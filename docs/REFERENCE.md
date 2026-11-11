# ugbench: Bedrock direct vs. Bedrock through Databricks Unity Gateway

> Developer reference (all options and internals). For the customer-facing reproduction guide, see the [README](../README.md).

A reproducible benchmark that measures **how much latency Databricks Unity Gateway adds**
when calling an LLM on Amazon Bedrock. It also measures the incremental cost of each gateway
feature (inference tables, rate limits, usage tracking). It produces a customer-shareable
report (HTML + Markdown + CSV + charts) from raw, per-request data.

```
                    ┌───────────────────────────── bedrock_direct ──────────────────────────────┐
 client (same  ─────┤                                                                           ├──► Amazon Bedrock
 AWS region)        └──► Databricks Unity Gateway ──► model provider service (Bedrock creds) ───┘
                          ├─ ug:baseline           (no optional features)
                          ├─ ug:inference_table    (+ payload logging to UC)
                          ├─ ug:rate_limits        (+ QPM limits, set high so they never trip)
                          ├─ ug:all_features       (+ both)
                          └─ ug_provider:direct    (call the provider service directly, no model service)
                         Legacy serving endpoints (optional): se:baseline / se:usage_tracking / ...
```

## Quick start

```bash
uv venv && uv pip install -e '.[dev]'        # or: pip install -e .
cp config.example.yaml config.yaml           # fill in profile, region, model, catalog/schema, credential

.venv/bin/ugbench targets                    # what will be benchmarked
.venv/bin/ugbench setup                      # create the UC model provider service + model services
.venv/bin/ugbench probe                      # 1 request per target: fix any errors before the real run
.venv/bin/ugbench health                     # 30 requests per target; must say HEALTHY before a benchmark
.venv/bin/ugbench run                        # interleaved benchmark + report
open results/<run_id>/report.html
```

Offline demo (no cloud calls, synthetic latencies): `ugbench -c config.mock.yaml run`.

### Optional follow-ups
```bash
ugbench run --stream                         # also measures time-to-first-token (report_ttft_ms.html)
ugbench run --concurrency 16                 # closed loop: 16 workers back to back
ugbench run --qps 100 --duration 600         # open loop: fixed 100 QPS total for 10 min (Poisson with --arrival poisson)
ugbench server-stats --warehouse-id <id>     # server-side breakdown from system.ai_gateway.usage (30-60 min later)
ugbench teardown --yes                       # delete the services/endpoints created by setup
```

## Where to run the client (this matters most)

The client-side delta = **gateway processing + any extra network distance**. To show the customer
what the gateway itself costs, put all three in **the same AWS region**:

| Component | Recommendation |
|---|---|
| Client | EC2 / Cloud9 / Databricks notebook in region X (`notebooks/run_benchmark.py`) |
| Databricks workspace | region X |
| Bedrock (`aws.region`) | region X. Use a regional model ID or `us.`/`eu.` inference profile consistently on both paths. |

The bundle (`databricks.yml`, `resources/ugbench_services.yml`) also creates the Unity Gateway services; see README §3.3–3.4. The simplest way to run is the bundle: `databricks bundle deploy -p <profile>`, then `databricks bundle run ugbench_benchmark -p <profile> --params qps=...,duration_s=...` (job defined in `resources/ugbench_benchmark.job.yml`; see README §3.6).
Without the bundle, run the notebook as a serverless job pinned to **environment version 4+**, which `dbutils` needs to read UC secrets:

```bash
databricks jobs submit --no-wait --json '{
  "run_name": "ugbench", "timeout_seconds": 14400,
  "environments": [{"environment_key": "env4", "spec": {"environment_version": "4"}}],
  "tasks": [{"task_key": "bench", "environment_key": "env4",
    "notebook_task": {"notebook_path": "/Workspace/Users/<you>/unity-gateway-benchmark/notebooks/run_benchmark",
      "base_parameters": {"config": "config.yaml", "aws_creds": "uc_secrets", "rounds": "2000", "setup": "no"}}}]}'
```

Running from a laptop works for a smoke test, but internet jitter (±20-50 ms) hides a single-digit ms signal.
If the customer's real application runs elsewhere, run a second benchmark from **their** location. That
number is their real-world experience.

## Prerequisites

* **Databricks:** Unity Gateway enabled (GA 2026-08-04) and `CREATE SERVICE` on `catalog.schema`
  (+ `CREATE TABLE` for inference tables, + `CREATE CONNECTION` if using inline AWS keys).
  The CLI profile from `~/.databrickscfg` is used for auth (OAuth or PAT, handled by the Databricks SDK).
* **Bedrock credentials, stored once in Unity Catalog** and shared by the gateway and the benchmark:
  * **Best:** a UC **service credential** (IAM role, short-lived credentials). This needs `CREATE SERVICE CREDENTIAL`,
    or `ACCESS` on an existing credential.
  * **Otherwise:** one AWS key pair stored as two **UC secrets** (`aws_uc_secrets` in the config). Create them once:
    ```bash
    export UGBENCH_AWS_ACCESS_KEY_ID=... UGBENCH_AWS_SECRET_ACCESS_KEY=...   # IAM user limited to bedrock:InvokeModel*
    ugbench store-aws-secrets && unset UGBENCH_AWS_ACCESS_KEY_ID UGBENCH_AWS_SECRET_ACCESS_KEY
    ```
    Set `unity_gateway.bedrock_auth.use_uc_secrets: true`. The provider service then references the secret access
    key by name (`secret_reference`), so the value is never copied, and rotating the secret needs no change to the
    service. The benchmark notebook reads both with `dbutils.secrets.get(catalog=, schema=, key=)`. Privileges:
    the provider-service owner needs `READ SECRET` on the secret access key, and the benchmark runner needs
    `READ SECRET` on both. Callers of the services need nothing on the secrets.
* **Bedrock auth for the direct path, outside Databricks:** any boto3 credential (`aws.profile`, env, instance role).
* Model access enabled in Bedrock for the chosen model in that region.
* For `server-stats`: a SQL warehouse and `SELECT` on `system.ai_gateway.usage` (account + metastore admin by default).

## What gets measured, and how

See [docs/METHODOLOGY.md](METHODOLOGY.md). In short:

* **Randomized interleaved rounds:** each round sends one request to every target in shuffled order.
* **Warm, kept-alive connections:** auth tokens are resolved outside the timer, and client retries are disabled.
* **Small outputs by default** (`max_tokens: 5`): the worst case for Databricks, because overhead is the
  largest share. Run again with a realistic `max_tokens` to show the overhead as a share of a real call.
* **Statistics:** P50/P90/P99 per target, percentile deltas with bootstrap 95% CIs, and a **paired
  median delta** (the most precise estimate of typical added latency).
* **Server-side cross-check:** `system.ai_gateway.usage` gives gateway total latency and upstream
  (Bedrock) latency per request, so `gateway_internal = total - upstream`. This is the same quantity
  the product team's reference numbers describe.

## Outputs (`results/<run_id>/`)

| File | Content |
|---|---|
| `raw.csv` | one row per request: target, round, latency_ms, ttfb/ttft, status, request_id, error |
| `meta.json` | fully resolved config, client environment, timestamps |
| `summary_latency_ms.csv`, `comparisons_latency_ms.csv` | the numbers behind the report |
| `report.html` / `report.md` | customer-shareable report (HTML is self-contained) |
| `charts/*.png` | overhead vs. direct, feature increments, distributions, timeline |
| `server.csv` | (after `server-stats`) per-request server-side breakdown |

## Unity Gateway vs. legacy serving endpoints

* `unity_gateway` (default) is the current product: UC **model provider service** (Bedrock) plus
  UC **model services** queried at `/ai-gateway/mlflow/v1/chat/completions` (or the native
  `/ai-gateway/anthropic/v1/messages` with `api: anthropic_messages`). Usage tracking is always on.
* `serving_endpoints` (opt-in) is an external-model serving endpoint with endpoint-level AI Gateway. Usage
  tracking, inference tables, and rate limits are separate toggles there, which matches the
  product team's per-feature breakdown.

## Credentials runbook (UC secrets)

**Rotate the AWS key.** No re-create needed:
```bash
KEYJSON=$(aws iam create-access-key --user-name <iam-user> --output json)
export UGBENCH_AWS_ACCESS_KEY_ID=$(jq -r .AccessKey.AccessKeyId <<<"$KEYJSON")
export UGBENCH_AWS_SECRET_ACCESS_KEY=$(jq -r .AccessKey.SecretAccessKey <<<"$KEYJSON"); unset KEYJSON
ugbench store-aws-secrets          # overwrite both UC secrets
ugbench setup --update-auth        # push the new key ID (plain field) to the provider service
unset UGBENCH_AWS_ACCESS_KEY_ID UGBENCH_AWS_SECRET_ACCESS_KEY
# wait until probes are clean on ALL targets, from where the benchmark runs (~17 min here), THEN delete the old IAM key
```
During propagation, some gateway instances pair the old key ID with the new secret, so expect a few
`403 signature does not match` errors. Keep the old IAM key active until every path is clean.

**Switching auth mode: use a NEW provider name, don't re-create in place.** Going from an inline key to a UC
secret reference, or between a secret reference and a service credential, can't be done in place: the API rejects
it. Don't delete and re-create under the same name. Gateway instances keep resolving the old name to the deleted
service's hidden connection (`<name>_do_not_modify`) and return `404 RESOURCE_DOES_NOT_EXIST`. Seen here: about
40% of requests from AWS us-east-1 were still failing 27 min later, even after the laptop path was clean.
Instead:
1. Set a new `unity_gateway.provider_service` name and run `ugbench setup`. This creates the new provider and
   repoints the existing model services in place.
2. **Keep the old provider service and the old IAM key** until `ugbench probe` is clean on every target, from the
   same place the benchmark will run. Model-service instances cache the resolved provider, connection and key.
3. Only then delete the old provider service and the old key.

If you do re-create model services, set a new `unity_gateway.inference_table_prefix`. Inference-table prefixes are
immutable, and new services can't attach to the existing `*_payload` tables.

**Always verify health right before a benchmark** from the benchmark's location: `ugbench health` (30 requests per target; exits non-zero unless every target is 100% OK). A config change can take
5-30 min to reach every gateway instance.

## Known gateway behaviors seen in testing (Nov 2026)

* **Traffic level matters.** At ~2 QPS, about 45% of requests took a ~80 ms slower path inside the gateway, for
  every variant including the provider service. At 100 QPS that dropped to ~1%, and gateway-internal time was
  P50 2 ms / P90 3 ms. Benchmark at the customer's real traffic level (`--qps`) and read gateway-internal time
  from `server-stats`.
* **Features are free at the measured resolution.** Inference tables and rate limits added 0 ± 2 ms.

## Re-using for another customer

1. Copy `config.example.yaml` to `config.<customer>.yaml`. Change `run_name`, region, model, catalog/schema.
2. `ugbench -c config.<customer>.yaml setup && ugbench -c config.<customer>.yaml run`
3. Fill in `docs/customer_brief_template.md` with numbers from `report.md`. Attach `report.html`.
4. `ugbench -c config.<customer>.yaml teardown --yes`

## Tests

`.venv/bin/pytest -q` covers the statistics, resource payloads, the HTTP client (mocked transport), and an
end-to-end mock run.
