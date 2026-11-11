# Unity Gateway latency benchmark: Amazon Bedrock direct vs. through Databricks

**Question answered:** if your application calls an LLM on Amazon Bedrock *through Databricks Unity Gateway*
instead of calling Bedrock directly, how much latency is added, and what do gateway features (inference tables,
rate limits, usage tracking) add on top?

**Short answer** (Claude Haiku 4.5, us-east-1, Nov 2026, details in [Results](#6-results-we-measured)):
Unity Gateway's own processing is **2 ms at the median and 4–8 ms at P99**. End to end, a client in the same AWS
region sees **about +28 ms**. Almost all of that is the extra network hop. Inference tables and rate limits add
**no measurable latency** (< 1 ms).

This repo (`ugbench`) is a reproducible tool. Point it at **your** Databricks workspace and **your** AWS
account, and it produces the same report from your environment, with every raw request kept for audit.

```
                       ┌──────────────── bedrock_direct (boto3 Converse) ────────────────┐
  client  ─────────────┤                                                                  ├──►  Amazon Bedrock
 (laptop or Databricks │  Unity Gateway  /ai-gateway/anthropic/v1/messages                │      (your account)
  notebook, same       └──► model service ──► model provider service (Bedrock creds) ────┘
  AWS region)               ├─ ug:baseline         no optional features
                            ├─ ug:inference_table  + request/response logging to Unity Catalog
                            ├─ ug:rate_limits      + rate limits (set very high: evaluated, never triggered)
                            ├─ ug:all_features     + both
                            └─ ug_provider:direct  provider service only (no model service)
```

---

## Contents
1. [How the measurement works](#1-how-the-measurement-works)
2. [Prerequisites](#2-prerequisites-in-your-environment)
3. [Step-by-step reproduction](#3-step-by-step-reproduction)
4. [Reading the report](#4-reading-the-report)
5. [Customer session plan](#5-customer-session-plan-60-90-min)
6. [Results we measured](#6-results-we-measured)
7. [Troubleshooting](#7-troubleshooting)
8. [Cleanup](#8-cleanup)

---

## 1. How the measurement works

* **Same request on every path.** Same model, prompt, `max_tokens`, temperature 0. Direct Bedrock uses boto3
  Converse. The Databricks paths use the native Anthropic Messages API (`/ai-gateway/anthropic/v1/messages`),
  which is what the Anthropic SDK uses.
* **Fixed request rate (open loop).** Requests are released on a Poisson schedule at e.g. 100 or 150 QPS, whether
  or not earlier ones have finished, so every path sees the same load. The report states whether the client
  actually held the rate ("send lag").
* **Interleaved rounds.** Each round sends one request to every path, in random order, so Bedrock's own drift hits
  all paths equally. The **paired median delta** (gateway request minus the direct request in the same round) is the
  headline number, with a bootstrap 95% confidence interval.
* **Warm connections, no hidden retries.** Connections are reused, auth tokens are fetched outside the timer, and
  client retries are disabled.
* **Server-side cross-check.** Every gateway request is tagged (`Databricks-Ai-Gateway-Request-Tags`) and matched to
  its row in `system.ai_gateway.usage`. That splits the end-to-end difference into **gateway processing**, the
  **gateway → Bedrock leg**, and the **client → Databricks leg**.

More detail: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

---

## 2. Prerequisites in your environment

| Area | What is needed |
|---|---|
| **Databricks** | A workspace with **Unity Gateway** enabled, **in the same AWS region as Bedrock** (e.g. both us-east-1). Databricks CLI ≥ 1.19 with a profile (`databricks auth login`). |
| **Unity Catalog** | A catalog/schema you can use, with `USE CATALOG`, `USE SCHEMA`, `CREATE SERVICE`, `CREATE TABLE` (inference tables), and either `CREATE SECRET` (UC secrets) or `ACCESS` on a service credential. |
| **Bedrock** | Model access for the chosen model (we used `us.anthropic.claude-haiku-4-5-20251001-v1:0`) in that region. |
| **Bedrock quota** | Requests/min and tokens/min for the model must cover your test rate, see the table below. |
| **AWS credentials** | **Best:** an IAM role registered as a UC **service credential**. **Alternative:** an IAM user limited to `bedrock:InvokeModel*`, stored once as two UC secrets. Your laptop also needs AWS credentials for the direct path, e.g. `aws sso login`. |
| **Server-side stats** *(optional)* | A SQL warehouse, and `SELECT` on `system.ai_gateway.usage` (account + metastore admin by default). |
| **Python** | Python ≥ 3.10 and [`uv`](https://docs.astral.sh/uv/) (or pip). |

**Quota sizing.** Bedrock's quota is shared by everything in the account. Leave headroom.

| Test | Requests/min | Tokens/min (approx.) |
|---|---|---|
| 100 QPS, tiny prompt (5-token reply) | 6,000 | ~0.2 M |
| 100 QPS, large prompt (566 in / 5 out) | 6,000 | ~3.4 M |
| 150 QPS, tiny prompt | 9,000 | ~0.3 M |
| 1,000 QPS, tiny prompt | 60,000 | ~2 M → needs a quota increase |

---

## 3. Step-by-step reproduction

### 3.1 Get the code and install
```bash
git clone <this repo> unity-gateway-benchmark && cd unity-gateway-benchmark   # or unzip the shared copy
uv venv && uv pip install -e '.[dev]'
.venv/bin/pytest -q                       # 16 tests, offline
.venv/bin/ugbench -c config.mock.yaml run # optional offline demo with synthetic latencies
```

### 3.2 Create the AWS credentials for the gateway

**Option A (recommended): IAM role + UC service credential.** No long-lived keys.
1. Create an IAM role with this permission policy. Replace `<ACCOUNT_ID>`:
   ```json
   {"Version": "2012-10-17", "Statement": [{
     "Effect": "Allow",
     "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
     "Resource": [
       "arn:aws:bedrock:*:<ACCOUNT_ID>:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0",
       "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0"]}]}
   ```
2. In Databricks, go to **Catalog → External data → Credentials → Create credential**, choose *Service credential*,
   and enter the role ARN. Copy the external ID it shows into the role's trust policy, as the UI instructs.
3. Grant yourself `ACCESS` on it. Set its name as `bedrock_service_credential` in the bundle's `dev` target (3.3).
   For laptop runs, also set `aws.service_credential: <name>` in `config.yaml`.

**Option B: one access key stored as UC secrets.** Use this when you can't create service credentials.
1. Create an IAM user with the same permission policy as above, plus one access key.
2. Store the key **once** in Unity Catalog. The values come from environment variables and are never printed or
   written to disk:
   ```bash
   KEYJSON=$(aws iam create-access-key --user-name <iam-user> --output json)
   export UGBENCH_AWS_ACCESS_KEY_ID=$(jq -r .AccessKey.AccessKeyId <<<"$KEYJSON")
   export UGBENCH_AWS_SECRET_ACCESS_KEY=$(jq -r .AccessKey.SecretAccessKey <<<"$KEYJSON"); unset KEYJSON
   .venv/bin/ugbench store-aws-secrets    # creates <catalog>.<schema>.ugbench_aws_{access_key_id,secret_access_key}
   unset UGBENCH_AWS_ACCESS_KEY_ID UGBENCH_AWS_SECRET_ACCESS_KEY
   ```
3. In `config.yaml`, set `unity_gateway.bedrock_auth.use_uc_secrets: true`. The provider service then
   **references** the secret by name and never copies it. The notebook reads both secrets with `dbutils.secrets`.
   Use the bundle target `dev_uc_secrets` (3.4).

### 3.3 Configure
The repo is a **Declarative Automation Bundle**. One `databricks bundle deploy` creates the gateway services
under test and the benchmark job. Two files hold your settings:

**`databricks.yml`**: edit the `variables` of the target you'll use. `dev` is for Option A, `dev_uc_secrets` for
Option B.

| Variable | Default | Meaning |
|---|---|---|
| `catalog`, `schema` | *(set per target)* | Where the services and inference tables are created |
| `bedrock_service_credential` | `CHANGE_ME` | Option A only: your UC service credential |
| `aws_region` | `us-east-1` | Bedrock region, the **same** as the workspace |
| `bedrock_model_id` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Model on every path |
| `service_prefix` | `ugbench_dab` | Model services `<prefix>_{baseline,inference_table,rate_limits,all_features}` |
| `provider_service` | `ugbench_dab_bedrock` | Model provider service |
| `inference_table_prefix` | `ugbench_dab` | Tables `<prefix>_<variant>_payload` |
| `rate_limit_rpm` | `100000` | Rate limit far above the test rate: evaluated on every request, never triggered |

You can also override any variable on the command line with `--var name=value`, or keep your values out of
`databricks.yml` in `.databricks/bundle/<target>/variable-overrides.json` (git-ignored), e.g.
`{"catalog": "main", "schema": "ug_benchmark"}`.

**`config.yaml`** is used by the laptop CLI, and for the notebook's client settings:
```bash
cp config.example.yaml config.yaml
```
| Key | Example |
|---|---|
| `run_name` | `acme-haiku45` |
| `databricks.profile` | your CLI profile |
| `aws.region`, `aws.bedrock_model_id` | same as the bundle variables |
| `aws.profile` | the AWS profile for direct calls from the laptop |
| `unity_gateway.catalog/schema/service_prefix/provider_service` | **the same values as the bundle variables** |
| `aws_uc_secrets.catalog/schema` | where `store-aws-secrets` put the key (Option B) |
| `unity_gateway.bedrock_auth` | Option A or B from 3.2 |
| `unity_gateway.api` | `anthropic_messages` (default) |

The Databricks job always gets the catalog, schema, service names, region and model from the bundle variables,
so it benchmarks exactly what the bundle deployed.

### 3.4 Deploy the services and the job, then check health
```bash
# Option A (service credential): the bundle creates the provider service + 4 model services + the job
databricks bundle validate --strict -p <profile>
databricks bundle deploy -p <profile>

# Option B (UC secrets): bundles can't reference a UC secret on a provider service yet, so create the
# provider once with the CLI (it reads config.yaml), then deploy the model services + the job
.venv/bin/ugbench setup --provider-only
databricks bundle deploy -p <profile> -t dev_uc_secrets

.venv/bin/ugbench targets        # the 6 paths that will be measured
.venv/bin/ugbench health         # 30 requests per path; must print HEALTHY before you benchmark
```
After the first deploy, configuration can take **5–30 minutes** to reach every gateway instance. Re-run `health`
until it reports HEALTHY. Run `bundle deploy` again after any change to the code, `config.yaml` or the variables.
For Option B, add `-t dev_uc_secrets` to **every** `bundle` command below. `dev` is the default target, and
deploying it in Option B would try to create a second provider service with the `CHANGE_ME` service credential.

*Without bundles:* `.venv/bin/ugbench setup` creates the provider and the 4 model services from `config.yaml` alone.

### 3.5 Run from the laptop (quick demo)
These are the exact runs we used. Each one writes `results/<run_id>/report.html`.
```bash
# 60 s at 100 QPS, tiny prompt (the quickest convincing run)
.venv/bin/ugbench run --workload tiny --qps 100 --duration 60 --arrival poisson

# 5 min at 100 QPS, large prompt (566 input tokens): shows request size doesn't matter
.venv/bin/ugbench run --workload large_prompt --qps 100 --duration 300 --arrival poisson

# 30 s at 150 QPS, tiny prompt
.venv/bin/ugbench run --workload tiny --qps 150 --duration 30 --arrival poisson
```
A laptop measures over the internet, so its numbers include your ISP's path. For the numbers to present, run from
inside AWS (3.6).

### 3.6 Run from Databricks in the same region (customer-grade numbers)
The bundle's job `ugbench-benchmark` runs `notebooks/run_benchmark` on **serverless environment version 4**,
which is required to read UC secrets. Every notebook setting is a job parameter. The defaults match our headline
run: 150 QPS for 20 minutes, tiny prompt, UC secrets.

1. Run it from the command line. `bundle run` waits and prints the run URL; add `--no-wait` to start it and return.
   ```bash
   # headline run (defaults: 150 QPS, 20 min, tiny prompt)
   databricks bundle run ugbench_benchmark -p <profile> --params run_id=acme-colocated-qps150-20min

   # any other variant, for example a 10-minute large-prompt run at 100 QPS
   databricks bundle run ugbench_benchmark -p <profile> --no-wait \
     --params qps=100,duration_s=600,workload=large_prompt,run_id=acme-colocated-large-10min
   ```
   | Parameter | Default | Meaning |
   |---|---|---|
   | `qps`, `duration_s`, `arrival` | `150`, `1200`, `poisson` | Open-loop request rate and length |
   | `workload` | `tiny` | `tiny`, `large_prompt` or `realistic` |
   | `run_id` | auto | Name of the results folder |
   | `health_n` | `10` | Health-gate requests per path before the run (the job fails if a path is unhealthy) |
   | `aws_creds` | `uc_secrets` | `service_credential` for Option A (set `bedrock_service_credential` too) |
   | `procs` | `1` | Client processes, if the report says the rate was not held |
   | `results_volume` | empty | Also copy results to `/Volumes/<cat>/<schema>/<vol>` |
   | `ug_catalog`, `ug_schema`, `ug_service_prefix`, `ug_provider_service`, `aws_region`, `bedrock_model_id` | bundle variables | What to benchmark; normally leave as is |
2. Results are written to `/Workspace/Users/<you>/.bundle/ugbench/<target>/files/results/<run_id>/`. `raw.csv` is
   refreshed every 60 s, so you can watch progress live. Download a finished run to the laptop with:
   ```bash
   databricks workspace export-dir \
     /Workspace/Users/<you>/.bundle/ugbench/<target>/files/results/<run_id> results/<run_id> -p <profile>
   ```
   Later deploys never touch the `results/` folder.

*Without bundles:* upload the folder with `databricks workspace import-dir` and start the notebook with
`databricks jobs submit`, using the same parameters as `base_parameters` and an `environments` block with
`environment_version: "4"`.

### 3.7 Server-side breakdown (30–60 min after a run)
```bash
.venv/bin/ugbench server-stats results/<run_id> --warehouse-id <sql-warehouse-id> --wait 5400
```
This waits until `system.ai_gateway.usage` covers the whole run, then adds the gateway-processing / Bedrock-leg /
network split to `report.html`. For a Databricks run, first download it with `export-dir` (3.6).

---

## 4. Reading the report

`results/<run_id>/report.html` is self-contained, so you can email it. The key parts:

| Section | What it tells you |
|---|---|
| **Summary** | Paired median added latency vs. direct, with 95% CI, as a % of the direct P50 |
| **Test setup → "Rate held"** | Whether the client kept the target rate (send lag P99 < 20 ms). If it says *NOT fully held*, the client was the bottleneck: use `--procs N` or a lower rate. |
| **Latency by target** | P50/P90/P99 per path, plus output tokens (should be equal on every path) |
| **Added latency vs. Bedrock direct** | Paired median and percentile deltas with CIs |
| **Incremental cost of gateway features** | Each feature vs. the no-features model service |
| **Server-side breakdown** | Gateway processing vs. network, from `system.ai_gateway.usage` |

Interpretation rules:
* **Trust the paired median first.** P50/P90 deltas need ≥ 1,000 requests per path to be meaningful.
* **A CI that includes 0** means "not distinguishable from zero". That's the expected result for the features.
* **Client-side P99 needs thousands of requests per path.** The report warns when there are too few. For P99, use
  the server-side gateway-processing P99, which excludes Bedrock's own tail.

---

## 5. Customer session plan (60–90 min)

**Before the session (customer side, ~30 min):**
- [ ] Workspace + Unity Gateway, in the same region as Bedrock; Bedrock model access; quota checked
- [ ] Catalog/schema with the privileges in §2; a SQL warehouse ID for server-side stats
- [ ] Credentials: a service credential (Option A) or an IAM user + `store-aws-secrets` (Option B)
- [ ] `databricks.yml` variables and `config.yaml` filled in
- [ ] `databricks bundle deploy` (plus `ugbench setup --provider-only` for Option B) run **at least 30 minutes
      before** the session (propagation time); `ugbench health` is HEALTHY

**During the session:**
| Time | Step | Command |
|---|---|---|
| 0–10 min | Walk through the method and the diagram above | n/a |
| 10–15 min | Health check, live | `ugbench health` |
| 15–20 min | Start the long, customer-grade run in Databricks | `databricks bundle run ugbench_benchmark --no-wait [-t dev_uc_secrets]` (3.6) |
| 20–25 min | Quick laptop run while it runs | `ugbench run --workload tiny --qps 100 --duration 60` |
| 25–40 min | Open the laptop `report.html` and explain each section (§4) | n/a |
| 40–50 min | Watch the Databricks job's `raw.csv` grow; discuss features and network | n/a |
| after ~60 min | Pull the server-side split | `ugbench server-stats …` |

**Optional:** run at the customer's real peak rate (`--qps <their peak>`), as long as their Bedrock quota allows it.

---

## 6. Results we measured

Environment: Databricks workspace and Bedrock both in **us-east-1**, model **Claude Haiku 4.5**
(cross-region inference profile), Anthropic Messages API, Nov 2026.

**Headline run: 150 QPS for 20 min, from a Databricks serverless notebook** (180,000 requests, 1 upstream 503, no
throttling):

| Path | P50 | P90 | P99 | Added vs. direct (paired median, 95% CI) |
|---|---|---|---|---|
| Bedrock direct | 295 ms | 340 ms | 614 ms | n/a |
| Model service, no features | 322 ms | 370 ms | 722 ms | **+27.8 ms** [27.3, 28.2] |
| + inference table | 323 ms | 370 ms | 657 ms | +27.8 ms [27.3, 28.3] |
| + rate limits | 323 ms | 371 ms | 664 ms | +28.0 ms [27.5, 28.5] |
| + inference table + rate limits | 323 ms | 370 ms | 652 ms | +28.1 ms [27.6, 28.6] |
| Provider service only | 323 ms | 371 ms | 699 ms | +28.1 ms [27.7, 28.6] |

**Where the +28 ms goes** (server-side, all 150,000 gateway requests):

| Part | Median |
|---|---|
| **Gateway processing** (P50 2 ms, P90 2 ms, **P99 4–8 ms**) | **2.0 ms** |
| Gateway → Bedrock network leg | 11.0 ms |
| Client → Databricks network leg | 14.4 ms |

**Other runs**

| Run | Added vs. direct (median) | Features |
|---|---|---|
| Laptop, 100 QPS, 10 min, tiny prompt | +24 ms | < 1 ms |
| Laptop, 100 QPS, 5 min, **large prompt (566 tokens)** | +22 ms | < 0.5 ms |
| Notebook, 100 QPS, 10 min, large prompt | +27 ms | < 0.5 ms |
| Laptop, Strands + Anthropic SDK, 30 QPS, 300-token replies (experimental layer) | +26 ms total / +24 ms TTFT (~1%) | 0–4 ms |

**Caveats to share openly:**
* **At very low traffic (~2 QPS), ~45% of requests took a ~80 ms slower path inside the gateway.** At 100–150 QPS
  that dropped to < 1%. Benchmark at your real traffic level.
* **Gateway P99.9 is ~158 ms.** Rare slow requests remain in the far tail.
* **About 25 of the 28 ms is network.** It depends on how your clients reach the workspace endpoint (e.g. public
  internet vs. PrivateLink).

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `health` shows `404 … Connection '<name>_do_not_modify' does not exist` | The provider service was deleted and **re-created under the same name**; some gateway instances still point at the old one. Use a **new** `provider_service` name and re-run `setup`. That re-points the model services in place. Keep the old one until `health` is clean, then delete it. |
| `403 signature does not match` / `security token invalid` after a key rotation | Propagation: some instances still hold the old key. Keep the **old IAM key active** until `health` is clean everywhere. |
| `Cannot change the UC Secret reference on an existing model provider service` | Switching auth mode needs a new provider service (see the first row). |
| Notebook: `dbutils.secrets.get() got an unexpected keyword argument 'catalog'` | The job isn't on **serverless environment version 4+**. Add the `environments` block from 3.6. |
| Notebook: `Unknown service: 'bedrock-runtime'` | Old boto3 in the image. The notebook's `%pip install --upgrade` line fixes it; keep it. |
| Many `429 Too many tokens` / `Too many requests` on **all** paths, including direct | Bedrock quota. Lower `--qps` or the prompt size, or request a quota increase. |
| Report says **"Rate NOT fully held"** | Client CPU bottleneck. Use `--procs 4` (or `auto`) or a lower `--qps`. |
| `health` shows stale 404s right after `bundle destroy` + `bundle deploy` | Same cause as the first row: set new `service_prefix` / `provider_service` (and `inference_table_prefix`) variables before re-deploying. |
| `bundle deploy` fails on `credentials/CHANGE_ME` | You deployed the default `dev` target. Set `bedrock_service_credential` (Option A), or add `-t dev_uc_secrets` (Option B). |
| `bundle validate` warns `unknown field: secret_reference` | Bundles can't reference a UC secret on a provider service yet. Use target `dev_uc_secrets` with `ugbench setup --provider-only`. |
| `setup` fails creating a model service with an inference table | Inference-table names are immutable. Set a new `unity_gateway.inference_table_prefix`. |
| `server-stats` finds no or partial rows | The usage table lags 30–60 minutes. Use `--wait 5400`; it keeps polling until the whole run is covered. |

---

## 8. Cleanup
```bash
databricks bundle destroy -p <profile> [-t dev_uc_secrets]   # the services it created + the job + the bundle
                                                            # folder (export results/ first); tables are kept
.venv/bin/ugbench teardown --yes   # Option B: also deletes the provider service (and anything made by `ugbench setup`)
```
Inference tables are never deleted automatically. List them and drop the ones you no longer need:
```bash
databricks tables list <catalog> <schema> -p <profile> -o json | jq -r '.[] | select(.name|endswith("_payload")) | .name'
databricks tables delete <catalog>.<schema>.ugbench_dab_inference_table_payload -p <profile>
databricks tables delete <catalog>.<schema>.ugbench_dab_all_features_payload -p <profile>
```
Then remove what you created by hand: the IAM user's access keys and the user (Option B), or the IAM role and
service credential (Option A), and the UC secrets (`databricks secrets-uc delete-secret …`).

---

**Further reading:** [docs/METHODOLOGY.md](docs/METHODOLOGY.md) (statistics, controls) ·
[docs/REFERENCE.md](docs/REFERENCE.md) (all CLI options, credential runbook, Strands layer notes) ·
[docs/customer_brief_template.md](docs/customer_brief_template.md) (one-page summary to fill in).
