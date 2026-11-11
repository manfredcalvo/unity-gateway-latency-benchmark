# Unity Gateway latency benchmark: Amazon Bedrock direct vs. through Databricks

**Question answered:** if your application calls an LLM on Amazon Bedrock *through Databricks Unity Gateway*
instead of calling Bedrock directly, how much latency is added, and what do gateway features (inference tables,
rate limits, usage tracking) add on top?

**Short answer** (Claude Haiku 4.5, us-east-1, Nov 2026, details in [Results](#6-results-we-measured)):
Unity Gateway's own processing is **2 ms at the median and 4–8 ms at P99**. End to end, a client in the same AWS
region sees **about +28 ms**. Almost all of that is the extra network hop. Inference tables and rate limits add
**no measurable latency** (< 1 ms).

This repo (`ugbench`) is a reproducible tool. Point it at **your** Databricks workspace, **your** AWS account and
**any Anthropic Claude model on Bedrock** (Haiku, Sonnet, Opus, any version), and it produces the same report from
your environment, with every raw request kept for audit. The model is a single setting: service names and
model-specific request fields follow from it ([3.3](#33-choose-the-model)).

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
3. [Step-by-step reproduction](#3-step-by-step-reproduction): install, credentials,
   [choose the model](#33-choose-the-model), configure, deploy, run
4. [Reading the report](#4-reading-the-report)
5. [Customer session plan](#5-customer-session-plan-6090-min)
6. [Results we measured](#6-results-we-measured)
7. [Troubleshooting](#7-troubleshooting)
8. [Cleanup](#8-cleanup)

---

## 1. How the measurement works

* **Same request on every path.** Same model, prompt, `max_tokens`, and the same model-specific fields
  (temperature 0 where the model accepts it, thinking kept off where it can be). Direct Bedrock uses boto3
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
| **Bedrock** | Access to the Anthropic Claude model you want to test, in that region: any model ID or inference profile (we measured `us.anthropic.claude-haiku-4-5-20251001-v1:0` and `us.anthropic.claude-sonnet-5-5`). |
| **Bedrock quota** | Requests/min and tokens/min **for that model** must cover your test rate, see the table below. Quotas are per model. |
| **AWS credentials** | **Best:** an IAM role registered as a UC **service credential**. **Alternative:** an IAM user limited to `bedrock:InvokeModel*`, stored once as two UC secrets. Your laptop also needs AWS credentials for the direct path, e.g. `aws sso login`. |
| **Server-side stats** *(optional)* | A SQL warehouse, and `SELECT` on `system.ai_gateway.usage` (account + metastore admin by default). |
| **Python** | Python ≥ 3.10 and [`uv`](https://docs.astral.sh/uv/) (or pip). |

**Quota sizing** (per model; the same for every Claude model at these sizes). Bedrock's quota is shared by
everything in the account, so leave headroom. Check yours with
`aws service-quotas list-service-quotas --service-code bedrock --query "Quotas[?contains(QuotaName,'<model name>')]"`.

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
.venv/bin/pytest -q                       # offline tests
.venv/bin/ugbench -c config.mock.yaml run # optional offline demo with synthetic latencies
```

### 3.2 Create the AWS credentials for the gateway

**Option A (recommended): IAM role + UC service credential.** No long-lived keys.
1. Create an IAM role with this permission policy. Replace `<ACCOUNT_ID>`. It allows invoking any Anthropic
   Claude model (inference profiles such as `us.`/`global.` and the underlying foundation models), so you can
   benchmark any of them; list specific ARNs instead to restrict it to the models you test:
   ```json
   {"Version": "2012-10-17", "Statement": [{
     "Effect": "Allow",
     "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
     "Resource": [
       "arn:aws:bedrock:*:<ACCOUNT_ID>:inference-profile/*.anthropic.claude-*",
       "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"]}]}
   ```
2. In Databricks, go to **Catalog → External data → Credentials → Create credential**, choose *Service credential*,
   and enter the role ARN. Copy the external ID it shows into the role's trust policy, as the UI instructs.
3. Grant yourself `ACCESS` on it. Set its name as `bedrock_service_credential` in your Option A bundle target (3.4).
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
   Use an Option B bundle target, without a `resources` line (3.4).

### 3.3 Choose the model
**The benchmark works with any Anthropic Claude model on Amazon Bedrock.** Pick its Bedrock model ID or inference
profile (for example `us.anthropic.claude-haiku-4-5-20251001-v1:0`, `us.anthropic.claude-sonnet-5-5`,
`us.anthropic.claude-opus-5-5`, or a `global.` profile). List what your account offers with
`aws bedrock list-inference-profiles --region <region>`. Everything else is derived from that ID:

* **Service names.** A short *model slug* (`us.anthropic.claude-sonnet-5-5` -> `sonnet_5_5`) names the services:
  model services `ugbench_<slug>_{baseline,inference_table,rate_limits,all_features}`, provider
  `ugbench_<slug>_bedrock`, inference tables `ugbench_<slug>_<variant>_payload`. **Each model gets its own set of
  services**, so you can benchmark several models side by side and remove each one independently.
* **Request fields.** Models differ in what a request may contain. The benchmark adjusts the fields per model and
  sends them **identically on every path** (`ugbench/models.py`), so the comparison stays fair:

  | Model | `temperature: 0` | Thinking (kept off for a short, low-variance reply) |
  |---|---|---|
  | Haiku 4.5, Sonnet/Opus 4.6 and older | sent | off by default; nothing sent |
  | Opus 4.7 / 4.8 | not sent (rejected) | off by default; nothing sent |
  | Sonnet 5, Opus 5, Haiku 5.5 | not sent | `thinking: {type: "disabled"}` |
  | Sonnet 5.5 | not sent | `thinking: {type: "between_tools"}` (`disabled` is rejected) |
  | Opus 5.5, Fable 5.x | not sent | can't be turned off: `output_config: {effort: "low"}` |
  | newer models | rules of the newest known generation of that tier | |

  To override, set `workload.thinking` / `workload.effort` in `config.yaml` (`null` = send nothing, or a value
  to send as given). Every report records the fields that were sent.

Check what a model resolves to before deploying anything (offline, no credentials needed):
```bash
.venv/bin/ugbench -c config.example.yaml --model us.anthropic.claude-sonnet-5-5 targets
# [model] us.anthropic.claude-sonnet-5-5 (sonnet_5_5): temperature not sent, thinking={'type': 'between_tools'}
# ug:baseline   ug_model_service   {"service": "<catalog>.<schema>.ugbench_sonnet_5_5_baseline", ...}
```
Two things must also cover the model: the **IAM policy** (the one in 3.2 allows every Claude model) and its
**Bedrock quota** at your test rate (§2). Larger models are slower (Sonnet 5.5 ≈ 2x Haiku 4.5 per request), so the
same QPS keeps more requests in flight; the client handles up to 512 by default.

### 3.4 Configure
The repo is a **Declarative Automation Bundle**. One `databricks bundle deploy` per model creates that model's
gateway services and the benchmark job. Two files hold your settings.

**`databricks.yml`: one target per model.** Copy the template at the bottom of the file and set the model:
```yaml
targets:
  sonnet_5_5:                                   # Option B (UC secrets)
    variables:
      catalog: main
      schema: ug_benchmark
      bedrock_model_id: us.anthropic.claude-sonnet-5-5
      model_slug: sonnet_5_5                    # the slug `ugbench --model <id> targets` prints

  sonnet_5_5_sc:                                # Option A (service credential): the bundle also creates the provider
    variables:
      catalog: main
      schema: ug_benchmark
      bedrock_model_id: us.anthropic.claude-sonnet-5-5
      model_slug: sonnet_5_5
      bedrock_service_credential: my_bedrock_credential
      provider_ref: ${resources.model_provider_services.bedrock.model_provider_service_id}
    resources: *sc_provider
```
The included `dev` target (Option A, Haiku 4.5) works the same way. All variables:

| Variable | Default | Meaning |
|---|---|---|
| `catalog`, `schema` | *(set per target)* | Where the services and inference tables are created |
| `bedrock_model_id` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | **The model**, on every path |
| `model_slug` | `haiku_4_5` | Short model name used in the service names |
| `service_prefix` | `ugbench_${var.model_slug}` | Model services `<prefix>_{baseline,inference_table,rate_limits,all_features}` |
| `provider_service` | `ugbench_${var.model_slug}_bedrock` | Model provider service |
| `inference_table_prefix` | `ugbench_${var.model_slug}` | Tables `<prefix>_<variant>_payload` |
| `bedrock_service_credential` | `CHANGE_ME` | Option A only: your UC service credential |
| `aws_region` | `us-east-1` | Bedrock region, the **same** as the workspace |
| `rate_limit_rpm` | `100000` | Rate limit far above the test rate: evaluated on every request, never triggered |

You can also override any variable with `--var name=value`, or keep your values out of `databricks.yml` in
`.databricks/bundle/<target>/variable-overrides.json` (git-ignored), e.g. `{"catalog": "main", "schema": "ug_benchmark"}`.

**`config.yaml`** is used by the laptop CLI, and for the notebook's client settings:
```bash
cp config.example.yaml config.yaml
```
| Key | Example |
|---|---|
| `run_name` | `acme-latency` |
| `databricks.profile` | your CLI profile |
| `aws.region` | same as the bundle's `aws_region` |
| `aws.bedrock_model_id` | your default model; any command takes `--model <id>` to switch |
| `aws.profile` | the AWS profile for direct calls from the laptop |
| `unity_gateway.catalog/schema` | **the same as the bundle variables** |
| `unity_gateway.provider_service/service_prefix` | keep the `{model_slug}` templates, so names match the bundle's |
| `aws_uc_secrets.catalog/schema` | where `store-aws-secrets` put the key (Option B) |
| `unity_gateway.bedrock_auth` | Option A or B from 3.2 |

The Databricks job always gets the catalog, schema, service names, region and model from its target's bundle
variables, so it benchmarks exactly what that target deployed.

### 3.5 Deploy the services and the job, then check health
Set `M` to your model and `T` to its bundle target, then:
```bash
M=us.anthropic.claude-sonnet-5-5; T=sonnet_5_5

# Option A (service credential): the bundle creates the provider service + 4 model services + the job
databricks bundle validate --strict -p <profile> -t $T
databricks bundle deploy -p <profile> -t $T

# Option B (UC secrets): bundles can't reference a UC secret on a provider service yet, so create the
# provider once with the CLI, then deploy the model services + the job
.venv/bin/ugbench --model $M setup --provider-only
databricks bundle deploy -p <profile> -t $T

.venv/bin/ugbench --model $M targets        # the 6 paths and the request fields
.venv/bin/ugbench --model $M health         # 30 requests per path; must print HEALTHY before you benchmark
```
After the first deploy, configuration can take **5–30 minutes** to reach every gateway instance. Re-run `health`
until it reports HEALTHY. Run `bundle deploy` again after any change to the code, `config.yaml` or the variables.
Always pass `-t <target>`: `dev` is the default target, and in Option B it would try to create a provider with
the `CHANGE_ME` service credential. To benchmark another model, repeat 3.4–3.5 with a new target.

*Without bundles:* `.venv/bin/ugbench --model $M setup` creates the provider and the 4 model services from
`config.yaml` alone.

### 3.6 Run from the laptop (quick demo)
Each run writes `results/<run_id>/report.html`. Add `--model <id>` to benchmark a model other than the one in
`config.yaml`.
```bash
# 60 s at 100 QPS, tiny prompt (the quickest convincing run)
.venv/bin/ugbench --model $M run --workload tiny --qps 100 --duration 60 --arrival poisson

# 5 min at 100 QPS, large prompt (566 input tokens): shows request size doesn't matter
.venv/bin/ugbench --model $M run --workload large_prompt --qps 100 --duration 300 --arrival poisson

# 30 s at 150 QPS, tiny prompt
.venv/bin/ugbench --model $M run --workload tiny --qps 150 --duration 30 --arrival poisson
```
A laptop measures over the internet, so its numbers include your ISP's path. For the numbers to present, run from
inside AWS (3.7).

### 3.7 Run from Databricks in the same region (customer-grade numbers)
Each target's job `ugbench-benchmark` runs `notebooks/run_benchmark` on **serverless environment version 4**,
which is required to read UC secrets, against that target's model and services. Every notebook setting is a job
parameter. The defaults match our headline run: 150 QPS for 20 minutes, tiny prompt, UC secrets.

1. Run it from the command line. `bundle run` waits and prints the run URL; add `--no-wait` to start it and return.
   ```bash
   # headline run (defaults: 150 QPS, 20 min, tiny prompt)
   databricks bundle run ugbench_benchmark -p <profile> -t $T --params run_id=acme-$T-qps150-20min

   # any other variant, for example 30 minutes at 100 QPS
   databricks bundle run ugbench_benchmark -p <profile> -t $T --no-wait \
     --params qps=100,duration_s=1800,run_id=acme-$T-qps100-30min
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
   | `bedrock_model_id`, `ug_catalog`, `ug_schema`, `ug_service_prefix`, `ug_provider_service`, `aws_region` | the target's variables | What to benchmark; leave as is |
2. Results are written to `/Workspace/Users/<you>/.bundle/ugbench/<target>/files/results/<run_id>/`. `raw.csv` is
   refreshed every 60 s, so you can watch progress live. Download a finished run to the laptop with:
   ```bash
   databricks workspace export-dir \
     /Workspace/Users/<you>/.bundle/ugbench/$T/files/results/<run_id> results/<run_id> -p <profile>
   ```
   Later deploys never touch the `results/` folder.

*Without bundles:* upload the folder with `databricks workspace import-dir` and start the notebook with
`databricks jobs submit`, using the same parameters as `base_parameters` and an `environments` block with
`environment_version: "4"`.

### 3.8 Server-side breakdown (30–60 min after a run)
```bash
.venv/bin/ugbench server-stats results/<run_id> --warehouse-id <sql-warehouse-id> --wait 5400
```
This waits until `system.ai_gateway.usage` covers the whole run, then adds the gateway-processing / Bedrock-leg /
network split to `report.html`. For a Databricks run, first download it with `export-dir` (3.7).

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
| 15–20 min | Start the long, customer-grade run in Databricks | `databricks bundle run ugbench_benchmark -t <target> --no-wait` (3.7) |
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
| `403 PERMISSION_DENIED … The model provider denied` on every gateway path, for a new model | The IAM policy doesn't allow that model yet (3.2). After updating it, IAM takes a few minutes to propagate; re-run `health`. |
| `400 … temperature is deprecated` / thinking errors for a new model | Its request rules aren't what `ugbench/models.py` assumes. Set `workload.thinking` / `workload.effort` (or `temperature: null`) in `config.yaml`. |
| `403 signature does not match` / `security token invalid` after a key rotation | Propagation: some instances still hold the old key. Keep the **old IAM key active** until `health` is clean everywhere. |
| `Cannot change the UC Secret reference on an existing model provider service` | Switching auth mode needs a new provider service (see the first row). |
| Notebook: `dbutils.secrets.get() got an unexpected keyword argument 'catalog'` | The job isn't on **serverless environment version 4+**. Deploy the job with the bundle (3.5), which sets it. |
| Notebook: `Unknown service: 'bedrock-runtime'` | Old boto3 in the image. The notebook's `%pip install --upgrade` line fixes it; keep it. |
| Many `429 Too many tokens` / `Too many requests` on **all** paths, including direct | Bedrock quota. Lower `--qps` or the prompt size, or request a quota increase. |
| Report says **"Rate NOT fully held"** | Client CPU bottleneck. Use `--procs 4` (or `auto`) or a lower `--qps`. |
| `health` shows stale 404s right after `bundle destroy` + `bundle deploy` | Same cause as the first row: set new `service_prefix` / `provider_service` (and `inference_table_prefix`) variables before re-deploying. |
| `bundle deploy` fails on `credentials/CHANGE_ME` | You deployed the default `dev` target. Set `bedrock_service_credential` (Option A), or pass `-t <your Option B target>`. |
| `bundle validate` warns `unknown field: secret_reference` | Bundles can't reference a UC secret on a provider service yet. Use an Option B target (no `resources` line) with `ugbench --model <id> setup --provider-only`. |
| `setup` fails creating a model service with an inference table | Inference-table names are immutable. Set a new `unity_gateway.inference_table_prefix`. |
| `server-stats` finds no or partial rows | The usage table lags 30–60 minutes. Use `--wait 5400`; it keeps polling until the whole run is covered. |

---

## 8. Cleanup
Per model target:
```bash
databricks bundle destroy -p <profile> -t <target>   # its model services (+ provider in Option A), the job and the
                                                     # bundle folder (export results/ first); tables are kept
.venv/bin/ugbench --model <id> teardown --yes        # Option B: also deletes that model's provider service
```
Inference tables are never deleted automatically. List them and drop the ones you no longer need:
```bash
databricks tables list <catalog> <schema> -p <profile> -o json | jq -r '.[] | select(.name|endswith("_payload")) | .name'
databricks tables delete <catalog>.<schema>.ugbench_<slug>_inference_table_payload -p <profile>
databricks tables delete <catalog>.<schema>.ugbench_<slug>_all_features_payload -p <profile>
```
Then remove what you created by hand: the IAM user's access keys and the user (Option B), or the IAM role and
service credential (Option A), and the UC secrets (`databricks secrets-uc delete-secret …`).

---

**Further reading:** [docs/METHODOLOGY.md](docs/METHODOLOGY.md) (statistics, controls) ·
[docs/REFERENCE.md](docs/REFERENCE.md) (all CLI options, credential runbook, Strands layer notes) ·
[docs/customer_brief_template.md](docs/customer_brief_template.md) (one-page summary to fill in).
