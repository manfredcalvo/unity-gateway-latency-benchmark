# Databricks notebook source
# MAGIC %md
# MAGIC # Unity Gateway latency benchmark (in-Databricks runner)
# MAGIC
# MAGIC Runs `ugbench` from Databricks compute in the **same AWS region** as the workspace and Bedrock,
# MAGIC so the measured delta is the gateway, not cross-region network.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC 1. Clone this repo as a Git folder in the workspace (or upload the folder).
# MAGIC 2. AWS credentials for Bedrock, stored once in Unity Catalog. Either the UC secrets in
# MAGIC    `aws_uc_secrets` (create them with `ugbench store-aws-secrets`; you need READ SECRET), or a UC
# MAGIC    **service credential**. The notebook uses them for the direct Bedrock calls, and the model provider
# MAGIC    service uses the same ones for the gateway path. Reading UC secrets needs serverless environment version 4+.
# MAGIC 3. `CREATE SERVICE` (and `CREATE TABLE` for inference tables) on the target schema.
# MAGIC
# MAGIC Results go to `results/<run_id>/` next to this repo. They can also be copied to a UC volume (last cell).

# COMMAND ----------

# MAGIC %pip install -q --upgrade "boto3>=1.35" "botocore>=1.35" httpx pyyaml "databricks-sdk>=0.160" "matplotlib>=3.10" "strands-agents[anthropic]>=1.60" "anthropic[bedrock]>=0.70"
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("config", "config.yaml", "Config file (relative to repo root)")
dbutils.widgets.text("bedrock_service_credential", "", "UC service credential for direct Bedrock calls")
dbutils.widgets.dropdown("aws_creds", "uc_secrets", ["uc_secrets", "service_credential", "secret_scope"],
                         "Direct-Bedrock creds: UC secrets (aws_uc_secrets in config) / service credential / scope")
dbutils.widgets.text("aws_secret_scope", "", "Only for aws_creds=secret_scope")
dbutils.widgets.text("rounds", "1000", "Measured rounds")
dbutils.widgets.text("run_id", "", "Optional run id")
dbutils.widgets.text("qps", "0", "Open-loop total QPS (0 = closed loop)")
dbutils.widgets.text("duration_s", "", "Seconds at qps (overrides rounds)")
dbutils.widgets.dropdown("arrival", "poisson", ["poisson", "uniform"], "Open-loop arrivals")
dbutils.widgets.dropdown("setup", "yes", ["yes", "no"], "Create/update UG services first")
dbutils.widgets.text("results_volume", "", "Optional: /Volumes/<cat>/<schema>/<vol> to copy results")
dbutils.widgets.dropdown("layer", "raw", ["raw", "strands", "both"], "Client stack: raw | strands (Strands + Anthropic SDK)")
dbutils.widgets.dropdown("workload", "tiny", ["tiny", "large_prompt", "realistic"], "Workload preset (tiny = 5 tokens, realistic = 300)")
dbutils.widgets.text("procs", "1", "Client processes for open-loop runs (int or 'auto')")
dbutils.widgets.text("health_n", "30", "Health gate: requests per target before the run (0 = 1 probe, no gate)")
# Set by the bundle job (from databricks.yml variables) so the job benchmarks exactly the services the bundle
# deployed. Empty = keep the values from the config file.
dbutils.widgets.text("ug_catalog", "", "Override: catalog of the UG services (+ the AWS UC secrets)")
dbutils.widgets.text("ug_schema", "", "Override: schema of the UG services (+ the AWS UC secrets)")
dbutils.widgets.text("ug_service_prefix", "", "Override: model services are <prefix>_<variant>")
dbutils.widgets.text("ug_provider_service", "", "Override: model provider service name")
dbutils.widgets.text("aws_region", "", "Override: Bedrock region")
dbutils.widgets.text("bedrock_model_id", "", "Override: Bedrock model / inference-profile ID")

# COMMAND ----------

import os
import sys
import time

nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/Workspace" + os.path.dirname(os.path.dirname(nb_path))
sys.path.insert(0, repo_root)
os.chdir(repo_root)

from ugbench.config import build_targets, load_config  # noqa: E402

cfg = load_config(dbutils.widgets.get("config"))
cfg["databricks"]["profile"] = None          # use the notebook's own auth
_ug = cfg["unity_gateway"]
if dbutils.widgets.get("bedrock_model_id"):
    from ugbench.config import set_model  # noqa: E402

    set_model(cfg, dbutils.widgets.get("bedrock_model_id"))   # first, so explicit names below win
for widget, key in (("ug_catalog", "catalog"), ("ug_schema", "schema"),
                    ("ug_service_prefix", "service_prefix"), ("ug_provider_service", "provider_service")):
    if dbutils.widgets.get(widget):
        _ug[key] = dbutils.widgets.get(widget)
        if key in ("catalog", "schema"):
            cfg["aws_uc_secrets"][key] = _ug[key]
if dbutils.widgets.get("aws_region"):
    cfg["aws"]["region"] = dbutils.widgets.get("aws_region")
from ugbench.models import describe  # noqa: E402

print("request fields:", describe(cfg))
print(f"Unity Gateway services: {_ug['catalog']}.{_ug['schema']}.{_ug['service_prefix']}_*  "
      f"provider: {_ug['provider_service']}  model: {_ug['model']}  region: {cfg['aws']['region']}")
cfg["aws"]["profile"] = None
mode = dbutils.widgets.get("aws_creds")
cfg["aws"]["use_uc_secrets"] = mode == "uc_secrets"
if mode == "service_credential":
    cfg["aws"]["service_credential"] = dbutils.widgets.get("bedrock_service_credential")
if mode == "secret_scope":
    cfg["aws"]["secret_scope"] = dbutils.widgets.get("aws_secret_scope")
cfg["layer"] = dbutils.widgets.get("layer")
from ugbench.config import apply_workload_profile  # noqa: E402

apply_workload_profile(cfg, dbutils.widgets.get("workload"))
cfg["run"]["rounds"] = int(dbutils.widgets.get("rounds"))
cfg["run"]["qps"] = float(dbutils.widgets.get("qps") or 0)
cfg["run"]["arrival"] = dbutils.widgets.get("arrival")
if dbutils.widgets.get("duration_s"):
    cfg["run"]["duration_s"] = float(dbutils.widgets.get("duration_s"))
cfg["run"]["output_dir"] = os.path.join(repo_root, "results")

for t in build_targets(cfg):
    print(f"{t.name:28s} {t.kind:22s} features={'+'.join(t.features) or '-'}")

# COMMAND ----------

if dbutils.widgets.get("setup") == "yes":
    from ugbench.setup_resources import setup_all

    setup_all(cfg)
    time.sleep(60)  # let gateway config propagate (rate limits take up to 60 s)

# COMMAND ----------

# Health gate: N requests per target, abort unless every target is 100% OK. Gateway config changes can take
# 5-30 min to reach every instance; benchmarking during that window measures propagation errors, not latency.
from collections import Counter  # noqa: E402

from ugbench.clients import make_client  # noqa: E402

n_health = int(dbutils.widgets.get("health_n") or 0)
unhealthy = []
for t in build_targets(cfg):
    c = make_client(t, cfg)
    st, errs = Counter(), Counter()
    for _ in range(max(n_health, 1)):
        try:
            r = c.call(-1, {"ugbench_run": "health"})
        except Exception as e:
            r = {"status": None, "error": f"{type(e).__name__}: {e}"}
        st[r.get("status")] += 1
        if r.get("status") != 200:
            errs[(r.get("error") or "")[:120]] += 1
    c.close()
    print(f"{t.name:28s} ok={st.get(200, 0)}/{max(n_health, 1)} {dict(st)}", *[f"\n    {k}x {e}" for e, k in errs.most_common(2)])
    if st.get(200, 0) != max(n_health, 1):
        unhealthy.append(t.name)
if unhealthy and n_health:
    raise RuntimeError(f"NOT HEALTHY: {unhealthy}. Wait for gateway config to propagate, then re-run.")
print("HEALTHY")

# COMMAND ----------

from ugbench.report import build_report  # noqa: E402
from ugbench.runner import run_benchmark, run_sharded  # noqa: E402

procs_w = dbutils.widgets.get("procs")
procs = min(8, os.cpu_count() or 1) if procs_w == "auto" else int(procs_w)
print(f"client machine: {os.cpu_count()} CPUs; using {procs} benchmark process(es)")
rid = dbutils.widgets.get("run_id") or None
if procs > 1 and cfg["run"]["qps"]:
    run_dir = run_sharded(cfg, build_targets(cfg), procs, run_id=rid)
else:
    run_dir = run_benchmark(cfg, build_targets(cfg), run_id=rid)
res = build_report(run_dir)
import pandas as pd  # noqa: E402

if pd.read_csv(os.path.join(run_dir, "raw.csv"))["ttft_ms"].notna().any():
    build_report(run_dir, metric="ttft_ms")
displayHTML(open(res["html"]).read())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: server-side breakdown
# MAGIC Run this **30-60 minutes later**, once `system.ai_gateway.usage` has caught up. It needs a SQL warehouse
# MAGIC and SELECT on the system table.

# COMMAND ----------

# from ugbench.server_side import fetch_server_side
# fetch_server_side(cfg, run_dir, warehouse_id="<warehouse-id>")
# displayHTML(open(build_report(run_dir)["html"]).read())

# COMMAND ----------

vol = dbutils.widgets.get("results_volume")
if vol:
    import shutil

    dst = os.path.join(vol, os.path.basename(run_dir))
    shutil.copytree(run_dir, dst, dirs_exist_ok=True)
    print("copied to", dst)
