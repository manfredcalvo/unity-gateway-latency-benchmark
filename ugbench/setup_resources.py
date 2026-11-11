"""Create, update, and delete the Databricks resources used by the benchmark.

Unity Gateway (recommended, GA 2026-08-04):

    model provider service  <catalog>.<schema>.<provider_service>   (Bedrock creds + allowed model)
        ^
        |  one model service per variant, each with different gateway features:
    <catalog>.<schema>.<prefix>_baseline         no inference table, no rate limits
    <catalog>.<schema>.<prefix>_inference_table  payload logging to UC
    <catalog>.<schema>.<prefix>_rate_limits      QPM limits set far above the benchmark rate
    <catalog>.<schema>.<prefix>_all_features     both

Usage tracking (``system.ai_gateway.usage``) is always on in Unity Gateway, so
every UG variant includes it. To measure usage tracking on/off in isolation, use
the legacy serving-endpoint variants, where it is a separate toggle.

All calls go through the Databricks SDK's authenticated API client (REST + JSON),
so this works with any SDK version that can authenticate.
"""

from __future__ import annotations

import os
import time
from urllib.parse import quote

from databricks.sdk import WorkspaceClient

from .config import databricks_hosted_model, se_endpoint_name, ug_fqn, ug_service_id

UC = "/api/2.1/unity-catalog"


def workspace_client(cfg: dict) -> WorkspaceClient:
    kw = {k: v for k, v in cfg["databricks"].items() if v and k in ("profile", "host")}
    return WorkspaceClient(**kw)


def _exists(w: WorkspaceClient, path: str) -> bool:
    try:
        w.api_client.do("GET", path)
        return True
    except Exception as e:  # NotFound / ResourceDoesNotExist
        if "not found" in str(e).lower() or "does not exist" in str(e).lower() or "404" in str(e):
            return False
        raise


# --------------------------------------------------------------------------------------
# Unity Gateway
# --------------------------------------------------------------------------------------
def _provider_config(cfg: dict, w: WorkspaceClient | None = None) -> dict:
    ug = cfg["unity_gateway"]
    auth = ug["bedrock_auth"]
    direct: dict = {"region": cfg["aws"]["region"]}
    if auth.get("service_credential"):
        direct["service_credential"] = {"name": f"credentials/{auth['service_credential']}"}
    elif auth.get("use_uc_secrets"):
        from . import uc_secrets

        # The secret access key is referenced by name and never copied onto the service.
        # The key ID is a plain field on the provider service, so it is resolved once here.
        direct["aws_access_key"] = {
            "access_key_id": uc_secrets.read(cfg, "access_key_id", w),
            "secret_access_key": {"secret_reference": {"name": f"secrets/{uc_secrets.fqn(cfg, 'secret_access_key')}"}},
        }
    else:
        key_id = os.environ.get(auth["access_key_id_env"])
        if auth.get("secret_access_key_uc_secret"):
            secret = {"secret_reference": {"name": auth["secret_access_key_uc_secret"]}}
        else:
            val = os.environ.get(auth["secret_access_key_env"])
            if not val:
                raise SystemExit(
                    "Bedrock auth for Unity Gateway: set unity_gateway.bedrock_auth.service_credential "
                    f"(recommended) or export {auth['access_key_id_env']} / {auth['secret_access_key_env']}.")
            secret = {"plaintext": val}
        if not key_id:
            raise SystemExit(f"Missing env var {auth['access_key_id_env']}")
        direct["aws_access_key"] = {"access_key_id": key_id, "secret_access_key": secret}
    conf = {
        "provider_type": "EXTERNAL_MODEL_PROVIDER_TYPE_AMAZON_BEDROCK",
        "amazon_bedrock": {"direct": direct},
        "targets": [{"model": ug["model"], "native_api_types": ug["native_api_types"]}],
    }
    if ug.get("allow_all_targets"):
        conf["allow_all_targets"] = True
    return conf


def _model_service_config(cfg: dict, variant: str, feats: dict) -> dict:
    ug = cfg["unity_gateway"]
    conf: dict = {
        "routing": {"destinations": [{
            "name": "bedrock",
            "destination_type": "DESTINATION_TYPE_EXTERNAL_FOUNDATION_MODEL",
            "external_model_config": {
                "model_provider_service": f"model-provider-services/{ug_fqn(cfg, ug['provider_service'])}",
                "target": {"model": ug["model"]},
            },
        }]},
    }
    if feats.get("inference_table"):
        # The prefix is immutable once the table exists. Change inference_table_prefix when re-creating services
        # so they log to new tables instead of clashing with the old ones.
        base = ug.get("inference_table_prefix") or ug["service_prefix"]
        conf["inference_table"] = {"parent": f"schemas/{ug['catalog']}.{ug['schema']}",
                                   "table_name_prefix": f"{base}_{variant}"}
    if feats.get("rate_limits"):
        rl = ug["rate_limit"]
        limits = []
        for key in ("RATE_LIMIT_KEY_SERVICE", "RATE_LIMIT_KEY_USER_DEFAULT"):
            item = {"key": key, "renewal_period": "RATE_LIMIT_RENEWAL_PERIOD_MINUTE"}
            if rl.get("requests_per_minute"):
                item["requests"] = int(rl["requests_per_minute"])
            if rl.get("tokens_per_minute"):
                item["tokens"] = int(rl["tokens_per_minute"])
            limits.append(item)
        conf["rate_limits"] = limits
    return conf


def setup_unity_gateway(cfg: dict, w: WorkspaceClient, recreate: bool = False, update_auth: bool = False,
                        provider_only: bool = False) -> None:
    ug = cfg["unity_gateway"]
    parent = f"schemas/{ug['catalog']}.{ug['schema']}"
    prov_fqn = ug_fqn(cfg, ug["provider_service"])
    prov_path = f"{UC}/model-provider-services/{prov_fqn}"

    if recreate and _exists(w, prov_path):
        teardown_unity_gateway(cfg, w)

    if _exists(w, prov_path):
        if update_auth:
            print(f"[setup] provider service exists: {prov_fqn} (updating targets + Bedrock auth)")
            conf = _provider_config(cfg, w)
            w.api_client.do("PATCH", prov_path, query={"update_mask": "config.targets,config.provider"},  # mask path for the provider block is config.provider
                            body={"config": {"targets": conf["targets"], "amazon_bedrock": conf["amazon_bedrock"]}})
        else:
            print(f"[setup] provider service exists: {prov_fqn} (updating targets; pass --update-auth to change auth)")
            ug_ = cfg["unity_gateway"]
            w.api_client.do("PATCH", prov_path, query={"update_mask": "config.targets"},
                            body={"config": {"targets": [{"model": ug_["model"],
                                                          "native_api_types": ug_["native_api_types"]}]}})
    else:
        print(f"[setup] creating provider service {prov_fqn}")
        w.api_client.do("POST", f"{UC}/model-provider-services",
                        query={"parent": parent, "model_provider_service_id": ug["provider_service"]},
                        body={"comment": "ugbench: Amazon Bedrock provider for latency benchmark",
                              "config": _provider_config(cfg, w)})

    if provider_only:
        return  # the bundle (resources/ugbench_services.yml) manages the model services
    hosted = databricks_hosted_model(cfg)
    if hosted:
        sid = ug_service_id(cfg, "dbx_hosted")
        path = f"{UC}/model-services/{ug_fqn(cfg, sid)}"
        conf = {"routing": {"destinations": [{
            "name": "databricks_hosted", "destination_type": "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL",
            "pay_per_token_config": {"model": f"models/system.ai.{hosted}"}}]}}
        if _exists(w, path):
            w.api_client.do("PATCH", path, query={"update_mask": "config.routing.destinations"}, body={"config": conf})
        else:
            print(f"[setup] creating model service {ug_fqn(cfg, sid)} -> Databricks-hosted {hosted}")
            w.api_client.do("POST", f"{UC}/model-services", query={"parent": parent, "model_service_id": sid},
                            body={"comment": f"ugbench: Databricks-hosted {hosted} (pay-per-token)", "config": conf})
    for variant, feats in ug["variants"].items():
        sid = ug_service_id(cfg, variant)
        fqn = ug_fqn(cfg, sid)
        path = f"{UC}/model-services/{fqn}"
        conf = _model_service_config(cfg, variant, feats or {})
        comment = f"ugbench variant '{variant}': {', '.join(k for k, v in (feats or {}).items() if v) or 'no features'}"
        if _exists(w, path):
            print(f"[setup] model service exists: {fqn} (updating config)")
            mask = ["config.routing.destinations"]
            if "rate_limits" in conf:
                mask.append("config.rate_limits")
            # The inference table destination is immutable once created, so it is only set on create.
            w.api_client.do("PATCH", path, query={"update_mask": ",".join(mask)},
                            body={"config": {k: v for k, v in conf.items() if k != "inference_table"}})
        else:
            print(f"[setup] creating model service {fqn}")
            w.api_client.do("POST", f"{UC}/model-services",
                            query={"parent": parent, "model_service_id": sid},
                            body={"comment": comment, "config": conf})


def teardown_unity_gateway(cfg: dict, w: WorkspaceClient) -> None:
    ug = cfg["unity_gateway"]
    for variant in list(ug["variants"]) + ["dbx_hosted"]:
        fqn = ug_fqn(cfg, ug_service_id(cfg, variant))
        path = f"{UC}/model-services/{fqn}"
        if _exists(w, path):
            print(f"[teardown] deleting model service {fqn}")
            w.api_client.do("DELETE", path)
    prov = ug_fqn(cfg, ug["provider_service"])
    path = f"{UC}/model-provider-services/{prov}"
    if _exists(w, path):
        print(f"[teardown] deleting provider service {prov}")
        w.api_client.do("DELETE", path)
    print("[teardown] note: inference tables (*_payload) are left in place, so drop them manually if needed.")


# --------------------------------------------------------------------------------------
# Legacy serving endpoints (external model + endpoint-level AI Gateway)
# --------------------------------------------------------------------------------------
def _bedrock_external_model(cfg: dict) -> dict:
    se = cfg["serving_endpoints"]
    auth = se["auth"]
    bc: dict = {"aws_region": cfg["aws"]["region"], "bedrock_provider": se["bedrock_provider"]}
    if auth.get("uc_service_credential_name"):
        bc["uc_service_credential_name"] = auth["uc_service_credential_name"]
    elif auth.get("instance_profile_arn"):
        bc["instance_profile_arn"] = auth["instance_profile_arn"]
    elif auth.get("aws_access_key_id_secret") and auth.get("aws_secret_access_key_secret"):
        bc["aws_access_key_id"] = auth["aws_access_key_id_secret"]
        bc["aws_secret_access_key"] = auth["aws_secret_access_key_secret"]
    else:
        raise SystemExit("serving_endpoints.auth: set uc_service_credential_name, instance_profile_arn, "
                         "or both aws_*_secret references ({{secrets/scope/key}}).")
    return {"name": se["model"], "provider": "amazon-bedrock", "task": "llm/v1/chat",
            "amazon_bedrock_config": bc}


def _se_ai_gateway(cfg: dict, variant: str, feats: dict) -> dict:
    se = cfg["serving_endpoints"]
    gw: dict = {"usage_tracking_config": {"enabled": bool(feats.get("usage_tracking"))}}
    if feats.get("inference_tables"):
        it = se["inference_table"]
        if not (it.get("catalog") and it.get("schema")):
            raise SystemExit("serving_endpoints.inference_table.catalog/schema required for inference_tables variant")
        gw["inference_table_config"] = {"enabled": True, "catalog_name": it["catalog"], "schema_name": it["schema"],
                                        "table_name_prefix": se_endpoint_name(cfg, variant).replace("-", "_")}
    if feats.get("rate_limits"):
        calls = int(se["rate_limit"]["calls_per_minute"])
        gw["rate_limits"] = [{"calls": calls, "key": "endpoint", "renewal_period": "minute"},
                             {"calls": calls, "key": "user", "renewal_period": "minute"}]
    return gw


def setup_serving_endpoints(cfg: dict, w: WorkspaceClient) -> None:
    se = cfg["serving_endpoints"]
    em = _bedrock_external_model(cfg)
    for variant, feats in se["variants"].items():
        name = se_endpoint_name(cfg, variant)
        gw = _se_ai_gateway(cfg, variant, feats or {})
        path = f"/api/2.0/serving-endpoints/{quote(name)}"
        if _exists(w, path):
            print(f"[setup] serving endpoint exists: {name} (updating AI Gateway config)")
            w.api_client.do("PUT", f"{path}/ai-gateway", body=gw)
        else:
            print(f"[setup] creating serving endpoint {name}")
            w.api_client.do("POST", "/api/2.0/serving-endpoints", body={
                "name": name,
                "config": {"served_entities": [{"name": "bedrock", "external_model": em}]},
                "ai_gateway": gw,
                "tags": [{"key": "created_by", "value": "ugbench"}, {"key": "variant", "value": variant}],
            })
    _wait_ready(w, [se_endpoint_name(cfg, v) for v in se["variants"]])


def _wait_ready(w: WorkspaceClient, names: list[str], timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    pending = set(names)
    while pending and time.time() < deadline:
        for n in list(pending):
            st = w.api_client.do("GET", f"/api/2.0/serving-endpoints/{quote(n)}").get("state", {})
            if st.get("ready") == "READY" and st.get("config_update") in (None, "NOT_UPDATING"):
                print(f"[setup] {n}: READY")
                pending.discard(n)
        if pending:
            time.sleep(10)
    if pending:
        print(f"[setup] WARNING: not ready after {timeout_s}s: {sorted(pending)}")


def teardown_serving_endpoints(cfg: dict, w: WorkspaceClient) -> None:
    for variant in cfg["serving_endpoints"]["variants"]:
        name = se_endpoint_name(cfg, variant)
        path = f"/api/2.0/serving-endpoints/{quote(name)}"
        if _exists(w, path):
            print(f"[teardown] deleting serving endpoint {name}")
            w.api_client.do("DELETE", path)


def setup_all(cfg: dict, recreate: bool = False, update_auth: bool = False, provider_only: bool = False) -> None:
    w = workspace_client(cfg)
    if cfg["unity_gateway"].get("enabled"):
        setup_unity_gateway(cfg, w, recreate=recreate, update_auth=update_auth, provider_only=provider_only)
    if cfg["serving_endpoints"].get("enabled") and not provider_only:
        setup_serving_endpoints(cfg, w)
    # Feature changes take ~20-60 s to propagate (rate limits up to 60 s).
    print("[setup] done. Wait ~60 s before benchmarking so gateway config has propagated.")


def teardown_all(cfg: dict) -> None:
    w = workspace_client(cfg)
    if cfg["unity_gateway"].get("enabled"):
        teardown_unity_gateway(cfg, w)
    if cfg["serving_endpoints"].get("enabled"):
        teardown_serving_endpoints(cfg, w)
