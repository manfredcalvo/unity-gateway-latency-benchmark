"""Configuration loading and target expansion.

A *target* is one thing we send requests to (direct Bedrock, one Unity Gateway
model service variant, one legacy serving endpoint variant, ...). The runner
interleaves requests across all enabled targets so that they experience the same
network/provider conditions.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

# Workload presets (select with workload.profile or `--workload`). "realistic" asks for more than max_tokens,
# so every response stops at exactly max_tokens: identical output length on every path.
_CONTEXT = (
    "Background notes for the support assistant. Our platform stores customer orders in a regional database "
    "cluster with three replicas. Writes go to the primary in the customer's home region and are replicated "
    "asynchronously to two secondaries. Read traffic is served from the nearest healthy replica. When a "
    "replica falls behind by more than five seconds it is removed from the read pool until it catches up. "
    "Failover to a secondary is automatic after thirty seconds of primary unavailability and requires a "
    "quorum of health checks from at least two independent monitors. Backups are taken every six hours and "
    "retained for thirty five days; point in time recovery is available for the last seven days. "
)

WORKLOAD_PROFILES: dict[str, dict] = {
    "tiny": {"prompt": "Reply with exactly one word: OK", "max_tokens": 5},
    # ~565 input tokens (measured), same 5-token reply: measures the effect of request size on the gateway path
    # while staying well inside a 5M tokens/min quota at 100 QPS.
    "large_prompt": {"prompt": _CONTEXT * 4 + "\n\nIgnore the notes above. Reply with exactly one word: OK",
                     "max_tokens": 5},
    "realistic": {
        "prompt": ("Write a detailed, roughly 600-word technical explanation of how a distributed database "
                   "replicates data across regions, covering consensus, conflict resolution and failure "
                   "handling. Write in plain paragraphs without headings or lists."),
        "max_tokens": 300,
    },
}

DEFAULTS: dict[str, Any] = {
    "run_name": "ugbench",
    # Which client stack to benchmark:
    #   raw     : boto3 Converse (direct) vs httpx + unified chat completions / Anthropic messages (gateway)
    #   strands : Strands Agents + Anthropic SDK on both paths (AsyncAnthropicBedrock vs AsyncAnthropic -> gateway)
    #   both    : all of the above in one interleaved run
    "layer": "raw",
    "databricks": {"profile": None, "host": None},
    "aws": {
        "region": "us-east-1",
        "profile": None,
        # Model ID / inference-profile ID used for the direct Bedrock call.
        "bedrock_model_id": None,
        # Optional: in a Databricks notebook, use a UC service credential for boto3.
        "service_credential": None,
        # Optional: read the AWS key pair from the UC secrets in `aws_uc_secrets` (recommended in Databricks).
        "use_uc_secrets": False,
        # Legacy: workspace secret scope with keys aws_access_key_id / aws_secret_access_key.
        "secret_scope": None,
    },
    # One AWS key pair stored once as UC secrets, used by the provider service (by reference)
    # and by the benchmark's direct-Bedrock client. catalog/schema default to unity_gateway's.
    "aws_uc_secrets": {
        "catalog": None,
        "schema": None,
        "access_key_id": "ugbench_aws_access_key_id",
        "secret_access_key": "ugbench_aws_secret_access_key",
    },
    "workload": {
        "profile": None,            # tiny | realistic: overrides prompt/max_tokens when set
        "prompt": "Reply with exactly one word: OK",
        "system": None,
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": False,
    },
    "run": {
        "rounds": 500,
        "warmup_rounds": 20,
        "concurrency": 1,
        "pause_between_requests_ms": 0,
        "timeout_s": 60,
        "shuffle": True,
        "seed": 42,
        "output_dir": "results",
        "tag_requests": True,
        # Open-loop fixed-rate mode: set qps > 0 (total across all targets).
        "qps": 0,
        "arrival": "uniform",       # uniform | poisson
        "max_in_flight": 512,
        "duration_s": None,         # if set (with qps), overrides rounds
        "warmup_s": 10,
    },
    "unity_gateway": {
        "enabled": True,
        "catalog": None,
        "schema": None,
        "provider_service": "ugbench_bedrock",
        "service_prefix": "ugbench",
        "inference_table_prefix": None,   # defaults to service_prefix; tables are <prefix>_<variant>_payload
        # Exactly one of service_credential / access_key must be set for `setup`.
        "bedrock_auth": {
            # Recommended without a service credential: reference the UC secrets in `aws_uc_secrets`.
            "use_uc_secrets": False,
            "service_credential": None,
            "access_key_id_env": "UGBENCH_AWS_ACCESS_KEY_ID",
            "secret_access_key_env": "UGBENCH_AWS_SECRET_ACCESS_KEY",
            "secret_access_key_uc_secret": None,  # secrets/{catalog}.{schema}.{secret}
        },
        # Provider-side model id; defaults to aws.bedrock_model_id.
        "model": None,
        "native_api_types": ["anthropic/v1/messages"],
        "allow_all_targets": False,
        # How clients call the model services:
        #   chat_completions   -> /ai-gateway/mlflow/v1/chat/completions (unified, provider-agnostic)
        #   anthropic_messages -> /ai-gateway/anthropic/v1/messages   (native Claude API)
        "api": "chat_completions",
        "include_provider_service_direct": True,
        "provider_api": "anthropic_messages",
        "rate_limit": {"requests_per_minute": 100000, "tokens_per_minute": None},
        "variants": {
            "baseline": {},
            "inference_table": {"inference_table": True},
            "rate_limits": {"rate_limits": True},
            "all_features": {"inference_table": True, "rate_limits": True},
        },
    },
    "serving_endpoints": {
        "enabled": False,
        "endpoint_prefix": "ugbench",
        "bedrock_provider": "anthropic",
        "model": None,
        # Exactly one: uc_service_credential_name | instance_profile_arn | secret refs
        "auth": {
            "uc_service_credential_name": None,
            "instance_profile_arn": None,
            "aws_access_key_id_secret": None,       # "{{secrets/scope/key}}"
            "aws_secret_access_key_secret": None,   # "{{secrets/scope/key}}"
        },
        "inference_table": {"catalog": None, "schema": None},
        "rate_limit": {"calls_per_minute": 100000},
        "variants": {
            "baseline": {},
            "usage_tracking": {"usage_tracking": True},
            "inference_tables": {"inference_tables": True},
            "rate_limits": {"rate_limits": True},
            "all_features": {"usage_tracking": True, "inference_tables": True, "rate_limits": True},
        },
    },
    # Synthetic targets for offline testing of the pipeline (no network).
    "mock": {"enabled": False, "targets": {}},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        # `variants` / mock targets are replaced wholesale so users can drop entries.
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("variants", "targets"):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, user)
    # Environment overrides that are handy in CI / notebooks.
    if os.environ.get("UGBENCH_DATABRICKS_PROFILE"):
        cfg["databricks"]["profile"] = os.environ["UGBENCH_DATABRICKS_PROFILE"]
    ug = cfg["unity_gateway"]
    ug["model"] = ug["model"] or cfg["aws"]["bedrock_model_id"]
    se = cfg["serving_endpoints"]
    se["model"] = se["model"] or cfg["aws"]["bedrock_model_id"]
    if cfg["workload"].get("profile"):
        apply_workload_profile(cfg, cfg["workload"]["profile"])
    return cfg


def apply_workload_profile(cfg: dict, name: str) -> dict:
    if name not in WORKLOAD_PROFILES:
        raise ValueError(f"unknown workload profile {name!r}; choose from {sorted(WORKLOAD_PROFILES)}")
    cfg["workload"].update(WORKLOAD_PROFILES[name])
    cfg["workload"]["profile"] = name
    return cfg


@dataclass
class Target:
    name: str          # unique label used in results, e.g. "ug:baseline"
    kind: str          # bedrock | ug_model_service | ug_provider_service | serving_endpoint | mock
    group: str         # "direct" | "unity_gateway" | "serving_endpoint" | "mock"
    features: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


def ug_service_id(cfg: dict, variant: str) -> str:
    return f"{cfg['unity_gateway']['service_prefix']}_{variant}"


def ug_fqn(cfg: dict, leaf: str) -> str:
    ug = cfg["unity_gateway"]
    return f"{ug['catalog']}.{ug['schema']}.{leaf}"


def se_endpoint_name(cfg: dict, variant: str) -> str:
    return f"{cfg['serving_endpoints']['endpoint_prefix']}-{variant.replace('_', '-')}"


def _strands_targets(cfg: dict) -> list[Target]:
    """Strands + Anthropic SDK on both paths. Names mirror the raw layer with a ``strands:`` prefix."""
    ug = cfg["unity_gateway"]
    out = [Target("strands:direct", "strands", "direct",
                  params={"backend": "bedrock", "model_id": cfg["aws"]["bedrock_model_id"]})]
    if ug.get("enabled"):
        if ug.get("include_provider_service_direct"):
            out.append(Target("strands:ug_provider", "strands", "unity_gateway", ["provider_service"],
                              params={"backend": "ug_provider_service", "model_id": ug["model"],
                                      "service": ug_fqn(cfg, ug["provider_service"])}))
        for variant, feats in ug["variants"].items():
            out.append(Target(f"strands:ug:{variant}", "strands", "unity_gateway",
                              [k for k, v in (feats or {}).items() if v],
                              params={"backend": "ug_model_service",
                                      "model_id": ug_fqn(cfg, ug_service_id(cfg, variant))}))
    return out


def build_targets(cfg: dict, only: list[str] | None = None) -> list[Target]:
    layer = cfg.get("layer", "raw")
    if layer not in ("raw", "strands", "both"):
        raise ValueError(f"layer must be raw | strands | both, got {layer!r}")
    targets = _build_raw_targets(cfg) if layer in ("raw", "both") else []
    if layer in ("strands", "both"):
        if not cfg["aws"].get("bedrock_model_id"):
            raise ValueError("layer=strands needs aws.bedrock_model_id")
        targets += _strands_targets(cfg)
    if only:
        wanted = set(only)
        targets = [t for t in targets if t.name in wanted or t.group in wanted]
    if not targets:
        raise ValueError("No targets enabled. Check aws.bedrock_model_id / unity_gateway / serving_endpoints.")
    return targets


def _build_raw_targets(cfg: dict) -> list[Target]:
    targets: list[Target] = []
    if cfg["aws"].get("bedrock_model_id"):
        targets.append(Target("bedrock_direct", "bedrock", "direct",
                              params={"model_id": cfg["aws"]["bedrock_model_id"]}))

    ug = cfg["unity_gateway"]
    if ug.get("enabled"):
        if not (ug.get("catalog") and ug.get("schema")):
            raise ValueError("unity_gateway.catalog and unity_gateway.schema are required")
        if ug.get("include_provider_service_direct"):
            targets.append(Target(
                "ug_provider:direct", "ug_provider_service", "unity_gateway", ["provider_service"],
                params={"service": ug_fqn(cfg, ug["provider_service"]), "model": ug["model"],
                        "api": ug["provider_api"]}))
        for variant, feats in ug["variants"].items():
            features = [k for k, v in (feats or {}).items() if v]
            targets.append(Target(
                f"ug:{variant}", "ug_model_service", "unity_gateway", features,
                params={"service": ug_fqn(cfg, ug_service_id(cfg, variant)), "api": ug["api"]}))

    se = cfg["serving_endpoints"]
    if se.get("enabled"):
        for variant, feats in se["variants"].items():
            features = [k for k, v in (feats or {}).items() if v]
            targets.append(Target(
                f"se:{variant}", "serving_endpoint", "serving_endpoint", features,
                params={"endpoint": se_endpoint_name(cfg, variant)}))

    mock = cfg.get("mock") or {}
    if mock.get("enabled"):
        for name, spec in (mock.get("targets") or {}).items():
            targets.append(Target(name, "mock", spec.get("group", "mock"),
                                  spec.get("features", []), params=spec))
    return targets
