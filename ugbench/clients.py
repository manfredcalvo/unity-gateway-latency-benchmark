"""Clients for every target type.

Timing rules, applied the same way to every client so the comparison is fair:

* Connections are created once and reused (HTTP keep-alive). TLS handshakes are
  paid during warmup, never inside a measured request.
* Credentials are resolved *before* the timer starts. The Databricks SDK refreshes
  OAuth tokens itself, and that refresh must not show up as gateway latency.
  SigV4 signing for Bedrock happens inside boto3's call. It is part of the real
  client cost and takes well under 1 ms.
* Client-side retries are disabled everywhere (boto3 max_attempts=1, no httpx
  retries), so a hidden retry can never inflate a sample.
* ``latency_ms`` is wall-clock from just before the request is sent until the full
  response body has been read (``time.perf_counter_ns``).
* ``ttfb_ms`` is when response headers arrived. With ``stream: true``, ``ttft_ms``
  is when the first content token arrived.
"""

from __future__ import annotations

import json
import math
import random
import time
import zlib
from typing import Any

import httpx

from .config import Target

REQUEST_ID_HEADERS = ("x-request-id", "x-databricks-request-id", "x-amzn-requestid", "request-id")


def _ms(ns: int) -> float:
    return ns / 1e6


def _request_id(headers) -> str | None:
    for h in REQUEST_ID_HEADERS:
        if h in headers:
            return headers[h]
    return None


class BaseClient:
    def __init__(self, target: Target, cfg: dict):
        self.target = target
        self.cfg = cfg
        self.workload = cfg["workload"]
        self.timeout = cfg["run"]["timeout_s"]

    def call(self, round_id: int, tags: dict[str, str]) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------------------
# AWS credentials, shared by every client that calls Bedrock directly
# --------------------------------------------------------------------------------------
def aws_session(cfg: dict):
    """boto3 Session from the configured credential source (first match wins):
    UC service credential > UC secrets > workspace secret scope > boto3 profile/default chain."""
    import boto3

    aws = cfg["aws"]
    if aws.get("_static_credentials"):
        # Set in memory by run_sharded: credentials resolved once in the parent process.
        sc = aws["_static_credentials"]
        return boto3.Session(aws_access_key_id=sc["access_key"], aws_secret_access_key=sc["secret_key"],
                             aws_session_token=sc.get("token"), region_name=aws["region"])
    if aws.get("service_credential"):
        # Inside a Databricks notebook/job: mint short-lived creds from a UC service credential.
        from databricks.sdk.runtime import dbutils  # type: ignore

        return boto3.Session(
            botocore_session=dbutils.credentials.getServiceCredentialsProvider(aws["service_credential"]),
            region_name=aws["region"])
    if aws.get("use_uc_secrets"):
        # One key pair stored as UC secrets (see ugbench/uc_secrets.py).
        from . import uc_secrets

        return boto3.Session(
            aws_access_key_id=uc_secrets.read(cfg, "access_key_id"),
            aws_secret_access_key=uc_secrets.read(cfg, "secret_access_key"),
            region_name=aws["region"])
    if aws.get("secret_scope"):
        # Legacy: key pair from a workspace secret scope.
        from databricks.sdk.runtime import dbutils  # type: ignore

        return boto3.Session(
            aws_access_key_id=dbutils.secrets.get(aws["secret_scope"], aws.get("access_key_id_key", "aws_access_key_id")).strip(),
            aws_secret_access_key=dbutils.secrets.get(aws["secret_scope"],
                                                      aws.get("secret_access_key_key", "aws_secret_access_key")).strip(),
            region_name=aws["region"])
    return boto3.Session(profile_name=aws.get("profile"), region_name=aws["region"])


# --------------------------------------------------------------------------------------
# Direct Amazon Bedrock (bedrock-runtime Converse API)
# --------------------------------------------------------------------------------------
class BedrockClient(BaseClient):
    def __init__(self, target: Target, cfg: dict):
        super().__init__(target, cfg)
        from botocore.config import Config as BotoConfig

        session = aws_session(cfg)
        self.client = session.client(
            "bedrock-runtime",
            config=BotoConfig(
                retries={"max_attempts": 1, "mode": "standard"},
                max_pool_connections=max(10, cfg["run"]["concurrency"] * 2),
                connect_timeout=10, read_timeout=self.timeout, tcp_keepalive=True))
        self.model_id = target.params["model_id"]

    def _kwargs(self) -> dict:
        w = self.workload
        kw: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": w["prompt"]}]}],
            "inferenceConfig": {"maxTokens": w["max_tokens"], "temperature": w["temperature"]},
        }
        if w.get("system"):
            kw["system"] = [{"text": w["system"]}]
        return kw

    def call(self, round_id, tags):
        kw = self._kwargs()
        out: dict[str, Any] = {}
        if not self.workload["stream"]:
            t0 = time.perf_counter_ns()
            resp = self.client.converse(**kw)
            t1 = time.perf_counter_ns()
            out["latency_ms"] = _ms(t1 - t0)
            out["output_tokens"] = resp.get("usage", {}).get("outputTokens")
            out["provider_latency_ms"] = resp.get("metrics", {}).get("latencyMs")
        else:
            t0 = time.perf_counter_ns()
            resp = self.client.converse_stream(**kw)
            ttfb = time.perf_counter_ns()
            ttft = None
            for ev in resp["stream"]:
                if ttft is None and "contentBlockDelta" in ev:
                    ttft = time.perf_counter_ns()
                if "metadata" in ev:
                    out["output_tokens"] = ev["metadata"].get("usage", {}).get("outputTokens")
                    out["provider_latency_ms"] = ev["metadata"].get("metrics", {}).get("latencyMs")
            t1 = time.perf_counter_ns()
            out["latency_ms"] = _ms(t1 - t0)
            out["ttfb_ms"] = _ms(ttfb - t0)
            out["ttft_ms"] = _ms(ttft - t0) if ttft else None
        out["status"] = resp["ResponseMetadata"]["HTTPStatusCode"]
        out["request_id"] = resp["ResponseMetadata"].get("RequestId")
        return out


# --------------------------------------------------------------------------------------
# Databricks HTTP targets (Unity Gateway model / provider services, legacy endpoints)
# --------------------------------------------------------------------------------------
class DatabricksHttpClient(BaseClient):
    _shared: dict[str, Any] = {}

    def __init__(self, target: Target, cfg: dict):
        super().__init__(target, cfg)
        from databricks.sdk.core import Config

        key = f"{cfg['databricks'].get('profile')}|{cfg['databricks'].get('host')}"
        if key not in DatabricksHttpClient._shared:
            kwargs = {k: v for k, v in cfg["databricks"].items() if v and k in ("profile", "host")}
            DatabricksHttpClient._shared[key] = Config(**kwargs)
        self.dbx = DatabricksHttpClient._shared[key]
        self.host = self.dbx.host.rstrip("/")
        n = max(4, cfg["run"]["concurrency"] * 2)
        # One pooled HTTP/1.1 client per target. Keep-alive is on by default.
        self.http = httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=10),
            limits=httpx.Limits(max_connections=n, max_keepalive_connections=n, keepalive_expiry=120))
        self.url, self.body, self.extra_headers, self.style = self._build()

    # -- request construction ---------------------------------------------------------
    def _messages(self) -> list[dict]:
        return [{"role": "user", "content": self.workload["prompt"]}]

    def _chat_body(self, model: str | None) -> dict:
        w = self.workload
        msgs = self._messages()
        if w.get("system"):
            msgs = [{"role": "system", "content": w["system"]}] + msgs
        body = {"messages": msgs, "max_tokens": w["max_tokens"], "temperature": w["temperature"]}
        if model:
            body["model"] = model
        if w["stream"]:
            body["stream"] = True
        return body

    def _anthropic_body(self, model: str) -> dict:
        w = self.workload
        body = {"model": model, "max_tokens": w["max_tokens"], "temperature": w["temperature"],
                "messages": self._messages()}
        if w.get("system"):
            body["system"] = w["system"]
        if w["stream"]:
            body["stream"] = True
        return body

    def _build(self):
        p = self.target.params
        kind = self.target.kind
        if kind == "ug_model_service":
            if p["api"] == "anthropic_messages":
                return (f"{self.host}/ai-gateway/anthropic/v1/messages", self._anthropic_body(p["service"]),
                        {"anthropic-version": "2023-06-01"}, "anthropic")
            return (f"{self.host}/ai-gateway/mlflow/v1/chat/completions", self._chat_body(p["service"]),
                    {}, "chat")
        if kind == "ug_provider_service":
            hdr = {"Databricks-Model-Provider-Service": p["service"]}
            if p["api"] == "anthropic_messages":
                hdr["anthropic-version"] = "2023-06-01"
                return (f"{self.host}/ai-gateway/anthropic/v1/messages", self._anthropic_body(p["model"]),
                        hdr, "anthropic")
            return (f"{self.host}/ai-gateway/openai/v1/chat/completions", self._chat_body(p["model"]),
                    hdr, "chat")
        if kind == "serving_endpoint":
            return (f"{self.host}/serving-endpoints/{p['endpoint']}/invocations", self._chat_body(None),
                    {}, "chat")
        raise ValueError(kind)

    # -- execution --------------------------------------------------------------------
    def headers(self, tags: dict[str, str]) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self.extra_headers}
        h.update(self.dbx.authenticate())  # resolved/refreshed OUTSIDE the timed section
        if tags and self.target.group == "unity_gateway" and self.cfg["run"]["tag_requests"]:
            h["Databricks-Ai-Gateway-Request-Tags"] = json.dumps(tags, separators=(",", ":"))
        return h

    def call(self, round_id, tags):
        headers = self.headers(tags)
        payload = json.dumps(self.body).encode()
        out: dict[str, Any] = {}
        ttft = None
        t0 = time.perf_counter_ns()
        with self.http.stream("POST", self.url, content=payload, headers=headers) as r:
            ttfb = time.perf_counter_ns()
            if self.workload["stream"] and r.status_code == 200:
                chunks = []
                for line in r.iter_lines():
                    if ttft is None and line.startswith("data:") and self._is_content(line[5:].strip()):
                        ttft = time.perf_counter_ns()
                    chunks.append(line)
                raw = "\n".join(chunks).encode()
            else:
                raw = r.read()
            t1 = time.perf_counter_ns()
            status = r.status_code
            rid = _request_id(r.headers)
        out.update(latency_ms=_ms(t1 - t0), ttfb_ms=_ms(ttfb - t0),
                   ttft_ms=_ms(ttft - t0) if ttft else None, status=status, request_id=rid)
        if status != 200:
            out["error"] = raw[:500].decode(errors="replace")
        elif not self.workload["stream"]:
            out["output_tokens"] = self._output_tokens(raw)
        return out

    def _is_content(self, data: str) -> bool:
        if not data or data == "[DONE]":
            return False
        try:
            ev = json.loads(data)
        except ValueError:
            return False
        if self.style == "anthropic":
            return ev.get("type") == "content_block_delta"
        ch = ev.get("choices") or []
        return bool(ch and (ch[0].get("delta") or {}).get("content"))

    def _output_tokens(self, raw: bytes):
        try:
            usage = json.loads(raw).get("usage") or {}
        except ValueError:
            return None
        return usage.get("completion_tokens", usage.get("output_tokens"))

    def close(self):
        self.http.close()


# --------------------------------------------------------------------------------------
# Mock target: synthetic latencies for offline pipeline tests and demos.
# --------------------------------------------------------------------------------------
class MockClient(BaseClient):
    """Simulates ``provider_latency(round) + overhead``. The provider part is shared
    by every target within a round, which mimics real provider drift and lets the
    paired analysis be exercised.

    params: base_ms, sigma (per-round lognormal), call_sigma (per-call lognormal), overhead_ms, overhead_jitter_ms, error_rate, sleep
    """

    def call(self, round_id, tags):
        p = self.target.params
        rng_round = random.Random(10_000 + round_id)            # shared across targets
        provider = p.get("base_ms", 400) * math.exp(rng_round.gauss(0, p.get("sigma", 0.25)))
        rng = random.Random(zlib.crc32(f"{self.target.name}:{round_id}".encode()))
        overhead = max(0.0, rng.gauss(p.get("overhead_ms", 0), p.get("overhead_jitter_ms", 0)))
        # Independent per-call model jitter: real generations vary call to call even in the same round.
        lat = provider * math.exp(rng.gauss(0, p.get("call_sigma", 0.0))) + overhead + abs(rng.gauss(0, 2))
        if p.get("sleep"):
            time.sleep(lat / 1000)
        if rng.random() < p.get("error_rate", 0.0):
            return {"latency_ms": lat, "status": 503, "error": "mock error"}
        return {"latency_ms": lat, "ttfb_ms": lat, "status": 200, "output_tokens": 1,
                "provider_latency_ms": provider}


def make_client(target: Target, cfg: dict) -> BaseClient:
    if target.kind == "strands":
        from .strands_client import StrandsClient

        return StrandsClient(target, cfg)
    if target.kind == "bedrock":
        return BedrockClient(target, cfg)
    if target.kind in ("ug_model_service", "ug_provider_service", "serving_endpoint"):
        return DatabricksHttpClient(target, cfg)
    if target.kind == "mock":
        return MockClient(target, cfg)
    raise ValueError(f"unknown target kind {target.kind}")
