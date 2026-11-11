"""Strands Agents + Anthropic SDK targets: the customer's real stack.

Both paths run identical Strands code (``AnthropicModel`` → ``client.messages.stream``). Only the Anthropic
client underneath differs:

* direct  : ``anthropic.AsyncAnthropicBedrock``  → Bedrock InvokeModelWithResponseStream (Anthropic Messages body)
* gateway : ``anthropic.AsyncAnthropic(base_url=<workspace>/ai-gateway/anthropic)``
            - model service    : ``model_id`` = UC model service name (``catalog.schema.service``)
            - provider service : ``model_id`` = Bedrock model ID, plus header ``Databricks-Model-Provider-Service``

Why not ``agent("...")``? Strands' sync ``Agent.__call__`` runs every call through ``asyncio.run`` in a new
thread (``strands/_async.py``). An async HTTP pool can't survive that, so each call would pay a fresh TCP+TLS
handshake. Instead, each benchmark worker thread keeps one **persistent event loop** with its own model and
client (connections are reused), and drives ``agent.stream_async()`` with ``loop.run_until_complete``.
A fresh ``Agent`` is built per request, before the timer starts, so conversation history never grows.

Timing: ``latency_ms`` covers the whole agent invocation until the stream ends. ``ttft_ms`` is the first
``"data"`` (text delta) event. Both include Strands' own per-event processing, the same on both paths.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import threading
import time
from typing import Any

import httpx

from .clients import BaseClient, _anthropic_extras, aws_session
from .config import Target

_TAGS: contextvars.ContextVar[dict | None] = contextvars.ContextVar("ugbench_tags", default=None)
logging.getLogger("strands").setLevel(logging.WARNING)


class _DatabricksAuth(httpx.Auth):
    """Adds a fresh Databricks bearer token (SDK-cached, refreshed when needed) and the per-request
    usage tags (``Databricks-Ai-Gateway-Request-Tags``) to every gateway request."""

    def __init__(self, dbx_config, add_tags: bool):
        self.dbx = dbx_config
        self.add_tags = add_tags

    def auth_flow(self, request: httpx.Request):
        request.headers.update(self.dbx.authenticate())
        tags = _TAGS.get()
        if self.add_tags and tags:
            request.headers["Databricks-Ai-Gateway-Request-Tags"] = json.dumps(tags, separators=(",", ":"))
        yield request


def _strands_params(w: dict) -> dict:
    params = _anthropic_extras(w)
    if w.get("temperature") is not None:
        params["temperature"] = w["temperature"]
    return params


def _status_of(exc: BaseException) -> int | None:
    """Find an HTTP status in an exception chain (Strands wraps Anthropic SDK errors)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        code = getattr(exc, "status_code", None)
        if isinstance(code, int):
            return code
        exc = exc.__cause__ or exc.__context__
    return None


class StrandsClient(BaseClient):
    _dbx_shared: dict[str, Any] = {}
    _test_transport: httpx.AsyncBaseTransport | None = None  # tests inject an httpx.MockTransport here

    def __init__(self, target: Target, cfg: dict):
        super().__init__(target, cfg)
        self.backend = target.params["backend"]          # bedrock | ug_model_service | ug_provider_service
        self.model_id = target.params["model_id"]
        n = int(cfg["run"].get("max_in_flight", 512)) if cfg["run"].get("qps") else cfg["run"]["concurrency"]
        self.pool = max(2, min(n, 64))                  # per-thread pool; each thread has 1 request in flight
        self._local = threading.local()
        if self.backend == "bedrock":
            creds = aws_session(cfg).get_credentials()
            if creds is None:
                raise SystemExit("no AWS credentials for the direct Bedrock path")
            self._aws = creds.get_frozen_credentials()
        else:
            from databricks.sdk.core import Config

            key = f"{cfg['databricks'].get('profile')}|{cfg['databricks'].get('host')}"
            if key not in StrandsClient._dbx_shared:
                kwargs = {k: v for k, v in cfg["databricks"].items() if v and k in ("profile", "host")}
                StrandsClient._dbx_shared[key] = Config(**kwargs)
            self.dbx = StrandsClient._dbx_shared[key]
            self.dbx.authenticate()  # warm the token cache once, outside any timed request

    # -- per-thread state --------------------------------------------------------------
    def _anthropic_client(self):
        import anthropic

        http = dict(timeout=httpx.Timeout(self.timeout, connect=10),
                    limits=httpx.Limits(max_connections=self.pool, max_keepalive_connections=self.pool,
                                        keepalive_expiry=120))
        if self._test_transport is not None:
            http["transport"] = self._test_transport
        if self.backend == "bedrock":
            return anthropic.AsyncAnthropicBedrock(
                aws_access_key=self._aws.access_key, aws_secret_key=self._aws.secret_key,
                aws_session_token=self._aws.token, aws_region=self.cfg["aws"]["region"],
                max_retries=0, timeout=self.timeout, http_client=httpx.AsyncClient(**http))
        headers = {}
        if self.backend == "ug_provider_service":
            headers["Databricks-Model-Provider-Service"] = self.target.params["service"]
        auth = _DatabricksAuth(self.dbx, add_tags=bool(self.cfg["run"].get("tag_requests", True)))
        return anthropic.AsyncAnthropic(
            api_key="unused",  # auth is the Databricks bearer token added by _DatabricksAuth
            base_url=f"{self.dbx.host.rstrip('/')}/ai-gateway/anthropic",
            default_headers=headers, max_retries=0, timeout=self.timeout,
            http_client=httpx.AsyncClient(auth=auth, **http))

    def _state(self):
        st = getattr(self._local, "st", None)
        if st is None:
            from strands.models.anthropic import AnthropicModel

            loop = asyncio.new_event_loop()
            w = self.workload
            model = AnthropicModel(client_args={"api_key": "unused"}, model_id=self.model_id,
                                   max_tokens=int(w["max_tokens"]), params=_strands_params(w))
            # Equivalent to the newer AnthropicModel(client=...) argument; works on strands-agents 1.x.
            model.client = self._anthropic_client()
            st = self._local.st = (loop, model)
        return st

    # -- one request -----------------------------------------------------------------
    def call(self, round_id, tags):
        # A Databricks/Jupyter notebook's main thread already runs an asyncio loop (IPython), so we can't
        # drive our own loop there. Hand the request to a dedicated helper thread with its own persistent
        # loop. Benchmark worker threads have no running loop and take the direct path below.
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False
        if in_loop:
            if getattr(self, "_helper", None) is None:
                from concurrent.futures import ThreadPoolExecutor

                self._helper = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ugb-strands")
            return self._helper.submit(self._call, round_id, tags).result()
        return self._call(round_id, tags)

    def _call(self, round_id, tags):
        from strands import Agent

        loop, model = self._state()
        w = self.workload
        # retry_strategy=None: Strands otherwise retries throttling/transient errors up to 6x with
        # 4-240 s backoff, which would silently inflate latency samples. Built outside the timer.
        agent = Agent(model=model, callback_handler=None, system_prompt=w.get("system"), retry_strategy=None)
        prompt = w["prompt"]

        async def run():
            from strands.types.exceptions import MaxTokensReachedException

            ttft = None
            try:
                async for ev in agent.stream_async(prompt):
                    if ttft is None and ev.get("data"):
                        ttft = time.perf_counter_ns()
            except MaxTokensReachedException:
                # Expected with the "realistic" workload: the prompt asks for more than max_tokens, so every
                # response stops at exactly max_tokens (equal length on all paths). The stream completed.
                pass
            return ttft

        token = _TAGS.set(tags or None)
        t0 = time.perf_counter_ns()
        try:
            ttft = loop.run_until_complete(run())
        except Exception as e:  # noqa: BLE001 - record any failure as a failed sample
            t1 = time.perf_counter_ns()
            return {"latency_ms": (t1 - t0) / 1e6, "status": _status_of(e) or 599,
                    "error": f"{type(e).__name__}: {e}"[:500]}
        finally:
            _TAGS.reset(token)
        t1 = time.perf_counter_ns()
        out_tokens = None
        try:
            out_tokens = agent.event_loop_metrics.accumulated_usage.get("outputTokens")
        except AttributeError:
            pass
        return {"latency_ms": (t1 - t0) / 1e6, "ttft_ms": (ttft - t0) / 1e6 if ttft else None,
                "status": 200, "output_tokens": out_tokens}

    def close(self):
        helper = getattr(self, "_helper", None)
        if helper is not None:
            helper.submit(self._close_local).result()
            helper.shutdown(wait=True)
            self._helper = None
        try:
            asyncio.get_running_loop()
            return  # main notebook thread: nothing of ours lives here
        except RuntimeError:
            self._close_local()

    def _close_local(self):
        st = getattr(self._local, "st", None)
        if st:
            loop, model = st
            try:
                loop.run_until_complete(model.client.close())
            except Exception:  # noqa: BLE001
                pass
            loop.close()
