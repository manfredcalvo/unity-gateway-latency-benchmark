import json
import os

import httpx
import numpy as np
import pandas as pd
import pytest

from ugbench import stats
from ugbench.clients import DatabricksHttpClient
from ugbench.config import DEFAULTS, _deep_merge, build_targets
from ugbench.setup_resources import _model_service_config, _provider_config, _se_ai_gateway

ROOT = os.path.dirname(os.path.dirname(__file__))


def real_cfg(**over):
    cfg = _deep_merge(DEFAULTS, {
        "databricks": {"profile": "x"},
        "aws": {"bedrock_model_id": "us.anthropic.claude-haiku-4-5"},
        "unity_gateway": {"catalog": "main", "schema": "bench",
                          "bedrock_auth": {"service_credential": "bedrock_cred"}},
        "serving_endpoints": {"enabled": True, "auth": {"uc_service_credential_name": "bedrock_cred"},
                              "inference_table": {"catalog": "main", "schema": "bench"}},
    })
    cfg = _deep_merge(cfg, over)
    cfg["unity_gateway"]["model"] = cfg["aws"]["bedrock_model_id"]
    cfg["serving_endpoints"]["model"] = cfg["aws"]["bedrock_model_id"]
    return cfg


def test_target_expansion():
    names = [t.name for t in build_targets(real_cfg())]
    assert names[0] == "bedrock_direct"
    assert "ug_provider:direct" in names and "ug:all_features" in names and "se:usage_tracking" in names
    assert [t.name for t in build_targets(real_cfg(), only=["direct"])] == ["bedrock_direct"]


def test_resource_payloads():
    cfg = real_cfg()
    prov = _provider_config(cfg)
    assert prov["provider_type"] == "EXTERNAL_MODEL_PROVIDER_TYPE_AMAZON_BEDROCK"
    assert prov["amazon_bedrock"]["direct"]["service_credential"]["name"] == "credentials/bedrock_cred"
    base = _model_service_config(cfg, "baseline", {})
    assert "rate_limits" not in base and "inference_table" not in base
    dest = base["routing"]["destinations"][0]["external_model_config"]
    assert dest["model_provider_service"] == "model-provider-services/main.bench.ugbench_bedrock"
    allf = _model_service_config(cfg, "all_features", {"inference_table": True, "rate_limits": True})
    assert allf["inference_table"]["parent"] == "schemas/main.bench"
    assert {r["key"] for r in allf["rate_limits"]} == {"RATE_LIMIT_KEY_SERVICE", "RATE_LIMIT_KEY_USER_DEFAULT"}
    gw = _se_ai_gateway(cfg, "baseline", {})
    assert gw == {"usage_tracking_config": {"enabled": False}}
    gw = _se_ai_gateway(cfg, "all_features", {"usage_tracking": True, "inference_tables": True, "rate_limits": True})
    assert gw["usage_tracking_config"]["enabled"] and gw["inference_table_config"]["enabled"]


def _df(a, b):
    rows = [{"target": "bedrock_direct", "group": "direct", "features": "", "round": i, "ok": 1, "latency_ms": v}
            for i, v in enumerate(a)]
    rows += [{"target": "ug:baseline", "group": "unity_gateway", "features": "", "round": i, "ok": 1,
              "latency_ms": v} for i, v in enumerate(b)]
    return pd.DataFrame(rows)


def test_compare_recovers_known_shift():
    rng = np.random.default_rng(1)
    base = rng.lognormal(np.log(400), 0.2, 3000)
    df = _df(base, base + 5.0)
    r = stats.compare(df, "ug:baseline", "bedrock_direct", n_boot=300)
    assert r["paired_median"] == pytest.approx(5.0)
    assert r["d_p50"] == pytest.approx(5.0, abs=0.01)
    assert r["d_p50_lo"] <= 5.0 <= r["d_p50_hi"]


def test_summary_and_errors():
    df = _df([100, 200, 300, 400], [110, 210, 310, 410])
    df.loc[0, "ok"] = 0
    s = stats.summarize(df).set_index("target")
    assert s.loc["bedrock_direct", "n_ok"] == 3
    assert s.loc["bedrock_direct", "error_rate"] == pytest.approx(0.25)
    assert stats.tail_reliability(1000, 99) == "low"
    assert stats.tail_reliability(5000, 99) == "ok"


class _FakeCfg:
    host = "https://example.cloud.databricks.com"

    def authenticate(self):
        return {"Authorization": "Bearer fake"}


def _client(cfg, target_name, handler):
    DatabricksHttpClient._shared = {"x|None": _FakeCfg()}
    tgt = next(t for t in build_targets(cfg) if t.name == target_name)
    c = DatabricksHttpClient(tgt, cfg)
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_http_client_chat_completions_and_tags():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        seen["headers"] = dict(req.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}],
                                         "usage": {"completion_tokens": 1}}, headers={"x-request-id": "abc"})

    c = _client(real_cfg(), "ug:baseline", handler)
    out = c.call(3, {"ugbench_run": "r1", "ugbench_target": "ug:baseline", "ugbench_round": "3"})
    assert out["status"] == 200 and out["output_tokens"] == 1 and out["request_id"] == "abc"
    assert seen["url"].endswith("/ai-gateway/mlflow/v1/chat/completions")
    assert seen["body"]["model"] == "main.bench.ugbench_baseline"
    assert json.loads(seen["headers"]["databricks-ai-gateway-request-tags"])["ugbench_round"] == "3"
    assert out["latency_ms"] > 0


def test_http_client_provider_service_streaming_ttft():
    sse = ("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
           "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"OK\"}}\n\n"
           "data: {\"type\":\"message_stop\"}\n\n")

    def handler(req):
        assert req.headers["databricks-model-provider-service"] == "main.bench.ugbench_bedrock"
        assert req.url.path == "/ai-gateway/anthropic/v1/messages"
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})

    cfg = real_cfg(workload={"stream": True})
    c = _client(cfg, "ug_provider:direct", handler)
    out = c.call(0, {})
    assert out["status"] == 200 and out["ttft_ms"] is not None and out["ttft_ms"] <= out["latency_ms"]


def test_http_client_error_is_recorded():
    c = _client(real_cfg(), "se:baseline", lambda req: httpx.Response(429, text="rate limited"))
    out = c.call(0, {})
    assert out["status"] == 429 and "rate limited" in out["error"]
    assert "/serving-endpoints/ugbench-baseline/invocations" in c.url


def test_end_to_end_mock(tmp_path):
    from ugbench.config import load_config
    from ugbench.report import build_report
    from ugbench.runner import run_benchmark

    cfg = load_config(os.path.join(ROOT, "config.mock.yaml"))
    cfg["run"].update(rounds=200, warmup_rounds=2, output_dir=str(tmp_path))
    out = run_benchmark(cfg, build_targets(cfg), run_id="t")
    res = build_report(out, n_boot=200)
    raw = pd.read_csv(os.path.join(out, "raw.csv"))
    assert len(raw) == 202 * 5
    assert os.path.exists(res["html"]) and len(res["charts"]) == 4
    c = res["comparisons"]
    pm = c[(c["target"] == "mock:baseline") & (c["kind"] == "vs_direct")]["paired_median"].iloc[0]
    assert 0 < pm < 15


def test_provider_config_uc_secret_reference(monkeypatch):
    from ugbench import uc_secrets
    cfg = real_cfg(unity_gateway={"bedrock_auth": {"service_credential": None, "use_uc_secrets": True}})
    cfg["aws_uc_secrets"] = dict(DEFAULTS["aws_uc_secrets"])
    monkeypatch.setattr(uc_secrets, "read", lambda c, which, w=None: "AKIAEXAMPLE" if which == "access_key_id" else "x")
    ak = _provider_config(cfg)["amazon_bedrock"]["direct"]["aws_access_key"]
    assert ak["access_key_id"] == "AKIAEXAMPLE"
    assert ak["secret_access_key"] == {"secret_reference": {"name": "secrets/main.bench.ugbench_aws_secret_access_key"}}
    assert "plaintext" not in str(ak)


def test_open_loop_holds_rate(tmp_path):
    from ugbench.config import load_config
    from ugbench.runner import run_benchmark

    cfg = load_config(os.path.join(ROOT, "config.mock.yaml"))
    cfg["run"].update(output_dir=str(tmp_path), qps=200, duration_s=2, warmup_s=0.5)
    out = run_benchmark(cfg, build_targets(cfg), run_id="ol")
    raw = pd.read_csv(os.path.join(out, "raw.csv"))
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["load"]["mode"] == "open_loop"
    span = (raw["sched_offset_ms"].max() - raw["sched_offset_ms"].min()) / 1000
    assert 150 < len(raw) / span < 250
    assert raw["send_lag_ms"].notna().all() and raw["latency_from_sched_ms"].notna().all()


# ---------------------------------------------------------------- Strands layer
def _anthropic_sse(text_chunks=("Hello", " world"), out_tokens=7):
    ev = []
    def add(t, d):
        ev.append(f"event: {t}\ndata: {json.dumps({'type': t, **d})}\n\n")
    add("message_start", {"message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                                      "content": [], "stop_reason": None, "stop_sequence": None,
                                      "usage": {"input_tokens": 12, "output_tokens": 1}}})
    add("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
    for c in text_chunks:
        add("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": c}})
    add("content_block_stop", {"index": 0})
    add("message_delta", {"delta": {"stop_reason": "max_tokens", "stop_sequence": None},
                          "usage": {"output_tokens": out_tokens}})
    add("message_stop", {})
    return "".join(ev).encode()


def _strands_client(cfg, name, handler):
    from ugbench.strands_client import StrandsClient
    StrandsClient._dbx_shared = {"x|None": _FakeCfg()}
    StrandsClient._test_transport = httpx.MockTransport(handler)
    tgt = next(t for t in build_targets(cfg) if t.name == name)
    return StrandsClient(tgt, cfg)


def test_strands_targets_and_comparisons():
    cfg = real_cfg(layer="strands")
    names = [t.name for t in build_targets(cfg)]
    assert names == ["strands:direct", "strands:ug_provider", "strands:ug:baseline", "strands:ug:inference_table",
                     "strands:ug:rate_limits", "strands:ug:all_features"]
    comps = stats.default_comparisons(names)
    assert ("strands:ug:baseline", "strands:direct", "vs_direct") in comps
    assert ("strands:ug:rate_limits", "strands:ug:baseline", "vs_baseline") in comps
    assert not any(ref == "bedrock_direct" for _, ref, _ in comps)


@pytest.mark.parametrize("name,expect_model,provider_hdr", [
    ("strands:ug:baseline", "main.bench.ugbench_baseline", None),
    ("strands:ug_provider", "us.anthropic.claude-haiku-4-5", "main.bench.ugbench_bedrock"),
])
def test_strands_gateway_request_and_ttft(name, expect_model, provider_hdr):
    pytest.importorskip("strands")
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["headers"] = dict(req.headers)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=_anthropic_sse(), headers={"content-type": "text/event-stream"})

    cfg = real_cfg(layer="strands", workload={"max_tokens": 50})
    c = _strands_client(cfg, name, handler)
    out = c.call(4, {"ugbench_run": "r", "ugbench_target": name, "ugbench_round": "4"})
    c.close()
    assert out["status"] == 200, out
    assert out["ttft_ms"] is not None and 0 < out["ttft_ms"] <= out["latency_ms"]
    assert out["output_tokens"] == 7
    assert seen["url"] == "https://example.cloud.databricks.com/ai-gateway/anthropic/v1/messages"
    assert seen["body"]["model"] == expect_model and seen["body"]["stream"] is True
    assert seen["body"]["max_tokens"] == 50
    assert seen["headers"]["authorization"] == "Bearer fake"
    assert json.loads(seen["headers"]["databricks-ai-gateway-request-tags"])["ugbench_round"] == "4"
    assert seen["headers"].get("databricks-model-provider-service") == provider_hdr


def test_strands_error_status_recorded():
    pytest.importorskip("strands")
    cfg = real_cfg(layer="strands")
    c = _strands_client(cfg, "strands:ug:baseline",
                        lambda req: httpx.Response(429, json={"type": "error", "error": {"type": "rate_limit_error",
                                                                                        "message": "slow down"}}))
    out = c.call(0, {})
    c.close()
    assert out["status"] == 429 and out.get("error")


def test_sharded_open_loop_merges(tmp_path):
    from ugbench.config import load_config
    from ugbench.runner import run_sharded

    cfg = load_config(os.path.join(ROOT, "config.mock.yaml"))
    cfg["run"].update(output_dir=str(tmp_path), qps=200, duration_s=2, warmup_s=0.5)
    out = run_sharded(cfg, build_targets(cfg), procs=2, run_id="sharded")
    raw = pd.read_csv(os.path.join(out, "raw.csv"))
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["load"]["procs"] == 2 and meta["load"]["target_qps"] == 200
    assert set(raw["run_id"]) == {"sharded"}
    # rounds from different shards never collide, and every round still contains every target
    per_round = raw.groupby("round")["target"].nunique()
    assert (per_round == raw["target"].nunique()).all()
    assert raw["round"].max() >= 1_000_000
    assert "_static_credentials" not in json.dumps(meta)


def test_strands_call_inside_running_event_loop():
    """Databricks/Jupyter notebooks already run an asyncio loop on the main thread."""
    pytest.importorskip("strands")
    import asyncio
    cfg = real_cfg(layer="strands", workload={"max_tokens": 50})
    c = _strands_client(cfg, "strands:ug:baseline",
                        lambda req: httpx.Response(200, content=_anthropic_sse(),
                                                   headers={"content-type": "text/event-stream"}))

    async def notebook_cell():
        return [c.call(i, {}) for i in range(2)]  # sync calls made while a loop is running

    outs = asyncio.run(notebook_cell())
    asyncio.run(asyncio.sleep(0))
    c.close()
    assert all(o["status"] == 200 and o["ttft_ms"] for o in outs), outs
