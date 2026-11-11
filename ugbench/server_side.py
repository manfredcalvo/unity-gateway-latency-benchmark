"""Server-side cross-check using ``system.ai_gateway.usage``.

Client-side numbers answer "what does my application see?". They include the
extra network hop (client -> Databricks -> Bedrock) as well as the gateway's own
processing. The usage table records, per request:

* ``latency_ms``: total time the gateway spent on the request
* ``routing_information.attempts[].latency_ms``: time spent waiting on the
  upstream provider (Bedrock) for each attempt

So ``latency_ms - sum(attempts.latency_ms)`` approximates the time the gateway
itself added, measured the same way as the product team's overhead numbers.
Requests are matched to client rows through the ``Databricks-Ai-Gateway-Request-Tags``
header that the runner sends (``ugbench_run``, ``ugbench_target``, ``ugbench_round``).

Requirements: a SQL warehouse, and SELECT on system.ai_gateway.usage (account plus
metastore admin by default). Data lands roughly 30-60 minutes after the run.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .setup_resources import workspace_client

SQL = """
SELECT
  request_tags['ugbench_target'] AS target,
  CAST(request_tags['ugbench_round'] AS INT) AS round,
  status_code,
  latency_ms           AS server_latency_ms,
  time_to_first_byte_ms AS server_ttfb_ms,
  -- time spent waiting on the upstream provider, summed over attempts (computed here to keep results small:
  -- inline statement results are capped at 25 MiB)
  aggregate(transform(routing_information.attempts, a -> CAST(a.latency_ms AS BIGINT)), 0L, (acc, x) -> acc + x)
    AS upstream_ms,
  event_time
FROM system.ai_gateway.usage
WHERE request_tags['ugbench_run'] = :run_id
  AND event_time >= current_timestamp() - INTERVAL 14 DAYS
"""


def _upstream_ms(routing_json: str | None):
    if not routing_json:
        return None
    try:
        attempts = (json.loads(routing_json) or {}).get("attempts") or []
    except ValueError:
        return None
    vals = [float(a["latency_ms"]) for a in attempts if a.get("latency_ms") is not None]
    return sum(vals) if vals else None


def fetch_server_side(cfg: dict, run_dir: str, warehouse_id: str, wait_s: int = 0) -> str:
    from databricks.sdk.service.sql import StatementParameterListItem, StatementState

    run_id = os.path.basename(os.path.normpath(run_dir))
    try:
        run_finished = json.load(open(os.path.join(run_dir, "meta.json"))).get("finished_utc")
    except (OSError, ValueError):
        run_finished = None
    w = workspace_client(cfg)
    deadline = time.time() + wait_s
    while True:
        resp = w.statement_execution.execute_statement(
            statement=SQL, warehouse_id=warehouse_id, wait_timeout="50s",
            parameters=[StatementParameterListItem(name="run_id", value=run_id)])
        while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
            time.sleep(2)
            resp = w.statement_execution.get_statement(resp.statement_id)
        if resp.status.state != StatementState.SUCCEEDED:
            raise SystemExit(f"query failed: {resp.status}")
        cols = [c.name for c in resp.manifest.schema.columns]
        rows = list(resp.result.data_array or [])
        chunk = resp.result
        while chunk.next_chunk_index is not None:
            chunk = w.statement_execution.get_statement_result_chunk_n(resp.statement_id, chunk.next_chunk_index)
            rows.extend(chunk.data_array or [])
        # The usage table fills in gradually. Keep polling until it covers the end of the run (or the deadline).
        complete = False
        if rows:
            ti = cols.index("event_time")
            latest = max(pd.Timestamp(r[ti]) for r in rows if r[ti])
            finished = pd.Timestamp(run_finished) if run_finished else None
            complete = finished is None or latest >= finished - pd.Timedelta(seconds=30)
            print(f"[server] {len(rows)} rows, latest event {latest:%H:%M:%S}; run ended "
                  f"{finished:%H:%M:%S}" if finished is not None else f"[server] {len(rows)} rows")
        if complete or time.time() >= deadline:
            if not complete and rows:
                print("[server] WARNING: deadline reached before the usage table covered the whole run (partial data)")
            break
        print("[server] usage table not complete yet (lags 30-60 min); retrying in 120 s ...")
        time.sleep(120)

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        print("[server] no usage rows found for this run yet. Try again later or pass --wait.")
    else:
        for c in ("server_latency_ms", "server_ttfb_ms", "status_code", "round", "upstream_ms"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df["upstream_ms"] <= 0, "upstream_ms"] = float("nan")  # no attempts recorded
        df["gateway_internal_ms"] = df["server_latency_ms"] - df["upstream_ms"]
        raw = pd.read_csv(os.path.join(run_dir, "raw.csv"))
        df = df.merge(raw[["target", "round", "latency_ms", "phase"]], on=["target", "round"], how="left")
        df["network_and_client_ms"] = df["latency_ms"] - df["server_latency_ms"]
    out = os.path.join(run_dir, "server.csv")
    df.to_csv(out, index=False)
    print(f"[server] {len(df)} rows -> {out}")
    return out


def summarize_server(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df[(df.get("phase", "measure") == "measure") & (df["status_code"] == 200)]
    rows = []
    for tgt, g in df.groupby("target"):
        row = {"target": tgt, "n": len(g)}
        for col in ("server_latency_ms", "upstream_ms", "gateway_internal_ms", "network_and_client_ms"):
            x = g[col].dropna().to_numpy()
            if len(x):
                row[f"{col}_p50"] = float(np.percentile(x, 50))
                row[f"{col}_p99"] = float(np.percentile(x, 99))
        rows.append(row)
    return pd.DataFrame(rows)
