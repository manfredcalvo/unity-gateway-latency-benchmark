"""Benchmark runner.

Design (why the numbers can be trusted):

* **Interleaved, randomized blocks.** Each round sends one request to every target,
  in a freshly shuffled order. Provider-side drift (Bedrock load, time of day)
  hits all targets equally instead of biasing whichever one ran last.
* **Warmup is discarded.** The first ``warmup_rounds`` open connections, warm
  DNS/TLS, and warm the gateway's caches. They are recorded with
  ``phase=warmup`` and excluded from the statistics.
* **Two load modes.**
  - *Closed loop* (default): ``run.concurrency`` workers each send, wait, send again.
    Throughput depends on latency (~2 QPS per worker for a ~450 ms model).
  - *Open loop* (``run.qps`` > 0): a dispatcher releases requests at a fixed total
    rate (``arrival: uniform`` or ``poisson``) no matter how many are still in flight,
    up to ``max_in_flight``. Each slot still picks targets in shuffled round-robin
    order. ``send_lag_ms`` records how late each request left compared with its
    schedule, so you can verify the client held the rate. ``latency_from_sched_ms``
    adds that lag to the latency, which avoids coordinated omission.
* **Everything is persisted raw** (one CSV row per request) next to the resolved
  config and environment metadata, so any number in the report can be recomputed.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import queue
import random
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

from . import __version__
from .clients import make_client
from .config import Target

FIELDS = ["run_id", "phase", "round", "seq", "worker", "target", "group", "features",
          "start_utc", "latency_ms", "ttfb_ms", "ttft_ms", "provider_latency_ms",
          "status", "ok", "output_tokens", "request_id", "error", "sched_offset_ms", "send_lag_ms",
          "latency_from_sched_ms"]


def _env_metadata(cfg: dict) -> dict:
    meta = {
        "ugbench_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "in_databricks": bool(os.environ.get("DATABRICKS_RUNTIME_VERSION")),
        "databricks_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
        "aws_region_env": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
    }
    # Best effort: EC2 / Databricks classic compute availability zone (IMDSv2). Ignored elsewhere.
    try:
        import httpx

        tok = httpx.put("http://169.254.169.254/latest/api/token",
                        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}, timeout=0.3).text
        meta["client_az"] = httpx.get("http://169.254.169.254/latest/meta-data/placement/availability-zone",
                                      headers={"X-aws-ec2-metadata-token": tok}, timeout=0.3).text
    except Exception:
        meta["client_az"] = None
    return meta


def _public_cfg(cfg: dict) -> dict:
    """Config as written to meta.json: in-memory credentials handed to shard processes are removed."""
    import copy

    c = copy.deepcopy(cfg)
    c.get("aws", {}).pop("_static_credentials", None)
    return c


def run_benchmark(cfg: dict, targets: list[Target], run_id: str | None = None,
                  progress_every: int = 25, tag_run_id: str | None = None, round_offset: int = 0) -> str:
    """Run one benchmark process. ``tag_run_id`` / ``round_offset`` are set by ``run_sharded`` so every
    shard tags requests with the parent run id and uses non-overlapping round numbers."""
    rcfg = cfg["run"]
    run_id = run_id or f"{cfg['run_name']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    final_dir = os.path.join(rcfg["output_dir"], run_id)
    os.makedirs(final_dir, exist_ok=True)
    # Workspace files and volumes only show a file once it's closed, so a cancelled job would lose its data.
    # For those, write to local disk and copy to the final location periodically and at the end.
    remote = final_dir.startswith(("/Workspace", "/Volumes", "/dbfs"))
    out_dir = os.path.join("/tmp/ugbench", run_id) if remote else final_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[ugbench] run_id={run_id}")
    print(f"[ugbench] targets: {', '.join(t.name for t in targets)}")
    if rcfg.get("qps"):
        # Size connection pools for the in-flight load of the open-loop mode.
        cfg = {**cfg, "run": {**rcfg, "concurrency": int(rcfg.get("max_in_flight", 256))}}
        if rcfg.get("duration_s"):
            per_round = len(targets) / float(rcfg["qps"])
            rcfg = {**rcfg, "rounds": max(1, int(rcfg["duration_s"] / per_round)),
                    "warmup_rounds": int(rcfg.get("warmup_s", 10) / per_round)}
        print(f"[ugbench] open loop: {rcfg['qps']} QPS total ({rcfg.get('arrival', 'uniform')}), "
              f"{rcfg['rounds']} rounds = {rcfg['rounds'] * len(targets) / float(rcfg['qps']):.0f}s measured")
    clients = {t.name: make_client(t, cfg) for t in targets}

    rng = random.Random(rcfg["seed"])
    warm, rounds = rcfg["warmup_rounds"], rcfg["rounds"]
    jobs: list[tuple[str, int, Target]] = []
    for r in range(warm + rounds):
        order = list(targets)
        if rcfg["shuffle"]:
            rng.shuffle(order)
        phase = "warmup" if r < warm else "measure"
        jobs.extend((phase, round_offset + r, t) for t in order)

    meta = {"run_id": run_id, "started_utc": datetime.now(timezone.utc).isoformat(),
            "targets": [t.__dict__ for t in targets], "config": _public_cfg(cfg), "env": _env_metadata(cfg)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    lock = threading.Lock()
    csv_path = os.path.join(out_dir, "raw.csv")
    stop_sync = threading.Event()

    def sync_loop():
        while not stop_sync.wait(int(rcfg.get("sync_every_s", 60))):
            _sync(out_dir, final_dir, lock, fh)

    if remote:
        threading.Thread(target=sync_loop, daemon=True).start()
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader()
    counts = {"ok": 0, "err": 0}
    t_start = time.time()

    def execute(seq: int, phase: str, rnd: int, tgt: Target, wid, sched: float | None = None, t0: float = 0.0):
        tags = {"ugbench_run": tag_run_id or run_id, "ugbench_target": tgt.name, "ugbench_round": str(rnd)}
        row = {"run_id": run_id, "phase": phase, "round": rnd, "seq": seq, "worker": wid,
               "target": tgt.name, "group": tgt.group, "features": "+".join(tgt.features),
               "start_utc": datetime.now(timezone.utc).isoformat()}
        if sched is not None:
            row["sched_offset_ms"] = (sched - t0) * 1000
            row["send_lag_ms"] = (time.perf_counter() - sched) * 1000
        try:
            res = clients[tgt.name].call(rnd, tags)
        except Exception as e:  # network errors, throttling surfaced as exceptions, etc.
            res = {"status": getattr(getattr(e, "response", None), "status_code", None)
                   or _boto_status(e), "error": f"{type(e).__name__}: {e}"[:500]}
            if os.environ.get("UGBENCH_DEBUG"):
                traceback.print_exc()
        row.update(res)
        if sched is not None and res.get("latency_ms") is not None:
            row["latency_from_sched_ms"] = row["send_lag_ms"] + res["latency_ms"]
        row["ok"] = int(res.get("status") == 200 and not res.get("error"))
        with lock:
            writer.writerow({k: row.get(k) for k in FIELDS})
            counts["ok" if row["ok"] else "err"] += 1
        return row

    qps = float(rcfg.get("qps") or 0)
    if qps > 0:
        meta["load"] = _run_open_loop(jobs, execute, qps, rcfg, fh, counts, lock)
    else:
        meta["load"] = {"mode": "closed_loop", "concurrency": rcfg["concurrency"]}
        q: queue.Queue = queue.Queue()
        for i, j in enumerate(jobs):
            q.put((i, *j))
        pause = rcfg["pause_between_requests_ms"] / 1000
        done_rounds = set()

        def worker(wid: int):
            while True:
                try:
                    seq, phase, rnd, tgt = q.get_nowait()
                except queue.Empty:
                    return
                execute(seq, phase, rnd, tgt, wid)
                with lock:
                    if rnd not in done_rounds and seq % len(targets) == len(targets) - 1:
                        done_rounds.add(rnd)
                        if (rnd + 1) % progress_every == 0:
                            fh.flush()
                            print(f"[ugbench] round {rnd + 1}/{warm + rounds}  ok={counts['ok']} "
                                  f"err={counts['err']}  elapsed={time.time() - t_start:.0f}s", flush=True)
                if pause:
                    time.sleep(pause)

        threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(rcfg["concurrency"])]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    stop_sync.set()
    fh.close()
    for c in clients.values():
        c.close()

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["counts"] = counts
    meta["config"] = _public_cfg(cfg)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[ugbench] done: ok={counts['ok']} err={counts['err']} -> {csv_path}")
    if remote:
        _sync(out_dir, final_dir, None, None)
        print(f"[ugbench] copied results to {final_dir}")
    return final_dir


def _sync(src: str, dst: str, lock, fh) -> None:
    """Copy raw.csv + meta.json to the final (workspace/volume) location. Best effort."""
    import shutil

    try:
        if lock is not None:
            with lock:
                fh.flush()
                shutil.copyfile(os.path.join(src, "raw.csv"), os.path.join(dst, "raw.csv"))
        else:
            shutil.copyfile(os.path.join(src, "raw.csv"), os.path.join(dst, "raw.csv"))
        shutil.copyfile(os.path.join(src, "meta.json"), os.path.join(dst, "meta.json"))
    except Exception as e:  # never let a sync failure kill the benchmark
        print(f"[ugbench] sync warning: {e}")


def _run_open_loop(jobs, execute, qps, rcfg, fh, counts, lock) -> dict:
    """Release jobs at a fixed total rate. Arrival times don't depend on completions (open loop)."""
    from concurrent.futures import ThreadPoolExecutor

    max_in_flight = int(rcfg.get("max_in_flight", 256))
    arrival = rcfg.get("arrival", "uniform")
    rng = random.Random(rcfg["seed"] + 1)
    slots = threading.BoundedSemaphore(max_in_flight)
    saturated = 0
    t0 = time.perf_counter() + 0.5
    offset = 0.0
    last_report = time.time()
    with ThreadPoolExecutor(max_workers=max_in_flight, thread_name_prefix="ugb") as pool:
        for seq, (phase, rnd, tgt) in enumerate(jobs):
            sched = t0 + offset
            offset += rng.expovariate(qps) if arrival == "poisson" else 1.0 / qps
            delay = sched - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            if not slots.acquire(blocking=False):  # client at max_in_flight: this request will be late
                saturated += 1
                slots.acquire()

            def run(seq=seq, phase=phase, rnd=rnd, tgt=tgt, sched=sched):
                try:
                    execute(seq, phase, rnd, tgt, threading.current_thread().name, sched=sched, t0=t0)
                finally:
                    slots.release()

            pool.submit(run)
            if time.time() - last_report > 15:
                last_report = time.time()
                with lock:
                    fh.flush()
                    print(f"[ugbench] t={time.perf_counter() - t0:.0f}s sent={seq + 1}/{len(jobs)} "
                          f"ok={counts['ok']} err={counts['err']}", flush=True)
        achieved = len(jobs) / max(offset, 1e-9)
    return {"mode": "open_loop", "target_qps": qps, "arrival": arrival, "max_in_flight": max_in_flight,
            "scheduled_qps": achieved, "saturated_dispatches": saturated}


def _boto_status(e: Exception):
    resp = getattr(e, "response", None)
    if isinstance(resp, dict):
        return resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return None


# ======================================================================================
# Multi-process sharding (open-loop mode)
# ======================================================================================
def _shard_main(cfg: dict, target_names: list[str], run_id: str, tag_run_id: str, round_offset: int) -> None:
    from .config import build_targets

    targets = [t for t in build_targets(cfg) if t.name in set(target_names)]
    run_benchmark(cfg, targets, run_id=run_id, tag_run_id=tag_run_id, round_offset=round_offset)


def _merge_shards(shard_dirs: list[str], final_dir: str, run_id: str, total_qps: float, procs: int,
                  final: bool) -> int:
    import pandas as pd

    frames = []
    for d in shard_dirs:
        p = os.path.join(d, "raw.csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            try:
                frames.append(pd.read_csv(p))
            except Exception:  # a shard may be mid-write; skip it this time
                pass
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    df["run_id"] = run_id
    os.makedirs(final_dir, exist_ok=True)
    tmp = os.path.join("/tmp/ugbench", run_id + ".merged.csv")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    df.to_csv(tmp, index=False)
    import shutil

    shutil.copyfile(tmp, os.path.join(final_dir, "raw.csv"))
    metas = []
    for d in shard_dirs:
        try:
            metas.append(json.load(open(os.path.join(d, "meta.json"))))
        except (OSError, ValueError):
            pass
    if metas:
        m = metas[0]
        m["run_id"] = run_id
        m["started_utc"] = min(x.get("started_utc", "") for x in metas)
        if final:
            m["finished_utc"] = max(x.get("finished_utc", "") for x in metas)
            m["counts"] = {k: sum(x.get("counts", {}).get(k, 0) for x in metas) for k in ("ok", "err")}
        load = dict(m.get("load") or {})
        load.update(target_qps=total_qps, procs=procs,
                    saturated_dispatches=sum((x.get("load") or {}).get("saturated_dispatches", 0) for x in metas))
        m["load"] = load
        m["config"]["run"]["qps"] = total_qps
        with open(os.path.join("/tmp/ugbench", run_id + ".meta.json"), "w") as f:
            json.dump(m, f, indent=2, default=str)
        shutil.copyfile(os.path.join("/tmp/ugbench", run_id + ".meta.json"), os.path.join(final_dir, "meta.json"))
    return len(df)


def run_sharded(cfg: dict, targets: list[Target], procs: int, run_id: str | None = None) -> str:
    """Split an open-loop run across ``procs`` processes, each sending qps/procs. One Python process can't
    drive hundreds of concurrent streaming requests (GIL + thread switching inflate client latency); see
    the report's send-lag line. Results are merged into one run directory as if from a single process."""
    import copy
    import multiprocessing as mp
    import sys as _sys

    rcfg = cfg["run"]
    if not rcfg.get("qps"):
        raise SystemExit("--procs requires open-loop mode (--qps)")
    run_id = run_id or f"{cfg['run_name']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    final_dir = os.path.join(rcfg["output_dir"], run_id)
    shard_root = os.path.join("/tmp/ugbench", run_id, "shards")

    # Resolve credentials ONCE in the parent; hand them to children in memory only (never written to disk).
    base = copy.deepcopy(cfg)
    if any(t.kind == "bedrock" or t.params.get("backend") == "bedrock" for t in targets):
        from .clients import aws_session

        fc = aws_session(cfg).get_credentials().get_frozen_credentials()
        base["aws"]["_static_credentials"] = {"access_key": fc.access_key, "secret_key": fc.secret_key,
                                              "token": fc.token}
    if any(t.group in ("unity_gateway", "serving_endpoint") for t in targets):
        # Pre-build and warm the Databricks auth config so forked children inherit a cached token.
        from databricks.sdk.core import Config

        from .clients import DatabricksHttpClient
        from .strands_client import StrandsClient

        key = f"{cfg['databricks'].get('profile')}|{cfg['databricks'].get('host')}"
        kwargs = {k: v for k, v in cfg["databricks"].items() if v and k in ("profile", "host")}
        conf = DatabricksHttpClient._shared.get(key) or Config(**kwargs)
        conf.authenticate()
        DatabricksHttpClient._shared[key] = conf
        StrandsClient._dbx_shared[key] = conf

    ctx = mp.get_context("fork" if _sys.platform.startswith("linux") else "spawn")
    shard_dirs, children = [], []
    print(f"[ugbench] sharded: {procs} processes x {rcfg['qps'] / procs:.1f} QPS = {rcfg['qps']} QPS total")
    for i in range(procs):
        c = copy.deepcopy(base)
        c["run"].update(qps=rcfg["qps"] / procs, seed=rcfg["seed"] + i, output_dir=shard_root,
                        max_in_flight=max(32, int(rcfg.get("max_in_flight", 512)) // procs))
        shard_dirs.append(os.path.join(shard_root, f"shard{i}"))
        p = ctx.Process(target=_shard_main, args=(c, [t.name for t in targets], f"shard{i}", run_id, i * 1_000_000),
                        daemon=False)
        p.start()
        children.append(p)
    last = time.time()
    while any(p.is_alive() for p in children):
        time.sleep(2)
        if time.time() - last > 60:
            last = time.time()
            n = _merge_shards(shard_dirs, final_dir, run_id, rcfg["qps"], procs, final=False)
            print(f"[ugbench] merged progress: {n} rows", flush=True)
    for p in children:
        p.join()
    bad = [i for i, p in enumerate(children) if p.exitcode != 0]
    n = _merge_shards(shard_dirs, final_dir, run_id, rcfg["qps"], procs, final=True)
    print(f"[ugbench] sharded run done: {n} rows -> {final_dir}" + (f"  (FAILED shards: {bad})" if bad else ""))
    if bad:
        raise SystemExit(f"shards failed: {bad}")
    return final_dir
