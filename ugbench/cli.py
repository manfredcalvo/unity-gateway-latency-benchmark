"""Command line entry point: ``ugbench <command> -c config.yaml``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import build_targets, load_config


def _latest_run(cfg) -> str:
    d = cfg["run"]["output_dir"]
    runs = sorted((os.path.join(d, x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))),
                  key=os.path.getmtime)
    if not runs:
        raise SystemExit(f"no runs in {d}")
    return runs[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ugbench", description=__doc__)
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--model", help="Bedrock model ID to benchmark, e.g. us.anthropic.claude-sonnet-5-5 "
                                    "(overrides aws.bedrock_model_id; names with {model_slug} follow it)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("targets", help="list the targets the config expands to")

    s = sub.add_parser("setup", help="create/update Unity Gateway services and/or legacy serving endpoints")
    s.add_argument("--recreate", action="store_true", help="delete and recreate Unity Gateway services")
    s.add_argument("--update-auth", action="store_true", help="re-apply Bedrock auth on an existing provider service")
    s.add_argument("--provider-only", action="store_true",
                   help="create/update only the model provider service (the bundle then deploys the model services)")

    sub.add_parser("store-aws-secrets",
                   help="store the AWS key pair from env UGBENCH_AWS_ACCESS_KEY_ID / UGBENCH_AWS_SECRET_ACCESS_KEY "
                        "as UC secrets (aws_uc_secrets), once")

    p = sub.add_parser("probe", help="send one request per target and print status, latency, headers")
    p.add_argument("--layer", choices=["raw", "strands", "both"], help="client stack to benchmark")
    p.add_argument("--workload", choices=["tiny", "large_prompt", "realistic", "throughput"], help="workload preset (prompt + max_tokens; throughput = 500 streamed tokens)")
    p.add_argument("--only", nargs="*")

    h = sub.add_parser("health", help="send N requests per target; exit 1 unless every target is 100%% OK")
    h.add_argument("--layer", choices=["raw", "strands", "both"], help="client stack to benchmark")
    h.add_argument("--workload", choices=["tiny", "large_prompt", "realistic", "throughput"], help="workload preset (prompt + max_tokens; throughput = 500 streamed tokens)")
    h.add_argument("--only", nargs="*")
    h.add_argument("-n", type=int, default=30)

    r = sub.add_parser("run", help="run the benchmark")
    r.add_argument("--layer", choices=["raw", "strands", "both"], help="client stack to benchmark")
    r.add_argument("--workload", choices=["tiny", "large_prompt", "realistic", "throughput"], help="workload preset (prompt + max_tokens; throughput = 500 streamed tokens)")
    r.add_argument("--only", nargs="*", help="target names or groups (direct, unity_gateway, serving_endpoint)")
    r.add_argument("--rounds", type=int)
    r.add_argument("--warmup", type=int)
    r.add_argument("--concurrency", type=int)
    r.add_argument("--stream", action="store_true", help="stream responses (adds TTFT measurements)")
    r.add_argument("--qps", type=float, help="open-loop fixed total rate across all targets, e.g. 100")
    r.add_argument("--duration", type=float, help="seconds to run at --qps (overrides --rounds)")
    r.add_argument("--arrival", choices=["uniform", "poisson"])
    r.add_argument("--max-in-flight", type=int)
    r.add_argument("--procs", default="1", help="processes for open-loop runs (int or 'auto' = CPU count, max 8)")
    r.add_argument("--run-id")
    r.add_argument("--no-report", action="store_true")

    a = sub.add_parser("report", help="(re)build report for a run directory")
    a.add_argument("run_dir", nargs="?")
    a.add_argument("--metric", default="latency_ms", choices=["latency_ms", "ttfb_ms", "ttft_ms", "output_tps"])
    a.add_argument("--boot", type=int, default=2000)

    v = sub.add_parser("server-stats", help="pull system.ai_gateway.usage rows for a run (needs a SQL warehouse)")
    v.add_argument("run_dir", nargs="?")
    v.add_argument("--warehouse-id", required=True)
    v.add_argument("--wait", type=int, default=0, help="seconds to keep polling while the table catches up")

    t = sub.add_parser("teardown", help="delete resources created by setup")
    t.add_argument("--yes", action="store_true")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.model:
        from .config import set_model
        set_model(cfg, args.model)
    if getattr(args, "layer", None):
        cfg["layer"] = args.layer
    if getattr(args, "workload", None):
        from .config import apply_workload_profile
        apply_workload_profile(cfg, args.workload)

    if args.cmd in ("targets", "health", "run", "probe"):
        from .models import describe
        print("[model]", describe(cfg))
    if args.cmd == "targets":
        for tg in build_targets(cfg):
            print(f"{tg.name:28s} {tg.kind:22s} features={'+'.join(tg.features) or '-'}  {json.dumps(tg.params)}")
        return

    if args.cmd == "setup":
        from .setup_resources import setup_all
        setup_all(cfg, recreate=args.recreate, update_auth=args.update_auth, provider_only=args.provider_only)
        return

    if args.cmd == "store-aws-secrets":
        from . import uc_secrets
        kid, sec = os.environ.get("UGBENCH_AWS_ACCESS_KEY_ID"), os.environ.get("UGBENCH_AWS_SECRET_ACCESS_KEY")
        if not (kid and sec):
            raise SystemExit("export UGBENCH_AWS_ACCESS_KEY_ID and UGBENCH_AWS_SECRET_ACCESS_KEY first")
        uc_secrets.store(cfg, kid, sec)
        return

    if args.cmd == "teardown":
        if not args.yes:
            raise SystemExit("teardown deletes the benchmark services/endpoints. Re-run with --yes.")
        from .setup_resources import teardown_all
        teardown_all(cfg)
        return

    if args.cmd == "probe":
        from .clients import make_client
        for tg in build_targets(cfg, args.only):
            try:
                c = make_client(tg, cfg)
                res = c.call(-1, {"ugbench_run": "probe", "ugbench_target": tg.name, "ugbench_round": "-1"})
                c.close()
            except Exception as e:  # show the real error so config problems are obvious
                res = {"error": f"{type(e).__name__}: {e}"}
            print(f"{tg.name:28s} {json.dumps(res, default=str)[:600]}")
        return

    if args.cmd == "health":
        from collections import Counter

        from .clients import make_client
        healthy = True
        for tg in build_targets(cfg, args.only):
            c = make_client(tg, cfg)
            st, errs = Counter(), Counter()
            for _ in range(args.n):
                try:
                    res = c.call(-1, {"ugbench_run": "health"})
                except Exception as e:
                    res = {"status": None, "error": f"{type(e).__name__}: {e}"}
                st[res.get("status")] += 1
                if res.get("status") != 200:
                    errs[(res.get("error") or "")[:120].replace("\n", " ")] += 1
            c.close()
            ok = st.get(200, 0)
            healthy &= ok == args.n
            print(f"{tg.name:28s} ok={ok}/{args.n}  {dict(st)}")
            for e, k in errs.most_common(2):
                print(f"{'':28s}   {k}x {e}")
        print("HEALTHY" if healthy else "NOT HEALTHY: wait for config to propagate before benchmarking")
        sys.exit(0 if healthy else 1)

    if args.cmd == "run":
        if args.rounds is not None:
            cfg["run"]["rounds"] = args.rounds
        if args.warmup is not None:
            cfg["run"]["warmup_rounds"] = args.warmup
        if args.concurrency is not None:
            cfg["run"]["concurrency"] = args.concurrency
        if args.stream:
            cfg["workload"]["stream"] = True
        if args.qps is not None:
            cfg["run"]["qps"] = args.qps
        if args.duration is not None:
            cfg["run"]["duration_s"] = args.duration
        if args.arrival:
            cfg["run"]["arrival"] = args.arrival
        if args.max_in_flight:
            cfg["run"]["max_in_flight"] = args.max_in_flight
        from .runner import run_benchmark, run_sharded
        procs = min(8, os.cpu_count() or 1) if args.procs == "auto" else int(args.procs)
        if procs > 1:
            out = run_sharded(cfg, build_targets(cfg, args.only), procs, run_id=args.run_id)
        else:
            out = run_benchmark(cfg, build_targets(cfg, args.only), run_id=args.run_id)
        if not args.no_report:
            from .report import build_report
            build_report(out)
            import pandas as pd
            if pd.read_csv(os.path.join(out, "raw.csv"))["ttft_ms"].notna().any():
                build_report(out, metric="ttft_ms")  # Strands always streams; raw layer with --stream
        return

    if args.cmd == "report":
        from .report import build_report
        build_report(args.run_dir or _latest_run(cfg), n_boot=args.boot, metric=args.metric)
        return

    if args.cmd == "server-stats":
        from .server_side import fetch_server_side
        from .report import build_report
        run_dir = args.run_dir or _latest_run(cfg)
        fetch_server_side(cfg, run_dir, args.warehouse_id, wait_s=args.wait)
        build_report(run_dir)
        return


if __name__ == "__main__":
    sys.exit(main())
