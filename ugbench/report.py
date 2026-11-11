"""Customer-shareable report: Markdown + self-contained HTML + CSVs + PNG charts."""

from __future__ import annotations

import base64
import html
import json
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats  # noqa: E402
from .server_side import summarize_server  # noqa: E402

# Validated reference palette (dataviz skill, light mode). Slot order is fixed.
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # blue, orange, aqua

PRODUCT_REFERENCE = pd.DataFrame([
    {"configuration": "No gateway features", "p50_ms": 4, "p99_ms": 9},
    {"configuration": "With usage tracking", "p50_ms": 7, "p99_ms": 15},
    {"configuration": "With inference tables", "p50_ms": 7, "p99_ms": 15},
    {"configuration": "With rate limits", "p50_ms": 6, "p99_ms": 9},
])

LABELS = {
    "bedrock_direct": "Bedrock direct",
    "ug_provider:direct": "Unity Gateway: provider service",
    "ug:baseline": "Unity Gateway: model service (no features)",
    "ug:inference_table": "Unity Gateway: + inference table",
    "ug:rate_limits": "Unity Gateway: + rate limits",
    "ug:all_features": "Unity Gateway: + inference table + rate limits",
    "se:baseline": "Serving endpoint: no features",
    "se:usage_tracking": "Serving endpoint: + usage tracking",
    "se:inference_tables": "Serving endpoint: + inference tables",
    "se:rate_limits": "Serving endpoint: + rate limits",
    "se:all_features": "Serving endpoint: all features",
    "strands:direct": "Strands + Anthropic SDK: Bedrock direct",
    "strands:ug_provider": "Strands: Unity Gateway provider service",
    "strands:ug:baseline": "Strands: Unity Gateway model service (no features)",
    "strands:ug:inference_table": "Strands: UG + inference table",
    "strands:ug:rate_limits": "Strands: UG + rate limits",
    "strands:ug:all_features": "Strands: UG + inference table + rate limits",
}


def label(t: str) -> str:
    return LABELS.get(t, t)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(h):
    fig, ax = plt.subplots(figsize=(9, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


# ------------------------------------------------------------------------------ charts
def chart_overhead(cmp: pd.DataFrame, path: str, kind: str, title: str) -> str | None:
    """Dot + CI whisker per target. P99 deltas stay in the tables: client-side P99 deltas
    carry model-jitter CIs of tens of ms and would visually drown the overhead signal."""
    c = cmp[(cmp["kind"] == kind) & cmp["paired_median"].notna()].iloc[::-1]
    if c.empty:
        return None
    fig, ax = _fig(0.6 * len(c) + 1.7)
    y = np.arange(len(c))
    series = (("paired_median", S1, "Paired median delta (most precise)", 0.13),
              ("d_p50", S2, "P50 delta", -0.13))
    for col_name, col, name, off in series:
        x = c[col_name].to_numpy()
        lo, hi = c[f"{col_name}_lo"].to_numpy(), c[f"{col_name}_hi"].to_numpy()
        ax.hlines(y + off, lo, hi, color=col, linewidth=2, alpha=0.45)
        ax.scatter(x, y + off, s=64, color=col, edgecolor=SURFACE, linewidth=2, zorder=3, label=name)
        for xi, hii, yi in zip(x, hi, y + off):
            ax.annotate(f"{xi:+.1f} ms", (hii, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, [label(t) for t in c["target"]], color=INK)
    ax.set_xlabel("Added latency per request (ms), whiskers = 95% bootstrap CI", color=INK2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    xmax = np.nanmax(c[["paired_median_hi", "d_p50_hi"]].to_numpy())
    ax.set_xlim(right=xmax + max(6, 0.25 * abs(xmax)))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_distribution(df: pd.DataFrame, path: str, metric: str = "latency_ms") -> str | None:
    ok = df[df["ok"] == 1]
    order = list(dict.fromkeys(df["target"]))[::-1]
    data = [ok[ok["target"] == t][metric].dropna().to_numpy() for t in order]
    if not any(len(d) for d in data):
        return None
    fig, ax = _fig(0.5 * len(order) + 1.6)
    bp = ax.boxplot(data, orientation="horizontal", whis=(1, 99), showfliers=False, widths=0.5, patch_artist=True,
                    medianprops={"color": INK, "linewidth": 2})
    for patch, t in zip(bp["boxes"], order):
        patch.set_facecolor(S2 if t == "bedrock_direct" else S1)
        patch.set_alpha(0.85)
        patch.set_edgecolor(SURFACE)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(MUTED)
    ax.set_yticks(range(1, len(order) + 1), [label(t) for t in order], color=INK)
    ax.set_xlabel(f"End-to-end {metric.replace('_ms', '')} (ms): box = P25 to P75, line = median, whiskers = P1 to P99",
                  color=INK2, fontsize=9)
    ax.set_title("Client-observed latency distribution", loc="left", color=INK, fontsize=11, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S2, label="Bedrock direct"), Patch(color=S1, label="Through Databricks")],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_timeline(df: pd.DataFrame, path: str, metric: str = "latency_ms") -> str | None:
    ok = df[df["ok"] == 1]
    names = list(dict.fromkeys(df["target"]))
    pick = [t for t in ("bedrock_direct", "ug:baseline", "ug:all_features") if t in names]
    if len(pick) < 2:
        pick = names[:3]
    fig, ax = _fig(3.4)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for t, col in zip(pick, (S2, S1, S3)):
        s = ok[ok["target"] == t].sort_values("round").set_index("round")[metric]
        ax.plot(s.index, s.rolling(100, min_periods=20).median(), color=col, linewidth=2, label=label(t))
    ax.set_xlabel("Round", color=INK2, fontsize=9)
    ax.set_ylabel("Rolling median (100 rounds), ms", color=INK2, fontsize=9)
    ax.set_title("Latency over the run (rolling median)",
                 loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------ tables
def _fmt_ci(v, lo, hi):
    if pd.isna(v):
        return "n/a"
    return f"{v:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def table_latency(summary: pd.DataFrame, df: pd.DataFrame | None = None) -> pd.DataFrame:
    t = pd.DataFrame({
        "Target": summary["target"].map(label),
        "Features": summary["features"].replace("", "none"),
        "Requests (ok/total)": summary.apply(lambda r: f"{int(r.n_ok)}/{int(r.n)}", axis=1),
        "Errors": (summary["error_rate"] * 100).map(lambda v: f"{v:.1f}%"),
    })
    for p in (50, 90, 99):
        t[f"P{p} (ms)"] = summary.get(f"p{p}", pd.Series(dtype=float)).map(lambda v: f"{v:.0f}" if pd.notna(v) else "n/a")
    if df is not None and "output_tokens" in df and df["output_tokens"].notna().any():
        tok = df[df["ok"] == 1].groupby("target")["output_tokens"].median()
        t["Output tokens P50"] = summary["target"].map(tok).map(lambda v: f"{v:.0f}" if pd.notna(v) else "n/a")
    return t


def table_cmp(cmp: pd.DataFrame, kind: str) -> pd.DataFrame:
    c = cmp[cmp["kind"] == kind]
    if c.empty:
        return c
    return pd.DataFrame({
        "Target": c["target"].map(label),
        "vs": c["reference"].map(label),
        "Paired median delta, ms [95% CI]": c.apply(lambda r: _fmt_ci(r.get("paired_median"), r.get("paired_median_lo"), r.get("paired_median_hi")), axis=1),
        "P50 delta, ms [95% CI]": c.apply(lambda r: _fmt_ci(r.get("d_p50"), r.get("d_p50_lo"), r.get("d_p50_hi")), axis=1),
        "P90 delta, ms [95% CI]": c.apply(lambda r: _fmt_ci(r.get("d_p90"), r.get("d_p90_lo"), r.get("d_p90_hi")), axis=1),
        "P99 delta, ms [95% CI]": c.apply(lambda r: _fmt_ci(r.get("d_p99"), r.get("d_p99_lo"), r.get("d_p99_hi")), axis=1),
        "P50 delta as % of reference P50": c["rel_p50_pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "n/a"),
    })


def table_server(srv: pd.DataFrame) -> pd.DataFrame:
    if srv is None or srv.empty:
        return pd.DataFrame()
    f = lambda v: f"{v:.0f}" if pd.notna(v) else "n/a"  # noqa: E731
    return pd.DataFrame({
        "Target": srv["target"].map(label),
        "n": srv["n"],
        "Gateway-internal P50 (ms)": srv.get("gateway_internal_ms_p50", pd.Series(dtype=float)).map(f),
        "Gateway-internal P99 (ms)": srv.get("gateway_internal_ms_p99", pd.Series(dtype=float)).map(f),
        "Upstream Bedrock P50 (ms)": srv.get("upstream_ms_p50", pd.Series(dtype=float)).map(f),
        "Network + client P50 (ms)": srv.get("network_and_client_ms_p50", pd.Series(dtype=float)).map(f),
    })


# ------------------------------------------------------------------------------ render
def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_no data_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _html_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p><em>no data</em></p>"
    return df.to_html(index=False, border=0, classes="t", escape=True)


def _load_line(meta: dict, df: pd.DataFrame) -> str:
    load = meta.get("load") or {"mode": "closed_loop", "concurrency": meta["config"]["run"]["concurrency"]}
    rounds = int(df["round"].nunique())
    if load["mode"] != "open_loop":
        return (f"{rounds} measured rounds after {meta['config']['run']['warmup_rounds']} warmup rounds; closed loop, "
                f"concurrency {load['concurrency']}; target order shuffled each round.")
    d = df.dropna(subset=["sched_offset_ms"])
    span_s = (d["sched_offset_ms"] + d["send_lag_ms"]).max() / 1000 - d["sched_offset_ms"].min() / 1000
    sent_qps = len(d) / span_s if span_s > 0 else float("nan")
    lag = d["send_lag_ms"]
    held = "held" if lag.quantile(0.99) < 20 and load.get("saturated_dispatches", 0) == 0 else "NOT fully held"
    return (f"Open loop at **{load['target_qps']:g} QPS** total ({load['arrival']} arrivals), measured send rate "
            f"{sent_qps:.1f} QPS, send lag P50 {lag.median():.1f} ms / P99 {lag.quantile(0.99):.1f} ms. Rate {held}. "
            f"{rounds} rounds, target order shuffled each round; max in flight {load['max_in_flight']}"
            + (f" per process, {load['procs']} client processes." if load.get("procs", 1) > 1 else "."))


def build_report(run_dir: str, n_boot: int = 2000, metric: str = "latency_ms") -> dict:
    raw = os.path.join(run_dir, "raw.csv")
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    df = stats.load_raw(raw)
    order = [t["name"] for t in meta.get("targets", [])]
    df["target"] = pd.Categorical(df["target"], categories=order or None, ordered=bool(order))
    df = df.sort_values(["target", "round"]).assign(target=lambda d: d["target"].astype(str))
    if metric not in df or df[metric].dropna().empty:
        raise SystemExit(f"no values for metric {metric}")
    summary = stats.summarize(df, metric)
    cmp = stats.all_comparisons(df, metric, n_boot=n_boot)
    summary.to_csv(os.path.join(run_dir, f"summary_{metric}.csv"), index=False)
    cmp.to_csv(os.path.join(run_dir, f"comparisons_{metric}.csv"), index=False)

    srv_path = os.path.join(run_dir, "server.csv")
    srv = summarize_server(srv_path) if os.path.exists(srv_path) else None

    sfx = "" if metric == "latency_ms" else f"_{metric}"
    cdir = os.path.join(run_dir, f"charts{sfx}")
    os.makedirs(cdir, exist_ok=True)
    charts = [p for p in (
        chart_overhead(cmp, os.path.join(cdir, "overhead_vs_direct.png"), "vs_direct",
                       "Added latency vs. calling Bedrock directly"),
        chart_overhead(cmp, os.path.join(cdir, "overhead_vs_baseline.png"), "vs_baseline",
                       "Incremental latency of each gateway feature"),
        chart_distribution(df, os.path.join(cdir, "distribution.png"), metric),
        chart_timeline(df, os.path.join(cdir, "timeline.png"), metric),
    ) if p]

    cfg = meta["config"]
    w = cfg["workload"]
    env = meta.get("env", {})
    n_meas = int(summary["n"].max()) if not summary.empty else 0

    # Headline numbers
    head = []
    vd = cmp[cmp["kind"] == "vs_direct"].set_index("target") if not cmp.empty else pd.DataFrame()
    p50s = summary.set_index("target").get("p50", pd.Series())
    for t in ("ug:baseline", "ug:all_features", "ug_provider:direct", "se:baseline", "se:all_features",
              "mock:baseline", "strands:ug:baseline", "strands:ug:all_features", "strands:ug_provider"):
        if t in vd.index and pd.notna(vd.loc[t].get("paired_median")):
            r = vd.loc[t]
            direct_p50 = p50s.get(r["reference"])
            head.append(f"**{label(t)}** adds a median of **{r['paired_median']:.1f} ms** "
                        f"(95% CI {r['paired_median_lo']:.1f} to {r['paired_median_hi']:.1f} ms) per request"
                        + (f", about {100 * r['paired_median'] / direct_p50:.1f}% of the {direct_p50:.0f} ms "
                           f"direct-Bedrock P50 ({metric})." if direct_p50 else "."))

    warn = []
    rel = stats.tail_reliability(n_meas, 99)
    if rel != "ok":
        warn.append(f"P99 is based on only ~{n_meas * 0.01:.0f} tail samples per target ({rel}). "
                    "Run at least 2,000 rounds for a stable P99.")
    errs = summary[summary["error_rate"] > 0.01]
    for r in errs.itertuples():
        warn.append(f"{label(r.target)} had {r.error_rate * 100:.1f}% errors. Check raw.csv `error` column.")

    sections = [
        ("Summary", head or ["No Databricks-vs-direct comparison available in this run."], None),
        ("Test setup", [
            f"Run `{meta['run_id']}`, {meta.get('started_utc', '')[:19]} to {meta.get('finished_utc', '')[:19]} UTC.",
            f"Model: `{cfg['aws'].get('bedrock_model_id')}` in AWS region `{cfg['aws']['region']}`.",
            f"Workload: {w['max_tokens']} max output tokens, temperature {w['temperature']}, "
            f"streaming={'on' if w['stream'] else 'off'}, prompt: \"{w['prompt'][:80]}\".",
            _load_line(meta, df),
            f"Client: {env.get('hostname')} ({env.get('platform')}), AZ: {env.get('client_az') or 'unknown'}, "
            f"Databricks runtime: {env.get('databricks_runtime') or 'no'}.",
        ], None),
        ("Latency by target", [f"Metric: `{metric}`. Successful requests only."], table_latency(summary, df)),
        ("Added latency vs. Bedrock direct", [
            "Paired median delta: within each round, the Databricks request minus the direct "
            "request, then the median across rounds. Provider noise largely cancels out, so this "
            "is the most precise estimate of typical added latency. The percentile deltas compare "
            "the two distributions at that percentile."], table_cmp(cmp, "vs_direct")),
        ("Incremental cost of gateway features", [
            "Each variant compared with the same path and no features."], table_cmp(cmp, "vs_baseline")),
    ]
    if srv is not None and not srv.empty:
        sections.append(("Server-side breakdown (system.ai_gateway.usage)", [
            "Gateway-internal = gateway total latency minus time spent waiting on Bedrock. This is "
            "measured the same way as the product team reference below."], table_server(srv)))
    sections.append(("Reference: Databricks product team internal measurement (gateway-internal overhead)", [
        "Server-side overhead added by the gateway itself, excluding network hops and model time."],
        PRODUCT_REFERENCE.rename(columns={"configuration": "Configuration", "p50_ms": "P50 (ms)", "p99_ms": "P99 (ms)"})))
    sections.append(("How to read these results", [
        "Client-side deltas include gateway processing **plus** the extra network hop "
        "(client to Databricks control plane to Bedrock). Keep the client, the Databricks workspace, "
        "and the Bedrock region in the same AWS region to measure only what the gateway adds.",
        "Model generation time varies by hundreds of ms between identical requests. A delta whose CI "
        "includes 0 means the overhead could not be told apart from that noise.",
        "Usage tracking is always on in Unity Gateway, so every `ug:*` row includes it. The `se:*` "
        "(legacy serving endpoint) rows isolate it as a separate toggle.",
        "Rate limits are set far above the benchmark rate, so they are evaluated on every request "
        "but never trigger.",
    ] + ([f"WARNING: {x}" for x in warn]), None))

    # ---- markdown
    md = [f"# Unity Gateway latency benchmark: {cfg['run_name']}\n",
          f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by ugbench._\n"]
    for title, paras, table in sections:
        md.append(f"## {title}\n")
        md += [("- " if title in ("Summary", "Test setup", "How to read these results") else "") + p + "\n"
               for p in paras]
        if table is not None:
            md.append(_md_table(table))
    md.append("## Charts\n")
    md += [f"![{os.path.basename(p)}](charts{sfx}/{os.path.basename(p)})\n" for p in charts]
    md_path = os.path.join(run_dir, f"report{sfx}.md")
    open(md_path, "w").write("\n".join(md))

    # ---- html (self-contained, images embedded)
    def img(p):
        return f'<img alt="{os.path.basename(p)}" src="data:image/png;base64,{base64.b64encode(open(p, "rb").read()).decode()}">'

    def inline_md(s):
        import re
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)

    body = []
    for title, paras, table in sections:
        body.append(f"<h2>{html.escape(title)}</h2>")
        if title in ("Summary", "Test setup", "How to read these results"):
            body.append("<ul>" + "".join(f"<li>{inline_md(p)}</li>" for p in paras) + "</ul>")
        else:
            body += [f"<p>{inline_md(p)}</p>" for p in paras]
        if table is not None:
            body.append(_html_table(table))
    body.append("<h2>Charts</h2>" + "".join(img(p) for p in charts))
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Unity Gateway latency benchmark: {html.escape(cfg['run_name'])}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;background:{SURFACE};color:{INK};
max-width:1100px;margin:32px auto;padding:0 24px;line-height:1.45}}
h1{{font-size:24px}} h2{{font-size:18px;margin-top:32px;border-bottom:1px solid {GRID};padding-bottom:4px}}
table.t{{border-collapse:collapse;font-size:13px;width:100%;margin:8px 0}}
table.t th{{text-align:left;color:{INK2};font-weight:600;border-bottom:2px solid {GRID};padding:6px 8px}}
table.t td{{border-bottom:1px solid {GRID};padding:6px 8px;font-variant-numeric:tabular-nums}}
code{{background:#f0efec;padding:1px 4px;border-radius:3px}} img{{max-width:100%;margin:12px 0}}
.muted{{color:{INK2}}}
</style></head><body>
<h1>Unity Gateway latency benchmark: {html.escape(cfg['run_name'])}</h1>
<p class="muted">Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by ugbench. Raw data: raw.csv</p>
{''.join(body)}
</body></html>"""
    html_path = os.path.join(run_dir, f"report{sfx}.html")
    open(html_path, "w").write(page)
    print(f"[report] {md_path}\n[report] {html_path}")
    return {"summary": summary, "comparisons": cmp, "md": md_path, "html": html_path, "charts": charts}
