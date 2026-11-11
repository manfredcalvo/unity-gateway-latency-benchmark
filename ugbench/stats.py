"""Statistics: percentiles, overhead deltas, and bootstrap confidence intervals.

Two complementary estimators of the overhead added by the gateway:

1. **Percentile delta** ``Pq(target) - Pq(reference)``. This is the number
   customers think in ("P50 went from X to Y"). The 95% CI comes from a bootstrap
   that resamples each distribution independently.
2. **Paired median delta** ``median(target_r - reference_r)`` over rounds ``r``
   where both succeeded. Both requests in a round ran seconds apart, so most of
   the provider's latency noise cancels out. This is the most precise estimate
   of the typical added latency.

LLM latency is noisy (hundreds of ms of jitter from the model) next to a gateway
cost of a few ms. Report the CI, not just the point estimate. A delta whose CI
straddles 0 means the overhead is too small to tell apart from noise at that
sample size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PCTS = (50, 90, 95, 99)
DELTA_PCTS = (50, 90, 99)


def load_raw(path: str, include_warmup: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not include_warmup:
        df = df[df["phase"] == "measure"]
    return df


def summarize(df: pd.DataFrame, metric: str = "latency_ms") -> pd.DataFrame:
    rows = []
    for tgt, g in df.groupby("target", sort=False):
        ok = g[g["ok"] == 1][metric].dropna().to_numpy()
        row = {"target": tgt, "group": g["group"].iloc[0], "features": g["features"].fillna("").iloc[0],
               "n": len(g), "n_ok": len(ok), "error_rate": 1 - len(ok) / max(len(g), 1)}
        if len(ok):
            row.update(mean=ok.mean(), std=ok.std(ddof=1) if len(ok) > 1 else 0.0, min=ok.min(), max=ok.max())
            for p in PCTS:
                row[f"p{p}"] = float(np.percentile(ok, p))
        rows.append(row)
    return pd.DataFrame(rows)


def _boot_pct_delta(a: np.ndarray, b: np.ndarray, q: float, n_boot: int, rng) -> tuple[float, float]:
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    d = np.percentile(a[ia], q, axis=1) - np.percentile(b[ib], q, axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _boot_median(x: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = np.median(x[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(df: pd.DataFrame, target: str, reference: str, metric: str = "latency_ms",
            n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ok = df[df["ok"] == 1]
    a = ok[ok["target"] == target][metric].dropna().to_numpy()
    b = ok[ok["target"] == reference][metric].dropna().to_numpy()
    out = {"target": target, "reference": reference, "metric": metric, "n_target": len(a), "n_ref": len(b)}
    if len(a) < 5 or len(b) < 5:
        return out
    # Bound bootstrap memory (n_boot x n index matrix) for very large runs: ~30M indices per resample set.
    # With tens of thousands of samples, a few hundred replicates already give a stable 95% CI.
    n_boot = max(200, min(n_boot, int(3e7 / max(len(a), len(b)))))
    for q in DELTA_PCTS:
        out[f"d_p{q}"] = float(np.percentile(a, q) - np.percentile(b, q))
        out[f"d_p{q}_lo"], out[f"d_p{q}_hi"] = _boot_pct_delta(a, b, q, n_boot, rng)
    pa = ok[ok["target"] == target][["round", metric]].dropna()
    pb = ok[ok["target"] == reference][["round", metric]].dropna()
    m = pa.merge(pb, on="round", suffixes=("_t", "_r"))
    if len(m) >= 5:
        d = (m[f"{metric}_t"] - m[f"{metric}_r"]).to_numpy()
        out["n_pairs"] = len(m)
        out["paired_median"] = float(np.median(d))
        out["paired_median_lo"], out["paired_median_hi"] = _boot_median(d, n_boot, rng)
        out["paired_p90_of_diff"] = float(np.percentile(d, 90))
    out["rel_p50_pct"] = 100 * out["d_p50"] / float(np.percentile(b, 50))
    return out


def _layer(name: str) -> str:
    return "strands:" if name.startswith("strands:") else ""


def default_comparisons(targets: list[str]) -> list[tuple[str, str, str]]:
    """Return (target, reference, kind) tuples. Each client layer is compared only within itself.

    kind = "vs_direct"   : total added latency of the Databricks path vs. calling Bedrock directly
                           (raw layer: vs ``bedrock_direct``; Strands layer: vs ``strands:direct``)
    kind = "vs_baseline" : incremental cost of a gateway feature vs. the same path with no features
    """
    comps = []
    direct_ref = {"": "bedrock_direct", "strands:": "strands:direct"}
    for t in targets:
        ref = direct_ref[_layer(t)]
        if t != ref and ref in targets:
            comps.append((t, ref, "vs_direct"))
    for prefix in ("ug:", "se:", "mock:", "strands:ug:"):
        base = f"{prefix}baseline"
        if base in targets:
            comps += [(t, base, "vs_baseline") for t in targets
                      if t.startswith(prefix) and t != base and _layer(t) == _layer(base)]
    return comps


def all_comparisons(df: pd.DataFrame, metric: str = "latency_ms", n_boot: int = 2000) -> pd.DataFrame:
    targets = list(dict.fromkeys(df["target"]))
    rows = []
    for t, ref, kind in default_comparisons(targets):
        r = compare(df, t, ref, metric=metric, n_boot=n_boot)
        r["kind"] = kind
        rows.append(r)
    return pd.DataFrame(rows)


def tail_reliability(n: int, q: float) -> str:
    tail = n * (100 - q) / 100
    if tail >= 20:
        return "ok"
    if tail >= 5:
        return "low"
    return "insufficient"
