"""Confidence intervals and paired comparisons from the per-image rows (`perimage_test.json`).

    python -m benchmark.report_stats

Unit of resampling = image (micro) or FIELD (capillary, field-level protocol). Writes into the results folder:
  stats_ci.csv           one row per run: point estimate and 95% bootstrap CI of F1, pooled Dice, area error, count error
  stats_vs_baseline.csv  every normalized run vs the SAME model without normalization: paired bootstrap of the
                         difference, two-sided p-value, Holm-adjusted p-value (family = all comparisons of a task)
With 6 test fields per mare and one fold, treat these intervals as a within-mare measure of precision only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.core import RESULTS, read_json

N_BOOT = 2000
STATS = ("f1", "dice", "area_err", "count_mae")


def stat_values(rows: pd.DataFrame, idx: np.ndarray | None = None) -> dict:
    r = rows if idx is None else rows.iloc[idx]
    tp, fp, fn = r["tp"].sum(), r["fp"].sum(), r["fn"].sum()
    gt = r["area_gt_sum"].replace(0, np.nan)
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "dice": 2 * r["inter"].sum() / max(r["area_pred_union"].sum() + r["area_gt_union"].sum(), 1),
            "area_err": float(((r["area_pred_sum"] - r["area_gt_sum"]).abs() / gt).mean()),
            "count_mae": float((r["n_pred"] - r["n_gt"]).abs().mean())}


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load(results: Path) -> list[dict]:
    runs = []
    for rj in sorted(results.glob("*/perimage_test.json")):
        cfg = read_json(rj.parent / "config.json")
        pj = read_json(rj)
        part = pj.get("field") or pj["image"]  # capillary: field level is the protocol
        runs.append({"run_id": rj.parent.name, "task": cfg["task"], "model": cfg["model"], "norm": cfg["norm"],
                     "unit": part["unit"], "rows": pd.DataFrame(part["rows"])})
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args()
    runs = load(a.results)
    rng = np.random.default_rng(0)
    # one set of resamples per sample size, shared by every run of that size => the comparisons are paired
    idx_by_n: dict[int, np.ndarray] = {}
    ci_rows, boots = [], {}
    for r in runs:
        n = len(r["rows"])
        idx = idx_by_n.setdefault(n, rng.integers(0, n, (N_BOOT, n)))
        b = pd.DataFrame([stat_values(r["rows"], i) for i in idx])
        boots[r["run_id"]] = b
        pt = stat_values(r["rows"])
        row = {k: r[k] for k in ("run_id", "task", "model", "norm", "unit")} | {"n": n}
        for s in STATS:
            row[s] = pt[s]
            row[f"{s}_lo"], row[f"{s}_hi"] = np.percentile(b[s], [2.5, 97.5])
        ci_rows.append(row)
    pd.DataFrame(ci_rows).to_csv(a.results / "stats_ci.csv", index=False)

    cmp_rows = []
    for task in sorted({r["task"] for r in runs}):
        base = {r["model"]: r for r in runs if r["task"] == task and r["norm"] == "none"}
        sub = []
        for r in runs:
            if r["task"] != task or r["norm"] == "none" or r["model"] not in base:
                continue
            b0, b1 = boots[base[r["model"]]["run_id"]], boots[r["run_id"]]
            for s in STATS:
                d = b1[s].to_numpy() - b0[s].to_numpy()
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                sub.append({"task": task, "model": r["model"], "norm": r["norm"], "stat": s,
                            "diff": stat_values(r["rows"])[s] - stat_values(base[r["model"]]["rows"])[s],
                            "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5), "p": max(p, 1 / N_BOOT)})
        if sub:
            df = pd.DataFrame(sub)
            df["p_holm"] = holm(df["p"].to_numpy())
            cmp_rows.append(df)
    cmp = pd.concat(cmp_rows, ignore_index=True) if cmp_rows else pd.DataFrame()
    cmp.to_csv(a.results / "stats_vs_baseline.csv", index=False)
    print(f"{len(ci_rows)} execuções -> stats_ci.csv | {len(cmp)} comparações -> stats_vs_baseline.csv")


if __name__ == "__main__":
    main()
