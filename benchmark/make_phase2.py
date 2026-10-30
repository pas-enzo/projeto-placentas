"""Builds the phase-2 matrix (stain-normalization crossing) from the phase-1 results.

    python -m benchmark.make_phase2 --top-micro 3 --top-capilar 2

Models are chosen by VALIDATION coco mask AP (never the test set), among baseline runs (norm none). Each chosen model is
trained under every normalization of benchmark/views/stain.py with the same recipe/epochs/batch it had in phase 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from benchmark.core import RESULTS

NORMS = ["whitebalance", "grayscale", "histmatch", "macenko", "reinhard", "vahadane"]
CRITERION = "val_report_coco_ap_ap"
EPOCHS = {"micro": 120, "capilar": 60}
BATCH = {"micro": 8, "capilar": 8}


def build(results: Path, tops: dict[str, int]) -> dict:
    df = pd.read_csv(results / "results.csv")
    df = df[(df["norm"] == "none") & (df["fold"] == "primary")]
    runs = []
    for task, k in tops.items():
        sub = df[df["task"] == task].sort_values(CRITERION, ascending=False).head(k)
        if sub.empty:
            continue
        print(f"{task}: " + ", ".join(f"{m} ({a:.3f})" for m, a in zip(sub["model"], sub[CRITERION])))
        for _, r in sub.iterrows():
            runs.append({"task": task, "model": r["model"], "norm": NORMS, "epochs": EPOCHS[task],
                         "batch": int(r["batch_used"]) if pd.notna(r["batch_used"]) else BATCH[task],
                         "recipe": r["recipe"], "tags": ["phase2", "stain"]})
    return {"defaults": {"family": "ultralytics", "fold": "primary", "seed": 0, "patience": 30}, "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--top-micro", type=int, default=3)
    ap.add_argument("--top-capilar", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "configs" / "phase2_stain.yaml")
    a = ap.parse_args()
    cfg = build(a.results, {"micro": a.top_micro, "capilar": a.top_capilar})
    header = "# GERADO por benchmark/make_phase2.py a partir dos resultados da fase 1 (seleção por AP de validação).\n"
    a.out.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"{sum(len(r['norm']) for r in cfg['runs'])} execuções -> {a.out}")


if __name__ == "__main__":
    main()
