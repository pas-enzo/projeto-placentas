"""Consolidate capillary SAHI field metrics into figs/results."""
import csv
import json
from pathlib import Path

ROOT = Path(r"D:\projeto_placentas_clayton\dev\projeto-placentas")
OUT = ROOT / "figs" / "results"
OUT.mkdir(parents=True, exist_ok=True)

sel = json.loads(
    (
        ROOT
        / "v3_capilar_yolo11s"
        / "artifacts_v4_field"
        / "benchmarks"
        / "selected_confidence_field.json"
    ).read_text(encoding="utf-8")
)
sweep = list(
    csv.DictReader(
        (
            ROOT
            / "v3_capilar_yolo11s"
            / "artifacts_v4_field"
            / "benchmarks"
            / "validation_conf_sweep_field.csv"
        ).open(encoding="utf-8")
    )
)[0]
totals = list(
    csv.DictReader(
        (
            ROOT
            / "v3_capilar_yolo11s"
            / "artifacts_v4_field"
            / "reports"
            / "capilar_field_totals_report.csv"
        ).open(encoding="utf-8")
    )
)
print("field totals n=", len(totals), "cols=", list(totals[0].keys()))
print("sample", totals[0])

out = OUT / "table_capilar_field_sahi.csv"
with out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "eval",
            "n_fov",
            "conf",
            "precision",
            "recall",
            "f1",
            "mean_iou",
            "area_rel_error",
            "tp",
            "fp",
            "fn",
            "overlap",
            "slice",
            "gt_area_um2",
            "pred_area_um2",
            "source_metrics",
            "source_totals",
            "note",
        ],
    )
    w.writeheader()
    w.writerow(
        {
            "eval": "sahi_field_overlap0.2",
            "n_fov": sel["n_fov"],
            "conf": sel["best_conf"],
            "precision": sel["precision"],
            "recall": sel["recall"],
            "f1": sel["f1"],
            "mean_iou": sel["mean_iou"],
            "area_rel_error": sel["area_rel_error"],
            "tp": sel["tp"],
            "fp": sel["fp"],
            "fn": sel["fn"],
            "overlap": sel["overlap"],
            "slice": f"{sel['slice'][0]}x{sel['slice'][1]}",
            "gt_area_um2": sweep["gt_area_um2"],
            "pred_area_um2": sweep["pred_area_um2"],
            "source_metrics": "v3_capilar_yolo11s/artifacts_v4_field/benchmarks/selected_confidence_field.json",
            "source_totals": "v3_capilar_yolo11s/artifacts_v4_field/reports/capilar_field_totals_report.csv",
            "note": (
                "SAHI sliding-window overlap=0.2 on native FOVs. "
                "NOT identical to non-overlapping 3x3 border-merge described as future work in early drafts; "
                "Methods primary remains tile-level."
            ),
        }
    )
print("wrote", out)

# Also dump per-FOV count/area summary if columns exist
count_cols = [c for c in totals[0] if "count" in c.lower() or c in ("GT", "AI", "Matched", "FP", "FN")]
area_cols = [c for c in totals[0] if "area" in c.lower() or "um2" in c.lower()]
print("count-like", count_cols)
print("area-like", area_cols)
