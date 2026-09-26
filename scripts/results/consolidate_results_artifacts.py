"""Consolidate Results artifacts from existing pipeline outputs (no re-training).

Sources (read-only):
  Microcot: v2_rfdetr_seg_large_opt_v1/artifacts/...
  Capilar:  v3_capilar_yolo11s/artifacts_v4/...

Writes under: figs/results/
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(r"D:\projeto_placentas_clayton\dev\projeto-placentas")
OUT = ROOT / "figs" / "results"
OUT.mkdir(parents=True, exist_ok=True)

MICRO_SWEEP = (
    ROOT
    / "v2_rfdetr_seg_large_opt_v1"
    / "artifacts"
    / "benchmarks"
    / "validation_conf_sweep_seg_large_r672_auto.csv"
)
MICRO_CHAMP = (
    ROOT / "v2_rfdetr_seg_large_opt_v1" / "artifacts" / "benchmarks" / "champion_run.json"
)
MICRO_FINAL = (
    ROOT / "v2_rfdetr_seg_large_opt_v1" / "artifacts" / "benchmarks" / "final_summary.json"
)
MICRO_INFER = (
    ROOT
    / "v2_rfdetr_seg_large_opt_v1"
    / "artifacts"
    / "benchmarks"
    / "inference_benchmark_seg_large_r672_auto.json"
)
MICRO_TOTALS = (
    ROOT
    / "v2_rfdetr_seg_large_opt_v1"
    / "artifacts"
    / "reports"
    / "seg_large_r672_auto"
    / "placenta_totals_report_rfdetr.csv"
)
MICRO_IOU = (
    ROOT
    / "v2_rfdetr_seg_large_opt_v1"
    / "artifacts"
    / "iou_viz"
    / "seg_large_r672_auto"
)

CAP_SEL = ROOT / "v3_capilar_yolo11s" / "artifacts_v4" / "benchmarks" / "selected_confidence.json"
CAP_SWEEP = ROOT / "v3_capilar_yolo11s" / "artifacts_v4" / "benchmarks" / "validation_conf_sweep.csv"
CAP_TOTALS = ROOT / "v3_capilar_yolo11s" / "artifacts_v4" / "reports" / "capilar_tile_totals_report.csv"
CAP_IOU = ROOT / "v3_capilar_yolo11s" / "artifacts_v4" / "iou_viz"

# Same FOV as Methods Figs 1–3 (stretch hash for micro; native hash for cap tiles)
MICRO_FIG_CANDIDATES = [
    "iou_viz_ROSILHA-M-D_006_jpg.rf.de71085683e9c7eb460be0f184fb31ac.png",
    "iou_viz_ROSILHA-M-G_001_jpg.rf.3dab74a9f6c203f6aca8afc07a353bd2.png",
    "iou_viz_ZAZA-G_010_jpg.rf.cd14098445a0a54c10014aaa7d8f975c.png",
]
CAP_FIG_CANDIDATES = [
    "iou_viz_ROSILHA-M-D_006_jpg.rf.4bf38cf5314617b13c7c11445fe40b04_r0c2.png",
    "iou_viz_TP-D_009_jpg.rf.e8cdb11635df3be6c85d743284d8ecfb_r1c2.png",
    "iou_viz_ZAINA-679-D_007_jpg.rf.34de0be2ff800849eeb730bc4890f94f_r2c1.png",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(x: str | float) -> float:
    return float(x)


def main() -> None:
    champ = json.loads(MICRO_CHAMP.read_text(encoding="utf-8"))
    final = json.loads(MICRO_FINAL.read_text(encoding="utf-8"))
    infer = json.loads(MICRO_INFER.read_text(encoding="utf-8"))
    micro_rows = read_csv(MICRO_SWEEP)
    micro_best = max(micro_rows, key=lambda r: fnum(r["score"]))
    assert abs(fnum(micro_best["conf"]) - fnum(champ["best_conf"])) < 1e-9

    micro_totals = read_csv(MICRO_TOTALS)
    gt_um2 = sum(fnum(r["GT_Area_um2"]) for r in micro_totals)
    ai_um2 = sum(fnum(r["AI_Area_um2"]) for r in micro_totals)
    gt_px = sum(fnum(r["GT_Area_px"]) for r in micro_totals)
    ai_px = sum(fnum(r["AI_Area_px"]) for r in micro_totals)
    area_factor_implied = gt_um2 / gt_px if gt_px else None
    area_rel_from_totals = abs(ai_um2 - gt_um2) / gt_um2 if gt_um2 else None

    cap_sel = json.loads(CAP_SEL.read_text(encoding="utf-8"))
    cap_rows = read_csv(CAP_SWEEP)
    cap_best = max(cap_rows, key=lambda r: fnum(r["score"]))
    assert abs(fnum(cap_best["conf"]) - fnum(cap_sel["best_conf"])) < 1e-9
    cap_totals = read_csv(CAP_TOTALS)

    # Operating-point table
    op_path = OUT / "table_operating_points.csv"
    with op_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "model",
                "run_name",
                "eval_unit",
                "n_units",
                "conf_star",
                "S_c",
                "precision",
                "recall",
                "f1",
                "mean_iou",
                "area_rel_error",
                "source",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "task": "microcotyledon",
                "model": "RF-DETR Seg Large",
                "run_name": champ["run_name"],
                "eval_unit": "FOV",
                "n_units": len(micro_totals),
                "conf_star": micro_best["conf"],
                "S_c": micro_best["score"],
                "precision": micro_best["precision"],
                "recall": micro_best["recall"],
                "f1": micro_best["f1"],
                "mean_iou": micro_best["mean_iou"],
                "area_rel_error": micro_best["area_rel_error"],
                "source": str(MICRO_SWEEP.relative_to(ROOT)),
            }
        )
        w.writerow(
            {
                "task": "capillary",
                "model": "YOLO11s-seg",
                "run_name": cap_sel["run_name"],
                "eval_unit": "tile",
                "n_units": len(cap_totals),
                "conf_star": cap_best["conf"],
                "S_c": cap_best["score"],
                "precision": cap_best["precision"],
                "recall": cap_best["recall"],
                "f1": cap_best["f1"],
                "mean_iou": cap_best["mean_iou"],
                "area_rel_error": cap_best["area_rel_error"],
                "source": str(CAP_SWEEP.relative_to(ROOT)),
            }
        )

    # Area totals table
    area_path = OUT / "table_area_totals.csv"
    with area_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "n_units",
                "gt_count_sum",
                "ai_count_sum",
                "matched_sum",
                "fp_sum",
                "fn_sum",
                "gt_area_um2",
                "ai_area_um2",
                "area_rel_error_from_totals",
                "area_factor_um2_per_px2",
                "source",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "task": "microcotyledon",
                "n_units": len(micro_totals),
                "gt_count_sum": sum(fnum(r["GT_Count"]) for r in micro_totals),
                "ai_count_sum": sum(fnum(r["AI_Count"]) for r in micro_totals),
                "matched_sum": sum(fnum(r["Matched_Count"]) for r in micro_totals),
                "fp_sum": sum(fnum(r["FP_Count"]) for r in micro_totals),
                "fn_sum": sum(fnum(r["FN_Count"]) for r in micro_totals),
                "gt_area_um2": gt_um2,
                "ai_area_um2": ai_um2,
                "area_rel_error_from_totals": area_rel_from_totals,
                "area_factor_um2_per_px2": area_factor_implied,
                "source": str(MICRO_TOTALS.relative_to(ROOT)),
            }
        )
        w.writerow(
            {
                "task": "capillary",
                "n_units": len(cap_totals),
                "gt_count_sum": sum(fnum(r["GT_Count"]) for r in cap_totals),
                "ai_count_sum": sum(fnum(r["AI_Count"]) for r in cap_totals),
                "matched_sum": sum(fnum(r["Matched"]) for r in cap_totals),
                "fp_sum": sum(fnum(r["FP"]) for r in cap_totals),
                "fn_sum": sum(fnum(r["FN"]) for r in cap_totals),
                "gt_area_um2": sum(fnum(r["GT_Area_um2"]) for r in cap_totals),
                "ai_area_um2": sum(fnum(r["AI_Area_um2"]) for r in cap_totals),
                "area_rel_error_from_totals": abs(
                    sum(fnum(r["AI_Area_um2"]) for r in cap_totals)
                    - sum(fnum(r["GT_Area_um2"]) for r in cap_totals)
                )
                / sum(fnum(r["GT_Area_um2"]) for r in cap_totals),
                "area_factor_um2_per_px2": cap_sel["area_factor_um2_per_px2"],
                "source": str(CAP_TOTALS.relative_to(ROOT)),
            }
        )

    # Inference table (micro filled; cap pending)
    ms_per_fov = 1000.0 * fnum(infer["elapsed_sec_mean"]) / fnum(infer["n_images"])
    infer_path = OUT / "table_inference.csv"
    with infer_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "unit",
                "n_units",
                "conf",
                "warmup",
                "repeats",
                "elapsed_sec_mean",
                "elapsed_sec_std",
                "ms_per_unit",
                "units_per_sec",
                "peak_gpu_mem_mb",
                "optimize_for_inference",
                "status",
                "source",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "task": "microcotyledon",
                "unit": "FOV",
                "n_units": infer["n_images"],
                "conf": infer["conf"],
                "warmup": infer["warmup"],
                "repeats": infer["repeats"],
                "elapsed_sec_mean": infer["elapsed_sec_mean"],
                "elapsed_sec_std": infer["elapsed_sec_std"],
                "ms_per_unit": ms_per_fov,
                "units_per_sec": infer["images_per_sec"],
                "peak_gpu_mem_mb": infer["peak_gpu_mem_mb"],
                "optimize_for_inference": "attempted_in_notebook_protocol",
                "status": "OK",
                "source": str(MICRO_INFER.relative_to(ROOT)),
            }
        )
        w.writerow(
            {
                "task": "capillary",
                "unit": "tile",
                "n_units": 243,
                "conf": cap_sel["best_conf"],
                "warmup": "",
                "repeats": "",
                "elapsed_sec_mean": "",
                "elapsed_sec_std": "",
                "ms_per_unit": "",
                "units_per_sec": "",
                "peak_gpu_mem_mb": "",
                "optimize_for_inference": "n/a",
                "status": "PENDING — no inference_benchmark JSON for YOLO11s v4",
                "source": "",
            }
        )

    # Figure manifest (paths only; do not rewrite PNGs yet)
    fig_manifest = {
        "microcotyledon_overlays": {
            "dir": str(MICRO_IOU.relative_to(ROOT)).replace("\\", "/"),
            "recommended": [
                str((MICRO_IOU / n).relative_to(ROOT)).replace("\\", "/")
                for n in MICRO_FIG_CANDIDATES
                if (MICRO_IOU / n).exists()
            ],
            "n_available": len(list(MICRO_IOU.glob("iou_viz_*.png"))),
            "note": "Existing GT vs pred IoU overlays from RF-DETR champion eval; not yet reframed for paper.",
        },
        "capillary_overlays": {
            "dir": str(CAP_IOU.relative_to(ROOT)).replace("\\", "/"),
            "recommended": [
                str((CAP_IOU / n).relative_to(ROOT)).replace("\\", "/")
                for n in CAP_FIG_CANDIDATES
                if (CAP_IOU / n).exists()
            ],
            "n_available": len(list(CAP_IOU.glob("iou_viz_*.png"))),
            "note": "Tile-level overlays only (Methods: no FOV merge).",
        },
    }
    (OUT / "figure_manifest.json").write_text(
        json.dumps(fig_manifest, indent=2), encoding="utf-8"
    )

    summary = {
        "generated_by": "scripts/results/consolidate_results_artifacts.py",
        "microcotyledon": {
            "champion_run.json": champ,
            "operating_point_from_conf_sweep": {
                k: (fnum(v) if k != "run_name" else v) for k, v in micro_best.items()
            },
            "area_totals": {
                "n_fov": len(micro_totals),
                "gt_area_um2": gt_um2,
                "ai_area_um2": ai_um2,
                "area_rel_error_from_totals": area_rel_from_totals,
                "implied_area_factor": area_factor_implied,
                "note": "AreaRelErr in sweep uses px totals; totals CSV also stores µm² with project factor (~0.3606).",
            },
            "inference": infer,
            "ms_per_fov_derived": ms_per_fov,
            "ms_per_fov_formula": "1000 * elapsed_sec_mean / n_images",
        },
        "capillary": {
            "selected_confidence.json": cap_sel,
            "operating_point_from_conf_sweep": {
                k: fnum(v) for k, v in cap_best.items()
            },
            "n_tiles": len(cap_totals),
            "inference": None,
            "inference_status": "PENDING",
        },
        "outputs": {
            "table_operating_points.csv": str(op_path.relative_to(ROOT)).replace("\\", "/"),
            "table_area_totals.csv": str(area_path.relative_to(ROOT)).replace("\\", "/"),
            "table_inference.csv": str(infer_path.relative_to(ROOT)).replace("\\", "/"),
            "figure_manifest.json": "figs/results/figure_manifest.json",
        },
    }
    (OUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    provenance = """# Results artifacts — provenance

Generated by `scripts/results/consolidate_results_artifacts.py` from existing pipeline outputs.
No models were re-trained. Capillary inference latency was NOT measured in this pass
(PyTorch import hung on this machine during the session).

## Microcotyledon (RF-DETR Seg Large, `seg_large_r672_auto`)

| Value | Source |
|-------|--------|
| conf* = 0.44 | `v2_rfdetr_seg_large_opt_v1/artifacts/benchmarks/champion_run.json` AND best `score` row of `validation_conf_sweep_seg_large_r672_auto.csv` |
| P/R/F1/IoU/AreaRelErr/S(c) | Same conf-sweep CSV row at conf=0.44 |
| n = 27 FOVs; GT/AI counts & µm² | `artifacts/reports/seg_large_r672_auto/placenta_totals_report_rfdetr.csv` |
| Latency / throughput / peak VRAM | `artifacts/benchmarks/inference_benchmark_seg_large_r672_auto.json` (warmup=1, repeats=3, n=27) |
| ms/FOV | Derived: `1000 * elapsed_sec_mean / n_images` |
| Overlays | `artifacts/iou_viz/seg_large_r672_auto/*.png` |

## Capillary (YOLO11s-seg, `capilar_yolo11s_tiled_v4`)

| Value | Source |
|-------|--------|
| conf* = 0.33 | `v3_capilar_yolo11s/artifacts_v4/benchmarks/selected_confidence.json` AND best `score` row of `validation_conf_sweep.csv` |
| P/R/F1/IoU/AreaRelErr/S(c) | Conf-sweep CSV row at conf=0.33 |
| n = 243 tiles | `artifacts_v4/reports/capilar_tile_totals_report.csv` |
| AREA_FACTOR | `selected_confidence.json` → 0.0115248… (= Methods ≈0.0115) |
| Inference latency | **PENDING** — no `inference_benchmark_*.json` under artifacts_v4 |
| Overlays | `artifacts_v4/iou_viz/*.png` (tiles) |

## Outputs in `figs/results/`

- `table_operating_points.csv`
- `table_area_totals.csv`
- `table_inference.csv`
- `figure_manifest.json`
- `results_summary.json`
"""
    (OUT / "PROVENANCE.md").write_text(provenance, encoding="utf-8")
    print("Wrote", OUT)
    print("micro conf*", micro_best["conf"], "F1", micro_best["f1"], "AreaRelErr", micro_best["area_rel_error"])
    print("cap  conf*", cap_best["conf"], "F1", cap_best["f1"], "AreaRelErr", cap_best["area_rel_error"])
    print("micro ms/FOV", round(ms_per_fov, 2), "peak_mb", infer["peak_gpu_mem_mb"])
    print("cap inference PENDING")


if __name__ == "__main__":
    main()
