"""Rescore capillary conf sweeps with Methods S(c) = 0.6*F1 + 0.3*IoU + 0.1*(1-ARE).

No re-inference. Same criterion as RF-DETR microcotyledon / paper Methods.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(r"D:\projeto_placentas_clayton\dev\projeto-placentas")
OUT = ROOT / "figs" / "results"
OUT.mkdir(parents=True, exist_ok=True)

TILE_SWEEP = ROOT / "v3_capilar_yolo11s" / "artifacts_v4" / "benchmarks" / "validation_conf_sweep.csv"
FIELD_SWEEP = (
    ROOT / "v3_capilar_yolo11s" / "artifacts_v4_field" / "benchmarks" / "validation_conf_sweep_field.csv"
)
TILE_SEL = ROOT / "v3_capilar_yolo11s" / "artifacts_v4" / "benchmarks" / "selected_confidence.json"
FIELD_SEL = (
    ROOT / "v3_capilar_yolo11s" / "artifacts_v4_field" / "benchmarks" / "selected_confidence_field.json"
)

FORMULA = "0.6*F1 + 0.3*mean_IoU + 0.1*(1-AreaRelErr)"
PREV = "0.5*F1 + 0.3*mean_IoU + 0.2*(1-AreaRelErr)"


def S(f1: float, iou: float, are: float) -> float:
    return 0.6 * f1 + 0.3 * iou + 0.1 * (1.0 - are)


def main() -> None:
    rows = list(csv.DictReader(TILE_SWEEP.open(encoding="utf-8")))
    for r in rows:
        r["score_old_0.5_0.3_0.2"] = r["score"]
        r["score"] = f"{S(float(r['f1']), float(r['mean_iou']), float(r['area_rel_error'])):.16g}"
        r["score_formula"] = FORMULA

    best = max(rows, key=lambda r: float(r["score"]))
    old_best = max(rows, key=lambda r: float(r["score_old_0.5_0.3_0.2"]))
    print("TILE old best conf", old_best["conf"], "score_old", old_best["score_old_0.5_0.3_0.2"])
    print("TILE new best conf", best["conf"], "score_new", best["score"])
    print(
        "  P/R/F1/IoU/ARE",
        best["precision"],
        best["recall"],
        best["f1"],
        best["mean_iou"],
        best["area_rel_error"],
    )

    fields = list(rows[0].keys())
    with TILE_SWEEP.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: float(x["conf"])):
            w.writerow(r)

    with (OUT / "capilar_tile_conf_sweep_S060301.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: -float(x["score"])):
            w.writerow(r)

    sel = {
        "model": "yolo11s-seg",
        "run_name": "capilar_yolo11s_tiled_v4",
        "checkpoint": str(
            ROOT / "v3_capilar_yolo11s" / "runs" / "capilar_yolo11s_tiled_v4" / "weights" / "best.pt"
        ),
        "best_conf": float(best["conf"]),
        "f1": float(best["f1"]),
        "mean_iou": float(best["mean_iou"]),
        "area_rel_error": float(best["area_rel_error"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "weighted_score": float(best["score"]),
        "score_formula": FORMULA,
        "score_formula_note": (
            "Aligned to Methods / RF-DETR microcotyledon criterion. "
            f"Was {PREV}. Rescored from existing sweep metrics; no re-inference."
        ),
        "previous_formula": PREV,
        "previous_best_conf": float(old_best["conf"]),
        "area_factor_um2_per_px2": 0.011524823461313616,
        "classes": [0],
        "notes": "Tile-level Capilar; FOV morphometry via SAHI separately.",
    }
    TILE_SEL.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    (OUT / "selected_confidence_capilar_tile_S060301.json").write_text(
        json.dumps(sel, indent=2), encoding="utf-8"
    )
    print("updated", TILE_SEL)

    frows = list(csv.DictReader(FIELD_SWEEP.open(encoding="utf-8")))
    for r in frows:
        r["score_old_0.5_0.3_0.2"] = r.get("score", "")
        r["score"] = f"{S(float(r['f1']), float(r['mean_iou']), float(r['area_rel_error'])):.16g}"
        r["score_formula"] = FORMULA
    with FIELD_SWEEP.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frows[0].keys()))
        w.writeheader()
        w.writerows(frows)

    fb = frows[0]
    field_sel = json.loads(FIELD_SEL.read_text(encoding="utf-8"))
    field_sel["weighted_score"] = float(fb["score"])
    field_sel["score_formula"] = FORMULA
    field_sel["score_formula_note"] = (
        "Rescored to Methods criterion; FOV/SAHI sweep currently has only conf=0.33."
    )
    field_sel["best_conf"] = float(fb["conf"])
    FIELD_SEL.write_text(json.dumps(field_sel, indent=2), encoding="utf-8")
    print("FIELD conf", fb["conf"], "score_new", fb["score"], "(single conf in FOV sweep)")

    op = OUT / "table_operating_points.csv"
    with op.open("w", encoding="utf-8", newline="") as f:
        cols = [
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
            "score_formula",
            "source",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(
            {
                "task": "microcotyledon",
                "model": "RF-DETR Seg Large",
                "run_name": "seg_large_r672_auto",
                "eval_unit": "FOV",
                "n_units": 27,
                "conf_star": 0.44,
                "S_c": 0.9116264146074771,
                "precision": 0.8928571428571429,
                "recall": 0.9096244131455399,
                "f1": 0.9011627906976744,
                "mean_iou": 0.9054170475820253,
                "area_rel_error": 0.0069637408573528055,
                "score_formula": FORMULA,
                "source": "validation_conf_sweep_seg_large_r672_auto.csv",
            }
        )
        w.writerow(
            {
                "task": "capillary",
                "model": "YOLO11s-seg",
                "run_name": "capilar_yolo11s_tiled_v4",
                "eval_unit": "FOV+SAHI",
                "n_units": 27,
                "conf_star": field_sel["best_conf"],
                "S_c": field_sel["weighted_score"],
                "precision": field_sel["precision"],
                "recall": field_sel["recall"],
                "f1": field_sel["f1"],
                "mean_iou": field_sel["mean_iou"],
                "area_rel_error": field_sel["area_rel_error"],
                "score_formula": FORMULA,
                "source": "selected_confidence_field.json (rescored)",
            }
        )
        w.writerow(
            {
                "task": "capillary",
                "model": "YOLO11s-seg",
                "run_name": "capilar_yolo11s_tiled_v4",
                "eval_unit": "tile",
                "n_units": 243,
                "conf_star": sel["best_conf"],
                "S_c": sel["weighted_score"],
                "precision": sel["precision"],
                "recall": sel["recall"],
                "f1": sel["f1"],
                "mean_iou": sel["mean_iou"],
                "area_rel_error": sel["area_rel_error"],
                "score_formula": FORMULA,
                "source": "validation_conf_sweep.csv rescored 0.6/0.3/0.1",
            }
        )
    print("updated", op)

    # patch notebooks for future runs
    for nb_path in [
        ROOT / "v3_capilar_yolo11s" / "notebooks" / "yolo11s_capilar_train.ipynb",
        ROOT / "v3_capilar_yolo11s" / "notebooks" / "yolo11s_capilar_field_eval.ipynb",
    ]:
        text = nb_path.read_text(encoding="utf-8")
        old = "0.5 * f1 + 0.3 * mean_iou + 0.2 * (1.0 - area_rel_err)"
        new = "0.6 * f1 + 0.3 * mean_iou + 0.1 * (1.0 - area_rel_err)"
        if old in text:
            nb_path.write_text(text.replace(old, new), encoding="utf-8")
            print("patched formula in", nb_path.name)
        else:
            print("formula already patched or missing in", nb_path.name)


if __name__ == "__main__":
    main()
