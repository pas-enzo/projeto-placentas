"""Slide figure: Original | prediction in the PIPELINE colors | difference against the annotation, for the champion
micro model (YOLO26s, primary fold) on one TEST field.

    python scripts/figs/make_slide_micro_example.py --out <png> [--field-index N] [--device cpu]

The field is, by default, the one whose F1 is closest to the median of the 30 test fields (a typical case, not the best).
Difference map: yellow = predicted and annotated, green = only predicted, red = only annotated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.adapters.ultralytics_adapter import UltralyticsPredictor  # noqa: E402
from benchmark.eval.instances import read_gt_polygons  # noqa: E402
from pipeline.viz import colorize_overlay  # noqa: E402

RESULTS = Path(r"D:\projeto_placentas_clayton\benchmark_results")
VIEW = Path(r"D:\projeto_placentas_clayton\datasets_v4\test-ZAZA_val-TOSTADA\micro_stretch640\test")


def _run_dir(model: str) -> Path:
    import pandas as pd

    df = pd.read_csv(RESULTS / "results.csv")
    r = df[(df.task == "micro") & (df.model == model) & (df.fold == "primary") & (df.norm == "none")]
    return RESULTS / r.run_id.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="yolo26s-seg.pt")
    ap.add_argument("--field-index", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    run = _run_dir(a.model)
    conf = json.loads((run / "metrics_val.json").read_text(encoding="utf-8"))["conf_star"]
    rows = json.loads((run / "perimage_test.json").read_text(encoding="utf-8"))["image"]["rows"]
    f1 = np.array([2 * r["tp"] / max(2 * r["tp"] + r["fp"] + r["fn"], 1) for r in rows])
    files = sorted(p for p in (VIEW / "images").iterdir() if p.suffix.lower() in {".jpg", ".png"})
    k = a.field_index if a.field_index is not None else int(np.argmin(np.abs(f1 - np.median(f1))))
    img_path = files[k]
    print(f"campo {k}: {img_path.name[:20]}  F1 do campo {f1[k]:.3f} (mediana {np.median(f1):.3f}), conf {conf}")

    bgr = cv2.imread(str(img_path))
    h, w = bgr.shape[:2]
    pr = UltralyticsPredictor(run / "train" / "run" / "weights" / "best.pt", conf=conf, device=a.device)
    pred = [i for i in pr.predict(bgr) if i.score >= conf]
    gt = read_gt_polygons(VIEW / "labels" / f"{img_path.stem}.txt", h, w, 0)
    pm = [i.full(h, w).astype(np.uint8) for i in pred]
    gm = [i.full(h, w).astype(np.uint8) for i in gt]
    pu = np.clip(np.sum(pm, 0), 0, 1).astype(bool) if pm else np.zeros((h, w), bool)
    gu = np.clip(np.sum(gm, 0), 0, 1).astype(bool) if gm else np.zeros((h, w), bool)

    overlay = colorize_overlay(bgr, micro_masks=pm)
    diff = np.zeros((h, w, 3), np.uint8)
    diff[pu & gu] = (0, 255, 255)   # BGR yellow: hit
    diff[pu & ~gu] = (0, 200, 0)    # green: only predicted
    diff[~pu & gu] = (0, 0, 220)    # red: only annotated
    iou_px = (pu & gu).sum() / max((pu | gu).sum(), 1)
    print(f"{len(pred)} previstos, {len(gt)} anotados; IoU de pixels da união {iou_px:.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, BLUE = "#FFFCF3", "#596AB6"
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5), dpi=200)
    fig.patch.set_facecolor(BG)
    for ax, im, t in zip(axs, (bgr, overlay, diff), ("Original", "Predição (YOLO26s)", "Diferença para a anotação")):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, color=BLUE, fontsize=14, fontweight="bold", pad=6)
        ax.axis("off")
    fig.text(0.5, 0.045, "amarelo: previsto e anotado   |   verde: só previsto   |   vermelho: só anotado"
             f"      (IoU de pixels: {iou_px:.2f}".replace(".", ",") + f"; {len(pred)} previstos, {len(gt)} anotados)",
             ha="center", color=BLUE, fontsize=11)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.03)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=BG)


if __name__ == "__main__":
    main()
