"""Slide figures for the capillary model (YOLO11s, primary fold).

  * f1 curve: F1 x confidence on the VALIDATION tiles for the three best capillary models of the primary fold.
  * example: one TEST field through the project's field protocol (16 SAHI windows + greedy merge); three panels cropped to
    the same 1380x1032 window: Original | prediction in the PIPELINE colors | difference against the annotation.
    The window is the one with the most annotated capillary area; the field is the one whose F1 is closest to the median.
    Difference map: yellow = predicted and annotated, green = only predicted, red = only annotated.

    python scripts/figs/make_slide_capilar_example.py --out-dir <dir>
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
from benchmark.eval import sliced  # noqa: E402
from benchmark.eval.instances import read_gt_polygons  # noqa: E402
from pipeline.viz import colorize_overlay  # noqa: E402

RESULTS = Path(r"D:\projeto_placentas_clayton\benchmark_results")
NATIVE = Path(r"D:\projeto_placentas_clayton\datasets_v4\test-ZAZA_val-TOSTADA\native\test")
BG, BLUE = "#FFFCF3", "#596AB6"
WIN_W, WIN_H = 1380, 1032


def _run(model: str) -> Path:
    import pandas as pd

    df = pd.read_csv(RESULTS / "results.csv")
    r = df[(df.task == "capilar") & (df.model == model) & (df.fold == "primary") & (df.norm == "none")]
    return RESULTS / r.run_id.iloc[0]


def curve(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"yolov8s-seg.pt": ("YOLOv8s", "#2FA4A0", 5), "yolo26s-seg.pt": ("YOLO26s", "#E8912D", 9), "yolo11s-seg.pt": ("YOLO11s", "#596AB6", 7)}
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=300)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for z, (m, (lab, col, ms)) in enumerate(names.items()):
        v = json.loads((_run(m) / "metrics_val.json").read_text(encoding="utf-8"))
        x = [a["conf"] for a in v["calibration"]]
        y = [a["f1"] for a in v["calibration"]]
        ax.plot(x, y, color=col, lw=2.2, zorder=2 + z, label=f"{lab}  (conf. {v['conf_star']:.2f}, F1 {max(y):.3f})")
        ax.plot([v["conf_star"]], [max(y)], "o", color=col, ms=ms, mec="white", mew=1.2, zorder=10 + z)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], frameon=False, fontsize=9, labelcolor=BLUE, loc="upper right")
    ax.set_xlabel("Confiança", color=BLUE, fontsize=10)
    ax.set_ylabel("F1 por ladrilho (validação)", color=BLUE, fontsize=10)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.3, 0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BLUE)
    ax.tick_params(colors=BLUE, labelsize=9)
    ax.grid(alpha=0.25, color=BLUE)
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.1)


def example(out: Path, device: str | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = _run("yolo11s-seg.pt")
    conf = json.loads((run / "metrics_val.json").read_text(encoding="utf-8"))["conf_star"]
    rows = json.loads((run / "perimage_test.json").read_text(encoding="utf-8"))["field"]["rows"]
    f1 = np.array([2 * r["tp"] / max(2 * r["tp"] + r["fp"] + r["fn"], 1) for r in rows])
    files = sorted(p for p in (NATIVE / "images").iterdir() if p.suffix.lower() in {".jpg", ".png"})
    k = int(np.argmin(np.abs(f1 - np.median(f1))))
    print(f"campo {k}: {files[k].name[:20]}  F1 do campo {f1[k]:.3f} (mediana {np.median(f1):.3f}), conf {conf}")
    bgr = cv2.imread(str(files[k]))
    h, w = bgr.shape[:2]
    pr = UltralyticsPredictor(run / "train" / "run" / "weights" / "best.pt", conf=conf, device=device)
    merged = sliced.merge(sliced.collect(lambda t: pr.predict(t, cls_id=0), bgr), conf)
    gt = read_gt_polygons(NATIVE / "labels" / f"{files[k].stem}.txt", h, w, 0)

    def union(insts) -> np.ndarray:
        u = np.zeros((h, w), np.uint8)
        for it in insts:
            u = np.maximum(u, it.full(h, w).astype(np.uint8))
        return u

    pu, gu = union(merged), union(gt)
    print(f"{len(merged)} previstos, {len(gt)} anotados no campo")
    # window with the most annotated area
    small = cv2.resize(gu.astype(np.float32), (w // 12, h // 12), interpolation=cv2.INTER_AREA)
    integ = cv2.integral(small)
    ww, wh = WIN_W // 12, WIN_H // 12
    best = max(((integ[y + wh, x + ww] - integ[y, x + ww] - integ[y + wh, x] + integ[y, x], x, y)
                for y in range(0, small.shape[0] - wh + 1, 4) for x in range(0, small.shape[1] - ww + 1, 4)))
    x0, y0 = best[1] * 12, best[2] * 12
    sl = (slice(y0, y0 + WIN_H), slice(x0, x0 + WIN_W))
    overlay = colorize_overlay(bgr, capilar_masks=[pu])  # pipeline colors (cyan fill, yellow outline)
    diff = np.zeros((h, w, 3), np.uint8)
    diff[(pu > 0) & (gu > 0)] = (0, 255, 255)
    diff[(pu > 0) & (gu == 0)] = (0, 200, 0)
    diff[(pu == 0) & (gu > 0)] = (0, 0, 220)
    iou_px = ((pu > 0) & (gu > 0)).sum() / max(((pu > 0) | (gu > 0)).sum(), 1)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5), dpi=200)
    fig.patch.set_facecolor(BG)
    for ax, im, t in zip(axs, (bgr, overlay, diff), ("Original", "Predição (YOLO11s)", "Diferença para a anotação")):
        ax.imshow(cv2.cvtColor(im[sl], cv2.COLOR_BGR2RGB))
        ax.set_title(t, color=BLUE, fontsize=14, fontweight="bold", pad=6)
        ax.axis("off")
    fig.text(0.5, 0.075, "amarelo: previsto e anotado   |   verde: só previsto   |   vermelho: só anotado",
             ha="center", color=BLUE, fontsize=11)
    fig.text(0.5, 0.025, f"Campo inteiro: IoU de pixels {iou_px:.2f}".replace(".", ",") + f"; {len(merged)} previstos e {len(gt)} anotados. "
             "Recorte de 1380×1032 px (≈ 147 × 110 µm).", ha="center", color=BLUE, fontsize=10)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.14, wspace=0.03)
    fig.savefig(out, facecolor=BG)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    curve(a.out_dir / "f1_confianca_capilar.png")
    example(a.out_dir / "exemplo_capilar_yolo11s.png", a.device)


if __name__ == "__main__":
    main()
