"""Slide assets: true 20% SAHI overlap on tissue + per-axis window layout schematic."""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from sahi.slicing import get_slice_bboxes  # noqa: E402

IMG_PATH = Path(
    r"D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size\valid\images"
    r"\ROSILHA-M-D_006_jpg.rf.4bf38cf5314617b13c7c11445fe40b04.jpg"
)
OUT_DIR = Path(__file__).resolve().parents[2] / "figs" / "slides"
OUT_W = 1600
SLICE_W, SLICE_H, OVERLAP = 1380, 1032, 0.2

WIN_BGR = [(0, 0, 220), (220, 110, 0), (0, 140, 255), (160, 0, 160)]
WIN_HEX = ["#dc0000", "#006edc", "#ff8c00", "#a000a0"]
YELLOW_BGR = (0, 215, 255)


def tissue_three_windows(bgr: np.ndarray, boxes: list) -> None:
    h, w = bgr.shape[:2]
    row = [b for b in boxes if b[1] == 0][:3]
    out = cv2.addWeighted(bgr, 0.45, np.full_like(bgr, 255), 0.55, 0)
    for x0, y0, x1, y1 in row:
        out[y0:y1, x0:x1] = bgr[y0:y1, x0:x1]
    layer = out.copy()
    for a, b in zip(row, row[1:]):
        layer[0 : a[3], b[0] : a[2]] = YELLOW_BGR
    ov = np.zeros((h, w), bool)
    for a, b in zip(row, row[1:]):
        ov[0 : a[3], b[0] : a[2]] = True
    out[ov] = cv2.addWeighted(out, 0.6, layer, 0.4, 0)[ov]
    thick = max(10, w // 400)
    for (x0, y0, x1, y1), c in zip(row, WIN_BGR):
        cv2.rectangle(out, (x0 + thick // 2, y0 + thick // 2), (x1 - thick // 2, y1 - thick // 2), c, thick)
    for i, (x0, _, _, y1) in enumerate(row):
        cv2.putText(out, str(i + 1), (x0 + 50, y1 - 60), cv2.FONT_HERSHEY_DUPLEX, 5, WIN_BGR[i], 12, cv2.LINE_AA)
    out = cv2.resize(out, (OUT_W, round(h * OUT_W / w)), interpolation=cv2.INTER_AREA)
    path = OUT_DIR / "sahi_sobreposicao_20pct.jpg"
    cv2.imwrite(str(path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    print("saved", path)


def axis_panel(ax, spans: list[tuple[int, int]], length: int, size: int, title: str) -> None:
    n = len(spans)
    for i, (a, b) in enumerate(spans):
        y = n - i
        ax.add_patch(Rectangle((a, y - 0.32), b - a, 0.64, color=WIN_HEX[i], alpha=0.9, zorder=2))
        ax.text(a + 40, y, f"Janela {i + 1}", va="center", ha="left", color="white", fontsize=13, fontweight="bold", zorder=3)
    for i, ((a0, a1), (b0, _)) in enumerate(zip(spans, spans[1:])):
        ov = a1 - b0
        ax.add_patch(Rectangle((b0, 0.5), ov, n, color="#ffd700", alpha=0.35, lw=0, zorder=0))
        pct = round(100 * ov / size)
        edge = i == n - 2 and pct != round(100 * OVERLAP)
        label = f"{ov} px\n({pct}%)" + ("\nborda" if edge else "")
        ax.text(b0 + ov / 2, n + 0.62, label, ha="center", va="bottom", fontsize=12,
                color="#a00000" if edge else "#7a6000", fontweight="bold")
    ax.add_patch(Rectangle((0, -0.25), length, 0.4, color="#999999"))
    ax.text(length / 2, -0.05, f"Campo: {length} px", ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")
    ax.set_xlim(-60, length + 60)
    ax.set_ylim(-0.5, n + 1.6)
    ax.set_yticks([])
    ax.set_title(title, fontsize=15, loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)


def schematic(boxes: list, w: int, h: int) -> None:
    xs = sorted({(b[0], b[2]) for b in boxes})
    ys = sorted({(b[1], b[3]) for b in boxes})
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.6), dpi=200)
    axis_panel(ax1, xs, w, SLICE_W, f"Eixo horizontal: janelas de {SLICE_W} px, passo {xs[1][0]} px")
    axis_panel(ax2, ys, h, SLICE_H, f"Eixo vertical: janelas de {SLICE_H} px, passo {ys[1][0]} px")
    ax2.set_xlabel("Posição no campo (px)", fontsize=12)
    fig.tight_layout(h_pad=2.0)
    path = OUT_DIR / "sahi_esquema_eixos.jpg"
    fig.savefig(path, pil_kwargs={"quality": 93})
    print("saved", path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(IMG_PATH))
    assert bgr is not None, IMG_PATH
    h, w = bgr.shape[:2]
    boxes = get_slice_bboxes(
        image_height=h, image_width=w, slice_height=SLICE_H, slice_width=SLICE_W,
        overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
    )
    tissue_three_windows(bgr, boxes)
    schematic(boxes, w, h)


if __name__ == "__main__":
    main()
