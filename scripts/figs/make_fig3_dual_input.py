"""Generate Fig. 3: dual input strategies (native / stretch / 3x3 tiles)."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

STEM = "ROSILHA-M-D_006_jpg.rf.4bf38cf5314617b13c7c11445fe40b04"
NATIVE_PATH = Path(
    rf"D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size\valid\images\{STEM}.jpg"
)
STRETCH_PATH = Path(
    r"D:\projeto_placentas_clayton\dataset_v2_ready_for_yolo\valid\images"
    r"\ROSILHA-M-D_006_jpg.rf.de71085683e9c7eb460be0f184fb31ac.jpg"
)
TILE_DIR = Path(r"D:\projeto_placentas_clayton\dataset_v3.1_yolo11_tiled_3x3\valid\images")
OUT_PATH = Path(
    r"D:\projeto_placentas_clayton\dev\projeto-placentas\figs"
    r"\fig3_dual_input_strategy_framed.jpg"
)

PANEL_W = 900
GAP = 36
OUTER = 48
OUTLINE = 8
LABEL_H = 70
CARD_PAD = 14
WHITE = (255, 255, 255)
BG = WHITE
CARD_BG = WHITE
OUTLINE_COLOR = (40, 40, 40)

LABELS = [
    "A  Native 4140\u00d73096",
    "B  Stretch 640\u00d7640 (micro)",
    "C  3\u00d73 tiles 1380\u00d71032 (cap)",
]


def fit_panel(img: np.ndarray, target_w: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = target_w / iw
    th = int(round(ih * scale))
    return cv2.resize(img, (target_w, th), interpolation=cv2.INTER_AREA)


def make_card(img_bgr: np.ndarray, label: str) -> Image.Image:
    card_inner_h = img_bgr.shape[0]
    canvas = np.full((card_inner_h, PANEL_W, 3), 255, dtype=np.uint8)
    y0 = (card_inner_h - img_bgr.shape[0]) // 2
    canvas[y0 : y0 + img_bgr.shape[0], :] = img_bgr

    card_w = PANEL_W + 2 * CARD_PAD
    card_h = LABEL_H + card_inner_h + 2 * CARD_PAD
    card = Image.new("RGB", (card_w, card_h), CARD_BG)
    draw = ImageDraw.Draw(card)
    try:
        font_b = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 28)
    except OSError:
        font_b = ImageFont.load_default()

    for t in range(OUTLINE):
        draw.rectangle([t, t, card_w - 1 - t, card_h - 1 - t], outline=OUTLINE_COLOR)

    draw.rectangle(
        [CARD_PAD, CARD_PAD, card_w - CARD_PAD, CARD_PAD + LABEL_H - 8],
        fill=WHITE,
    )
    bbox = draw.textbbox((0, 0), label, font=font_b)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (card_w - tw) // 2
    ty = CARD_PAD + (LABEL_H - 8 - th) // 2 - bbox[1]
    draw.text((tx, ty), label, fill=(20, 20, 20), font=font_b)

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    card.paste(Image.fromarray(img_rgb), (CARD_PAD, CARD_PAD + LABEL_H))
    return card


def main() -> None:
    native = cv2.imread(str(NATIVE_PATH))
    assert native is not None, NATIVE_PATH
    h, w = native.shape[:2]
    print(f"native: {w}x{h}")

    stretch = cv2.imread(str(STRETCH_PATH))
    if stretch is None:
        stretch = cv2.resize(native, (640, 640), interpolation=cv2.INTER_LINEAR)
    print(f"stretch: {stretch.shape[1]}x{stretch.shape[0]}")

    tiles = []
    for r in range(3):
        row = []
        for c in range(3):
            tp = TILE_DIR / f"{STEM}_r{r}c{c}.jpg"
            t = cv2.imread(str(tp))
            assert t is not None, tp
            row.append(t)
        tiles.append(row)
    th, tw = tiles[0][0].shape[:2]
    seam = max(6, tw // 80)
    grid_h = 3 * th + 2 * seam
    grid_w = 3 * tw + 2 * seam
    panel_c = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for r in range(3):
        for c in range(3):
            y0 = r * (th + seam)
            x0 = c * (tw + seam)
            panel_c[y0 : y0 + th, x0 : x0 + tw] = tiles[r][c]
    print(f"tiled grid: {grid_w}x{grid_h} (tile {tw}x{th})")

    fitted = [fit_panel(p, PANEL_W) for p in (native, stretch, panel_c)]
    cards = [make_card(fitted[i], LABELS[i]) for i in range(3)]
    card_ws = [c.size[0] for c in cards]
    row_h = max(c.size[1] for c in cards)
    total_w = OUTER * 2 + sum(card_ws) + GAP * 2
    total_h = OUTER * 2 + row_h

    fig = Image.new("RGB", (total_w, total_h), BG)
    x = OUTER
    for c in cards:
        y = OUTER + (row_h - c.size[1]) // 2
        fig.paste(c, (x, y))
        x += c.size[0] + GAP

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.save(OUT_PATH, quality=92, optimize=True)
    print(f"saved {OUT_PATH}  size={fig.size}")


if __name__ == "__main__":
    main()
