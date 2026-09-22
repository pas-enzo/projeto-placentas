"""Generate Fig. 2: manual annotation grid (framed, white backdrop)."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG_PATH = Path(
    r"D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size\valid\images"
    r"\ROSILHA-M-D_006_jpg.rf.4bf38cf5314617b13c7c11445fe40b04.jpg"
)
LBL_PATH = Path(
    r"D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size\valid\labels"
    r"\ROSILHA-M-D_006_jpg.rf.4bf38cf5314617b13c7c11445fe40b04.txt"
)
OUT_PATH = Path(
    r"D:\projeto_placentas_clayton\dev\projeto-placentas\figs"
    r"\fig2_annotations_grid_framed.jpg"
)

WHITE = (255, 255, 255)
GAP = 40
MARGIN = 52
BORDER = 8
BORDER_COLOR = (40, 40, 40)


def main() -> None:
    bgr = cv2.imread(str(IMG_PATH))
    assert bgr is not None, IMG_PATH
    h, w = bgr.shape[:2]
    polys: dict[int, list] = {0: [], 1: []}
    for line in LBL_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
        pts = np.array(
            [[int(coords[i] * w), int(coords[i + 1] * h)] for i in range(0, len(coords), 2)],
            np.int32,
        )
        if cls in polys:
            polys[cls].append(pts)

    def draw_overlay(base, pts_list, color, alpha=0.4):
        overlay = base.copy()
        for pts in pts_list:
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(overlay, [pts], True, color, 2, cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    def resize_max(im, max_w=1480):
        hh, ww = im.shape[:2]
        if ww > max_w:
            im = cv2.resize(im, (max_w, int(hh * max_w / ww)), interpolation=cv2.INTER_AREA)
        return im

    font = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 40)

    def add_label(bgr_im, text):
        rgb = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = 18, 14
        box_w = tw + 2 * pad_x
        box_h = th + 2 * pad_y
        x0, y0 = 14, 14
        draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=WHITE)
        draw.text(
            (x0 + box_w / 2, y0 + box_h / 2),
            text,
            fill=(25, 25, 25),
            font=font,
            anchor="mm",
        )
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    panels = [
        ("A  Original", bgr.copy()),
        ("B  Microcotyledon GT", draw_overlay(bgr, polys[1], (255, 180, 0), 0.38)),
        ("C  Capillary GT", draw_overlay(bgr, polys[0], (0, 0, 255), 0.45)),
        (
            "D  Both classes",
            draw_overlay(draw_overlay(bgr, polys[1], (255, 180, 0), 0.32), polys[0], (0, 0, 255), 0.45),
        ),
    ]

    framed = [add_label(resize_max(im), txt) for txt, im in panels]
    max_h = max(p.shape[0] for p in framed)
    max_w = max(p.shape[1] for p in framed)

    def place(im):
        out = np.full((max_h, max_w, 3), 255, np.uint8)
        out[: im.shape[0], : im.shape[1]] = im
        cv2.rectangle(out, (0, 0), (max_w - 1, max_h - 1), BORDER_COLOR, BORDER)
        return out

    cards = [place(p) for p in framed]
    ch, cw = cards[0].shape[:2]
    canvas_h = MARGIN * 2 + ch * 2 + GAP
    canvas_w = MARGIN * 2 + cw * 2 + GAP
    canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    coords = [
        (MARGIN, MARGIN),
        (MARGIN, MARGIN + cw + GAP),
        (MARGIN + ch + GAP, MARGIN),
        (MARGIN + ch + GAP, MARGIN + cw + GAP),
    ]
    for (y, x), card in zip(coords, cards):
        canvas[y : y + ch, x : x + cw] = card

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
