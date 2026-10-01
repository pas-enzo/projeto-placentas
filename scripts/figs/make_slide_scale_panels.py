"""Slide assets: Fig. 4 panels A/B/C as separate same-size images (Portuguese labels)."""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fig4_scale_calibration import (  # noqa: E402
    STRETCH_PATH,
    annotate_bar,
    crop_scale_bar,
    render_equation_panel,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "figs" / "slides"
W, H = 900, 560
FONT_SCALE = 1.9
OUTLINE = (40, 40, 40)


def bar_panel(annotated_bgr: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    card = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(card)
    m = round(0.015 * W)
    for t in range(4):
        draw.rectangle([m + t, m + t, W - 1 - m - t, H - 1 - m - t], outline=OUTLINE)
    for text, path, size, y, fill in (
        (title, r"C:\Windows\Fonts\segoeuib.ttf", 25, 0.12, (20, 20, 20)),
        (subtitle, r"C:\Windows\Fonts\segoeui.ttf", 16.7, 0.24, (68, 68, 68)),
    ):
        font = ImageFont.truetype(path, round(size * FONT_SCALE))
        tb = draw.textbbox((0, 0), text, font=font)
        draw.text(((W - (tb[2] - tb[0])) // 2, round(y * H) - (tb[3] - tb[1]) // 2 - tb[1]),
                  text, fill=fill, font=font)

    top, pad = round(0.32 * H), 30
    box_w, box_h = W - 2 * pad, H - top - pad
    h, w = annotated_bgr.shape[:2]
    s = min(box_w / w, box_h / h)
    img = cv2.resize(annotated_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    card.paste(rgb, ((W - rgb.width) // 2, top + (box_h - rgb.height) // 2))
    return cv2.cvtColor(np.array(card), cv2.COLOR_RGB2BGR)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stretch = cv2.imread(str(STRETCH_PATH))
    assert stretch is not None, STRETCH_PATH
    annotated = annotate_bar(crop_scale_bar(stretch), note=None)

    panels = {
        "escala_A_barra.jpg": bar_panel(annotated, "A  Barra de escala", "Imagem 640\u00d7640 (eixo horizontal)"),
        "escala_B_microcotiledones.jpg": render_equation_panel(
            title="B  Microcotil\u00e9dones",
            subtitle="Fator anisotr\u00f3pico (stretch 640\u00d7640)",
            formula_lines=[
                r"$A_{\mu\mathrm{m}^{2}}^{(\mathrm{micro})} ="
                r" A_{\mathrm{px}}\times\left(\dfrac{50}{72}\right)^{2}"
                r"\times\left(\dfrac{3096}{4140}\right)$",
            ],
            factor="\u2248  0,3606 \u00b5m\u00b2 / px\u00b2",
            width_px=W,
            height_px=H,
            full_bleed=True,
            font_scale=FONT_SCALE,
        ),
        "escala_C_capilares.jpg": render_equation_panel(
            title="C  Capilares",
            subtitle="Fator isotr\u00f3pico (geometria nativa)",
            formula_lines=[
                r"$A_{\mu\mathrm{m}^{2}}^{(\mathrm{cap})} ="
                r" A_{\mathrm{px}}\times\left(\dfrac{50}{72}\right)^{2}"
                r"\times\left(\dfrac{640}{4140}\right)^{2}$",
            ],
            factor="\u2248  0,0115 \u00b5m\u00b2 / px\u00b2",
            width_px=W,
            height_px=H,
            full_bleed=True,
            font_scale=FONT_SCALE,
        ),
    }
    for name, im in panels.items():
        path = OUT_DIR / name
        cv2.imwrite(str(path), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print("saved", path, im.shape[1], "x", im.shape[0])


if __name__ == "__main__":
    main()
