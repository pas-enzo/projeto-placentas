"""Generate Fig. 4: scale-bar calibration and task-specific area conversion factors."""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont

STRETCH_PATH = Path(
    r"D:\projeto_placentas_clayton\dataset_v2_ready_for_yolo\valid\images"
    r"\ROSILHA-M-D_006_jpg.rf.de71085683e9c7eb460be0f184fb31ac.jpg"
)
OUT_PATH = Path(
    r"D:\projeto_placentas_clayton\dev\projeto-placentas\figs"
    r"\fig4_scale_calibration_framed.jpg"
)

WHITE = (255, 255, 255)
OUTLINE = (40, 40, 40)


def crop_scale_bar(img_bgr: np.ndarray) -> np.ndarray:
    """Bottom-right crop of the embedded 50 µm bar on the 640×640 export."""
    h, w = img_bgr.shape[:2]
    # Include a bit of tissue context above/left of the bar box
    return img_bgr[h - 140 : h, w - 260 : w].copy()


def annotate_bar(crop_bgr: np.ndarray) -> np.ndarray:
    """Upscale crop and overlay a 72-px measurement callout on the black bar."""
    # Upscale for print clarity
    scale = 4
    big = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 36)
        font_sm = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    # Approximate bar location in the upscaled crop (measured visually on 640 export):
    # In the 140x260 crop of bottom-right, the white label box sits near the BR corner.
    # On 640 image, bar is inside white box; horizontal black line ~72 px long.
    # Crop coords relative to full 640: x in [380,640], y in [500,640]
    # Bar roughly centered in the white box horizontally.
    # We'll detect the black bar automatically.
    gray = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
    # Find dark horizontal segment (bar)
    # Threshold very dark pixels
    dark = gray < 40
    # Restrict to lower-right region of crop (where the label sits)
    hh, ww = gray.shape
    roi = dark[int(hh * 0.45) :, int(ww * 0.25) :]
    ys, xs = np.where(roi)
    if len(xs) == 0:
        # fallback approximate
        y_bar = int(hh * 0.72)
        x0, x1 = int(ww * 0.38), int(ww * 0.38 + 72 * scale)
    else:
        # take the row with most dark pixels
        row_counts = {}
        for y, x in zip(ys, xs):
            row_counts.setdefault(y, []).append(x)
        best_y = max(row_counts, key=lambda r: len(row_counts[r]))
        xs_row = sorted(row_counts[best_y])
        # contiguous span
        x0_roi, x1_roi = xs_row[0], xs_row[-1]
        # refine to longest run
        runs = []
        start = xs_row[0]
        prev = xs_row[0]
        for x in xs_row[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append((start, prev))
                start = prev = x
        runs.append((start, prev))
        x0_roi, x1_roi = max(runs, key=lambda t: t[1] - t[0])
        y_bar = best_y + int(hh * 0.45)
        x0 = x0_roi + int(ww * 0.25)
        x1 = x1_roi + int(ww * 0.25)

    bar_len_px_export = (x1 - x0) / scale
    print(f"detected bar length on export canvas: {bar_len_px_export:.1f} px (expect ~72)")

    # Measurement bracket above the bar
    y_br = y_bar - int(28 * scale / 4)
    color = (220, 30, 30)
    draw.line([(x0, y_br), (x1, y_br)], fill=color, width=4)
    # end ticks
    tick = 14
    draw.line([(x0, y_br - tick), (x0, y_br + tick)], fill=color, width=4)
    draw.line([(x1, y_br - tick), (x1, y_br + tick)], fill=color, width=4)

    label = "72 px  =  50 \u00b5m"
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx = (x0 + x1) // 2
    tx = cx - tw // 2
    ty = y_br - th - 18
    pad = 10
    draw.rectangle(
        [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
        fill=WHITE,
        outline=color,
        width=3,
    )
    draw.text((tx, ty), label, fill=color, font=font)

    # small footer note
    note = "640\u00d7640 Roboflow export (horizontal axis)"
    nb = draw.textbbox((0, 0), note, font=font_sm)
    nw, nh = nb[2] - nb[0], nb[3] - nb[1]
    nx, ny = 12, 12
    draw.rectangle([nx - 6, ny - 4, nx + nw + 6, ny + nh + 4], fill=WHITE)
    draw.text((nx, ny), note, fill=(40, 40, 40), font=font_sm)

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def render_equation_panel(
    title: str,
    formula_lines: list[str],
    factor: str,
    subtitle: str,
    width_px: int = 900,
    height_px: int = 520,
) -> np.ndarray:
    """Render a white card with title + formula using matplotlib mathtext."""
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Outer outline
    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.015),
            0.97,
            0.97,
            boxstyle="square,pad=0",
            linewidth=3.5,
            edgecolor="#282828",
            facecolor="white",
            transform=ax.transAxes,
            clip_on=False,
        )
    )

    ax.text(
        0.5,
        0.88,
        title,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        fontfamily="Segoe UI",
        color="#141414",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.76,
        subtitle,
        ha="center",
        va="center",
        fontsize=12,
        fontfamily="Segoe UI",
        color="#444444",
        transform=ax.transAxes,
    )

    # Main formula(s)
    y = 0.52
    for line in formula_lines:
        ax.text(
            0.5,
            y,
            line,
            ha="center",
            va="center",
            fontsize=15,
            color="#141414",
            transform=ax.transAxes,
        )
        y -= 0.16

    # Factor highlight box
    ax.add_patch(
        FancyBboxPatch(
            (0.12, 0.08),
            0.76,
            0.16,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#282828",
            facecolor="#F5F5F5",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.16,
        factor,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        fontfamily="Segoe UI",
        color="#141414",
        transform=ax.transAxes,
    )

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def make_labeled_card(img_bgr: np.ndarray, label: str, panel_w: int = 900) -> Image.Image:
    """Framed card with centered label chip (same language as Figs 2–3)."""
    h, w = img_bgr.shape[:2]
    scale = panel_w / w
    nh = int(round(h * scale))
    resized = cv2.resize(img_bgr, (panel_w, nh), interpolation=cv2.INTER_AREA)

    label_h = 70
    pad = 14
    outline = 8
    card_w = panel_w + 2 * pad
    card_h = label_h + nh + 2 * pad
    card = Image.new("RGB", (card_w, card_h), WHITE)
    draw = ImageDraw.Draw(card)
    try:
        font_b = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 28)
    except OSError:
        font_b = ImageFont.load_default()

    for t in range(outline):
        draw.rectangle([t, t, card_w - 1 - t, card_h - 1 - t], outline=OUTLINE)

    draw.rectangle([pad, pad, card_w - pad, pad + label_h - 8], fill=WHITE)
    bbox = draw.textbbox((0, 0), label, font=font_b)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (card_w - tw) // 2
    ty = pad + (label_h - 8 - th) // 2 - bbox[1]
    draw.text((tx, ty), label, fill=(20, 20, 20), font=font_b)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    card.paste(Image.fromarray(rgb), (pad, pad + label_h))
    return card


def main() -> None:
    stretch = cv2.imread(str(STRETCH_PATH))
    assert stretch is not None, STRETCH_PATH
    crop = crop_scale_bar(stretch)
    annotated = annotate_bar(crop)

    eq_micro = render_equation_panel(
        title="B  Microcotyledon conversion",
        subtitle="Anisotropic stretch 640\u00d7640  (full-field)",
        formula_lines=[
            r"$A_{\mu\mathrm{m}^{2}}^{(\mathrm{micro})} ="
            r" A_{\mathrm{px}}\times\left(\dfrac{50}{72}\right)^{2}"
            r"\times\left(\dfrac{3096}{4140}\right)$",
        ],
        factor="\u2248  0.3606 \u00b5m\u00b2 / px\u00b2",
        width_px=900,
        height_px=420,
    )

    eq_cap = render_equation_panel(
        title="C  Capillary conversion",
        subtitle="Isotropic tile space  (native geometry)",
        formula_lines=[
            r"$A_{\mu\mathrm{m}^{2}}^{(\mathrm{cap})} ="
            r" A_{\mathrm{px}}\times\left(\dfrac{50}{72}\right)^{2}"
            r"\times\left(\dfrac{640}{4140}\right)^{2}$",
        ],
        factor="\u2248  0.0115 \u00b5m\u00b2 / px\u00b2",
        width_px=900,
        height_px=420,
    )

    # Wider A so the annotated bar fills the top row; B|C below.
    card_a = make_labeled_card(annotated, "A  Scale bar on 640\u00d7640 export", panel_w=1400)

    def frame_only(img_bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    card_b = frame_only(eq_micro)
    card_c = frame_only(eq_cap)

    gap = 36
    outer = 48

    # Match B and C widths to half of A (minus gap)
    target_half = (card_a.size[0] - gap) // 2
    for name, card in (("b", card_b), ("c", card_c)):
        scale = target_half / card.size[0]
        new_size = (target_half, int(round(card.size[1] * scale)))
        if name == "b":
            card_b = card.resize(new_size, Image.Resampling.LANCZOS)
        else:
            card_c = card.resize(new_size, Image.Resampling.LANCZOS)

    total_w = outer * 2 + card_a.size[0]
    total_h = outer * 2 + card_a.size[1] + gap + card_b.size[1]
    fig = Image.new("RGB", (total_w, total_h), WHITE)

    fig.paste(card_a, (outer, outer))
    y2 = outer + card_a.size[1] + gap
    fig.paste(card_b, (outer, y2))
    fig.paste(card_c, (outer + card_b.size[0] + gap, y2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.save(OUT_PATH, quality=93, optimize=True)
    print(f"saved {OUT_PATH}  size={fig.size}")


if __name__ == "__main__":
    main()
