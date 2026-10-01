"""Slide assets: SAHI 4×4 slicing grid over a native capillary field."""
from pathlib import Path

import cv2
import numpy as np
from sahi.slicing import get_slice_bboxes

IMG_PATH = Path(
    r"D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size\valid\images"
    r"\ROSILHA-M-D_006_jpg.rf.4bf38cf5314617b13c7c11445fe40b04.jpg"
)
OUT_DIR = Path(__file__).resolve().parents[2] / "figs" / "slides"
OUT_W = 1600
SLICE_W, SLICE_H, OVERLAP = 1380, 1032, 0.2

RED, BLUE, YELLOW, DARK = (0, 0, 220), (220, 120, 0), (0, 215, 255), (40, 40, 40)


def save(im: np.ndarray, name: str) -> None:
    h, w = im.shape[:2]
    im = cv2.resize(im, (OUT_W, round(h * OUT_W / w)), interpolation=cv2.INTER_AREA)
    path = OUT_DIR / name
    cv2.imwrite(str(path), im, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    print("saved", path)


def tint(im: np.ndarray, mask: np.ndarray, color: tuple, alpha: float) -> np.ndarray:
    out = im.copy()
    layer = np.zeros_like(im)
    layer[:] = color
    out[mask] = cv2.addWeighted(im, 1 - alpha, layer, alpha, 0)[mask]
    return out


def faded(bgr: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(bgr, 0.45, np.full_like(bgr, 255), 0.55, 0)


def frame(im: np.ndarray, box, color, thick: int) -> None:
    x0, y0, x1, y1 = box
    h = thick // 2
    cv2.rectangle(im, (x0 + h, y0 + h), (x1 - h - 1, y1 - h - 1), color, thick)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(IMG_PATH))
    assert bgr is not None, IMG_PATH
    h, w = bgr.shape[:2]
    boxes = get_slice_bboxes(
        image_height=h,
        image_width=w,
        slice_height=SLICE_H,
        slice_width=SLICE_W,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
    )
    print(f"field {w}x{h}, {len(boxes)} windows")
    for b in boxes:
        print("  ", b)
    thick = max(8, w // 450)

    coverage = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in boxes:
        coverage[y0:y1, x0:x1] += 1
    grid = tint(bgr, coverage > 1, YELLOW, 0.35)
    for b in boxes:
        frame(grid, b, DARK, thick // 2)
    save(grid, "sahi_grade_4x4.jpg")

    single = faded(bgr)
    x0, y0, x1, y1 = boxes[0]
    single[y0:y1, x0:x1] = bgr[y0:y1, x0:x1]
    frame(single, boxes[0], RED, thick)
    save(single, "sahi_janela_unica.jpg")

    a, b = boxes[0], boxes[1]
    pair = faded(bgr)
    for x0, y0, x1, y1 in (a, b):
        pair[y0:y1, x0:x1] = bgr[y0:y1, x0:x1]
    ov = np.zeros((h, w), bool)
    ov[max(a[1], b[1]) : min(a[3], b[3]), max(a[0], b[0]) : min(a[2], b[2])] = True
    pair = tint(pair, ov, YELLOW, 0.40)
    frame(pair, a, RED, thick)
    frame(pair, b, BLUE, thick)
    save(pair, "sahi_sobreposicao.jpg")


if __name__ == "__main__":
    main()
