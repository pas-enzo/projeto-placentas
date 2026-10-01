"""Slide assets: original field + GT masks (app colors), one image per class."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.viz import colorize_overlay  # noqa: E402

IMG_PATH = Path(
    r"D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size\valid\images"
    r"\ROSILHA-M-D_006_jpg.rf.4bf38cf5314617b13c7c11445fe40b04.jpg"
)
LBL_PATH = Path(
    r"D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size\valid\labels"
    r"\ROSILHA-M-D_006_jpg.rf.4bf38cf5314617b13c7c11445fe40b04.txt"
)
OUT_DIR = Path(__file__).resolve().parents[2] / "figs" / "slides"
OUT_W = 1600
CLS_CAPILAR, CLS_MICRO = 0, 1


def read_masks(h: int, w: int) -> dict[int, list[np.ndarray]]:
    masks: dict[int, list[np.ndarray]] = {CLS_CAPILAR: [], CLS_MICRO: []}
    for line in LBL_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        if cls not in masks:
            continue
        xy = np.array(parts[1:], dtype=np.float32).reshape(-1, 2) * [w, h]
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [xy.round().astype(np.int32)], 1)
        masks[cls].append(m)
    return masks


def save(im: np.ndarray, name: str) -> None:
    h, w = im.shape[:2]
    im = cv2.resize(im, (OUT_W, round(h * OUT_W / w)), interpolation=cv2.INTER_AREA)
    path = OUT_DIR / name
    cv2.imwrite(str(path), im, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    print("saved", path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(IMG_PATH))
    assert bgr is not None, IMG_PATH
    masks = read_masks(*bgr.shape[:2])
    print("micro:", len(masks[CLS_MICRO]), "capilar:", len(masks[CLS_CAPILAR]))
    save(bgr, "dataset_original.jpg")
    save(colorize_overlay(bgr, micro_masks=masks[CLS_MICRO]), "dataset_mask_microcotiledones.jpg")
    save(colorize_overlay(bgr, capilar_masks=masks[CLS_CAPILAR]), "dataset_mask_capilares.jpg")


if __name__ == "__main__":
    main()
