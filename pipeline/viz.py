from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return im


def stretch_to_square(bgr: np.ndarray, size: int = 640) -> np.ndarray:
    """Non-proportional resize matching Roboflow Stretch-to-640 export."""
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)


def masks_square_to_native(
    masks: list[np.ndarray],
    native_w: int,
    native_h: int,
    canvas: int = 640,
) -> list[np.ndarray]:
    """Map masks predicted on stretch-square canvas back to native FOV size."""
    out: list[np.ndarray] = []
    for m in masks:
        mm = m.astype(np.uint8)
        if mm.ndim != 2:
            continue
        if mm.shape[0] != canvas or mm.shape[1] != canvas:
            mm = cv2.resize(mm, (canvas, canvas), interpolation=cv2.INTER_NEAREST)
        resized = cv2.resize(mm, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
        out.append((resized > 0).astype(np.uint8))
    return out


def colorize_overlay(
    bgr: np.ndarray,
    micro_masks: list[np.ndarray] | None = None,
    capilar_masks: list[np.ndarray] | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Draw micro (green) and capillary (red) masks on a copy of the image."""
    out = bgr.copy()
    h, w = out.shape[:2]

    def _union(masks: list[np.ndarray] | None) -> np.ndarray:
        u = np.zeros((h, w), dtype=np.uint8)
        if not masks:
            return u
        for m in masks:
            mm = m
            if mm.shape[0] != h or mm.shape[1] != w:
                mm = cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            u = np.maximum(u, (mm > 0).astype(np.uint8))
        return u

    g = _union(micro_masks)
    r = _union(capilar_masks)
    if g.any():
        out[g > 0] = (
            (1.0 - alpha) * out[g > 0] + alpha * np.array([40, 200, 40], dtype=np.float32)
        ).astype(np.uint8)
    if r.any():
        out[r > 0] = (
            (1.0 - alpha) * out[r > 0] + alpha * np.array([40, 40, 220], dtype=np.float32)
        ).astype(np.uint8)
    both = np.logical_and(g > 0, r > 0)
    if both.any():
        out[both] = (
            (1.0 - alpha) * out[both] + alpha * np.array([40, 200, 220], dtype=np.float32)
        ).astype(np.uint8)
    return out


def save_side_by_side(bgr: np.ndarray, overlay: np.ndarray, path: Path, max_w: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = np.hstack([bgr, overlay])
    if vis.shape[1] > max_w:
        scale = max_w / vis.shape[1]
        vis = cv2.resize(
            vis,
            (max_w, int(vis.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(path), vis)
