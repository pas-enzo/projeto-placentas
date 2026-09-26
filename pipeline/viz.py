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


def _mask_union(
    masks: list[np.ndarray] | None, h: int, w: int, dilate_px: int = 0
) -> np.ndarray:
    u = np.zeros((h, w), dtype=np.uint8)
    if not masks:
        return u
    for m in masks:
        mm = m
        if mm.shape[0] != h or mm.shape[1] != w:
            mm = cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        u = np.maximum(u, (mm > 0).astype(np.uint8))
    if dilate_px > 0 and u.any():
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        u = cv2.dilate(u, kernel, iterations=1)
    return u


def _blend(out: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> None:
    if not mask.any():
        return
    c = np.array(color_bgr, dtype=np.float32)
    out[mask > 0] = ((1.0 - alpha) * out[mask > 0] + alpha * c).astype(np.uint8)


def _draw_contours(
    out: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    if not mask.any():
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color_bgr, thickness, lineType=cv2.LINE_AA)


def colorize_overlay(
    bgr: np.ndarray,
    micro_masks: list[np.ndarray] | None = None,
    capilar_masks: list[np.ndarray] | None = None,
    *,
    micro_alpha: float = 0.80,
    capilar_alpha: float = 0.80,
) -> np.ndarray:
    """Draw prediction overlays.

    Micro = neon lime fill + yellow outline.
    Capillary = cyan fill + yellow outline (drawn on top so it stays visible
    inside microcotyledons; H&E is already red/pink).
    """
    out = bgr.copy()
    h, w = out.shape[:2]
    # Display-only dilation so thin capillary masks survive downscaling in side-by-side.
    g = _mask_union(micro_masks, h, w, dilate_px=0)
    r = _mask_union(capilar_masks, h, w, dilate_px=2)

    _blend(out, g, (0, 255, 0), micro_alpha)  # BGR neon green
    # Contour thickness scales with FOV so it survives side-by-side downscale.
    micro_t = max(3, int(round(min(h, w) * 0.0025)))
    cap_t = max(2, int(round(min(h, w) * 0.0015)))
    _draw_contours(out, g, (0, 255, 255), thickness=micro_t)

    _blend(out, r, (255, 255, 0), capilar_alpha)  # BGR pure cyan
    _draw_contours(out, r, (0, 255, 255), thickness=cap_t)
    return out


def save_side_by_side(bgr: np.ndarray, overlay: np.ndarray, path: Path, max_w: int = 2400) -> None:
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
