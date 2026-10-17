"""Per-instance geometry in native-FOV pixels (bbox, centroid, perimeter, ellipse axes, polygon)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import UM_PER_PX_NATIVE, PipelineConfig
from .types import CropMasks, ImageResult

GEOMETRY_FIELDS = [
    "area_px_native",
    "centroid_x_px",
    "centroid_y_px",
    "bbox_x_px",
    "bbox_y_px",
    "bbox_w_px",
    "bbox_h_px",
    "perimeter_um",
    "major_axis_um",
    "minor_axis_um",
    "circularity",
    "solidity",
]


def _bbox_crop(mask: np.ndarray) -> tuple[int, int, np.ndarray] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    crop = np.ascontiguousarray(mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] > 0)
    return int(cols[0]), int(rows[0]), crop.astype(np.uint8)


def instance_geometry(
    crop: np.ndarray, ox: int, oy: int, sx: float, sy: float, eps: float
) -> tuple[dict, list[list[float]]]:
    """Geometry of one binary mask crop located at (ox, oy) on a grid scaled by (sx, sy)
    to native pixels. Returns (metrics, polygons) with polygons as flat [x, y, ...] lists."""
    empty = {k: None for k in GEOMETRY_FIELDS}
    if crop.size == 0 or not crop.any():
        return empty, []

    h, w = crop.shape
    mom = cv2.moments(crop, binaryImage=True)
    geo = {
        "area_px_native": float(crop.sum()) * sx * sy,
        "centroid_x_px": (mom["m10"] / mom["m00"] + ox + 0.5) * sx,
        "centroid_y_px": (mom["m01"] / mom["m00"] + oy + 0.5) * sy,
        "bbox_x_px": ox * sx,
        "bbox_y_px": oy * sy,
        "bbox_w_px": w * sx,
        "bbox_h_px": h * sy,
    }

    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys: list[np.ndarray] = []
    for c in contours:
        a = cv2.approxPolyDP(c, eps, True)
        if len(a) < 3:
            a = c
        if len(a) < 3:
            continue
        pts = (a.reshape(-1, 2).astype(np.float64) + [ox + 0.5, oy + 0.5]) * [sx, sy]
        polys.append(pts.astype(np.float32))
    if not polys:
        # Degenerate (1-2 px wide) mask: fall back to its bbox as the polygon.
        x0, y0, x1, y1 = ox * sx, oy * sy, (ox + w) * sx, (oy + h) * sy
        polys = [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)]

    poly_area = sum(abs(cv2.contourArea(p)) for p in polys)
    perim = sum(cv2.arcLength(p, True) for p in polys)
    hull_area = cv2.contourArea(cv2.convexHull(np.concatenate(polys)))
    major = minor = None
    big = max(polys, key=cv2.contourArea)
    if len(big) >= 5:
        try:
            _, (d1, d2), _ = cv2.fitEllipse(big)
            major, minor = max(d1, d2) * UM_PER_PX_NATIVE, min(d1, d2) * UM_PER_PX_NATIVE
        except cv2.error:
            pass
    geo.update(
        {
            "perimeter_um": perim * UM_PER_PX_NATIVE,
            "major_axis_um": major,
            "minor_axis_um": minor,
            "circularity": min(1.0, 4 * math.pi * poly_area / perim**2) if perim else None,
            "solidity": min(1.0, poly_area / hull_area) if hull_area else None,
        }
    )
    geo = {k: (None if v is None else round(float(v), 3 if k.endswith("_um") or k in ("circularity", "solidity") else 1)) for k, v in geo.items()}
    return geo, [np.round(p, 1).reshape(-1).tolist() for p in polys]


def compute_geometry(res: ImageResult, cfg: PipelineConfig) -> None:
    """Fill `geometry` and `polygons` on each head of `res` (call before drop_masks)."""
    if res.micro is not None:
        canvas = int(cfg.micro_canvas)
        sx, sy = res.width / canvas, res.height / canvas
        geoms, polys = [], []
        for m in res.micro.masks:
            bc = _bbox_crop(m)
            g, p = instance_geometry(bc[2], bc[0], bc[1], sx, sy, eps=0.5) if bc else instance_geometry(np.zeros((0, 0), np.uint8), 0, 0, sx, sy, 0.5)
            geoms.append(g)
            polys.append(p)
        res.micro.geometry, res.micro.polygons = geoms, polys
    if res.capilar is not None:
        masks = res.capilar.masks
        geoms, polys = [], []
        for i in range(len(masks)):
            if isinstance(masks, CropMasks):
                y0, x0, crop = masks.crop(i)
            else:
                bc = _bbox_crop(masks[i])
                x0, y0, crop = bc if bc else (0, 0, np.zeros((0, 0), np.uint8))
            g, p = instance_geometry(crop, x0, y0, 1.0, 1.0, eps=1.0)
            geoms.append(g)
            polys.append(p)
        res.capilar.geometry, res.capilar.polygons = geoms, polys
