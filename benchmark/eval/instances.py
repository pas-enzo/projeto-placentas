"""Instance containers shared by every adapter and by the metrics (one format for all model families).

An instance is a boolean mask stored as a bounding-box crop plus its offset in the image, and a confidence.
Ground truth uses the same container with score = 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Inst:
    x0: int
    y0: int
    mask: np.ndarray  # bool crop, shape (h, w)
    score: float = 1.0

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x0 + self.mask.shape[1], self.y0 + self.mask.shape[0]

    def full(self, h: int, w: int) -> np.ndarray:
        out = np.zeros((h, w), dtype=bool)
        out[self.y0: self.y0 + self.mask.shape[0], self.x0: self.x0 + self.mask.shape[1]] = self.mask
        return out


def from_full(mask: np.ndarray, score: float = 1.0) -> Inst | None:
    """Tight crop of a full-size boolean mask; None if the mask is empty."""
    m = mask > 0
    rows = np.flatnonzero(m.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(m.any(axis=0))
    return Inst(int(cols[0]), int(rows[0]), m[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1].copy(), float(score))


def inter_area(a: Inst, b: Inst) -> int:
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((a.mask[y0 - a.y0: y1 - a.y0, x0 - a.x0: x1 - a.x0]
                & b.mask[y0 - b.y0: y1 - b.y0, x0 - b.x0: x1 - b.x0]).sum())


def iou(a: Inst, b: Inst) -> float:
    i = inter_area(a, b)
    return i / (a.area + b.area - i) if i else 0.0


def union_mask(insts: list[Inst], h: int, w: int) -> np.ndarray:
    u = np.zeros((h, w), dtype=bool)
    for it in insts:
        u[it.y0: it.y0 + it.mask.shape[0], it.x0: it.x0 + it.mask.shape[1]] |= it.mask
    return u


def read_gt_polygons(label_file: Path, h: int, w: int, cls_id: int = 0) -> list[Inst]:
    """YOLO-seg label file (normalized polygons) -> list of Inst rasterized at (h, w); other classes ignored."""
    out: list[Inst] = []
    if not Path(label_file).exists():
        return out
    for line in Path(label_file).read_text(encoding="utf-8").splitlines():
        tok = line.split()
        if len(tok) < 7 or int(float(tok[0])) != cls_id:
            continue
        pts = (np.array(tok[1:], dtype=float).reshape(-1, 2) * [w, h]).astype(np.int32)
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)
        it = from_full(m)
        if it is not None:
            out.append(it)
    return out
