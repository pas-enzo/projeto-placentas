"""Common sliced predictor: the SAME slicing and merging for every model family (fair capillary comparison).

Slices come from SAHI's `get_slice_bboxes`; predictions of neighbouring slices are merged with SAHI's own
`GreedyNMMPostprocess` (class_agnostic=False, as in `get_sliced_prediction`), so for the paper configuration
(1380x1032, overlap 0.2, GREEDYNMM / IOS 0.5) the result matches the published pipeline.

Design: the expensive part (network inference per slice) runs ONCE and is cached as `TilePredictions`;
`merge(conf)` then filters by confidence BEFORE merging (what SAHI does) for any number of thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .instances import Inst, from_full

SLICE_W, SLICE_H, OVERLAP = 1380, 1032, 0.2
MATCH_METRIC, MATCH_THRESHOLD = "IOS", 0.5


@dataclass
class TilePredictions:
    """Raw (low-confidence) instances of every slice, in slice coordinates."""
    shape: tuple[int, int]
    tiles: list[tuple[int, int, list[Inst]]] = field(default_factory=list)  # (x0, y0, instances)

    @property
    def n_instances(self) -> int:
        return sum(len(t[2]) for t in self.tiles)


def slice_boxes(h: int, w: int, slice_h: int = SLICE_H, slice_w: int = SLICE_W, overlap: float = OVERLAP):
    from sahi.slicing import get_slice_bboxes

    return get_slice_bboxes(h, w, slice_h, slice_w, False, overlap, overlap)


def collect(predict_fn, bgr: np.ndarray, slice_h: int = SLICE_H, slice_w: int = SLICE_W, overlap: float = OVERLAP,
            progress=None) -> TilePredictions:
    """Run `predict_fn(tile_bgr) -> list[Inst]` on every slice (instances in slice coordinates)."""
    h, w = bgr.shape[:2]
    boxes = slice_boxes(h, w, slice_h, slice_w, overlap)
    out = TilePredictions((h, w))
    for k, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        out.tiles.append((x0, y0, predict_fn(bgr[y0:y1, x0:x1])))
        if progress:
            progress(k, len(boxes))
    return out


def _polygons(it: Inst) -> list[list[float]]:
    cnts, _ = cv2.findContours(it.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [(c.reshape(-1, 2) + [it.x0, it.y0]).astype(float).ravel().tolist() for c in cnts if len(c) >= 3]


def merge(tp: TilePredictions, conf: float = 0.0, match_metric: str = MATCH_METRIC,
          match_threshold: float = MATCH_THRESHOLD) -> list[Inst]:
    """Filter by confidence, shift to the full image and merge across slices with SAHI's GreedyNMM."""
    from sahi.postprocess.combine import GreedyNMMPostprocess
    from sahi.prediction import ObjectPrediction

    h, w = tp.shape
    preds = []
    for x0, y0, insts in tp.tiles:
        for it in insts:
            if it.score < conf:
                continue
            polys = _polygons(it)
            if not polys:
                continue
            op = ObjectPrediction(segmentation=polys, category_id=0, category_name="obj", score=float(it.score),
                                  shift_amount=[x0, y0], full_shape=[h, w])
            preds.append(op.get_shifted_object_prediction())
    if not preds:
        return []
    merged = GreedyNMMPostprocess(match_threshold=match_threshold, match_metric=match_metric,
                                  class_agnostic=False)(preds)
    out = []
    for p in merged:
        it = from_full(p.mask.bool_mask, p.score.value)
        if it is not None:
            out.append(it)
    return out
