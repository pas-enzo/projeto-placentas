from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import PipelineConfig
from .types import InstanceResult
from .viz import stretch_to_square


def _parse_masks_and_conf(det) -> tuple[list[np.ndarray], list[float]]:
    masks: list[np.ndarray] = []
    confs: list[float] = []

    pred_mask = getattr(det, "mask", None)
    pred_conf = getattr(det, "confidence", None)

    if pred_mask is not None:
        arr = np.asarray(pred_mask)
        if arr.ndim == 2:
            masks.append((arr > 0).astype(np.uint8))
        elif arr.ndim == 3:
            for m in arr:
                masks.append((m > 0).astype(np.uint8))

    if pred_conf is not None:
        confs = [float(x) for x in np.asarray(pred_conf).reshape(-1)]

    if len(confs) != len(masks):
        confs = [1.0] * len(masks)
    return masks, confs


class MicrocotyledonModel:
    """RF-DETR Seg Large — stretch FOV → 640, predict, report area in stretch px."""

    def __init__(self, cfg: PipelineConfig | None = None, device: str | None = None):
        from rfdetr import RFDETRSegLarge

        self.cfg = cfg or PipelineConfig()
        ckpt = self.cfg.micro_ckpt_path
        if not ckpt.is_file():
            raise FileNotFoundError(f"Microcotyledon checkpoint not found: {ckpt}")

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # Champion was trained at resolution=672; positional_encoding_size must match
        # (default SegLarge config is 504 / 42 and will size-mismatch the checkpoint).
        res = int(self.cfg.micro_resolution)
        self.model = RFDETRSegLarge(
            pretrain_weights=str(ckpt),
            resolution=res,
            positional_encoding_size=res // 12,
            num_classes=2,
        )

    def predict_bgr(self, bgr: np.ndarray, conf: float | None = None) -> InstanceResult:
        conf = float(self.cfg.micro_conf if conf is None else conf)
        canvas = int(self.cfg.micro_canvas)
        stretched = stretch_to_square(bgr, size=canvas)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), stretched)
            det = self.model.predict(str(tmp_path), threshold=conf)
        finally:
            tmp_path.unlink(missing_ok=True)

        masks, confs = _parse_masks_and_conf(det)
        keep = [(m, c) for m, c in zip(masks, confs) if c >= conf]
        masks = [m for m, _ in keep]
        confs = [c for _, c in keep]

        # Area factor is defined in stretch-canvas pixel space.
        fixed: list[np.ndarray] = []
        for m in masks:
            mm = m.astype(np.uint8)
            if mm.shape[0] != canvas or mm.shape[1] != canvas:
                mm = cv2.resize(mm, (canvas, canvas), interpolation=cv2.INTER_NEAREST)
            fixed.append((mm > 0).astype(np.uint8))

        area_px = float(sum(int(m.sum()) for m in fixed))
        area_um2 = area_px * float(self.cfg.micro_area_um2_per_px2)
        return InstanceResult(
            count=len(fixed),
            area_px=area_px,
            area_um2=area_um2,
            masks=fixed,
            confidences=confs,
        )

    def predict_path(self, path: Path, conf: float | None = None) -> InstanceResult:
        from .viz import read_bgr

        return self.predict_bgr(read_bgr(path), conf=conf)
