from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .config import PipelineConfig
from .types import InstanceResult


def _masks_from_sahi(result, w: int, h: int, cls_keep: int) -> tuple[list[np.ndarray], list[float]]:
    masks: list[np.ndarray] = []
    confs: list[float] = []
    for op in result.object_prediction_list:
        if int(op.category.id) != cls_keep:
            continue
        if op.mask is None:
            continue
        m = op.mask.bool_mask.astype(np.uint8)
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append((m > 0).astype(np.uint8))
        score = getattr(op, "score", None)
        confs.append(float(score.value) if score is not None else 1.0)
    return masks, confs


class CapilarModel:
    """YOLO11s-seg + SAHI sliced inference on native FOV."""

    def __init__(self, cfg: PipelineConfig | None = None, device: str | None = None):
        from sahi import AutoDetectionModel

        self.cfg = cfg or PipelineConfig()
        ckpt = self.cfg.capilar_ckpt_path
        if not ckpt.is_file():
            raise FileNotFoundError(f"Capillary checkpoint not found: {ckpt}")

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.det = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(ckpt),
            confidence_threshold=float(self.cfg.capilar_conf),
            device=self.device,
            image_size=int(self.cfg.capilar_imgsz),
            task="segment",
        )

    def predict_path(self, path: Path, conf: float | None = None) -> InstanceResult:
        from sahi.predict import get_sliced_prediction

        from .viz import read_bgr

        conf = float(self.cfg.capilar_conf if conf is None else conf)
        self.det.confidence_threshold = conf

        bgr = read_bgr(path)
        h, w = bgr.shape[:2]
        result = get_sliced_prediction(
            str(path),
            self.det,
            slice_height=int(self.cfg.capilar_slice_h),
            slice_width=int(self.cfg.capilar_slice_w),
            overlap_height_ratio=float(self.cfg.capilar_overlap),
            overlap_width_ratio=float(self.cfg.capilar_overlap),
            perform_standard_pred=False,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
            exclude_classes_by_id=[1],
            verbose=0,
        )
        masks, confs = _masks_from_sahi(result, w, h, cls_keep=int(self.cfg.capilar_cls_id))
        area_px = float(sum(int(m.sum()) for m in masks))
        area_um2 = area_px * float(self.cfg.capilar_area_um2_per_px2)
        return InstanceResult(
            count=len(masks),
            area_px=area_px,
            area_um2=area_um2,
            masks=masks,
            confidences=confs,
        )

    def predict_bgr(self, bgr: np.ndarray, conf: float | None = None) -> InstanceResult:
        """Write a temp PNG and run SAHI (API expects a path)."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), bgr)
            return self.predict_path(tmp_path, conf=conf)
        finally:
            tmp_path.unlink(missing_ok=True)
