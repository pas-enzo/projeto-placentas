from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import PipelineConfig
from .types import InstanceResult
from .viz import stretch_to_square

MICRO_BACKENDS = ("rfdetr", "yolo")


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


def _stretch_result(
    masks: list[np.ndarray],
    confs: list[float],
    conf: float,
    cfg: PipelineConfig,
    model_name: str,
) -> InstanceResult:
    """Filter by conf and report area in stretch-canvas pixels (anisotropic factor)."""
    canvas = int(cfg.micro_canvas)
    fixed: list[np.ndarray] = []
    kept: list[float] = []
    for m, c in zip(masks, confs):
        if c < conf:
            continue
        mm = m.astype(np.uint8)
        if mm.shape[0] != canvas or mm.shape[1] != canvas:
            mm = cv2.resize(mm, (canvas, canvas), interpolation=cv2.INTER_NEAREST)
        fixed.append((mm > 0).astype(np.uint8))
        kept.append(c)

    area_px = float(sum(int(m.sum()) for m in fixed))
    return InstanceResult(
        count=len(fixed),
        area_px=area_px,
        area_um2=area_px * float(cfg.micro_area_um2_per_px2),
        masks=fixed,
        confidences=kept,
        model_name=model_name,
    )


class MicrocotyledonModel:
    """RF-DETR Seg Large — stretch FOV → 640, predict, report area in stretch px."""

    backend = "rfdetr"

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
        stretched = stretch_to_square(bgr, size=int(self.cfg.micro_canvas))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), stretched)
            det = self.model.predict(str(tmp_path), threshold=conf)
        finally:
            tmp_path.unlink(missing_ok=True)

        masks, confs = _parse_masks_and_conf(det)
        return _stretch_result(masks, confs, conf, self.cfg, "RF-DETR Seg Large")

    def predict_path(self, path: Path, conf: float | None = None) -> InstanceResult:
        from .viz import read_bgr

        return self.predict_bgr(read_bgr(path), conf=conf)


class MicrocotyledonYOLOModel:
    """YOLO11s-seg — same stretch-640 input and area factor as the RF-DETR head."""

    backend = "yolo"

    def __init__(self, cfg: PipelineConfig | None = None, device: str | None = None):
        from ultralytics import YOLO

        self.cfg = cfg or PipelineConfig()
        ckpt = self.cfg.micro_yolo_ckpt_path
        if not ckpt.is_file():
            raise FileNotFoundError(f"Microcotyledon YOLO checkpoint not found: {ckpt}")

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(str(ckpt))

    def predict_bgr(self, bgr: np.ndarray, conf: float | None = None) -> InstanceResult:
        conf = float(self.cfg.micro_yolo_conf if conf is None else conf)
        stretched = stretch_to_square(bgr, size=int(self.cfg.micro_canvas))

        r = self.model.predict(
            stretched,
            conf=conf,
            imgsz=int(self.cfg.micro_yolo_imgsz),
            retina_masks=True,
            device=self.device,
            verbose=False,
        )[0]
        if r.masks is None:
            masks, confs = [], []
        else:
            masks = [(m > 0.5).astype(np.uint8) for m in r.masks.data.cpu().numpy()]
            confs = [float(c) for c in r.boxes.conf.cpu().numpy()]
        return _stretch_result(masks, confs, conf, self.cfg, "YOLO11s-seg")

    def predict_path(self, path: Path, conf: float | None = None) -> InstanceResult:
        from .viz import read_bgr

        return self.predict_bgr(read_bgr(path), conf=conf)


def load_micro_model(
    cfg: PipelineConfig | None = None,
    device: str | None = None,
    backend: str | None = None,
) -> MicrocotyledonModel | MicrocotyledonYOLOModel:
    cfg = cfg or PipelineConfig()
    backend = (backend or cfg.micro_backend).lower().strip()
    if backend == "rfdetr":
        return MicrocotyledonModel(cfg, device=device)
    if backend == "yolo":
        return MicrocotyledonYOLOModel(cfg, device=device)
    raise ValueError(f"micro backend must be one of {MICRO_BACKENDS}, got {backend!r}")
