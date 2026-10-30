"""Ultralytics (YOLOv8 / YOLO11 / YOLO26 / YOLOE / YOLO12 ...) predictor producing the common `Inst` format.

`predict(bgr)` returns every instance above a very low confidence (default 0.001) at the resolution of the
image given, so that the benchmark can calibrate the confidence on validation and compute COCO AP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark.eval.instances import Inst, from_full


class UltralyticsPredictor:
    family = "ultralytics"

    def __init__(self, weights: str | Path, imgsz: int = 640, conf: float = 0.001, device: str | None = None,
                 max_det: int = 1000, iou: float = 0.7, retina_masks: bool = True):
        import torch
        from ultralytics import YOLO

        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.cpu_model = None  # created only if an image does not fit on the GPU
        self.imgsz, self.conf, self.max_det, self.iou, self.retina = imgsz, conf, max_det, iou, retina_masks
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def _run(self, model, bgr: np.ndarray, device: str):
        return model.predict(bgr, conf=self.conf, imgsz=self.imgsz, iou=self.iou, max_det=self.max_det,
                             retina_masks=self.retina, device=device, verbose=False)[0]

    def predict(self, bgr: np.ndarray, cls_id: int | None = None) -> list[Inst]:
        import torch

        try:
            r = self._run(self.model, bgr, self.device)
        except torch.OutOfMemoryError:
            # conf 0.001 can yield hundreds of full-resolution masks (> 1 GiB) on some models. Same settings on the CPU
            # keeps the result identical; only this image is slower.
            torch.cuda.empty_cache()
            if self.cpu_model is None:
                from ultralytics import YOLO

                self.cpu_model = YOLO(self.weights)
            r = self._run(self.cpu_model, bgr, "cpu")
        if r.masks is None:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        h, w = bgr.shape[:2]
        out: list[Inst] = []
        for m, s, c in zip(masks, scores, classes):
            if cls_id is not None and c != cls_id:
                continue
            if m.shape != (h, w):  # without retina_masks the mask comes at the network resolution
                import cv2

                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            it = from_full(m, float(s))
            if it is not None:
                out.append(it)
        return out
