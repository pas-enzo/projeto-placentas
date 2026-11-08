"""RF-DETR Seg family (Nano / Small / Medium) through the common benchmark interface.

Models are named `rfdetr-seg-nano|small|medium` (spec.model). Data: the Roboflow-style COCO export of the view.
Training uses the library's own loop (`model.train`), so the augmentation is the library default for every size.
`supports_resume = True`: the library writes `checkpoint.pth` every epoch and `train(resume=...)` continues from it.

STATUS: written against rfdetr 1.6.2 but NOT yet exercised end to end; run the smoke config before trusting results.
Known open issue from the paper pipeline: RF-DETR did not reproduce the article's numbers inside the project pipeline
(cause unknown), so compare RF-DETR results with care.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from benchmark.core import RunSpec, view_dir, write_json
from benchmark.eval.instances import Inst, from_full

CLASSES = {"rfdetr-seg-nano": "RFDETRSegNano", "rfdetr-seg-small": "RFDETRSegSmall", "rfdetr-seg-medium": "RFDETRSegMedium"}
# The library downloads its pretrained file to the path it is given (the current directory by default): keep it on D:.
WEIGHTS_DIR = Path(r"D:\projeto_placentas_clayton\weights_rfdetr")


def _new_model(name: str, weights: str | None = None, resolution: int | None = None):
    import rfdetr

    if name not in CLASSES:
        raise KeyError(f"modelo RF-DETR desconhecido {name!r}; opções: {list(CLASSES)}")
    kw = {}
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    kw["pretrain_weights"] = weights or str(WEIGHTS_DIR / f"rf-detr-seg-{name.rsplit('-', 1)[1]}.pt")
    if resolution:
        kw["resolution"] = resolution
    return getattr(rfdetr, CLASSES[name])(**kw)


class RFDetrFamily:
    name = "rfdetr"
    supports_resume = True

    def out_dir(self, run_dir: Path) -> Path:
        return run_dir / "train"

    def best(self, run_dir: Path) -> Path:
        d = self.out_dir(run_dir)
        for n in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
            if (d / n).exists():
                return d / n
        raise FileNotFoundError(f"nenhum checkpoint 'best' em {d}")

    def can_resume(self, run_dir: Path) -> bool:
        return (self.out_dir(run_dir) / "checkpoint.pth").exists()

    def train(self, spec: RunSpec, run_dir: Path, resume: bool, batch: int) -> dict:
        import torch

        out = self.out_dir(run_dir)
        out.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.set_per_process_memory_fraction(0.92)
        resolution = spec.extra.get("resolution")
        model = _new_model(spec.model, resolution=resolution)
        write_json(out / "meta.json", {"name": spec.model, "resolution": resolution})
        args = dict(dataset_dir=str(view_dir(spec, coco=True)), epochs=spec.epochs, batch_size=batch,
                    grad_accum_steps=max(1, 8 // batch), output_dir=str(out), seed=spec.seed, num_workers=2,
                    early_stopping=True, early_stopping_patience=spec.patience, progress_bar=False, tensorboard=False)
        args.update({k: v for k, v in spec.extra.items() if k != "resolution"})
        if resume and self.can_resume(run_dir):
            args["resume"] = str(out / "checkpoint.pth")
        t0 = time.time()
        model.train(**args)
        write_json(out / "timing.json", {"train_seconds": time.time() - t0})
        return {"vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None}

    def model_info(self, run_dir: Path) -> dict:
        import torch

        ck = torch.load(self.best(run_dir), map_location="cpu", weights_only=False)
        sd = ck.get("model") or ck.get("ema_model") or ck
        n = sum(v.numel() for v in sd.values() if hasattr(v, "dtype") and v.dtype.is_floating_point)
        return {"params_millions": round(n / 1e6, 2), "weights_mb": round(self.best(run_dir).stat().st_size / 1e6, 1)}

    def make_predictor(self, spec: RunSpec, run_dir: Path, conf: float = 0.001):
        meta = json.loads((self.out_dir(run_dir) / "meta.json").read_text(encoding="utf-8"))
        return RFDetrPredictor(spec.model, self.best(run_dir), meta.get("resolution"), conf)

    def training_log(self, run_dir: Path) -> dict:
        f = self.out_dir(run_dir) / "timing.json"
        t = json.loads(f.read_text(encoding="utf-8"))["train_seconds"] if f.exists() else None
        log = self.out_dir(run_dir) / "metrics.csv"
        return {"train_seconds": t, "epochs_run": (sum(1 for _ in open(log, encoding="utf-8")) - 1) if log.exists() else None}


class RFDetrPredictor:
    family = "rfdetr"

    def __init__(self, name: str, weights: Path, resolution: int | None, conf: float = 0.001):
        self.model = _new_model(name, str(weights), resolution)
        self.conf = conf
        try:
            self.model.optimize_for_inference()
        except Exception:  # optional speed-up, not required for correctness
            pass

    def predict(self, bgr: np.ndarray, cls_id: int | None = None) -> list[Inst]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det = self.model.predict(rgb, threshold=self.conf)
        if det.mask is None or len(det) == 0:
            return []
        out = []
        for m, s in zip(det.mask, det.confidence):
            it = from_full(np.asarray(m, bool), float(s))
            if it is not None:
                out.append(it)
        return out
