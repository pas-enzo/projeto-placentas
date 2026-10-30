"""Training side of the Ultralytics family (YOLOv8 / YOLO11 / YOLO26 / YOLOE / YOLO12 ...).

`supports_resume = True`: an interrupted run continues from `last.pt` (model, optimizer, EMA and epoch are
restored by Ultralytics). Resumed runs are flagged `resumed: true` in run.json (equivalent, not bit-identical).
"""

from __future__ import annotations

import os
from pathlib import Path

from benchmark import recipes
from benchmark.core import RunSpec, view_dir

LOCAL_WEIGHT_DIRS = [Path(r"D:\projeto_placentas_clayton\weights_ultralytics"),
                     Path(__file__).resolve().parents[2] / "archive" / "weights_base"]


def resolve_weights(name: str) -> str:
    """Local pretrained file if we have it; otherwise the bare name (Ultralytics downloads it)."""
    p = Path(name)
    if p.exists():
        return str(p)
    for d in LOCAL_WEIGHT_DIRS:
        if (d / name).exists():
            return str(d / name)
    return name


class UltralyticsFamily:
    name = "ultralytics"
    supports_resume = True

    def best(self, run_dir: Path) -> Path:
        return run_dir / "train" / "run" / "weights" / "best.pt"

    def last(self, run_dir: Path) -> Path:
        return run_dir / "train" / "run" / "weights" / "last.pt"

    def can_resume(self, run_dir: Path) -> bool:
        return self.last(run_dir).exists()

    def train(self, spec: RunSpec, run_dir: Path, resume: bool, batch: int) -> dict:
        import torch
        from ultralytics import YOLO

        device = 0 if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # Windows lets the GPU spill into shared system memory (silently ~100x slower). A hard cap makes an
            # out-of-memory situation raise immediately so the runner can retry with a smaller batch.
            torch.cuda.set_per_process_memory_fraction(0.92)
        if resume and self.can_resume(run_dir):
            YOLO(str(self.last(run_dir))).train(resume=True)
        else:
            args = dict(data=str(view_dir(spec) / "data.yaml"), epochs=spec.epochs, patience=spec.patience,
                        imgsz=spec.imgsz, batch=batch, seed=spec.seed, project=str(run_dir / "train"), name="run",
                        exist_ok=True, workers=2, plots=False, device=device, verbose=False)
            args.update(recipes.get(spec.recipe))
            args.update(spec.extra)
            model = YOLO(resolve_weights(spec.model))
            stop_at = int(os.environ.get("BENCH_TEST_INTERRUPT_EPOCH", 0))  # test hook: simulate Ctrl+C after N epochs
            if stop_at:
                def _interrupt(trainer):
                    if trainer.epoch + 1 >= stop_at:
                        raise KeyboardInterrupt("interrupção simulada (teste)")

                model.add_callback("on_fit_epoch_end", _interrupt)
            model.train(**args)
        return {"vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None}

    def model_info(self, run_dir: Path) -> dict:
        from ultralytics import YOLO

        w = self.best(run_dir)
        m = YOLO(str(w)).model
        return {"params_millions": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
                "weights_mb": round(w.stat().st_size / 1e6, 1)}

    def make_predictor(self, spec: RunSpec, run_dir: Path, conf: float = 0.001):
        from benchmark.adapters.ultralytics_adapter import UltralyticsPredictor

        return UltralyticsPredictor(self.best(run_dir), imgsz=spec.imgsz, conf=conf)

    def training_log(self, run_dir: Path) -> dict:
        import pandas as pd

        f = run_dir / "train" / "run" / "results.csv"
        if not f.exists():
            return {}
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        return {"epochs_run": int(len(d)), "train_seconds": float(d["time"].iloc[-1]) if "time" in d else None}
