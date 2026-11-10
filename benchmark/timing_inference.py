"""Clean inference-time measurement for the slides / efficiency chapter.

The benchmark's own `ms_per_image` is NOT a usage time: it predicts at confidence 0.001 (needed for COCO AP), which decodes
hundreds of masks per image. Here each trained model is timed the way it would be deployed:
  * confidence = the value calibrated on validation (conf*, from metrics_val.json);
  * GPU warm-up discarded; wall-clock around the whole call (pre-processing, network, post-processing, masks to CPU,
    conversion to instances); torch.cuda.synchronize before and after;
  * FP32 and FP16 (`half=True`);
  * micro: the 30 test images of the 640x640 view, one by one;
  * capillary: the 30 native test fields, with the project's slicing + greedy merge (the same code as the field protocol).

    python -m benchmark.timing_inference                 # default models
    python -m benchmark.timing_inference --models micro:yolo26s-seg.pt capilar:yolo11s-seg.pt

Output: <results>/inference_timing.json and a printed table. Run it with nothing else using the GPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from benchmark.adapters.ultralytics_adapter import UltralyticsPredictor
from benchmark.core import RESULTS, read_json, write_json
from benchmark.eval import sliced

FOLD = Path(r"D:\projeto_placentas_clayton\datasets_v4\test-ZAZA_val-TOSTADA")
DEFAULT = ["micro:yolo26s-seg.pt", "micro:yolo26m-seg.pt", "micro:yolo11m-seg.pt",
           "capilar:yolo11s-seg.pt", "capilar:yolo26s-seg.pt"]
EXTS = {".jpg", ".jpeg", ".png"}


class HalfPredictor(UltralyticsPredictor):
    """Same predictor as the benchmark, plus the FP16 switch."""

    half = False

    def _run(self, model, bgr, device):
        return model.predict(bgr, conf=self.conf, imgsz=self.imgsz, iou=self.iou, max_det=self.max_det,
                             retina_masks=self.retina, device=device, half=self.half and device != "cpu", verbose=False)[0]


def _run_dir(results: Path, task: str, model: str) -> Path:
    import pandas as pd

    df = pd.read_csv(results / "results.csv")
    r = df[(df.task == task) & (df.model == model) & (df.fold == "primary") & (df.norm == "none")]
    if len(r) != 1:
        raise RuntimeError(f"esperava 1 execução para {task}/{model} (fold principal, sem norm.), achei {len(r)}")
    return results / r.run_id.iloc[0]


def _files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_one(task: str, model: str, results: Path) -> dict:
    import torch

    run = _run_dir(results, task, model)
    conf = read_json(run / "metrics_val.json")["conf_star"]
    weights = run / "train" / "run" / "weights" / "best.pt"
    if task == "micro":
        imgs = _files(FOLD / "micro_stretch640" / "test" / "images")
        work = lambda pr, bgr: pr.predict(bgr, cls_id=None)  # noqa: E731
        warm = 5
    else:
        imgs = _files(FOLD / "native" / "test" / "images")
        work = lambda pr, bgr: sliced.merge(sliced.collect(lambda t: pr.predict(t, cls_id=0), bgr), conf)  # noqa: E731
        warm = 1
    out = {"task": task, "model": model, "run": run.name, "conf": conf, "n_items": len(imgs), "unit": "imagem" if task == "micro" else "campo"}
    for half in (False, True):
        pr = HalfPredictor(weights, imgsz=640, conf=conf)
        pr.half = half
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for f in imgs[:warm]:  # warm-up, discarded
            work(pr, cv2.imread(str(f)))
        times = []
        for f in imgs:
            bgr = cv2.imread(str(f))  # disk read excluded from the timing
            _sync()
            t = time.perf_counter()
            work(pr, bgr)
            _sync()
            times.append(time.perf_counter() - t)
        t = np.array(times)
        out["fp16" if half else "fp32"] = {"mean_s": float(t.mean()), "sd_s": float(t.std(ddof=1)), "median_s": float(np.median(t)),
                                           "p95_s": float(np.percentile(t, 95)), "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20)}
        print(f"  {task} {model} {'FP16' if half else 'FP32'}: {t.mean() * 1000:.0f} ms (dp {t.std(ddof=1) * 1000:.0f}) "
              f"por {out['unit']}, VRAM {out['fp16' if half else 'fp32']['vram_peak_mb']} MB", flush=True)
        del pr
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT, help="task:model, ex.: micro:yolo26s-seg.pt")
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("sem GPU: a medição precisa da GPU do experimento")
    rows = [time_one(*m.split(":", 1), args.results) for m in args.models]
    write_json(args.results / "inference_timing.json",
               {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "measured": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows})
    print("salvo em", args.results / "inference_timing.json")


if __name__ == "__main__":
    main()
