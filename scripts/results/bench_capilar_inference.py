"""Benchmark YOLO11s-seg capillary inference (tile-level) and save JSON + update tables.

Protocol aligned with RF-DETR bench: warmup=1, repeats=3, batch=1, conf=conf*.
Unit = tile (243 validation tiles). Does NOT convert to FOV.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(r"D:\projeto_placentas_clayton\dev\projeto-placentas")
CKPT = ROOT / "v3_capilar_yolo11s" / "runs" / "capilar_yolo11s_tiled_v4" / "weights" / "best.pt"
TILE_DIR = Path(r"D:\projeto_placentas_clayton\dataset_v3.1_yolo11_tiled_3x3\valid\images")
OUT_DIR = ROOT / "v3_capilar_yolo11s" / "artifacts_v4" / "benchmarks"
RESULTS = ROOT / "figs" / "results"
CONF = 0.33
IMGSZ = 640
WARMUP = 1
REPEATS = 3


def main() -> None:
    assert CKPT.exists(), CKPT
    assert TILE_DIR.exists(), TILE_DIR
    imgs = sorted(TILE_DIR.glob("*.jpg"))
    assert len(imgs) == 243, f"expected 243 tiles, got {len(imgs)}"

    device = 0 if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        if torch.cuda.is_available()
        else None
    )
    print(f"device={device} gpu={gpu_name} vram_gb={vram_gb} n_tiles={len(imgs)}")

    model = YOLO(str(CKPT))

    def run_pass() -> None:
        for p in imgs:
            _ = model.predict(
                source=str(p),
                conf=CONF,
                imgsz=IMGSZ,
                device=device,
                verbose=False,
                half=False,
            )

    print("warmup...")
    for _ in range(WARMUP):
        run_pass()

    timings = []
    peak_mem_mb = None
    for i in range(REPEATS):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_pass()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        print(f"repeat {i+1}/{REPEATS}: {elapsed:.3f}s  ({len(imgs)/elapsed:.2f} tiles/s)")

    mean_elapsed = float(np.mean(timings))
    std_elapsed = float(np.std(timings))
    tps = len(imgs) / mean_elapsed if mean_elapsed > 0 else 0.0
    ms_per_tile = 1000.0 * mean_elapsed / len(imgs)

    payload = {
        "model": "yolo11s-seg",
        "run_name": "capilar_yolo11s_tiled_v4",
        "checkpoint": str(CKPT),
        "eval_unit": "tile",
        "conf": CONF,
        "imgsz": IMGSZ,
        "n_images": len(imgs),
        "repeats": REPEATS,
        "warmup": WARMUP,
        "batch_size": 1,
        "half": False,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_vram_gb": vram_gb,
        "elapsed_sec_mean": mean_elapsed,
        "elapsed_sec_std": std_elapsed,
        "images_per_sec": tps,
        "ms_per_unit": ms_per_tile,
        "peak_gpu_mem_mb": peak_mem_mb,
        "dataset_dir": str(TILE_DIR),
        "notes": "Tile-level only; do not multiply by 9 as FOV latency without stating estimate.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "inference_benchmark_capilar_yolo11s_tiled_v4.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("saved", out_json)

    # Update figs/results tables
    RESULTS.mkdir(parents=True, exist_ok=True)
    infer_csv = RESULTS / "table_inference.csv"
    # rewrite both rows with current micro + new cap
    micro_json = (
        ROOT
        / "v2_rfdetr_seg_large_opt_v1"
        / "artifacts"
        / "benchmarks"
        / "inference_benchmark_seg_large_r672_auto.json"
    )
    micro = json.loads(micro_json.read_text(encoding="utf-8"))
    micro_ms = 1000.0 * micro["elapsed_sec_mean"] / micro["n_images"]

    with infer_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "unit",
                "n_units",
                "conf",
                "warmup",
                "repeats",
                "elapsed_sec_mean",
                "elapsed_sec_std",
                "ms_per_unit",
                "units_per_sec",
                "peak_gpu_mem_mb",
                "gpu_name",
                "gpu_vram_gb",
                "status",
                "source",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "task": "microcotyledon",
                "unit": "FOV",
                "n_units": micro["n_images"],
                "conf": micro["conf"],
                "warmup": micro["warmup"],
                "repeats": micro["repeats"],
                "elapsed_sec_mean": micro["elapsed_sec_mean"],
                "elapsed_sec_std": micro["elapsed_sec_std"],
                "ms_per_unit": micro_ms,
                "units_per_sec": micro["images_per_sec"],
                "peak_gpu_mem_mb": micro["peak_gpu_mem_mb"],
                "gpu_name": "training_machine_16GB_edual_path",
                "gpu_vram_gb": 16,
                "status": "OK_from_prior_benchmark_json",
                "source": str(micro_json.relative_to(ROOT)),
            }
        )
        w.writerow(
            {
                "task": "capillary",
                "unit": "tile",
                "n_units": payload["n_images"],
                "conf": payload["conf"],
                "warmup": payload["warmup"],
                "repeats": payload["repeats"],
                "elapsed_sec_mean": payload["elapsed_sec_mean"],
                "elapsed_sec_std": payload["elapsed_sec_std"],
                "ms_per_unit": payload["ms_per_unit"],
                "units_per_sec": payload["images_per_sec"],
                "peak_gpu_mem_mb": payload["peak_gpu_mem_mb"],
                "gpu_name": payload["gpu_name"],
                "gpu_vram_gb": payload["gpu_vram_gb"],
                "status": "OK",
                "source": str(out_json.relative_to(ROOT)),
            }
        )
    print("updated", infer_csv)

    # Also copy a flat summary into figs/results
    (RESULTS / "inference_capilar_yolo11s_tiled_v4.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
