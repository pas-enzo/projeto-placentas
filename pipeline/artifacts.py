"""Downloadable artifacts of a run, shared by the CLI, the single-image tab and the batch tab.

Layout of a run folder (single image = batch of one):

    run_info.json            config, versions, git sha, date (reproducibility)
    results.csv              one row per image
    instances.csv            one row per instance (area, confidence, geometry in native px)
    annotations.coco.json    polygons (native px) + area_um2 + score, COCO format
    overlays/<stem>_overlay.jpg   full-resolution prediction overlay
    lado_a_lado/<stem>.jpg        original | prediction (downscaled)
    masks/<stem>_micro.png, <stem>_capilar.png   binary union masks, native size
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .config import (
    NATIVE_H,
    NATIVE_W,
    SCALE_BAR_PX_NATIVE,
    SCALE_BAR_UM,
    PipelineConfig,
)
from .geometry import GEOMETRY_FIELDS
from .types import CropMasks, ImageResult
from .viz import save_side_by_side

COCO_CATEGORIES = [
    {"id": 1, "name": "microcotiledone"},
    {"id": 2, "name": "Capilar"},
]


def unique_stem(stem: str, used: set[str]) -> str:
    """Same file name in different folders must not overwrite another image's artifacts."""
    name, k = stem, 1
    while name in used:
        k += 1
        name = f"{stem}_{k}"
    used.add(name)
    return name


def new_run_dir(root: Path, prefix: str) -> Path:
    stamp = time.strftime(f"{prefix}_%Y%m%d_%H%M%S")
    out, k = root / stamp, 1
    while out.exists():
        k += 1
        out = root / f"{stamp}_{k}"
    out.mkdir(parents=True)
    return out


def _union_mask_native(res: ImageResult, key: str, cfg: PipelineConfig) -> np.ndarray | None:
    inst = getattr(res, key)
    if inst is None:
        return None
    if key == "micro":
        canvas = int(cfg.micro_canvas)
        u = np.zeros((canvas, canvas), dtype=np.uint8)
        for m in inst.masks:
            u |= (m > 0).astype(np.uint8)
        # Nearest-neighbour is pointwise, so resizing the union equals the union of resizes.
        return cv2.resize(u, (res.width, res.height), interpolation=cv2.INTER_NEAREST)
    u = np.zeros((res.height, res.width), dtype=np.uint8)
    masks = inst.masks
    for i in range(len(masks)):
        if isinstance(masks, CropMasks):
            y0, x0, crop = masks.crop(i)
            u[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] |= crop
        else:
            u |= (masks[i] > 0).astype(np.uint8)
    return u


def save_image_artifacts(
    res: ImageResult,
    bgr: np.ndarray,
    overlay: np.ndarray | None,
    run_dir: Path,
    stem: str,
    cfg: PipelineConfig,
    *,
    save_overlays: bool = True,
    save_masks: bool = True,
) -> Path | None:
    """Write the image-level files; returns the side-by-side path (for galleries) or None.

    Must run before `res.drop_masks()` (masks feed the PNGs)."""
    res.extras["stem"] = stem
    side = None
    if save_overlays and overlay is not None:
        full = run_dir / "overlays" / f"{stem}_overlay.jpg"
        full.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(full), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
        side = run_dir / "lado_a_lado" / f"{stem}.jpg"
        save_side_by_side(bgr, overlay, side)
        res.overlay_path = Path("overlays") / full.name
    if save_masks:
        (run_dir / "masks").mkdir(parents=True, exist_ok=True)
        for key in ("micro", "capilar"):
            u = _union_mask_native(res, key, cfg)
            if u is not None:
                cv2.imwrite(
                    str(run_dir / "masks" / f"{stem}_{key}.png"),
                    u * 255,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )
    return side


# ---------------------------------------------------------------- run-level files


def write_results_csv(results: Sequence[ImageResult], csv_path: Path) -> None:
    rows = [r.to_row() for r in results]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    # Union keys if modes differ (shouldn't, but safe).
    for row in rows[1:]:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_instances_csv(results: Sequence[ImageResult], csv_path: Path) -> None:
    """One row per instance; `index` is the same number shown in the hover tooltip (#n)."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "head", "index", "area_um2", "confidence", *GEOMETRY_FIELDS])
        for r in results:
            for head, inst in (("micro", r.micro), ("capilar", r.capilar)):
                if inst is None:
                    continue
                for i, (a, c) in enumerate(zip(inst.areas_um2, inst.confidences)):
                    g = inst.geometry[i] if i < len(inst.geometry) else {}
                    writer.writerow(
                        [r.image_path.name, head, i + 1, round(a, 4), round(c, 4)]
                        + ["" if g.get(k) is None else g[k] for k in GEOMETRY_FIELDS]
                    )


def write_coco_json(results: Sequence[ImageResult], path: Path) -> None:
    """COCO instances: polygons in native px, area in native px, plus area_um2 and score."""
    images, anns = [], []
    ann_id = 1
    cat = {"micro": 1, "capilar": 2}
    for image_id, r in enumerate(results, start=1):
        images.append(
            {"id": image_id, "file_name": r.image_path.name, "width": r.width, "height": r.height}
        )
        for head, inst in (("micro", r.micro), ("capilar", r.capilar)):
            if inst is None:
                continue
            for i, polys in enumerate(inst.polygons):
                g = inst.geometry[i]
                if not polys or g.get("bbox_x_px") is None:
                    continue
                anns.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cat[head],
                        "segmentation": polys,
                        "bbox": [g["bbox_x_px"], g["bbox_y_px"], g["bbox_w_px"], g["bbox_h_px"]],
                        "area": g["area_px_native"],
                        "iscrowd": 0,
                        "score": round(inst.confidences[i], 4),
                        "area_um2": round(inst.areas_um2[i], 4),
                        "instance_index": i + 1,
                    }
                )
                ann_id += 1
    doc = {
        "info": {"description": "Equine placental histomorphometry predictions", "date_created": time.strftime("%Y-%m-%d %H:%M:%S")},
        "images": images,
        "categories": COCO_CATEGORIES,
        "annotations": anns,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def _git(cfg: PipelineConfig, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(cfg.repo_root), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict[str, str]:
    out = {}
    for pkg in ("torch", "ultralytics", "rfdetr", "sahi", "opencv-python", "numpy", "pandas", "gradio"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "n/a"
    return out


def write_run_info(
    run_dir: Path,
    cfg: PipelineConfig,
    *,
    mode: str,
    results: Sequence[ImageResult],
    micro_conf: float | None = None,
    capilar_conf: float | None = None,
    device: str | None = None,
) -> None:
    import torch

    yolo = cfg.micro_backend == "yolo"
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "n_images": len(results),
        "images": [r.image_path.name for r in results],
        "device": device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git": {
            "commit": _git(cfg, "rev-parse", "HEAD"),
            "dirty": bool(_git(cfg, "status", "--porcelain")),
        },
        "micro": {
            "backend": cfg.micro_backend,
            "checkpoint": cfg.micro_yolo_ckpt if yolo else cfg.micro_ckpt,
            "conf": micro_conf if micro_conf is not None else (cfg.micro_yolo_conf if yolo else cfg.micro_conf),
            "input": f"stretch to {cfg.micro_canvas}x{cfg.micro_canvas}",
            "resolution": cfg.micro_yolo_imgsz if yolo else cfg.micro_resolution,
        },
        "capilar": {
            "checkpoint": cfg.capilar_ckpt,
            "conf": capilar_conf if capilar_conf is not None else cfg.capilar_conf,
            "imgsz": cfg.capilar_imgsz,
            "sahi": {
                "slice": [cfg.capilar_slice_w, cfg.capilar_slice_h],
                "overlap": cfg.capilar_overlap,
                "postprocess": cfg.capilar_postprocess,
                "match_metric": cfg.capilar_match_metric,
                "match_threshold": cfg.capilar_match_threshold,
            },
        },
        "calibration": {
            "scale_bar_um": SCALE_BAR_UM,
            "scale_bar_px_native": SCALE_BAR_PX_NATIVE,
            "native_fov_px": [NATIVE_W, NATIVE_H],
            "micro_area_um2_per_px2_stretch640": cfg.micro_area_um2_per_px2,
            "capilar_area_um2_per_px2_native": cfg.capilar_area_um2_per_px2,
            "note": "Area per image = sum of instance areas; union columns discount overlaps.",
        },
        "versions": {"python": platform.python_version(), **_versions()},
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_run_artifacts(
    results: Sequence[ImageResult],
    run_dir: Path,
    cfg: PipelineConfig,
    *,
    mode: str,
    micro_conf: float | None = None,
    capilar_conf: float | None = None,
    device: str | None = None,
) -> dict[str, Path]:
    """Write results.csv, instances.csv, annotations.coco.json and run_info.json."""
    paths = {
        "results": run_dir / "results.csv",
        "instances": run_dir / "instances.csv",
        "coco": run_dir / "annotations.coco.json",
    }
    write_results_csv(results, paths["results"])
    write_instances_csv(results, paths["instances"])
    write_coco_json(results, paths["coco"])
    write_run_info(
        run_dir, cfg, mode=mode, results=results,
        micro_conf=micro_conf, capilar_conf=capilar_conf, device=device,
    )
    return paths
