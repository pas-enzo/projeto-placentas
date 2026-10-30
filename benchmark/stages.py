"""Evaluation stages (protocol of docs/BENCHMARK_PLAN.md, sections 5 and 5.1-5.2).

  eval_val  : predict the validation split, calibrate the confidence by maximizing F1 (area is NOT a criterion),
              report validation metrics. Capillary: validation is at TILE level (fast), as in the article.
  eval_test : with the confidence fixed on validation, report the test split. Capillary additionally runs the
              FIELD-level protocol (common slicing + SAHI greedy merge) on the native test fields.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from benchmark.core import RunSpec, native_dir, read_json, view_dir, write_json
from benchmark.eval import metrics, sliced
from benchmark.eval.instances import read_gt_polygons

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _files(directory: Path, limit: int | None = None) -> list[Path]:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS)
    return files[:limit] if limit else files


def _cls(spec: RunSpec) -> int | None:
    return 0 if spec.task == "capilar" else None  # capillary models also predict the micro class: keep class 0


def predict_split(predictor, view: Path, split: str, cls_id: int | None, limit: int | None = None):
    images, ms = [], []
    for f in _files(view / split / "images", limit):
        bgr = cv2.imread(str(f))
        h, w = bgr.shape[:2]
        t = time.time()
        pred = predictor.predict(bgr, cls_id=cls_id)
        ms.append(1000 * (time.time() - t))
        gt = read_gt_polygons(view / split / "labels" / f"{f.stem}.txt", h, w, 0)
        images.append((pred, gt, (h, w)))
    return images, ms


def split_report(images, conf: float) -> dict:
    return {"conf": conf, "instance": metrics.instance_metrics(images, conf), "coco_ap": metrics.coco_ap(images),
            "pixel": metrics.pixel_metrics(images, conf), "morphometry": metrics.morphometry(images, conf)}


def eval_val(spec: RunSpec, run_dir: Path, family) -> dict:
    predictor = family.make_predictor(spec, run_dir, conf=0.001)
    images, ms = predict_split(predictor, view_dir(spec), "valid", _cls(spec), spec.eval.get("max_images"))
    conf, table = metrics.calibrate_conf(images)
    res = {"level": "tile" if spec.task == "capilar" else "image", "n_images": len(images), "conf_star": conf,
           "ms_per_image": float(np.mean(ms)), "report": split_report(images, conf), "calibration": table}
    write_json(run_dir / "metrics_val.json", res)
    return res


def _field_level(spec: RunSpec, predictor, conf: float, limit: int | None) -> dict:
    nat = native_dir(spec)
    normalizer = None
    if spec.norm != "none":  # the very transformation a deployed pipeline would apply, fitted on TRAIN fields only
        from benchmark.views.stain import StainNormalizer

        normalizer = StainNormalizer(spec.norm).fit(_files(nat / "train" / "images"))
    images, secs = [], []
    for f in _files(nat / "test" / "images", limit):
        bgr = cv2.imread(str(f))
        if normalizer is not None:
            rgb, _ = normalizer.transform(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        t = time.time()
        merged = sliced.merge(sliced.collect(lambda tile: predictor.predict(tile, cls_id=0), bgr), conf)
        secs.append(time.time() - t)
        images.append((merged, read_gt_polygons(nat / "test" / "labels" / f"{f.stem}.txt", h, w, 0), (h, w)))
    return {"level": "field", "n_fields": len(images), "seconds_per_field": float(np.mean(secs)),
            "instance": metrics.instance_metrics(images), "pixel": metrics.pixel_metrics(images),
            "morphometry": metrics.morphometry(images)}


def eval_test(spec: RunSpec, run_dir: Path, family) -> dict:
    conf = read_json(run_dir / "metrics_val.json")["conf_star"]
    predictor = family.make_predictor(spec, run_dir, conf=0.001)
    images, ms = predict_split(predictor, view_dir(spec), "test", _cls(spec), spec.eval.get("max_images"))
    res = {"level": "tile" if spec.task == "capilar" else "image", "n_images": len(images), "conf_star": conf,
           "ms_per_image": float(np.mean(ms)), "report": split_report(images, conf)}
    if spec.task == "capilar":
        predictor.conf = conf  # NMS-equivalent shortcut: the merge only keeps >= conf anyway
        res["field"] = _field_level(spec, predictor, conf, spec.eval.get("max_fields"))
    write_json(run_dir / "metrics_test.json", res)
    return res
