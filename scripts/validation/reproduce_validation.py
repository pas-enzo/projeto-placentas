"""Reproduce the validation metrics with the unified pipeline (end-to-end check).

Micro:   27 valid images (640x640 Roboflow stretch export) with YOLO and RF-DETR.
Capilar: 27 valid native FOVs with YOLO11s-seg + SAHI.
Matching: greedy by IoU >= 0.5 (same protocol as the evaluation notebooks); area error on pixel totals.

Usage: python scripts/validation/reproduce_validation.py [out.json]
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from pipeline.capilar import CapilarModel  # noqa: E402
from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.micro import load_micro_model  # noqa: E402
from pipeline.types import CropMasks  # noqa: E402
from pipeline.viz import read_bgr  # noqa: E402

MICRO_VALID = Path(r"D:\projeto_placentas_clayton\dataset_v2_ready_for_yolo\valid")
CAP_VALID = Path(r"D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size\valid")


def crop_of(mask):
    r = np.flatnonzero(mask.any(1))
    if r.size == 0:
        return None
    c = np.flatnonzero(mask.any(0))
    return (int(c[0]), int(r[0]), mask[r[0] : r[-1] + 1, c[0] : c[-1] + 1] > 0)


def gt_crops(split: Path, img: str, cls: int, h: int, w: int):
    lab = split / "labels" / (Path(img).stem + ".txt")
    out = []
    for ln in lab.read_text().splitlines():
        p = ln.split()
        if not p or int(p[0]) != cls:
            continue
        pts = (np.array(p[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        c = crop_of(m)
        if c:
            out.append(c)
    return out


def pred_crops(masks):
    out = []
    if isinstance(masks, CropMasks):
        for i in range(len(masks)):
            y0, x0, crop = masks.crop(i)
            if crop.size:
                out.append((x0, y0, crop > 0))
        return out
    for m in masks:
        c = crop_of(m)
        if c:
            out.append(c)
    return out


def inter(a, b):
    ax, ay, am = a
    bx, by, bm = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + am.shape[1], bx + bm.shape[1]), min(ay + am.shape[0], by + bm.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((am[y0 - ay : y1 - ay, x0 - ax : x1 - ax] & bm[y0 - by : y1 - by, x0 - bx : x1 - bx]).sum())


def area(c):
    return int(c[2].sum())


def match(pred, gt):
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            it = inter(p, g)
            if it:
                pairs.append((it / (area(p) + area(g) - it), i, j))
    pairs.sort(reverse=True)
    up, ug, ious = set(), set(), []
    for iou, i, j in pairs:
        if iou < 0.5:
            break
        if i in up or j in ug:
            continue
        up.add(i)
        ug.add(j)
        ious.append(iou)
    tp = len(ious)
    return tp, len(pred) - tp, len(gt) - tp, ious


def union_px(crops, h, w):
    u = np.zeros((h, w), bool)
    for x, y, m in crops:
        u[y : y + m.shape[0], x : x + m.shape[1]] |= m
    return int(u.sum())


def summarize(rows, label):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    ious = [i for r_ in rows for i in r_["ious"]]
    pa, ga = sum(r_["pred_area"] for r_ in rows), sum(r_["gt_area"] for r_ in rows)
    ua = sum(r_["union_area"] for r_ in rows)
    out = {
        "label": label, "n_images": len(rows), "P": round(p, 4), "R": round(r, 4),
        "F1": round(2 * p * r / max(p + r, 1e-9), 4), "mean_IoU": round(float(np.mean(ious)), 4),
        "area_rel_error_sum": round(abs(pa - ga) / ga, 4),
        "area_rel_error_union": round(abs(ua - ga) / ga, 4),
        "pred_over_gt_sum": round(pa / ga, 4), "pred_over_gt_union": round(ua / ga, 4),
        "overlap_pct_of_sum": round((1 - ua / pa) * 100, 2) if pa else None,
        "pred_count": tp + fp, "gt_count": tp + fn,
    }
    out["S"] = round(0.6 * out["F1"] + 0.3 * out["mean_IoU"] + 0.1 * (1 - out["area_rel_error_sum"]), 4)
    return out


def run_micro(backend: str):
    cfg = PipelineConfig()
    model = load_micro_model(cfg, backend=backend)
    rows = []
    for f in sorted(glob.glob(str(MICRO_VALID / "images" / "*"))):
        bgr = read_bgr(Path(f))
        h, w = bgr.shape[:2]
        res = model.predict_bgr(bgr)
        pc = pred_crops(res.micro.masks if hasattr(res, "micro") else res.masks)
        gc = gt_crops(MICRO_VALID, f, 0, h, w)
        tp, fp, fn, ious = match(pc, gc)
        rows.append(dict(img=Path(f).name[:20], tp=tp, fp=fp, fn=fn, ious=ious,
                         pred_area=sum(map(area, pc)), gt_area=sum(map(area, gc)),
                         union_area=union_px(pc, h, w)))
    return summarize(rows, f"micro-{backend}")


def run_capilar():
    cfg = PipelineConfig()
    model = CapilarModel(cfg)
    rows = []
    for k, f in enumerate(sorted(glob.glob(str(CAP_VALID / "images" / "*"))), 1):
        t = time.time()
        res = model.predict_path(Path(f))
        h, w = read_bgr(Path(f)).shape[:2]
        pc = pred_crops(res.masks)
        gc = gt_crops(CAP_VALID, f, 0, h, w)
        tp, fp, fn, ious = match(pc, gc)
        rows.append(dict(img=Path(f).name[:20], tp=tp, fp=fp, fn=fn, ious=ious,
                         pred_area=sum(map(area, pc)), gt_area=sum(map(area, gc)),
                         union_area=union_px(pc, h, w)))
        print(f"  capilar {k}/27 {Path(f).name[:18]} n={len(pc)} ({time.time()-t:.0f}s)", flush=True)
    return summarize(rows, "capilar-yolo+sahi")


if __name__ == "__main__":
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "validation_repro.json"
    results = []
    for backend in ("yolo", "rfdetr"):
        r = run_micro(backend)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
    r = run_capilar()
    print(json.dumps(r, ensure_ascii=False), flush=True)
    results.append(r)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", out_path)
