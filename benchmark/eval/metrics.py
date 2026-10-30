"""Metrics for the benchmark (see docs/BENCHMARK_PLAN.md, section 5).

Input everywhere: `images`, a list of per-image records `(pred, gt, (h, w))` with `pred`/`gt` lists of `Inst`.

  * instance level: greedy IoU>=0.5 matching -> P, R, F1, mean IoU of matched pairs
  * COCO mask AP (AP50-95, AP50, AP75) via pycocotools - threshold-free, used for the ranking
  * pixel level: Dice / IoU of the UNION of instance masks (common to every technique)
  * morphometry per field: area (sum and union), count, relative/signed errors, Pearson, Lin's CCC, Bland-Altman
  * confidence calibration on validation by maximizing F1; bootstrap CIs over fields
"""

from __future__ import annotations

import numpy as np

from .instances import Inst, inter_area, iou, union_mask

Image = tuple[list[Inst], list[Inst], tuple[int, int]]


# ------------------------------------------------------------------ instance matching
def match(pred: list[Inst], gt: list[Inst], thr: float = 0.5) -> tuple[int, int, int, list[float]]:
    """Greedy one-to-one matching by descending IoU. Returns (tp, fp, fn, matched IoUs)."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if inter_area(p, g):
                pairs.append((iou(p, g), i, j))
    pairs.sort(reverse=True)
    used_p, used_g, ious = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(v)
    tp = len(ious)
    return tp, len(pred) - tp, len(gt) - tp, ious


def filter_conf(images: list[Image], conf: float) -> list[Image]:
    return [([p for p in pred if p.score >= conf], gt, hw) for pred, gt, hw in images]


def instance_metrics(images: list[Image], conf: float | None = None, thr: float = 0.5) -> dict:
    if conf is not None:
        images = filter_conf(images, conf)
    tp = fp = fn = 0
    ious: list[float] = []
    for pred, gt, _ in images:
        a, b, c, v = match(pred, gt, thr)
        tp, fp, fn = tp + a, fp + b, fn + c
        ious += v
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"conf": conf, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-12), "mean_iou": float(np.mean(ious)) if ious else 0.0}


# ------------------------------------------------------------------ confidence calibration
def calibrate_conf(val_images: list[Image], grid=None) -> tuple[float, list[dict]]:
    """Confidence that maximizes F1 on VALIDATION (area is NOT part of the criterion)."""
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2) if grid is None else grid
    table = [instance_metrics(val_images, float(c)) for c in grid]
    top = max(t["f1"] for t in table)
    plateau = [t["conf"] for t in table if t["f1"] >= top - 1e-12]
    return float(np.median(plateau)), table  # ties: the middle of the plateau is the most robust choice


# ------------------------------------------------------------------ pixel level (union masks)
def pixel_metrics(images: list[Image], conf: float | None = None) -> dict:
    if conf is not None:
        images = filter_conf(images, conf)
    inter = up = ug = 0
    dice_i, iou_i = [], []
    for pred, gt, (h, w) in images:
        mp, mg = union_mask(pred, h, w), union_mask(gt, h, w)
        i, a, b = int((mp & mg).sum()), int(mp.sum()), int(mg.sum())
        inter, up, ug = inter + i, up + a, ug + b
        dice_i.append(2 * i / (a + b) if a + b else 1.0)
        iou_i.append(i / (a + b - i) if a + b - i else 1.0)
    return {"dice_pooled": 2 * inter / max(up + ug, 1), "iou_pooled": inter / max(up + ug - inter, 1),
            "dice_per_image_mean": float(np.mean(dice_i)), "iou_per_image_mean": float(np.mean(iou_i))}


# ------------------------------------------------------------------ COCO mask AP
def coco_ap(images: list[Image]) -> dict:
    """COCO mask AP with a single category. Scores must be real confidences (not available for semantic nets)."""
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_doc = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "obj"}]}
    dets, ann_id = [], 1

    def rle(it: Inst, h: int, w: int) -> dict:
        r = mu.encode(np.asfortranarray(it.full(h, w).astype(np.uint8)))
        r["counts"] = r["counts"].decode("ascii")
        return r

    for k, (pred, gt, (h, w)) in enumerate(images):
        gt_doc["images"].append({"id": k, "height": h, "width": w})
        for g in gt:
            r = rle(g, h, w)
            gt_doc["annotations"].append({"id": ann_id, "image_id": k, "category_id": 1, "segmentation": r,
                                          "area": float(g.area), "bbox": list(mu.toBbox(r)), "iscrowd": 0})
            ann_id += 1
        for p in pred:
            dets.append({"image_id": k, "category_id": 1, "segmentation": rle(p, h, w), "score": float(p.score)})
    if not dets or not gt_doc["annotations"]:
        return {"ap": 0.0, "ap50": 0.0, "ap75": 0.0}
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_doc
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.maxDets = [10, 100, 1000]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {"ap": float(s[0]), "ap50": float(s[1]), "ap75": float(s[2])}


# ------------------------------------------------------------------ morphometry per field
def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (agreement, not just correlation)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    sxy = np.cov(x, y, bias=True)[0, 1]
    denom = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return float(2 * sxy / denom) if denom else float("nan")


def bland_altman(x: np.ndarray, y: np.ndarray) -> dict:
    """pred (x) vs gt (y): bias and 95% limits of agreement of the differences."""
    d = np.asarray(x, float) - np.asarray(y, float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return {"bias": float(d.mean()), "loa_low": float(d.mean() - 1.96 * sd), "loa_high": float(d.mean() + 1.96 * sd)}


def per_field(images: list[Image], conf: float | None = None) -> dict[str, np.ndarray]:
    """Per-field arrays: counts and areas (px) of prediction and ground truth, as SUM of instances and as UNION."""
    if conf is not None:
        images = filter_conf(images, conf)
    cols = {k: [] for k in ("n_pred", "n_gt", "area_pred_sum", "area_gt_sum", "area_pred_union", "area_gt_union")}
    for pred, gt, (h, w) in images:
        cols["n_pred"].append(len(pred))
        cols["n_gt"].append(len(gt))
        cols["area_pred_sum"].append(sum(p.area for p in pred))
        cols["area_gt_sum"].append(sum(g.area for g in gt))
        cols["area_pred_union"].append(int(union_mask(pred, h, w).sum()))
        cols["area_gt_union"].append(int(union_mask(gt, h, w).sum()))
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def morphometry(images: list[Image], conf: float | None = None) -> dict:
    f = per_field(images, conf)
    out: dict = {"n_fields": len(f["n_gt"])}
    for kind in ("sum", "union"):
        p, g = f[f"area_pred_{kind}"], f[f"area_gt_{kind}"]
        ok = g > 0
        rel = (p[ok] - g[ok]) / g[ok]
        out[f"area_{kind}"] = {
            "mean_abs_rel_error": float(np.abs(rel).mean()), "median_abs_rel_error": float(np.median(np.abs(rel))),
            "signed_rel_error_mean": float(rel.mean()),
            "aggregate_rel_error": float(abs(p.sum() - g.sum()) / max(g.sum(), 1)),
            "pearson": float(np.corrcoef(p, g)[0, 1]) if len(p) > 2 and p.std() and g.std() else float("nan"),
            "ccc": lin_ccc(p, g), **{f"ba_{k}": v for k, v in bland_altman(p, g).items()},
        }
    n_p, n_g = f["n_pred"], f["n_gt"]
    out["count"] = {"mean_abs_error": float(np.abs(n_p - n_g).mean()), "signed_error_mean": float((n_p - n_g).mean()),
                    "pearson": float(np.corrcoef(n_p, n_g)[0, 1]) if len(n_p) > 2 and n_p.std() and n_g.std() else float("nan"),
                    "ccc": lin_ccc(n_p, n_g), **{f"ba_{k}": v for k, v in bland_altman(n_p, n_g).items()}}
    return out


# ------------------------------------------------------------------ uncertainty
def bootstrap_ci(images: list[Image], stat, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of `stat(list_of_images)` resampling FIELDS with replacement."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(images), len(images))
        vals.append(stat([images[i] for i in idx]))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def full_report(val: list[Image], test: list[Image]) -> dict:
    """Everything the plan asks for: conf calibrated on VAL (F1), then every metric on TEST."""
    conf, table = calibrate_conf(val)
    return {
        "conf_star": conf,
        "val_f1_at_conf": max(t["f1"] for t in table),
        "test": {
            "instance": instance_metrics(test, conf),
            "coco_ap": coco_ap(test),
            "pixel": pixel_metrics(test, conf),
            "morphometry": morphometry(test, conf),
        },
    }
