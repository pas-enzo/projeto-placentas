"""Known-answer tests for the metrics (no models, no GPU): python -m benchmark.eval.selftest"""

import numpy as np

from .instances import Inst
from .metrics import (bland_altman, calibrate_conf, coco_ap, instance_metrics, lin_ccc, match, morphometry,
                      pixel_metrics)


def sq(x, y, s=10, score=1.0):
    return Inst(x, y, np.ones((s, s), dtype=bool), score)


def close(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b}"


HW = (100, 100)

# 1) perfect prediction
gt = [sq(0, 0), sq(30, 30), sq(60, 60)]
perfect = [([sq(0, 0), sq(30, 30), sq(60, 60)], gt, HW), ([sq(10, 10)], [sq(10, 10)], HW)]
m = instance_metrics(perfect)
assert (m["tp"], m["fp"], m["fn"]) == (4, 0, 0) and m["f1"] == 1.0 and m["mean_iou"] == 1.0
assert pixel_metrics(perfect)["dice_pooled"] == 1.0
close(coco_ap(perfect)["ap"], 1.0, 1e-3)
mo = morphometry(perfect)
assert mo["area_sum"]["mean_abs_rel_error"] == 0 and mo["count"]["mean_abs_error"] == 0
print("1 perfeito: OK")

# 2) IoU thresholds: shift 2 px -> IoU 80/120 = 0.667 (match); shift 5 px -> 50/150 = 0.333 (no match)
tp, fp, fn, ious = match([sq(2, 0)], [sq(0, 0)])
assert (tp, fp, fn) == (1, 0, 0); close(ious[0], 80 / 120)
tp, fp, fn, _ = match([sq(5, 0)], [sq(0, 0)])
assert (tp, fp, fn) == (0, 1, 1)
print("2 limiares de IoU: OK")

# 3) overlapping predictions: SUM counts the common pixels twice, UNION once
a, b = sq(0, 0), sq(5, 0)  # overlap 5x10 = 50 px
ov = [([a, b], [sq(0, 0, 10), sq(5, 0, 10)], HW)]
f = morphometry(ov)
assert f["area_sum"]["aggregate_rel_error"] == 0  # same overlapping GT -> sums equal
only_union_pred = [([a, b], [Inst(0, 0, np.ones((10, 15), dtype=bool))], HW)]
mm = morphometry(only_union_pred)
close(mm["area_sum"]["signed_rel_error_mean"], (200 - 150) / 150)  # sum 200 vs gt 150
close(mm["area_union"]["signed_rel_error_mean"], 0.0)  # union 150 vs gt 150
print("3 soma x união: OK")

# 4) calibration: true detections score 0.9, false positives 0.2 -> F1 = 1 in the plateau 0.25..0.90
true_det = [sq(0, 0, score=0.9), sq(30, 30, score=0.9)]
fps = [sq(70, 70, score=0.2), sq(80, 10, score=0.2)]
val = [(true_det + fps, [sq(0, 0), sq(30, 30)], HW)]
conf, table = calibrate_conf(val)
assert 0.25 <= conf <= 0.9 and instance_metrics(val, conf)["f1"] == 1.0, conf
assert instance_metrics(val, 0.05)["f1"] < 1.0  # accepting the false positives hurts
print(f"4 calibração do limiar: OK (conf*={conf})")

# 5) CCC and Bland-Altman
x = np.array([1.0, 2.0, 3.0])
close(lin_ccc(x, x), 1.0)
close(lin_ccc(x + 1, x), 2 * (2 / 3) / (2 * (2 / 3) + 1))  # 0.5714: agreement drops with a constant bias
ba = bland_altman(x + 1, x)
close(ba["bias"], 1.0); close(ba["loa_low"], 1.0); close(ba["loa_high"], 1.0)
print("5 CCC e Bland-Altman: OK")

# 6) COCO AP with one missed instance out of 3 (perfect masks otherwise): AP ~ 2/3
miss = [([sq(0, 0), sq(30, 30)], gt, HW)]
ap = coco_ap(miss)["ap"]
assert 0.6 < ap < 0.72, ap
print(f"6 AP com 1 omissão em 3: OK ({ap:.3f})")

print("\nTODOS OS AUTOTESTES PASSARAM")
