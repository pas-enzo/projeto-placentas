"""Acceptance test: the new metrics must reproduce the known YOLO-micro numbers on the OLD validation set.

Known (scripts/validation/reproduce_validation.py, conf 0.46, 27 images, greedy IoU>=0.5):
  P 0.9061  R 0.8721  F1 0.8888  mean IoU 0.8809  area err (sum) 0.0062  area err (union) 0.0657
Run: python -m benchmark.eval.accept_micro
"""

import glob
import json
from pathlib import Path

import cv2

from benchmark.adapters.ultralytics_adapter import UltralyticsPredictor
from benchmark.eval.instances import read_gt_polygons
from benchmark.eval.metrics import calibrate_conf, coco_ap, instance_metrics, morphometry, pixel_metrics

ROOT = Path(__file__).resolve().parents[2]
VALID = Path(r"D:\projeto_placentas_clayton\dataset_v2_ready_for_yolo\valid")
WEIGHTS = ROOT / "archive/experiments/v2_yolo_early/runs/segment/placentas_v11_aug_v2/weights/best.pt"

pred = UltralyticsPredictor(WEIGHTS, imgsz=640, conf=0.001)
images = []
for f in sorted(glob.glob(str(VALID / "images" / "*"))):
    bgr = cv2.imread(f)
    h, w = bgr.shape[:2]
    images.append((pred.predict(bgr), read_gt_polygons(VALID / "labels" / (Path(f).stem + ".txt"), h, w), (h, w)))
print(f"{len(images)} imagens; instâncias previstas com conf>=0.001: {sum(len(p) for p, _, _ in images)}")

m = instance_metrics(images, 0.46)
mo = morphometry(images, 0.46)
print("\n@0.46  precisão %.4f  revocação %.4f  F1 %.4f  IoU médio %.4f" % (m["precision"], m["recall"], m["f1"], m["mean_iou"]))
print("       erro de área agregado: soma %.4f | união %.4f   (referência: 0.0062 | 0.0657)"
      % (mo["area_sum"]["aggregate_rel_error"], mo["area_union"]["aggregate_rel_error"]))
ok = (abs(m["precision"] - 0.9061) < 2e-3 and abs(m["recall"] - 0.8721) < 2e-3 and abs(m["f1"] - 0.8888) < 2e-3
      and abs(m["mean_iou"] - 0.8809) < 2e-3 and abs(mo["area_sum"]["aggregate_rel_error"] - 0.0062) < 1e-3)
print("ACEITAÇÃO:", "PASSOU (reproduz os números conhecidos)" if ok else "FALHOU")

conf, _ = calibrate_conf(images)
print(f"\nNOVO (o que o benchmark faz): conf* que maximiza o F1 = {conf} (o artigo usou 0.46, escolhido pelo S(c))")
print("  F1 em conf*: %.4f" % instance_metrics(images, conf)["f1"])
ap = coco_ap(images)
print("  AP máscara: AP50-95 %.4f | AP50 %.4f | AP75 %.4f" % (ap["ap"], ap["ap50"], ap["ap75"]))
px = pixel_metrics(images, conf)
print("  pixel (união): Dice %.4f | IoU %.4f" % (px["dice_pooled"], px["iou_pooled"]))
mc = morphometry(images, conf)
print("  por campo: erro de área |médio| soma %.3f união %.3f | CCC área %.3f | CCC contagem %.3f"
      % (mc["area_sum"]["mean_abs_rel_error"], mc["area_union"]["mean_abs_rel_error"], mc["area_sum"]["ccc"], mc["count"]["ccc"]))
