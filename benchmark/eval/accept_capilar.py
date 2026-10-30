"""Acceptance test of the common sliced predictor against the published SAHI pipeline (capillary YOLO11s).

For N validation fields: pipeline.CapilarModel (SAHI, conf 0.33) vs benchmark.eval.sliced (same weights, same
slicing, SAHI's merge). Compares counts, summed area and instance-level agreement.
Run: python -m benchmark.eval.accept_capilar [n_fields]
"""

import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from benchmark.adapters.ultralytics_adapter import UltralyticsPredictor
from benchmark.eval import sliced
from benchmark.eval.instances import from_full, iou
from benchmark.eval.metrics import match
from pipeline.capilar import CapilarModel
from pipeline.config import PipelineConfig

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
VALID = Path(r"D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size\valid\images")
cfg = PipelineConfig()
files = sorted(glob.glob(str(VALID / "*")))[:: max(1, 27 // N)][:N]

ours = UltralyticsPredictor(cfg.capilar_ckpt_path, imgsz=640, conf=0.001)
pipe = CapilarModel(cfg)
tot = {"n_pipe": 0, "n_ours": 0, "a_pipe": 0, "a_ours": 0, "matched90": 0}
for f in files:
    bgr = cv2.imread(f)
    h, w = bgr.shape[:2]
    t = time.time()
    res = pipe.predict_path(Path(f))
    t_pipe = time.time() - t
    ref = [i for i in (from_full(np.asarray(m), 1.0) for m in res.masks) if i is not None]
    t = time.time()
    tp = sliced.collect(lambda tile: ours.predict(tile, cls_id=0), bgr)
    t_collect = time.time() - t
    t = time.time()
    mine = sliced.merge(tp, conf=cfg.capilar_conf)
    t_merge = time.time() - t
    tpn, fp, fn, ious = match(mine, ref, 0.5)
    tot["n_pipe"] += len(ref); tot["n_ours"] += len(mine)
    tot["a_pipe"] += sum(i.area for i in ref); tot["a_ours"] += sum(i.area for i in mine)
    tot["matched90"] += sum(1 for v in ious if v >= 0.9)
    print(f"{Path(f).name[:20]}: pipeline={len(ref):3d} inst | ours={len(mine):3d} inst | pareadas IoU>=0.5: {tpn:3d} "
          f"(IoU>=0.9: {sum(1 for v in ious if v >= 0.9):3d}) | área ours/pipe={sum(i.area for i in mine)/max(sum(i.area for i in ref),1):.4f} "
          f"| t: SAHI {t_pipe:.0f}s, coleta {t_collect:.0f}s, fusão {t_merge:.1f}s")
print(f"\nTOTAL: instâncias pipeline={tot['n_pipe']} ours={tot['n_ours']} | área ours/pipe={tot['a_ours']/tot['a_pipe']:.4f} "
      f"| {100*tot['matched90']/max(tot['n_pipe'],1):.1f}% das instâncias do pipeline reproduzidas com IoU>=0.9")
