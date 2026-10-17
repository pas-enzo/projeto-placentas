"""Consistency check: CLI vs single-image tab vs batch tab vs hover on the same FOVs.

Usage: python scripts/validation/check_interfaces.py [n_fovs]
"""
from __future__ import annotations

import glob
import html
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from pipeline import app_gradio as app
from pipeline.run import run_pipeline

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
fovs = sorted(glob.glob(r"D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size\valid\images\*"))[:n]
out = ROOT / "outputs" / "_check_interfaces"
tmp = out / "in"
tmp.mkdir(parents=True, exist_ok=True)

ok = True


def check(name, a, b, tol=0.0):
    global ok
    good = abs(a - b) <= tol
    ok &= good
    print(f"  {'OK ' if good else 'FAIL'} {name}: {a} vs {b}")


# CLI on the folder of originals (SAHI reads the original file)
for f in fovs:
    (tmp / Path(f).name).write_bytes(Path(f).read_bytes())
cli = run_pipeline(tmp, out / "cli", mode="both", save_overlays=False, save_masks=False)
cli_by = {r.image_path.name: r for r in cli}

# Batch tab (analyze_image on files)
_, _, _, _, dl, _ = app.predict_batch(fovs, "both")
batch = pd.read_csv(dl[0])
inst = pd.read_csv(dl[1])

for f in fovs:
    name = Path(f).name
    print(name[:30])
    c = cli_by[name]
    b = batch[batch.image == name].iloc[0]
    check("batch micro count vs CLI", int(b.micro_count), c.micro.count)
    check("batch micro area vs CLI", float(b.micro_area_um2), round(c.micro.area_um2, 4), 1e-3)
    check("batch capilar count vs CLI", int(b.capilar_count), c.capilar.count)
    check("batch capilar area vs CLI", float(b.capilar_area_um2), round(c.capilar.area_um2, 4), 1e-3)
    # instances.csv sums == results.csv
    sub = inst[inst.image == name]
    check("instances.csv micro sum vs results", round(sub[sub["head"] == "micro"].area_um2.sum(), 2), float(b.micro_area_um2), 0.02)
    check("instances.csv capilar sum vs results", round(sub[sub["head"] == "capilar"].area_um2.sum(), 2), float(b.capilar_area_um2), 0.02)

    # Single tab: upload decoded with PIL (as Gradio does) -> PNG temp -> SAHI
    arr = np.array(Image.open(f).convert("RGB"))
    _, _, _, table, _, _, hover = app.predict(arr, "both")
    vals = {r["Métrica"]: r for _, r in table.iterrows()}
    cnt_m, cnt_c = int(vals["Contagem"].iloc[1]), int(vals["Contagem"].iloc[2])
    check("single (PIL->PNG) micro count vs CLI", cnt_m, c.micro.count)
    check("single (PIL->PNG) capilar count vs CLI", cnt_c, c.capilar.count)
    a_c = float(str(vals["Área total, soma das instâncias (µm²)"].iloc[2]).replace(",", ""))
    check("single capilar area vs CLI (um2)", a_c, round(c.capilar.area_um2, 2), 0.02 * c.capilar.area_um2)
    # Hover tooltips sum == totals
    tips = [html.unescape(t) for t in re.findall(r'data-tip="([^"]*)"', hover)]
    for lab, ref in (("Microcotil", c.micro), ("Capilar", c.capilar)):
        a = [float(re.search(r"Área: ([\d,\.]+)", t).group(1).replace(",", "")) for t in tips if t.startswith(lab)]
        check(f"hover {lab} tooltips count", len(a), ref.count)
        check(f"hover {lab} sum vs total", round(sum(a), 1), round(ref.area_um2, 1), 0.05 * max(len(a), 1))

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
