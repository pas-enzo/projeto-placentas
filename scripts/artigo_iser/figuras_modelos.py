"""Figuras da selecao de modelos (a partir de benchmark_results).

  fig_model_comparison.png    AP de mascara (validacao = ponto cheio, teste = ponto vazio) dos modelos comparados no fold principal, micro e capilar
  fig_f1_vs_confidence.png    F1 x limiar de confianca na validacao dos 6 folds (YOLO26s micro, YOLO11s capilar), com o limiar escolhido
Cores: paleta de referencia do guia dataviz (slots 1-3 validam todos os pares): azul #2a78d6, laranja #eb6834, verde-agua #1baf7a.
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

R = "D:/projeto_placentas_clayton/benchmark_results/"
FIG = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\artigo\figs")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3de", "#f0efea"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK})
FAM_COLOR = {"YOLO": BLUE, "Mask R-CNN": ORANGE, "RF-DETR": AQUA}
ADOPTED = {"micro": "YOLO26s", "capilar": "YOLO11s"}
FINALISTS = {"micro": 3, "capilar": 2}          # make_phase2.py: top-3 micro, top-2 capilar pelo AP de validacao


def pretty(model):
    m = re.match(r"yolov(\d+)([a-z])", model)
    if m:
        return f"YOLOv{m.group(1)}{m.group(2)}", "YOLO"
    m = re.match(r"yolo(\d+)([a-z])", model)
    if m:
        return f"YOLO{m.group(1)}{m.group(2)}", "YOLO"
    if model.startswith("maskrcnn"):
        return ("Mask R-CNN v2" if model.endswith("_v2") else "Mask R-CNN"), "Mask R-CNN"
    if model.startswith("rfdetr"):
        return "RF-DETR Nano", "RF-DETR"
    return model, "YOLO"


# ------------------------------------------------------------------ Figura: comparacao dos modelos
d = pd.read_csv(R + "results.csv")
p = d[(d.fold == "primary") & (d.norm == "none")].copy()
ep = {}
for rd in glob.glob(R + "*__none__s0__*"):
    c = json.load(open(rd + "/config.json"))
    ep[rd.split("\\")[-1].split("/")[-1]] = c.get("epochs")
p["epochs"] = p.run_id.map(ep)
fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.6), dpi=300, gridspec_kw={"height_ratios": [13, 8], "hspace": .42})
for ax, task, title in zip(axes, ("micro", "capilar"), ("Microcotyledons", "Capillaries (tile level)")):
    s = p[p.task == task].sort_values("val_report_coco_ap_ap", ascending=False).reset_index(drop=True)
    labels, fams = [], []
    for _, r in s.iterrows():
        name, fam = pretty(r.model)
        if r.model.startswith("maskrcnn") and task == "capilar":
            name = f"{name} ({int(r.epochs)} ep.)"
        labels.append(name)
        fams.append(fam)
    y = np.arange(len(s))[::-1]
    k = FINALISTS[task]
    ax.axhspan(y[k - 1] - .5, y[0] + .5, color=BAND, zorder=0)
    ax.grid(axis="x", color=GRID, lw=.6, zorder=0)
    for yi, (_, r), fam in zip(y, s.iterrows(), fams):
        c = FAM_COLOR[fam]
        ax.plot([r.val_report_coco_ap_ap, r.test_report_coco_ap_ap], [yi, yi], color=c, lw=1.0, alpha=.45, zorder=1)
        ax.scatter(r.val_report_coco_ap_ap, yi, s=26, color=c, edgecolor="white", linewidth=.8, zorder=3)
        ax.scatter(r.test_report_coco_ap_ap, yi, s=26, facecolor="white", edgecolor=c, linewidth=1.3, zorder=3)
    ax.set_yticks(y, labels)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if lab == ADOPTED[task]:
            t.set_fontweight("bold")
            t.set_text(lab + " \u2605")
    ax.set_yticklabels([l + " \u2605" if l == ADOPTED[task] else l for l in labels])
    for t, lab in zip(ax.get_yticklabels(), labels):
        if lab == ADOPTED[task]:
            t.set_fontweight("bold")
    ax.set_ylim(-.7, len(s) - .3)
    lo, hi = min(s.val_report_coco_ap_ap.min(), s.test_report_coco_ap_ap.min()), max(s.val_report_coco_ap_ap.max(), s.test_report_coco_ap_ap.max())
    pad = (hi - lo) * .08
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("Mask AP (IoU 0.50\u20130.95)")
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
handles = ([Patch(color=c, label=f) for f, c in FAM_COLOR.items()] +
           [Line2D([], [], marker="o", ls="", color=INK2, label="Validation"),
            Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, label="Test"),
            Patch(color=BAND, label="Finalists (best validation AP)")])
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.005), columnspacing=1.4)
fig.subplots_adjust(left=.21, right=.97, top=.965, bottom=.14)
fig.savefig(FIG / "fig_model_comparison.png", dpi=300)
plt.close(fig)
print("fig_model_comparison ok:", {t: len(p[p.task == t]) for t in ("micro", "capilar")})

# ------------------------------------------------------------------ Figura: F1 x confianca (validacao), 6 folds
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), dpi=300, sharey=True)
for ax, task, model, color, title in ((axes[0], "micro", "yolo26s-seg", BLUE, "Microcotyledons (YOLO26s, whole fields)"),
                                       (axes[1], "capilar", "yolo11s-seg", YELLOW, "Capillaries (YOLO11s, tiles)")):
    runs = sorted(glob.glob(R + f"{task}__ultralytics__{model}__none__*"))
    ax.grid(color=GRID, lw=.6, zorder=0)
    stars = []
    for rd in runs:
        mv = json.load(open(rd + "/metrics_val.json"))
        cal = pd.DataFrame(mv["calibration"])
        ax.plot(cal.conf, cal.f1, color=color, lw=1.1, alpha=.55, zorder=2)
        cs = mv["conf_star"]
        f1s = float(cal.iloc[(cal.conf - cs).abs().argmin()].f1)
        ax.scatter(cs, f1s, s=22, color=color, edgecolor=INK, linewidth=.8, zorder=4)
        stars.append(cs)
    ax.set_xlabel("Confidence threshold")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.text(.02, .06, f"6 folds; chosen threshold {min(stars):.2f}\u2013{max(stars):.2f}", transform=ax.transAxes, fontsize=7, color=INK2)
axes[0].set_ylabel("F1-score on the validation mare")
fig.legend(handles=[Line2D([], [], color=INK2, lw=1.1, label="One curve per fold"),
                    Line2D([], [], marker="o", ls="", color=INK2, markeredgecolor=INK, label="Threshold that maximizes F1")],
           loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=.085, right=.985, top=.90, bottom=.27, wspace=.08)
fig.savefig(FIG / "fig_f1_vs_confidence.png", dpi=300)
plt.close(fig)
print("fig_f1_vs_confidence ok")
