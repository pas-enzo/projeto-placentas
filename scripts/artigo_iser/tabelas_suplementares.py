"""Tabelas de selecao de modelos, geradas de benchmark_results.

  1) Tabela pequena (tab:finalists), inserida em methods_06_models_en.tex no marcador '% AUTO-TABLE-FINALISTS' (idempotente):
     modelos avaliados nos 6 folds, com parametros e AP de validacao (media +- DP entre folds). Mostra a regra do um erro-padrao.
  2) Tabelas grandes (todos os modelos da triagem; configuracao de treino), gravadas em artigo/suplemento_tabelas.tex para o material suplementar.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"D:\projeto_placentas_clayton\dev\projeto-placentas")
sys.path.insert(0, str(REPO))
from benchmark.recipes import RECIPES  # noqa: E402

R = "D:/projeto_placentas_clayton/benchmark_results/"
ART = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\artigo")
M6, SUP = ART / "methods_06_models_en.tex", ART / "suplemento_tabelas.tex"
FINALISTS = {"micro": 3, "capilar": 2}
SIX = {"micro": ["yolo26m-seg.pt", "yolo26s-seg.pt", "yolo11m-seg.pt"], "capilar": ["yolo11s-seg.pt", "yolo26s-seg.pt"]}
ADOPTED = {"micro": "yolo26s-seg.pt", "capilar": "yolo11s-seg.pt"}


def pretty(model):
    m = re.match(r"yolov(\d+)([a-z])", model)
    if m:
        return f"YOLOv{m.group(1)}{m.group(2)}"
    m = re.match(r"yolo(\d+)([a-z])", model)
    if m:
        return f"YOLO{m.group(1)}{m.group(2)}"
    if model.startswith("maskrcnn"):
        return "Mask R-CNN v2" if model.endswith("_v2") else "Mask R-CNN"
    if model.startswith("rfdetr"):
        return "RF-DETR Nano"
    return model


d = pd.read_csv(R + "results.csv")
n = d[d.norm == "none"].copy()
params = {}
for rid, mdl in zip(n.run_id, n.model):
    params[mdl] = json.load(open(R + rid + "/run.json"))["model"].get("params_millions")

# ------------------------------------------------------------------ 1) tabela pequena
rows, se = [], {}
for task, title in (("micro", "Microcotyledons"), ("capilar", "Capillaries")):
    first = True
    vals = {m: n[(n.task == task) & (n.model == m)].val_report_coco_ap_ap for m in SIX[task]}
    for m in SIX[task]:
        s = vals[m]
        assert len(s) == 6, (task, m, len(s))
        name = pretty(m)
        label = rf"\textbf{{{name}}}" if m == ADOPTED[task] else name
        mean, sd = s.mean(), s.std()
        rows.append(f"{title if first else ''} & {label} & {params[m]:.1f} & {mean:.3f} $\\pm$ {sd:.3f} \\\\")
        first = False
    best = max(vals, key=lambda k: vals[k].mean())
    se[task] = vals[best].std() / np.sqrt(len(vals[best]))
    rows.append(r"\hline")
t_fin = r"""\begin{table}[H]
\centering
\caption{Models evaluated in the six folds: number of parameters and mean ($\pm$ standard deviation across folds) of the mask average precision (AP) on
the validation mare of each fold. The adopted model (bold) is the simplest one within one standard error of the best (standard deviation across folds divided by
$\sqrt{6}$: %.3f for microcotyledons and %.3f for capillaries).}
\label{tab:finalists}
\vspace{8pt}
\begin{tabular}{llcc}
\hline
Structure & Model & Parameters (millions) & Validation AP \\
\hline
""" % (se["micro"], se["capilar"]) + "\n".join(rows) + r"""
\end{tabular}
\end{table}
"""
txt = M6.read_text(encoding="utf-8")
block = "% BEGIN auto-table finalists\n" + t_fin + "% END auto-table finalists\n"
if "% AUTO-TABLE-FINALISTS" in txt:
    txt = txt.replace("% AUTO-TABLE-FINALISTS\n", block)
else:
    txt = re.sub(r"% BEGIN auto-table finalists\n.*?% END auto-table finalists\n", lambda _: block, txt, flags=re.S)
M6.write_text(txt, encoding="utf-8")
print("tabela pequena ok | erro-padrao do melhor:", {k: round(v, 4) for k, v in se.items()})

# ------------------------------------------------------------------ 2) tabelas grandes (suplemento)
p = n[n.fold == "primary"].copy()
info = {rid: (params[mdl], json.load(open(R + rid + "/run.json"))["config"].get("epochs")) for rid, mdl in zip(p.run_id, p.model)}
p["params"] = p.run_id.map(lambda r: info[r][0])
p["epochs"] = p.run_id.map(lambda r: info[r][1])
big = []
for task, title in (("micro", "Microcotyledons"), ("capilar", "Capillaries (tile level)")):
    s = p[p.task == task].sort_values("val_report_coco_ap_ap", ascending=False).reset_index(drop=True)
    big.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{title}}}}} \\")
    for i, r in s.iterrows():
        name = pretty(r.model) + (f" ({int(r.epochs)} ep.)" if r.model.startswith("maskrcnn") and task == "capilar" else "")
        bt = int(r.batch_used) if pd.notna(r.batch_used) else ""
        big.append(f"{name} & {r.params:.1f} & {int(r.epochs)} & {bt} & {r.val_report_coco_ap_ap:.3f} & {r.test_report_coco_ap_ap:.3f} "
                   f"& {r.test_report_instance_f1:.3f} & {'yes' if i < FINALISTS[task] else ''} \\\\")
    big.append(r"\hline")
T1 = r"""\begin{table}[H]
\centering
\caption{Models compared in the first stage of model selection (primary fold, without stain normalization): parameters in millions, epochs and batch size of
each training, mask average precision (AP) on the validation and test mares, and F1-score on the test mare. Capillary values are at tile level.}
\label{tab:supp-models}
\setlength{\tabcolsep}{4pt}
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
\begin{tabular}{lccccccc}
\hline
Model & Params (M) & Epochs & Batch & Validation AP & Test AP & Test F1 & Finalist \\
\hline
""" + "\n".join(big) + "\n\\end{tabular}%\n}\n\\end{table}\n"
mb, cc = RECIPES["micro_bench"], RECIPES["capilar_champion"]
pr = lambda v: f"{v:g}"  # noqa: E731
spec = [("Initialization", "COCO weights", "COCO weights"), ("Input size (px)", "$640 \\times 640$", "$640 \\times 640$"),
        ("Maximum epochs", "120", "60"), ("Early stopping patience (epochs)", "30", "30"), ("Batch size", "8", "8"), ("Optimizer", "automatic", "automatic"),
        ("Mosaic probability", pr(mb["mosaic"]), pr(cc["mosaic"])), ("Mosaic off in the last epochs", pr(mb["close_mosaic"]), pr(cc["close_mosaic"])),
        ("Mixup probability", pr(mb["mixup"]), "--"), ("Copy-paste probability", "--", pr(cc["copy_paste"])),
        ("Rotation (degrees)", f"$\\pm${mb['degrees']:g}", f"$\\pm${cc['degrees']:g}"),
        ("Translation (fraction)", pr(mb["translate"]), pr(cc["translate"])), ("Scale (fraction)", pr(mb["scale"]), pr(cc["scale"])),
        ("Flip probability (vertical / horizontal)", f"{mb['flipud']:g} / {mb['fliplr']:g}", f"{cc['flipud']:g} / {cc['fliplr']:g}"),
        ("HSV jitter (hue / saturation / value)", f"{mb['hsv_h']:g} / {mb['hsv_s']:g} / {mb['hsv_v']:g}", f"{cc['hsv_h']:g} / {cc['hsv_s']:g} / {cc['hsv_v']:g}"),
        ("Random erasing probability", pr(mb["erasing"]), pr(cc["erasing"])), ("RandAugment", "yes", "yes"),
        ("Overlapping masks", "yes" if mb["overlap_mask"] else "no", "yes" if cc["overlap_mask"] else "no"),
        ("Mask ratio", pr(mb["mask_ratio"]), pr(cc["mask_ratio"])), ("Maximum detections", "300", pr(cc["max_det"]))]
T2 = r"""\begin{table}[H]
\centering
\caption{Training settings of the YOLO models (Ultralytics), which were the same for all YOLO models of a task. The batch size was 8 and was reduced
automatically when the GPU memory was insufficient.}
\label{tab:supp-training}
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
\begin{tabular}{lcc}
\hline
Setting & Microcotyledons & Capillaries \\
\hline
""" + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in spec) + "\n\\hline\n\\end{tabular}%\n}\n\\end{table}\n"
SUP.write_text("% Tabelas para o material suplementar (geradas por scripts/artigo_iser/tabelas_suplementares.py); fora do texto principal.\n\n" + T1 + "\n" + T2,
               encoding="utf-8")
print("suplemento ok:", SUP)
