"""Resultados dos 6 folds (test = cada egua uma vez) dos modelos adotados: micro YOLO26s (campo inteiro) e capilar YOLO11s (campo, SAHI).

Le benchmark_results (results.csv e perimage_test.json) e grava:
  artigo/figs/fig_resultados_concordancia.png   dispersao GT x predito por campo (area e contagem; micro e capilar), cor = egua
  artigo/figs/fig_resultados_por_egua.png       F1 e erro de area por egua de teste
  temp_ml/artigo_iser/resultados_seisfolds.json valores usados no texto (para conferencia)
A area usada e a da UNIAO das mascaras (Metodos, Evaluation metrics). Conversao px -> um2: micro 0.3557 (canvas 640x640 esticado), capilar 0.01137.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "D:/projeto_placentas_clayton/benchmark_results/"
FIG = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\artigo\figs")
OUTJ = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\resultados_seisfolds.json")
F_MICRO = (50 / 469) ** 2 * (4140 / 640) * (3096 / 640)
F_CAP = (50 / 469) ** 2
FIELD_UM2 = 4140 * 3096 * (50 / 469) ** 2
MARES = {"ROSILHA-M": 1, "TORDILHA": 2, "TP": 3, "ZAINA-679": 4, "ZAZA": 5, "TOSTADA": 6}
PRIMARY_TEST = "ZAZA"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cc2", "#0b0b0b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK})


def mare_of(fold):
    return PRIMARY_TEST if fold == "primary" else fold.split("_val-")[0].replace("test-", "")


def ccc(x, y):
    mx, my = x.mean(), y.mean()
    return 2 * np.mean((x - mx) * (y - my)) / (x.var() + y.var() + (mx - my) ** 2)


d = pd.read_csv(R + "results.csv")
d = d[d.norm == "none"]
runs = {"micro": d[(d.task == "micro") & (d.model == "yolo26s-seg.pt")], "capilar": d[(d.task == "capilar") & (d.model == "yolo11s-seg.pt")]}
fields = []
for task, df in runs.items():
    assert len(df) == 6
    for _, r in df.iterrows():
        p = json.load(open(R + r.run_id + "/perimage_test.json"))
        rows = p["image"]["rows"] if task == "micro" else p["field"]["rows"]
        assert len(rows) == 30, (task, r.fold, len(rows))
        f = F_MICRO if task == "micro" else F_CAP
        for i, row in enumerate(rows):
            fields.append(dict(task=task, mare=mare_of(r.fold), idx=i, gt_area=row["area_gt_union"] * f, pr_area=row["area_pred_union"] * f,
                               gt_n=row["n_gt"], pr_n=row["n_pred"], tp=row["tp"], fp=row["fp"], fn=row["fn"]))
F = pd.DataFrame(fields)
F["mare_n"] = F.mare.map(MARES)
F["gt_frac"] = F.gt_area / FIELD_UM2 * 100
F["pr_frac"] = F.pr_area / FIELD_UM2 * 100

out = {"checks": {}, "per_structure": {}, "per_mare": {}}
# conferencia com results.csv (erro de area por fold e CCC)
for task, df in runs.items():
    col = "test_report_morphometry" if task == "micro" else "test_field_morphometry"
    for _, r in df.iterrows():
        s = F[(F.task == task) & (F.mare == mare_of(r.fold))]
        mine = float((abs(s.pr_area - s.gt_area) / s.gt_area).mean())
        out["checks"][f"{task}_{mare_of(r.fold)}"] = {"erro_area_meu": round(mine, 4), "erro_area_csv": round(float(r[col + "_area_union_mean_abs_rel_error"]), 4),
                                                       "ccc_meu": round(float(ccc(s.gt_area.values, s.pr_area.values)), 3), "ccc_csv": round(float(r[col + "_area_union_ccc"]), 3)}

summ = {}
for task, df in runs.items():
    s = F[F.task == task]
    rep = "test_report" if task == "micro" else "test_field"
    per_fold = df.assign(mare=df.fold.map(mare_of)).set_index("mare")
    area_err = (abs(s.pr_area - s.gt_area) / s.gt_area)
    cnt_err = abs(s.pr_n - s.gt_n)
    da, dn = s.pr_area - s.gt_area, s.pr_n - s.gt_n
    summ[task] = dict(
        n_fields=int(len(s)),
        gt_area_mean=float(s.gt_area.mean()), pr_area_mean=float(s.pr_area.mean()),
        gt_n_mean=float(s.gt_n.mean()), pr_n_mean=float(s.pr_n.mean()),
        gt_frac_mean=float(s.gt_frac.mean()), pr_frac_mean=float(s.pr_frac.mean()),
        area_ccc=float(ccc(s.gt_area.values, s.pr_area.values)), area_r=float(np.corrcoef(s.gt_area, s.pr_area)[0, 1]),
        area_bias=float(da.mean()), area_loa=[float(da.mean() - 1.96 * da.std()), float(da.mean() + 1.96 * da.std())],
        area_bias_pct=float((da / s.gt_area).mean() * 100),
        area_err_mean_pct=float(area_err.mean() * 100), area_err_median_pct=float(area_err.median() * 100),
        count_ccc=float(ccc(s.gt_n.values.astype(float), s.pr_n.values.astype(float))), count_r=float(np.corrcoef(s.gt_n, s.pr_n)[0, 1]),
        count_bias=float(dn.mean()), count_loa=[float(dn.mean() - 1.96 * dn.std()), float(dn.mean() + 1.96 * dn.std())],
        count_err_mean=float(cnt_err.mean()), count_err_median=float(cnt_err.median()),
        # media +- DP entre as 6 egua de teste (metricas por fold do benchmark)
        ap=[float(per_fold[rep + "_coco_ap_ap"].mean()), float(per_fold[rep + "_coco_ap_ap"].std())] if task == "micro" else None,
        ap50=[float(per_fold[rep + "_coco_ap_ap50"].mean()), float(per_fold[rep + "_coco_ap_ap50"].std())] if task == "micro" else None,
    )
    mets = {"f1": "_instance_f1", "dice": "_pixel_dice_pooled", "precision": "_instance_precision", "recall": "_instance_recall"}
    fcol = "test_report" if task == "micro" else "test_field"
    for k, suf in mets.items():
        col = fcol + suf
        summ[task][k] = [float(per_fold[col].mean()), float(per_fold[col].std())]
    # por egua
    pm = {}
    for m in MARES:
        sm = s[s.mare == m]
        ae = abs(sm.pr_area - sm.gt_area) / sm.gt_area
        pm[m] = dict(mare_n=MARES[m], f1=float(per_fold.loc[m, fcol + "_instance_f1"]), dice=float(per_fold.loc[m, fcol + "_pixel_dice_pooled"]),
                     area_err_pct=float(ae.mean() * 100), count_err=float(abs(sm.pr_n - sm.gt_n).mean()),
                     area_ccc=float(ccc(sm.gt_area.values, sm.pr_area.values)), count_ccc=float(ccc(sm.gt_n.values.astype(float), sm.pr_n.values.astype(float))),
                     gt_n=float(sm.gt_n.mean()), pr_n=float(sm.pr_n.mean()))
    out["per_mare"][task] = pm
    # tp/fp/fn totais
    summ[task]["tp_fp_fn"] = [int(s.tp.sum()), int(s.fp.sum()), int(s.fn.sum())]
    summ[task]["n_gt_total"], summ[task]["n_pred_total"] = int(s.gt_n.sum()), int(s.pr_n.sum())
out["per_structure"] = summ

# fracao capilar/micro (area capilar / area micro, por campo)
mi, ca = F[F.task == "micro"].reset_index(drop=True), F[F.task == "capilar"].reset_index(drop=True)
assert (mi.mare == ca.mare).all() and (mi.idx == ca.idx).all()
gt_fr, pr_fr = ca.gt_area / mi.gt_area * 100, ca.pr_area / mi.pr_area * 100
out["capillary_fraction"] = dict(gt_mean=float(gt_fr.mean()), pr_mean=float(pr_fr.mean()), ccc=float(ccc(gt_fr.values, pr_fr.values)),
                                 r=float(np.corrcoef(gt_fr, pr_fr)[0, 1]), abs_err_pts=float(abs(pr_fr - gt_fr).mean()))
OUTJ.write_text(json.dumps(out, indent=1), encoding="utf-8")

# ------------------------------------------------------------------ figura 1: dispersao GT x predito
fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.4), dpi=300)
panels = [("micro", "area", "Microcotyledons, area (10$^3$ $\\mu$m$^2$)", 1e-3), ("capilar", "area", "Capillaries, area (10$^3$ $\\mu$m$^2$)", 1e-3),
          ("micro", "n", "Microcotyledons, profiles per field", 1), ("capilar", "n", "Capillaries, profiles per field", 1)]
for ax, (task, kind, title, sc) in zip(axes.ravel(), panels):
    s = F[F.task == task]
    x = (s.gt_area if kind == "area" else s.gt_n).values * sc
    y = (s.pr_area if kind == "area" else s.pr_n).values * sc
    for m, n in MARES.items():
        sm = s.mare == m
        ax.scatter(x[sm.values], y[sm.values], s=11, color=COLORS[n - 1], edgecolor="white", linewidth=.4, zorder=3, label=f"Mare {n}")
    lim = [0, max(x.max(), y.max()) * 1.05]
    ax.plot(lim, lim, color=INK2, lw=.8, ls="--", zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=.6, zorder=0)
    c = ccc(x, y)
    ax.text(.04, .93, f"CCC = {c:.2f}", transform=ax.transAxes, fontsize=7.5, va="top")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=4)
    ax.set_xlabel("Manual annotation")
    ax.set_ylabel("Model")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
for ax, letter in zip(axes.ravel(), "ABCD"):
    ax.text(-.2, 1.1, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, -.005), handletextpad=.2, columnspacing=1.0)
fig.subplots_adjust(left=.1, right=.97, top=.95, bottom=.1, hspace=.38, wspace=.3)
fig.savefig(FIG / "fig_resultados_concordancia.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ figura 2: por egua de teste
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), dpi=300)
for ax, key, ylab in ((axes[0], "f1", "F1-score"), (axes[1], "area_err_pct", "Area error (%)")):
    for task, color, lab, off in (("micro", "#2a78d6", "Microcotyledons", -.1), ("capilar", "#eda100", "Capillaries", .1)):
        pm = out["per_mare"][task]
        xs = np.arange(1, 7) + off
        ys = [pm[m][key] for m in MARES]
        ax.scatter(xs, ys, s=22, color=color, edgecolor=INK, linewidth=.6, zorder=3, label=lab)
        ax.plot(xs, ys, color=color, lw=.8, alpha=.5, zorder=2)
    ax.set_xticks(range(1, 7), [f"Mare {n}" for n in range(1, 7)], fontsize=7)
    ax.set_ylabel(ylab)
    ax.grid(axis="y", color=GRID, lw=.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
axes[0].set_ylim(0, 1)
axes[1].set_ylim(bottom=0)
for ax, letter in zip(axes, "AB"):
    ax.text(-.2, 1.08, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, -.01))
fig.subplots_adjust(left=.09, right=.985, top=.88, bottom=.27, wspace=.28)
fig.savefig(FIG / "fig_resultados_por_egua.png", dpi=300)
plt.close(fig)
print("ok", OUTJ)
