"""Figuras do Capitulo 4 (Analise de Resultados) da monografia, em portugues e fundo branco, geradas dos dados brutos.

Uso:  python figuras_cap4.py [nome ...]      (sem argumentos gera todas)

  ranking      cap4_ranking_variantes.png      AP de validacao e de teste de todos os modelos do fold principal (2 tarefas)
  selecao      cap4_selecao_seis_folds.png     finalistas nas 6 rotacoes e a regra do 1 erro-padrao
  egua         cap4_desempenho_por_egua.png    F1 e erro de area por egua de teste (finalistas)
  capilar      cap4_capilar_contagem_egua.png  capilares anotados x preditos por egua (a TOSTADA)
  normaliz     cap4_normalizacao_delta.png     variacao do F1 de cada metodo de normalizacao vs sem normalizacao
  concord      cap4_concordancia.png           dispersao anotacao x modelo por campo (area e contagem)
  blandaltman  cap4_bland_altman.png           diferenca x media por campo, vies e limites de concordancia
  tempos       cap4_tempos_inferencia.png      tempo de inferencia FP32/FP16 e memoria da GPU
  eficiencia   cap4_eficiencia.png             tempo manual de anotacao x tempo do pipeline por campo
  exemplos     cap4_exemplos.png               campos anotados x preditos (egua tipica, TOSTADA, zoom)

Area 'soma' = soma das areas das instancias (definicao do Cap. 3). Calibracao: micro (50/469)^2*(4140/640)*(3096/640),
capilar (50/469)^2. Cores iguais as do Cap. 3 (azul = microcotiledones, amarelo = capilares).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FIG = Path(r"D:\Downloads\projeto_tcc_enzo_template_correto\graduacao\figs")
R = Path(r"D:\projeto_placentas_clayton\benchmark_results")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PURPLE = "#8a5cc2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
MARES = ["ROSILHA-M", "TORDILHA", "TP", "ZAINA-679", "ZAZA", "TOSTADA"]
MCOL = [BLUE, ORANGE, AQUA, YELLOW, PURPLE, INK]
F_MICRO = (50 / 469) ** 2 * (4140 / 640) * (3096 / 640)
F_CAP = (50 / 469) ** 2
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
FIG.mkdir(parents=True, exist_ok=True)
VIRG = lambda x, n=2: f"{x:.{n}f}".replace(".", ",")  # noqa: E731  virgula decimal

NOME = {"yolov8n-seg.pt": "YOLOv8n", "yolov8s-seg.pt": "YOLOv8s", "yolov8m-seg.pt": "YOLOv8m", "yolov9c-seg.pt": "YOLOv9c",
        "yolo11n-seg.pt": "YOLO11n", "yolo11s-seg.pt": "YOLO11s", "yolo11m-seg.pt": "YOLO11m", "yolo26n-seg.pt": "YOLO26n",
        "yolo26s-seg.pt": "YOLO26s", "yolo26m-seg.pt": "YOLO26m", "maskrcnn_resnet50_fpn": "Mask R-CNN v1",
        "maskrcnn_resnet50_fpn_v2": "Mask R-CNN v2", "rfdetr-seg-nano": "RF-DETR-Seg Nano"}


def clean(ax, grid="x"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.6, zorder=0)
    ax.set_axisbelow(True)


def letra(ax, t, dx=-.02, dy=1.04):
    ax.text(dx, dy, t, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right")


def load():
    d = pd.read_csv(R / "results.csv")
    d["mare"] = d.fold.map(lambda f: "ZAZA" if f == "primary" else f.split("_val-")[0].replace("test-", ""))
    return d


D = load()
BASE = D[D.norm == "none"]


def six(task, model):
    s = BASE[(BASE.task == task) & (BASE.model == model)].set_index("mare").loc[MARES]
    assert len(s) == 6
    return s


def virg_axis(ax, which="x", nd=2):
    fm = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.{nd}f}".replace(".", ","))
    (ax.xaxis if which == "x" else ax.yaxis).set_major_formatter(fm)


def stats_norm():
    """Bootstrap pareado (2000 reamostragens dos 30 campos) de cada normalizacao contra o MESMO modelo sem normalizacao,
    ambos no fold principal. Mesma regra de benchmark/report_stats.py (seed 0, p bicaudal, Holm sobre todas as comparacoes
    de uma tarefa), mas com a linha de base correta: report_stats.py usa a ultima execucao 'none' do modelo, que para os modelos
    de seis rotacoes e a de outra egua."""
    from benchmark.report_stats import stat_values, holm, STATS
    cache = Path("D:/projeto_placentas_clayton/temp_ml/tcc/stats_norm_primary.csv")
    if cache.exists():
        return pd.read_csv(cache)
    prim = D[D.fold == "primary"]
    idx = np.random.default_rng(0).integers(0, 30, (2000, 30))
    rowsof, boots = {}, {}
    for _, r in prim.iterrows():
        f = R / r.run_id / "perimage_test.json"
        if not f.exists():
            continue
        pj = json.load(open(f))
        part = pj.get("field") or pj["image"]
        df = pd.DataFrame(part["rows"])
        if len(df) != 30:
            continue
        rowsof[r.run_id] = df
        boots[r.run_id] = pd.DataFrame([stat_values(df, i) for i in idx])
    out = []
    for task in ("micro", "capilar"):
        sub = []
        for _, r in prim[(prim.task == task) & (prim.norm != "none")].iterrows():
            b = prim[(prim.task == task) & (prim.model == r.model) & (prim.norm == "none")]
            if r.run_id not in boots or b.empty or b.iloc[0].run_id not in boots:
                continue
            b0, b1 = boots[b.iloc[0].run_id], boots[r.run_id]
            for st in STATS:
                dd = b1[st].to_numpy() - b0[st].to_numpy()
                pv = max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1 / 2000)
                sub.append(dict(task=task, model=r.model, norm=r.norm, stat=st,
                                diff=stat_values(rowsof[r.run_id])[st] - stat_values(rowsof[b.iloc[0].run_id])[st],
                                ci_lo=np.percentile(dd, 2.5), ci_hi=np.percentile(dd, 97.5), p=pv))
        t = pd.DataFrame(sub)
        t["p_holm"] = holm(t.p.to_numpy())
        out.append(t)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(cache, index=False)
    return res


# ------------------------------------------------------------------ ranking dos modelos no fold principal
def fig_ranking():
    prim = BASE[BASE.fold == "primary"]
    finalistas = {"micro": {"yolo26m-seg.pt", "yolo26s-seg.pt", "yolo11m-seg.pt"}, "capilar": {"yolo11s-seg.pt", "yolo26s-seg.pt"}}
    depois = {"micro": {"maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2", "rfdetr-seg-nano"},
              "capilar": {"maskrcnn_resnet50_fpn", "yolov8m-seg.pt", "yolov9c-seg.pt", "yolo11m-seg.pt", "yolo26m-seg.pt"}}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.4), dpi=300, gridspec_kw={"width_ratios": [1, 1]})
    for ax, task, cor, tit in ((axes[0], "micro", BLUE, "Microcotilédones"), (axes[1], "capilar", YELLOW, "Capilares")):
        s = prim[prim.task == task].copy()
        s["nome"] = s.model.map(NOME)
        if task == "capilar":    # as duas execucoes da Mask R-CNN (20 e 40 epocas)
            m = s.model == "maskrcnn_resnet50_fpn"
            s.loc[m, "nome"] = [f"Mask R-CNN v1 ({int(e)} ép.)" for e in s.loc[m, "train_epochs_run"]]
        s = s.sort_values("val_report_coco_ap_ap", ascending=True).reset_index(drop=True)
        y = np.arange(len(s))
        for i, r in s.iterrows():
            if r.model in finalistas[task]:
                ax.axhspan(i - .45, i + .45, color="#fdf6e3" if task == "capilar" else "#eef3fb", zorder=0)
        ax.scatter(s.test_report_coco_ap_ap, y, s=26, facecolor="white", edgecolor=AQUA, linewidth=1.3, zorder=3, label="Teste (ZAZA)")
        ax.scatter(s.val_report_coco_ap_ap, y, s=26, color=ORANGE, zorder=4, label="Validação (TOSTADA)")
        lab = []
        for _, r in s.iterrows():
            t = r.nome
            if r.model in finalistas[task]:
                t += " ★"
            elif r.model in depois[task]:
                t += " †"
            lab.append(t)
        ax.set_yticks(y)
        ax.set_yticklabels(lab, fontsize=7.4)
        ax.set_ylim(-.7, len(s) - .3)
        lo = min(s.val_report_coco_ap_ap.min(), s.test_report_coco_ap_ap.min())
        hi = max(s.val_report_coco_ap_ap.max(), s.test_report_coco_ap_ap.max())
        ax.set_xlim(lo - .03, hi + .02)
        ax.set_xlabel("AP de máscara" + (" (por ladrilho)" if task == "capilar" else ""), fontsize=7.6)
        virg_axis(ax, "x", 2)
        ax.set_title(tit, loc="left", fontsize=8.8, fontweight="bold", color=cor if cor != YELLOW else "#a87500")
        clean(ax, "x")
    h = [Line2D([], [], marker="o", color=ORANGE, ls="", ms=5, label="Validação (égua TOSTADA)"),
         Line2D([], [], marker="o", mfc="white", mec=AQUA, mew=1.3, ls="", ms=5, label="Teste (égua ZAZA)"),
         Rectangle((0, 0), 1, 1, fc="#eef3fb", ec=INK2, lw=.4, label="★ finalista"),
         Line2D([], [], ls="", label="† treinado depois da escolha dos finalistas")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7.4, bbox_to_anchor=(.5, -.005))
    fig.subplots_adjust(left=.185, right=.985, top=.95, bottom=.17, wspace=.62)
    fig.savefig(FIG / "cap4_ranking_variantes.png", dpi=300)
    plt.close(fig)
    print("ranking ok")


# ------------------------------------------------------------------ finalistas nas seis rotacoes e a regra do 1 erro-padrao
def fig_selecao():
    spec = {"micro": (["yolo26m-seg.pt", "yolo26s-seg.pt", "yolo11m-seg.pt"], "yolo26s-seg.pt", BLUE, "Microcotilédones"),
            "capilar": (["yolo11s-seg.pt", "yolo26s-seg.pt"], "yolo11s-seg.pt", YELLOW, "Capilares")}
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.0), dpi=300, gridspec_kw={"height_ratios": [3, 2]})
    out = {}
    for ax, (task, (mods, adot, cor, tit)) in zip(axes, spec.items()):
        vals = {m: six(task, m).val_report_coco_ap_ap.values for m in mods}
        par = {m: float(six(task, m).model_params_millions.iloc[0]) for m in mods}
        best = max(mods, key=lambda m: vals[m].mean())
        se = vals[best].std(ddof=1) / np.sqrt(6)
        thr = vals[best].mean() - se
        out[task] = dict(best=best, se=se, thr=thr, mean={m: vals[m].mean() for m in mods}, sd={m: vals[m].std(ddof=1) for m in mods}, params=par)
        y = np.arange(len(mods))[::-1]
        ax.axvspan(thr, vals[best].mean() + .12, color="#e8f6f0", zorder=0)
        ax.axvline(thr, color=AQUA, lw=1.1, ls=(0, (4, 2)), zorder=1)
        for yi, m in zip(y, mods):
            v = vals[m]
            ax.scatter(v, np.full(6, yi), s=12, color=INK2, alpha=.45, zorder=3, edgecolor="none")
            ax.errorbar(v.mean(), yi, xerr=v.std(ddof=1), fmt="D", ms=6.5, color=cor if cor != YELLOW else "#a87500", ecolor=INK, elinewidth=1.1,
                        capsize=3, zorder=5, mec=INK, mew=.8)
        ax.set_yticks(y)
        ax.set_yticklabels([NOME[m] + (" ★" if m == adot else "") + f"\n{VIRG(par[m], 1)} M de parâmetros" for m in mods], fontsize=7.4)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.05))
        virg_axis(ax, "x", 2)
        for t, m in zip(ax.get_yticklabels(), mods):
            if m == adot:
                t.set_fontweight("bold")
        lo = min(six(task, m).val_report_coco_ap_ap.min() for m in mods)
        hi = max(six(task, m).val_report_coco_ap_ap.max() for m in mods)
        ax.set_xlim(lo - .03, hi + .03)
        ax.set_ylim(-.6, len(mods) - .4)
        ax.set_title(tit, loc="left", fontsize=8.8, fontweight="bold", color=cor if cor != YELLOW else "#a87500")
        ax.text(thr, 1.02, "limite: melhor média − 1 erro-padrão", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.8, color="#0f7a55")
        clean(ax, "x")
    axes[1].set_xlabel("AP de máscara na validação (pontos: seis éguas; losango: média ± desvio-padrão)", fontsize=7.4)
    fig.subplots_adjust(left=.235, right=.985, top=.93, bottom=.12, hspace=.7)
    fig.savefig(FIG / "cap4_selecao_seis_folds.png", dpi=300)
    plt.close(fig)
    json.dump(out, open(r"D:\projeto_placentas_clayton\temp_ml\tcc\cap4_selecao.json", "w"), indent=1, default=float)
    print("selecao ok", {k: (v["best"], round(v["se"], 4), round(v["thr"], 4)) for k, v in out.items()})


# ------------------------------------------------------------------ desempenho por egua
def _pm(task, m):
    pre = "test_report" if task == "micro" else "test_field"
    s = six(task, m)
    return dict(f1=s[pre + "_instance_f1"].values, ae=s[pre + "_morphometry_area_sum_mean_abs_rel_error"].values * 100,
                ce=s[pre + "_morphometry_count_mean_abs_error"].values)


def fig_egua():
    spec = {"micro": (["yolo26m-seg.pt", "yolo11m-seg.pt", "yolo26s-seg.pt"], "yolo26s-seg.pt", BLUE, "Microcotilédones"),
            "capilar": (["yolo26s-seg.pt", "yolo11s-seg.pt"], "yolo11s-seg.pt", YELLOW, "Capilares")}
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.3), dpi=300, sharex=True)
    x = np.arange(6)
    for j, (task, (mods, adot, cor, tit)) in enumerate(spec.items()):
        cc = cor if cor != YELLOW else "#a87500"
        for i, (key, yl) in enumerate((("f1", "F1"), ("ae", "Erro de área (%)"))):
            ax = axes[i, j]
            for m in mods:
                v = _pm(task, m)[key]
                if m == adot:
                    ax.plot(x, v, "-o", color=cor, lw=1.8, ms=5.5, mec=INK, mew=.7, zorder=4, label=NOME[m] + " (adotado)")
                else:
                    ax.plot(x, v, "-", color=INK2, lw=.9, alpha=.55, zorder=2, marker="o", ms=3, mec="none", label=NOME[m])
            if i == 0:
                ax.set_title(tit, loc="left", fontsize=8.8, fontweight="bold", color=cc)
                ax.set_ylim(.4 if task == "capilar" else .78, .95 if task == "micro" else .85)
                virg_axis(ax, "y", 2)
            else:
                ax.set_ylim(bottom=0)
            ax.set_ylabel(yl, fontsize=8)
            clean(ax, "y")
            if i == 0:
                ax.legend(frameon=False, fontsize=6.6, loc="lower left" if task == "capilar" else "lower right", handlelength=1.6, labelspacing=.25)
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(MARES, rotation=38, ha="right", fontsize=7)
    for ax, t in zip(axes.ravel(), "ACBD"):
        letra(ax, t, dx=-.13, dy=1.05)
    fig.subplots_adjust(left=.1, right=.985, top=.95, bottom=.15, hspace=.12, wspace=.3)
    fig.savefig(FIG / "cap4_desempenho_por_egua.png", dpi=300)
    plt.close(fig)
    print("egua ok")


# ------------------------------------------------------------------ capilares anotados x preditos por egua (TOSTADA)
def fig_capilar():
    s = six("capilar", "yolo11s-seg.pt")
    rows = []
    for m in MARES:
        p = json.load(open(R / s.loc[m, "run_id"] / "perimage_test.json"))["field"]["rows"]
        r = pd.DataFrame(p)
        rows.append(dict(mare=m, anot=r.n_gt.mean(), pred=r.n_pred.mean(), rec=r.tp.sum() / (r.tp.sum() + r.fn.sum()), prec=r.tp.sum() / (r.tp.sum() + r.fp.sum())))
    t = pd.DataFrame(rows)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.7), dpi=300, gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(6)
    a.bar(x - .2, t.anot, .38, color="#d9d8d2", edgecolor=INK2, lw=.6, label="Anotados", zorder=3)
    a.bar(x + .2, t.pred, .38, color=YELLOW, edgecolor=INK, lw=.6, label="Detectados pelo modelo", zorder=3)
    for xi, g, p_ in zip(x, t.anot, t.pred):
        a.text(xi + .2, p_ + 3, f"{p_:.0f}".replace(".", ","), ha="center", fontsize=6.6)
        a.text(xi - .2, g + 3, f"{g:.0f}".replace(".", ","), ha="center", fontsize=6.6, color=INK2)
    a.set_xticks(x)
    a.set_xticklabels(MARES, rotation=35, ha="right", fontsize=7)
    a.set_ylabel("Capilares por campo (média)")
    a.legend(frameon=False, fontsize=7, loc="upper left", ncol=1)
    a.set_ylim(0, t.anot.max() * 1.3)
    clean(a, "y")
    letra(a, "A", dx=-.12)
    b.scatter(t.rec, t.prec, s=36, color=[c for c in MCOL], edgecolor=INK, linewidth=.6, zorder=3)
    off = {"ROSILHA-M": (-5, 5, "right"), "TORDILHA": (5, 4, "left"), "TP": (4, -11, "left"), "ZAINA-679": (-5, 5, "right"),
           "ZAZA": (5, 4, "left"), "TOSTADA": (5, 4, "left")}
    for _, r in t.iterrows():
        dx, dy, ha = off[r.mare]
        b.annotate(r.mare, (r.rec, r.prec), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=6.6)
    virg_axis(b, "x", 1)
    virg_axis(b, "y", 2)
    b.set_xlabel("Revocação")
    b.set_ylabel("Precisão")
    b.set_xlim(.35, .85)
    b.set_ylim(.6, .86)
    clean(b, "both")
    letra(b, "B", dx=-.2)
    fig.subplots_adjust(left=.09, right=.985, top=.9, bottom=.25, wspace=.32)
    fig.savefig(FIG / "cap4_capilar_contagem_egua.png", dpi=300)
    plt.close(fig)
    print("capilar ok", t.round(3).to_dict("records")[-1])


# ------------------------------------------------------------------ normalizacao: delta de F1
def fig_normaliz():
    st = stats_norm()
    st = st[st.stat == "f1"]
    nm = {"whitebalance": "Balanço\nde branco", "grayscale": "Escala\nde cinza", "histmatch": "Histmatch", "reinhard": "Reinhard",
          "macenko": "Macenko", "vahadane": "Vahadane"}
    order = ["whitebalance", "grayscale", "histmatch", "reinhard", "macenko", "vahadane"]
    spec = {"micro": ["yolo26s-seg.pt", "yolo26m-seg.pt", "yolo11m-seg.pt"], "capilar": ["yolo11n-seg.pt", "yolo11s-seg.pt"]}
    prim = BASE[BASE.fold == "primary"]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.2), dpi=300, gridspec_kw={"height_ratios": [3, 2]})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [ORANGE, "#f1f0ec", BLUE])
    norm_ = matplotlib.colors.TwoSlopeNorm(vmin=-.18, vcenter=0, vmax=.03)
    rep = {}
    for ax, (task, mods), tit in zip(axes, spec.items(), ("Microcotilédones", "Capilares")):
        M = np.full((len(mods), 6), np.nan)
        P = np.full((len(mods), 6), np.nan)
        base_f1 = []
        for i, m in enumerate(mods):
            pre = "test_report" if task == "micro" else "test_field"
            base_f1.append(float(prim[(prim.task == task) & (prim.model == m)][pre + "_instance_f1"].iloc[0]))
            for k, n in enumerate(order):
                r = st[(st.task == task) & (st.model == m) & (st.norm == n)]
                if len(r):
                    M[i, k], P[i, k] = float(r["diff"].iloc[0]), float(r["p_holm"].iloc[0])
                else:    # sem teste pareado: diferenca direta de F1 pelo results.csv
                    rr = D[(D.task == task) & (D.model == m) & (D.norm == n) & (D.fold == "primary")]
                    if len(rr):
                        M[i, k] = float(rr[pre + "_instance_f1"].iloc[0]) - base_f1[-1]
        ax.imshow(M, cmap=cmap, norm=norm_, aspect="auto")
        for i in range(len(mods)):
            for k in range(6):
                if np.isnan(M[i, k]):
                    continue
                sig = (not np.isnan(P[i, k])) and P[i, k] < .05
                sem = np.isnan(P[i, k])
                txt = ("+" if M[i, k] > 0 else "−" if M[i, k] < 0 else "") + VIRG(abs(M[i, k]), 3) + ("*" if sig else "") + ("†" if sem else "")
                ax.text(k, i, txt, ha="center", va="center", fontsize=7.2, fontweight="bold" if sig else "normal",
                        color="white" if M[i, k] < -.1 else INK)
        ax.set_xticks(range(6))
        ax.set_xticklabels([nm[n] for n in order], fontsize=7.4)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(mods)))
        ax.set_yticklabels([f"{NOME[m]}  (F1 {VIRG(b, 3)})" for m, b in zip(mods, base_f1)], fontsize=7.6)
        ax.set_title(tit, loc="left", fontsize=8.8, fontweight="bold", color=BLUE if task == "micro" else "#a87500", pad=24 if task == "micro" else 4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(mods), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.6)
        ax.tick_params(which="minor", length=0)
        if task == "capilar":
            ax.xaxis.set_visible(False)
        rep[task] = M
    fig.subplots_adjust(left=.27, right=.985, top=.86, bottom=.1, hspace=.5)
    fig.savefig(FIG / "cap4_normalizacao_delta.png", dpi=300)
    plt.close(fig)
    print("normaliz ok")


# ------------------------------------------------------------------ campos (6 folds) dos modelos adotados
def campos(task, model):
    f = F_MICRO if task == "micro" else F_CAP
    rows = []
    s = six(task, model)
    for m in MARES:
        p = json.load(open(R / s.loc[m, "run_id"] / "perimage_test.json"))
        rr = p["image"]["rows"] if task == "micro" else p["field"]["rows"]
        for x in rr:
            rows.append(dict(mare=m, n_gt=x["n_gt"], n_pr=x["n_pred"], a_gt=x["area_gt_sum"] * f, a_pr=x["area_pred_sum"] * f))
    return pd.DataFrame(rows)


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return 2 * np.mean((x - x.mean()) * (y - y.mean())) / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


PAIN = [("micro", "yolo26s-seg.pt", "a", "Microcotilédones, área (10³ µm²)", 1e-3), ("capilar", "yolo11s-seg.pt", "a", "Capilares, área (10³ µm²)", 1e-3),
        ("micro", "yolo26s-seg.pt", "n", "Microcotilédones, por campo", 1), ("capilar", "yolo11s-seg.pt", "n", "Capilares, por campo", 1)]


def fig_concord():
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.3), dpi=300)
    for ax, (task, mod, k, tit, sc), t in zip(axes.ravel(), PAIN, "ABCD"):
        C = campos(task, mod)
        x, y = C[f"{k}_gt"].values * sc, C[f"{k}_pr"].values * sc
        for m, c in zip(MARES, MCOL):
            sel = (C.mare == m).values
            ax.scatter(x[sel], y[sel], s=11, color=c, edgecolor="white", linewidth=.4, zorder=3, label=m)
        lim = [0, max(x.max(), y.max()) * 1.05]
        ax.plot(lim, lim, color=INK2, lw=.8, ls="--", zorder=2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.text(.04, .94, f"CCC = {VIRG(ccc(x, y))}\nr = {VIRG(np.corrcoef(x, y)[0, 1])}", transform=ax.transAxes, fontsize=7.6, va="top")
        ax.set_title(tit, loc="left", fontsize=8.2, fontweight="bold", pad=4)
        ax.set_xlabel("Anotação manual")
        ax.set_ylabel("Modelo")
        clean(ax, "both")
        letra(ax, t, dx=-.14, dy=1.08)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(.5, -.003), handletextpad=.1, columnspacing=.8)
    fig.subplots_adjust(left=.1, right=.97, top=.95, bottom=.09, hspace=.36, wspace=.3)
    fig.savefig(FIG / "cap4_concordancia.png", dpi=300)
    plt.close(fig)
    print("concord ok")


def fig_blandaltman():
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.4), dpi=300)
    for ax, (task, mod, k, tit, sc), t in zip(axes.ravel(), PAIN, "ABCD"):
        C = campos(task, mod)
        g, p = C[f"{k}_gt"].values * sc, C[f"{k}_pr"].values * sc
        mean, dif = (g + p) / 2, p - g
        b, s = dif.mean(), dif.std(ddof=1)
        for m, c in zip(MARES, MCOL):
            sel = (C.mare == m).values
            ax.scatter(mean[sel], dif[sel], s=11, color=c, edgecolor="white", linewidth=.4, zorder=3, label=m)
        ax.axhline(0, color=INK2, lw=.7, zorder=1)
        ax.axhline(b, color=INK, lw=1.1, zorder=2)
        for lv in (b - 1.96 * s, b + 1.96 * s):
            ax.axhline(lv, color=INK2, lw=.9, ls=(0, (4, 2)), zorder=2)
        xr = ax.get_xlim()[1]
        for lv, tx in ((b, f"viés {VIRG(b, 1 if sc == 1 else 1)}"), (b + 1.96 * s, f"+1,96 dp: {VIRG(b + 1.96 * s, 1)}"), (b - 1.96 * s, f"−1,96 dp: {VIRG(b - 1.96 * s, 1)}")):
            ax.text(xr, lv, tx, ha="right", va="bottom", fontsize=6.4, color=INK)
        ax.set_title(tit, loc="left", fontsize=8.2, fontweight="bold", pad=4)
        ax.set_xlabel("Média entre modelo e anotação")
        ax.set_ylabel("Modelo − anotação")
        clean(ax, "y")
        letra(ax, t, dx=-.14, dy=1.08)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(.5, -.003), handletextpad=.1, columnspacing=.8)
    fig.subplots_adjust(left=.11, right=.97, top=.95, bottom=.1, hspace=.4, wspace=.32)
    fig.savefig(FIG / "cap4_bland_altman.png", dpi=300)
    plt.close(fig)
    print("blandaltman ok")


# ------------------------------------------------------------------ tempos de inferencia
def fig_tempos():
    j = json.load(open(R / "inference_timing.json"))["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), dpi=300, gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, task, cor, tit, un, sc in ((axes[0], "micro", BLUE, "Microcotilédones (imagem 640 × 640)", "ms", 1e3),
                                       (axes[1], "capilar", YELLOW, "Capilares (campo nativo)", "s", 1)):
        rows = [r for r in j if r["task"] == task]
        x = np.arange(len(rows))
        for dx, key, c, lab in ((-.2, "fp32", cor, "FP32"), (.2, "fp16", "#d9d8d2", "FP16")):
            v = [r[key]["mean_s"] * sc for r in rows]
            ax.bar(x + dx, v, .38, color=c, edgecolor=INK, lw=.6, zorder=3, label=lab)
            for xi, vi in zip(x + dx, v):
                ax.text(xi, vi, VIRG(vi, 0 if sc == 1e3 else 1), ha="center", va="bottom", fontsize=6.8)
        ax.set_xticks(x)
        ax.set_xticklabels([NOME[r["model"]] + f"\n{r['fp32']['vram_peak_mb']:.0f} MB".replace(".", ",") for r in rows], fontsize=7.2)
        ax.set_ylabel(f"Tempo médio por unidade ({un})")
        ax.set_title(tit, loc="left", fontsize=8.2, fontweight="bold")
        ax.set_ylim(0, max(max(r["fp32"]["mean_s"], r["fp16"]["mean_s"]) for r in rows) * sc * 1.18)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        clean(ax, "y")
    fig.text(.01, .01, "Sob o nome de cada modelo: pico de memória da GPU em FP32.", fontsize=6.8, color=INK2)
    fig.subplots_adjust(left=.1, right=.985, top=.9, bottom=.2, wspace=.3)
    fig.savefig(FIG / "cap4_tempos_inferencia.png", dpi=300)
    plt.close(fig)
    print("tempos ok")


# ------------------------------------------------------------------ eficiencia: manual x pipeline
def fig_eficiencia():
    # tempos manuais medios por campo (s), nos 180 campos (Planilha completa.xlsx, 2026-11-24): micro 313,8; capilar 2103,4; total 2417,2
    man = {"Campo completo": 2417.2, "Capilares": 2103.4, "Microcotilédones": 313.8}
    j = json.load(open(R / "inference_timing.json"))["rows"]
    tm = next(r for r in j if r["task"] == "micro" and r["model"] == "yolo26s-seg.pt")["fp32"]["mean_s"]
    tc = next(r for r in j if r["task"] == "capilar" and r["model"] == "yolo11s-seg.pt")["fp32"]["mean_s"]
    pip = {"Campo completo": tm + tc, "Capilares": tc, "Microcotilédones": tm}
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=300)
    y = np.arange(3)[::-1]
    def fmt(s_):
        return (f"{s_ / 60:.1f} min" if s_ >= 60 else f"{s_:.1f} s" if s_ >= 1 else f"{s_ * 1000:.0f} ms").replace(".", ",")
    for yi, k in zip(y, man):
        ax.plot([pip[k], man[k]], [yi, yi], color=INK2, lw=1.2, zorder=2)
        ax.scatter(man[k], yi, s=70, marker="s", color="#d9d8d2", edgecolor=INK, linewidth=.9, zorder=4)
        ax.scatter(pip[k], yi, s=70, color=AQUA, edgecolor=INK, linewidth=.9, zorder=4)
        ax.text(man[k] * 1.25, yi, fmt(man[k]), va="center", ha="left", fontsize=7.4)
        ax.text(pip[k] / 1.25, yi, fmt(pip[k]), va="center", ha="right", fontsize=7.4)
        ax.text(np.sqrt(man[k] * pip[k]), yi + .17, f"≈ {int(float(format(man[k] / pip[k], '.2g'))):,} vezes".replace(",", "."), ha="center", va="bottom", fontsize=7.8,
                fontweight="bold", color="#0f7a55")
    ax.set_xscale("log")
    ax.set_xlim(.01, 4e4)
    ax.set_ylim(-.6, 2.7)
    tk = [.1, 1, 10, 60, 600, 3600]
    ax.set_xticks(tk)
    ax.set_xticklabels(["0,1 s", "1 s", "10 s", "1 min", "10 min", "1 h"], fontsize=7.4)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(y)
    ax.set_yticklabels(list(man), fontsize=8)
    ax.set_xlabel("Tempo por campo (escala logarítmica)", fontsize=7.8)
    ax.legend(handles=[Line2D([], [], marker="s", color="#d9d8d2", mec=INK, ls="", ms=7, label="Anotação manual"),
                       Line2D([], [], marker="o", color=AQUA, mec=INK, ls="", ms=7, label="Pipeline (só inferência)")],
              frameon=False, fontsize=7.4, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.02))
    clean(ax, "x")
    fig.subplots_adjust(left=.19, right=.985, top=.96, bottom=.17)
    fig.savefig(FIG / "cap4_eficiencia.png", dpi=300)
    plt.close(fig)
    print("eficiencia ok", {k: round(man[k] / pip[k]) for k in man})


# ------------------------------------------------------------------ exemplos qualitativos (predicoes fora da dobra)
def fig_exemplos():
    from scripts.artigo_iser import dataset_vs_modelo as V
    from scripts.artigo_iser import fluxo as Fx
    gt = pd.read_csv(V.OUT / "gt.csv")
    pr = pd.DataFrame([json.load(open(p, encoding="utf-8")) for p in sorted((V.OUT / "pred" / "OOF_none").glob("*.json"))])
    m = gt.merge(pr, on="lamina", suffixes=("_gt", "_pr"))
    m["erro_n"] = (m.capilar_n_pr - m.capilar_n_gt)
    # campos escolhidos: ZAZA com erro de contagem proximo da mediana da egua; TOSTADA idem (subestima, como a egua toda)
    pick = {}
    for eg in ("ZAZA", "TOSTADA"):
        s = m[m.egua_gt == eg]
        alvo = s.erro_n.median()
        pick[eg] = s.iloc[(s.erro_n - alvo).abs().argmin()]
        print(eg, pick[eg].lamina, "micro", int(pick[eg].micro_n_gt), "->", int(pick[eg].micro_n_pr), "| cap", int(pick[eg].capilar_n_gt), "->", int(pick[eg].capilar_n_pr))
    fields = {f["lamina"]: f for f in V.list_fields(None)}
    PW = 1000
    rows = []
    for eg in ("ZAZA", "TOSTADA"):
        r = pick[eg]
        fl = fields[r.lamina]
        img = cv2.imread(str(V.input_path(fl, "none")))
        H, W = img.shape[:2]
        z = np.load(V.OUT / "pred" / "OOF_none" / f"{r.lamina}.npz")
        gm, gc = V.gt_instances(fl)
        rows.append((eg, r, img, gm, gc, Fx.unpack(z, "micro"), Fx.unpack(z, "cap")))
    fig, axes = plt.subplots(3, 3, figsize=(6.4, 5.7), dpi=300)
    ds = 4140 / PW
    def painted(img, mi, ci, x0=0, y0=0, w=None, h=None, scale=1.0):
        w, h = w or img.shape[1], h or img.shape[0]
        crop = img[y0:y0 + h, x0:x0 + w]
        out = cv2.resize(crop, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        # _paint subtrai a origem (ox, oy) no espaco de origem e so depois aplica a escala: a origem do recorte vai em pixels nativos
        if mi is not None:
            Fx._paint(out, mi, (4140 / 640) * scale, (3096 / 640) * scale, x0 / (4140 / 640), y0 / (3096 / 640), _bgr(BLUE), .35)
        if ci is not None:
            Fx._paint(out, ci, scale, scale, x0, y0, _bgr(YELLOW), .6)
        return out
    titles = ["Campo original", "Anotação manual", "Modelo"]
    for i, (eg, r, img, gm, gc, pm, pc) in enumerate(rows):
        s = 1 / ds
        ims = [painted(img, None, None, scale=s), painted(img, gm, gc, scale=s), painted(img, pm, pc, scale=s)]
        for j, im in enumerate(ims):
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if i == 0:
                ax.set_title(titles[j], fontsize=8, pad=3)
        cnt = (None, f"{int(r.micro_n_gt)} micro, {int(r.capilar_n_gt)} cap.", f"{int(r.micro_n_pr)} micro, {int(r.capilar_n_pr)} cap.")
        for j in (1, 2):
            axes[i, j].text(.015, .02, cnt[j], transform=axes[i, j].transAxes, fontsize=6.8, va="bottom", color="white",
                            bbox=dict(boxstyle="square,pad=.2", fc="black", ec="none", alpha=.6))
        axes[i, 0].text(-.02, .5, f"{eg}\n({r.lamina.split('-B_')[0] if False else r.lamina})", transform=axes[i, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=7.2, fontweight="bold")
    # zoom: ladrilho de 1380 x 1032 da TOSTADA com mais capilares anotados
    eg, r, img, gm, gc, pm, pc = rows[1]
    best, bxy = -1, (0, 0)
    for ty in range(3):
        for tx in range(3):
            x0, y0 = tx * 1380, ty * 1032
            n = sum(1 for it in gc if x0 <= it.x0 + it.mask.shape[1] / 2 < x0 + 1380 and y0 <= it.y0 + it.mask.shape[0] / 2 < y0 + 1032)
            if n > best:
                best, bxy = n, (x0, y0)
    x0, y0 = bxy
    zs = 1000 / 1380
    zi = [painted(img, None, None, x0, y0, 1380, 1032, zs), painted(img, None, gc, x0, y0, 1380, 1032, zs), painted(img, None, pc, x0, y0, 1380, 1032, zs)]
    for j, im in enumerate(zi):
        ax = axes[2, j]
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    axes[2, 0].text(-.02, .5, "TOSTADA, ladrilho\nampliado (capilares)", transform=axes[2, 0].transAxes, rotation=90, va="center", ha="right",
                    fontsize=7.2, fontweight="bold")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=BLUE, alpha=.7, label="Microcotilédone"), Rectangle((0, 0), 1, 1, fc=YELLOW, alpha=.9, label="Capilar")],
               loc="lower center", ncol=2, frameon=False, fontsize=7.6, bbox_to_anchor=(.5, -.003))
    fig.subplots_adjust(left=.07, right=.995, top=.96, bottom=.045, wspace=.03, hspace=.06)
    fig.savefig(FIG / "cap4_exemplos.png", dpi=300)
    plt.close(fig)
    print("exemplos ok", (x0, y0), best)


def _bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


ALL = dict(ranking=fig_ranking, selecao=fig_selecao, egua=fig_egua, capilar=fig_capilar, normaliz=fig_normaliz, concord=fig_concord,
           blandaltman=fig_blandaltman, tempos=fig_tempos, eficiencia=fig_eficiencia, exemplos=fig_exemplos)

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(ALL)):
        ALL[k]()
