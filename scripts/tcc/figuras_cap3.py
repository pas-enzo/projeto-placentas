"""Figuras do Capitulo 3 (Materiais e Metodos) da monografia, em portugues e fundo branco, geradas dos dados.

Uso:  python figuras_cap3.py [nome ...]      (sem argumentos gera todas)

  fluxo       cap3_fluxo_geral.png        fluxograma das etapas do estudo
  composicao  cap3_composicao.png         microcotiledones e capilares por campo, por egua (substitui a tabela de composicao)
  rotacoes    cap3_rotacoes.png           esquema da validacao cruzada por egua
  selecao     cap3_selecao.png            protocolo de selecao de modelos em duas etapas
  normaliz    cap3_normalizacoes.png      o mesmo campo em cada metodo de normalizacao + mosaico de referencia
  curvas      cap3_curvas_treino.png      perdas e AP de validacao por epoca dos modelos adotados (fold principal)
  limiar      cap3_f1_limiar.png          F1 x limiar de confianca na validacao (6 folds)
  pipeline    cap3_pipeline.png           arquitetura do pipeline e da interface
Cores (paleta de referencia do guia dataviz): azul #2a78d6 (microcotiledones), amarelo #eda100 (capilares),
laranja #eb6834 (validacao), verde-agua #1baf7a (teste).
"""
import glob
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FIG = Path(r"D:\Downloads\projeto_tcc_enzo_template_correto\graduacao\figs")
BR = "D:/projeto_placentas_clayton/benchmark_results/"
DS = Path(r"D:\projeto_placentas_clayton\datasets_v4\test-ZAZA_val-TOSTADA")
CSV = r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\dataset_vs_modelo\lamina_anotada.csv"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
MARES = ["ROSILHA-M", "TORDILHA", "TP", "ZAINA-679", "ZAZA", "TOSTADA"]
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
FIG.mkdir(parents=True, exist_ok=True)
fmt = lambda x: f"{x:,.0f}".replace(",", ".")  # noqa: E731  milhar com ponto


def box(ax, x, y, w, h, title, body="", fc="#eef3fb", ec=BLUE, n=None, bs=6.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
    ty = y + h - 0.22
    if n is not None:
        ax.text(x + 0.13, ty, str(n), ha="left", va="top", fontsize=9, fontweight="bold", color=ec)
    ax.text(x + ((0.42 if n < 10 else 0.62) if n is not None else 0.13), ty, title, ha="left", va="top", fontsize=8.6, fontweight="bold")
    if body:
        ax.text(x + 0.13, ty - 0.42, body, ha="left", va="top", fontsize=bs, color=INK2, linespacing=1.35)


def arrow(ax, p0, p1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.1,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ fluxo geral
def fig_fluxo():
    W, H, bw, bh, gx, gy = 3.0, 1.55, 3.0, 1.55, 0.55, 0.62
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=300)
    ax.set_xlim(-0.1, 3 * bw + 2 * gx + 0.1)
    ax.set_ylim(-0.1, 4 * bh + 3 * gy + 0.1)
    ax.axis("off")
    ax.set_aspect("equal")
    c_dado, c_prep, c_mod, c_aval = ("#eef3fb", BLUE), ("#fdf6e3", YELLOW), ("#fdeee7", ORANGE), ("#e8f6f0", AQUA)
    spec = [  # (linha 0 = topo, coluna, titulo, corpo, cores)
        (0, 0, "Aquisição", "180 campos H&E, 20×\n6 éguas × 3 regiões × 10 campos\n4140 × 3096 px; 0,1066 µm/px", c_dado),
        (0, 1, "Anotação manual", "Polígonos no Roboflow\n5.411 microcotilédones\n36.400 capilares (referência)", c_dado),
        (0, 2, "Partição por égua", "Treino, validação e teste sem\nmistura de animais; 6 rotações\n(fold principal: teste ZAZA)", c_dado),
        (1, 2, "Vistas dos dados", "Microcotilédones: campo esticado\npara 640 × 640 px\nCapilares: 3 × 3 ladrilhos de\n1380 × 1032 px", c_prep),
        (1, 1, "Normalização de cor", "Opcional: 6 métodos, com a referência\nestimada só nas éguas de treino\n(vs. imagem original)", c_prep),
        (1, 0, "Treinamento", "YOLO (e Mask R-CNN, RF-DETR)\ncom pesos COCO; uma receita por\ntarefa, só o modelo muda", c_mod),
        (2, 0, "Seleção dos modelos", "Etapa 1: todos no fold principal\nEtapa 2: finalistas nas 6 rotações;\nregra do 1 erro-padrão", c_mod),
        (2, 1, "Calibração do limiar", "Limiar de confiança que maximiza\no F1 na égua de validação;\nfixado antes do teste", c_mod),
        (2, 2, "Inferência", "Microcotilédones: campo inteiro\nCapilares: SAHI, 16 janelas\ncom 20% de sobreposição", c_aval),
        (3, 2, "Medidas", "Área, perímetro, eixos,\ncircularidade, solidez; por campo:\ncontagem, área e ocupação", c_aval),
        (3, 1, "Métricas", "F1, Dice, AP, erro de área e de\ncontagem, CCC, Bland–Altman;\nbootstrap e médias entre éguas", c_aval),
        (3, 0, "Tempo de inferência", "De ponta a ponta, por campo,\nem FP32 e FP16; comparação\ncom o tempo de anotação", c_aval),
    ]
    pos = {}
    for i, (r, c, t, b, (fc, ec)) in enumerate(spec, 1):
        x = c * (bw + gx)
        y = (3 - r) * (bh + gy)
        box(ax, x, y, bw, bh, t, b, fc=fc, ec=ec, n=i)
        pos[i] = (x, y)
    mid = lambda i: (pos[i][0] + bw / 2, pos[i][1] + bh / 2)  # noqa: E731
    for i in (1, 2):                       # linha 1: esquerda -> direita
        arrow(ax, (pos[i][0] + bw, pos[i][1] + bh / 2), (pos[i + 1][0], pos[i + 1][1] + bh / 2))
    arrow(ax, (pos[3][0] + bw / 2, pos[3][1]), (pos[4][0] + bw / 2, pos[4][1] + bh))
    for i in (4, 5):                       # linha 2: direita -> esquerda
        arrow(ax, (pos[i][0], pos[i][1] + bh / 2), (pos[i + 1][0] + bw, pos[i + 1][1] + bh / 2))
    arrow(ax, (pos[6][0] + bw / 2, pos[6][1]), (pos[7][0] + bw / 2, pos[7][1] + bh))
    for i in (7, 8):
        arrow(ax, (pos[i][0] + bw, pos[i][1] + bh / 2), (pos[i + 1][0], pos[i + 1][1] + bh / 2))
    arrow(ax, (pos[9][0] + bw / 2, pos[9][1]), (pos[10][0] + bw / 2, pos[10][1] + bh))
    for i in (10, 11):
        arrow(ax, (pos[i][0], pos[i][1] + bh / 2), (pos[i + 1][0] + bw, pos[i + 1][1] + bh / 2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(FIG / "cap3_fluxo_geral.png", dpi=300)
    plt.close(fig)
    print("fluxo ok")


# ------------------------------------------------------------------ composicao por egua
def fig_composicao():
    d = pd.read_csv(CSV, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.7), dpi=300)
    rng = np.random.default_rng(0)
    for ax, col, color, title in ((axes[0], "micro_n", BLUE, "Microcotilédones por campo"),
                                  (axes[1], "capilar_n", YELLOW, "Capilares por campo")):
        ax.grid(axis="y", color=GRID, lw=.6, zorder=0)
        for i, m in enumerate(MARES):
            v = d[d.egua == m][col].to_numpy()
            ax.scatter(i + rng.uniform(-.22, .22, len(v)), v, s=11, color=color, alpha=.7, edgecolor="white", linewidth=.4, zorder=2)
            ax.hlines(v.mean(), i - .34, i + .34, color=INK, lw=1.6, zorder=3)
        ax.set_xticks(range(6), [f"{m}\n(total: {fmt(d[d.egua == m][col].sum())})" for m in MARES], fontsize=7.4)
        ax.set_title(title, loc="left", fontsize=8.6, fontweight="bold")
        ax.set_ylim(0, None)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, alpha=.7, label="Um campo"),
                        Line2D([], [], color=INK, lw=1.6, label="Média da égua")],
               loc="lower center", ncol=2, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=.075, right=.985, top=.95, bottom=.12, hspace=.42)
    fig.savefig(FIG / "cap3_composicao.png", dpi=300)
    plt.close(fig)
    print("composicao ok:", d.micro_n.sum(), d.capilar_n.sum(), len(d))


# ------------------------------------------------------------------ rotacoes
def fig_rotacoes():
    n = 6
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=300)
    col = {"tr": BLUE, "va": ORANGE, "te": AQUA}
    txt = {"tr": "Treino", "va": "Valid.", "te": "Teste"}
    tc = {"tr": "white", "va": INK, "te": INK}
    for f in range(n):
        for m in range(n):
            r = "te" if m == f else ("va" if m == (f + 1) % n else "tr")
            ax.add_patch(Rectangle((m, n - 1 - f), 1, 1, fc=col[r], ec="white", lw=2))
            ax.text(m + .5, n - 1 - f + .5, txt[r], ha="center", va="center", fontsize=7.6, color=tc[r])
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([i + .5 for i in range(n)], MARES, fontsize=7.4)
    ax.set_yticks([n - 1 - i + .5 for i in range(n)], [f"Fold {i + 1}" for i in range(n)])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(Rectangle((-0.04, n - 1 - 4 - .04), n + .08, 1.08, fill=False, ec=INK, lw=1.3, ls=(0, (4, 2))))
    ax.text(n + .12, n - 1 - 4 + .5, "fold\nprincipal", ha="left", va="center", fontsize=7.6)
    h = [Rectangle((0, 0), 1, 1, fc=col[k], ec="none") for k in ("tr", "va", "te")]
    ax.legend(h, ["Treino (4 éguas, 120 campos)", "Validação (1 égua, 30 campos)", "Teste (1 égua, 30 campos)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, fontsize=7.6, handlelength=1.2, columnspacing=1.5)
    fig.subplots_adjust(left=.1, right=.88, top=.90, bottom=.12)
    fig.savefig(FIG / "cap3_rotacoes.png", dpi=300)
    plt.close(fig)
    print("rotacoes ok")


# ------------------------------------------------------------------ protocolo de selecao
def fig_selecao():
    fig, ax = plt.subplots(figsize=(6.4, 2.95), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(1.2, 6.5)
    ax.axis("off")
    ax.text(0.0, 6.25, "Etapa 1 · fold principal", fontsize=8.8, fontweight="bold", va="center")
    ax.text(6.55, 6.25, "Etapa 2 · seis rotações", fontsize=8.8, fontweight="bold", va="center")
    rows = [("Microcotilédones", BLUE, 3.95,
             "10 modelos YOLO", "YOLO26m\nYOLO26s\nYOLO11m",
             "AP de máscara\nna validação do\nfold principal"),
            ("Capilares", YELLOW, 1.45,
             "6 variantes YOLO\n(n e s de v8, 11 e 26)", "YOLO11s\nYOLO26s",
             "AP de máscara na\nvalidação; exceções\ndescritas no texto")]
    for name, c, y, s1, s2, rk in rows:
        ax.text(0.0, y + 1.78, name, fontsize=8.4, fontweight="bold", color=c if c != YELLOW else "#a87500")
        box(ax, 0.0, y, 3.1, 1.5, "Treino único", s1, fc="white", ec=c)
        box(ax, 3.6, y, 2.5, 1.5, "Ranking", rk, fc="white", ec=c)
        box(ax, 6.55, y, 2.6, 1.5, "Finalistas", s2, fc="white", ec=c)
        box(ax, 9.6, y, 2.4, 1.5, "Escolha", "1 erro-padrão da\nmelhor média; menos\nparâmetros", fc="white", ec=INK)
        arrow(ax, (3.1, y + .75), (3.6, y + .75))
        arrow(ax, (6.1, y + .75), (6.55, y + .75))
        arrow(ax, (9.15, y + .75), (9.6, y + .75))
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    fig.savefig(FIG / "cap3_selecao.png", dpi=300)
    plt.close(fig)
    print("selecao ok")


# ------------------------------------------------------------------ normalizacoes
def fig_normaliz():
    import os
    test_dir = DS / "native" / "test" / "images"
    stem = sorted(os.listdir(test_dir))[2].split("_jpg")[0]          # ZAZA-B_003
    read = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)  # noqa: E731
    first = lambda d: next(p for p in sorted(Path(d).iterdir()) if p.name.startswith(stem))  # noqa: E731
    panels = [("Original", read(first(test_dir)))]
    for key, name in (("whitebalance", "Balanço de branco"), ("grayscale", "Escala de cinza"), ("histmatch", "Histmatch"),
                      ("reinhard", "Reinhard"), ("macenko", "Macenko"), ("vahadane", "Vahadane")):
        panels.append((name, read(first(DS / f"native_{key}" / "test" / "images"))))
    # mosaico de referencia: 120 campos de treino, 368x276, 12 colunas
    tr = sorted((DS / "native" / "train" / "images").iterdir())
    tiles = [cv2.resize(read(p), (368, 276), interpolation=cv2.INTER_AREA) for p in tr]
    rows = [np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)]
    mosaic = np.vstack(rows)
    small = lambda a: cv2.resize(a, (900, round(a.shape[0] * 900 / a.shape[1])), interpolation=cv2.INTER_AREA)  # noqa: E731
    fig, axes = plt.subplots(2, 4, figsize=(6.4, 3.0), dpi=300)
    for ax, (t, im) in zip(axes.ravel(), panels):
        ax.imshow(small(im))
        ax.set_title(t, fontsize=7.8, pad=2)
        ax.axis("off")
    axes[1, 3].imshow(small(mosaic))
    axes[1, 3].set_title("Mosaico de referência\n(120 campos de treino)", fontsize=7.8, pad=2)
    axes[1, 3].axis("off")
    fig.subplots_adjust(left=.005, right=.995, top=.93, bottom=.005, wspace=.03, hspace=.2)
    fig.savefig(FIG / "cap3_normalizacoes.png", dpi=300)
    plt.close(fig)
    print("normaliz ok:", stem, mosaic.shape)


# ------------------------------------------------------------------ curvas de treino
def fig_curvas():
    runs = {"micro": ("micro__ultralytics__yolo26s-seg__none__s0__d6fbe2c1", BLUE, "Microcotilédones (YOLO26s)"),
            "capilar": ("capilar__ultralytics__yolo11s-seg__none__s0__1b192800", YELLOW, "Capilares (YOLO11s)")}
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.2), dpi=300, sharex="col")
    for j, (task, (rid, color, title)) in enumerate(runs.items()):
        r = pd.read_csv(BR + rid + "/train/run/results.csv")
        r.columns = [c.strip() for c in r.columns]
        ep = r["epoch"]
        fit = sum(0.1 * r[f"metrics/mAP50({k})"] + 0.9 * r[f"metrics/mAP50-95({k})"] for k in "BM")   # fitness do Ultralytics
        best = int(fit.idxmax())
        ax = axes[0, j]
        ax.plot(ep, r["train/seg_loss"], color=color, lw=1.4, label="treino")
        ax.plot(ep, r["val/seg_loss"], color=INK2, lw=1.2, ls="--", label="validação")
        ax.set_title(title, loc="left", fontsize=8.6, fontweight="bold")
        ax.set_ylabel("Perda de máscara")
        ax = axes[1, j]
        ax.plot(ep, r["metrics/mAP50-95(M)"], color=color, lw=1.4)
        ax.scatter(ep[best], r["metrics/mAP50-95(M)"][best], s=26, color=color, edgecolor=INK, zorder=4)
        ax.text(.97, .05, f"● melhor época: {int(ep[best])} de {len(r)}", transform=ax.transAxes, ha="right", fontsize=7.4, color=INK2)
        ax.set_xlabel("Época")
        ax.set_ylabel("AP de máscara na validação")
        for a in axes[:, j]:
            a.grid(color=GRID, lw=.6)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.tick_params(length=0)
        print(task, "épocas rodadas:", len(r), "melhor:", int(ep[best]))
    axes[0, 0].legend(frameon=False, fontsize=7.4)
    fig.subplots_adjust(left=.09, right=.985, top=.94, bottom=.09, wspace=.27, hspace=.12)
    fig.savefig(FIG / "cap3_curvas_treino.png", dpi=300)
    plt.close(fig)
    print("curvas ok")


# ------------------------------------------------------------------ F1 x limiar
def fig_limiar():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), dpi=300, sharey=True)
    for ax, task, model, color, title in ((axes[0], "micro", "yolo26s-seg", BLUE, "Microcotilédones (YOLO26s)"),
                                           (axes[1], "capilar", "yolo11s-seg", YELLOW, "Capilares (YOLO11s)")):
        stars = []
        ax.grid(color=GRID, lw=.6, zorder=0)
        for rd in sorted(glob.glob(BR + f"{task}__ultralytics__{model}__none__*")):
            mv = json.load(open(rd + "/metrics_val.json"))
            cal = pd.DataFrame(mv["calibration"])
            ax.plot(cal.conf, cal.f1, color=color, lw=1.1, alpha=.6, zorder=2)
            cs = mv["conf_star"]
            ax.scatter(cs, float(cal.iloc[(cal.conf - cs).abs().argmin()].f1), s=22, color=color, edgecolor=INK, linewidth=.8, zorder=4)
            stars.append(cs)
        ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold")
        ax.set_xlabel("Limiar de confiança")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        ax.text(.03, .06, f"{len(stars)} rotações; limiar escolhido {min(stars):.2f}–{max(stars):.2f}".replace(".", ","),
                transform=ax.transAxes, fontsize=7, color=INK2)
    axes[0].set_ylabel("F1 na égua de validação")
    fig.legend(handles=[Line2D([], [], color=INK2, lw=1.1, label="Uma curva por rotação"),
                        Line2D([], [], marker="o", ls="", color=INK2, markeredgecolor=INK, label="Limiar que maximiza o F1")],
               loc="lower center", ncol=2, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=.085, right=.97, top=.90, bottom=.27, wspace=.08)
    fig.savefig(FIG / "cap3_f1_limiar.png", dpi=300)
    plt.close(fig)
    print("limiar ok")


# ------------------------------------------------------------------ pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.4, 3.7), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    cin, cmi, cca, cme, cou = ("#eef3fb", BLUE), ("#eef3fb", BLUE), ("#fdf6e3", YELLOW), ("#e8f6f0", AQUA), ("#f3f2ee", INK2)
    box(ax, 0.0, 2.65, 2.0, 1.5, "Entrada", "campos nativos\n4140 × 3096 px\n(um ou vários)", fc=cin[0], ec=cin[1], bs=6.6)
    box(ax, 2.6, 4.55, 3.0, 1.9, "Microcotilédones", "esticar para 640 × 640\nYOLO26s\nmáscaras no campo nativo", fc=cmi[0], ec=cmi[1], bs=6.6)
    box(ax, 2.6, 0.35, 3.0, 1.9, "Capilares", "janelas de 1380 × 1032\nYOLO11s e fusão (SAHI)\nmáscaras no campo nativo", fc=cca[0], ec=cca[1], bs=6.6)
    box(ax, 6.2, 2.65, 2.4, 1.5, "Medidas", "por instância e\npor campo, em µm\n(0,1066 µm/px)", fc=cme[0], ec=cme[1], bs=6.6)
    box(ax, 9.1, 3.45, 2.9, 3.0, "Pacote de saída", "tabela por imagem\ntabela por instância\nanotações COCO\nsobreposições\nmáscaras binárias\nconfiguração e versões", fc=cou[0], ec=cou[1], bs=6.6)
    box(ax, 9.1, 0.35, 2.9, 2.55, "Interface Gradio", "imagem única ou lote\nmáscaras sobre o tecido\nárea e confiança ao passar\no mouse; tabelas e\ndownload do pacote", fc="white", ec=INK, bs=6.6)
    arrow(ax, (2.0, 3.55), (2.6, 5.3))
    arrow(ax, (2.0, 3.25), (2.6, 1.6))
    arrow(ax, (5.6, 5.2), (6.2, 3.85))
    arrow(ax, (5.6, 1.7), (6.2, 2.95))
    arrow(ax, (8.6, 3.4), (9.1, 4.6))
    arrow(ax, (10.55, 3.45), (10.55, 2.9))
    fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.005)
    fig.savefig(FIG / "cap3_pipeline.png", dpi=300)
    plt.close(fig)
    print("pipeline ok")


def _campo_teste(stem="ZAZA-B_003"):
    import os
    d = DS / "native" / "test" / "images"
    f = next(p for p in sorted(os.listdir(d)) if p.startswith(stem))
    return cv2.cvtColor(cv2.imread(str(d / f)), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ vistas dos dados
def fig_vistas():
    nat = _campo_teste()
    H, W = nat.shape[:2]
    small = cv2.resize(nat, (1000, round(H * 1000 / W)), interpolation=cv2.INTER_AREA)
    stretch = cv2.resize(nat, (640, 640), interpolation=cv2.INTER_AREA)
    tiled = small.copy()
    sh, sw = tiled.shape[:2]
    for k in (1, 2):
        cv2.line(tiled, (round(sw * k / 3), 0), (round(sw * k / 3), sh), (255, 255, 255), 5)
        cv2.line(tiled, (0, round(sh * k / 3)), (sw, round(sh * k / 3)), (255, 255, 255), 5)
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.05), dpi=300, gridspec_kw={"width_ratios": [4, 3.0, 4]})
    for ax, im, t in ((axes[0], small, "A  Campo nativo\n4140 × 3096 px"),
                      (axes[1], stretch, "B  Esticado para 640 × 640 px\n(microcotilédones)"),
                      (axes[2], tiled, "C  Grade 3 × 3 de ladrilhos\nde 1380 × 1032 px (capilares)")):
        ax.imshow(im)
        ax.set_title(t, fontsize=7.4, pad=3, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(INK2)
            sp.set_linewidth(.8)
    fig.subplots_adjust(left=.005, right=.995, top=.80, bottom=.01, wspace=.06)
    fig.savefig(FIG / "cap3_vistas.png", dpi=300)
    plt.close(fig)
    print("vistas ok")


# ------------------------------------------------------------------ janelas do SAHI
def fig_sahi():
    nat = _campo_teste()
    H, W = nat.shape[:2]
    ww, wh, ov = 1380, 1032, 0.2
    def starts(total, win):
        step = int(win * (1 - ov))
        xs = list(range(0, total - win + 1, step))
        if xs[-1] + win < total:
            xs.append(total - win)
        return xs
    xs, ys = starts(W, ww), starts(H, wh)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), dpi=300, gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.imshow(nat)
    cols = [BLUE, ORANGE]
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            ax.add_patch(Rectangle((x, y), ww, wh, fill=False, ec=cols[(i + j) % 2], lw=.9, alpha=.9))
    ax.set_title(f"{len(xs) * len(ys)} janelas de {ww} × {wh} px ({len(xs)} × {len(ys)})", fontsize=7.6, loc="left", pad=3)
    ax.axis("off")
    ax = axes[1]
    ax.imshow(nat[0:wh, 0:xs[1] + ww])
    ax.add_patch(Rectangle((xs[0], 0), ww, wh, fc=BLUE, ec=BLUE, lw=1.4, alpha=.28))
    ax.add_patch(Rectangle((xs[1], 0), ww, wh, fc=ORANGE, ec=ORANGE, lw=1.4, alpha=.28))
    ax.add_patch(Rectangle((xs[1], 0), xs[0] + ww - xs[1], wh, fc="none", ec=INK, lw=1.2, ls=(0, (4, 2))))
    ax.text((xs[1] + xs[0] + ww) / 2, wh + 60, "sobreposição de 20%", ha="center", va="top", fontsize=7.2)
    ax.set_xlim(0, xs[1] + ww)
    ax.set_ylim(wh + 230, 0)
    ax.set_title("Duas janelas vizinhas", fontsize=7.6, loc="left", pad=3)
    ax.axis("off")
    fig.subplots_adjust(left=.005, right=.995, top=.90, bottom=.01, wspace=.04)
    fig.savefig(FIG / "cap3_sahi.png", dpi=300)
    plt.close(fig)
    print("sahi ok:", xs, ys)


# ------------------------------------------------------------------ anotacao manual 2x2 e medidas (precisam dos modulos do artigo)
def _campo_anotado(lamina="ROSILHA-M-D_006"):
    sys.path.insert(0, r"D:\projeto_placentas_clayton\dev\projeto-placentas")
    from scripts.artigo_iser import dataset_vs_modelo as D
    fl = next(f for f in D.list_fields() if f["lamina"] == lamina)
    nat = cv2.imread(str(fl["path"]))
    gm, gc = D.gt_instances(fl)
    return nat, gm, gc


def _pintar(img, items, sx, sy, color, alpha, thick):
    H, W = img.shape[:2]
    for it in items:
        x0, y0 = int(round(it.x0 * sx)), int(round(it.y0 * sy))
        w, h = max(int(round(it.mask.shape[1] * sx)), 1), max(int(round(it.mask.shape[0] * sy)), 1)
        if x0 >= W or y0 >= H or x0 + w <= 0 or y0 + h <= 0:
            continue
        m = cv2.resize(it.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
        m = m[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0]
        if not m.any():
            continue
        roi = img[cy0:cy1, cx0:cx1]
        fill = roi.copy()
        fill[m > 0] = color
        roi[:] = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, cnts, -1, color, thick, cv2.LINE_AA)


def _rgb_to_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def fig_anotacao():
    nat, gm, gc = _campo_anotado()
    PW = 1200
    PH = round(nat.shape[0] * PW / nat.shape[1])
    ds = nat.shape[1] / PW
    base = cv2.resize(nat, (PW, PH), interpolation=cv2.INTER_AREA)
    sxm, sym = PW / 640, PH / 640
    pa, pb, pc, pd = base.copy(), base.copy(), base.copy(), base.copy()
    _pintar(pb, gm, sxm, sym, _rgb_to_bgr(BLUE), .45, 2)
    _pintar(pc, gc, 1 / ds, 1 / ds, _rgb_to_bgr(YELLOW), .65, 1)
    _pintar(pd, gm, sxm, sym, _rgb_to_bgr(BLUE), .35, 2)
    _pintar(pd, gc, 1 / ds, 1 / ds, _rgb_to_bgr(YELLOW), .65, 1)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.9), dpi=300)
    for ax, im, t in zip(axes.ravel(), (pa, pb, pc, pd),
                         ("A  Campo original", f"B  Microcotilédones ({len(gm)})", f"C  Capilares ({len(gc)})", "D  As duas classes")):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=7.8, loc="left", pad=2)
        ax.axis("off")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=BLUE, alpha=.7, label="Microcotilédone"),
                        Rectangle((0, 0), 1, 1, fc=YELLOW, alpha=.9, label="Capilar")],
               loc="lower center", ncol=2, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=.005, right=.995, top=.96, bottom=.05, wspace=.03, hspace=.12)
    fig.savefig(FIG / "cap3_anotacao.png", dpi=300)
    plt.close(fig)
    print("anotacao ok:", len(gm), "micro,", len(gc), "capilares")


def fig_medidas():
    nat, gm, _ = _campo_anotado()
    Hn, Wn = nat.shape[:2]
    fx, fy = Wn / 640, Hn / 640
    um = 50 / 469
    cands = []
    for it in gm:
        h, w = it.mask.shape
        x0, y0 = it.x0, it.y0
        if x0 > 6 and y0 > 6 and x0 + w < 634 and y0 + h < 634:
            cands.append((int(it.mask.sum()), it))
    cands.sort(key=lambda c: c[0])
    it = cands[int(len(cands) * 0.75)][1]
    h640, w640 = it.mask.shape
    from pipeline.geometry import instance_geometry          # a mesma funcao usada pelo pipeline
    geo, polys = instance_geometry(it.mask.astype(np.uint8), int(it.x0), int(it.y0), fx, fy, 0.5)
    poly = np.array(polys[0], np.float32).reshape(-1, 2)
    mg = 60
    x0p, y0p = poly.min(0)
    x1p, y1p = poly.max(0)
    cx0, cy0, cx1, cy1 = max(int(x0p) - mg, 0), max(int(y0p) - mg, 0), min(int(x1p) + mg, Wn), min(int(y1p) + mg, Hn)
    crop = cv2.cvtColor(nat[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2RGB).copy()
    cc = (poly - [cx0, cy0]).astype(np.float32)
    hh = cv2.convexHull(cc)[:, 0, :]
    (ex, ey), (ea, eb), ang = cv2.fitEllipse(cc)
    fig, (ax, at) = plt.subplots(1, 2, figsize=(6.4, 3.4), dpi=300, gridspec_kw={"width_ratios": [1.25, 1]})
    ax.imshow(crop)
    ax.fill(*cc.T, color=BLUE, alpha=.28, lw=0)
    ax.plot(*np.vstack([cc, cc[:1]]).T, color=BLUE, lw=1.4, label="Contorno (área e perímetro)")
    ax.plot(*np.vstack([hh, hh[:1]]).T, color=ORANGE, lw=1.1, ls=(0, (4, 2)), label="Fecho convexo (solidez)")
    t_ = np.linspace(0, 2 * np.pi, 200)
    th = np.deg2rad(ang)
    exx = ex + (ea / 2) * np.cos(t_) * np.cos(th) - (eb / 2) * np.sin(t_) * np.sin(th)
    eyy = ey + (ea / 2) * np.cos(t_) * np.sin(th) + (eb / 2) * np.sin(t_) * np.cos(th)
    ax.plot(exx, eyy, color=AQUA, lw=1.1, label="Elipse ajustada (eixos)")
    ax.scatter([ex], [ey], s=12, color=INK, zorder=5)
    ax.set_xlim(0, cx1 - cx0)
    ax.set_ylim(cy1 - cy0, 0)
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1, frameon=False, fontsize=6.8)
    at.axis("off")
    br = lambda v, d=0: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")  # noqa: E731
    rows = [("Área", br(geo["area_px_native"] * um * um) + " µm²"),
            ("Perímetro", br(geo["perimeter_um"]) + " µm"),
            ("Eixo maior", br(geo["major_axis_um"]) + " µm"),
            ("Eixo menor", br(geo["minor_axis_um"]) + " µm"),
            ("Circularidade", br(geo["circularity"], 2)),
            ("Solidez", br(geo["solidity"], 2))]
    at.text(0.0, 0.97, "Medidas deste microcotilédone", fontsize=8, fontweight="bold", va="top")
    for k, (a, b) in enumerate(rows):
        y = 0.80 - k * 0.125
        at.text(0.0, y, a, fontsize=7.8, va="center", color=INK2)
        at.text(0.62, y, b, fontsize=7.8, va="center", ha="left")
        at.plot([0, 1], [y - 0.06, y - 0.06], color=GRID, lw=.6, transform=at.transData)
    at.set_xlim(0, 1)
    at.set_ylim(0, 1)
    fig.subplots_adjust(left=.005, right=.99, top=.98, bottom=.21, wspace=.08)
    fig.savefig(FIG / "cap3_medidas.png", dpi=300)
    plt.close(fig)
    print("medidas ok:", dict(rows))


ALL = {"fluxo": fig_fluxo, "composicao": fig_composicao, "rotacoes": fig_rotacoes, "selecao": fig_selecao,
       "normaliz": fig_normaliz, "curvas": fig_curvas, "limiar": fig_limiar, "pipeline": fig_pipeline,
       "vistas": fig_vistas, "sahi": fig_sahi, "anotacao": fig_anotacao, "medidas": fig_medidas}

if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
