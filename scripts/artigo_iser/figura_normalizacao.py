"""Figura de normalizacao de cor: campo de treino (referencia), imagem externa original e os 6 metodos (8 paineis, 2 x 4).

Coluna da esquerda: campo de treino (ROSILHA-M-D_006, o mesmo das Figs. 1 e 2) e a imagem externa original.
Tres colunas a direita: linha 1 = metodos de estatistica global da cor (Reinhard, histogram matching, balanco de branco);
                        linha 2 = separacao de corantes (Macenko, Vahadane) e escala de cinza.
Imagem externa: a mais proxima da MEDIANA da distancia de cor entre as 90 (um caso tipico). Recorte: o ladrilho 3x3 com mais tecido (sem o canto da barra).
Rotulo: distancia de Wasserstein em L*a*b* ao treino (media das 90 imagens externas; campos de treino: 6.2).
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ISER_RUN=run2 -> 150 imagens com coloracao boa (imagens_teste_artigo_iser_v2), saida com sufixo _run2; padrao = run1 (90 imagens)
_RUN = os.environ.get("ISER_RUN", "run1")
RUN1 = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser") / _RUN
IN_DIR = Path(r"D:\projeto_placentas_clayton\imagens_teste_artigo_iser" + ("_v2" if _RUN == "run2" else ""))
TRAIN = list(Path(r"D:\projeto_placentas_clayton\datasets_v4\test-ZAZA_val-TOSTADA\native").glob("*/images/ROSILHA-M-D_006_*"))[0]
OUT = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\artigo\figs") / ("fig_stain_normalization" + ("" if _RUN == "run1" else "_" + _RUN) + ".png")
INK, INK2, BLUE = "#0b0b0b", "#52514e", "#2a78d6"
W, H = 4140, 3096

cor = pd.read_csv(RUN1 / "cor.csv")
new = cor[cor.set == "new"]
dist = new.groupby("method").wd_mean.mean().to_dict()
dist_train = cor[cor.set == "train"].wd_mean.mean()
# imagem externa tipica
n0 = new[new.method == "none"].copy()
med = n0.wd_mean.median()
row = n0.iloc[(n0.wd_mean - med).abs().argmin()]
files = {p.stem: p for p in IN_DIR.iterdir() if p.is_file()}
key = Path(str(row.image)).stem
ext = files.get(key) or next(p for s, p in files.items() if s.startswith(key[:20]))
print("imagem externa:", ext.name, "| distancia", round(float(row.wd_mean), 2), "| mediana", round(float(med), 2))


def read(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def best_tile(img):
    """Indice (r, c) do ladrilho 3x3 com mais tecido, sem o ladrilho da barra (canto inferior direito)."""
    small = cv2.resize(img, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)[..., 0].astype(float)
    tissue = L < (np.percentile(L, 90) - 14)             # tecido = mais escuro que o fundo (funciona com coloracao pouco saturada)
    h, w = tissue.shape
    best, arg = -1, (0, 0)
    for r in range(3):
        for c in range(3):
            if (r, c) == (2, 2):
                continue
            f = float(tissue[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3].mean())
            if f > best:
                best, arg = f, (r, c)
    return arg


def crop(img, rc):
    r, c = rc
    return img[r * H // 3:(r + 1) * H // 3, c * W // 3:(c + 1) * W // 3]


img_ext = read(ext)
tile_ext = best_tile(img_ext)
img_tr = read(TRAIN)
tile_tr = best_tile(img_tr)
print("ladrilho externo", tile_ext, "| ladrilho treino", tile_tr)

methods = [("reinhard", "Reinhard"), ("histmatch", "Histogram matching"), ("whitebalance", "White balance"),
           ("macenko", "Macenko"), ("vahadane", "Vahadane"), ("grayscale", "Grayscale")]
panels = {"A": (crop(img_tr, tile_tr), "Training field", dist_train), "B": (crop(img_ext, tile_ext), "External, original", dist["none"])}
for letter, (m, name) in zip("CDEFGH", methods):
    panels[letter] = (crop(read(RUN1 / "norm" / m / f"{ext.stem}.png"), tile_ext), name, dist[m])

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
# posicoes exatas (em polegadas): paineis 4:3 sem espaco morto, rotulo logo abaixo de cada imagem
FW, MARG, GAP, LAB, RGAP, TOP, BOT = 7.4, 0.08, 0.06, 0.40, 0.10, 0.06, 0.04
PW = (FW - 2 * MARG - 3 * GAP) / 4
PH = PW * 0.75
FH = TOP + 2 * (PH + LAB) + RGAP + BOT
fig = plt.figure(figsize=(FW, FH), dpi=300)
layout = {"A": (0, 0), "B": (0, 1), "C": (0, 2), "D": (0, 3), "E": (1, 0), "F": (1, 1), "G": (1, 2), "H": (1, 3)}   # ordem de leitura
for letter, (r, c) in layout.items():
    x0 = MARG + c * (PW + GAP)
    y0 = FH - TOP - PH - r * (PH + LAB + RGAP)
    ax = fig.add_axes([x0 / FW, y0 / FH, PW / FW, PH / FH])
    im, name, d = panels[letter]
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if letter == "C":                                     # metodo adotado
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor(BLUE)
            s.set_linewidth(1.8)
    ax.add_patch(Rectangle((0, 0), 0.115, 0.14, transform=ax.transAxes, color="white", zorder=3))
    ax.text(0.057, 0.07, letter, transform=ax.transAxes, ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
    sub = f"distance {d:.1f}" + (" (adopted)" if letter == "C" else "")
    ax.text(0.5, -0.03, name, transform=ax.transAxes, ha="center", va="top", fontsize=7.2, fontweight="bold")
    ax.text(0.5, -0.03 - 0.115 * (1.316 / PH), sub, transform=ax.transAxes, ha="center", va="top", fontsize=6.6, color=INK2)
fig.savefig(OUT, dpi=300)
print("salvo:", OUT, "| distancias:", {k: round(v, 1) for k, v in dist.items()}, "| treino", round(float(dist_train), 1))
