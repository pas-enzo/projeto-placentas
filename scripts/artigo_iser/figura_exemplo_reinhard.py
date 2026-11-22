"""Figura dos Resultados (imagens externas): exemplo(s) das 90 imagens externas.

Colunas: (A) imagem original; (B) imagem normalizada (Reinhard); (C) segmentacao do modelo treinado com Reinhard (modelo adotado) sobre a imagem normalizada.
As imagens externas nao tem anotacao manual. Cores iguais as das figuras dos Metodos (azul = micro, amarelo = capilar).
Uso: python figura_exemplo_reinhard.py "ILUMINADA 25 CU 5" ["outra imagem" ...]
Saida: artigo/figs/fig_exemplo_reinhard.png
"""
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(r"D:\projeto_placentas_clayton")
# ISER_RUN=run2 -> 150 imagens com coloracao boa (imagens_teste_artigo_iser_v2), saida com sufixo _run2; padrao = run1 (90 imagens)
_RUN = os.environ.get("ISER_RUN", "run1")
RUN = ROOT / "temp_ml" / "artigo_iser" / _RUN
ORIG = ROOT / ("imagens_teste_artigo_iser" + ("_v2" if _RUN == "run2" else ""))
OUT = ROOT / "temp_ml" / "artigo_iser" / "artigo" / "figs" / ("fig_exemplo_reinhard" + ("" if _RUN == "run1" else "_" + _RUN) + ".png")
BLUE, YELLOW = "#2a78d6", "#eda100"
PW = 1200


def bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def load_ann(variant):
    j = json.load(open(RUN / "pipeline" / variant / "annotations.coco.json", encoding="utf-8"))
    names = {i["id"]: i["file_name"] for i in j["images"]}
    by = {}
    for a in j["annotations"]:
        by.setdefault(names[a["image_id"]], []).append(a)
    return by


def paint(img, anns, scale):
    layer = img.copy()
    for cat, color in ((1, BLUE), (2, YELLOW)):
        for a in anns:
            if a["category_id"] != cat:
                continue
            for seg in a["segmentation"]:
                pts = (np.array(seg, dtype=np.float32).reshape(-1, 2) * scale).astype(np.int32)
                cv2.fillPoly(layer, [pts], bgr(color))
    out = cv2.addWeighted(layer, 0.40, img, 0.60, 0)
    for cat, color in ((1, BLUE), (2, YELLOW)):
        for a in anns:
            if a["category_id"] != cat:
                continue
            for seg in a["segmentation"]:
                pts = (np.array(seg, dtype=np.float32).reshape(-1, 2) * scale).astype(np.int32)
                cv2.polylines(out, [pts], True, bgr(color), 2, cv2.LINE_AA)
    return out


def read_small(path):
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    s = PW / im.shape[1]
    return cv2.resize(im, (PW, int(round(im.shape[0] * s))), interpolation=cv2.INTER_AREA), s


def main(names):
    ann = load_ann("A_reinhard")
    rows = []
    for nm in names:
        o, s = read_small(ORIG / f"{nm}.jpg")
        n, _ = read_small(RUN / "norm" / "reinhard" / f"{nm}.png")
        rows.append([o, n, paint(n, ann[f"{nm}.jpg"], s)])
    titles = ["Original image", "Reinhard-normalized image", "Segmentation (model trained with normalized images)"]
    h, w = rows[0][0].shape[:2]
    nr = len(rows)
    fig, axes = plt.subplots(nr, 3, figsize=(3 * 3.4, nr * 3.4 * h / w + 0.6), dpi=200, squeeze=False)
    for r in range(nr):
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(rows[r][c], cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=8.5, pad=4)
            ax.text(0.012, 0.985, "ABCDEFGHI"[r * 3 + c], transform=ax.transAxes, va="top", ha="left", fontsize=10, fontweight="bold",
                    color="white", bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.65))
    fig.legend(handles=[Patch(fc=BLUE, ec=BLUE, alpha=0.6, label="Microcotyledons"), Patch(fc=YELLOW, ec=YELLOW, label="Capillaries")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.003, right=0.997, top=0.90 if nr == 1 else 0.94, bottom=0.12 if nr == 1 else 0.07, wspace=0.015, hspace=0.015)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("salvo:", OUT)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ILUMINADA 25 CU 5"])
