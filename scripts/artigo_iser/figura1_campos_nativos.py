"""Figura 1 do artigo: 4 campos nativos (2x2), cada um de uma egua diferente.

Painel A: o campo ROSILHA-M-D_006 (o mesmo das Figuras 2 e 3). Paineis B-D: sorteia 3 das outras 5 eguas (semente fixa) e, em cada uma,
usa o campo cuja fracao de area microcotiledonar anotada e a mais proxima da mediana daquela egua (um campo "tipico").
A barra de 50 um fica como veio do microscopio.
Saida: temp_ml/artigo_iser/artigo/figs/fig1_native_fields_2x2.jpg
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FD = Path(r"D:\projeto_placentas_clayton\datasets_v4\test-ZAZA_val-TOSTADA\native")
GT = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\dataset_vs_modelo\lamina_anotada.csv")
OUT = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\artigo\figs\fig1_native_fields_2x2.jpg")
SEED, N = 0, 4
PW = 1200                      # largura de cada painel (campo nativo 4140x3096 -> 1200x897)

g = pd.read_csv(GT, encoding="utf-8-sig")
rng = np.random.default_rng(SEED)
# painel A fixo: o campo que as Figuras 2 e 3 (anotacoes e estrategia de entrada) mostram, para as legendas poderem remeter a ele
FIXO = "ROSILHA-M-D_006"
fixo = g[g.lamina == FIXO].iloc[0]
eguas = sorted(set(g.egua.unique()) - {fixo.egua})
escolhidas = list(rng.choice(eguas, N - 1, replace=False))
sel = [fixo]
for e in escolhidas:
    d = g[g.egua == e]
    med = d.micro_pct_do_campo.median()
    sel.append(d.iloc[(d.micro_pct_do_campo - med).abs().argmin()])
sel = pd.DataFrame(sel)
print(sel[["lamina", "egua", "grupo", "micro_pct_do_campo", "micro_n", "capilar_n"]].to_string(index=False))


def find(arquivo: str) -> Path:
    p = list(FD.glob(f"*/images/{arquivo}"))
    assert len(p) == 1, arquivo
    return p[0]


def painel(path: Path, letra: str) -> np.ndarray:
    img = cv2.imread(str(path))
    ph = round(img.shape[0] * PW / img.shape[1])
    img = cv2.resize(img, (PW, ph), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (86, 86), (255, 255, 255), -1)
    cv2.putText(img, letra, (14, 68), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 5, cv2.LINE_AA)
    return img


paineis = [painel(find(a), l) for a, l in zip(sel.arquivo, "ABCD")]
gut = 14
H, W = paineis[0].shape[:2]
canvas = np.full((2 * H + gut, 2 * W + gut, 3), 255, np.uint8)
for i, p in enumerate(paineis):
    r, c = divmod(i, 2)
    canvas[r * (H + gut): r * (H + gut) + H, c * (W + gut): c * (W + gut) + W] = p
OUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("salvo:", OUT, canvas.shape, f"{OUT.stat().st_size / 1e6:.2f} MB")
