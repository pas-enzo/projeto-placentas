"""Quanto do capilar cai FORA do microcotiledone, e quao confiavel e a fracao capilar/microcotiledone?

Compara, nas 180 laminas, a anotacao manual com a predicao 'OOF_none' (cada lamina predita pelo modelo do fold em que a egua dela foi o TESTE:
unica estimativa honesta para as 180). Duas definicoes da fracao capilar:
  independente = area dos capilares / area dos microcotiledones           (como a entrega e o TCC calcularam: duas medidas separadas)
  dentro       = area dos capilares DENTRO da mascara de micro / area de micro   (definicao da Isadora: capilares no tecido microcotiledonar)
Tambem mede a fracao de capilar fora do micro, na anotacao e na predicao. Resolucao de trabalho: metade do campo nativo (2070 x 1548).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(r"D:\projeto_placentas_clayton\dev\projeto-placentas")
sys.path.insert(0, str(REPO))
from scripts.artigo_iser import dataset_vs_modelo as D, fluxo as F  # noqa: E402

TW, TH = 2070, 1548
VAR = sys.argv[1] if len(sys.argv) > 1 else "OOF_none"


def raster(items, sx, sy):
    m = np.zeros((TH, TW), np.uint8)
    for it in items:
        x0, y0 = int(round(it.x0 * sx)), int(round(it.y0 * sy))
        w, h = max(int(round(it.mask.shape[1] * sx)), 1), max(int(round(it.mask.shape[0] * sy)), 1)
        if x0 >= TW or y0 >= TH or x0 + w <= 0 or y0 + h <= 0:
            continue
        r = cv2.resize(it.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, TW), min(y0 + h, TH)
        m[cy0:cy1, cx0:cx1] |= r[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0]
    return m.astype(bool)


def medidas(micro, cap):
    a_m, a_c = int(micro.sum()), int(cap.sum())
    a_in = int((cap & micro).sum())
    return a_m, a_c, a_in


rows = []
for i, fl in enumerate(D.list_fields(), 1):
    gm, gc = D.gt_instances(fl)
    z = np.load(D.OUT / "pred" / VAR / f"{fl['lamina']}.npz")
    pm, pc = F.unpack(z, "micro"), F.unpack(z, "cap")
    for quem, im, ic in (("anotado", gm, gc), ("modelo", pm, pc)):
        m = raster(im, TW / 640, TH / 640)
        c = raster(ic, TW / F.FIELD_W, TH / F.FIELD_H)
        a_m, a_c, a_in = medidas(m, c)
        rows.append({"lamina": fl["lamina"], "egua": fl["egua"], "quem": quem, "micro": a_m, "cap": a_c, "cap_dentro": a_in})
    if i % 30 == 0:
        print(i, "/ 180", flush=True)

d = pd.DataFrame(rows)
w = d.pivot(index=["lamina", "egua"], columns="quem", values=["micro", "cap", "cap_dentro"])
w.columns = [f"{a}_{b}" for a, b in w.columns]
w = w.reset_index()
for q in ("anotado", "modelo"):
    w[f"fora_{q}_pct"] = 100 * (1 - w[f"cap_dentro_{q}"] / w[f"cap_{q}"])
    w[f"indep_{q}"] = 100 * w[f"cap_{q}"] / w[f"micro_{q}"]
    w[f"dentro_{q}"] = 100 * w[f"cap_dentro_{q}"] / w[f"micro_{q}"]


def ccc(x, y):
    sxy = np.cov(x, y, ddof=0)[0, 1]
    return 2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


print(f"\n== Predicao: {VAR} | {len(w)} laminas | resolucao {TW}x{TH}")
print("\nCapilar FORA da mascara de micro (% da area de capilar por campo)")
for q in ("anotado", "modelo"):
    s = w[f"fora_{q}_pct"]
    print(f"  {q:8s}: media {s.mean():5.1f}% | mediana {s.median():5.1f}% | p90 {s.quantile(.9):5.1f}% | max {s.max():5.1f}%")
print("\nFracao capilar (%), modelo contra anotado")
print(f"  {'definicao':12s} {'media anot.':>11s} {'media modelo':>12s} {'vies':>7s} {'erro abs':>9s} {'r':>6s} {'CCC':>6s}")
for nome, ca, cm in (("independente", "indep_anotado", "indep_modelo"), ("dentro", "dentro_anotado", "dentro_modelo")):
    a, m = w[ca].values, w[cm].values
    print(f"  {nome:12s} {a.mean():11.1f} {m.mean():12.1f} {np.mean(m - a):+7.1f} {np.mean(np.abs(m - a)):9.1f} {np.corrcoef(a, m)[0, 1]:6.2f} {ccc(a, m):6.2f}")
print("\nPor egua (erro absoluto medio da fracao, pontos percentuais)")
g = w.assign(e_indep=(w.indep_modelo - w.indep_anotado).abs(), e_dentro=(w.dentro_modelo - w.dentro_anotado).abs(),
             fora_m=w.fora_modelo_pct, fora_a=w.fora_anotado_pct).groupby("egua")[["e_indep", "e_dentro", "fora_a", "fora_m"]].mean().round(1)
print(g.to_string())
w.round(2).to_csv(Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser") / f"capilar_dentro_do_micro_{VAR}.csv", index=False)
