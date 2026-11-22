"""Monta a pasta de entrega (so Reinhard A) para a veterinaria.

Le o que ja existe (nao roda modelo). Saida: temp_ml/artigo_iser/ENTREGA_REINHARD/
  LEIA-ME.txt
  1_comparacao_por_lamina.csv / _BR.csv     180 laminas: anotado x inferido (contagem, area, % do campo)
  (fora da entrega) ../metricas_reinhard_anotadas.csv   erro medio, vies, correlacao etc. por grupo e por egua
  3_imagens_comparativas_180/               imagem de entrada | anotado | modelo
  4_imagens_novas_90/                       medidas por imagem (csv) e imagens lado a lado
CSVs so no formato brasileiro (; entre colunas, virgula decimal). Nao ha medidas por instancia na entrega.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser")
DVM, RUN1 = BASE / "dataset_vs_modelo", BASE / "run1"
OUT = BASE / "ENTREGA_REINHARD"
VAR = "A_reinhard"
FATOR_ESCALA_90 = (469 / 466) ** 2   # barra de 50 um = 466 px nas 90 imagens novas (469 nas 180 anotadas)
GRUPO_TXT = {"teste": "teste (ZAZA)", "validacao": "validacao (TOSTADA)", "treino": "treino (4 eguas)"}


def save(df: pd.DataFrame, name: str) -> None:
    # formato brasileiro (ponto e virgula entre colunas, virgula decimal): abre direto no Excel em portugues
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxy = np.cov(x, y, ddof=0)[0, 1]
    return 2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def metricas(a, m):
    a, m = np.asarray(a, float), np.asarray(m, float)
    d = m - a
    ok = a != 0
    r = {"n_laminas": len(a), "media_anotado": a.mean(), "media_inferido": m.mean(), "vies_medio": d.mean(),
         "vies_pct": 100 * d.mean() / a.mean(), "erro_absoluto_medio": np.abs(d).mean(),
         "erro_absoluto_medio_pct": 100 * np.abs(d[ok] / a[ok]).mean(),
         "limite_inferior_95": d.mean() - 1.96 * d.std(ddof=1), "limite_superior_95": d.mean() + 1.96 * d.std(ddof=1)}
    r["correlacao_pearson"] = np.corrcoef(a, m)[0, 1] if len(a) > 2 else np.nan
    r["concordancia_ccc"] = ccc(a, m) if len(a) > 2 else np.nan
    return r


def main() -> None:
    OUT.mkdir(exist_ok=True)
    c = pd.read_csv(DVM / "lamina_comparacao.csv", encoding="utf-8-sig")
    c = c[c.variante == VAR].copy()
    assert len(c) == 180, len(c)
    inf = pd.read_csv(DVM / "lamina_inferida.csv", encoding="utf-8-sig")
    inf = inf[inf.variante == VAR].set_index("lamina")

    # ---------------- 1) por lamina
    t = pd.DataFrame({"lamina": c.lamina, "egua": c.egua, "grupo": c.grupo.map(GRUPO_TXT)})
    for cls, p in (("micro", "micro"), ("capilar", "capilar")):
        t[f"{cls}_contagem_anotada"] = c[f"{p}_n_anotado"]
        t[f"{cls}_contagem_inferida"] = c[f"{p}_n_modelo"]
        t[f"{cls}_contagem_diferenca"] = c[f"{p}_n_dif"]
        t[f"{cls}_area_um2_anotada"] = c[f"{p}_area_uniao_um2_anotado"]
        t[f"{cls}_area_um2_inferida"] = c[f"{p}_area_uniao_um2_modelo"]
        t[f"{cls}_area_diferenca_pct"] = c[f"{p}_area_uniao_um2_dif_pct"]
    t["micro_pct_do_campo_anotado"] = c.micro_pct_do_campo_anotado
    t["micro_pct_do_campo_inferido"] = c.micro_pct_do_campo_modelo
    t["micro_pct_do_campo_diferenca_pontos"] = c.micro_pct_do_campo_dif_pontos
    t["capilar_pct_da_area_micro_anotado"] = c.capilar_pct_da_area_micro_anotado
    t["capilar_pct_da_area_micro_inferido"] = c.capilar_pct_da_area_micro_modelo
    t["capilar_pct_da_area_micro_diferenca_pontos"] = c.capilar_pct_da_area_micro_dif_pontos
    t["limiar_confianca_micro"] = t.lamina.map(inf.conf_micro)
    t["limiar_confianca_capilar"] = t.lamina.map(inf.conf_capilar)
    ordem = {"teste (ZAZA)": 0, "validacao (TOSTADA)": 1, "treino (4 eguas)": 2}
    t = t.sort_values(["grupo", "egua", "lamina"], key=lambda s: s.map(ordem) if s.name == "grupo" else s).round(2)
    save(t, "1_comparacao_por_lamina")

    # ---------------- 2) metricas
    medidas = [("micro", "contagem", "micro_contagem_anotada", "micro_contagem_inferida"),
               ("micro", "area (um2)", "micro_area_um2_anotada", "micro_area_um2_inferida"),
               ("micro", "% do campo", "micro_pct_do_campo_anotado", "micro_pct_do_campo_inferido"),
               ("capilar", "contagem", "capilar_contagem_anotada", "capilar_contagem_inferida"),
               ("capilar", "area (um2)", "capilar_area_um2_anotada", "capilar_area_um2_inferida"),
               ("capilar", "% da area micro", "capilar_pct_da_area_micro_anotado", "capilar_pct_da_area_micro_inferido")]
    rows = []
    cortes = [(g, t[t.grupo == g]) for g in ordem] + [("ZAZA + TOSTADA (60 laminas, nao vistas no treino)",
                                                      t[t.grupo.isin(["teste (ZAZA)", "validacao (TOSTADA)"])]),
                                                     ("todas as 180", t)]
    cortes += [(f"egua {e}", d) for e, d in t.groupby("egua")]
    for nome, d in cortes:
        for cls, med, ca, cm in medidas:
            rows.append({"conjunto": nome, "classe": cls, "medida": med} | metricas(d[ca], d[cm]))
    # as metricas sao para nos (secao de Resultados), nao para a veterinaria: ficam FORA da pasta de entrega
    pd.DataFrame(rows).round(3).to_csv(BASE / "metricas_reinhard_anotadas.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")

    # ---------------- 3) imagens comparativas
    dst = OUT / "3_imagens_comparativas_180"
    dst.mkdir(exist_ok=True)
    for f in (DVM / "imagens_comparativas" / VAR).glob("*.jpg"):
        shutil.copy2(f, dst / f.name)

    # ---------------- 4) 90 imagens novas (escala corrigida)
    d90 = OUT / "4_imagens_novas_90"
    (d90 / "imagens_lado_a_lado").mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(RUN1 / "pipeline" / VAR / "results.csv")
    n = pd.DataFrame({"imagem": r.image.str.replace(".jpg", "", regex=False)})
    n["animal"] = n.imagem.str.split().str[0]
    n["regiao"] = n.imagem.str.split().str[2]
    n["micro_contagem"] = r.micro_count
    n["micro_area_um2"] = r.micro_area_union_um2 * FATOR_ESCALA_90
    n["capilar_contagem"] = r.capilar_count
    n["capilar_area_um2"] = r.capilar_area_union_um2 * FATOR_ESCALA_90
    campo = 4140 * 3096 * (50 / 466) ** 2
    n["micro_pct_do_campo"] = 100 * n.micro_area_um2 / campo
    n["capilar_pct_da_area_micro"] = 100 * n.capilar_area_um2 / n.micro_area_um2
    n = n.round(2)
    n.to_csv(d90 / "medidas_por_imagem.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")
    for f in (RUN1 / "pipeline" / VAR / "lado_a_lado").glob("*.jpg"):
        shutil.copy2(f, d90 / "imagens_lado_a_lado" / f.name)
    print("pronto:", OUT)


if __name__ == "__main__":
    main()
