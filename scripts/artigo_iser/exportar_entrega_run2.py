"""Entrega das 150 imagens externas com coloracao boa (run2), so Reinhard A, no mesmo formato de
ENTREGA_REINHARD/4_imagens_novas_90 (exportar_entrega_reinhard.py, secao 4). Le o que ja existe.

Saida: temp_ml/artigo_iser/ENTREGA_REINHARD_150/
  medidas_por_imagem.csv          150 linhas, formato brasileiro (; entre colunas, virgula decimal)
  resumo_por_egua_regiao.csv      media e desvio-padrao por egua x regiao
  imagens_lado_a_lado/            entrada | modelo
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser")
RUN2 = BASE / "run2"
OUT = BASE / "ENTREGA_REINHARD_150"
VAR = "A_reinhard"
FATOR_ESCALA = (469 / 466) ** 2   # barra de 50 um = 466 px nas 150 imagens (469 nas 180 anotadas)
BR = dict(index=False, encoding="utf-8-sig", sep=";", decimal=",")


def main() -> None:
    (OUT / "imagens_lado_a_lado").mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(RUN2 / "pipeline" / VAR / "results.csv")
    n = pd.DataFrame({"imagem": r.image.str.replace(".jpg", "", regex=False)})
    t = n.imagem.str.split()
    n["egua"] = t.str[0] + " " + t.str[1]
    n["regiao"] = t.str[2]
    n["campo"] = t.str[3].astype(int)
    n["micro_contagem"] = r.micro_count
    n["micro_area_um2"] = r.micro_area_union_um2 * FATOR_ESCALA
    n["capilar_contagem"] = r.capilar_count
    n["capilar_area_um2"] = r.capilar_area_union_um2 * FATOR_ESCALA
    campo = 4140 * 3096 * (50 / 466) ** 2
    n["micro_pct_do_campo"] = 100 * n.micro_area_um2 / campo
    n["capilar_pct_da_area_micro"] = 100 * n.capilar_area_um2 / n.micro_area_um2
    n = n.sort_values(["egua", "regiao", "campo"]).round(2)
    n.to_csv(OUT / "medidas_por_imagem.csv", **BR)

    cols = ["micro_contagem", "micro_area_um2", "micro_pct_do_campo",
            "capilar_contagem", "capilar_area_um2", "capilar_pct_da_area_micro"]
    g = n.groupby(["egua", "regiao"])[cols].agg(["mean", "std"]).round(2)
    g.columns = [f"{c}_{'media' if s == 'mean' else 'dp'}" for c, s in g.columns]
    g.insert(0, "n_campos", n.groupby(["egua", "regiao"]).size())
    g.reset_index().to_csv(OUT / "resumo_por_egua_regiao.csv", **BR)

    for f in (RUN2 / "pipeline" / VAR / "lado_a_lado").glob("*.jpg"):
        shutil.copy2(f, OUT / "imagens_lado_a_lado" / f.name)
    print("pronto:", OUT, len(n), "imagens")


if __name__ == "__main__":
    main()
