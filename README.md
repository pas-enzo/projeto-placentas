# projeto placentas — histomorfometria equina

Repositório do TCC/projeto: segmentação de **microcotilédones** (RF-DETR Seg Large) e **capilares** (YOLO11s-seg + SAHI), figuras e texto Methods/Results.

## Onde está o quê (raiz)

| Pasta | Papel |
|-------|-------|
| [`v2_rfdetr_seg_large_opt_v1/`](v2_rfdetr_seg_large_opt_v1/) | **Pipeline 1 — campeão** microcotiledone |
| [`v3_capilar_yolo11s/`](v3_capilar_yolo11s/) | **Pipeline 2 — campeão** capilar (tile + FOV/SAHI) |
| [`figs/`](figs/) | Figuras Methods/Results (JPGs, CSVs) |
| [`scripts/`](scripts/) | Utilitários (data, figs, results) |
| [`paper/`](paper/) | LaTeX do artigo |
| [`docs/`](docs/) | Inventário de experimentos |
| [`archive/`](archive/) | Experimentos legados, backups, pesos base |

Mapa detalhado run → notebook → artefatos: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Campeões (paper)

| Pipeline | Pasta | Notebook | Run | conf\* |
|----------|-------|----------|-----|-------|
| Microcotiledone | `v2_rfdetr_seg_large_opt_v1` | `notebooks/rfdetr_seg_large_opt_v1_train_eval.ipynb` | `seg_large_r672_auto` | 0.44 |
| Capilar (tile) | `v3_capilar_yolo11s` | `notebooks/01_yolo11s_capilar_train_v4.ipynb` | `capilar_yolo11s_tiled_v4` | 0.33 |
| Capilar (FOV) | idem | `notebooks/02_yolo11s_capilar_field_sahi.ipynb` | mesmo peso + SAHI | 0.33 |

## Scripts

```text
scripts/data/      # split / tiling de dataset
scripts/figs/      # gera fig2–fig4 em figs/
scripts/results/   # consolida métricas e plots em figs/results/
```

## O que vai no git

| Versionar | Não versionar |
|-----------|----------------|
| `figs/results/` (tabelas/plots do paper) | `**/iou_viz/`, `**/iou_imaging/` |
| `*/artifacts*/benchmarks/` e `reports/` (CSV/JSON leves) | `**/runs/`, `*.pt`, `*.pth` |
| Notebooks, scripts, READMEs | `archive/weights_base/*.pt` |

Overlays IoU regeneram pelos notebooks; métricas leves e figuras do paper ficam no repo.
