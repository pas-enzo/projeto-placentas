# Inventário de experimentos

## Ativos (paper)

| Pipeline | Pasta | Notebook(s) | Run campeão | Artefatos | conf\* |
|----------|-------|-------------|-------------|-----------|--------|
| Microcotiledone | `v2_rfdetr_seg_large_opt_v1` | `notebooks/rfdetr_seg_large_opt_v1_train_eval.ipynb` | `seg_large_r672_auto` | `artifacts/` | 0.44 |
| Microcotiledone (ablation) | idem | mesmo notebook (`RUN_MATRIX`) | `seg_large_r504_auto` | `artifacts/` | — |
| Capilar tile | `v3_capilar_yolo11s` | `notebooks/01_yolo11s_capilar_train_v4.ipynb` | `capilar_yolo11s_tiled_v4` | `artifacts_v4/` | 0.33 |
| Capilar FOV+SAHI | idem | `notebooks/02_yolo11s_capilar_field_sahi.ipynb` | mesmo peso | `artifacts_v4_field/` | 0.33 |
| Capilar Macenko | idem | `notebooks/03_yolo11s_capilar_macenko.ipynb` | `capilar_yolo11s_tiled_macenko_v1` | `artifacts_macenko/` | — |
| Capilar YOLO26s | idem | `notebooks/04_yolo26s_capilar_train.ipynb` | `capilar_yolo26s_tiled_v1` | `artifacts_yolo26s/` | 0.33 |

### Capilar — parâmetros SAHI (morphometry)

- Slice: `1380×1032`
- Overlap: `0.2`
- Postprocess: `GREEDYNMM`, match metric `IOS`, threshold `0.5`

### Fatores de área

- Microcotiledone (stretch 640): \(\approx 0.3606\,\mu m^2/px^2\) (anisotropia)
- Capilar (tile / SAHI nativo): \(\approx 0.0115\,\mu m^2/px^2\) (isotrópico)

## Arquivados

Ver [`archive/README.md`](../archive/README.md).

## Figuras Results

Geradas / consolidadas sob `figs/results/` pelos scripts em `scripts/results/`.
