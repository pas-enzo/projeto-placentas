# Pipeline 2 — Capilares (YOLO11s-seg + tiles)

Dataset externo (não versionado no git):

`D:/projeto_placentas_clayton/dataset_v3.1_yolo11_tiled_3x3`

## Notebooks (ordem)

| # | Notebook | Papel | Run / artefatos |
|---|----------|-------|-----------------|
| 01 | `notebooks/01_yolo11s_capilar_train_v4.ipynb` | **Campeão** tile train/eval | `runs/capilar_yolo11s_tiled_v4` → `artifacts_v4/` |
| 02 | `notebooks/02_yolo11s_capilar_field_sahi.ipynb` | Morphometry FOV + SAHI | mesmo peso → `artifacts_v4_field/` |
| 03 | `notebooks/03_yolo11s_capilar_macenko.ipynb` | Ablation Macenko | `…_macenko_v1` → `artifacts_macenko/` |
| 04 | `notebooks/04_yolo26s_capilar_train.ipynb` | Ablation YOLO26s | `capilar_yolo26s_tiled_v1` → `artifacts_yolo26s/` |

Paper / Results usam **01 + 02** (conf\* tile = 0.33; métricas morphometricas no FOV via SAHI).

## Como rodar (campeão)

1. Abra `01_yolo11s_capilar_train_v4.ipynb` (`DO_TRAIN = False` para só avaliar o `best.pt` do v4).
2. Para morphometry por campo: `02_yolo11s_capilar_field_sahi.ipynb` (slice `1380×1032`, overlap `0.2`, GREEDYNMM/IOS `0.5`).

Pesos base baixados ficam em `archive/weights_base/` (gitignored).

## Calibração de área

Tiles / SAHI em resolução nativa (pixel isotrópico):

```text
AREA_FACTOR = (50/72)**2 * (640/4140)**2   # µm² / px²
```

Não use o fator do pipeline 1 (canvas 640 esticado).

## Observações

- Treino em tile (`1380×1032` → letterbox 640). Contagem/área por FOV: notebook 02 (SAHI).
- Artefatos de treino em `v3_capilar_yolo11s/runs/` (gitignored).
- YAMLs: `data_capilar_tiled.yaml`, `data_capilar_tiled_macenko.yaml`.
