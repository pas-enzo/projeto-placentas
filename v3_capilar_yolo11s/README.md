# Pipeline 2 — Capilares (YOLO11s-seg + tiles)

Dataset externo (não versionado no git):

`D:/projeto_placentas_clayton/dataset_v3.1_yolo11_tiled_3x3`

## Como rodar

Abra e execute:

`v3_capilar_yolo11s/notebooks/yolo11s_capilar_train.ipynb`

O notebook:

1. treina **YOLO11s-seg** só na classe `Capilar` (`classes=[0]`, ignora `microcotiledone`)
2. faz confidence sweep no valid (tiles)
3. exporta F1 / IoU / erro de área e o melhor `conf`

## Calibração de área

Tiles estão em resolução nativa (pixel isotrópico). O fator é:

```text
AREA_FACTOR = (50/72)**2 * (640/4140)**2   # µm² / px² no espaço do tile
```

Não use o fator do pipeline 1 (canvas 640 esticado).

## Observações

- Treino em tile (`1380×1032` → letterbox 640). Contagem por campo completo exige merge de borda na inferência agregada (próximo passo).
- Artefatos de treino ficam em `v3_capilar_yolo11s/runs/` (gitignored).
