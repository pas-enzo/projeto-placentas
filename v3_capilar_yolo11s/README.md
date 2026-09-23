# Pipeline 2 — Capilares (YOLO11s-seg + tiles)

Dataset externo (não versionado no git):

`D:/projeto_placentas_clayton/dataset_v3.1_yolo11_tiled_3x3`

## Como rodar

Abra e execute:

`v3_capilar_yolo11s/notebooks/yolo11s_capilar_train.ipynb`

O notebook:

1. treina só a classe `Capilar` (`classes=[0]`)
2. **v4** (run `capilar_yolo11s_tiled_v4`): YOLO11s campeão (mask mAP50 0,778)
3. **v5** (run `capilar_yolo26s_tiled_v1`): YOLO26s-seg; artefatos em `artifacts_yolo26s/`
4. faz confidence sweep no valid (tiles) e exporta F1 / IoU / erro de área

Modo atual do notebook: **`DO_TRAIN = False`**, sweep do **v4** em `artifacts_v4/` (não sobrescreve v3/26s). Para retreinar, `DO_TRAIN = True` e ajuste `run_name`.

## Calibração de área

Tiles estão em resolução nativa (pixel isotrópico). O fator é:

```text
AREA_FACTOR = (50/72)**2 * (640/4140)**2   # µm² / px² no espaço do tile
```

Não use o fator do pipeline 1 (canvas 640 esticado).

## Observações

- Treino em tile (`1380×1032` → letterbox 640). Contagem por campo completo exige merge de borda na inferência agregada (próximo passo).
- Artefatos de treino ficam em `v3_capilar_yolo11s/runs/` (gitignored).
