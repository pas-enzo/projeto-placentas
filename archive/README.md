# Archive

Experimentos e arquivos que **não** são o pipeline campeão do paper.
Mantidos para histórico / reprodutibilidade; não editar como caminho principal.

## `experiments/`

| Pasta | O que era |
|-------|-----------|
| `v2_yolo_early/` | YOLO11 early (`v2/`), export stretch, IoU viz iniciais |
| `v2.1_yolo11m/` | YOLO11m segment runs |
| `v2_rf_detr_early/` | Primeiro RF-DETR Seg Small no COCO v2 |
| `v2_rfdetr_seg_small/` | Otimização RF-DETR Seg Small (4GB) — baseline do Large |
| `v2_yolo11s_baseline/` | YOLO11s optimization / showcase (baseline micro) |
| `v2_yolo26_empty/` | Shell vazio YOLO26 (não usado) |
| `v2_newmodels_bench/` | Bancada de modelos alternativos |
| `root_runs/` | `runs/` que estava na raiz do repo |

## `root_notebooks/`

Notebooks soltos da raiz (`yolo_11_aug`, `yolo_model`, `test_packages`).

## `backups/`

- `um2_correction_20260907/` — CSVs pré-correção de área µm²
- `placenta_disparity_report.csv`

## `weights_base/`

Pesos `.pt` baixados / base (gitignored). Exemplos: `yolo11s-seg.pt`, `yolo26s-seg.pt`, `rf-detr-seg-large.pt`, YOLOv8\*.

## Campeões atuais (fora deste archive)

- Microcotiledone: `v2_rfdetr_seg_large_opt_v1/`
- Capilar: `v3_capilar_yolo11s/`
