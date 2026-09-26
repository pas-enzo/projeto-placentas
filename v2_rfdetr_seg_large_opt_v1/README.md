# RF-DETR Seg Large — Pipeline 1 (microcotiledones)

Champion workspace for automated microcotyledon instance segmentation
(Roboflow stretch export 640×640 → RF-DETR Seg Large).

Baseline RF-DETR Seg Small (4GB) is archived at
`archive/experiments/v2_rfdetr_seg_small/`.

## Folder

- `notebooks/rfdetr_seg_large_opt_v1_train_eval.ipynb`: preprocess + train + evaluate + benchmark
- `artifacts/`: runs, reports, IoU visualizations, preprocessing checks, benchmarks

## Runs in the notebook

| Run | Role | Notes |
|-----|------|-------|
| `seg_large_r504_auto` | Ablation | input resolution 504 px |
| `seg_large_r672_auto` | **Champion** (paper) | input 672 px; conf\* = 0.44 |

## What this notebook covers

1. Dataset preprocessing + COCO integrity validation.
2. Ground-truth mask preview overlays.
3. Training sweep on `RFDETRSegLarge` (`RUN_MATRIX`: r504 + r672).
4. Per-run confidence sweep on validation data.
5. Champion run selection (`seg_large_r672_auto`).
6. Per-instance and totals reports (area in px and µm²; anisotropic stretch factor).
7. IoU overlay visualizations.
8. Inference-time studies (latency, throughput, peak VRAM).

## Model

`RFDETRSegLarge` (DINOv2 backbone, `dec_layers=5`). Other variants (Nano→2XLarge)
can be selected via `SEG_MODEL_NAME` in the config cell; resolution must be
divisible by 24 (except Nano).

## Hardware notes

- Designed for ~16GB VRAM: batch + grad accum + `gradient_checkpointing=True`.
- Mixed precision via RF-DETR (`amp_dtype='auto'`).

## Dataset

Roboflow COCO export `dataset_v2_coco_rf_detr` (class `microcotiledone`), outside git.
Path resolution: `PLACENTA_COCO_DIR`, then common layouts.

## Artifact contract

- `artifacts/benchmarks/validation_conf_sweep_<run>.csv`
- `artifacts/benchmarks/run_ranking.csv`
- `artifacts/benchmarks/champion_run.json`
- `artifacts/reports/<run>/placenta_instance_report_rfdetr.csv`
- `artifacts/reports/<run>/placenta_totals_report_rfdetr.csv`
- `artifacts/iou_viz/<run>/iou_viz_*.png`
- `artifacts/preprocessing/dataset_validation.json` and `gt_preview/*.png`
- `artifacts/benchmarks/inference_benchmark_<run>.json`
- `artifacts/benchmarks/final_summary.json`
