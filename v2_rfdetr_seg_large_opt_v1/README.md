# RF-DETR Seg Large Optimization Iteration v1

Transformer segmentation workflow upgraded from the 4GB `RFDETRSegSmall` baseline
(`v2_rfdetr_opt_v1`) to a larger, more robust backbone now that training runs on a
16GB GPU (RTX 5060 Ti).

## Folder

- `notebooks/rfdetr_seg_large_opt_v1_train_eval.ipynb`: preprocess + train + evaluate + benchmark
- `artifacts/`: generated runs, reports, IoU visualizations, preprocessing checks, and benchmark files

## What this notebook covers

1. Dataset preprocessing + COCO integrity validation (files, sizes, categories, instance stats).
2. Ground-truth mask preview overlays (sanity check before spending GPU hours).
3. Constrained training sweep on `RFDETRSegLarge`.
4. Per-run confidence sweep on validation data.
5. Champion run selection.
6. Per-instance and totals reports (area in px and µm²).
7. IoU overlay visualizations.
8. Inference-time studies (latency, throughput, peak VRAM).

## Model

Default is `RFDETRSegLarge` (`dec_layers=5`, `resolution=504`, DINOv2 backbone).
Change `SEG_MODEL_NAME` in the config cell to scale up/down across the available
segmentation variants:

| Variant | dec_layers | default res | notes |
|---------|-----------|-------------|-------|
| `RFDETRSegNano` | 4 | 312 | res divisible by 12 |
| `RFDETRSegSmall` | 4 | 384 | prior 4GB baseline |
| `RFDETRSegMedium` | 5 | 432 | |
| `RFDETRSegLarge` | 5 | 504 | **default here** |
| `RFDETRSegXLarge` | 6 | 624 | |
| `RFDETRSeg2XLarge` | 6 | 768 | heaviest |

Segmentation variants (except Nano) require `resolution` divisible by 24; the notebook
asserts this for every entry in `RUN_MATRIX`.

## Hardware notes

- Designed for ~16GB VRAM: real `batch_size` + `grad_accum_steps` + `gradient_checkpointing=True`.
- Mixed precision is handled internally by RF-DETR (`amp_dtype='auto'` → bf16 on this GPU).
- `tensorboard=False` in `train()` because the logger package is not installed in the `pytorch` env.

## Dataset

Uses the Roboflow COCO export at `dataset_v2_coco_rf_detr` (single class `microcotiledone`).
Path resolution is portable: it checks `PLACENTA_COCO_DIR`, then common layouts, and finally
`C:/Users/edual/projeto-placentas/datasets/dataset_v2_coco_rf_detr`. RF-DETR applies its own
resize + ImageNet normalization (tied to the pretrained backbone), so no manual normalization
is applied.

## Artifact contract

Kept identical to `v2_rfdetr_opt_v1` for cross-iteration comparability:

- `artifacts/benchmarks/validation_conf_sweep_<run>.csv`
- `artifacts/benchmarks/run_ranking.csv`
- `artifacts/benchmarks/champion_run.json`
- `artifacts/reports/<run>/placenta_instance_report_rfdetr.csv`
- `artifacts/reports/<run>/placenta_totals_report_rfdetr.csv`
- `artifacts/iou_viz/<run>/iou_viz_*.png`
- `artifacts/preprocessing/dataset_validation.json` and `gt_preview/*.png` (new in this iteration)
- `artifacts/benchmarks/inference_benchmark_<run>.json`
- `artifacts/benchmarks/final_summary.json`
