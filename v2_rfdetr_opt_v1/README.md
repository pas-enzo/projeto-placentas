# RF-DETR Optimization Iteration v1

Notebook-first transformer optimization workflow for 4GB VRAM hardware.

## Folder

- `notebooks/rfdetr_opt_v1_train_eval.ipynb`: train + evaluate + benchmark
- `artifacts/`: generated runs, reports, IoU visualizations, and benchmark files

## What this notebook covers

1. Multi-run constrained sweep (RFDETRSegSmall).
2. Per-run confidence sweep on validation data.
3. Champion run selection.
4. Per-instance and totals reports.
5. IoU overlay visualizations.
6. Inference-time studies (latency, throughput, peak VRAM).

## Notes

- Designed for `batch_size=1` + gradient accumulation + checkpointing.
- Keeps segmentation-first evaluation for area-sensitive metrics.
- Uses relative paths from repo root whenever possible.
