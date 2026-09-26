# RF-DETR Optimization Iteration v1

Notebook-first transformer optimization workflow for 4GB VRAM hardware.

## Folder

- `notebooks/rfdetr_opt_v1_train_eval.ipynb`: train + evaluate + benchmark
- `notebooks/select_best_model.ipynb`: champion-model selector from sweep artifacts
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

## Select champion model (notebook)

Open and run:
`v2_rfdetr_opt_v1/notebooks/select_best_model.ipynb`

This reads `validation_conf_sweep_*.csv`, ranks runs, and writes:

- `v2_rfdetr_opt_v1/artifacts/benchmarks/run_ranking.csv`
- `v2_rfdetr_opt_v1/artifacts/benchmarks/champion_run.json`

Academic quality-first default uses weighted quality metrics with speed weight `0.0`.
