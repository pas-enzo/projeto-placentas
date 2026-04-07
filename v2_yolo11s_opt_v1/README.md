# YOLO11s Optimization Iteration v1

This folder contains a clean, reproducible iteration of the YOLO11s segmentation workflow focused on:

- preserving high-resolution masks (`retina_masks=True`) for area metrics
- running baseline vs FP16 inference comparisons
- generating the same core validation artifacts:
  - per-instance report
  - image-level totals
  - IoU visualization images

## Structure

- `notebooks/yolo11s_opt_v1.ipynb`: primary workflow (recommended)
- `artifacts/`: all outputs for this iteration

## Recommended run mode (Notebook)

Open and run:

`v2_yolo11s_opt_v1/notebooks/yolo11s_opt_v1.ipynb`

It performs:

1. confidence sweep on validation set (post-training),
2. selection of best confidence for this specific model run,
3. baseline and FP16 reports,
4. IoU visualizations,
5. benchmark exports.

## Why confidence is not fixed in advance

Confidence should be selected after training because it varies by model checkpoint/run.
The notebook saves:

- `artifacts/benchmarks/conf_sweep.csv`
- `artifacts/benchmarks/selected_confidence.json`

and then uses that selected confidence for all downstream validations.
