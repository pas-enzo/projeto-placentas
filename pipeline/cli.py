from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .run import run_pipeline, summarize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description=(
            "Equine placental histomorphometry: microcotyledon (RF-DETR) "
            "+ capillary (YOLO11s-seg / SAHI)."
        ),
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Image file or directory of FOV images (native resolution).",
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output directory (results.csv + overlays/).",
    )
    p.add_argument(
        "--mode",
        choices=("micro", "capilar", "both"),
        default="both",
        help="Which head(s) to run (default: both).",
    )
    p.add_argument("--micro-conf", type=float, default=None, help="Override micro conf* (default 0.44).")
    p.add_argument("--capilar-conf", type=float, default=None, help="Override capilar conf* (default 0.33).")
    p.add_argument("--micro-ckpt", type=str, default=None, help="Override RF-DETR checkpoint path.")
    p.add_argument("--capilar-ckpt", type=str, default=None, help="Override YOLO weights path.")
    p.add_argument("--device", type=str, default=None, help="cuda:0 | cpu (auto if omitted).")
    p.add_argument("--no-overlays", action="store_true", help="Skip side-by-side overlays.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PipelineConfig()
    if args.micro_ckpt:
        cfg.micro_ckpt = args.micro_ckpt
    if args.capilar_ckpt:
        cfg.capilar_ckpt = args.capilar_ckpt

    results = run_pipeline(
        args.input,
        args.output,
        mode=args.mode,
        cfg=cfg,
        save_overlays=not args.no_overlays,
        micro_conf=args.micro_conf,
        capilar_conf=args.capilar_conf,
        device=args.device,
    )
    summary = summarize(results)
    summary_path = Path(args.output).resolve() / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
