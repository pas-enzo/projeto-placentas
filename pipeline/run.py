from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from .capilar import CapilarModel
from .config import PipelineConfig
from .micro import MicrocotyledonModel
from .types import ImageResult
from .viz import colorize_overlay, masks_square_to_native, read_bgr, save_side_by_side

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def collect_images(input_path: Path) -> list[Path]:
    p = input_path.resolve()
    if p.is_file():
        if p.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image type: {p}")
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Input not found: {p}")
    files = sorted(
        x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        raise FileNotFoundError(f"No images under {p}")
    return files


def run_pipeline(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    mode: str = "both",
    cfg: PipelineConfig | None = None,
    save_overlays: bool = True,
    micro_conf: float | None = None,
    capilar_conf: float | None = None,
    device: str | None = None,
) -> list[ImageResult]:
    """Run histomorphometry inference on one image or a folder.

    mode: "micro" | "capilar" | "both"
    """
    cfg = cfg or PipelineConfig()
    mode = mode.lower().strip()
    if mode not in {"micro", "capilar", "both"}:
        raise ValueError("mode must be micro|capilar|both")

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    overlay_dir = out / "overlays"
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(Path(input_path))
    micro_model: MicrocotyledonModel | None = None
    cap_model: CapilarModel | None = None
    if mode in {"micro", "both"}:
        micro_model = MicrocotyledonModel(cfg, device=device)
    if mode in {"capilar", "both"}:
        cap_model = CapilarModel(cfg, device=device)

    results: list[ImageResult] = []
    for i, img_path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {img_path.name}", flush=True)
        bgr = read_bgr(img_path)
        h, w = bgr.shape[:2]
        res = ImageResult(image_path=img_path, width=w, height=h)

        micro_native = None
        if micro_model is not None:
            res.micro = micro_model.predict_bgr(bgr, conf=micro_conf)
            # Overlay needs native-resolution masks.
            micro_native = masks_square_to_native(
                res.micro.masks, native_w=w, native_h=h, canvas=cfg.micro_canvas
            )

        if cap_model is not None:
            res.capilar = cap_model.predict_path(img_path, conf=capilar_conf)

        if save_overlays:
            overlay = colorize_overlay(
                bgr,
                micro_masks=micro_native,
                capilar_masks=None if res.capilar is None else res.capilar.masks,
            )
            overlay_path = overlay_dir / f"{img_path.stem}_overlay.jpg"
            save_side_by_side(bgr, overlay, overlay_path)
            res.overlay_path = overlay_path

        results.append(res)

    csv_path = out / "results.csv"
    rows = [r.to_row() for r in results]
    if rows:
        fieldnames = list(rows[0].keys())
        # Union keys if modes differ (shouldn't, but safe).
        for row in rows[1:]:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {csv_path} ({len(results)} images)", flush=True)
    return results


def summarize(results: Sequence[ImageResult]) -> dict:
    n = len(results)
    out: dict = {"n_images": n}
    if n and results[0].micro is not None:
        out["micro_count_sum"] = sum(r.micro.count for r in results if r.micro)
        out["micro_area_um2_sum"] = round(
            sum(r.micro.area_um2 for r in results if r.micro), 4
        )
    if n and results[0].capilar is not None:
        out["capilar_count_sum"] = sum(r.capilar.count for r in results if r.capilar)
        out["capilar_area_um2_sum"] = round(
            sum(r.capilar.area_um2 for r in results if r.capilar), 4
        )
    return out
