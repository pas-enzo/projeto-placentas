"""
Tile a YOLO-seg dataset into a non-overlapping grid.

Designed for placenta capillary images at native 4140x3096:
  - default grid 3x3 -> tiles of 1380x1032
  - clips polygons to each tile (Sutherland-Hodgman)
  - drops tiny border scraps (fraction of original area)
  - keeps class ids and YOLO-seg polygon format
  - does NOT stretch tiles (pixels stay isotropic)

Usage:
  python tile_yolo_dataset.py ^
    --src D:/projeto_placentas_clayton/datasat_v3.1_yolo11_og_size ^
    --dst D:/projeto_placentas_clayton/dataset_v3.1_yolo11_tiled_3x3 ^
    --grid 3 3
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class Stats:
    images_in: int = 0
    tiles_out: int = 0
    empty_tiles: int = 0
    polys_in: int = 0
    polys_kept: int = 0
    polys_dropped_small: int = 0
    polys_dropped_no_intersect: int = 0
    area_in_by_class: dict | None = None
    area_out_by_class: dict | None = None

    def __post_init__(self):
        if self.area_in_by_class is None:
            self.area_in_by_class = {}
        if self.area_out_by_class is None:
            self.area_out_by_class = {}


def shoelace(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def clip_edge(
    pts: list[tuple[float, float]],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    edge: str,
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip against one axis-aligned edge."""

    def inside(p: tuple[float, float]) -> bool:
        x, y = p
        if edge == "left":
            return x >= x_min
        if edge == "right":
            return x <= x_max
        if edge == "top":
            return y >= y_min
        return y <= y_max  # bottom

    def intersect(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if edge == "left":
            t = 0.0 if dx == 0 else (x_min - x1) / dx
            return (x_min, y1 + t * dy)
        if edge == "right":
            t = 0.0 if dx == 0 else (x_max - x1) / dx
            return (x_max, y1 + t * dy)
        if edge == "top":
            t = 0.0 if dy == 0 else (y_min - y1) / dy
            return (x1 + t * dx, y_min)
        t = 0.0 if dy == 0 else (y_max - y1) / dy
        return (x1 + t * dx, y_max)

    if not pts:
        return []
    out: list[tuple[float, float]] = []
    prev = pts[-1]
    prev_in = inside(prev)
    for cur in pts:
        cur_in = inside(cur)
        if cur_in:
            if not prev_in:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect(prev, cur))
        prev, prev_in = cur, cur_in
    return out


def clip_poly_to_rect(
    pts: list[tuple[float, float]],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> list[tuple[float, float]]:
    for edge in ("left", "right", "top", "bottom"):
        pts = clip_edge(pts, x_min, y_min, x_max, y_max, edge)
        if not pts:
            return []
    # drop near-duplicate consecutive points
    cleaned: list[tuple[float, float]] = []
    for p in pts:
        if not cleaned or abs(cleaned[-1][0] - p[0]) > 1e-6 or abs(cleaned[-1][1] - p[1]) > 1e-6:
            cleaned.append(p)
    if len(cleaned) >= 2 and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-6 and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-6:
        cleaned.pop()
    return cleaned


def parse_yolo_seg_line(line: str, img_w: int, img_h: int):
    toks = line.strip().split()
    if len(toks) < 7 or (len(toks) - 1) % 2 != 0:
        return None
    cls = int(float(toks[0]))
    coords = [float(v) for v in toks[1:]]
    pts = []
    for i in range(0, len(coords), 2):
        pts.append((coords[i] * img_w, coords[i + 1] * img_h))
    return cls, pts


def format_yolo_seg_line(cls: int, pts: list[tuple[float, float]], tile_w: int, tile_h: int) -> str:
    parts = [str(cls)]
    for x, y in pts:
        # clamp tiny float noise to [0, 1]
        xn = min(1.0, max(0.0, x / tile_w))
        yn = min(1.0, max(0.0, y / tile_h))
        parts.append(f"{xn:.10f}")
        parts.append(f"{yn:.10f}")
    return " ".join(parts)


def process_split(
    src_split: Path,
    dst_split: Path,
    grid_x: int,
    grid_y: int,
    min_area_frac: float,
    min_area_px: float,
    jpeg_quality: int,
    stats: Stats,
) -> None:
    img_dir = src_split / "images"
    lbl_dir = src_split / "labels"
    out_img = dst_split / "images"
    out_lbl = dst_split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    for img_path in images:
        stats.images_in += 1
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            img_w, img_h = im.size
            if img_w % grid_x != 0 or img_h % grid_y != 0:
                raise ValueError(
                    f"{img_path.name}: size {img_w}x{img_h} not divisible by grid {grid_x}x{grid_y}"
                )
            tile_w = img_w // grid_x
            tile_h = img_h // grid_y

            label_path = lbl_dir / f"{img_path.stem}.txt"
            polys = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    parsed = parse_yolo_seg_line(line, img_w, img_h)
                    if parsed is None:
                        continue
                    cls, pts = parsed
                    area = shoelace(pts)
                    stats.polys_in += 1
                    stats.area_in_by_class[cls] = stats.area_in_by_class.get(cls, 0.0) + area
                    polys.append((cls, pts, area))

            stem = img_path.stem
            for ty in range(grid_y):
                for tx in range(grid_x):
                    x0 = tx * tile_w
                    y0 = ty * tile_h
                    x1 = x0 + tile_w
                    y1 = y0 + tile_h
                    tile = im.crop((x0, y0, x1, y1))
                    tile_name = f"{stem}_r{ty}c{tx}"
                    tile_img_path = out_img / f"{tile_name}.jpg"
                    tile_lbl_path = out_lbl / f"{tile_name}.txt"

                    kept_lines = []
                    for cls, pts, area_orig in polys:
                        clipped = clip_poly_to_rect(pts, x0, y0, x1, y1)
                        if len(clipped) < 3:
                            stats.polys_dropped_no_intersect += 1
                            continue
                        # shift into tile coordinates
                        local = [(x - x0, y - y0) for x, y in clipped]
                        area_clip = shoelace(local)
                        if area_clip < min_area_px or area_clip < (min_area_frac * area_orig):
                            stats.polys_dropped_small += 1
                            continue
                        kept_lines.append(format_yolo_seg_line(cls, local, tile_w, tile_h))
                        stats.polys_kept += 1
                        stats.area_out_by_class[cls] = stats.area_out_by_class.get(cls, 0.0) + area_clip

                    tile.save(tile_img_path, format="JPEG", quality=jpeg_quality, subsampling=0)
                    tile_lbl_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
                    stats.tiles_out += 1
                    if not kept_lines:
                        stats.empty_tiles += 1


def write_data_yaml(dst: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {dst.as_posix()}",
        "train: train/images",
        "val: valid/images",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    (dst / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_names_from_src_yaml(src: Path) -> list[str]:
    yaml_path = src / "data.yaml"
    if not yaml_path.exists():
        return ["Capilar", "microcotiledone"]
    text = yaml_path.read_text(encoding="utf-8")
    # support both list and map forms
    for line in text.splitlines():
        if line.strip().startswith("names:"):
            rest = line.split(":", 1)[1].strip()
            if rest.startswith("[") and rest.endswith("]"):
                inner = rest[1:-1]
                return [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
    names = []
    in_names = False
    for line in text.splitlines():
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not line.startswith(" ") and not line.startswith("\t"):
                break
            if ":" in line:
                names.append(line.split(":", 1)[1].strip().strip("'\""))
    return names or ["Capilar", "microcotiledone"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--grid", nargs=2, type=int, default=[3, 3], metavar=("GX", "GY"))
    ap.add_argument("--min-area-frac", type=float, default=0.15, help="keep clipped poly if area >= frac*original")
    ap.add_argument("--min-area-px", type=float, default=64.0, help="absolute min clipped area in px^2")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    grid_x, grid_y = args.grid
    if args.dst.exists():
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True, exist_ok=True)

    names = parse_names_from_src_yaml(args.src)
    stats = Stats()

    for split in ("train", "valid", "test"):
        src_split = args.src / split
        if not (src_split / "images").exists():
            continue
        process_split(
            src_split,
            args.dst / split,
            grid_x,
            grid_y,
            args.min_area_frac,
            args.min_area_px,
            args.jpeg_quality,
            stats,
        )

    write_data_yaml(args.dst, names)

    report = {
        "src": str(args.src),
        "dst": str(args.dst),
        "grid": [grid_x, grid_y],
        "min_area_frac": args.min_area_frac,
        "min_area_px": args.min_area_px,
        "images_in": stats.images_in,
        "tiles_out": stats.tiles_out,
        "empty_tiles": stats.empty_tiles,
        "polys_in": stats.polys_in,
        "polys_kept": stats.polys_kept,
        "polys_dropped_small": stats.polys_dropped_small,
        "polys_dropped_no_intersect": stats.polys_dropped_no_intersect,
        "area_in_by_class_px2": {str(k): v for k, v in sorted(stats.area_in_by_class.items())},
        "area_out_by_class_px2": {str(k): v for k, v in sorted(stats.area_out_by_class.items())},
        "area_conservation": {
            str(k): {
                "in": stats.area_in_by_class.get(k, 0.0),
                "out": stats.area_out_by_class.get(k, 0.0),
                "rel_diff": (
                    abs(stats.area_out_by_class.get(k, 0.0) - stats.area_in_by_class.get(k, 0.0))
                    / stats.area_in_by_class[k]
                    if stats.area_in_by_class.get(k, 0.0) > 0
                    else None
                ),
            }
            for k in sorted(set(stats.area_in_by_class) | set(stats.area_out_by_class))
        },
        "class_names": names,
        "notes": [
            "Tiles saved at native tile resolution (no stretch).",
            "Border scraps below min_area_frac/min_area_px are dropped and cause small area loss.",
            "For inference counting across tiles, merge instances that touch shared borders.",
        ],
    }
    (args.dst / "tiling_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
