"""Export a YOLO-seg view (images/ + labels/ per split) to Roboflow-style COCO.

Layout expected by RF-DETR (`dataset_file: roboflow`), Mask R-CNN / Mask2Former loaders and MMDetection:

    <out>/<split>/<image files>            (hard-linked from the view, no copy)
    <out>/<split>/_annotations.coco.json   category 0 = placeholder 'objects', real classes from id 1

Usage:
    python -m benchmark.views.coco_export --view D:/.../micro_stretch640 --out D:/.../coco/micro_stretch640
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "valid", "test")


def read_names(view: Path) -> list[str]:
    """Class names from the view's data.yaml (`names:` as an `id: name` map)."""
    names: dict[int, str] = {}
    in_names = False
    for line in (view / "data.yaml").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not line.startswith((" ", "\t")):
                break
            k, _, v = line.strip().partition(":")
            names[int(k)] = v.strip().strip("'\"")
    return [names[i] for i in sorted(names)]


def shoelace(pts: list[tuple[float, float]]) -> float:
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # same drive: no extra disk
    except OSError:
        shutil.copy2(src, dst)


def export_split(view: Path, out: Path, split: str, names: list[str]) -> dict:
    img_dir, lbl_dir = view / split / "images", view / split / "labels"
    dst = out / split
    dst.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    ann_id = 1
    for image_id, img in enumerate(sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)):
        h, w = cv2.imread(str(img), cv2.IMREAD_COLOR).shape[:2]
        link_or_copy(img, dst / img.name)
        images.append({"id": image_id, "license": 1, "file_name": img.name, "height": h, "width": w})
        lbl = lbl_dir / f"{img.stem}.txt"
        for line in lbl.read_text(encoding="utf-8").splitlines() if lbl.exists() else []:
            tok = line.split()
            if len(tok) < 7:
                continue
            cls = int(float(tok[0]))
            vals = [float(v) for v in tok[1:]]
            pts = [(vals[i] * w, vals[i + 1] * h) for i in range(0, len(vals) - 1, 2)]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x0, y0 = min(xs), min(ys)
            anns.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cls + 1,  # 0 is the Roboflow placeholder
                "bbox": [round(x0, 3), round(y0, 3), round(max(xs) - x0, 3), round(max(ys) - y0, 3)],
                "area": round(shoelace(pts), 3),
                "segmentation": [[round(c, 3) for p in pts for c in p]],
                "iscrowd": 0,
            })
            ann_id += 1
    categories = [{"id": 0, "name": "objects", "supercategory": "none"}] + [
        {"id": i + 1, "name": n, "supercategory": "objects"} for i, n in enumerate(names)
    ]
    doc = {
        "info": {"description": f"{view.name}/{split}", "date_created": dt.datetime.now().isoformat(timespec="seconds")},
        "licenses": [{"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}],
        "categories": categories,
        "images": images,
        "annotations": anns,
    }
    (dst / "_annotations.coco.json").write_text(json.dumps(doc), encoding="utf-8")
    return {"images": len(images), "annotations": len(anns)}


def export_view(view: Path, out: Path) -> dict:
    names = read_names(view)
    report = {split: export_split(view, out, split, names) for split in SPLITS if (view / split / "images").exists()}
    (out / "export_report.json").write_text(json.dumps({"view": str(view), "classes": names, **report}, indent=1),
                                           encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(export_view(args.view, args.out), indent=1))


if __name__ == "__main__":
    main()
