"""Build training views of the native dataset for one fold of the mare-level split.

Single source of truth = native images + labels (og_size) + the split manifest. Views are
derived, reproducible and never edited by hand:

  native           copy of the native fields, grouped by role (train / valid / test)
  micro_stretch640 non-proportional stretch to 640x640 (same cv2.INTER_LINEAR as the pipeline),
                   only the microcotyledon class (remapped to id 0); labels are normalized so they
                   stay valid after the stretch
  capilar_tiles3x3 non-overlapping 3x3 grid of 1380x1032 tiles (polygons clipped per tile),
                   built AFTER the split so all nine tiles of a field share its role

Usage:
  python scripts/data/build_views.py --fold primary
  python scripts/data/build_views.py --fold test-ZAZA_val-TOSTADA --views micro,capilar
  python scripts/data/build_views.py --all-folds
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from split_common import (  # noqa: E402
    CLASS_CAPILAR,
    CLASS_MICRO,
    DEFAULT_MANIFEST,
    DEFAULT_SRC,
    IMAGE_EXTS,
    field_stem,
    fields_of,
    get_fold,
    load_manifest,
)
from verify_split import verify_manifest, verify_view  # noqa: E402

DEFAULT_OUT = Path(r"D:\projeto_placentas_clayton\datasets_v4")
ROLE_DIR = {"train": "train", "val": "valid", "test": "test"}
ALL_VIEWS = ("native", "micro", "capilar")


def _load_tiler():
    spec = importlib.util.spec_from_file_location("tile_yolo_dataset", HERE / "tile_yolo_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tile_yolo_dataset"] = mod
    spec.loader.exec_module(mod)
    return mod


def index_source(src: Path) -> dict[str, tuple[Path, Path | None]]:
    """field stem -> (image path, label path) across every split of the source dataset."""
    out: dict[str, tuple[Path, Path | None]] = {}
    for split in ("train", "valid", "test"):
        d = src / split / "images"
        if not d.exists():
            continue
        for img in d.iterdir():
            if img.suffix.lower() in IMAGE_EXTS:
                lbl = src / split / "labels" / f"{img.stem}.txt"
                out[field_stem(img.name)] = (img, lbl if lbl.exists() else None)
    return out


def write_yaml(dst: Path, names: list[str]) -> None:
    lines = [f"path: {dst.as_posix()}", "train: train/images", "val: valid/images", "test: test/images",
             f"nc: {len(names)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (dst / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def valid_label(line: str) -> bool:
    """A segmentation polygon needs a class id and >= 3 (x, y) points."""
    tok = line.split()
    return len(tok) >= 7 and (len(tok) - 1) % 2 == 0


def build_native(manifest: dict, fold: dict, source: dict, dst: Path) -> dict:
    """Copy the fields of each role. Degenerate polygons (< 3 points, i.e. lines) are dropped here, once,
    so every derived view (YOLO, tiles, COCO) carries exactly the same labels."""
    dropped = {"capilar": 0, "micro": 0}
    for role, d in ROLE_DIR.items():
        (dst / d / "images").mkdir(parents=True, exist_ok=True)
        (dst / d / "labels").mkdir(parents=True, exist_ok=True)
        for stem in fields_of(manifest, fold[role]):
            img, lbl = source[stem]
            shutil.copy2(img, dst / d / "images" / img.name)
            target = dst / d / "labels" / f"{img.stem}.txt"
            kept = []
            for line in (lbl.read_text(encoding="utf-8").splitlines() if lbl else []):
                if not line.strip():
                    continue
                if valid_label(line):
                    kept.append(line)
                else:
                    dropped["capilar" if int(float(line.split()[0])) == CLASS_CAPILAR else "micro"] += 1
            target.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    write_yaml(dst, ["Capilar", "microcotiledone"])
    return dropped


def build_micro(native: Path, dst: Path, size: int = 640) -> dict:
    stats = {"fields": 0, "instances": 0}
    for d in ROLE_DIR.values():
        (dst / d / "images").mkdir(parents=True, exist_ok=True)
        (dst / d / "labels").mkdir(parents=True, exist_ok=True)
        for img in sorted((native / d / "images").iterdir()):
            bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
            stretched = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(dst / d / "images" / f"{img.stem}.png"), stretched)
            lines = []
            lbl = native / d / "labels" / f"{img.stem}.txt"
            for line in lbl.read_text(encoding="utf-8").splitlines():
                tok = line.split()
                if tok and int(float(tok[0])) == CLASS_MICRO:
                    lines.append(" ".join(["0", *tok[1:]]))  # normalized coords: unchanged by the stretch
            (dst / d / "labels" / f"{img.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            stats["fields"] += 1
            stats["instances"] += len(lines)
    write_yaml(dst, ["microcotiledone"])
    return stats


def build_capilar(native: Path, dst: Path, grid=(3, 3)) -> dict:
    tiler = _load_tiler()
    stats = tiler.Stats()
    for d in ROLE_DIR.values():
        tiler.process_split(native / d, dst / d, grid[0], grid[1], 0.15, 64.0, 95, stats)
    tiler.write_data_yaml(dst, ["Capilar", "microcotiledone"])
    # same data.yaml layout as the other views (adds the test key)
    write_yaml(dst, ["Capilar", "microcotiledone"])
    return {"fields": stats.images_in, "tiles": stats.tiles_out, "polygons_kept": stats.polys_kept,
            "polygons_dropped_small": stats.polys_dropped_small}


def build_fold(manifest: dict, fold_name: str, src: Path, out_root: Path, views: set[str], force: bool) -> None:
    fold = get_fold(manifest, fold_name)
    root = out_root / fold["name"]
    if root.exists() and not force:
        raise SystemExit(f"{root} já existe (use --force para recriar)")
    if root.exists():
        shutil.rmtree(root)
    source = index_source(src)
    print(f"\n== fold {fold['name']}: train={fold['train']} val={fold['val']} test={fold['test']}")

    native = root / "native"
    dropped = build_native(manifest, fold, source, native)
    verify_view(manifest, fold["name"], native)
    report = {"fold": fold, "manifest_hash": manifest["hash"], "source": str(src),
              "dropped_degenerate_polygons": dropped, "views": {}}
    print(f"polígonos degenerados descartados: {dropped}")

    if "micro" in views:
        v = root / "micro_stretch640"
        report["views"]["micro_stretch640"] = build_micro(native, v)
        verify_view(manifest, fold["name"], v)
    if "capilar" in views:
        v = root / "capilar_tiles3x3"
        report["views"]["capilar_tiles3x3"] = build_capilar(native, v)
        verify_view(manifest, fold["name"], v, tiles_per_field=9)
    if "native" not in views:
        shutil.rmtree(native)

    (root / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["views"], ensure_ascii=False))


def build_norm(manifest: dict, fold_name: str, out_root: Path, methods: list[str], views: set[str],
               keep_native: bool, coco: bool, force: bool) -> None:
    """Add stain-normalized views (native_<m> -> micro_stretch640_<m> / capilar_tiles3x3_<m>) to an existing fold.

    The reference of every method is fitted on the TRAIN fields only and applied unchanged to all roles."""
    import time

    sys.path.insert(0, str(HERE.parents[1]))
    from benchmark.views.coco_export import export_view
    from benchmark.views.stain import StainNormalizer, normalize_dir

    fold = get_fold(manifest, fold_name)
    root = out_root / fold["name"]
    native = root / "native"
    if not native.exists():
        raise SystemExit(f"{native} não existe; gere primeiro as vistas base (sem --norm)")
    train = sorted((native / "train" / "images").iterdir())
    for m in methods:
        t0 = time.time()
        micro_dir, cap_dir = root / f"micro_stretch640_{m}", root / f"capilar_tiles3x3_{m}"
        done = (micro_dir.exists() or "micro" not in views) and (cap_dir.exists() or "capilar" not in views)
        if done and not force:
            print(f"[{m}] já existe (use --force para recriar)", flush=True)
            continue
        norm_native = root / f"native_{m}"
        for d in (norm_native, micro_dir, cap_dir):
            if d.exists():
                shutil.rmtree(d)
        print(f"\n[{m}] ajustando a referência com {len(train)} campos de treino...", flush=True)
        norm = StainNormalizer(m).fit(train)
        norm.save_reference(root / f"reference_{m}.json", train)
        print(f"[{m}] normalizando os campos nativos...", flush=True)
        rep = normalize_dir(native, norm_native, norm)
        verify_view(manifest, fold["name"], norm_native)
        report = {"method": m, "normalize": {k: v for k, v in rep.items() if k != "diagnostics"}, "views": {}}
        if "micro" in views:
            report["views"][micro_dir.name] = build_micro(norm_native, micro_dir)
            verify_view(manifest, fold["name"], micro_dir)
        if "capilar" in views:
            report["views"][cap_dir.name] = build_capilar(norm_native, cap_dir)
            verify_view(manifest, fold["name"], cap_dir, tiles_per_field=9)
        if coco:
            for v in (micro_dir, cap_dir):
                if v.exists():
                    export_view(v, root / "coco" / v.name)
        if not keep_native:
            shutil.rmtree(norm_native)
        report["seconds"] = round(time.time() - t0)
        (root / f"norm_report_{m}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{m}] pronto em {report['seconds']} s | imagens no plano B: {len(rep['fallbacks'])}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fold", default="primary")
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--views", default="native,micro,capilar", help="subset of native,micro,capilar")
    ap.add_argument("--force", action="store_true", help="recria uma pasta de fold já existente")
    ap.add_argument("--norm", default="", help="métodos de normalização (vírgula); ver benchmark/views/stain.py")
    ap.add_argument("--keep-native-norm", action="store_true", help="mantém native_<método> (ocupa ~GBs)")
    ap.add_argument("--coco", action="store_true", help="também exporta COCO das vistas geradas")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    verify_manifest(manifest, args.src)
    views = {v.strip() for v in args.views.split(",")}
    unknown = views - set(ALL_VIEWS)
    if unknown:
        raise SystemExit(f"vistas desconhecidas: {unknown}")
    names = [f["name"] for f in manifest["folds"]] if args.all_folds else [args.fold]
    methods = [m.strip() for m in args.norm.split(",") if m.strip()]
    if methods:
        for n in names:
            build_norm(manifest, n, args.out, methods, views, args.keep_native_norm, args.coco, args.force)
        return
    for n in names:
        build_fold(manifest, n, args.src, args.out, views, args.force)


if __name__ == "__main__":
    main()
