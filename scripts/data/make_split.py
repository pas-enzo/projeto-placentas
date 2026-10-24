"""Build the mare-level split manifest (no mare appears in two roles).

The 180 native fields (6 mares x 30 fields, 10 per region) are grouped by mare. Folds rotate the
roles: fold i has test = order[i], val = order[i+1], train = the other four mares (120/30/30 fields).
The `primary` fold is the fixed split for fast benchmarking; the 6 folds give mean +- sd across
animals for the finalists.

Usage: python scripts/data/make_split.py [--src DIR] [--out manifest.json]
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_common import (  # noqa: E402
    CLASS_CAPILAR,
    CLASS_MICRO,
    DEFAULT_MANIFEST,
    DEFAULT_SRC,
    IMAGE_EXTS,
    field_stem,
    manifest_hash,
    parse_field,
)

# Rotation order: primary fold = (test ZAZA, val TOSTADA). TORDILHA (densest mare) is placed so it is
# never in the primary fold's val/test roles.
DEFAULT_ORDER = ["ZAZA", "TOSTADA", "ROSILHA-M", "TORDILHA", "TP", "ZAINA-679"]


def scan(src: Path) -> dict[str, dict]:
    """field stem -> {mare, region, n_capilar, n_micro, source_split} from every split of the source."""
    fields: dict[str, dict] = {}
    for split in ("train", "valid", "test"):
        img_dir, lbl_dir = src / split / "images", src / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            stem = field_stem(img.name)
            if stem in fields:
                raise SystemExit(f"Campo duplicado entre splits da origem: {stem}")
            mare, region = parse_field(stem)
            counts = collections.Counter()
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        counts[int(float(line.split()[0]))] += 1
            fields[stem] = {
                "mare": mare,
                "region": region,
                "n_capilar": counts[CLASS_CAPILAR],
                "n_micro": counts[CLASS_MICRO],
                "source_split": split,
            }
    return fields


def build_manifest(src: Path, order: list[str]) -> dict:
    fields = scan(src)
    mares: dict[str, dict] = {}
    for stem, f in sorted(fields.items()):
        m = mares.setdefault(
            f["mare"], {"fields": [], "n_fields": 0, "instances": {"capilar": 0, "micro": 0}, "regions": {}}
        )
        m["fields"].append(stem)
        m["n_fields"] += 1
        m["instances"]["capilar"] += f["n_capilar"]
        m["instances"]["micro"] += f["n_micro"]
        m["regions"][f["region"]] = m["regions"].get(f["region"], 0) + 1
    if set(order) != set(mares):
        raise SystemExit(f"order {order} não coincide com as éguas encontradas {sorted(mares)}")
    sizes = {m: v["n_fields"] for m, v in mares.items()}
    if len(set(sizes.values())) != 1:
        print(f"AVISO: éguas com números de campos diferentes: {sizes}")

    folds = []
    n = len(order)
    for i in range(n):
        test, val = order[i], order[(i + 1) % n]
        train = [m for m in order if m not in (test, val)]
        folds.append({"name": f"test-{test}_val-{val}", "train": train, "val": [val], "test": [test]})

    manifest = {
        "version": 1,
        "description": "Divisão por égua (sem vazamento entre animais). Papéis por rodízio 4/1/1.",
        "source": str(src),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "order": order,
        "primary_fold": folds[0]["name"],
        "mares": mares,
        "folds": folds,
    }
    manifest["hash"] = manifest_hash(manifest)
    return manifest


def report(manifest: dict) -> None:
    mares = manifest["mares"]
    tot_f = sum(v["n_fields"] for v in mares.values())
    tot_c = sum(v["instances"]["capilar"] for v in mares.values())
    tot_m = sum(v["instances"]["micro"] for v in mares.values())
    print(f"{tot_f} campos | {len(mares)} éguas | capilares {tot_c} | micros {tot_m}\n")
    print(f"{'égua':12s}{'campos':>7s}{'capilares':>10s}{'%':>7s}{'micros':>8s}{'%':>7s}")
    for m in manifest["order"]:
        v = mares[m]
        print(f"{m:12s}{v['n_fields']:7d}{v['instances']['capilar']:10d}{100*v['instances']['capilar']/tot_c:6.1f}%"
              f"{v['instances']['micro']:8d}{100*v['instances']['micro']/tot_m:6.1f}%")
    print("\nFolds (% das instâncias capilar / micro em cada papel):")
    for f in manifest["folds"]:
        row = []
        for role in ("train", "val", "test"):
            c = sum(mares[m]["instances"]["capilar"] for m in f[role]) / tot_c * 100
            mi = sum(mares[m]["instances"]["micro"] for m in f[role]) / tot_m * 100
            row.append(f"{role} {c:4.1f}/{mi:4.1f}")
        mark = "  <- primary" if f["name"] == manifest["primary_fold"] else ""
        print(f"  {f['name']:34s}" + " | ".join(row) + mark)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--order", nargs="+", default=DEFAULT_ORDER)
    args = ap.parse_args()
    manifest = build_manifest(args.src, args.order)
    report(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifesto salvo em {args.out} (hash {manifest['hash']})")


if __name__ == "__main__":
    main()
