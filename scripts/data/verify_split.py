"""Fail loudly if any mare appears in more than one role.

Checks (1) the manifest itself, (2) the manifest against the source files, and optionally
(3) a built dataset view (images AND labels of train/valid/test) against a fold.

Usage:
  python scripts/data/verify_split.py                         # manifest + source
  python scripts/data/verify_split.py --view DIR --fold primary
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_common import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_SRC,
    IMAGE_EXTS,
    field_stem,
    fields_of,
    get_fold,
    load_manifest,
    manifest_hash,
    mare_of,
)

ROLE_DIR = {"train": "train", "val": "valid", "test": "test"}


class SplitError(AssertionError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SplitError(msg)


def verify_manifest(manifest: dict, src: Path | None) -> None:
    mares = set(manifest["mares"])
    # the hash is computed before the 'hash' key is added, so recompute it the same way
    core = {k: v for k, v in manifest.items() if k != "hash"}
    check(manifest["hash"] == manifest_hash(core), "hash do manifesto não confere (arquivo editado à mão?)")

    seen = collections.Counter(s for m in manifest["mares"].values() for s in m["fields"])
    check(all(c == 1 for c in seen.values()), f"campo repetido em éguas: {[s for s, c in seen.items() if c > 1]}")
    for stem in seen:
        check(mare_of(stem) in mares and stem in manifest["mares"][mare_of(stem)]["fields"], f"{stem} na égua errada")

    test_count, val_count = collections.Counter(), collections.Counter()
    for f in manifest["folds"]:
        roles = {r: set(f[r]) for r in ("train", "val", "test")}
        for a in roles:
            for b in roles:
                if a < b:
                    check(not (roles[a] & roles[b]), f"{f['name']}: égua em {a} e {b}: {roles[a] & roles[b]}")
        check(set().union(*roles.values()) == mares, f"{f['name']}: papéis não cobrem todas as éguas")
        test_count.update(f["test"])
        val_count.update(f["val"])
    check(all(test_count[m] == 1 for m in mares), f"cada égua deve ser teste uma vez: {dict(test_count)}")
    check(all(val_count[m] == 1 for m in mares), f"cada égua deve ser validação uma vez: {dict(val_count)}")

    if src is not None:
        found = set()
        for split in ("train", "valid", "test"):
            d = src / split / "images"
            if d.exists():
                found |= {field_stem(p.name) for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        check(found == set(seen), f"campos da origem != manifesto (faltam {sorted(set(seen) - found)[:3]}, "
                                  f"sobram {sorted(found - set(seen))[:3]})")
    print(f"OK manifesto: {len(seen)} campos, {len(mares)} éguas, {len(manifest['folds'])} folds, "
          f"sem égua em dois papéis (hash {manifest['hash']})")


def verify_view(manifest: dict, fold_name: str, view: Path, tiles_per_field: int | None = None) -> None:
    fold = get_fold(manifest, fold_name)
    for role, d in ROLE_DIR.items():
        expected_mares = set(fold[role])
        expected_fields = set(fields_of(manifest, fold[role]))
        for sub in ("images", "labels"):
            p = view / d / sub
            check(p.exists(), f"{p} não existe")
            names = [x for x in p.iterdir() if x.is_file()]
            stems = {field_stem(x.name) for x in names}
            check(stems <= expected_fields,
                  f"{view.name}/{d}/{sub}: campos fora do papel '{role}': {sorted(stems - expected_fields)[:3]}")
            check({mare_of(s) for s in stems} <= expected_mares,
                  f"{view.name}/{d}/{sub}: égua fora do papel '{role}'")
            check(stems == expected_fields,
                  f"{view.name}/{d}/{sub}: faltam campos: {sorted(expected_fields - stems)[:3]}")
            if tiles_per_field and sub == "images":
                check(len(names) == tiles_per_field * len(expected_fields),
                      f"{view.name}/{d}: {len(names)} ladrilhos != {tiles_per_field}x{len(expected_fields)}")
    print(f"OK vista {view}: nenhum vazamento entre éguas (fold {fold['name']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--no-src", action="store_true", help="não confere os arquivos da origem")
    ap.add_argument("--view", type=Path, default=None)
    ap.add_argument("--fold", default="primary")
    ap.add_argument("--tiles-per-field", type=int, default=None)
    args = ap.parse_args()
    manifest = load_manifest(args.manifest)
    try:
        verify_manifest(manifest, None if args.no_src else args.src)
        if args.view:
            verify_view(manifest, args.fold, args.view, args.tiles_per_field)
    except SplitError as e:
        print(f"FALHA: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
