"""Shared helpers for the mare-level (leakage-free) dataset split."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

DEFAULT_SRC = Path(r"D:\projeto_placentas_clayton\datasat_v3.1_yolo11_og_size")
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "splits" / "split_v1_por_egua.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_CAPILAR, CLASS_MICRO = 0, 1  # og_size data.yaml: names ['Capilar', 'microcotiledone']

_STEM_RE = re.compile(r"^(?P<mare>.+)-(?P<region>[BDG])_(?P<idx>\d+)$")


def field_stem(name: str) -> str:
    """'ROSILHA-M-B_011_jpg.rf.<hash>.jpg' (or a tile '..._r0c1') -> 'ROSILHA-M-B_011'."""
    stem = Path(name).stem
    stem = re.sub(r"_jpg\.rf\..*$", "", stem)
    stem = re.sub(r"_r\dc\d$", "", stem)  # tile suffix from tile_yolo_dataset.py
    return stem


def parse_field(stem: str) -> tuple[str, str]:
    """Field stem -> (mare, region). Raises if the naming convention is not matched."""
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Nome fora do padrão <egua>-<B|D|G>_<n>: {stem!r}")
    return m.group("mare"), m.group("region")


def mare_of(name: str) -> str:
    return parse_field(field_stem(name))[0]


def manifest_hash(manifest: dict) -> str:
    """Stable hash of the split content (ignores creation date)."""
    core = {k: v for k, v in manifest.items() if k != "created"}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_fold(manifest: dict, name: str) -> dict:
    if name == "primary":
        name = manifest["primary_fold"]
    for f in manifest["folds"]:
        if f["name"] == name:
            return f
    raise KeyError(f"fold {name!r} não existe; opções: {[f['name'] for f in manifest['folds']]}")


def fields_of(manifest: dict, mares: list[str]) -> list[str]:
    return sorted(s for m in mares for s in manifest["mares"][m]["fields"])
