"""Core of the benchmark runner: run specification, stable run ids, on-disk state and dataset resolution.

State lives on disk only (no database): per run a folder `<results>/<run_id>/` with

    config.json        the run specification (hash is part of the id)
    status.json        state: pending | running | interrupted | failed | done, current stage, timings
    train.done / eval_val.done / eval_test.done   stage markers, written when the stage is complete
    DONE               written LAST, after run.json: only then is the run complete
    run.json           everything the article needs (config, versions, hardware, timings, metrics)

Everything is written atomically (temp file + rename), so an interruption never leaves a half-written file.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = Path(r"D:\projeto_placentas_clayton\datasets_v4")
RESULTS = Path(r"D:\projeto_placentas_clayton\benchmark_results")
sys.path.insert(0, str(REPO / "scripts" / "data"))

TASKS = ("micro", "capilar")
LISTABLE = ("task", "model", "norm", "seed", "fold")  # keys that expand into a cartesian product


@dataclass
class RunSpec:
    task: str                      # micro | capilar
    family: str                    # ultralytics | ...
    model: str                     # weights name/path or architecture id
    norm: str = "none"             # see benchmark/views/stain.py
    fold: str = "primary"
    seed: int = 0
    epochs: int = 100
    patience: int = 30
    imgsz: int = 640
    batch: int = 2
    recipe: str = "default"        # augmentation/loss recipe, see benchmark/recipes.py
    extra: dict = field(default_factory=dict)  # family-specific training arguments
    eval: dict = field(default_factory=dict)   # e.g. {"max_fields": 2} for smoke tests
    tags: list = field(default_factory=list)

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task deve ser {TASKS}")

    def config(self) -> dict:
        """What defines the experiment (tags and eval limits are bookkeeping, not part of identity... except eval)."""
        d = asdict(self)
        d.pop("tags")
        return d

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.config(), sort_keys=True).encode("utf-8")).hexdigest()[:8]

    def run_id(self) -> str:
        stem = Path(self.model).stem
        fold = "" if self.fold == "primary" else f"__{self.fold}"
        return f"{self.task}__{self.family}__{stem}__{self.norm}{fold}__s{self.seed}__{self.hash()}"


def expand_matrix(cfg: dict) -> list[RunSpec]:
    """`defaults` + `runs`; list values of LISTABLE keys expand into every combination."""
    defaults = cfg.get("defaults", {})
    specs: list[RunSpec] = []
    for entry in cfg["runs"]:
        d = {**defaults, **entry}
        keys = [k for k in LISTABLE if isinstance(d.get(k), list)]
        for combo in itertools.product(*[d[k] for k in keys]) if keys else [()]:
            specs.append(RunSpec(**{**d, **dict(zip(keys, combo))}))
    seen = set()
    for s in specs:
        if s.run_id() in seen:
            raise ValueError(f"execução duplicada na matriz: {s.run_id()}")
        seen.add(s.run_id())
    return specs


# ------------------------------------------------------------------ atomic json / state
def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def mark(run_dir: Path, name: str) -> None:
    (run_dir / name).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def is_done(run_dir: Path) -> bool:
    return (run_dir / "DONE").exists()


def set_status(run_dir: Path, state: str, **fields) -> dict:
    st = read_json(run_dir / "status.json", {})
    st.update({"state": state, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), **fields})
    write_json(run_dir / "status.json", st)
    return st


# ------------------------------------------------------------------ datasets
def fold_name(fold: str) -> str:
    from split_common import get_fold, load_manifest

    return get_fold(load_manifest(), fold)["name"]


def view_dir(spec: RunSpec, coco: bool = False) -> Path:
    """Directory of the dataset view this run trains on."""
    base = "micro_stretch640" if spec.task == "micro" else "capilar_tiles3x3"
    suffix = "" if spec.norm == "none" else f"_{spec.norm}"
    root = DATASETS / fold_name(spec.fold)
    return (root / "coco" / f"{base}{suffix}") if coco else (root / f"{base}{suffix}")


def native_dir(spec: RunSpec) -> Path:
    return DATASETS / fold_name(spec.fold) / "native"


# ------------------------------------------------------------------ environment record
def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def versions() -> dict:
    from importlib import metadata

    out = {"python": sys.version.split()[0]}
    for p in ("torch", "torchvision", "ultralytics", "rfdetr", "sahi", "torchstain", "staintools", "scikit-image",
              "pycocotools", "numpy", "opencv-python"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def hardware() -> dict:
    import torch

    hw = {"cuda": torch.cuda.is_available()}
    if hw["cuda"]:
        p = torch.cuda.get_device_properties(0)
        hw.update({"gpu": p.name, "vram_gb": round(p.total_memory / 1e9, 1)})
    return hw
