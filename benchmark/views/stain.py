"""Stain / color normalization of native fields, using established libraries.

Methods (names used by `--norm` in scripts/data/build_views.py) and where each comes from:
  none         identity (baseline)
  macenko      torchstain (numpy backend) MacenkoNormalizer   - Macenko et al., ISBI 2009
  reinhard     torchstain (numpy backend) ReinhardNormalizer  - Reinhard et al., IEEE CG&A 2001
  vahadane     staintools (spams) VahadaneStainNormalizer     - Vahadane et al., IEEE TMI 2016
  histmatch    skimage.exposure.match_histograms              - per-channel histogram matching
  whitebalance background white balance (small own function: bright, low-saturation pixels -> neutral)
  grayscale    OpenCV luminance replicated to 3 channels      - ablation: how much does color matter?

These libraries take ONE reference image. For a reference that comes from the TRAIN mares only (never from
validation/test), the reference is a *mosaic* of downscaled train fields: Macenko/Reinhard/histogram
statistics are then estimated over the pooled pixels of all training animals, not over a single slide.
The same fitted reference is applied to train, validation and test.

CLI:
  python -m benchmark.views.stain --native <fold>/native --report            # color spread between mares
  python -m benchmark.views.stain --native <fold>/native --preview grid.jpg  # same crop under each method
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_METHODS = ("none", "whitebalance", "reinhard", "macenko", "vahadane", "histmatch", "grayscale")
# Chains run left to right ("whitebalance+macenko": neutralize the background cast, then Macenko).
METHODS = BASE_METHODS + ("whitebalance+macenko", "whitebalance+reinhard", "whitebalance+vahadane")
MOSAIC_CELL = (368, 276)  # (w, h) of each downscaled train field inside the reference mosaic
MOSAIC_COLS = 12


# ------------------------------------------------------------------ io helpers
def _read_rgb(path: Path, reduced: int = 1) -> np.ndarray:
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
            8: cv2.IMREAD_REDUCED_COLOR_8}[reduced]
    bgr = cv2.imread(str(path), flag)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def reference_mosaic(train_paths: list[Path]) -> np.ndarray:
    """Grid of downscaled TRAIN fields (RGB uint8) used as the single 'reference image' of the libraries."""
    cells = [cv2.resize(_read_rgb(p, 8), MOSAIC_CELL, interpolation=cv2.INTER_AREA) for p in train_paths]
    while len(cells) % MOSAIC_COLS:
        cells.append(cells[len(cells) % len(train_paths)])  # pad with repeats so the grid is rectangular
    rows = [np.hstack(cells[i:i + MOSAIC_COLS]) for i in range(0, len(cells), MOSAIC_COLS)]
    return np.vstack(rows)


# ------------------------------------------------------------------ white balance (own, tiny)
def _background_color(rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(rgb, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    mx, mn = small.max(-1).astype(np.int16), small.min(-1).astype(np.int16)
    m = (mn > 150) & ((mx - mn) < 60)
    px = small[m] if m.sum() > 500 else small.reshape(-1, 3)[np.argsort(small.reshape(-1, 3).sum(1))[-2000:]]
    return np.median(px, axis=0)


# ------------------------------------------------------------------ fit / apply
class _Step:
    """One method, fitted on the reference mosaic (see StainNormalizer for chains)."""

    def __init__(self, method: str):
        self.method = method
        self.ref: dict = {}
        self._lib = None
        self._mosaic = None

    def fit_mosaic(self, mosaic: np.ndarray | None, saved: dict | None = None) -> None:
        """`saved`: reference stored by a previous fit; used to RESTORE the exact same Vahadane reference
        (its dictionary learning is not guaranteed to be reproducible across runs)."""
        m = self.method
        if m in ("none", "grayscale"):
            return
        if m == "whitebalance":
            self.ref = {"target_luminance": float(np.mean(_background_color(mosaic)))}
        elif m == "histmatch":
            self._mosaic = mosaic
        elif m == "vahadane":
            import staintools

            s = 2400 / max(mosaic.shape[:2])  # the stain matrix is global: a smaller mosaic is enough
            small = mosaic if s >= 1 else cv2.resize(mosaic, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            self._lib = staintools.StainNormalizer(method="vahadane")
            target = staintools.LuminosityStandardizer.standardize(small)
            if saved and "stain_matrix" in saved:
                # restore: same stain matrix as the original fit; maxC_target is a deterministic function of it
                from staintools.miscellaneous.get_concentrations import get_concentrations

                self._lib.stain_matrix_target = np.asarray(saved["stain_matrix"])
                conc = get_concentrations(target, self._lib.stain_matrix_target)
                self._lib.maxC_target = np.percentile(conc, 99, axis=0).reshape((1, 2))
            else:
                self._lib.fit(target)
            sm = getattr(self._lib, "stain_matrix_target", None)
            self.ref = {"stain_matrix": np.asarray(sm).tolist()} if sm is not None else {}
        else:
            import torchstain

            cls = torchstain.normalizers.MacenkoNormalizer if m == "macenko" else torchstain.normalizers.ReinhardNormalizer
            self._lib = cls(backend="numpy")
            self._lib.fit(mosaic)
            if m == "macenko":
                self.ref = {"HERef": np.asarray(self._lib.HERef).tolist(), "maxCRef": np.asarray(self._lib.maxCRef).tolist()}
            else:
                self.ref = {"target_means": np.asarray(self._lib.target_means).tolist(),
                            "target_stds": np.asarray(self._lib.target_stds).tolist()}

    def transform(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        """(normalized RGB uint8, diagnostics). On failure the image is returned unchanged and flagged."""
        m, diag = self.method, {}
        try:
            if m == "none":
                return rgb, diag
            if m == "grayscale":
                g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                return cv2.merge([g, g, g]), diag
            if m == "whitebalance":
                gain = self.ref["target_luminance"] / np.maximum(_background_color(rgb), 1.0)
                diag["gain_rgb"] = [round(float(x), 3) for x in gain]
                return np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8), diag
            if m == "histmatch":
                from skimage.exposure import match_histograms

                out = match_histograms(rgb, self._mosaic, channel_axis=-1)
                return np.clip(out, 0, 255).astype(np.uint8), diag
            if m == "vahadane":
                import staintools

                out = self._lib.transform(staintools.LuminosityStandardizer.standardize(rgb))
                return np.clip(out, 0, 255).astype(np.uint8), diag
            if m == "macenko":
                out = self._lib.normalize(I=rgb, stains=False)[0]  # the library returns (Inorm, H, E)
                return np.clip(out, 0, 255).astype(np.uint8), diag
            if m == "reinhard":
                return np.clip(self._lib.normalize(rgb), 0, 255).astype(np.uint8), diag
        except Exception as e:  # e.g. too little tissue for the SVD
            diag["fallback"] = f"{type(e).__name__}: {e}"
            return rgb, diag
        raise AssertionError(m)


class StainNormalizer:
    """Fit once on the TRAIN fields of a fold, then transform any RGB image. `method` may be a chain 'a+b'."""

    def __init__(self, method: str):
        parts = method.split("+")
        bad = [p for p in parts if p not in BASE_METHODS]
        if bad:
            raise ValueError(f"método desconhecido: {bad}; opções: {METHODS}")
        self.method = method
        self.steps = [_Step(p) for p in parts]
        self.n_train = 0

    @property
    def ref(self) -> dict:
        return {"n_train_images": self.n_train, **{s.method: s.ref for s in self.steps if s.ref}}

    def fit(self, train_paths: list[Path], saved_ref: dict | None = None) -> "StainNormalizer":
        self.n_train = len(train_paths)
        if all(s.method in ("none", "grayscale") for s in self.steps):
            return self
        mosaic = reference_mosaic(train_paths)
        for k, s in enumerate(self.steps):
            s.fit_mosaic(mosaic, (saved_ref or {}).get(s.method))
            if k < len(self.steps) - 1:
                mosaic = s.transform(mosaic)[0]  # later steps are fitted on already-normalized pixels
        return self

    def transform(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        diag: dict = {}
        for s in self.steps:
            rgb, d = s.transform(rgb)
            if d:
                diag[s.method] = d
        return rgb, diag

    def save_reference(self, path: Path, train_paths: list[Path]) -> None:
        import importlib.metadata as md

        libs = {k: md.version(k) for k in ("torchstain", "scikit-image", "staintools", "spams-bin", "opencv-python", "numpy")}
        path.write_text(json.dumps({"method": self.method, "reference": self.ref, "libraries": libs,
                                    "train_images": [p.name[:30] for p in train_paths]}, indent=1), encoding="utf-8")


def _valid_output(target: Path, src: Path) -> bool:
    """A previous run's output is reusable only if it decodes and has the size of the source field."""
    if not target.exists() or target.stat().st_size == 0:
        return False
    out, ref = cv2.imread(str(target)), cv2.imread(str(src))
    return out is not None and ref is not None and out.shape == ref.shape


def normalize_dir(src: Path, dst: Path, norm: StainNormalizer, resume: bool = False) -> dict:
    """Normalize every field of src/<split>/images into dst (PNG, same stem); labels are copied unchanged."""
    report = {"method": norm.method, "images": 0, "fallbacks": [], "diagnostics": {}}
    for split in ("train", "valid", "test"):
        img_dir = src / split / "images"
        if not img_dir.exists():
            continue
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in sorted(img_dir.iterdir()):
            target = dst / split / "images" / (img.stem + ".png")
            if resume and _valid_output(target, img):  # already normalized (with the SAME restored reference)
                report["images"] += 1
                report["resumed_images"] = report.get("resumed_images", 0) + 1
                continue
            out, diag = norm.transform(_read_rgb(img))
            tmp = target.with_name(target.stem + ".tmp.png")
            cv2.imwrite(str(tmp), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            import os

            os.replace(tmp, target)  # atomic: a kill mid-write never leaves a truncated image under the final name
            report["images"] += 1
            if diag:
                report["diagnostics"][img.stem[:24]] = diag
            if any("fallback" in v for v in diag.values()):
                report["fallbacks"].append(img.stem[:24])
        for lbl in (src / split / "labels").glob("*.txt"):
            shutil.copy2(lbl, dst / split / "labels" / lbl.name)
    if (src / "data.yaml").exists():
        shutil.copy2(src / "data.yaml", dst / "data.yaml")
    return report


# ------------------------------------------------------------------ analysis CLI
def _stats(rgb: np.ndarray) -> dict:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    od = -np.log((rgb.astype(np.float32) + 1) / 256.0)
    tissue = (od > 0.15).all(-1)
    t = lab[tissue] if tissue.sum() > 500 else lab.reshape(-1, 3)
    bg = _background_color(rgb)
    return {"L": float(t[:, 0].mean()), "a": float(t[:, 1].mean() - 128), "b": float(t[:, 2].mean() - 128),
            "bgR": float(bg[0]), "bgG": float(bg[1]), "bgB": float(bg[2])}


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data"))
    from split_common import mare_of  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--native", type=Path, required=True, help="vista nativa do fold (<fold>/native)")
    ap.add_argument("--report", action="store_true", help="variação de cor entre éguas, por método")
    ap.add_argument("--preview", type=Path, default=None, help="grade: mesmo recorte sob cada método")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--per-mare", type=int, default=3, help="campos por égua no --report (Vahadane é lento)")
    args = ap.parse_args()
    methods = [m for m in args.methods.split(",") if m]
    train = sorted((args.native / "train" / "images").iterdir())
    norms = {m: StainNormalizer(m).fit(train) for m in methods}
    print("referências ajustadas só com as éguas de treino:", {m: n.ref.get("n_train_images", "-") for m, n in norms.items()})
    all_imgs = [p for sp in ("train", "valid", "test") for p in sorted((args.native / sp / "images").iterdir())]

    if args.report:
        import collections

        groups: dict[str, list] = {}
        for p in sorted(all_imgs, key=lambda q: q.name):
            groups.setdefault(mare_of(p.name), []).append(p)
        # deterministic, evenly spaced fields from every mare
        sample = [g[i] for g in groups.values() for i in np.linspace(0, len(g) - 1, args.per_mare).astype(int)]

        print("\nVariação de cor ENTRE éguas (desvio-padrão das médias por égua; menor = mais homogêneo)")
        print(f"{'método':14s}{'L':>8s}{'a*':>8s}{'b*':>8s}{'fundoR':>9s}{'fundoG':>9s}{'fundoB':>9s}")
        for m in methods:
            per = collections.defaultdict(list)
            for p in sample:
                out, _ = norms[m].transform(_read_rgb(p, 4))  # global statistics: a quarter-scale copy is enough
                per[mare_of(p.name)].append(_stats(out))
            means = {k: {s: np.mean([x[s] for x in v]) for s in v[0]} for k, v in per.items()}
            sd = {s: float(np.std([mm[s] for mm in means.values()])) for s in ("L", "a", "b", "bgR", "bgG", "bgB")}
            print(f"{m:14s}{sd['L']:8.2f}{sd['a']:8.2f}{sd['b']:8.2f}{sd['bgR']:9.2f}{sd['bgG']:9.2f}{sd['bgB']:9.2f}")

    if args.preview:
        by_mare: dict[str, Path] = {}
        for p in all_imgs:
            by_mare.setdefault(mare_of(p.name), p)
        rows = []
        for mare, p in list(by_mare.items())[:4]:
            img = _read_rgb(p, 2)
            h, w = img.shape[:2]
            crop_in = img[h // 2 - 225: h // 2 + 225, w // 2 - 300: w // 2 + 300]
            cells = []
            for m in methods:
                out, _ = norms[m].transform(crop_in if m == "none" else img)
                crop = out if m == "none" else out[h // 2 - 225: h // 2 + 225, w // 2 - 300: w // 2 + 300]
                cell = cv2.cvtColor(cv2.resize(crop, (330, 248), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
                for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                    cv2.putText(cell, f"{mare} | {m}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th)
                cells.append(cell)
            rows.append(np.hstack(cells))
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), np.vstack(rows))
        print("preview salvo em", args.preview)


if __name__ == "__main__":
    main()
