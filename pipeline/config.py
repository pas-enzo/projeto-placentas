from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Calibration: the 50 um scale bar burned into every native FOV measures 469 px
# (constant over all 180 images), so native pixels are isotropic at 50/469 um/px.
SCALE_BAR_UM = 50.0
SCALE_BAR_PX_NATIVE = 469.0
NATIVE_W, NATIVE_H = 4140, 3096
MICRO_CANVAS = 640
UM_PER_PX_NATIVE = SCALE_BAR_UM / SCALE_BAR_PX_NATIVE


def fov_size_warning(width: int, height: int) -> str | None:
    """Area factors are calibrated for native FOVs only; warn on any other size."""
    if (width, height) == (NATIVE_W, NATIVE_H):
        return None
    return (
        f"FOV {width}×{height} px diferente do nativo ({NATIVE_W}×{NATIVE_H} px): os fatores de área "
        "(µm²) são calibrados para o FOV nativo e podem estar incorretos."
    )


def _is_repo_root(p: Path) -> bool:
    return (p / ".git").exists() and (p / "v3_capilar_yolo11s").exists()


def discover_repo_root(start: Path | None = None) -> Path:
    # Prefer the package location so the repo is found regardless of the working directory.
    starts = [Path(start)] if start else [Path(__file__).resolve().parent, Path.cwd()]
    for s in starts:
        p = s.resolve()
        while True:
            if _is_repo_root(p):
                return p
            if p == p.parent:
                break
            p = p.parent
    raise RuntimeError("Repo root not found (.git + v3_capilar_yolo11s)")


@dataclass
class PipelineConfig:
    """Champion operating points and paths used in the paper."""

    repo_root: Path = field(default_factory=discover_repo_root)

    # Microcotyledon backend: "yolo" (TCC: YOLO family only) | "rfdetr" (paper champion)
    micro_backend: str = "yolo"

    # Microcotyledon (RF-DETR Seg Large)
    micro_ckpt: str = (
        "v2_rfdetr_seg_large_opt_v1/artifacts/runs/seg_large_r672_auto/checkpoint_best_total.pth"
    )
    micro_conf: float = 0.44
    micro_resolution: int = 672

    # Microcotyledon (YOLO11s-seg, same stretch-640 Roboflow export)
    micro_yolo_ckpt: str = (
        "archive/experiments/v2_yolo_early/runs/segment/placentas_v11_aug_v2/weights/best.pt"
    )
    micro_yolo_conf: float = 0.46
    micro_yolo_imgsz: int = 640

    # Stretch-space area factor (Roboflow 640×640 anisotropic export); both backends.
    # One stretched px covers (NATIVE_W/640) x (NATIVE_H/640) native px.
    micro_area_um2_per_px2: float = (
        UM_PER_PX_NATIVE**2 * (NATIVE_W / MICRO_CANVAS) * (NATIVE_H / MICRO_CANVAS)
    )  # ≈ 0.3557

    # Capillary (YOLO11s-seg + SAHI on native FOV)
    capilar_ckpt: str = "v3_capilar_yolo11s/runs/capilar_yolo11s_tiled_v4/weights/best.pt"
    capilar_conf: float = 0.33
    capilar_imgsz: int = 640
    capilar_slice_w: int = 1380
    capilar_slice_h: int = 1032
    capilar_overlap: float = 0.2
    capilar_postprocess: str = "GREEDYNMM"
    capilar_match_metric: str = "IOS"
    capilar_match_threshold: float = 0.5
    capilar_cls_id: int = 0
    # Native isotropic tile/FOV factor
    capilar_area_um2_per_px2: float = UM_PER_PX_NATIVE**2  # ≈ 0.011365

    # Preprocess for micro: non-proportional stretch like Roboflow export
    micro_canvas: int = MICRO_CANVAS

    # Scratch files for model APIs that only take a path (kept off C:, which is nearly full)
    tmp_dir: str = "outputs/tmp"

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_file():
            return p.resolve()
        return (self.repo_root / rel).resolve()

    @property
    def tmp_dir_path(self) -> Path:
        p = self.resolve(self.tmp_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def micro_ckpt_path(self) -> Path:
        return self.resolve(self.micro_ckpt)

    @property
    def micro_yolo_ckpt_path(self) -> Path:
        return self.resolve(self.micro_yolo_ckpt)

    @property
    def capilar_ckpt_path(self) -> Path:
        return self.resolve(self.capilar_ckpt)
