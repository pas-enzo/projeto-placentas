from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def discover_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    while p != p.parent:
        if (p / ".git").exists() and (p / "v3_capilar_yolo11s").exists():
            return p
        p = p.parent
    raise RuntimeError("Repo root not found (.git + v3_capilar_yolo11s)")


@dataclass
class PipelineConfig:
    """Champion operating points and paths used in the paper."""

    repo_root: Path = field(default_factory=discover_repo_root)

    # Microcotyledon (RF-DETR Seg Large)
    micro_ckpt: str = (
        "v2_rfdetr_seg_large_opt_v1/artifacts/runs/seg_large_r672_auto/checkpoint_best_total.pth"
    )
    micro_conf: float = 0.44
    micro_resolution: int = 672
    # Stretch-space area factor (Roboflow 640×640 anisotropic export)
    micro_area_um2_per_px2: float = (50 / 72) ** 2 * (3096 / 4140)  # ≈ 0.3606

    # Capillary (YOLO11s-seg + SAHI on native FOV)
    capilar_ckpt: str = "v3_capilar_yolo11s/runs/capilar_yolo11s_tiled_v4/weights/best.pt"
    capilar_conf: float = 0.33
    capilar_imgsz: int = 640
    capilar_slice_w: int = 1380
    capilar_slice_h: int = 1032
    capilar_overlap: float = 0.2
    capilar_cls_id: int = 0
    # Native isotropic tile/FOV factor
    capilar_area_um2_per_px2: float = (50 / 72) ** 2 * (640 / 4140) ** 2  # ≈ 0.0115

    # Preprocess for micro: non-proportional stretch like Roboflow export
    micro_canvas: int = 640

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_file():
            return p.resolve()
        return (self.repo_root / rel).resolve()

    @property
    def micro_ckpt_path(self) -> Path:
        return self.resolve(self.micro_ckpt)

    @property
    def capilar_ckpt_path(self) -> Path:
        return self.resolve(self.capilar_ckpt)
