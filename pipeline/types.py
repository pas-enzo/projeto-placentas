from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class InstanceResult:
    count: int
    area_px: float
    area_um2: float
    masks: list[np.ndarray] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)


@dataclass
class ImageResult:
    image_path: Path
    width: int
    height: int
    micro: InstanceResult | None = None
    capilar: InstanceResult | None = None
    overlay_path: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "image": self.image_path.name,
            "path": str(self.image_path),
            "width": self.width,
            "height": self.height,
        }
        if self.micro is not None:
            row.update(
                {
                    "micro_count": self.micro.count,
                    "micro_area_px": round(self.micro.area_px, 2),
                    "micro_area_um2": round(self.micro.area_um2, 4),
                }
            )
        if self.capilar is not None:
            row.update(
                {
                    "capilar_count": self.capilar.count,
                    "capilar_area_px": round(self.capilar.area_px, 2),
                    "capilar_area_um2": round(self.capilar.area_um2, 4),
                }
            )
        if self.overlay_path is not None:
            row["overlay"] = str(self.overlay_path)
        return row
