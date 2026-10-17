from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class CropMasks(Sequence):
    """Binary masks stored as bbox crops; indexing returns the full-size uint8 mask.

    Native capillary masks are ~13 MB each as full arrays; crops are a few KB.
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = (int(shape[0]), int(shape[1]))
        self._items: list[tuple[int, int, np.ndarray]] = []

    def append(self, mask: np.ndarray) -> int:
        """Store one mask; returns its area in px."""
        rows = np.flatnonzero(mask.any(axis=1))
        if rows.size == 0:
            self._items.append((0, 0, np.zeros((0, 0), dtype=np.uint8)))
            return 0
        cols = np.flatnonzero(mask.any(axis=0))
        y0, x0 = int(rows[0]), int(cols[0])
        crop = (mask[y0 : int(rows[-1]) + 1, x0 : int(cols[-1]) + 1] > 0).astype(np.uint8)
        self._items.append((y0, x0, crop))
        return int(crop.sum())

    def crop(self, i: int) -> tuple[int, int, np.ndarray]:
        """(y0, x0, crop) of instance i."""
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        y0, x0, crop = self._items[i]
        out = np.zeros(self.shape, dtype=np.uint8)
        out[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
        return out


@dataclass
class InstanceResult:
    count: int
    area_px: float
    area_um2: float
    masks: Sequence[np.ndarray] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    model_name: str = ""
    areas_um2: list[float] = field(default_factory=list)
    # Area of the mask union (pixels covered by 2+ instances counted once); area_px is the
    # per-instance sum used in the paper.
    area_union_px: float = 0.0
    area_union_um2: float = 0.0
    # Per-instance geometry / polygons in native-FOV px (see geometry.compute_geometry).
    geometry: list[dict] = field(default_factory=list)
    polygons: list[list[list[float]]] = field(default_factory=list)


@dataclass
class ImageResult:
    image_path: Path
    width: int
    height: int
    micro: InstanceResult | None = None
    capilar: InstanceResult | None = None
    overlay_path: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def drop_masks(self) -> None:
        """Release mask arrays (native capillary masks are ~13 MB each)."""
        for head in (self.micro, self.capilar):
            if head is not None:
                head.masks = []

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
                    "micro_model": self.micro.model_name,
                    "micro_count": self.micro.count,
                    "micro_area_px": round(self.micro.area_px, 2),
                    "micro_area_um2": round(self.micro.area_um2, 4),
                    "micro_area_union_um2": round(self.micro.area_union_um2, 4),
                }
            )
        if self.capilar is not None:
            row.update(
                {
                    "capilar_count": self.capilar.count,
                    "capilar_area_px": round(self.capilar.area_px, 2),
                    "capilar_area_um2": round(self.capilar.area_um2, 4),
                    "capilar_area_union_um2": round(self.capilar.area_union_um2, 4),
                }
            )
        if self.extras.get("warning"):
            row["warning"] = self.extras["warning"]
        if self.overlay_path is not None:
            row["overlay"] = str(self.overlay_path)
        return row
