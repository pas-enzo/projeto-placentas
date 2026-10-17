"""Written metrics: per-image and batch summary tables, shared by the CLI and the Gradio app."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import PipelineConfig
from .types import ImageResult, InstanceResult

HEADS = (("micro", "Microcotilédones"), ("capilar", "Capilares"))
_NA = "—"


def _present(results: Sequence[ImageResult]) -> list[tuple[str, str]]:
    return [(k, label) for k, label in HEADS if any(getattr(r, k) is not None for r in results)]


def _fov_um2(res: ImageResult, cfg: PipelineConfig) -> float:
    # Native isotropic pixel factor; only meaningful for native-size FOVs (see fov_size_warning).
    return res.width * res.height * float(cfg.capilar_area_um2_per_px2)


def _mean_sd(v: Sequence[float], digits: int = 1) -> str:
    if len(v) == 0:
        return _NA
    sd = f" ± {np.std(v, ddof=1):,.{digits}f}" if len(v) > 1 else ""
    return f"{np.mean(v):,.{digits}f}{sd}"


def _inst_stats(areas: Sequence[float], confs: Sequence[float]) -> list[str]:
    """Rows: mean ± sd, median, min – max, mean confidence (min)."""
    if len(areas) == 0:
        return [_NA] * 4
    a = np.asarray(areas, dtype=float)
    c = np.asarray(confs, dtype=float)
    return [
        _mean_sd(a),
        f"{np.median(a):,.1f}",
        f"{a.min():,.1f} – {a.max():,.1f}",
        f"{c.mean():.2f} (mín {c.min():.2f})" if c.size else _NA,
    ]


_INST_LABELS = [
    "Área por instância, média ± dp (µm²)",
    "Área por instância, mediana (µm²)",
    "Área por instância, mín – máx (µm²)",
    "Confiança, média (mín)",
]


# Same rows, same order and same labels in the single-image and batch tables.
_ROW_LABELS = [
    "Modelo",
    "Contagem",
    "Área total, soma das instâncias (µm²)",
    "Área total, união das máscaras (µm²)",
    "Sobreposição entre instâncias (%)",
    "Ocupação do FOV (união, %)",
    *_INST_LABELS,
]


def _overlap_pct(h: InstanceResult) -> float | None:
    return (1 - h.area_union_px / h.area_px) * 100 if h.area_px else None


def _table(cols: list[list[str]], heads: list[tuple[str, str]], extra_bottom: list[list[str]] = ()):
    header = ["Métrica", *[label for _, label in heads]]
    rows = [[lab, *[c[i] for c in cols]] for i, lab in enumerate(_ROW_LABELS)]
    return header, rows + [list(r) for r in extra_bottom]


def image_table(res: ImageResult, cfg: PipelineConfig) -> tuple[list[str], list[list[str]]]:
    """Metric × head table for one image."""
    heads = _present([res])
    fov = _fov_um2(res, cfg)
    cols: list[list[str]] = []
    for key, _ in heads:
        h: InstanceResult = getattr(res, key)
        ov = _overlap_pct(h)
        cols.append(
            [
                h.model_name,
                f"{h.count}",
                f"{h.area_um2:,.2f}",
                f"{h.area_union_um2:,.2f}",
                f"{ov:.1f}" if ov is not None else _NA,
                f"{h.area_union_um2 / fov * 100:.1f}",
                *_inst_stats(h.areas_um2, h.confidences),
            ]
        )
    return _table(cols, heads)


def batch_table(
    results: Sequence[ImageResult], cfg: PipelineConfig
) -> tuple[list[str], list[list[str]]]:
    """Same table aggregated over a batch.

    Count, areas, overlap and occupancy are mean ± sd across images; the per-instance
    rows pool every instance of the batch. Extra top rows (at the end): images and total count.
    """
    heads = _present(results)
    cols: list[list[str]] = []
    n_images: list[str] = []
    totals: list[str] = []
    for key, _ in heads:
        rs = [(r, getattr(r, key)) for r in results if getattr(r, key) is not None]
        pooled_a = [a for _, h in rs for a in h.areas_um2]
        pooled_c = [c for _, h in rs for c in h.confidences]
        n_images.append(f"{len(rs)}")
        totals.append(f"{sum(h.count for _, h in rs)}")
        cols.append(
            [
                rs[0][1].model_name if rs else _NA,
                _mean_sd([h.count for _, h in rs]),
                _mean_sd([h.area_um2 for _, h in rs]),
                _mean_sd([h.area_union_um2 for _, h in rs]),
                _mean_sd([ov for _, h in rs if (ov := _overlap_pct(h)) is not None]),
                _mean_sd([h.area_union_um2 / _fov_um2(r, cfg) * 100 for r, h in rs]),
                *_inst_stats(pooled_a, pooled_c),
            ]
        )
    return _table(cols, heads, extra_bottom=[["Imagens", *n_images], ["Contagem total", *totals]])
