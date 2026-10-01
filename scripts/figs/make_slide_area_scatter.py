"""Slide asset: per-field area agreement (PT-BR, large fonts, app colors)."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MICRO_TOTALS = ROOT / "v2_rfdetr_seg_large_opt_v1/artifacts/reports/seg_large_r672_auto/placenta_totals_report_rfdetr.csv"
CAP_TOTALS = ROOT / "v3_capilar_yolo11s/artifacts_v4_field/reports/capilar_field_totals_report.csv"
OUT = ROOT / "figs/slides/concordancia_area_campo.jpg"


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    gt = np.array([float(r["GT_Area_um2"]) for r in rows])
    ai = np.array([float(r["AI_Area_um2"]) for r in rows])
    return gt, ai


def main() -> None:
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 15})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), dpi=200)
    thousands = FuncFormatter(lambda v, _: f"{v / 1000:.0f}")
    panels = [
        (axes[0], *load(MICRO_TOTALS), "#00aa00", "A  Microcotilédones"),
        (axes[1], *load(CAP_TOTALS), "#00a0be", "B  Capilares (SAHI)"),
    ]
    for ax, gt, ai, color, title in panels:
        r = np.corrcoef(gt, ai)[0, 1]
        err = np.abs(ai - gt) / gt
        print(f"{title}: n={len(gt)} r={r:.3f} mean={100*err.mean():.1f}% median={100*np.median(err):.1f}%")
        lo, hi = min(gt.min(), ai.min()), max(gt.max(), ai.max())
        pad = 0.06 * (hi - lo)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color="#888888", lw=1.6, ls="--", zorder=1)
        ax.scatter(gt, ai, s=85, color=color, alpha=0.9, edgecolors="#333333", linewidths=0.8, zorder=2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_formatter(thousands)
        ax.yaxis.set_major_formatter(thousands)
        ax.set_xlabel("Área anotada (×10³ µm²)")
        ax.set_ylabel("Área prevista (×10³ µm²)")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=17)
        ax.text(
            0.04, 0.96,
            f"r = {r:.2f}\nerro médio = {100 * err.mean():.1f}%".replace(".", ","),
            transform=ax.transAxes, va="top", ha="left", fontsize=15,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"),
        )
        ax.text(0.96, 0.04, "n = 27 campos", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=13, color="#555555")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=3)
    fig.savefig(OUT, pil_kwargs={"quality": 93})
    print("saved", OUT)


if __name__ == "__main__":
    main()
