"""Generate optional Results plots (scatter, Bland-Altman, conf sweeps).

Outputs under figs/results/ — paper-ready white background, Segoe UI when available.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(r"D:\projeto_placentas_clayton\dev\projeto-placentas")
OUT = ROOT / "figs" / "results"
OUT.mkdir(parents=True, exist_ok=True)

MICRO_TOTALS = (
    ROOT
    / "v2_rfdetr_seg_large_opt_v1"
    / "artifacts"
    / "reports"
    / "seg_large_r672_auto"
    / "placenta_totals_report_rfdetr.csv"
)
CAP_FIELD_TOTALS = (
    ROOT
    / "v3_capilar_yolo11s"
    / "artifacts_v4_field"
    / "reports"
    / "capilar_field_totals_report.csv"
)
MICRO_SWEEP = (
    ROOT
    / "v2_rfdetr_seg_large_opt_v1"
    / "artifacts"
    / "benchmarks"
    / "validation_conf_sweep_seg_large_r672_auto.csv"
)
CAP_TILE_SWEEP = (
    ROOT / "v3_capilar_yolo11s" / "artifacts_v4" / "benchmarks" / "validation_conf_sweep.csv"
)

# Operating points (for vertical markers on sweeps)
MICRO_CONF = 0.44
CAP_CONF = 0.33

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#282828",
        "axes.labelcolor": "#141414",
        "xtick.color": "#141414",
        "ytick.color": "#141414",
        "text.color": "#141414",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.8,
    }
)
# Prefer Segoe UI on Windows
try:
    plt.rcParams["font.family"] = "Segoe UI"
except Exception:
    pass


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fcol(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def style_ax(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("#282828")
        spine.set_linewidth(1.2)


def save(fig, name: str) -> Path:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)
    return path


def plot_area_scatter() -> None:
    micro = read_csv(MICRO_TOTALS)
    cap = read_csv(CAP_FIELD_TOTALS)
    gt_m, ai_m = fcol(micro, "GT_Area_um2"), fcol(micro, "AI_Area_um2")
    gt_c, ai_c = fcol(cap, "GT_Area_um2"), fcol(cap, "AI_Area_um2")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    panels = [
        (axes[0], gt_m, ai_m, "A  Microcotyledon (RF-DETR, per FOV)", MICRO_CONF),
        (axes[1], gt_c, ai_c, "B  Capillary (YOLO11s + SAHI, per FOV)", CAP_CONF),
    ]
    for ax, gt, ai, title, conf_star in panels:
        lo = float(min(gt.min(), ai.min()))
        hi = float(max(gt.max(), ai.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color="#888888", lw=1.2, ls="--", label="Identity", zorder=1)
        ax.scatter(gt, ai, s=36, c="#1F4E79", alpha=0.85, edgecolors="white", linewidths=0.6, zorder=2)
        r = np.corrcoef(gt, ai)[0, 1]
        rel = np.abs(ai.sum() - gt.sum()) / gt.sum()
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"GT area ($\mathrm{\mu m}^{2}$)")
        ax.set_ylabel(r"Predicted area ($\mathrm{\mu m}^{2}$)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(
            0.04,
            0.96,
            f"n={len(gt)}, conf*={conf_star}\nr = {r:.3f}\nAreaRelErr (agg.) = {100*rel:.2f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCCCCC"),
        )
        ax.legend(loc="lower right", frameon=True)
        style_ax(ax)

    save(fig, "figR_area_scatter_fov.jpg")


def bland_altman(ax, gt: np.ndarray, ai: np.ndarray, title: str) -> None:
    mean = (gt + ai) / 2.0
    diff = ai - gt
    md = float(diff.mean())
    sd = float(diff.std(ddof=1))
    ax.scatter(mean, diff, s=36, c="#1F4E79", alpha=0.85, edgecolors="white", linewidths=0.6)
    ax.axhline(md, color="#C0392B", lw=1.4, label=f"Mean diff = {md:.1f}")
    ax.axhline(md + 1.96 * sd, color="#888888", lw=1.1, ls="--", label=f"+1.96 SD = {md+1.96*sd:.1f}")
    ax.axhline(md - 1.96 * sd, color="#888888", lw=1.1, ls="--", label=f"−1.96 SD = {md-1.96*sd:.1f}")
    ax.set_xlabel(r"Mean of GT and AI ($\mathrm{\mu m}^{2}$)")
    ax.set_ylabel(r"AI − GT ($\mathrm{\mu m}^{2}$)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(loc="best", fontsize=8, frameon=True)
    style_ax(ax)


def plot_bland_altman() -> None:
    micro = read_csv(MICRO_TOTALS)
    cap = read_csv(CAP_FIELD_TOTALS)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    bland_altman(
        axes[0],
        fcol(micro, "GT_Area_um2"),
        fcol(micro, "AI_Area_um2"),
        "A  Microcotyledon Bland–Altman (per FOV)",
    )
    bland_altman(
        axes[1],
        fcol(cap, "GT_Area_um2"),
        fcol(cap, "AI_Area_um2"),
        "B  Capillary Bland–Altman (per FOV, SAHI)",
    )
    save(fig, "figR_bland_altman_area_fov.jpg")


def plot_conf_sweeps() -> None:
    micro = read_csv(MICRO_SWEEP)
    cap = read_csv(CAP_TILE_SWEEP)
    # sort by conf
    micro = sorted(micro, key=lambda r: float(r["conf"]))
    cap = sorted(cap, key=lambda r: float(r["conf"]))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=False)

    for ax, rows, conf_star, title, unit in [
        (axes[0], micro, MICRO_CONF, "A  Microcotyledon confidence sweep", "FOV"),
        (axes[1], cap, CAP_CONF, "B  Capillary confidence sweep (tile)", "tile"),
    ]:
        conf = fcol(rows, "conf")
        f1 = fcol(rows, "f1")
        iou = fcol(rows, "mean_iou")
        are = fcol(rows, "area_rel_error")
        ax.plot(conf, f1, "o-", color="#1F4E79", lw=1.6, ms=4, label="F1")
        ax.plot(conf, iou, "s-", color="#2E7D32", lw=1.6, ms=4, label="mean IoU")
        ax.plot(conf, are, "^-", color="#C0392B", lw=1.6, ms=4, label="AreaRelErr")
        ax.axvline(conf_star, color="#666666", ls="--", lw=1.2, label=f"conf* = {conf_star}")
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("Score")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlim(conf.min() - 0.02, conf.max() + 0.02)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", fontsize=8)
        ax.text(
            0.98,
            0.04,
            f"eval unit: {unit}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        style_ax(ax)

    save(fig, "figR_confidence_sweeps.jpg")

    # Note file: capillary FOV SAHI sweep only has 1 row currently
    note = OUT / "NOTES_plots.txt"
    note.write_text(
        "figR_confidence_sweeps.jpg panel B uses TILE-level sweep "
        "(artifacts_v4/validation_conf_sweep.csv), because the FOV/SAHI sweep "
        "currently has a single operating point (conf=0.33 only).\n"
        "Scatter and Bland–Altman for capillaries use FOV/SAHI totals "
        "(artifacts_v4_field/capilar_field_totals_report.csv).\n",
        encoding="utf-8",
    )
    print("saved", note)


def main() -> None:
    plot_area_scatter()
    plot_bland_altman()
    plot_conf_sweeps()
    # update manifest
    manifest_path = OUT / "figure_manifest.json"
    import json

    man = {}
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
    man["optional_plots"] = {
        "area_scatter_fov": "figs/results/figR_area_scatter_fov.jpg",
        "bland_altman_area_fov": "figs/results/figR_bland_altman_area_fov.jpg",
        "confidence_sweeps": "figs/results/figR_confidence_sweeps.jpg",
        "notes": "figs/results/NOTES_plots.txt",
    }
    manifest_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print("updated", manifest_path)


if __name__ == "__main__":
    main()
