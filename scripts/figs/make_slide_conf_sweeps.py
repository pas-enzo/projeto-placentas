"""Slide asset: confidence sweeps (PT-BR, large fonts, shared legend)."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MICRO_SWEEP = (
    ROOT / "v2_rfdetr_seg_large_opt_v1/artifacts/benchmarks/validation_conf_sweep_seg_large_r672_auto.csv"
)
CAP_TILE_SWEEP = ROOT / "v3_capilar_yolo11s/artifacts_v4/benchmarks/validation_conf_sweep.csv"
OUT = ROOT / "figs/slides/limiar_varredura.jpg"
MICRO_CONF, CAP_CONF = 0.44, 0.33

C_F1, C_IOU, C_ARE, C_S = "#1F4E79", "#2E7D32", "#C0392B", "#6A1B9A"


def comma(fmt: str):
    return FuncFormatter(lambda v, _: format(v, fmt).replace(".", ","))


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["conf"]))
    d = {k: np.array([float(r[k]) for r in rows]) for k in ("conf", "f1", "mean_iou", "area_rel_error")}
    d["S"] = 0.6 * d["f1"] + 0.3 * d["mean_iou"] + 0.1 * (1 - d["area_rel_error"])
    return d


def main() -> None:
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 15})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=200)
    panels = [
        (axes[0], load(MICRO_SWEEP), MICRO_CONF, "A  Microcotilédones (campo)"),
        (axes[1], load(CAP_TILE_SWEEP), CAP_CONF, "B  Capilares (tile)"),
    ]
    for ax, d, c_star, title in panels:
        c = d["conf"]
        ax.plot(c, d["f1"], "o-", color=C_F1, lw=2.2, ms=5)
        ax.plot(c, d["mean_iou"], "s-", color=C_IOU, lw=2.2, ms=5)
        ax.plot(c, d["area_rel_error"], "^-", color=C_ARE, lw=2.2, ms=5)
        ax.plot(c, d["S"], "D-", color=C_S, lw=2.6, ms=5)
        ax.axvline(c_star, color="#555555", ls="--", lw=1.6)
        i = int(np.argmin(np.abs(c - c_star)))
        ax.scatter([c_star], [d["S"][i]], s=110, color=C_S, zorder=5, edgecolor="white", lw=1.5)
        ax.text(c_star + 0.012, 0.50, f"c* = {c_star:.2f}".replace(".", ","), fontsize=15,
                fontweight="bold", color="#333333")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=17)
        ax.set_xlabel("Limiar de confiança")
        ax.set_xlim(c.min() - 0.02, c.max() + 0.02)
        ax.set_ylim(0, 1.03)
        ax.xaxis.set_major_formatter(comma(".1f"))
        ax.yaxis.set_major_formatter(comma(".1f"))
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Valor")

    handles = [
        Line2D([], [], color=C_F1, marker="o", lw=2.2, label="F1"),
        Line2D([], [], color=C_IOU, marker="s", lw=2.2, label="IoU médio"),
        Line2D([], [], color=C_ARE, marker="^", lw=2.2, label="Erro relativo de área"),
        Line2D([], [], color=C_S, marker="D", lw=2.6, label="S(c)"),
        Line2D([], [], color="#555555", ls="--", lw=1.6, label="Limiar escolhido"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=14,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT, pil_kwargs={"quality": 93})
    print("saved", OUT)


if __name__ == "__main__":
    main()
