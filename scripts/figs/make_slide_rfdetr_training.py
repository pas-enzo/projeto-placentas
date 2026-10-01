"""Slide asset: RF-DETR Seg Large (672 px) training curves with early-stopping markers."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "v2_rfdetr_seg_large_opt_v1/artifacts/runs/seg_large_r672_auto/metrics.csv"
OUT = ROOT / "figs/slides/rfdetr_curva_treino.jpg"
PATIENCE = 20

comma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))


def main() -> None:
    df = pd.read_csv(METRICS)
    df["epoch"] = df["epoch"] + 1
    tr = df.dropna(subset=["train/loss"]).groupby("epoch", as_index=False)["train/loss"].last()
    va = df.dropna(subset=["val/segm_mAP_50_95"]).groupby("epoch", as_index=False).last()
    va["best_of"] = va[["val/segm_mAP_50_95", "val/ema_segm_mAP_50_95"]].max(axis=1)
    best = va.loc[va["best_of"].idxmax()]
    best_ep, last_ep = int(best["epoch"]), int(va["epoch"].max())
    print(f"best epoch {best_ep} (score {best['best_of']:.4f}), last epoch {last_ep}")

    plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans"})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True, dpi=200)

    ax1.plot(tr["epoch"], tr["train/loss"], color="#444444", lw=2.2, label="Treino")
    ax1.plot(va["epoch"], va["val/loss"], color="#c0392b", lw=2.2, label="Validação")
    ax1.set_ylabel("Perda total")
    ax1.legend(frameon=False, loc="upper right")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))

    ax2.plot(va["epoch"], va["val/segm_mAP_50_95"], color="#7f8c8d", lw=2.0, label="Regular")
    ax2.plot(va["epoch"], va["val/ema_segm_mAP_50_95"], color="#00aa00", lw=2.6, label="EMA")
    ax2.set_ylabel("mAP@50:95 (máscara)")
    ax2.set_xlabel("Época")
    ax2.yaxis.set_major_formatter(comma)
    ax2.legend(frameon=False, loc="lower right")

    for ax in (ax1, ax2):
        ax.axvline(best_ep, color="#00aa00", ls="--", lw=1.6)
        ax.axvspan(best_ep, last_ep, color="#00aa00", alpha=0.06)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    ax2.scatter([best_ep], [best["best_of"]], s=90, color="#00aa00", zorder=5)
    ax2.set_ylim(top=best["best_of"] + 0.006)
    ax2.text(
        best_ep + 0.6,
        best["best_of"],
        f"Melhor época: {best_ep}",
        ha="left",
        va="center",
        fontsize=14,
        color="#006600",
        fontweight="bold",
    )
    ax2.text(
        last_ep - 0.4,
        best["best_of"] + 0.005,
        f"paciência = {PATIENCE} épocas\n(parada na época {last_ep})",
        ha="right",
        va="top",
        fontsize=13,
        color="#006600",
    )
    ax2.set_xlim(1, last_ep)

    fig.tight_layout()
    fig.savefig(OUT, pil_kwargs={"quality": 93})
    print("saved", OUT)


if __name__ == "__main__":
    main()
