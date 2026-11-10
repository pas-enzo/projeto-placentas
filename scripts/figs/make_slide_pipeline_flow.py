"""Slide figure: flowchart of the inference pipeline (pipeline/ package), in the slide theme.

Steps are taken from the code: pipeline/run.py (analyze_bgr), micro.py (stretch 640 + YOLO-seg), capilar.py (16 SAHI windows
1380x1032, overlap 0.2, GREEDYNMM merge), geometry.py (per-instance metrics), artifacts.py (outputs).

    python scripts/figs/make_slide_pipeline_flow.py --out <png>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BG, BLUE, HEAD, WHITE = "#FFFCF3", "#596AB6", "#C7D8E1", "#FFFFFF"
GREEN, CYAN = "#7BD96B", "#5FD3E8"  # the pipeline overlay colors (micro lime, capillary cyan)


def box(ax, x, y, w, h, text, fc=WHITE, fs=13.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=BLUE, lw=2.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=BLUE,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=BLUE,
                                 shrinkA=2, shrinkB=2))


def tag(ax, x, y, text, color):
    ax.text(x, y, text, ha="left", va="center", fontsize=11.5, color=BLUE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none"))


def note(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="top", fontsize=10.5, color=BLUE, style="italic", linespacing=1.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(15.2, 6.0), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    H = 1.45
    ym, yc = 3.9, 1.6  # lane bottoms (micro, capillary)
    # input
    box(ax, 0.1, 2.45, 2.4, 1.9, "Imagem do campo\n4140 × 3096 px\n(uma ou várias)", fc=HEAD, bold=True, fs=12)
    # micro lane
    tag(ax, 3.0, 5.65, "MICROCOTILÉDONES", GREEN)
    box(ax, 3.0, ym, 2.4, H, "Redimensiona\npara 640 × 640")
    box(ax, 5.9, ym, 2.6, H, "YOLO-seg\nmicrocotilédones")
    note(ax, 7.2, ym - 0.1, "confiança calibrada\nna validação")
    # capillary lane
    tag(ax, 3.0, 3.35, "CAPILARES", CYAN)
    box(ax, 3.0, yc, 2.4, H, "Divide o campo\nem 16 janelas")
    note(ax, 4.2, yc - 0.1, "1380 × 1032 px,\nsobreposição 0,2")
    box(ax, 5.9, yc, 2.6, H, "YOLO-seg capilares\n+ fusão")
    note(ax, 7.2, yc - 0.1, "fusão GreedyNMM\nentre janelas")
    # measures and outputs
    box(ax, 9.2, 2.45, 2.6, 1.9, "Medidas por\ninstância", fc=HEAD, bold=True)
    note(ax, 10.5, 2.35, "área (µm²), perímetro, eixos,\ncircularidade; calibração:\n50 µm = 469 px na imagem original")
    box(ax, 12.5, 2.45, 2.6, 1.9, "Resultados\npor campo e\npor instância", fc=HEAD, bold=True)
    note(ax, 13.8, 2.35, "contagem, área, tabela (CSV),\nanotações COCO,\nsobreposição e máscaras")

    arrow(ax, 2.5, 3.8, 3.0, ym + H / 2)
    arrow(ax, 2.5, 3.0, 3.0, yc + H / 2)
    arrow(ax, 5.4, ym + H / 2, 5.9, ym + H / 2)
    arrow(ax, 5.4, yc + H / 2, 5.9, yc + H / 2)
    arrow(ax, 8.5, ym + H / 2, 9.2, 3.9)
    arrow(ax, 8.5, yc + H / 2, 9.2, 2.9)
    arrow(ax, 11.8, 3.4, 12.5, 3.4)
    ax.text(7.6, 0.2, "Interface Gradio (imagem única ou lote)  ·  linha de comando  ·  ZIP com todos os arquivos",
            ha="center", va="center", fontsize=12, color=BLUE, fontweight="bold")
    fig.savefig(a.out, facecolor=BG, bbox_inches="tight", pad_inches=0.15)


if __name__ == "__main__":
    main()
