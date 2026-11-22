"""Figuras 2, 3 e 4 do artigo (secao de Metodos), geradas dos nossos dados.

  fig2_annotations_2x2.jpg   campo ROSILHA-M-D_006 (painel A da Fig. 1): original | mascaras de micro | de capilar | as duas (anotacao manual)
  fig3_cv_design.png         esquema da validacao cruzada por egua (6 folds x 6 eguas: treino / validacao / teste)
  fig4_method_overview.png   visao geral: campo nativo -> entrada de cada ramo -> predicao -> medidas (predicao do modelo sem normalizacao
                             do fold em que ROSILHA-M foi a egua de TESTE, isto e, uma predicao que nao viu essa egua)
Cores categoricas da paleta de referencia do guia dataviz (slots 1 a 3 validam todos os pares): azul #2a78d6, laranja #eb6834, verde-agua #1baf7a;
amarelo #eda100 (slot 4) para os capilares.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO = Path(r"D:\projeto_placentas_clayton\dev\projeto-placentas")
sys.path.insert(0, str(REPO))
from scripts.artigo_iser import dataset_vs_modelo as D, fluxo as F  # noqa: E402

FIG = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\artigo\figs")
FIG.mkdir(parents=True, exist_ok=True)
LAMINA = "ROSILHA-M-D_006"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2 = "#0b0b0b", "#52514e"


def bgr(hexcolor: str):
    h = hexcolor.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def paint(img, items, sx, sy, color, alpha, thick):
    """Mascaras guardadas como recortes (y0, x0, mask); preenchimento + contorno."""
    H, W = img.shape[:2]
    for it in items:
        x0, y0 = int(round(it.x0 * sx)), int(round(it.y0 * sy))
        w, h = max(int(round(it.mask.shape[1] * sx)), 1), max(int(round(it.mask.shape[0] * sy)), 1)
        if x0 >= W or y0 >= H or x0 + w <= 0 or y0 + h <= 0:
            continue
        m = cv2.resize(it.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
        m = m[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0]
        if not m.any():
            continue
        roi = img[cy0:cy1, cx0:cx1]
        fill = roi.copy()
        fill[m > 0] = color
        roi[:] = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, cnts, -1, color, thick, cv2.LINE_AA)


fl = next(f for f in D.list_fields() if f["lamina"] == LAMINA)
native = cv2.imread(str(fl["path"]))
gt_micro, gt_cap = D.gt_instances(fl)
z = np.load(D.OUT / "pred" / "OOF_none" / f"{LAMINA}.npz")
pr_micro, pr_cap = F.unpack(z, "micro"), F.unpack(z, "cap")
print(LAMINA, "| anotado:", len(gt_micro), "micro,", len(gt_cap), "capilares | modelo:", len(pr_micro), "micro,", len(pr_cap), "capilares")

# ------------------------------------------------------------------ Figura 2: anotacao manual
PW = 1200
PH = round(native.shape[0] * PW / native.shape[1])
ds = native.shape[1] / PW
base = cv2.resize(native, (PW, PH), interpolation=cv2.INTER_AREA)
SXM, SYM = PW / 640, PH / 640


def tag_letter(img, letter):
    cv2.rectangle(img, (0, 0), (86, 86), (255, 255, 255), -1)
    cv2.putText(img, letter, (14, 68), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 5, cv2.LINE_AA)


def tag_legend(img, entries):
    """Legenda dentro do painel (canto superior direito): quadrado colorido + nome."""
    x = img.shape[1] - 20
    y = 20
    for color, text in entries:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x0 = x - tw - 70
        cv2.rectangle(img, (x0, y), (x, y + 56), (255, 255, 255), -1)
        cv2.rectangle(img, (x0 + 10, y + 12), (x0 + 42, y + 44), color, -1)
        cv2.putText(img, text, (x0 + 54, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        y += 66


pa, pb, pc, pd = base.copy(), base.copy(), base.copy(), base.copy()
paint(pb, gt_micro, SXM, SYM, bgr(BLUE), .45, 2)
paint(pc, gt_cap, 1 / ds, 1 / ds, bgr(YELLOW), .65, 1)
paint(pd, gt_micro, SXM, SYM, bgr(BLUE), .35, 2)
paint(pd, gt_cap, 1 / ds, 1 / ds, bgr(YELLOW), .65, 1)
for p, l in zip((pa, pb, pc, pd), "ABCD"):
    tag_letter(p, l)
tag_legend(pb, [(bgr(BLUE), "Microcotyledons")])
tag_legend(pc, [(bgr(YELLOW), "Capillaries")])
tag_legend(pd, [(bgr(BLUE), "Microcotyledons"), (bgr(YELLOW), "Capillaries")])
gut = 14
canvas = np.full((2 * PH + gut, 2 * PW + gut, 3), 255, np.uint8)
for i, p in enumerate((pa, pb, pc, pd)):
    r, c = divmod(i, 2)
    canvas[r * (PH + gut): r * (PH + gut) + PH, c * (PW + gut): c * (PW + gut) + PW] = p
cv2.imwrite(str(FIG / "fig2_annotations_2x2.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("fig2 ok")

# ------------------------------------------------------------------ Figura 3: validacao cruzada por egua
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "text.color": INK})
n = 6
fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=300)
role_color = {"train": BLUE, "val": ORANGE, "test": AQUA}
role_text = {"train": "Train", "val": "Val.", "test": "Test"}
text_col = {"train": "white", "val": INK, "test": INK}
for f in range(n):                       # fold f: egua f e o teste, egua f+1 (ciclico) e a validacao
    for m in range(n):
        role = "test" if m == f else ("val" if m == (f + 1) % n else "train")
        ax.add_patch(Rectangle((m, n - 1 - f), 1, 1, facecolor=role_color[role], edgecolor="white", linewidth=2))
        ax.text(m + .5, n - 1 - f + .5, role_text[role], ha="center", va="center", fontsize=7.5, color=text_col[role])
ax.set_xlim(0, n)
ax.set_ylim(0, n)
ax.set_xticks([i + .5 for i in range(n)], [f"Mare {i + 1}" for i in range(n)])
ax.set_yticks([n - 1 - i + .5 for i in range(n)], [f"Fold {i + 1}" for i in range(n)])
ax.xaxis.tick_top()
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
# fold principal (egua 5 = teste, egua 6 = validacao): usado na normalizacao de cor e nas imagens externas
ax.add_patch(Rectangle((-0.04, n - 1 - 4 - .04), n + .08, 1.08, fill=False, edgecolor=INK, linewidth=1.3, linestyle=(0, (4, 2))))
ax.text(n + .12, n - 1 - 4 + .5, "primary\nfold", ha="left", va="center", fontsize=7.5)
handles = [Rectangle((0, 0), 1, 1, facecolor=role_color[k], edgecolor="none") for k in ("train", "val", "test")]
ax.legend(handles, ["Training (4 mares, 120 fields)", "Validation (1 mare, 30 fields)", "Test (1 mare, 30 fields)"],
          loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=7.5, handlelength=1.2, columnspacing=1.6)
fig.subplots_adjust(left=.09, right=.88, top=.90, bottom=.12)
fig.savefig(FIG / "fig3_cv_design.png", dpi=300)
plt.close(fig)
print("fig3 ok (ordem das eguas: ROSILHA-M, TORDILHA, TP, ZAINA-679, ZAZA, TOSTADA = Mare 1..6)")

# ------------------------------------------------------------------ Figura 4: visao geral do metodo
DW = 1000
DH = round(native.shape[0] * DW / native.shape[1])
dds = native.shape[1] / DW
full = cv2.resize(native, (DW, DH), interpolation=cv2.INTER_AREA)
stretch = cv2.resize(native, (640, 640), interpolation=cv2.INTER_AREA)
s_pred = stretch.copy()
paint(s_pred, pr_micro, 1, 1, bgr(BLUE), .45, 2)
tiled = full.copy()
for k in (1, 2):                                      # grade 3x3 de ladrilhos 1380x1032
    cv2.line(tiled, (round(DW * k / 3), 0), (round(DW * k / 3), DH), (255, 255, 255), 4)
    cv2.line(tiled, (0, round(DH * k / 3)), (DW, round(DH * k / 3)), (255, 255, 255), 4)
c_pred = full.copy()
paint(c_pred, pr_cap, 1 / dds, 1 / dds, bgr(YELLOW), .65, 1)
rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  # noqa: E731

fig = plt.figure(figsize=(7.5, 4.3), dpi=300)
gs = fig.add_gridspec(2, 4, width_ratios=[1.3, 1, 1, 0.95], wspace=.28, hspace=.32, left=.015, right=.985, top=.90, bottom=.03)
a0 = fig.add_subplot(gs[:, 0])
a11, a12, a13 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])
a21, a22, a23 = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])
for a, im in ((a0, full), (a11, stretch), (a12, s_pred), (a21, tiled), (a22, c_pred)):
    a.imshow(rgb(im))
    a.axis("off")
for a, t in ((a0, "Digitized field\n(4140 × 3096 px)"), (a11, "Whole field resized\nto 640 × 640 px"),
             (a12, "Predicted\nmicrocotyledons"), (a21, "Field divided into 3 × 3\ntiles of 1380 × 1032 px"),
             (a22, "Predicted capillaries\n(tiles merged)")):
    a.set_title(t, fontsize=7.5, color=INK, pad=3)
for a, lines in ((a13, ["Microcotyledons", "", "• area (µm²)", "• share of the field (%)", "• number of profiles"]),
                 (a23, ["Capillaries", "", "• area (µm²)", "• share of the", "   microcotyledon area (%)", "• number of profiles"])):
    a.axis("off")
    a.text(0.0, 0.5, "\n".join(lines), ha="left", va="center", fontsize=7.5, color=INK, linespacing=1.5)
a13.set_title("Measurements per field", fontsize=7.5, color=INK, pad=3)
fig.canvas.draw()


def arrow(src, dst, dy_src=0.0):
    """Da borda direita de `src` (deslocada em dy_src) ao meio da borda esquerda de `dst`."""
    p0, p1 = src.get_position(), dst.get_position()
    fig.add_artist(FancyArrowPatch((p0.x1 + .004, (p0.y0 + p0.y1) / 2 + dy_src), (p1.x0 - .004, (p1.y0 + p1.y1) / 2),
                                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1.0))


arrow(a0, a11, dy_src=.05)
arrow(a0, a21, dy_src=-.05)
arrow(a11, a12)
arrow(a12, a13)
arrow(a21, a22)
arrow(a22, a23)
fig.savefig(FIG / "fig4_method_overview.png", dpi=300)
plt.close(fig)
print("fig4 ok")
