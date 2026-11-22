"""Folhas comparativas separadas por classe (microcotiledone e capilar), CAMPO INTEIRO, para as 90 imagens novas.

Le apenas o que ja existe (run1/norm e run1/pred); nao roda modelo.

Cada folha: 3 colunas x 7 linhas (uma linha por normalizacao: none, reinhard, macenko, vahadane, whitebalance,
histmatch, grayscale):
  coluna 1  imagem de entrada (ja normalizada), sem mascara
  coluna 2  B_<norma>: modelo original aplicado a imagem normalizada
  coluna 3  A_<norma>: modelo treinado com aquela normalizacao (na linha 'none' e o modelo original)

Saida:
  run1/folhas_por_classe/micro/<imagem>.jpg    so microcotiledones (verde)
  run1/folhas_por_classe/capilar/<imagem>.jpg  so capilares (amarelo)
O numero no titulo de cada celula e a contagem de instancias do campo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fluxo as F  # noqa: E402

VERDE = (40, 210, 40)      # BGR
AMARELO = (0, 235, 255)    # BGR
CELL_W = 800


def paint(img, items, sx, sy, color, alpha, thick):
    """Mascara guardada como recortes (y0, x0, mask); preenchimento forte + contorno grosso."""
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


def sheet(out: Path, img: Path, cls: str, dest: Path) -> None:
    ds = F.FIELD_W / CELL_W
    cell_h = int(round(F.FIELD_H / ds))
    rows = []
    for n in ["none"] + F.NORM_ORDER:
        src = cv2.imread(str(F.norm_path(out, n, img)))
        if src is None:
            continue
        base = cv2.resize(src, (CELL_W, cell_h), interpolation=cv2.INTER_AREA)
        cells = [F._label(base, f"{n}: entrada")]
        for mode in ("B", "A"):
            v = "A_none" if n == "none" else f"{mode}_{n}"
            if n == "none" and mode == "B":
                cells.append(F._label(np.full_like(base, 255), "(igual a A)"))
                continue
            f = out / "pred" / v / f"{img.stem}.npz"
            c = base.copy()
            if not f.exists():
                cells.append(F._label(c, f"{v}: sem predicao"))
                continue
            z = np.load(f)
            if cls == "micro":
                items = F.unpack(z, "micro")  # espaco 640x640 esticado
                paint(c, items, CELL_W / 640, cell_h / 640, VERDE, .5, 2)
                txt = f"{v}: {len(items)} micro"
            else:
                items = F.unpack(z, "cap")    # espaco nativo
                paint(c, items, 1 / ds, 1 / ds, AMARELO, .65, 1)
                txt = f"{v}: {len(items)} cap"
            cells.append(F._label(c, txt))
        rows.append(np.hstack(cells))
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=F.DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for i, p in enumerate(F.list_images(a.limit), 1):
        if F.stop_requested(a.out):
            print("STOP", flush=True)
            return
        for cls, sub in (("micro", "micro"), ("cap", "capilar")):
            dest = a.out / "folhas_por_classe" / sub / f"{p.stem}.jpg"
            if a.force or not dest.exists():
                sheet(a.out, p, cls, dest)
        print(f"{i} {p.stem}", flush=True)
    print("FOLHAS POR CLASSE COMPLETAS", flush=True)


if __name__ == "__main__":
    main()
