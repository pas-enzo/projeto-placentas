"""Regenera os campos NATIVOS normalizados do fold principal (datasets_v4/<fold>/native_<metodo>).

O build_views apagava essas pastas por padrao depois de gerar as vistas. Aqui elas sao recriadas com a MESMA referencia
salva (reference_<metodo>.json; o Vahadane restaura a matriz de coloracao), e MANTIDAS.
Seguro: so escreve em native_<metodo>; nunca toca nas vistas (micro_stretch640_*, capilar_tiles3x3_*) nem em native/.
Retomavel (campos ja gerados e validos sao pulados).

  python scripts/artigo_iser/regenerar_nativos.py --verify            # confere 1 campo contra a vista de treino
  python scripts/artigo_iser/regenerar_nativos.py                      # todos os metodos (Vahadane por ultimo)
  python scripts/artigo_iser/regenerar_nativos.py --methods reinhard   # um metodo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.views.stain import StainNormalizer, _read_rgb, normalize_dir  # noqa: E402

FD = Path(r"D:\projeto_placentas_clayton\datasets_v4\test-ZAZA_val-TOSTADA")
ORDER = ["whitebalance", "histmatch", "grayscale", "reinhard", "macenko", "vahadane"]


def make_norm(m: str, train: list[Path]) -> StainNormalizer:
    ref = json.loads((FD / f"reference_{m}.json").read_text(encoding="utf-8"))["reference"]
    return StainNormalizer(m).fit(train, saved_ref=ref)


def verify(methods: list[str]) -> None:
    """Normaliza 1 campo de treino, estica para 640x640 (como o build_micro) e compara com a vista salva."""
    train = sorted((FD / "native" / "train" / "images").iterdir())
    p = train[5]
    for m in methods:
        view = next((FD / f"micro_stretch640_{m}" / "train" / "images").glob(p.stem + ".*"))
        rgb, _ = make_norm(m, train).transform(_read_rgb(p))
        mine = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (640, 640), interpolation=cv2.INTER_LINEAR)
        saved = cv2.imread(str(view))
        d = np.abs(mine.astype(int) - saved.astype(int))
        print(f"{m}: campo {p.stem[:20]} | max diff {d.max()} | media {d.mean():.4f} | identico={d.max() == 0}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="*", default=ORDER)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        return verify(a.methods)
    native = FD / "native"
    train = sorted((native / "train" / "images").iterdir())
    for m in a.methods:
        dst = FD / f"native_{m}"
        t0 = time.time()
        print(f"[{m}] ajustando referencia (restaurada) e normalizando 180 campos em {dst}", flush=True)
        rep = normalize_dir(native, dst, make_norm(m, train), resume=True)
        rep.pop("diagnostics", None)
        rep["seconds"] = round(time.time() - t0)
        (dst / "_regenerado.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"[{m}] pronto: {rep['images']} campos, fallbacks={len(rep['fallbacks'])}, {rep['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
