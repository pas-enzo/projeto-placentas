"""Gera, a partir das predicoes JA gravadas (pred/<variante>/*.npz), TUDO que o pipeline gera, sem rodar o modelo:

  pipeline/<variante>/
      results.csv                 uma linha por imagem (contagem, area total, fracao...)
      instances.csv               uma linha por instancia (area, confianca, perimetro, eixos, circularidade, solidez)
      annotations.coco.json       poligonos nativos + area_um2 + score
      overlays/<imagem>_overlay.jpg     sobreposicao em resolucao nativa (verde = microcotiledone, ciano = capilar)
      lado_a_lado/<imagem>.jpg          entrada do modelo | sobreposicao
      masks/<imagem>_micro.png, _capilar.png   mascaras binarias (uniao), tamanho nativo
      run_info.json               configuracao e versoes

A "entrada do modelo" e a imagem ja normalizada da variante (a que o modelo viu). Usa as funcoes do proprio pipeline
(compute_geometry, colorize_overlay, save_image_artifacts, write_run_artifacts): mesmo formato do app.
Retomavel: sobreposicoes ja gravadas sao puladas; os CSV/COCO sao sempre reescritos com todas as imagens da variante.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.artigo_iser import fluxo as F  # noqa: E402

PRIORIDADE = ["A_none", "A_reinhard", "B_reinhard"]


def rebuild(img: Path, z, cfg, w: int, h: int):
    from pipeline.types import CropMasks, ImageResult, InstanceResult

    canvas = int(cfg.micro_canvas)
    mi, ci = F.unpack(z, "micro"), F.unpack(z, "cap")

    mm = [m.full(canvas, canvas).astype(np.uint8) for m in mi]
    ap = [float(m.sum()) for m in mm]
    un = np.zeros((canvas, canvas), bool)
    for m in mm:
        un |= m > 0
    f = float(cfg.micro_area_um2_per_px2)
    micro = InstanceResult(count=len(mm), area_px=sum(ap), area_um2=sum(ap) * f, masks=mm,
                           confidences=[i.score for i in mi], model_name=cfg.micro_yolo_name,
                           areas_um2=[a * f for a in ap], area_union_px=float(un.sum()), area_union_um2=float(un.sum()) * f)

    cm = CropMasks((h, w))
    u = np.zeros((h, w), np.uint8)
    cp = []
    for i in ci:
        crop = i.mask.astype(np.uint8)
        cm._items.append((i.y0, i.x0, crop))
        u[i.y0: i.y0 + crop.shape[0], i.x0: i.x0 + crop.shape[1]] |= crop
        cp.append(float(crop.sum()))
    g = float(cfg.capilar_area_um2_per_px2)
    cap = InstanceResult(count=len(ci), area_px=sum(cp), area_um2=sum(cp) * g, masks=cm,
                         confidences=[i.score for i in ci], model_name="YOLO11s-seg + SAHI",
                         areas_um2=[a * g for a in cp], area_union_px=float(u.sum()), area_union_um2=float(u.sum()) * g)
    return ImageResult(image_path=img, width=w, height=h, micro=micro, capilar=cap)


def export_variant(args) -> str:
    from pipeline.artifacts import save_image_artifacts, write_run_artifacts
    from pipeline.config import PipelineConfig
    from pipeline.geometry import compute_geometry
    from pipeline.viz import colorize_overlay

    out, v, limit, force = args
    mn, imn = F.variant_norms(v)
    table = F.model_table()
    _, mconf, mw = table[("micro", mn)]
    _, cconf, cw = table[("capilar", mn)]
    cfg = PipelineConfig(micro_yolo_ckpt=str(mw), micro_yolo_conf=mconf, capilar_ckpt=str(cw), capilar_conf=cconf)
    dest = out / "pipeline" / v
    results, bad = [], []
    t0 = time.time()
    imgs = [p for p in F.list_images(limit) if (out / "pred" / v / f"{p.stem}.json").exists()]
    for k, img in enumerate(imgs, 1):
        meta = json.loads((out / "pred" / v / f"{img.stem}.json").read_text(encoding="utf-8"))
        w, h = (int(x) for x in meta["size"].split("x"))
        res = rebuild(img, np.load(out / "pred" / v / f"{img.stem}.npz"), cfg, w, h)
        # conferencia: o que reconstruimos tem de bater com o que foi medido na inferencia
        if res.micro.count != meta["n_micro"] or res.capilar.count != meta["n_cap"] or \
                abs(res.capilar.area_um2 - meta["cap_area_um2"]) > 1e-6 * max(meta["cap_area_um2"], 1) or \
                abs(res.micro.area_um2 - meta["micro_area_um2"]) > 1e-6 * max(meta["micro_area_um2"], 1):
            bad.append(img.stem)
        compute_geometry(res, cfg)
        stem = img.stem
        ov = dest / "overlays" / f"{stem}_overlay.jpg"
        if force or not ov.exists():
            bgr = cv2.imread(str(F.norm_path(out, imn, img)))
            u640 = np.zeros((cfg.micro_canvas,) * 2, np.uint8)
            for m in res.micro.masks:
                u640 |= (m > 0).astype(np.uint8)
            # a uniao redimensionada (vizinho mais proximo) e igual a uniao das mascaras redimensionadas
            micro_native = [cv2.resize(u640, (w, h), interpolation=cv2.INTER_NEAREST)]
            overlay = colorize_overlay(bgr, micro_masks=micro_native, capilar_masks=res.capilar.masks)
            save_image_artifacts(res, bgr, overlay, dest, stem, cfg, save_overlays=True, save_masks=True)
        else:
            res.extras["stem"] = stem
            res.overlay_path = Path("overlays") / ov.name
        res.drop_masks()
        results.append(res)
        if k % 10 == 0 or k == len(imgs):
            print(f"[{v}] {k}/{len(imgs)} ({time.time() - t0:.0f} s)", flush=True)
    if results:
        write_run_artifacts(results, dest, cfg, mode="both", micro_conf=mconf, capilar_conf=cconf)
    return f"{v}: {len(results)} imagens | divergencias de reconstrucao: {len(bad)} {bad[:3]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=F.DEFAULT_OUT)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="refaz as sobreposicoes ja existentes")
    a = ap.parse_args()
    have = [d.name for d in (a.out / "pred").iterdir() if d.is_dir() and any(d.glob("*.json"))]
    todo = [v for v in PRIORIDADE + [x for x in F.VARIANTS if x not in PRIORIDADE] if v in have and (not a.variants or v in a.variants)]
    (a.out / "pipeline").mkdir(exist_ok=True)
    print("variantes:", todo, flush=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(export_variant, [(a.out, v, a.limit, a.force) for v in todo]):
            print("PRONTO", msg, flush=True)


if __name__ == "__main__":
    main()
