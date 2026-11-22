"""Fluxo de validacao em imagens SEM rotulo (artigo ISER): campeoes x normalizacoes de cor.

Etapas (todas retomaveis; cada uma pula o que ja existe):
  normalize  normaliza as imagens nativas com cada metodo (referencia = mosaico dos 120 campos de treino do fold
             principal, a mesma do benchmark) e mede o deslocamento de cor em relacao ao treino.
  infer      roda os campeoes (micro YOLO26s + capilar YOLO11s com SAHI) em cada variante e salva as mascaras.
  report     tabelas (por variante, pareadas contra o original, concordancia entre variantes) e figuras.
  panels     folhas comparativas por imagem (entrada normalizada + sobreposicao de cada variante).

Variantes (nome = <modo>_<norma>):
  A_<n>  modelo TREINADO com a normalizacao n (fase 2 do benchmark) + entrada normalizada com n
  B_<n>  modelo base (treinado sem normalizacao) + entrada normalizada com n (so na entrada)
  A_none = modelo base + imagem original.

Sem rotulo nao ha acuracia: as metricas sao descritivas (contagens, areas, confianca, concordancia entre variantes,
distancia de cor ao treino). O criterio de parada e o arquivo STOP na pasta de saida.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BENCH = Path(r"D:\projeto_placentas_clayton\benchmark_results")
FOLD_DIR = Path(r"D:\projeto_placentas_clayton\datasets_v4\test-ZAZA_val-TOSTADA")
# Conjunto de imagens externas: run1 = 90 imagens originais (padrao); run2 = 150 imagens com coloracao boa (2026-11-20),
# escolhido pela variavel de ambiente ISER_IN_DIR (os outros scripts importam este modulo e herdam a escolha).
IN_DIR = Path(os.environ.get("ISER_IN_DIR", r"D:\projeto_placentas_clayton\imagens_teste_artigo_iser"))
DEFAULT_OUT = Path(os.environ.get("ISER_OUT", r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\run1"))

NORM_ORDER = ["reinhard", "macenko", "vahadane", "whitebalance", "histmatch", "grayscale"]
VARIANTS = ["A_none"] + [f"{m}_{n}" for n in NORM_ORDER for m in ("A", "B")]
MICRO_MODEL, CAP_MODEL = "yolo26s-seg.pt", "yolo11s-seg.pt"
FIELD_W, FIELD_H = 4140, 3096


def variant_norms(v: str) -> tuple[str, str]:
    """(norma com que o modelo foi treinado, norma aplicada a imagem)."""
    mode, n = v.split("_", 1)
    return ("none" if mode == "B" else n), n


def list_images(limit: int | None = None) -> list[Path]:
    files = sorted(p for p in IN_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"))
    return files[:limit] if limit else files


def image_info(p: Path) -> dict:
    t = p.stem.split()
    return {"image": p.stem, "animal": t[0], "region": t[2], "idx": int(t[3])}


def norm_path(out: Path, method: str, img: Path) -> Path:
    return img if method == "none" else out / "norm" / method / f"{img.stem}.png"


def stop_requested(out: Path) -> bool:
    return (out / "STOP").exists()


# ------------------------------------------------------------------ checkpoints do benchmark
def model_table() -> dict:
    """{(task, norm): (run_id, conf*, weights)} dos campeoes no fold principal (conf* = o da validacao)."""
    df = pd.read_csv(BENCH / "results.csv")
    df["fold"] = df["fold"].fillna("primary")
    out = {}
    for task, model in (("micro", MICRO_MODEL), ("capilar", CAP_MODEL)):
        d = df[(df.task == task) & (df.model == model) & (df.fold == "primary")]
        for _, r in d.iterrows():
            w = BENCH / r.run_id / "train" / "run" / "weights" / "best.pt"
            out[(task, r.norm)] = (r.run_id, float(r.val_conf_star), w)
    return out


# ------------------------------------------------------------------ normalizacao
def get_normalizer(method: str):
    from benchmark.views.stain import StainNormalizer

    train = sorted((FOLD_DIR / "native" / "train" / "images").iterdir())
    ref = FOLD_DIR / f"reference_{method}.json"
    saved = json.loads(ref.read_text(encoding="utf-8"))["reference"] if ref.exists() else None
    return StainNormalizer(method).fit(train, saved_ref=saved)


def cmd_normalize(a) -> None:
    from benchmark.views.stain import _read_rgb, _valid_output

    out = a.out
    imgs = list_images(a.limit)
    for m in [n for n in NORM_ORDER if not a.norms or n in a.norms]:
        d = out / "norm" / m
        d.mkdir(parents=True, exist_ok=True)
        todo = [p for p in imgs if not _valid_output(norm_path(out, m, p), p)]
        rep_file = d / "_report.json"
        rep = json.loads(rep_file.read_text()) if rep_file.exists() else {"fallbacks": [], "seconds": {}}
        print(f"[{m}] {len(imgs) - len(todo)} prontas, {len(todo)} a normalizar", flush=True)
        if not todo:
            continue
        t0 = time.time()
        norm = get_normalizer(m)
        print(f"[{m}] referencia ajustada em {time.time() - t0:.0f} s", flush=True)
        for i, p in enumerate(todo, 1):
            if stop_requested(out):
                print("STOP encontrado: saindo.", flush=True)
                return
            t1 = time.time()
            rgb, diag = norm.transform(_read_rgb(p))
            tgt = norm_path(out, m, p)
            tmp = tgt.with_name(tgt.stem + ".tmp.png")
            cv2.imwrite(str(tmp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            tmp.replace(tgt)
            rep["seconds"][p.stem] = round(time.time() - t1, 2)
            if any("fallback" in v for v in diag.values()):
                rep["fallbacks"].append({"image": p.stem, "diag": diag})
            rep_file.write_text(json.dumps(rep, indent=1), encoding="utf-8")
            print(f"[{m}] {i}/{len(todo)} {p.stem} {rep['seconds'][p.stem]:.1f} s", flush=True)
    cmd_color(a)


def _lab_sample(bgr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return lab[rng.choice(len(lab), min(n, len(lab)), replace=False)].astype(np.float32)


def cmd_color(a) -> None:
    """Distancia de cor ao treino: Wasserstein-1 por canal Lab (escala 0-255 do OpenCV) entre os pixels da imagem
    (reduzida 8x) e os do mosaico de treino bruto. Referencia interna: os proprios campos de treino brutos."""
    from scipy.stats import wasserstein_distance as wd

    from benchmark.views.stain import reference_mosaic

    out = a.out
    f = out / "cor.csv"
    rng = np.random.default_rng(0)
    train = sorted((FOLD_DIR / "native" / "train" / "images").iterdir())
    ref = _lab_sample(cv2.cvtColor(reference_mosaic(train), cv2.COLOR_RGB2BGR), 200_000, rng)
    rows = []

    def one(bgr, subset, method, name):
        s = _lab_sample(bgr, 60_000, rng)
        d = [wd(s[:, c], ref[:, c]) for c in range(3)]
        rows.append({"set": subset, "method": method, "image": name, "wd_L": d[0], "wd_a": d[1], "wd_b": d[2],
                     "wd_mean": float(np.mean(d)), "L": s[:, 0].mean(), "a": s[:, 1].mean(), "b": s[:, 2].mean()})

    for p in train:
        one(cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8), "train", "none", p.stem[:24])
    for p in list_images(a.limit):
        one(cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8), "new", "none", p.stem)
        for m in NORM_ORDER:
            q = norm_path(out, m, p)
            if q.exists():
                one(cv2.imread(str(q), cv2.IMREAD_REDUCED_COLOR_8), "new", m, p.stem)
    pd.DataFrame(rows).to_csv(f, index=False)
    print(f"cor.csv: {len(rows)} linhas", flush=True)


# ------------------------------------------------------------------ mascaras compactas
def pack(items: list[tuple[int, int, np.ndarray]], confs: list[float]) -> dict:
    meta = np.array([[y0, x0, c.shape[0], c.shape[1]] for y0, x0, c in items], dtype=np.int32).reshape(-1, 4)
    bits = [np.packbits(c.astype(bool).ravel()) for _, _, c in items]
    lens = np.array([len(b) for b in bits], dtype=np.int64)
    return {"meta": meta, "lens": lens, "bits": np.concatenate(bits) if bits else np.zeros(0, np.uint8),
            "conf": np.asarray(confs, dtype=np.float32)}


def unpack(z, prefix: str):
    from benchmark.eval.instances import Inst

    meta, lens, bits, conf = z[f"{prefix}_meta"], z[f"{prefix}_lens"], z[f"{prefix}_bits"], z[f"{prefix}_conf"]
    off, out = 0, []
    for (y0, x0, h, w), n, c in zip(meta, lens, conf):
        m = np.unpackbits(bits[off: off + n])[: h * w].reshape(h, w).astype(bool)
        out.append(Inst(int(x0), int(y0), m, float(c)))
        off += int(n)
    return out


def _crops_from_full(masks) -> list[tuple[int, int, np.ndarray]]:
    items = []
    for m in masks:
        rows = np.flatnonzero(m.any(axis=1))
        if rows.size == 0:
            continue
        cols = np.flatnonzero(m.any(axis=0))
        items.append((int(rows[0]), int(cols[0]), m[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1] > 0))
    return items


# ------------------------------------------------------------------ inferencia
def cmd_infer(a) -> None:
    import torch

    from pipeline.capilar import CapilarModel
    from pipeline.config import PipelineConfig
    from pipeline.micro import MicrocotyledonYOLOModel

    out = a.out
    imgs = list_images(a.limit)
    table = model_table()
    wanted = [v for v in VARIANTS if not a.variants or v in a.variants]
    for v in wanted:
        mn, imn = variant_norms(v)
        pdir = out / "pred" / v
        pdir.mkdir(parents=True, exist_ok=True)
        todo = [p for p in imgs if not (pdir / f"{p.stem}.json").exists()]
        print(f"[{v}] {len(imgs) - len(todo)} prontas, {len(todo)} a rodar", flush=True)
        if not todo:
            continue
        _, mconf, mw = table[("micro", mn)]
        _, cconf, cw = table[("capilar", mn)]
        for w in (mw, cw):
            if not w.is_file():
                raise FileNotFoundError(w)
        cfg = PipelineConfig(micro_yolo_ckpt=str(mw), micro_yolo_conf=mconf, capilar_ckpt=str(cw), capilar_conf=cconf)
        micro, cap = MicrocotyledonYOLOModel(cfg), CapilarModel(cfg)
        field_um2 = FIELD_W * FIELD_H * cfg.micro_area_um2_per_px2 / ((FIELD_W / cfg.micro_canvas) * (FIELD_H / cfg.micro_canvas))
        for i, p in enumerate(todo, 1):
            if stop_requested(out):
                print("STOP encontrado: saindo.", flush=True)
                return
            src = norm_path(out, imn, p)
            if not src.exists():
                print(f"[{v}] imagem normalizada ausente: {src} (rode normalize)", flush=True)
                continue
            bgr = cv2.imread(str(src))
            h, w = bgr.shape[:2]
            t0 = time.time()
            rm = micro.predict_bgr(bgr)
            t1 = time.time()
            rc = cap.predict_path(src)
            t2 = time.time()
            mi = _crops_from_full(rm.masks)
            ci = [rc.masks.crop(k) for k in range(len(rc.masks))]
            np.savez_compressed(
                pdir / f"{p.stem}.npz",
                **{f"micro_{k}": x for k, x in pack(mi, rm.confidences[: len(mi)]).items()},
                **{f"cap_{k}": x for k, x in pack(ci, rc.confidences[: len(ci)]).items()},
            )
            row = {**image_info(p), "variant": v, "model_norm": mn, "input_norm": imn, "size": f"{w}x{h}",
                   "conf_micro": mconf, "conf_cap": cconf,
                   "n_micro": rm.count, "micro_area_um2": rm.area_um2, "micro_union_um2": rm.area_union_um2,
                   "micro_frac_pct": 100 * rm.area_union_um2 / field_um2,
                   "micro_conf_mean": float(np.mean(rm.confidences)) if rm.confidences else np.nan,
                   "micro_area_med_um2": float(np.median(rm.areas_um2)) if rm.areas_um2 else np.nan,
                   "n_cap": rc.count, "cap_area_um2": rc.area_um2, "cap_union_um2": rc.area_union_um2,
                   "cap_over_micro_pct": 100 * rc.area_um2 / rm.area_um2 if rm.area_um2 else np.nan,
                   "cap_conf_mean": float(np.mean(rc.confidences)) if rc.confidences else np.nan,
                   "cap_area_med_um2": float(np.median(rc.areas_um2)) if rc.areas_um2 else np.nan,
                   "t_micro_s": t1 - t0, "t_cap_s": t2 - t1}
            (pdir / f"{p.stem}.json").write_text(json.dumps(row), encoding="utf-8")
            print(f"[{v}] {i}/{len(todo)} {p.stem}: micro {rm.count}, cap {rc.count}, {t2 - t0:.1f} s", flush=True)
        del micro, cap
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ relatorio
def load_rows(out: Path) -> pd.DataFrame:
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((out / "pred").glob("*/*.json"))]
    df = pd.DataFrame(rows)
    df["variant"] = pd.Categorical(df["variant"], [v for v in VARIANTS if v in set(df["variant"])], ordered=True)
    return df.sort_values(["variant", "image"]).reset_index(drop=True)


def f1_between(a, b, thr: float = 0.5) -> float:
    """F1 de instancias entre duas listas de Inst (casamento guloso por IoU, simetrico)."""
    from benchmark.eval.instances import iou

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    A, B = np.array([x.box for x in a]), np.array([x.box for x in b])
    ox = np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]) > 0
    oy = np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]) > 0
    pairs = sorted(((iou(a[i], b[j]), i, j) for i, j in np.argwhere(ox & oy)), reverse=True)
    ua, ub, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i in ua or j in ub:
            continue
        ua.add(i), ub.add(j)
        tp += 1
    return 2 * tp / (len(a) + len(b))


def union_small(insts, h: int, w: int, step: int) -> np.ndarray:
    from benchmark.eval.instances import union_mask

    return union_mask(insts, h, w)[::step, ::step]


def cmd_report(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon

    out = a.out
    rep = out / "relatorio"
    rep.mkdir(exist_ok=True)
    df = load_rows(out)
    df.to_csv(rep / "linhas_por_imagem.csv", index=False)
    variants = [v for v in VARIANTS if v in set(df.variant)]
    cols = ["n_micro", "micro_frac_pct", "micro_conf_mean", "micro_area_med_um2", "n_cap", "cap_area_um2",
            "cap_over_micro_pct", "cap_conf_mean", "cap_area_med_um2", "t_cap_s"]

    # 1) resumo por variante (e quantas imagens ficaram quase vazias)
    g = df.groupby("variant", observed=True)
    summ = g[cols].agg(["mean", "std", "median"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ["n_images"] = g.size()
    summ["micro_lt5"] = g["n_micro"].apply(lambda s: int((s < 5).sum()))
    summ["cap_lt50"] = g["n_cap"].apply(lambda s: int((s < 50).sum()))
    summ.to_csv(rep / "resumo_por_variante.csv")

    # 2) diferencas pareadas contra o original (A_none), por imagem
    base = df[df.variant == "A_none"].set_index("image")
    prow = []
    for v in variants[1:]:
        d = df[df.variant == v].set_index("image")
        common = base.index.intersection(d.index)
        for c in ("n_micro", "micro_frac_pct", "n_cap", "cap_over_micro_pct", "cap_conf_mean", "micro_conf_mean"):
            diff = (d.loc[common, c] - base.loc[common, c]).dropna()
            try:
                p = wilcoxon(diff).pvalue if len(diff) > 5 and (diff != 0).any() else np.nan
            except ValueError:
                p = np.nan
            prow.append({"variant": v, "metric": c, "n": len(diff), "base_mean": base.loc[common, c].mean(),
                         "mean_diff": diff.mean(), "median_diff": diff.median(), "wilcoxon_p": p})
    pd.DataFrame(prow).to_csv(rep / "pareado_vs_original.csv", index=False)

    # 3) concordancia entre variantes (F1 de instancias par a par; Dice contra consenso leave-one-out)
    stems = sorted(df.image.unique())
    f1m = {t: np.full((len(variants), len(variants), len(stems)), np.nan) for t in ("micro", "cap")}
    dice = {t: np.full((len(variants), len(stems)), np.nan) for t in ("micro", "cap")}
    for k, s in enumerate(stems):
        z = {}
        for v in variants:
            f = out / "pred" / v / f"{s}.npz"
            if f.exists():
                z[v] = np.load(f)
        if len(z) < len(variants):
            continue
        inst = {t: {v: unpack(z[v], t) for v in variants} for t in ("micro", "cap")}
        for t, (hh, ww, step) in {"micro": (640, 640, 1), "cap": (FIELD_H, FIELD_W, 4)}.items():
            for i, vi in enumerate(variants):
                for j, vj in enumerate(variants):
                    if j > i:
                        f1m[t][i, j, k] = f1m[t][j, i, k] = f1_between(inst[t][vi], inst[t][vj])
                f1m[t][i, i, k] = 1.0
            un = np.stack([union_small(inst[t][v], hh, ww, step) for v in variants]).astype(np.uint8)
            tot = un.sum(0)
            n = len(variants)
            for i in range(n):
                cons = (tot - un[i]) * 2 > (n - 1)
                inter = (cons & (un[i] > 0)).sum()
                dice[t][i, k] = 2 * inter / max(cons.sum() + un[i].sum(), 1)
        print(f"concordancia {k + 1}/{len(stems)}", flush=True)
    crow = []
    for t in ("micro", "cap"):
        pd.DataFrame(np.nanmean(f1m[t], axis=2), index=variants, columns=variants).to_csv(rep / f"f1_instancias_{t}.csv")
        for i, v in enumerate(variants):
            crow.append({"task": t, "variant": v, "dice_vs_consenso_mean": np.nanmean(dice[t][i]),
                         "dice_vs_consenso_sd": np.nanstd(dice[t][i], ddof=1), "f1_vs_original_mean": np.nanmean(f1m[t][i, 0]),
                         "f1_vs_original_sd": np.nanstd(f1m[t][i, 0], ddof=1)})
    pd.DataFrame(crow).to_csv(rep / "concordancia_por_variante.csv", index=False)

    # 4) figuras
    colors = {v: plt.cm.tab20(i) for i, v in enumerate(variants)}
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (c, t) in zip(axs.ravel(), (("n_micro", "Microcotiledones por campo"), ("micro_frac_pct", "Area microcotiledonar (% do campo)"),
                                         ("n_cap", "Capilares por campo"), ("cap_over_micro_pct", "Area capilar / microcotiledonar (%)"))):
        ax.boxplot([df[df.variant == v][c].dropna() for v in variants], tick_labels=variants, showfliers=True)
        ax.set_title(t)
        ax.tick_params(axis="x", rotation=60)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(rep / "fig_medidas_por_variante.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, t in zip(axs, ("micro", "cap")):
        m = np.nanmean(f1m[t], axis=2)
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(variants)), variants, rotation=60, ha="right")
        ax.set_yticks(range(len(variants)), variants)
        for i in range(len(variants)):
            for j in range(len(variants)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if m[i, j] < .6 else "k")
        ax.set_title("F1 de instancias entre variantes: " + ("microcotiledones" if t == "micro" else "capilares"))
    fig.colorbar(im, ax=axs, shrink=.8)
    fig.savefig(rep / "fig_concordancia.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    if (out / "cor.csv").exists():
        cor = pd.read_csv(out / "cor.csv")
        labs = [("train", "none", "treino bruto")] + [("new", m, "novas " + ("brutas" if m == "none" else m))
                                                      for m in ["none"] + NORM_ORDER]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot([cor[(cor.set == s) & (cor.method == m)].wd_mean for s, m, _ in labs], tick_labels=[l for *_, l in labs])
        ax.set_ylabel("Wasserstein-1 medio em Lab (0-255) ao mosaico de treino")
        ax.tick_params(axis="x", rotation=40)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(rep / "fig_distancia_cor.png", dpi=130)
        plt.close(fig)
        cor.groupby(["set", "method"]).wd_mean.agg(["mean", "std", "median"]).to_csv(rep / "distancia_cor.csv")
    print("relatorio em", rep, flush=True)


# ------------------------------------------------------------------ folhas comparativas
def _paint(img: np.ndarray, items, sx: float, sy: float, ox: float, oy: float, color, alpha: float = .25) -> None:
    """Pinta mascaras guardadas como recortes (y0, x0, mask) de um espaco com escala (sx, sy) e origem (ox, oy)."""
    H, W = img.shape[:2]
    for it in items:
        x0, y0 = int(round((it.x0 - ox) * sx)), int(round((it.y0 - oy) * sy))
        w, h = max(int(round(it.mask.shape[1] * sx)), 1), max(int(round(it.mask.shape[0] * sy)), 1)
        if x0 >= W or y0 >= H or x0 + w <= 0 or y0 + h <= 0:
            continue
        m = cv2.resize(it.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x0 + w, W), min(y0 + h, H)
        m = m[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0]
        if not m.any():
            continue
        roi = img[cy0:cy1, cx0:cx1]
        fill = roi.copy()
        fill[m > 0] = color
        roi[:] = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, cnts, -1, color, 1)


def _label(cell: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((34, cell.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, cell])


def make_sheet(out: Path, img: Path, region: tuple[int, int, int, int], cell_w: int, dest: Path) -> None:
    """Colunas = original + 6 normalizacoes; linhas = imagem de entrada, modelo base (B), modelo treinado com a norma (A)."""
    x0, y0, x1, y1 = region
    ds = (x1 - x0) / cell_w
    cell_h = int(round((y1 - y0) / ds))
    norms = ["none"] + NORM_ORDER
    cols = []
    for n in norms:
        src = cv2.imread(str(norm_path(out, n, img)))
        if src is None:
            continue
        base = cv2.resize(src[y0:y1, x0:x1], (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        cells = [_label(base, f"{n}: entrada")]
        for mode in ("B", "A"):
            v = "A_none" if n == "none" else f"{mode}_{n}"
            if n == "none" and mode == "B":
                cells.append(_label(np.full_like(base, 255), "(igual a A)"))
                continue
            f = out / "pred" / v / f"{img.stem}.npz"
            c = base.copy()
            if f.exists():
                z = np.load(f)
                mi, ci = unpack(z, "micro"), unpack(z, "cap")
                # micro: espaco 640x640 esticado -> nativo; capilar: nativo
                _paint(c, mi, FIELD_W / 640 / ds, FIELD_H / 640 / ds, x0 / (FIELD_W / 640), y0 / (FIELD_H / 640), (60, 200, 60), .2)
                _paint(c, ci, 1 / ds, 1 / ds, x0, y0, (255, 140, 0), .35)
                txt = f"{v}: {len(mi)} micro, {len(ci)} cap"
            else:
                txt = f"{v}: sem predicao"
            cells.append(_label(c, txt))
        cols.append(np.vstack(cells))
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 88])


def cmd_panels(a) -> None:
    out = a.out
    imgs = list_images(a.limit)
    for p in imgs:
        if stop_requested(out):
            return
        t = out / "folhas_detalhe" / f"{p.stem}.jpg"
        if a.force or not t.exists():
            make_sheet(out, p, (1380, 1032, 2760, 2064), 690, t)  # tile central 1380x1032 a 1/2 escala
            print("detalhe", p.stem, flush=True)
        if p.stem.endswith(" 1"):  # uma representante por animal-regiao
            t = out / "folhas_campo" / f"{p.stem}.jpg"
            if a.force or not t.exists():
                make_sheet(out, p, (0, 0, FIELD_W, FIELD_H), 690, t)
                print("campo", p.stem, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["normalize", "color", "infer", "report", "panels", "all"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="usa so as N primeiras imagens (teste rapido)")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--norms", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="panels: refaz as folhas existentes")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    steps = {"all": [cmd_normalize, cmd_infer, cmd_report, cmd_panels]}.get(
        a.cmd, [{"normalize": cmd_normalize, "color": cmd_color, "infer": cmd_infer, "report": cmd_report,
                 "panels": cmd_panels}[a.cmd]])
    for s in steps:
        s(a)


if __name__ == "__main__":
    main()
