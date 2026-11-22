"""Lamina anotada x lamina inferida: area e contagem de microcotiledones e capilares, para as 180 laminas anotadas.

Compara o que o ESPECIALISTA anotou com o que o MODELO detecta, para cada tipo de normalizacao (mesmas variantes do fluxo
das 90 imagens novas), para que se possa decidir qual normalizacao usar com numeros e imagens.

Etapas (retomaveis; cada uma pula o que ja existe):
  gt        mede as anotacoes (contagem e areas) de cada lamina            -> lamina_anotada.csv
  infer     roda os modelos campeoes em cada lamina, para cada variante    -> pred/<variante>/
  report    monta as tabelas (padrao e formato brasileiro)                 -> lamina_inferida/comparacao/resumo
  imagens   imagens comparativas: original | anotacao do especialista | deteccao do modelo (grupo de teste)

Variantes:  A_<n> = modelo treinado com a normalizacao n, imagem normalizada com n;  B_<n> = modelo normal (sem normalizacao)
aplicado em imagem normalizada com n;  A_none = original;  OOF_none = cada lamina e prevista pelo modelo do fold em que a
egua dela foi o TESTE (a unica estimativa honesta para as 180). As laminas do grupo 'treino' foram vistas pelo modelo
(fold principal): o erro nelas e otimista. Parar: criar o arquivo STOP na pasta de saida.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.artigo_iser import fluxo as F  # noqa: E402

FD = Path(r"D:\projeto_placentas_clayton\datasets_v4\test-ZAZA_val-TOSTADA")
BENCH = F.BENCH
OUT = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\dataset_vs_modelo")
UM_PX = 50.0 / 469.0                       # calibracao: barra de 50 um = 469 px no campo nativo
CAP_FACTOR = UM_PX ** 2                    # um2 por px2 nativo (capilar)
MICRO_FACTOR = UM_PX ** 2 * (F.FIELD_W / 640) * (F.FIELD_H / 640)   # um2 por px2 do campo esticado 640x640 (micro)
FIELD_UM2 = F.FIELD_W * F.FIELD_H * CAP_FACTOR
GRUPO = {"train": "treino", "valid": "validacao", "test": "teste"}
PRIORITY = ["A_none", "A_reinhard", "B_reinhard", "OOF_none"]
ALL_VARIANTS = PRIORITY + [v for v in F.VARIANTS if v not in PRIORITY]
TAIL = None


# ------------------------------------------------------------------ laminas
def list_fields(limit: int | None = None) -> list[dict]:
    out = []
    for split in ("train", "valid", "test"):
        for img in sorted((FD / "native" / split / "images").iterdir()):
            short = re.match(r"(.+?_\d+)_jpg", img.name).group(1)           # ex.: ROSILHA-M-B_001
            out.append({"lamina": short, "egua": re.sub(r"-[BDG]_\d+$", "", short), "grupo": GRUPO[split], "split": split,
                        "arquivo": img.name, "stem": img.stem, "path": img,
                        "label": FD / "native" / split / "labels" / (img.stem + ".txt")})
    if TAIL:
        return out[-TAIL:]
    return out[:limit] if limit else out


def input_path(fl: dict, norm: str) -> Path:
    return fl["path"] if norm == "none" else FD / f"native_{norm}" / fl["split"] / "images" / (fl["stem"] + ".png")


def native_ready(norm: str) -> bool:
    d = FD / f"native_{norm}"
    return norm == "none" or (d.exists() and sum(1 for _ in d.glob("*/images/*.png")) >= 180)


def stop_requested() -> bool:
    return (OUT / "STOP").exists()


# ------------------------------------------------------------------ anotacoes
def gt_instances(fl: dict):
    from benchmark.eval.instances import read_gt_polygons
    return (read_gt_polygons(fl["label"], 640, 640, 1),                       # micro: espaco 640x640 (como o benchmark)
            read_gt_polygons(fl["label"], F.FIELD_H, F.FIELD_W, 0))            # capilar: campo nativo


def measure_insts(insts, h, w, factor):
    from benchmark.eval.instances import union_mask
    s = float(sum(i.area for i in insts))
    u = float(union_mask(insts, h, w).sum()) if insts else 0.0
    return len(insts), s * factor, u * factor


def cmd_gt(a) -> None:
    rows = []
    for fl in list_fields(a.limit):
        mi, ci = gt_instances(fl)
        nm, ams, amu = measure_insts(mi, 640, 640, MICRO_FACTOR)
        nc, acs, acu = measure_insts(ci, F.FIELD_H, F.FIELD_W, CAP_FACTOR)
        rows.append({"lamina": fl["lamina"], "egua": fl["egua"], "grupo": fl["grupo"], "arquivo": fl["arquivo"],
                     "campo_area_um2": FIELD_UM2, "micro_n": nm, "micro_area_um2": ams, "micro_area_uniao_um2": amu,
                     "micro_pct_do_campo": 100 * amu / FIELD_UM2, "capilar_n": nc, "capilar_area_um2": acs,
                     "capilar_area_uniao_um2": acu, "capilar_pct_da_area_micro": 100 * acs / ams if ams else np.nan})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "gt.csv", index=False)
    print(f"gt.csv: {len(rows)} laminas", flush=True)


# ------------------------------------------------------------------ modelos
def per_fold_models() -> dict:
    """{egua: (micro_w, micro_conf, cap_w, cap_conf)} dos modelos 'none' de cada fold, pela egua que foi TESTE."""
    df = pd.read_csv(BENCH / "results.csv")
    df["fold"] = df["fold"].fillna("primary")
    out = {}
    for _, r in df[(df.norm == "none")].iterrows():
        if (r.task, r.model) not in (("micro", F.MICRO_MODEL), ("capilar", F.CAP_MODEL)):
            continue
        mare = "ZAZA" if r.fold == "primary" else re.match(r"test-(.+?)_val-", r.fold).group(1)
        w = BENCH / r.run_id / "train" / "run" / "weights" / "best.pt"
        e = out.setdefault(mare, {})
        e[r.task] = (w, float(r.val_conf_star))
    return {m: (v["micro"][0], v["micro"][1], v["capilar"][0], v["capilar"][1]) for m, v in out.items()}


def run_group(variant: str, fields: list[dict], mw, mconf, cw, cconf, input_norm: str, pdir: Path) -> None:
    import torch

    from pipeline.capilar import CapilarModel
    from pipeline.config import PipelineConfig
    from pipeline.micro import MicrocotyledonYOLOModel

    todo = [f for f in fields if not (pdir / f"{f['lamina']}.json").exists()]
    if not todo:
        return
    cfg = PipelineConfig(micro_yolo_ckpt=str(mw), micro_yolo_conf=mconf, capilar_ckpt=str(cw), capilar_conf=cconf)
    micro, cap = MicrocotyledonYOLOModel(cfg), CapilarModel(cfg)
    for i, fl in enumerate(todo, 1):
        if stop_requested():
            print("STOP encontrado: saindo.", flush=True)
            sys.exit(0)
        src = input_path(fl, input_norm)
        bgr = cv2.imread(str(src))
        t0 = time.time()
        rm = micro.predict_bgr(bgr)
        t1 = time.time()
        rc = cap.predict_path(src)
        t2 = time.time()
        mi = F._crops_from_full(rm.masks)
        ci = [rc.masks.crop(k) for k in range(len(rc.masks))]
        np.savez_compressed(pdir / f"{fl['lamina']}.npz",
                            **{f"micro_{k}": x for k, x in F.pack(mi, rm.confidences[: len(mi)]).items()},
                            **{f"cap_{k}": x for k, x in F.pack(ci, rc.confidences[: len(ci)]).items()})
        row = {"lamina": fl["lamina"], "egua": fl["egua"], "grupo": fl["grupo"], "arquivo": fl["arquivo"], "variante": variant,
               "conf_micro": mconf, "conf_capilar": cconf,
               "micro_n": rm.count, "micro_area_um2": rm.area_um2, "micro_area_uniao_um2": rm.area_union_um2,
               "micro_pct_do_campo": 100 * rm.area_union_um2 / FIELD_UM2, "capilar_n": rc.count,
               "capilar_area_um2": rc.area_um2, "capilar_area_uniao_um2": rc.area_union_um2,
               "capilar_pct_da_area_micro": 100 * rc.area_um2 / rm.area_um2 if rm.area_um2 else float("nan"),
               "micro_conf_media": float(np.mean(rm.confidences)) if rm.confidences else float("nan"),
               "capilar_conf_media": float(np.mean(rc.confidences)) if rc.confidences else float("nan"),
               "tempo_s": round(t2 - t0, 2)}
        (pdir / f"{fl['lamina']}.json").write_text(json.dumps(row), encoding="utf-8")
        print(f"[{variant}] {i}/{len(todo)} {fl['lamina']} ({fl['grupo']}): micro {rm.count}, capilar {rc.count}, {t2 - t0:.1f} s", flush=True)
    del micro, cap
    torch.cuda.empty_cache()


def cmd_infer(a) -> None:
    fields = list_fields(a.limit)
    table = F.model_table()
    folds = per_fold_models()
    for v in [x for x in ALL_VARIANTS if not a.variants or x in a.variants]:
        pdir = OUT / "pred" / v
        pdir.mkdir(parents=True, exist_ok=True)
        if v == "OOF_none":
            for mare in sorted({f["egua"] for f in fields}):
                mw, mc, cw, cc = folds[mare]
                print(f"[OOF_none] egua {mare}: modelos do fold em que ela foi o teste", flush=True)
                run_group(v, [f for f in fields if f["egua"] == mare], mw, mc, cw, cc, "none", pdir)
            continue
        mn, imn = F.variant_norms(v)
        if not native_ready(imn):
            print(f"[{v}] pulada: native_{imn} ainda nao esta completo (180 campos)", flush=True)
            continue
        _, mc, mw = table[("micro", mn)]
        _, cc, cw = table[("capilar", mn)]
        run_group(v, fields, mw, mc, cw, cc, imn, pdir)


# ------------------------------------------------------------------ tabelas
KEYS = ["micro_n", "micro_area_um2", "micro_area_uniao_um2", "micro_pct_do_campo", "capilar_n", "capilar_area_um2",
        "capilar_area_uniao_um2", "capilar_pct_da_area_micro"]


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    return 2 * np.cov(x, y, ddof=0)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    df.to_csv(OUT / f"{name}_BR.csv", index=False, encoding="utf-8-sig", sep=";", decimal=",")


def cmd_report(a) -> None:
    gt = pd.read_csv(OUT / "gt.csv")
    save(gt.round(2), "lamina_anotada")
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((OUT / "pred").glob("*/*.json"))]
    pr = pd.DataFrame(rows)
    pr["variante"] = pd.Categorical(pr["variante"], [v for v in ALL_VARIANTS if v in set(pr["variante"])], ordered=True)
    pr = pr.sort_values(["variante", "grupo", "lamina"]).reset_index(drop=True)
    save(pr.round(2), "lamina_inferida")
    m = pr.merge(gt, on=["lamina", "egua", "grupo", "arquivo"], suffixes=("_modelo", "_anotado"))
    comp = m[["lamina", "egua", "grupo", "variante"]].copy()
    for k in KEYS:
        comp[f"{k}_anotado"], comp[f"{k}_modelo"] = m[f"{k}_anotado"], m[f"{k}_modelo"]
        if k.endswith("_n"):
            comp[f"{k}_dif"] = m[f"{k}_modelo"] - m[f"{k}_anotado"]
        elif k.endswith("pct_do_campo") or k.endswith("pct_da_area_micro"):
            comp[f"{k}_dif_pontos"] = m[f"{k}_modelo"] - m[f"{k}_anotado"]
        else:
            comp[f"{k}_dif_pct"] = 100 * (m[f"{k}_modelo"] - m[f"{k}_anotado"]) / m[f"{k}_anotado"]
    save(comp.round(2), "lamina_comparacao")
    res = []
    for (v, g), d in comp.groupby(["variante", "grupo"], observed=True):
        r = {"variante": v, "grupo": g, "laminas": len(d)}
        for tag, n, ar in (("micro", "micro_n", "micro_area_um2"), ("capilar", "capilar_n", "capilar_area_um2")):
            r[f"{tag}_erro_abs_medio_area_pct"] = d[f"{ar}_dif_pct"].abs().mean()
            r[f"{tag}_vies_medio_area_pct"] = d[f"{ar}_dif_pct"].mean()
            r[f"{tag}_erro_abs_medio_contagem"] = d[f"{n}_dif"].abs().mean()
            r[f"{tag}_vies_medio_contagem"] = d[f"{n}_dif"].mean()
            r[f"{tag}_correlacao_area"] = np.corrcoef(d[f"{ar}_anotado"], d[f"{ar}_modelo"])[0, 1] if len(d) > 2 else np.nan
            r[f"{tag}_correlacao_contagem"] = np.corrcoef(d[f"{n}_anotado"], d[f"{n}_modelo"])[0, 1] if len(d) > 2 else np.nan
            r[f"{tag}_ccc_area"] = ccc(d[f"{ar}_anotado"], d[f"{ar}_modelo"])
        r["capilar_pct_da_area_micro_erro_abs_medio_pontos"] = d["capilar_pct_da_area_micro_dif_pontos"].abs().mean()
        res.append(r)
    save(pd.DataFrame(res).round(3), "resumo_por_variante_e_grupo")
    print(f"tabelas em {OUT}: {len(gt)} laminas anotadas, {len(pr)} inferencias, {pr['variante'].nunique()} variantes", flush=True)


# ------------------------------------------------------------------ imagens comparativas
def cmd_imagens(a) -> None:
    ds = 4
    for v in [x for x in (a.variants or ["A_none", "A_reinhard"])]:
        mn, imn = (("none", "none") if v == "OOF_none" else F.variant_norms(v))
        for fl in [f for f in list_fields(a.limit) if a.todos or f["grupo"] == "teste"]:
            dest = OUT / "imagens_comparativas" / v / f"{fl['lamina']}.jpg"
            npz = OUT / "pred" / v / f"{fl['lamina']}.npz"
            if dest.exists() or not npz.exists() or not input_path(fl, imn).exists():
                continue
            img = cv2.imread(str(input_path(fl, imn)))
            base = cv2.resize(img, (F.FIELD_W // ds, F.FIELD_H // ds), interpolation=cv2.INTER_AREA)
            z = np.load(npz)
            gm, gc = gt_instances(fl)
            sxm, sym = F.FIELD_W / 640 / ds, F.FIELD_H / 640 / ds
            g, p = base.copy(), base.copy()
            F._paint(g, gm, sxm, sym, 0, 0, (60, 200, 60), .25)
            F._paint(g, gc, 1 / ds, 1 / ds, 0, 0, (255, 140, 0), .45)
            F._paint(p, F.unpack(z, "micro"), sxm, sym, 0, 0, (60, 200, 60), .25)
            F._paint(p, F.unpack(z, "cap"), 1 / ds, 1 / ds, 0, 0, (255, 140, 0), .45)
            row = json.loads((OUT / "pred" / v / f"{fl['lamina']}.json").read_text(encoding="utf-8"))
            panels = [F._label(base, f"{fl['lamina']} ({fl['grupo']}) | {v}: imagem de entrada"),
                      F._label(g, f"anotado: {len(gm)} micro, {len(gc)} capilares"),
                      F._label(p, f"modelo: {row['micro_n']} micro, {row['capilar_n']} capilares")]
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print("imagem", v, fl["lamina"], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gt", "infer", "report", "imagens", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--todos", action="store_true", help="imagens: todas as 180 laminas (padrao: so o grupo de teste)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ultimas", type=int, default=None, help="so as ultimas N laminas (teste rapido)")
    a = ap.parse_args()
    global OUT, TAIL
    TAIL = a.ultimas
    if a.out:
        OUT = a.out
    OUT.mkdir(parents=True, exist_ok=True)
    steps = {"gt": [cmd_gt], "infer": [cmd_infer], "report": [cmd_report], "imagens": [cmd_imagens],
             "all": [cmd_gt, cmd_infer, cmd_report, cmd_imagens]}[a.cmd]
    for s in steps:
        s(a)


if __name__ == "__main__":
    main()
