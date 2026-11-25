"""Tempo de inferencia por imagem e por classe, para a entrega ao laboratorio.

Mede, para cada imagem, o pipeline Reinhard A (o mesmo modelo que gerou as medidas das entregas):
  tempo_normalizacao_s : normalizacao de cor Reinhard (referencia ajustada so com as eguas de treino do fold principal)
  tempo_esticar_s      : campo nativo -> 640x640 (so os microcotiledones usam)
  tempo_micro_s        : rede + pos-processamento dos microcotiledones (o mesmo que o artigo mede; sem o esticar)
  tempo_capilar_s      : divisao em janelas + rede + fusao dos capilares, no campo nativo
Protocolo identico ao de benchmark/timing_inference.py: confianca calibrada na validacao, FP32, GPU sincronizada antes e
depois de cada chamada, aquecimento descartado (5 imagens nos microcotiledones, 1 campo nos capilares), leitura do arquivo
fora do relogio. Rode com a GPU livre de outros programas.

    python scripts/artigo_iser/tempos_por_imagem.py run       # mede (retomavel: grava uma linha por imagem em tempos.jsonl)
    python scripts/artigo_iser/tempos_por_imagem.py export    # junta com as medidas das entregas e grava os 2 CSV
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\ENTREGA_TEMPOS")
JSONL = OUT / "tempos.jsonl"
ENTREGA_ANTIGA = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\ENTREGA_REINHARD")
ENTREGA_150 = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\ENTREGA_REINHARD_150")
DIR_150 = Path(r"D:\projeto_placentas_clayton\imagens_teste_artigo_iser_v2")
TEMPOS = ["tempo_normalizacao_s", "tempo_esticar_s", "tempo_micro_s", "tempo_capilar_s"]


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn):
    _sync()
    t = time.perf_counter()
    r = fn()
    _sync()
    return r, time.perf_counter() - t


def _imagens() -> list[tuple[str, str, Path]]:
    """(conjunto, chave, caminho): 150 imagens de teste e 180 campos anotados."""
    from scripts.artigo_iser import dataset_vs_modelo as V

    novas = sorted(Path(p) for p in glob.glob(str(DIR_150 / "**" / "*.jp*g"), recursive=True))
    out = [("150", p.stem, p) for p in novas]
    out += [("180", fl["lamina"], fl["path"]) for fl in V.list_fields(None)]
    return out


def cmd_run() -> None:
    import torch

    from benchmark.eval import sliced
    from benchmark.timing_inference import HalfPredictor
    from scripts.artigo_iser import fluxo as F

    if not torch.cuda.is_available():
        raise SystemExit("sem GPU")
    OUT.mkdir(parents=True, exist_ok=True)
    feitos = set()
    if JSONL.exists():
        for ln in JSONL.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            feitos.add((r["conjunto"], r["chave"]))
    itens = _imagens()
    todo = [x for x in itens if (x[0], x[1]) not in feitos]
    print(f"{len(itens)} imagens, {len(feitos)} ja medidas, {len(todo)} a medir", flush=True)
    if not todo:
        return

    tabela = F.model_table()
    _, mconf, mw = tabela[("micro", "reinhard")]
    _, cconf, cw = tabela[("capilar", "reinhard")]
    print("modelos Reinhard A:", mw.parent.parent.parent.name, "| conf", mconf, "/", cw.parent.parent.parent.name, "| conf", cconf, flush=True)
    norm = F.get_normalizer("reinhard")
    pm, pc = HalfPredictor(mw, imgsz=640, conf=mconf), HalfPredictor(cw, imgsz=640, conf=cconf)
    pm.half = pc.half = False

    def normalizar(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out, _ = norm.transform(rgb)
        return cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_RGB2BGR)

    def esticar(bgr):
        return cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR)

    cap = lambda bgr: sliced.merge(sliced.collect(lambda t: pc.predict(t, cls_id=0), bgr), cconf)  # noqa: E731

    # aquecimento (descartado): 5 imagens nos microcotiledones, 1 campo nos capilares, 1 normalizacao
    for _, _, p in todo[:5]:
        b = normalizar(cv2.imread(str(p)))
        pm.predict(esticar(b), cls_id=None)
    cap(normalizar(cv2.imread(str(todo[0][2]))))
    torch.cuda.reset_peak_memory_stats()
    print("aquecimento ok; medindo", flush=True)

    t_ini = time.time()
    with JSONL.open("a", encoding="utf-8") as f:
        for i, (conj, chave, p) in enumerate(todo, 1):
            bgr = cv2.imread(str(p))                       # leitura do arquivo fora do relogio
            nb, t_norm = _timed(lambda: normalizar(bgr))
            st, t_est = _timed(lambda: esticar(nb))
            rm, t_mic = _timed(lambda: pm.predict(st, cls_id=None))
            rc, t_cap = _timed(lambda: cap(nb))
            row = {"conjunto": conj, "chave": chave, "tempo_normalizacao_s": t_norm, "tempo_esticar_s": t_est,
                   "tempo_micro_s": t_mic, "tempo_capilar_s": t_cap, "n_micro": len(rm), "n_capilar": len(rc)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                dec = time.time() - t_ini
                print(f"{i}/{len(todo)} {chave}: norm {t_norm:.2f} s, micro {t_mic * 1000:.0f} ms, capilar {t_cap:.1f} s "
                      f"(decorridos {dec / 60:.0f} min)", flush=True)
    print("VRAM pico (MB):", round(torch.cuda.max_memory_allocated() / 2 ** 20), flush=True)


JSONL_MICRO = OUT / "tempos_micro.jsonl"
RUN2_NORM = Path(r"D:\projeto_placentas_clayton\temp_ml\artigo_iser\run2\norm\reinhard")
FD = Path(r"D:\projeto_placentas_clayton\datasets_v4\test-ZAZA_val-TOSTADA")


def cmd_micro() -> None:
    """Segunda passada so dos microcotiledons, no protocolo do artigo: imagens 640x640 ja na memoria e chamadas em sequencia
    (a GPU fica em clock alto). Na passada 'run' o microcotiledone vem logo depois dos ~5 s de normalizacao na CPU, com a GPU
    ociosa, e o tempo medido fica ~4x maior que o do artigo (0,27 s contra ~0,06 s)."""
    import torch

    from benchmark.timing_inference import HalfPredictor
    from scripts.artigo_iser import dataset_vs_modelo as V
    from scripts.artigo_iser import fluxo as F

    tab = F.model_table()
    _, mconf, mw = tab[("micro", "reinhard")]
    pr = HalfPredictor(mw, imgsz=640, conf=mconf)
    pr.half = False
    itens = [("150", p.stem, RUN2_NORM / (p.stem + ".png")) for _, _, p in [x for x in _imagens() if x[0] == "150"]]
    itens += [("180", fl["lamina"], FD / "native_reinhard" / fl["split"] / "images" / (fl["stem"] + ".png")) for fl in V.list_fields(None)]
    print(len(itens), "imagens normalizadas; carregando e esticando para 640x640", flush=True)
    lote = []
    for conj, chave, p in itens:
        b = cv2.imread(str(p))
        assert b is not None, p
        lote.append((conj, chave, cv2.resize(b, (640, 640), interpolation=cv2.INTER_LINEAR)))
    for _, _, b in lote[:5]:           # aquecimento descartado
        pr.predict(b, cls_id=None)
    with JSONL_MICRO.open("w", encoding="utf-8") as f:
        for conj, chave, b in lote:
            _sync()
            t = time.perf_counter()
            r = pr.predict(b, cls_id=None)
            _sync()
            f.write(json.dumps({"conjunto": conj, "chave": chave, "tempo_micro_s": time.perf_counter() - t, "n_micro": len(r)}) + "\n")
    print("microcotiledons medidos:", len(lote), flush=True)


def _ler_br(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep=";", decimal=",", encoding="utf-8-sig")


def cmd_export() -> None:
    rows = pd.DataFrame([json.loads(ln) for ln in JSONL.read_text(encoding="utf-8").splitlines()])
    rows = rows.drop_duplicates(["conjunto", "chave"], keep="last")
    if JSONL_MICRO.exists():      # tempo dos microcotiledons no protocolo do artigo (segunda passada)
        mi = pd.DataFrame([json.loads(ln) for ln in JSONL_MICRO.read_text(encoding="utf-8").splitlines()])
        rows = rows.drop(columns=["tempo_micro_s"]).merge(mi[["conjunto", "chave", "tempo_micro_s"]], on=["conjunto", "chave"], how="left")
        assert rows["tempo_micro_s"].notna().all()
    rows["tempo_total_s"] = rows[TEMPOS].sum(axis=1)
    cols_t = TEMPOS + ["tempo_total_s"]
    jobs = [("180", ENTREGA_ANTIGA / "1_comparacao_por_lamina.csv", "lamina", "tempos_180_anotadas.csv", "n_micro_medidas", "n_capilar_medidas"),
            ("150", ENTREGA_150 / "medidas_por_imagem.csv", "imagem", "tempos_150_teste.csv", None, None)]
    for conj, base_csv, chave, nome, _, _ in jobs:
        base = _ler_br(base_csv)
        t = rows[rows.conjunto == conj].rename(columns={"chave": chave})[[chave] + cols_t + ["n_micro", "n_capilar"]]
        m = base.merge(t, on=chave, how="left", validate="one_to_one")
        falta = m[cols_t[0]].isna().sum()
        # conferencia: as contagens da execucao de medida batem com as medidas ja entregues?
        cm = "micro_contagem_inferida" if conj == "180" else "micro_contagem"
        cc = "capilar_contagem_inferida" if conj == "180" else "capilar_contagem"
        print(f"[{conj}] {len(m)} linhas, {falta} sem tempo | contagens iguais: micro {(m[cm] == m['n_micro']).mean():.1%}, "
              f"capilar {(m[cc] == m['n_capilar']).mean():.1%}")
        m = m.drop(columns=["n_micro", "n_capilar"])
        for c in cols_t:
            m[c] = m[c].round(3)
        OUT.mkdir(parents=True, exist_ok=True)
        m.to_csv(OUT / nome, sep=";", decimal=",", index=False, encoding="utf-8-sig")
        d = m[cols_t]
        print(d.agg(["mean", "std", "median", "min", "max"]).round(3).to_string())
    print("gravado em", OUT)


if __name__ == "__main__":
    {"run": cmd_run, "micro": cmd_micro, "export": cmd_export}[sys.argv[1]]()
