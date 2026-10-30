"""Resumable experiment queue.

    python -m benchmark.runner run    benchmark/configs/smoke.yaml     # run / continue the queue
    python -m benchmark.runner status benchmark/configs/smoke.yaml     # what is done / running / pending
    python -m benchmark.runner table                                   # rebuild results.csv from run.json files

Interrupt freely: Ctrl+C at any moment, or create the file `<results>/STOP` to finish the CURRENT run and exit.
Run again and the queue continues: completed runs (marker DONE) are skipped, a stage already finished is not
repeated, and an interrupted training continues from `last.pt` when the family supports it.
"""

from __future__ import annotations

import argparse
import shutil
import time
import traceback
from pathlib import Path

import yaml

from benchmark import stages
from benchmark.adapters import get_family
from benchmark.core import (RESULTS, RunSpec, expand_matrix, git_info, hardware, is_done, mark, read_json,
                            set_status, versions, write_json)

NOW = lambda: time.strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731


def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def _flatten(prefix: str, d, out: dict) -> None:
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            _flatten(f"{prefix}{k}_", v, out)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[f"{prefix}{k}"] = v


def execute(spec: RunSpec, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = read_json(run_dir / "config.json")
    if existing and existing != spec.config():
        raise RuntimeError(f"{run_dir.name}: configuração diferente da registrada para o mesmo id")
    write_json(run_dir / "config.json", spec.config())
    family = get_family(spec.family)
    st = read_json(run_dir / "status.json", {})
    set_status(run_dir, "running", stage="train", started=st.get("started") or NOW(), error=None)
    t_run = time.time()
    try:
        if not (run_dir / "train.done").exists():
            batch = st.get("batch_used", spec.batch)
            resume = bool(family.supports_resume and family.can_resume(run_dir))
            if resume:
                set_status(run_dir, "running", resumed=True)
            while True:
                try:
                    info = family.train(spec, run_dir, resume=resume, batch=batch)
                    break
                except Exception as e:  # out of memory: retry from scratch with half the batch, and record it
                    if _is_oom(e) and batch > 1:
                        import torch

                        batch = max(1, batch // 2)
                        shutil.rmtree(run_dir / "train", ignore_errors=True)
                        torch.cuda.empty_cache()
                        resume = False
                        set_status(run_dir, "running", batch_used=batch, oom_retries=(st.get("oom_retries") or 0) + 1)
                        continue
                    raise
            set_status(run_dir, "running", batch_used=batch, **info)
            mark(run_dir, "train.done")
        if not (run_dir / "eval_val.done").exists():
            set_status(run_dir, "running", stage="eval_val")
            stages.eval_val(spec, run_dir, family)
            mark(run_dir, "eval_val.done")
        if not (run_dir / "eval_test.done").exists():
            set_status(run_dir, "running", stage="eval_test")
            stages.eval_test(spec, run_dir, family)
            mark(run_dir, "eval_test.done")
        st = read_json(run_dir / "status.json", {})
        write_json(run_dir / "run.json", {
            "run_id": run_dir.name, "config": spec.config(), "tags": spec.tags, "git": git_info(),
            "versions": versions(), "hardware": hardware(), "status": {**st, "state": "done", "stage": "done"},
            "model": family.model_info(run_dir), "training": family.training_log(run_dir),
            "metrics_val": read_json(run_dir / "metrics_val.json"), "metrics_test": read_json(run_dir / "metrics_test.json"),
            "finished": NOW(), "wall_seconds_last_session": round(time.time() - t_run),
        })
        mark(run_dir, "DONE")  # LAST: only now is the run complete
        set_status(run_dir, "done", stage="done")
    except KeyboardInterrupt:
        set_status(run_dir, "interrupted")
        raise
    except Exception:
        set_status(run_dir, "failed", error=traceback.format_exc())
        print(f"  FALHOU: {run_dir.name}\n{traceback.format_exc()}", flush=True)


def load_specs(config_path: Path) -> list[RunSpec]:
    return expand_matrix(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))


def state_of(run_dir: Path) -> str:
    if is_done(run_dir):
        return "done"
    st = read_json(run_dir / "status.json")
    if not st:
        return "pending"
    return st["state"] + (f" ({st.get('stage')})" if st.get("stage") else "")


def cmd_run(specs: list[RunSpec], results: Path, only: str | None, retry_failed: bool) -> None:
    stop = results / "STOP"
    results.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs if not only or only in s.run_id()]
    print(f"{len(todo)} execuções na fila ({sum(is_done(results / s.run_id()) for s in todo)} já concluídas)", flush=True)
    for i, spec in enumerate(todo, 1):
        rd = results / spec.run_id()
        if stop.exists():
            print("arquivo STOP encontrado: encerrando antes da próxima execução", flush=True)
            return
        if is_done(rd):
            continue
        st = read_json(rd / "status.json", {})
        if st.get("state") == "failed" and not retry_failed:
            print(f"[{i}/{len(todo)}] pulando (falhou antes; use --retry-failed): {rd.name}", flush=True)
            continue
        print(f"[{i}/{len(todo)}] {NOW()} {rd.name}", flush=True)
        execute(spec, rd)
        print(f"    -> {state_of(rd)}", flush=True)


def cmd_status(specs: list[RunSpec], results: Path) -> None:
    counts: dict[str, int] = {}
    for s in specs:
        state = state_of(results / s.run_id())
        counts[state.split(" ")[0]] = counts.get(state.split(" ")[0], 0) + 1
        print(f"{state:24s} {s.run_id()}")
    print("\nresumo:", counts)


def cmd_table(results: Path) -> Path:
    import pandas as pd

    rows = []
    for rj in sorted(results.glob("*/run.json")):
        r = read_json(rj)
        row = {"run_id": r["run_id"], **{k: r["config"][k] for k in ("task", "family", "model", "norm", "fold", "seed", "recipe")}}
        _flatten("val_", r["metrics_val"], row)
        _flatten("test_", r["metrics_test"], row)
        _flatten("model_", r["model"], row)
        _flatten("train_", r["training"], row)
        row["vram_peak_mb"] = (r.get("status") or {}).get("vram_peak_mb")
        row["batch_used"] = (r.get("status") or {}).get("batch_used")
        row["resumed"] = (r.get("status") or {}).get("resumed", False)
        rows.append(row)
    for col_drop in ("val_calibration",):
        pass
    df = pd.DataFrame(rows)
    out = results / "results.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} execuções concluídas -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "status"):
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
        p.add_argument("--results", type=Path, default=RESULTS)
        if name == "run":
            p.add_argument("--only", default=None, help="substring do run_id")
            p.add_argument("--retry-failed", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    if args.cmd == "table":
        cmd_table(args.results)
        return
    specs = load_specs(args.config)
    if args.cmd == "status":
        cmd_status(specs, args.results)
    else:
        cmd_run(specs, args.results, args.only, args.retry_failed)
        cmd_table(args.results)


if __name__ == "__main__":
    main()
