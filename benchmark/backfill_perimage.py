"""Writes `perimage_test.json` for runs finished before per-image rows were stored (evaluation only, no training).

    python -m benchmark.backfill_perimage                      # every DONE run that lacks the file
    python -m benchmark.backfill_perimage --only micro__      # substring of the run id

The stored `metrics_test.json` is never modified. As an integrity check the new pass must reproduce its main numbers;
a run that does not is reported and its rows are NOT written.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmark import stages
from benchmark.adapters import get_family
from benchmark.core import RESULTS, RunSpec, is_done, read_json, write_json


def _same(a: dict, b: dict) -> bool:
    keys = [("report", "instance", "tp"), ("report", "instance", "fp"), ("report", "instance", "fn")]
    if "field" in b:
        keys += [("field", "instance", "tp"), ("field", "instance", "fp"), ("field", "instance", "fn")]
    def get(d, path):
        for k in path:
            d = d[k]
        return d
    return all(get(a, k) == get(b, k) for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    todo = [d for d in sorted(a.results.iterdir()) if d.is_dir() and is_done(d) and not (d / "perimage_test.json").exists()
            and (not a.only or a.only in d.name)]
    print(f"{len(todo)} execuções sem linhas por imagem", flush=True)
    for i, rd in enumerate(todo, 1):
        t = time.time()
        spec = RunSpec(**read_json(rd / "config.json"))
        res, rows = stages.test_pass(spec, rd, get_family(spec.family))
        if _same(read_json(rd / "metrics_test.json"), res):
            write_json(rd / "perimage_test.json", rows)
            status = "ok"
        else:
            status = "DIVERGE do metrics_test.json (nada gravado)"
        print(f"[{i}/{len(todo)}] {rd.name}: {status} ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
