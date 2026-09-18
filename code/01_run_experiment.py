#!/usr/bin/env python
"""Run grid cells of one pre-specified experiment and write one parquet file.

Example:
  python code/01_run_experiment.py --experiment E1 --config code/configs/base.yaml \
      --cells 0:10 --out results/2026-09-18_pilot
Use --list-cells to print the grid. One parquet file per invocation (chunk of
cells), never one per cell (inode quota).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

import duomaxsim  # noqa: E402
from duomaxsim import config as C  # noqa: E402
from duomaxsim.experiments import run_cells  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=C.EXPERIMENTS)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cells", default="all", help="i:j (half-open), i, or all")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--n-rep", type=int, default=None,
                    help="override replicates per cell (patients for E1-E4, studies for E5-E6); pilots only")
    ap.add_argument("--list-cells", action="store_true")
    a = ap.parse_args(argv)

    cfg = C.load_config(a.config)
    cells = C.experiment_cells(cfg, a.experiment)
    if a.list_cells:
        for i, c in enumerate(cells):
            print(i, json.dumps(c))
        print(f"# {len(cells)} cells")
        return 0
    if not a.out:
        ap.error("--out is required unless --list-cells")
    sel = C.parse_cells(a.cells, len(cells))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.experiment}_cells_{sel.start:05d}-{sel.stop:05d}"
    if a.n_rep:
        stem += f"_nrep{a.n_rep}"
    t0 = time.perf_counter()
    df = run_cells(cfg, a.experiment, sel, n_rep=a.n_rep)
    df.to_parquet(out / f"{stem}.parquet", index=False)
    manifest = {
        "command": " ".join(sys.argv),
        "experiment": a.experiment,
        "cells": [sel.start, sel.stop],
        "n_rep_override": a.n_rep,
        "config_path": a.config,
        "config": cfg,
        "duomaxsim": duomaxsim.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_s_total": time.perf_counter() - t0,
        "rows": len(df),
    }
    with open(out / f"{stem}.manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    print(f"wrote {out / (stem + '.parquet')} rows={len(df)} wall={manifest['wall_s_total']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
