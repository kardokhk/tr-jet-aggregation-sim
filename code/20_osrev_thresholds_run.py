#!/usr/bin/env python
"""Protocol amendment 3, package A3-3 (thresholds): generate count tables.

Post hoc reporting detail requested by the senior author (2026-10-05, comment C201).
Reads code/configs/osrev_thresholds.yaml; writes parquet count tables to
results/2026-10-05_osrev/thresholds/raw/.

  python code/20_osrev_thresholds_run.py published              (login node, about 20 CPU-s)
  python code/20_osrev_thresholds_run.py replicates --workers 120   (Slurm: code/slurm/osrev_thresholds.sh)
  python code/20_osrev_thresholds_run.py replicates --n-replicates 2 --n-patients 4000 --out /tmp/x --workers 2   (pilot)

published   Rebuilds the patients of each stored E3 cell with its original seed, asserts
            that the rebuilt summary rows equal the stored parquet rows exactly, and
            tabulates counts (sample = "published_seed", 100,000 patients per scenario).
replicates  New patients: SeedSequence(entropy 20261005, spawn_key (73, 1, replicate)),
            the same key for every scenario (common random numbers across scenarios),
            100 replicates of 100,000 patients per scenario (sample = "amend3").
Within a scenario the four view rules are evaluated on the same patients and reads.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from duomaxsim import config as C  # noqa: E402
from duomaxsim import osrev_thresholds as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "code" / "configs" / "osrev_thresholds.yaml"
OUT_DEFAULT = ROOT / "results" / "2026-10-05_osrev" / "thresholds" / "raw"
TABLES = ("overall", "band", "bins", "hist")
_CFG_CACHE: dict = {}


def load_oc() -> dict:
    with open(CFG_PATH) as fh:
        return yaml.safe_load(fh)


def _cfg(path: str) -> dict:
    if path not in _CFG_CACHE:
        _CFG_CACHE[path] = C.load_config(ROOT / path)
    return _CFG_CACHE[path]


def scenario_params(sc: dict) -> dict:
    src = sc.get("published") or sc["params_from"]
    p = C.cell_params(_cfg(src["config"]), "E3", int(src["cell"]))
    p.update(copy.deepcopy(sc.get("overrides") or {}))
    return p


def tabulate(oc: dict, pat: dict) -> dict:
    a = oc["analysis"]
    return T.count_tables(pat["S"], pat["est"], a["cutoffs_mm"], rules=tuple(a["rules"]), ref=a["comparator"],
                          bands=tuple(a["band_half_widths_mm"]), bin_width=a["distance_bin_mm"],
                          max_dist=a["distance_max_mm"], hist_width=a["hist_bin_mm"], hist_max=a["hist_max_mm"])


def _frames(results, extra_cols) -> dict:
    out = {}
    for t in TABLES:
        rows = []
        for meta, tabs in results:
            rows += [{**meta, **r} for r in tabs[t]]
        out[t] = pd.DataFrame(rows)
    return out


def _write(frames: dict, out: Path, prefix: str, manifest: dict):
    out.mkdir(parents=True, exist_ok=True)
    for t, df in frames.items():
        df.to_parquet(out / f"{prefix}_{t}.parquet", index=False)
    manifest.update(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    host=platform.node(), date=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    command=" ".join(sys.argv), rows={t: len(df) for t, df in frames.items()})
    with open(out / f"{prefix}_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)


# ------------------------------------------------------------------ published seeds

def run_published(oc: dict, out: Path):
    results, checks = [], []
    for sid, sc in oc["scenarios"].items():
        pub = sc.get("published")
        if not pub:
            continue
        cfg = _cfg(pub["config"])
        cell = int(pub["cell"])
        p = scenario_params(sc)
        ss = C.cell_seed(cfg, "E3", cell)
        n = int(cfg["experiments"]["E3"]["n_patients"])
        chunk = int(cfg["meta"]["chunk_size"])
        t0 = time.process_time()
        pat = T.simulate_e3_patients(p, ss, n, chunk)
        cpu = time.process_time() - t0
        # exact agreement with the stored summary rows of this cell
        new = pd.DataFrame(T.e3_rows(pat["S"], pat["est"], p["cutoffs_mm"]))
        files = sorted(glob.glob(str(ROOT / pub["results"] / "E3_cells_*.parquet")))
        old = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        old = old[old.cell == cell].reset_index(drop=True)
        assert len(old) == len(new) == 294, (sid, len(old), len(new))
        for c in ("estimator", "axis", "metric"):
            assert (old[c].to_numpy() == new[c].to_numpy()).all(), (sid, c)
        assert (old["cutoff_mm"].to_numpy() == new["cutoff_mm"].to_numpy()).all()
        ident = all(np.array_equal(old[c].to_numpy(), new[c].to_numpy()) for c in ("value", "mcse", "n"))
        maxdiff = float(np.max(np.abs(old["value"].to_numpy() - new["value"].to_numpy())))
        checks.append(dict(scenario=sid, config=pub["config"], cell=cell,
                           seed_entropy=int(cfg["meta"]["master_seed"]),
                           seed_spawn_key=",".join(map(str, ss.spawn_key)), n_patients=n, rows_compared=len(new),
                           bit_identical=bool(ident), max_abs_diff_value=maxdiff, cpu_s=cpu))
        print(f"{sid}: cell {cell} of {pub['config']}: bit-identical to stored rows = {ident} "
              f"(max |diff| {maxdiff:.3g}), cpu {cpu:.1f} s", flush=True)
        if not ident:
            raise SystemExit(f"published rows of {sid} not reproduced")
        meta = dict(sample="published_seed", scenario=sid, replicate=0, n_patients=n,
                    seed_entropy=int(cfg["meta"]["master_seed"]), seed_spawn_key=",".join(map(str, ss.spawn_key)))
        results.append((meta, tabulate(oc, pat)))
        # per-patient true spans of the base case, for empirical quantiles of the case mix
        if sid == "base":
            q = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
            emp = pd.DataFrame({"quantile": q, "AP_mm": np.quantile(pat["S"][:, 0], q),
                                "SL_mm": np.quantile(pat["S"][:, 1], q)})
            emp.loc[len(emp)] = ["mean", pat["S"][:, 0].mean(), pat["S"][:, 1].mean()]
            emp.loc[len(emp)] = ["sd", pat["S"][:, 0].std(ddof=1), pat["S"][:, 1].std(ddof=1)]
            emp.to_csv(out / "published_base_true_span_empirical.csv", index=False)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(out / "published_reproduction_check.csv", index=False)
    _write(_frames(results, None), out, "published", dict(mode="published", config=oc))


# ------------------------------------------------------------------ replicates

def _task(args):
    oc, sid, p, rep, n, chunk = args
    key = tuple(oc["meta"]["spawn_prefix"]) + (rep,)
    ss = np.random.SeedSequence(entropy=int(oc["meta"]["seed_entropy"]), spawn_key=key)
    t0 = time.process_time()
    pat = T.simulate_e3_patients(p, ss, n, chunk)
    tabs = tabulate(oc, pat)
    meta = dict(sample="amend3", scenario=sid, replicate=rep, n_patients=n,
                seed_entropy=int(oc["meta"]["seed_entropy"]), seed_spawn_key=",".join(map(str, key)))
    return meta, tabs, time.process_time() - t0


def run_replicates(oc: dict, out: Path, workers: int, n_rep: int | None, n_pat: int | None):
    n_rep = int(n_rep or oc["meta"]["n_replicates"])
    n = int(n_pat or oc["meta"]["n_patients"])
    chunk = int(oc["meta"]["chunk_size"])
    tasks = []
    for sid, sc in oc["scenarios"].items():
        if sc.get("replicate_alias"):
            continue
        p = scenario_params(sc)
        tasks += [(oc, sid, p, rep, n, chunk) for rep in range(n_rep)]
    print(f"tasks: {len(tasks)}, workers: {workers}", flush=True)
    w0 = time.perf_counter()
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_task, tasks, chunksize=1)
    cpu = sum(r[2] for r in res)
    wall = time.perf_counter() - w0
    print(f"cpu {cpu:.0f} s over workers, wall {wall:.0f} s", flush=True)
    _write(_frames([(m, t) for m, t, _ in res], None), out, "replicates",
           dict(mode="replicates", config=oc, n_replicates=n_rep, n_patients=n, workers=workers,
                cpu_s_workers=cpu, wall_s=wall))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("published", "replicates"))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-replicates", type=int, default=None, help="pilots only")
    ap.add_argument("--n-patients", type=int, default=None, help="pilots only")
    a = ap.parse_args(argv)
    oc = load_oc()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "published":
        run_published(oc, out)
    else:
        run_replicates(oc, out, a.workers, a.n_replicates, a.n_patients)
    return 0


if __name__ == "__main__":
    sys.exit(main())
