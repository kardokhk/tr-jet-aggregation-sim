#!/usr/bin/env python
"""Amendment 3, package A3-2 (view rules): simulation runner.

  python code/20_osrev_views_run.py --list                 # print "task_id family first:last"
  python code/20_osrev_views_run.py --task 17 --out DIR    # run one task, write DIR/parts/task_0017.pkl
  python code/20_osrev_views_run.py --merge --out DIR      # merge parts into four parquet files

Families (code/configs/osrev_views.yaml): repro_e1b and repro_main re-simulate stored cells with
their original seeds; zero_under, decomp and cond are new (entropy 20261005, spawn key
(72, family code, cell)). Library: code/lib/duomaxsim/osrev_views.py.
Outputs after --merge: sim_metrics.parquet (one row per cell, arm, rule, axis, metric),
sim_jack.parquet (leave-one-group-out replicates of bias, SD, RMSE, MAE and q95|e|),
sim_decomp.parquet (decomposition terms), sim_cond.parquet (conditional quantiles).
"""
import argparse
import itertools
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from duomaxsim import config as C  # noqa: E402
from duomaxsim import osrev_views as V  # noqa: E402

CFG_PATH = ROOT / "code" / "configs" / "osrev_views.yaml"
TASK_SIZE = {"repro_e1b": 8, "repro_main": 2, "zero_under": 4, "decomp": 3, "cond": 1}
PKEYS = ["u_anchor", "u_long", "K", "beat_cv", "view_over", "view_under", "window", "S_median_mm",
         "rhythm", "rr_cv_af"]


def load():
    with open(CFG_PATH) as fh:
        oc = yaml.safe_load(fh)
    return oc, C.load_config(ROOT / oc["meta"]["parameters_from"])


def u_scale(a, L):
    return [a, L, L, round(1.2 * L, 10)]


def scenarios(oc):
    return [(a, L) for a in oc["u_anchor"] for L in oc["u_long"]]


def build_cells(oc, cfg):
    """All cells as dicts: family, cell, p (full parameter dict), seed (SeedSequence), arms, cond."""
    cells = []
    m = oc["meta"]
    ent, pre = int(m["master_seed"]), int(m["spawn_prefix"])
    # --- stored cells, original seeds
    for fam in ("repro_e1b", "repro_main"):
        f = oc["families"][fam]
        c0 = C.load_config(ROOT / f["config"])
        grid = C.experiment_cells(c0, f["experiment"])
        for ci, ov in enumerate(grid):
            p = C.cell_params(c0, f["experiment"], ci)
            if "select" in f and any(p[k] != v for k, v in f["select"].items()):
                continue
            if "u_anchor" not in p:           # main run: labels from the scale vector
                p["u_anchor"], p["u_long"] = p["u_scale"][0], p["u_scale"][1]
            cells.append(dict(family=fam, cell=ci, p=p, seed=C.cell_seed(c0, f["experiment"], ci),
                              chunk=int(c0["meta"]["chunk_size"]), arms=("full",), cond=False))
    base = C.base_params(cfg)
    base.update(oc["fixed"])
    chunk = int(m["chunk_size"])
    # --- zero underestimation
    f = oc["families"]["zero_under"]
    v = f["vary"]
    for ci, (vo, K, cv, w) in enumerate(itertools.product(v["view_over"], v["K"], v["beat_cv"], v["window"])):
        p = dict(base, **f["fixed"], view_over=vo, K=K, beat_cv=cv, window=w, u_anchor=0.0, u_long=0.0)
        cells.append(dict(family="zero_under", cell=ci, p=p, chunk=chunk, arms=("full",), cond=False,
                          seed=np.random.SeedSequence(entropy=ent, spawn_key=(pre, f["code"], ci))))
    # --- decomposition
    f = oc["families"]["decomp"]
    ci = 0
    for si, (a, L) in enumerate(scenarios(oc)):
        for ki, K in enumerate(f["K"]):
            for vo in f["view_over"]:
                p = dict(base, **f["fixed"], K=K, view_over=vo, u_scale=u_scale(a, L), u_anchor=a, u_long=L)
                key = (pre, f["code"], 3 * si + ki)
                cells.append(dict(family="decomp", cell=ci, p=p, chunk=chunk, arms=tuple(V.ARMS), cond=False,
                                  seed=np.random.SeedSequence(entropy=ent, spawn_key=key)))
                ci += 1
    # --- conditional quantiles
    f = oc["families"]["cond"]
    for ci, (va, w, rh) in enumerate(itertools.product(f["view_accuracy"], f["window"], f["rhythm"])):
        p = dict(base, **f["fixed"], window=w, **rh, u_scale=u_scale(va["u_anchor"], va["u_long"]), **va)
        cells.append(dict(family="cond", cell=ci, p=p, chunk=chunk, arms=("full",), cond=True,
                          seed=np.random.SeedSequence(entropy=ent, spawn_key=(pre, f["code"], ci))))
    return cells


def build_tasks(cells):
    tasks = []
    for fam, size in TASK_SIZE.items():
        idx = [i for i, c in enumerate(cells) if c["family"] == fam]
        # order decomp tasks so that the two overestimation settings of a seed stay together
        if fam == "decomp":
            size = 6
        for j in range(0, len(idx), size):
            tasks.append((fam, idx[j:j + size]))
    return tasks


def run_cell(c, n_total, n_groups):
    p = c["p"]
    t0 = time.process_time()
    sim = V.simulate_view_rules(p, c["seed"], n_total, c["chunk"], arms=c["arms"])
    meta = dict(family=c["family"], cell=c["cell"], seed_entropy=str(c["seed"].entropy),
                seed_spawn_key=",".join(map(str, c["seed"].spawn_key)), n_patients=n_total)
    for k in PKEYS:
        v = p.get(k, True if k == "view_under" else None)
        meta[k] = np.nan if (k == "window" and v is None) else v
    if p["rhythm"] != "AF":
        meta["rr_cv_af"] = np.nan
    met, jack, dec, cond = [], [], [], []
    S = sim["S"]
    for arm in c["arms"]:
        for d, ax in enumerate(V.AXES):
            for r in V.RULES:
                e = sim["est"][arm][r][:, d] - S[:, d]
                rows, reps = V.error_summary(e, n_groups)
                for row in rows:
                    met.append({**meta, "arm": arm, "rule": r, "axis": ax, **row})
                for nm in V.JACK_KEEP:
                    k = V.STAT_NAMES.index(nm)
                    for g in range(reps.shape[0]):
                        jack.append({**meta, "arm": arm, "rule": r, "axis": ax, "metric": nm, "group": g,
                                     "value": float(reps[g, k])})
                if c["cond"]:
                    for row in V.conditional_summary(e, S[:, d], n_groups=n_groups):
                        cond.append({**meta, "arm": arm, "rule": r, "axis": ax, **row})
            # excess of the largest view mean (and of the reviewed rule) over the anchor-view mean
            for a_ in ("A3", "A4"):
                x = sim["est"][arm][a_][:, d] - sim["est"][arm]["A1"][:, d]
                v_, se = V.paired_mean(x)
                met.append({**meta, "arm": arm, "rule": f"{a_}-A1", "axis": ax, "metric": "excess_mm",
                            "value": v_, "mcse": se, "mcse_method": "analytic", "count": np.nan, "n": x.size})
            for nm, flag in (("p_sel_nonanchor", sim["sel"][arm][:, d] != 0),
                             ("p_sel_true_best", sim["sel_true_best"][arm][:, d]),
                             ("p_sel_overestimated", sim["sel_over"][arm][:, d])):
                k_, n_ = int(flag.sum()), int(flag.size)
                pr = k_ / n_
                met.append({**meta, "arm": arm, "rule": "A3", "axis": ax, "metric": nm, "value": pr,
                            "mcse": float(np.sqrt(pr * (1 - pr) / n_)), "mcse_method": "binomial",
                            "count": k_, "n": n_})
    # library-accumulator values for the exact comparison with stored parquet rows
    for (r, d), acc in sim["acc"].items():
        for nm, (v_, se) in acc.summary().items():
            met.append({**meta, "arm": "full", "rule": r, "axis": V.AXES[d], "metric": "acc_" + nm,
                        "value": float(v_), "mcse": float(se), "mcse_method": "ErrAcc", "count": np.nan,
                        "n": acc.n})
    for d in range(2):
        v_, se, n_ = sim["infl"][d].summary()
        met.append({**meta, "arm": "full", "rule": "A3-A1", "axis": V.AXES[d], "metric": "acc_selection_inflation_mm",
                    "value": float(v_), "mcse": float(se), "mcse_method": "MeanAcc", "count": np.nan, "n": n_})
        v_, se, n_ = sim["lim_anchor"][d].summary()
        met.append({**meta, "arm": "full", "rule": "rule", "axis": V.AXES[d], "metric": "acc_p_limited_anchor",
                    "value": float(v_), "mcse": float(se), "mcse_method": "PropAcc",
                    "count": sim["lim_anchor"][d].k, "n": n_})
    if len(c["arms"]) == len(V.ARMS):
        for d, ax in enumerate(V.AXES):
            for a_ in ("A3", "A4"):
                dec.append({**meta, "axis": ax, "contrast": f"{a_}-A1", **V.decomposition(sim, d, a_, "A1")})
    cpu = time.process_time() - t0
    for rows in (met, jack, dec, cond):
        for r in rows:
            r["cpu_s"] = cpu
    return met, jack, dec, cond


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--n-rep", type=int, default=None, help="pilots only")
    a = ap.parse_args(argv)
    oc, cfg = load()
    cells = build_cells(oc, cfg)
    tasks = build_tasks(cells)
    if a.list:
        for i, (fam, idx) in enumerate(tasks):
            print(i, fam, f"{cells[idx[0]]['cell']}:{cells[idx[-1]]['cell'] + 1}")
        print(f"# {len(tasks)} tasks, {len(cells)} cells", file=sys.stderr)
        return 0
    out = Path(a.out)
    parts = out / "parts"
    if a.merge:
        files = sorted(parts.glob("task_*.pkl"))
        assert len(files) == len(tasks), (len(files), len(tasks))
        tabs = {"metrics": [], "jack": [], "decomp": [], "cond": []}
        for f in files:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            for k in tabs:
                tabs[k] += d[k]
        for k, rows in tabs.items():
            df = pd.DataFrame(rows)
            df.to_parquet(out / f"sim_{k}.parquet", index=False)
            print(k, df.shape)
        n_cells = pd.DataFrame(tabs["metrics"])[["family", "cell"]].drop_duplicates().shape[0]
        assert n_cells == len(cells), (n_cells, len(cells))
        return 0
    parts.mkdir(parents=True, exist_ok=True)
    fam, idx = tasks[a.task]
    n_total = int(a.n_rep or oc["meta"]["n_patients"])
    res = {"metrics": [], "jack": [], "decomp": [], "cond": []}
    t0 = time.perf_counter()
    for i in idx:
        met, jack, dec, cond = run_cell(cells[i], n_total, int(oc["meta"]["jackknife_groups"]))
        res["metrics"] += met
        res["jack"] += jack
        res["decomp"] += dec
        res["cond"] += cond
    tmp = parts / f"task_{a.task:04d}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(res, fh)
    tmp.rename(parts / f"task_{a.task:04d}.pkl")
    print(f"task {a.task} {fam} cells={len(idx)} wall={time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
