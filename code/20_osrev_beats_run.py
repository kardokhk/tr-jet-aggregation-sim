#!/usr/bin/env python
"""Protocol amendment 3, package A3-1 (beat rules): run all conditions and write tidy tables.

Post hoc sensitivity analysis requested by the senior author (2026-10-05). Reads
code/configs/osrev_beats.yaml and code/configs/base.yaml; writes csv tables to
results/2026-10-05_osrev/beats/. One worker process per condition.

  python code/20_osrev_beats_run.py --workers 120                 (Slurm: code/slurm/osrev_beats.sh)
  python code/20_osrev_beats_run.py --n-patients 4000 --out /tmp/x --grids 1,3 --workers 4   (pilot)

Design. Within a condition every arm (plain means, windowed rules at each budget, matched
arms) is computed on the same simulated patients, beats, reader bias and caliper errors
(common random numbers); paired contrasts use paired Monte Carlo standard errors. Conditions
are independently seeded: SeedSequence(entropy 20261005, spawn_key (71, grid, condition,
chunk, stream)), stream 0 = model and measurement, 1 = auxiliary permutation, 2 = bootstrap.
"""
from __future__ import annotations

import argparse
import copy
import glob
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from duomaxsim import config as C  # noqa: E402
from duomaxsim import osrev_beats as OB  # noqa: E402
from duomaxsim.experiments import run_cells  # noqa: E402
from duomaxsim.model import AXES  # noqa: E402
from duomaxsim.rules import beat_rule  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "code" / "configs" / "osrev_beats.yaml"
OUT_DEFAULT = ROOT / "results" / "2026-10-05_osrev" / "beats"
PUBLISHED = ROOT / "results" / "2026-09-18_full"


def gen(entropy, key):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=entropy, spawn_key=tuple(key))))


def load():
    with open(CFG_PATH) as fh:
        oc = yaml.safe_load(fh)
    base = C.load_config(ROOT / oc["meta"]["base_config"])
    return oc, base


def params_for(oc, base, overrides):
    p = C.base_params(base)
    p.update(copy.deepcopy(oc["fixed"]))
    p.update(copy.deepcopy(overrides))
    return p


def chunk_sizes(n_total, chunk):
    return [min(chunk, n_total - i) for i in range(0, n_total, chunk)]


def cond_desc(p):
    return dict(rhythm=p["rhythm"], rr_cv_af=p["rr_cv_af"] if p["rhythm"] == "AF" else np.nan,
                beat_cv=p["beat_cv"], K=int(p["K"]), data_mode=p["data_mode"],
                beat_ar1_rho=float(p.get("beat_ar1_rho", 0.0)), resp_amp=float(p.get("resp_amp", 0.0)),
                resp_mode=p.get("resp_mode", "add") if float(p.get("resp_amp", 0.0)) > 0 else "none",
                cal_noise_mode=p.get("cal_noise_mode", "constant"),
                total_beat_cv=OB.total_beat_cv(p), residual_beat_log_sd=OB.residual_beat_log_sd(p),
                cal_ref_span_mm=(OB.expected_beat_span_mm(p) if p.get("cal_noise_mode", "constant") != "constant"
                                 else np.nan))


# ------------------------------------------------------------------ prospective conditions

def simulate_arms(p, rule, n_total, chunk, entropy, key):
    """Per-patient estimates of every arm on common random numbers.

    Returns est (dict arm -> (n, 2)), truths (dict -> (n, 2)), flags (dict arm ->
    dict of (n, 2) arrays: used, accepted, met_first) and cal_sd summaries.
    """
    W = float(rule["W"])
    est, flags = {}, {}
    T1, Tv, calsd = [], [], []

    def put(d, k, v):
        d.setdefault(k, []).append(v)

    for ci, n in enumerate(chunk_sizes(n_total, chunk)):
        rng = gen(entropy, tuple(key) + (ci, 0))
        aux = gen(entropy, tuple(key) + (ci, 1))
        lat = OB.draw_latent_ext(p, n, rng)
        rb = p["sigma_rb_mm"] * rng.standard_normal(n)
        meas = OB.measure_ext(lat, p, rng, rb)
        x = meas[:, :, 0, :]                    # anchor view, (n, 2, B)
        B = x.shape[-1]
        av = lat.avail[:, None, 0]              # (n, 1)
        T1.append(lat.S)
        Tv.append(lat.g[:, None] * lat.mu[:, :, 0])
        sd = OB.caliper_sd(lat.spans[:, :, 0, :], p)
        calsd.append(np.broadcast_to(np.asarray(sd, dtype=float), x.shape).mean(axis=(0, 2)))
        for k in rule["plain_n"]:
            if k <= B:
                put(est, f"plain_{k}", OB.plain_mean_first(x, k))
        perm = aux.permutation(n)
        for N in rule["N"]:
            for bud in rule["budgets"]:
                if bud < N or bud > B:
                    continue
                br = OB.window_rule_budget(x, av, N, W, bud, "all_acquired")
                a = f"win{N}_b{bud}"
                put(est, a, br.value)
                put(est, f"win{N}fn_b{bud}", np.where(br.limited, OB.plain_mean_first(x, N), br.value))
                fl = flags.setdefault(a, {})
                put(fl, "used", br.used)
                put(fl, "accepted", br.accepted)
                put(fl, "met_first", br.met_first)
                if bud == B:
                    # same acquisition, no selection: mean of all beats that this view acquired
                    put(est, f"sameacq{N}", OB.plain_mean_first(x, br.used))
                    # same distribution of acquired beats, unrelated to this view's values
                    put(est, f"permacq{N}", OB.plain_mean_first(x, br.used[perm]))
    est = {k: np.concatenate(v) for k, v in est.items()}
    flags = {a: {k: np.concatenate(v) for k, v in d.items()} for a, d in flags.items()}
    truths = {"T1": np.concatenate(T1), "Tv": np.concatenate(Tv)}
    return est, truths, flags, np.mean(calsd, axis=0)


def pair_list(rule, arms):
    P = []
    for N in rule["N"]:
        ref = f"plain_{N}"
        for bud in rule["budgets"]:
            w, wf, pb = f"win{N}_b{bud}", f"win{N}fn_b{bud}", f"plain_{bud}"
            if w not in arms:
                continue
            P += [(w, pb, "window vs plain mean of all budget beats"),
                  (w, ref, "window vs plain mean of first N"),
                  (pb, ref, "plain mean of all budget beats vs plain mean of first N"),
                  (wf, pb, "window (first-N fallback) vs plain mean of all budget beats"),
                  (wf, ref, "window (first-N fallback) vs plain mean of first N")]
        top = f"win{N}_b{max(rule['budgets'])}"
        if f"sameacq{N}" in arms:
            P += [(top, f"sameacq{N}", "selection: window vs mean of all beats the same view acquired"),
                  (f"sameacq{N}", ref, "acquisition: mean of all acquired beats vs plain mean of first N"),
                  (top, f"permacq{N}", "window vs plain mean with the same distribution of beat counts"),
                  (f"permacq{N}", ref, "plain mean with the window's beat-count distribution vs first N")]
    seen, out = set(), []
    for a, b, lab in P:
        if a != b and (a, b) not in seen and a in arms and b in arms:
            seen.add((a, b))
            out.append((a, b, lab))
    return out


def run_prospective(task):
    oc, base, grid, cidx, cid, extra, n_total = task
    t0 = time.perf_counter()
    cond = extra["cond"]
    p = params_for(oc, base, extra["overrides"])
    rule = oc["rule"]
    ent = oc["meta"]["seed_entropy"]
    key = (oc["meta"]["spawn_prefix"], grid, cidx)
    est, truths, flags, calsd = simulate_arms(p, rule, n_total, oc["meta"]["chunk_size"], ent, key)
    desc = dict(grid=oc["grids"][grid]["name"], condition=cid, family=cond.get("family", ""),
                seed_entropy=ent, seed_spawn_key=",".join(map(str, key)), **cond_desc(p))
    arms_rows, pair_rows, rule_rows, boot_rows = [], [], [], []
    for d, ax in enumerate(AXES):
        for tn, tv in truths.items():
            for a, v in est.items():
                arms_rows.append(dict(**desc, axis=ax, estimand=tn, arm=a,
                                      **OB.error_summary(v[:, d] - tv[:, d], tv[:, d])))
            for a, b, lab in pair_list(rule, est):
                pair_rows.append(dict(**desc, axis=ax, estimand=tn, arm_a=a, arm_b=b, contrast=lab,
                                      **OB.paired_summary(est[a][:, d] - tv[:, d], est[b][:, d] - tv[:, d],
                                                          tv[:, d])))
        for a, fl in flags.items():
            u = fl["used"][:, d].astype(float)
            acc = fl["accepted"][:, d]
            n = u.size
            N = int(a[3])
            bud = int(a.split("_b")[1])
            rule_rows.append(dict(
                **desc, axis=ax, arm=a, N=N, W=rule["W"], budget=bud, n_views=n,
                n_met_first_N=int(fl["met_first"][:, d].sum()), n_accepted=int(acc.sum()),
                n_limited=int((~acc).sum()),
                sum_beats_acquired=int(u.sum()), mean_beats_acquired=float(u.mean()),
                mcse_beats_acquired=float(u.std(ddof=1) / np.sqrt(n)),
                mean_beats_acquired_accepted=float(u[acc].mean()) if acc.any() else np.nan,
                median_beats_acquired=float(np.median(u)), p90_beats_acquired=float(np.percentile(u, 90)),
                max_beats_acquired=int(u.max()), mean_caliper_sd_mm=float(calsd[d])))
    if cond.get("bootstrap"):
        brng = gen(ent, key + (0, 2))
        for d, ax in enumerate(AXES):
            for tn, tv in truths.items():
                for a, b in (("win3_b30", "plain_3"), ("win5_b30", "plain_5")):
                    ea, eb = est[a][:, d] - tv[:, d], est[b][:, d] - tv[:, d]
                    ps = OB.paired_summary(ea, eb, tv[:, d])
                    bs = OB.bootstrap_rmse_diff(ea, eb, int(oc["meta"]["n_boot"]), brng)
                    boot_rows.append(dict(**desc, axis=ax, estimand=tn, arm_a=a, arm_b=b, n=ps["n"],
                                          d_rmse_mm=ps["d_rmse_mm"], mcse_paired_delta=ps["d_rmse_mcse"],
                                          mcse_indep_formula=ps["d_rmse_mcse_indep"],
                                          corr_sq_err=ps["corr_sq_err"], **bs,
                                          boot_seed_spawn_key=",".join(map(str, key + (0, 2)))))
    return dict(kind="prospective", arms=arms_rows, pairs=pair_rows, rules=rule_rows, boot=boot_rows,
                log=f"{oc['grids'][grid]['name']}/{cid}: {time.perf_counter() - t0:.1f} s")


# ------------------------------------------------------------------ retrospective denominators

def run_retro(task):
    oc, base, grid, cidx, cid, extra, n_total = task
    t0 = time.perf_counter()
    p = params_for(oc, base, extra["overrides"])
    rule = oc["rule"]
    W = float(rule["W"])
    ent = oc["meta"]["seed_entropy"]
    key = (oc["meta"]["spawn_prefix"], grid, cidx)
    desc = dict(grid=oc["grids"][grid]["name"], condition=cid, seed_entropy=ent,
                seed_spawn_key=",".join(map(str, key)), **cond_desc(p))
    store = {}

    def put(k, v):
        store.setdefault(k, []).append(v)

    for ci, n in enumerate(chunk_sizes(n_total, oc["meta"]["chunk_size"])):
        rng = gen(ent, key + (ci, 0))
        lat = OB.draw_latent_ext(p, n, rng)
        rb = p["sigma_rb_mm"] * rng.standard_normal(n)
        meas = OB.measure_ext(lat, p, rng, rb)
        x = meas[:, :, 0, :]
        av = lat.avail[:, None, 0]
        put("avail", lat.avail[:, 0])
        put("T1", lat.S)
        put("Tv", lat.g[:, None] * lat.mu[:, :, 0])
        for N in rule["N"]:
            b0 = beat_rule(x, av, N, None)
            b1 = beat_rule(x, av, N, W)
            put(f"plain_val_{N}", b0.value)
            put(f"plain_lim_{N}", b0.limited)
            put(f"win_val_{N}", b1.value)
            put(f"win_lim_{N}", b1.limited)
            put(f"win_used_{N}", b1.used)
    S = {k: np.concatenate(v) for k, v in store.items()}
    avail = S["avail"]
    n = avail.size
    rows, err_rows = [], []
    for d, ax in enumerate(AXES):
        for N in rule["N"]:
            short = avail < N
            pl = S[f"plain_lim_{N}"][:, d]
            wl = S[f"win_lim_{N}"][:, d]
            rows.append(dict(
                **desc, axis=ax, N=N, W=W, n_views=n,
                mean_available_beats=float(avail.mean()),
                mcse_available_beats=float(avail.std(ddof=1) / np.sqrt(n)),
                n_available_lt_N=int(short.sum()),
                n_limited_no_window=int(pl.sum()),
                n_limited_window=int(wl.sum()),
                n_limited_window_available_lt_N=int((wl & short).sum()),
                n_limited_window_available_ge_N=int((wl & ~short).sum()),
                n_views_available_ge_N=int((~short).sum()),
                mean_beats_measured_window=float(S[f"win_used_{N}"][:, d].mean()),
                corr_available_T1=float(np.corrcoef(avail, S["T1"][:, d])[0, 1])))
            for tn in ("T1", "Tv"):
                tv = S[tn][:, d]
                for arm, val, lim in ((f"plain_{N}", S[f"plain_val_{N}"][:, d], pl),
                                      (f"win{N}", S[f"win_val_{N}"][:, d], wl)):
                    for sub, msk in (("all", np.ones(n, bool)), ("not_limited", ~lim), ("limited", lim)):
                        if msk.sum() > 1:
                            err_rows.append(dict(**desc, axis=ax, estimand=tn, arm=arm, subset=sub,
                                                 **OB.error_summary(val[msk] - tv[msk], tv[msk])))
                err_rows.append(dict(**desc, axis=ax, estimand=tn, arm=f"win{N} minus plain_{N}", subset="all",
                                     **{k: v for k, v in OB.paired_summary(
                                         S[f"win_val_{N}"][:, d] - tv, S[f"plain_val_{N}"][:, d] - tv, tv).items()}))
    dist = pd.Series(avail).value_counts().sort_index()
    dist_rows = [dict(**desc, available_beats=int(k), n_views=int(v), n_total=n) for k, v in dist.items()]
    return dict(kind="retro", retro=rows, retro_err=err_rows, avail=dist_rows,
                log=f"retro/{cid}: {time.perf_counter() - t0:.1f} s")


# ------------------------------------------------------------------ exhaustive-subset check

def run_exhaustive(task):
    oc, base, grid, cidx, cid, extra, n_total = task
    t0 = time.perf_counter()
    p = params_for(oc, base, extra["overrides"])
    rule = oc["rule"]
    W = float(rule["W"])
    bud = int(oc["grids"][grid]["budget"])
    ent = oc["meta"]["seed_entropy"]
    key = (oc["meta"]["spawn_prefix"], grid, cidx)
    acc = {}
    for ci, n in enumerate(chunk_sizes(n_total, oc["meta"]["chunk_size"])):
        rng = gen(ent, key + (ci, 0))
        lat = OB.draw_latent_ext(p, n, rng)
        rb = p["sigma_rb_mm"] * rng.standard_normal(n)
        x = OB.measure_ext(lat, p, rng, rb)[:, :, 0, :bud]
        tv = lat.g[:, None] * lat.mu[:, :, 0]
        for N in rule["N"]:
            a = beat_rule(x, bud, N, W)
            b = OB.beat_rule_exhaustive(x, bud, N, W)
            for d in range(2):
                s = acc.setdefault((N, d), dict(n=0, acc_lib=0, acc_exh=0, acc_differs=0, used_differs=0,
                                                value_differs=0, e_lib=[], e_exh=[], tv=[]))
                s["n"] += n
                s["acc_lib"] += int(a.accepted[:, d].sum())
                s["acc_exh"] += int(b.accepted[:, d].sum())
                s["acc_differs"] += int((a.accepted[:, d] != b.accepted[:, d]).sum())
                s["used_differs"] += int((a.used[:, d] != b.used[:, d]).sum())
                s["value_differs"] += int((np.abs(a.value[:, d] - b.value[:, d]) > 1e-9).sum())
                s["e_lib"].append(a.value[:, d] - tv[:, d])
                s["e_exh"].append(b.value[:, d] - tv[:, d])
                s["tv"].append(tv[:, d])
    rows = []
    for (N, d), s in acc.items():
        ea, eb, tv = (np.concatenate(s[k]) for k in ("e_lib", "e_exh", "tv"))
        ps = OB.paired_summary(ea, eb, tv)
        rows.append(dict(grid="check_exhaustive", condition=cid, seed_entropy=ent,
                         seed_spawn_key=",".join(map(str, key)), axis=AXES[d], N=N, W=W, budget=bud,
                         n_views=s["n"], n_accepted_contiguous=s["acc_lib"], n_accepted_exhaustive=s["acc_exh"],
                         n_accept_decision_differs=s["acc_differs"], n_beats_acquired_differs=s["used_differs"],
                         n_value_differs=s["value_differs"], estimand="Tv",
                         d_rmse_mm_contiguous_minus_exhaustive=ps["d_rmse_mm"], d_rmse_mcse=ps["d_rmse_mcse"],
                         d_bias_mm_contiguous_minus_exhaustive=ps["d_bias_mm"], d_bias_mcse=ps["d_bias_mcse"]))
    return dict(kind="exhaustive", exhaustive=rows, log=f"exhaustive/{cid}: {time.perf_counter() - t0:.1f} s")


# ------------------------------------------------------------------ reproduction of published cells

def repro_cells(oc, base):
    cells = C.experiment_cells(base, "E2")
    want = []
    r = oc["reproduce"]
    for rs, rov in r["rhythm_states"].items():
        for cv in r["beat_cv"]:
            for w in r["window"]:
                for N in r["N_beats"]:
                    for dm in r["data_mode"]:
                        target = dict(rov, beat_cv=cv, window=w, N_beats=N, data_mode=dm)
                        hit = [i for i, c in enumerate(cells) if c == target]
                        assert len(hit) == 1, (target, hit)
                        want.append((hit[0], rs))
    return want


def run_repro(task):
    oc, base, cell, rs, n_total = task
    t0 = time.perf_counter()
    new = run_cells(base, "E2", [cell], n_rep=n_total)
    return dict(kind="repro", cell=cell, rhythm_state=rs, df=new,
                log=f"repro/E2 cell {cell}: {time.perf_counter() - t0:.1f} s")


def compare_repro(results, n_total, base):
    files = sorted(glob.glob(str(PUBLISHED / "E2_cells_*.parquet")))
    pub = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    keys = ["cell", "estimator", "subset", "axis", "estimand", "metric"]
    rows = []
    for res in results:
        new = res["df"]
        old = pub[pub["cell"] == res["cell"]]
        m = new.merge(old, on=keys, how="outer", suffixes=("_rerun", "_published"), indicator=True)
        full = n_total == int(base["experiments"]["E2"]["n_patients"])
        for _, r in m.iterrows():
            rows.append(dict(
                cell=res["cell"], rhythm_state=res["rhythm_state"],
                beat_cv=r.get("p_beat_cv_rerun"), window=r.get("p_window_rerun"),
                N_beats=r.get("p_N_beats_rerun"), data_mode=r.get("p_data_mode_rerun"),
                estimator=r["estimator"], subset=r["subset"], axis=r["axis"], estimand=r["estimand"],
                metric=r["metric"], value_published=r.get("value_published"), value_rerun=r.get("value_rerun"),
                mcse_published=r.get("mcse_published"), n_published=r.get("n_published"),
                n_rerun=r.get("n_rerun"), seed_entropy=r.get("seed_entropy_rerun"),
                seed_spawn_key=r.get("seed_spawn_key_rerun"), merge=r["_merge"],
                identical=bool(full and r["_merge"] == "both"
                               and (r["value_rerun"] == r["value_published"]
                                    or (pd.isna(r["value_rerun"]) and pd.isna(r["value_published"]))))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main

def dispatch(task):
    kind = task[0]
    return {"prospective": run_prospective, "retro": run_retro, "exhaustive": run_exhaustive,
            "repro": run_repro}[kind](task[1])


def build_tasks(oc, base, n_total, grids):
    tasks = []
    for g, spec in oc["grids"].items():
        if grids and g not in grids:
            continue
        name = spec["name"]
        if name == "sens":
            i = 0
            for rs, rov in spec["rhythms"].items():
                for cond in spec["conditions"]:
                    ov = dict(beat_cv=0.15, **rov, **cond["overrides"])
                    tasks.append(("prospective", (oc, base, g, i, f"{rs}_{cond['id']}",
                                                  dict(cond=cond, overrides=ov), n_total)))
                    i += 1
        else:
            kind = {"budget": "prospective", "check_K3": "prospective", "retro": "retro",
                    "check_exhaustive": "exhaustive"}[name]
            for i, cond in enumerate(spec["conditions"]):
                tasks.append((kind, (oc, base, g, i, cond["id"], dict(cond=cond, overrides=cond["overrides"]),
                                     n_total)))
    if not grids or 0 in grids:
        for cell, rs in repro_cells(oc, base):
            tasks.append(("repro", (oc, base, cell, rs, n_total)))
    return tasks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-patients", type=int, default=None, help="pilot override")
    ap.add_argument("--grids", default="", help="comma-separated grid numbers (0 = reproduction); default all")
    a = ap.parse_args(argv)
    oc, base = load()
    n_total = int(a.n_patients or oc["meta"]["n_patients"])
    grids = [int(g) for g in a.grids.split(",") if g != ""]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(oc, base, n_total, grids)
    print(f"tasks: {len(tasks)}; workers: {a.workers}; n per condition: {n_total}", flush=True)
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(a.workers) as pool:
        results = []
        for r in pool.imap(dispatch, tasks, chunksize=1):
            print(r["log"], flush=True)
            results.append(r)
    tables = {"arms": [], "pairs": [], "rules": [], "boot": [], "retro": [], "retro_err": [], "avail": [],
              "exhaustive": []}
    repro = []
    for r in results:
        if r["kind"] == "repro":
            repro.append(r)
            continue
        for k in tables:
            tables[k] += r.get(k, [])
    names = {"arms": "beats_arm_errors.csv", "pairs": "beats_paired_contrasts.csv",
             "rules": "beats_window_rule_counts.csv", "boot": "beats_rmse_difference_intervals.csv",
             "retro": "beats_retro_limited_denominators.csv", "retro_err": "beats_retro_errors.csv",
             "avail": "beats_retro_available_distribution.csv", "exhaustive": "beats_check_exhaustive_subsets.csv"}
    for k, rows in tables.items():
        if rows:
            pd.DataFrame(rows).to_csv(out / names[k], index=False)
            print(f"wrote {names[k]} rows={len(rows)}")
    if repro:
        df = compare_repro(repro, n_total, base)
        df.to_csv(out / "beats_reproduction_E2_cells.csv", index=False)
        print(f"wrote beats_reproduction_E2_cells.csv rows={len(df)} identical={int(df['identical'].sum())}")
    print(f"total wall {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
