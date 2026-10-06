#!/usr/bin/env python
"""Software-verification reruns for the senior-author review (comment C191; amendment 3, A3-1c).

Post hoc. Nothing in code/lib/duomaxsim/, the existing scripts, configs or results is modified;
library functions are imported read-only. Three parts, all run in one process pool:

  e1     Rerun of tests/independent/indep_check_e1.py (independent reimplementation of the E1
         core, provisional parameters, its own seeds SeedSequence(987654321).spawn(12)) and of
         the library on the same six conditions of the frozen provisional configuration
         tests/fixtures/config_v01.yaml (cells 0, 409, 754, 842, 953, 1077; library seeds
         SeedSequence(20260918, spawn_key=(1, cell))). 240 comparisons; z = difference /
         sqrt(MCSE_lib^2 + MCSE_indep^2).
  amend2 Rerun of tests/independent/indep_check_amend2.py (its own seeds: default_rng(1) for
         E5b, default_rng(2) for E6b, default_rng(3) for the SciPy check) against the stored
         results/2026-09-18_amend2/ parquet files.
  crn    Window (+-15%) versus no window, anchor-view mean (A1), base case (sinus, beat CV 15%,
         three beats, prospective, AP axis), RMSE against T1 and Tv:
         (i) exact regeneration of the stored cells E2/147 (window) and E2/123 (no window) from
             their stored seeds, per patient, giving the independent-cells interval;
         (ii) a paired no-window arm computed on the patients and beats of cell 147 (stored seed);
         (iii) 100 new replicates of 100,000 patients, SeedSequence(20261005, spawn_key=(191, r)),
             each with both arms on the same patients and beats (common random numbers), to
             compare the analytic paired and independent MCSE with the empirical SD.

MCSE of an RMSE: delta method, sd(e^2) / (2 RMSE sqrt(n)).
MCSE of a paired RMSE difference: sd(psi) / sqrt(n), psi_i = e_w,i^2 / (2 RMSE_w) - e_0,i^2 / (2 RMSE_0).
MCSE of an independent RMSE difference: sqrt(MCSE_w^2 + MCSE_0^2).

Run from the project root (see code/slurm/osrev_verification.sh):
  /project/home/p201509/envs/duomax-sim/bin/python code/osrev_verification.py --workers 120
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from duomaxsim import config as C  # noqa: E402
from duomaxsim.experiments import run_cells  # noqa: E402
from duomaxsim.model import draw_latent, measure  # noqa: E402
from duomaxsim.rules import beat_rule  # noqa: E402

OUT = ROOT / "results" / "2026-10-05_osrev" / "verification"
ENTROPY = 20261005
PREFIX = 191
E1_SETTINGS = [(0, 1, 0.05, False, 0.0, 1), (409, 2, 0.15, True, 0.0, 3), (754, 3, 0.20, True, 0.3, 10),
               (842, 3, 0.30, True, 0.0, 5), (953, 4, 0.10, True, 0.6, 13), (1077, 4, 0.25, False, 0.9, 7)]
E1_SDFR = [(0.8, 0.1), (1.0, 0.0), (0.0, 0.3)]
N_E1 = 100_000
CELL_W, CELL_0 = 147, 123
N_REP = 100


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / "independent" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ E1
def e1_indep(task):
    i, j = task
    mod = _load("indep_check_e1")
    seeds = mod.rng_master.spawn(len(E1_SETTINGS) * 2)          # as in the script's __main__
    cell, K, cv, over, rho, N = E1_SETTINGS[i]
    axis = ["AP", "SL"][j]
    t0 = time.process_time()
    rows = [dict(cell=cell, estimator=r[0], axis=r[1], estimand=r[2], bias=r[3], se_bias=r[4],
                 rmse=r[5], se_rmse=r[6])
            for r in mod.simulate(N_E1, K, cv, over, rho, N, axis, seeds[2 * i + j], E1_SDFR)]
    return "e1_indep", rows, time.process_time() - t0


def e1_lib(cell):
    cfg = C.load_config(ROOT / "tests" / "fixtures" / "config_v01.yaml")
    t0 = time.process_time()
    df = run_cells(cfg, "E1", [cell], n_rep=N_E1)
    df = df[df.metric.isin(["bias_mm", "rmse_mm"]) & df.estimator.isin(["A1", "A3", "A4"])]
    keep = df.estimator.isin(["A1", "A3"])
    for s, f in E1_SDFR:
        keep |= (df.estimator == "A4") & (df.s_det == s) & (df.f_rej == f)
    df = df[keep].copy()
    df["est"] = np.where(df.estimator == "A4", "A4_" + df.s_det.astype(str) + "_" + df.f_rej.astype(str),
                         df.estimator)
    return "e1_lib", df[["cell", "est", "axis", "estimand", "metric", "value", "mcse", "n",
                         "seed_entropy", "seed_spawn_key"]].to_dict("records"), time.process_time() - t0


# ------------------------------------------------------------------ amendment 2
def am_e5(cell):
    import os
    os.chdir(ROOT)
    mod = _load("indep_check_amend2")
    t0 = time.process_time()
    return "am_e5", mod.e5(cell), time.process_time() - t0


def am_e6(cell):
    import os
    os.chdir(ROOT)
    mod = _load("indep_check_amend2")
    t0 = time.process_time()
    return "am_e6", dict(cell=cell, res=mod.e6(cell)), time.process_time() - t0


def am_scipy(_):
    import os
    os.chdir(ROOT)
    mod = _load("indep_check_amend2")
    return "am_scipy", mod.scipy_agreement(), 0.0


# ------------------------------------------------------------------ CRN
def _arms(ss, cell, n_total, both):
    """Errors of the anchor-view mean (AP) against T1 and Tv, following run_E2's draw order.

    Returns dict arm -> (e_T1, e_Tv). With both=True the no-window rule is also applied to the
    same measured beats (first N of the acquired sequence): common random numbers.
    """
    cfg = C.load_config(ROOT / "code" / "configs" / "base.yaml")
    p = C.cell_params(cfg, "E2", cell)
    chunk = int(cfg["meta"]["chunk_size"])
    kids = ss.spawn(-(-n_total // chunk))
    out = {"own": [[], []], "nowin_paired": [[], []]}
    for i, ch in enumerate(kids):
        n = min(chunk, n_total - i * chunk)
        rng = np.random.Generator(np.random.PCG64(ch))
        lat = draw_latent(p, n, rng)
        rb = p["sigma_rb_mm"] * rng.standard_normal(n)
        meas = measure(lat, p, rng, rb)
        T1 = lat.S[:, 0]
        Tv = lat.g * lat.mu[:, 0, 0]
        br = beat_rule(meas, lat.avail[:, None, :], p["N_beats"], p["window"], p["limited_value"])
        v = br.value[:, 0, 0]
        out["own"][0].append(v - T1)
        out["own"][1].append(v - Tv)
        if both:
            b0 = beat_rule(meas, lat.avail[:, None, :], p["N_beats"], None, p["limited_value"])
            v0 = b0.value[:, 0, 0]
            out["nowin_paired"][0].append(v0 - T1)
            out["nowin_paired"][1].append(v0 - Tv)
    return {k: tuple(np.concatenate(a) for a in v) for k, v in out.items() if v[0]}


def _rmse(e):
    n = e.size
    m2 = np.mean(e * e)
    r = np.sqrt(m2)
    return r, np.sqrt(max(np.mean(e ** 4) - m2 * m2, 0.0) / n) / (2 * r)


def _pair(ew, e0):
    rw, sw = _rmse(ew)
    r0, s0 = _rmse(e0)
    psi = ew * ew / (2 * rw) - e0 * e0 / (2 * r0)
    return dict(rmse_win=rw, mcse_win=sw, rmse_nowin=r0, mcse_nowin=s0, diff=rw - r0,
                mcse_diff_paired=float(np.std(psi, ddof=1) / np.sqrt(psi.size)),
                mcse_diff_independent_formula=float(np.hypot(sw, s0)),
                corr_sq_err=float(np.corrcoef(ew * ew, e0 * e0)[0, 1]), n=int(ew.size))


def crn_stored(_):
    cfg = C.load_config(ROOT / "code" / "configs" / "base.yaml")
    t0 = time.process_time()
    w = _arms(C.cell_seed(cfg, "E2", CELL_W), CELL_W, 100_000, both=True)
    z = _arms(C.cell_seed(cfg, "E2", CELL_0), CELL_0, 100_000, both=False)
    rows = []
    for k, tn in enumerate(("T1", "Tv")):
        rw, sw = _rmse(w["own"][k])
        r0, s0 = _rmse(z["own"][k])
        rows.append(dict(design="independent cells (stored seeds 2,147 and 2,123)", estimand=tn,
                         rmse_win=rw, mcse_win=sw, rmse_nowin=r0, mcse_nowin=s0, diff=rw - r0,
                         mcse_diff=float(np.hypot(sw, s0)), n=100_000))
        q = _pair(w["own"][k], w["nowin_paired"][k])
        rows.append(dict(design="paired on the patients and beats of cell 147 (stored seed 2,147)",
                         estimand=tn, rmse_win=q["rmse_win"], mcse_win=q["mcse_win"],
                         rmse_nowin=q["rmse_nowin"], mcse_nowin=q["mcse_nowin"], diff=q["diff"],
                         mcse_diff=q["mcse_diff_paired"], n=q["n"],
                         mcse_diff_if_treated_independent=q["mcse_diff_independent_formula"],
                         corr_sq_err=q["corr_sq_err"]))
    return "crn_stored", rows, time.process_time() - t0


def crn_rep(r):
    ss = np.random.SeedSequence(entropy=ENTROPY, spawn_key=(PREFIX, r))
    t0 = time.process_time()
    a = _arms(ss, CELL_W, 100_000, both=True)
    rows = []
    for k, tn in enumerate(("T1", "Tv")):
        rows.append(dict(rep=r, estimand=tn, seed_entropy=ENTROPY, seed_spawn_key=f"{PREFIX},{r}",
                         **_pair(a["own"][k], a["nowin_paired"][k])))
    return "crn_rep", rows, time.process_time() - t0


def _run(task):
    fn, arg = task
    return globals()[fn](arg)


# ------------------------------------------------------------------ assembly
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=120)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [("e1_indep", (i, j)) for i in range(6) for j in range(2)]
    tasks += [("e1_lib", s[0]) for s in E1_SETTINGS]
    tasks += [("am_e5", c) for c in (9, 10, 11, 12, 13, 14)] + [("am_e6", c) for c in (22, 0)]
    tasks += [("am_scipy", 0), ("crn_stored", 0)] + [("crn_rep", r) for r in range(N_REP)]
    w0 = time.perf_counter()
    with Pool(a.workers) as pool:
        res = pool.map(_run, tasks, chunksize=1)
    by = {}
    cpu = {}
    for kind, payload, c in res:
        by.setdefault(kind, []).append(payload)
        cpu[kind] = cpu.get(kind, 0.0) + c

    # ---- E1: 240 comparisons
    ind = pd.DataFrame([r for rows in by["e1_indep"] for r in rows])
    ind = ind.melt(id_vars=["cell", "estimator", "axis", "estimand"], value_vars=["bias", "rmse"],
                   var_name="metric", value_name="indep")
    se = pd.DataFrame([r for rows in by["e1_indep"] for r in rows]).melt(
        id_vars=["cell", "estimator", "axis", "estimand"], value_vars=["se_bias", "se_rmse"],
        var_name="metric", value_name="indep_mcse")
    se["metric"] = se["metric"].str.replace("se_", "")
    ind = ind.merge(se, on=["cell", "estimator", "axis", "estimand", "metric"])
    lib = pd.DataFrame([r for rows in by["e1_lib"] for r in rows]).rename(
        columns={"est": "estimator", "value": "lib", "mcse": "lib_mcse"})
    lib["metric"] = lib["metric"].str.replace("_mm", "")
    e1 = ind.merge(lib, on=["cell", "estimator", "axis", "estimand", "metric"], how="outer", indicator=True)
    assert (e1["_merge"] == "both").all() and len(e1) == 240, (len(e1), e1["_merge"].value_counts())
    e1 = e1.drop(columns="_merge")
    e1["diff"] = e1["indep"] - e1["lib"]
    e1["mcse_diff"] = np.hypot(e1["indep_mcse"], e1["lib_mcse"])
    e1["z"] = e1["diff"] / e1["mcse_diff"]
    e1 = e1.sort_values(["cell", "axis", "estimand", "metric", "estimator"]).reset_index(drop=True)
    e1.to_csv(OUT / "verif_e1_indep_vs_library.csv", index=False, float_format="%.8g")

    # ---- amendment 2
    st5 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "results/2026-09-18_amend2/E5_*.parquet")))])
    st6 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "results/2026-09-18_amend2/E6_*.parquet")))])
    rows = []

    def one(df, **sel):
        t = df
        for k, v in sel.items():
            t = t[t[k] == v]
        assert len(t) == 1, (sel, len(t))
        return float(t["value"].iloc[0]), float(t["mcse"].iloc[0])

    for o in by["am_e5"]:
        c, lam = o["cell"], o["lambda"]
        s = st5[st5.cell == c]
        q = [("apparent MAE vs single read, independent AI", o["mae_indep"],
              one(s, ai="independent", sigma_ai_mm=2.0, ref="single", metric="apparent_mae")),
             ("apparent MAE vs single read, shared AI", o["mae_shared"],
              one(s, ai="shared", sigma_ai_mm=2.0, ref="single", metric="apparent_mae")),
             ("MAE of R_img vs T1", o["mae_rimg_vs_T1"], one(s, ref="R_img", metric="ref_mae_vs_T1"))]
        if lam > 0:
            q += [("paired MAE difference, shared minus independent", o["paired_diff"],
                   one(s, ai="shared", sigma_ai_mm=2.0, ref="single", metric="paired_mae_diff_mean")),
                  ("proportion of studies with lower MAE, shared AI", o["p_lower"],
                   one(s, ai="shared", sigma_ai_mm=2.0, ref="single", metric="p_lower_mae_than_independent"))]
        for name, (v, sv), (lv, ls) in q:
            rows.append(dict(part="E5b", cell=c, quantity=name, indep=v, indep_mcse=sv, lib=lv, lib_mcse=ls,
                             lam=lam))
    for o in by["am_e6"]:
        c = o["cell"]
        s = st6[(st6.cell == c) & (st6.design == "sentinel")]
        for key, (p2, p1) in o["res"].items():
            ns, al = key.split("_a")
            ns, al = int(ns[2:]), float(al)
            for lab, v, m in (("two-sided", p2, "power_reject_bf"), ("one-sided", p1, "power_reject_bf_onesided")):
                lv, ls = one(s, sentinel_n=ns, anchor_alpha=al, metric=m)
                rows.append(dict(part="E6b", cell=c, quantity=f"Brown-Forsythe rejection rate, {lab}, n_s={ns}, alpha={al}",
                                 indep=float(v), indep_mcse=float(np.sqrt(v * (1 - v) / 1000)), lib=lv, lib_mcse=ls))
    am = pd.DataFrame(rows)
    am["diff"] = am["indep"] - am["lib"]
    am["mcse_diff"] = np.hypot(am["indep_mcse"], am["lib_mcse"])
    am["z"] = np.where(am["mcse_diff"] > 0, am["diff"] / am["mcse_diff"], np.nan)
    am.to_csv(OUT / "verif_amend2_indep_vs_stored.csv", index=False, float_format="%.8g")

    # ---- CRN
    st = pd.DataFrame(by["crn_stored"][0])
    st.to_csv(OUT / "verif_window_rmse_stored_seeds.csv", index=False, float_format="%.10g")
    rep = pd.DataFrame([r for rows in by["crn_rep"] for r in rows]).sort_values(["estimand", "rep"])
    rep.to_csv(OUT / "verif_window_rmse_crn_replicates.csv", index=False, float_format="%.10g")
    summ = []
    for tn, g in rep.groupby("estimand"):
        g = g.sort_values("rep")
        d = g["diff"].to_numpy()
        # independent design: window arm of replicate r minus no-window arm of replicate r+1 (disjoint seeds)
        di = g["rmse_win"].to_numpy() - np.roll(g["rmse_nowin"].to_numpy(), -1)
        R = len(d)
        summ.append(dict(estimand=tn, replicates=R, n_per_replicate=100_000,
                         mean_paired_diff=d.mean(), se_of_mean_paired_diff=d.std(ddof=1) / np.sqrt(R),
                         empirical_sd_paired_diff=d.std(ddof=1),
                         mean_analytic_mcse_paired=g["mcse_diff_paired"].mean(),
                         empirical_sd_independent_diff=di.std(ddof=1),
                         mean_analytic_mcse_independent=g["mcse_diff_independent_formula"].mean(),
                         rel_se_of_empirical_sd=1 / np.sqrt(2 * (R - 1)),
                         coverage_paired_95=float(np.mean(np.abs(d - d.mean()) <= 1.96 * g["mcse_diff_paired"])),
                         coverage_independent_formula_on_paired=float(
                             np.mean(np.abs(d - d.mean()) <= 1.96 * g["mcse_diff_independent_formula"])),
                         mean_corr_sq_err=g["corr_sq_err"].mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "verif_window_rmse_crn_summary.csv", index=False, float_format="%.10g")

    meta = dict(date=time.strftime("%Y-%m-%dT%H:%M:%S%z"), wall_s=time.perf_counter() - w0, cpu_s_by_part=cpu,
                workers=a.workers, seed_entropy=ENTROPY, seed_spawn_key_prefix=PREFIX, n_rep=N_REP,
                scipy_rel_err=by["am_scipy"][0], numpy=np.__version__, pandas=pd.__version__,
                python=sys.version.split()[0])
    with open(OUT / "verif_manifest.json", "w") as fh:
        json.dump(meta, fh, indent=1)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("E1: n comparisons", len(e1), "max |z|", e1.z.abs().max(), "n |z|>2", int((e1.z.abs() > 2).sum()),
          "n |z|>3", int((e1.z.abs() > 3).sum()))
    print(e1.reindex(e1.z.abs().sort_values(ascending=False).index).head(12).to_string())
    print(e1.groupby("cell").z.agg(lambda z: z.abs().max()).to_string())
    for part, g in am.groupby("part"):
        print(part, "n", len(g), "max |z|", g.z.abs().max(), "n |z|>2", int((g.z.abs() > 2).sum()),
              "n |z|>3", int((g.z.abs() > 3).sum()))
    print(am[am.part == "E5b"].to_string())
    print(am.reindex(am.z.abs().sort_values(ascending=False).index).head(8).to_string())
    print(st.to_string())
    print(summ.T.to_string())
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
