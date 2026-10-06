#!/usr/bin/env python
"""Protocol amendment 3, package A3-5 (AI validation and reference sets): simulations.

Post hoc sensitivity analyses requested by the senior author (2026-10-05; comments C189, C209,
C321, C399). Reads code/configs/osrev_ai_reference.yaml; writes raw parquet tables to
results/2026-10-05_osrev/ai_reference/raw/ (analysed by code/21_osrev_ai_reference_analyse.py).

  python code/20_osrev_ai_reference_run.py --workers 120          (Slurm: code/slurm/osrev_ai_reference.sh)
  python code/20_osrev_ai_reference_run.py --pilot --out <dir> --workers 4

Parts (each a set of independent tasks, one worker process per task):
  repro     stored seeds; recomputes published values with the extension module (options off)
  e5        AI-validation studies: component moments, analytic check, matched model pairs
  nested    reader pairs reading repeated case samples (conditional versus marginal precision)
  multi     panels of six readers (designs with 2, 3, 4 or 6 readers at 400 reads)
  sentinel  size and power of the F, Brown-Forsythe, Welch and permutation tests, 54 cells

Common random numbers: within an e5 cell every AI model and reference design is evaluated on the
same patients, reads and noise arrays; within a sentinel cell every test, sentinel size and drift
level uses the same production sets and permutations; within the multi part every design uses the
same panels and cases. Cells, scenarios and blocks are independently seeded.
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
from scipy import stats  # noqa: E402

from duomaxsim import config as C  # noqa: E402
from duomaxsim import osrev_ai_reference as OA  # noqa: E402
from duomaxsim.metrics import bland_altman  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "code" / "configs" / "osrev_ai_reference.yaml"
OUT_DEFAULT = ROOT / "results" / "2026-10-05_osrev" / "ai_reference"


def load():
    with open(CFG_PATH) as fh:
        oc = yaml.safe_load(fh)
    return oc


def seedseq(oc, *key):
    return np.random.SeedSequence(entropy=oc["meta"]["seed_entropy"],
                                  spawn_key=(oc["meta"]["spawn_prefix"],) + tuple(int(k) for k in key))


def gen(oc, *key):
    return np.random.Generator(np.random.PCG64(seedseq(oc, *key)))


def find_cell(cfg, exp, match):
    cells = C.experiment_cells(cfg, exp)
    hit = [i for i, c in enumerate(cells) if all(c.get(k) == v for k, v in match.items())
           and (exp != "E5" or c.get("ai_shared_lambda", 0.0) == 0.0)]
    if len(hit) != 1:
        raise ValueError(f"{exp} {match}: {len(hit)} cells")
    return hit[0]


def seed_label(oc, *key):
    return f"SeedSequence({oc['meta']['seed_entropy']}, spawn_key=({oc['meta']['spawn_prefix']}, " \
           + ", ".join(str(int(k)) for k in key) + "))"


# ====================================================================== repro

def task_repro(a):
    oc, spec, pilot = a
    cfg = C.load_config(ROOT / oc["meta"][spec["config"]])
    exp = spec["experiment"]
    cell = spec["cell"] if "cell" in spec else find_cell(cfg, exp, spec["match"])
    p = C.cell_params(cfg, exp, cell)
    ss = C.cell_seed(cfg, exp, cell)
    chunk = int(cfg["meta"]["chunk_size"])
    n_studies = int(cfg["experiments"][exp]["n_studies"])
    grids = {k: v.get("grid", [v["value"]]) for k, v in cfg["parameters"].items()}
    files = sorted(glob.glob(str(ROOT / spec["results"] / f"{exp}_cells_*.parquet")))
    st = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    st = st[st.cell == cell]
    assert len(st) and int(st.n_studies.iloc[0]) == n_studies
    rows = []
    base = dict(id=spec["id"], experiment=exp, cell=cell, results=spec["results"],
                seed=f"SeedSequence({cfg['meta']['master_seed']}, spawn_key=({C.EXPERIMENTS.index(exp) + 1}, {cell}))",
                n_studies=n_studies)
    if exp == "E5":
        lam = p.get("ai_shared_lambda")
        lam = None if lam is None else float(lam)
        d = OA.e5_draws(p, ss, n_studies, chunk)
        m = OA.e5_library_metrics(d, p, grids["sigma_ai_mm"], grids["adj_tol_mm"], lam=lam)
        for key, val in m.items():
            if key[0] == "ref":
                r = st[st.ai.isna() & (st.ref == key[1]) & (st.metric == key[2])] if "ai" in st else \
                    st[(st.ref == key[1]) & (st.metric == key[2])]
                name, v = f"ref|{key[1]}|{key[2]}", val[0]
            elif key[0] == "paired":
                r = st[(st.ai == key[1]) & (st.sigma_ai_mm == key[2]) & (st.ref == key[3])
                       & (st.metric == "p_lower_mae_than_independent")]
                name, v = f"paired|{key[1]}|{key[2]:g}|{key[3]}|p_lower_mae_than_independent", val
                if lam is None:
                    continue
            else:
                r = st[(st.ai == key[0]) & (st.sigma_ai_mm == key[1]) & (st.ref == key[2]) & (st.metric == key[3])]
                name, v = f"{key[0]}|{key[1]:g}|{key[2]}|{key[3]}", val[0]
            if len(r) != 1:
                rows.append(dict(**base, quantity=name, stored=np.nan, recomputed=v, note="no unique stored row"))
            else:
                rows.append(dict(**base, quantity=name, stored=float(r.value.iloc[0]), recomputed=float(v), note=""))
        # component variances behind the published 3.07 and 3.88 mm^2 (pooled, ddof 0, as script 10)
        e_img = d["R_img"] - d["T1"]
        e_ref = d["r1"] - d["T1"]
        for nm, x in (("var_e_img_pooled", e_img), ("var_e_ref_single_pooled", e_ref)):
            rows.append(dict(**base, quantity=f"component|{nm}", stored=np.nan, recomputed=float(np.ravel(x).var()),
                             note="compare with results/2026-09-18_amend2/analysis/E5bE6b_decomp_reference_error.csv"))
    else:
        d = OA.e6_draws(p, ss, n_studies, chunk)
        m = OA.e6_library_metrics(d, p)
        for key, val in m.items():
            if key[0] == "double":
                r = st[(st.n_double == key[1]) & (st.metric == key[2])]
                name = f"double|{key[1]}|{key[2]}"
                r = r.iloc[:1] if r.value.nunique() == 1 else r
            else:
                if "power_" + key[3] not in set(st.metric):
                    continue
                r = st[(st.sentinel_n == key[1]) & (st.anchor_alpha == key[2]) & (st.metric == "power_" + key[3])]
                name = f"sentinel|{key[1]}|{key[2]:g}|{key[3]}"
            if len(r) != 1:
                rows.append(dict(**base, quantity=name, stored=np.nan, recomputed=float(val), note="no unique stored row"))
            else:
                rows.append(dict(**base, quantity=name, stored=float(r.value.iloc[0]), recomputed=float(val), note=""))
    df = pd.DataFrame(rows)
    df["abs_diff"] = (df.stored - df.recomputed).abs()
    return {"repro": df}


# ====================================================================== e5

def _moments(x):
    x = np.ravel(x)
    m = x.mean()
    c = x - m
    v = np.mean(c * c)
    return dict(mean=float(m), var=float(v), skew=float(np.mean(c ** 3) / v ** 1.5) if v > 0 else np.nan,
                exkurt=float(np.mean(c ** 4) / v ** 2 - 3.0) if v > 0 else np.nan)


def _cov(x, y):
    x, y = np.ravel(x), np.ravel(y)
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def _jack(fn, arrays, G):
    """Delete-one-block jackknife SE (blocks of whole studies) of the dict returned by fn."""
    S = arrays[0].shape[0]
    idx = np.array_split(np.arange(S), G)
    reps = []
    for g in range(G):
        keep = np.concatenate([idx[h] for h in range(G) if h != g])
        reps.append(fn(*[x[keep] for x in arrays]))
    return {k: float(np.sqrt((G - 1) / G * np.sum((np.array([r[k] for r in reps]) - np.mean([r[k] for r in reps])) ** 2)))
            for k in reps[0]}


def _component_stats(e_img, e_rd, b, e_ref):
    w = e_rd - b
    mi, mr, mf = _moments(e_img), _moments(e_rd), _moments(e_ref)
    return dict(bias_img=mi["mean"], var_img=mi["var"], skew_img=mi["skew"], exkurt_img=mi["exkurt"],
                bias_read=mr["mean"], var_read=mr["var"],
                var_reader_offset=_moments(b)["var"], var_read_noise=_moments(w)["var"],
                two_cov_offset_noise=2 * _cov(b, w), two_cov_img_read=2 * _cov(e_img, e_rd),
                bias_ref=mf["mean"], var_ref=mf["var"], skew_ref=mf["skew"], exkurt_ref=mf["exkurt"],
                cov_img_ref=_cov(e_img, e_ref), ratio_var_img_ref=mi["var"] / mf["var"],
                corr_img_ref=_cov(e_img, e_ref) / np.sqrt(mi["var"] * mf["var"]),
                within_study_var_ref=float(e_ref.var(axis=1, ddof=1).mean()),
                between_study_var_ref_mean=float(e_ref.mean(axis=1).var(ddof=1)))


def task_e5(a):
    oc, ci, pilot = a
    ec = oc["e5"]
    spec = ec["cells"][ci]
    cfg = C.load_config(ROOT / oc["meta"]["base_config"])
    lib_cell = find_cell(cfg, "E5", spec["overrides"])
    p = C.cell_params(cfg, "E5", lib_cell)
    chunk = int(oc["meta"]["chunk_size"])
    S = 60 if pilot else int(ec["n_studies"])
    Sc = 60 if pilot else int(ec["n_matching_studies"])
    G = 6 if pilot else int(ec["jackknife_blocks"])
    d = OA.e5_draws(p, seedseq(oc, oc["streams"]["e5_validation"], ci), S, chunk, extra_noise=True)
    cal = OA.e5_draws(p, seedseq(oc, oc["streams"]["e5_matching"], ci), Sc, chunk)
    T1, z, z2 = d["T1"], d["z"], d["z2"]
    e_img = d["R_img"] - T1
    inh = d["a_int"] + (d["b_slope"] - 1.0) * T1
    sig0 = float(ec["sigma_reference_mm"])
    target = sig0 * np.sqrt(2.0 / np.pi)
    key = dict(cell_id=spec["id"], p_beat_cv=p["beat_cv"], p_view_over=bool(p["view_over"]), n_studies=S,
               n_per_study=int(p["n_val"]), seed=seed_label(oc, oc["streams"]["e5_validation"], ci))

    # ---- own-error SDs matched on true MAE (separate calibration sample)
    cal_img = cal["R_img"] - cal["T1"]
    cal_inh = d["a_int"] + (d["b_slope"] - 1.0) * cal["T1"]      # the evaluation run's calibration line
    fam = {f"share_lam{lam:g}": (float(lam), lam * e_img, lam * cal_img) for lam in ec["lambdas"]}
    fam["inherit"] = (np.nan, inh, cal_inh)
    models = [dict(model=f"indep_sd{s:g}", family="independent", lam=0.0, match="none", noise="z",
                   sigma_own=float(s), sys=np.zeros_like(T1), zz=z) for s in ec["sigma_independent_mm"]]
    match_rows = []
    for fname, (lam, sys_, sys_cal) in fam.items():
        try:
            s_m = OA.match_sigma_for_mae(sys_cal, cal["z"], target)
        except ValueError:
            # the systematic part alone already has a larger MAE than the comparator: no match exists
            s_m = np.nan
        match_rows.append(dict(**key, family=fname, lam=lam, target_true_mae=target, sigma_own_matched=s_m,
                               matchable=bool(np.isfinite(s_m)),
                               mae_systematic_alone_calib=float(np.abs(sys_cal).mean()),
                               n_calibration_patients=int(cal["z"].size),
                               seed_matching=seed_label(oc, oc["streams"]["e5_matching"], ci)))
        for match, s in (("eqsd", sig0), ("eqmae", s_m)):
            if not np.isfinite(s):
                continue
            for noise, zz in (("ind", z2), ("crn", z)):
                models.append(dict(model=f"{fname}_{match}_{noise}", family=fname, lam=lam, match=match,
                                   noise=noise, sigma_own=float(s), sys=sys_, zz=zz))

    refs = {"true": T1}
    refs.update(OA.reference_designs(d["r1"], d["r2"], d["r3"], ec["adj_tol_mm"], p["adjudication_value"]))
    c_between = float(p["sigma_rb_mm"]) ** 2 / 4.0

    study_rows, ana_rows = [], []
    for mdl in models:
        s = mdl["sigma_own"]
        ai = T1 + mdl["sys"] + s * mdl["zz"]
        ms = _moments(mdl["sys"])
        info = {k: mdl[k] for k in ("model", "family", "lam", "match", "noise", "sigma_own")}
        for rn, ref in refs.items():
            q = OA.study_agreement(ai, ref)
            study_rows.append(pd.DataFrame(dict(cell_id=spec["id"], model=mdl["model"], ref=rn, study=np.arange(S),
                                                mae=q["mae"], bias=q["bias"], sd=q["sd"], mse=q["mse"])))
            e_ref = ref - T1
            mr = _moments(e_ref)
            cov = _cov(mdl["sys"], e_ref)
            var_ai = ms["var"] + s * s
            pr = OA.predicted_apparent(var_ai, mr["var"], cov, ms["mean"], mr["mean"])
            # per-study prediction with the noise cross terms at their expectation
            sysd = mdl["sys"] - e_ref
            pred_s = (sysd * sysd).mean(-1) + s * s
            dm = q["mse"] - pred_s
            dd = ai - ref
            md = _moments(dd)
            cb = 0.0 if rn == "true" else c_between
            var_within_pred = pr["var"] - cb
            n = S
            ana_rows.append(dict(
                **key, **info, ref=rn,
                var_model_error=var_ai, bias_model_error=ms["mean"], var_ref_error=mr["var"], bias_ref_error=mr["mean"],
                cov_model_ref=cov, rho_model_ref=(cov / np.sqrt(var_ai * mr["var"]) if mr["var"] > 0 else np.nan),
                var_diff_pred=float(pr["var"]), bias_diff_pred=float(pr["bias"]),
                mse_pred=float(pr["mse"]), mse_sim=float(q["mse"].mean()), mse_sim_mcse=float(q["mse"].std(ddof=1) / np.sqrt(n)),
                mse_sim_minus_pred=float(dm.mean()), mse_sim_minus_pred_mcse=float(dm.std(ddof=1) / np.sqrt(n)),
                bias_sim=float(q["bias"].mean()), bias_sim_mcse=float(q["bias"].std(ddof=1) / np.sqrt(n)),
                mae_normal_pred=float(pr["mae_normal"]), mae_sim=float(q["mae"].mean()),
                mae_sim_mcse=float(q["mae"].std(ddof=1) / np.sqrt(n)),
                loa_width_pred_total=float(pr["loa_width"]),
                between_study_cov_assumed=cb,
                loa_width_pred_within=float(2 * OA.Z975 * np.sqrt(var_within_pred)),
                var_within_sim=float((q["sd"] ** 2).mean()), var_within_sim_mcse=float((q["sd"] ** 2).std(ddof=1) / np.sqrt(n)),
                var_within_pred=float(var_within_pred),
                loa_width_sim=float((2 * OA.Z975 * q["sd"]).mean()),
                loa_width_sim_mcse=float((2 * OA.Z975 * q["sd"]).std(ddof=1) / np.sqrt(n)),
                skew_diff=md["skew"], exkurt_diff=md["exkurt"]))

    # ---- components of the reference error
    comp_rows = []
    rd = {1: d["r1"] - d["R_img"], 2: d["r2"] - d["R_img"]}
    comp_sets = {"single": (rd[1], d["b1"]), "mean2": (0.5 * (rd[1] + rd[2]), 0.5 * (d["b1"] + d["b2"]))}
    for rn, ref in refs.items():
        if rn == "true":
            continue
        e_ref = ref - T1
        if rn in comp_sets:
            e_rd, bb = comp_sets[rn]
        else:
            e_rd = ref - d["R_img"]
            dis = np.abs(d["r1"] - d["r2"]) > float(rn.replace("adj_tol", ""))
            bb = np.where(dis, d["b3"], 0.5 * (d["b1"] + d["b2"]))
        full = _component_stats(e_img, e_rd, bb, e_ref)
        se = _jack(_component_stats, (e_img, e_rd, bb, e_ref), G)
        row = dict(**key, ref=rn, n_patients=int(T1.size), jackknife_blocks=G)
        for k_, v_ in full.items():
            row[k_] = v_
            row["mcse_" + k_] = se[k_]
        if rn.startswith("adj"):
            row["p_adjudicated"] = float(dis.mean())
        comp_rows.append(row)
    cal_row = dict(**key, inherited_intercept_mm=d["a_int"], inherited_slope=d["b_slope"], calib_n=int(p["calib_n"]),
                   sigma_rb_mm=float(p["sigma_rb_mm"]), sigma_cal_mm=float(p["sigma_cal_mm"]))
    mods = pd.DataFrame([{k: m[k] for k in ("model", "family", "lam", "match", "noise", "sigma_own")} for m in models])
    mods.insert(0, "cell_id", spec["id"])
    return {"e5_study": pd.concat(study_rows, ignore_index=True), "e5_analytic": pd.DataFrame(ana_rows),
            "e5_components": pd.DataFrame(comp_rows), "e5_matching": pd.DataFrame(match_rows),
            "e5_models": mods, "e5_calibration": pd.DataFrame([cal_row])}


# ====================================================================== nested reader pairs

def _e6_base_params(oc, sigma_rb):
    cfg = C.load_config(ROOT / oc["meta"]["base_config"])
    cell = find_cell(cfg, "E6", {"beat_cv": 0.15, "sigma_rb_mm": 0.75, "ai_draft_sigma_mm": 2.0})
    p = C.cell_params(cfg, "E6", cell)
    p["sigma_rb_mm"] = float(sigma_rb)
    return p


def task_nested(a):
    oc, si, block, pilot = a
    nc = oc["loa_nested"]
    sc = nc["scenarios"][si]
    p = _e6_base_params(oc, sc["sigma_rb_mm"])
    P = int(nc["pairs_per_block"])
    M = 6 if pilot else int(nc["n_rep"])
    n = int(nc["n_cases"])
    reads, b = OA.reader_panel_reads(p, gen(oc, oc["streams"]["loa_nested"], si, block), P, M, n, 2)
    dd = reads[..., 0] - reads[..., 1]                       # (P, M, n)
    rows = []
    pair = block * P + np.arange(P)
    for nn in nc["sizes"]:
        bias, sd, lo, hi, hw = bland_altman(dd[..., :nn])
        rows.append(pd.DataFrame(dict(scenario=sc["id"], pair=np.repeat(pair, M), rep=np.tile(np.arange(M), P),
                                      n_cases=nn, bias=bias.ravel(), sd=sd.ravel(), loa_hi=hi.ravel(),
                                      ci_halfwidth=hw.ravel())))
    pairs = pd.DataFrame(dict(scenario=sc["id"], pair=pair, block=block, b1=b[:, 0], b2=b[:, 1],
                              seed=seed_label(oc, oc["streams"]["loa_nested"], si, block)))
    return {"nested_reps": pd.concat(rows, ignore_index=True), "nested_pairs": pairs}


# ====================================================================== multi-reader designs

def task_multi(a):
    oc, block, pilot = a
    mc = oc["loa_multireader"]
    p = _e6_base_params(oc, mc["sigma_rb_mm"])
    S = int(mc["studies_per_block"])
    Rmax = int(mc["n_readers_max"])
    n = int(mc["n_cases_drawn"])
    tot = int(mc["total_reads"])
    reads, b = OA.reader_panel_reads(p, gen(oc, oc["streams"]["loa_multireader"], block), S, 1, n, Rmax)
    reads = reads[:, 0]                                       # (S, n, Rmax)
    study = block * S + np.arange(S)
    rows = []
    for R in mc["readers"]:
        for design, mask in (("crossed", OA.crossed_allocation(R, tot)), ("rotating", OA.rotating_allocation(R, tot))):
            if R == 2 and design == "rotating":
                continue                                      # identical to crossed
            nc_ = mask.shape[0]
            q = OA.between_reader_mom(reads[:, :nc_, :R], mask)
            rows.append(pd.DataFrame(dict(study=study, design=design, n_readers=R, n_cases=nc_,
                                          total_reads=int(mask.sum()), n_reader_pairs=q["n_pairs"],
                                          cases_per_pair_min=min(int((mask[:, j] & mask[:, k]).sum())
                                                                 for j in range(R) for k in range(j + 1, R)),
                                          cases_per_pair_max=max(int((mask[:, j] & mask[:, k]).sum())
                                                                 for j in range(R) for k in range(j + 1, R)),
                                          tau2=q["tau2"], sigw2=q["sigw2"], sd_b=q["sd_b"],
                                          loa_marginal=q["loa_marginal"],
                                          panel_sd_b_true=b[:, :R].std(axis=1, ddof=1))))
    # conventional two-reader Bland-Altman upper limit (readers 0 and 1, 200 cases)
    bias, sd, lo, hi, hw = bland_altman(reads[:, :, 0] - reads[:, :, 1])
    conv = pd.DataFrame(dict(study=study, design="conventional_two_reader", n_readers=2, n_cases=n, total_reads=2 * n,
                             n_reader_pairs=1, cases_per_pair_min=n, cases_per_pair_max=n,
                             tau2=np.nan, sigw2=sd ** 2, sd_b=np.nan, loa_marginal=hi,
                             panel_sd_b_true=b[:, :2].std(axis=1, ddof=1)))
    conv["loa_lo"] = lo
    conv["ci_halfwidth"] = hw
    rows.append(conv)
    out = pd.concat(rows, ignore_index=True)
    out["seed"] = seed_label(oc, oc["streams"]["loa_multireader"], block)
    return {"multi": out}


# ====================================================================== sentinel sets

def task_sentinel(a):
    oc, cell, pilot = a
    sc = oc["sentinel"]
    cfg = C.load_config(ROOT / oc["meta"]["base_config"])
    p = C.cell_params(cfg, "E6", cell)
    S = 100 if pilot else int(sc["n_studies"])
    B = 99 if pilot else int(sc["n_perm"])
    lvl = float(sc["level"])
    d = OA.e6_draws(p, seedseq(oc, oc["streams"]["sentinel_data"], cell), S, int(oc["meta"]["chunk_size"]),
                    skip_double=True)
    dman = d["man"] - d["ai"]
    ns_list = [int(x) for x in p["sentinel_n"]]
    w_list = [float(x) for x in p["anchor_alpha"]]
    pv_perm = OA.perm_variance_ratio_pvalues(dman, ns_list, w_list, B,
                                             rng=gen(oc, oc["streams"]["sentinel_perm"], cell))
    key = dict(cell=cell, p_beat_cv=p["beat_cv"], p_sigma_rb_mm=p["sigma_rb_mm"],
               p_ai_draft_sigma_mm=p["ai_draft_sigma_mm"], p_ai_draft_bias_mm=p["ai_draft_bias_mm"],
               production_n=int(p["production_n"]), level=lvl, n_perm=B)
    rows = []
    for ns in ns_list:
        for w in w_list:
            pv = OA.sentinel_tests(dman, ns, w)
            rej = {"f_onesided": pv["f"] < lvl, "bf_onesided": pv["bf_one"] < lvl, "bf_twosided": pv["bf_two"] < lvl,
                   "welch_mean": pv["welch"] < lvl, "perm_var_ratio_onesided": pv_perm[(ns, w)] <= lvl}
            for tn, r in rej.items():
                k = int(r.sum())
                rate = k / S
                rows.append(dict(**key, sentinel_n=ns, assisted_n=int(p["production_n"]) - ns, w_drift=w, test=tn,
                                 n_reject=k, n_studies=S, rate=rate, mcse=float(np.sqrt(rate * (1 - rate) / S))))
            # agreement between the calibrated test and the Brown-Forsythe decision (same studies)
            rows.append(dict(**key, sentinel_n=ns, assisted_n=int(p["production_n"]) - ns, w_drift=w,
                             test="perm_and_bf_onesided_both", n_reject=int((rej["perm_var_ratio_onesided"] & rej["bf_onesided"]).sum()),
                             n_studies=S, rate=float((rej["perm_var_ratio_onesided"] & rej["bf_onesided"]).mean()), mcse=np.nan))
    c = dman - dman.mean(axis=1, keepdims=True)
    v = np.mean(c * c)
    shape = dict(**key, n_studies=S, var_dman_within_study=float(v), mean_dman=float(dman.mean()),
                 skew_dman=float(np.mean(c ** 3) / v ** 1.5), exkurt_dman=float(np.mean(c ** 4) / v ** 2 - 3.0),
                 seed_data=seed_label(oc, oc["streams"]["sentinel_data"], cell),
                 seed_perm=seed_label(oc, oc["streams"]["sentinel_perm"], cell))
    return {"sentinel": pd.DataFrame(rows), "sentinel_shape": pd.DataFrame([shape])}


# ====================================================================== driver

def _run(t):
    kind, arg = t
    t0 = time.perf_counter()
    c0 = time.process_time()
    out = {"repro": task_repro, "e5": task_e5, "nested": task_nested, "multi": task_multi,
           "sentinel": task_sentinel}[kind](arg)
    return kind, out, time.perf_counter() - t0, time.process_time() - c0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--parts", default="repro,e5,nested,multi,sentinel")
    ap.add_argument("--pilot", action="store_true", help="tiny replicate counts; not for results")
    a = ap.parse_args(argv)
    oc = load()
    parts = a.parts.split(",")
    pilot = a.pilot
    tasks = []
    if "sentinel" in parts:
        cfg = C.load_config(ROOT / oc["meta"]["base_config"])
        cells = range(len(C.experiment_cells(cfg, "E6")))
        tasks += [("sentinel", (oc, c, pilot)) for c in (list(cells)[20:24] if pilot else cells)]
    if "e5" in parts:
        tasks += [("e5", (oc, i, pilot)) for i in range(1 if pilot else len(oc["e5"]["cells"]))]
    if "repro" in parts and not pilot:
        tasks += [("repro", (oc, s, pilot)) for s in oc["reproduction"]]
    if "nested" in parts:
        for si, sc in enumerate(oc["loa_nested"]["scenarios"]):
            nb = 2 if pilot else int(sc["n_pairs"]) // int(oc["loa_nested"]["pairs_per_block"])
            tasks += [("nested", (oc, si, b, pilot)) for b in range(nb)]
    if "multi" in parts:
        nb = 2 if pilot else int(oc["loa_multireader"]["n_studies"]) // int(oc["loa_multireader"]["studies_per_block"])
        tasks += [("multi", (oc, b, pilot)) for b in range(nb)]
    out = Path(a.out) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    print(f"tasks: {len(tasks)}; workers: {a.workers}; pilot: {pilot}", flush=True)
    t0 = time.perf_counter()
    frames, cost = {}, {}
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=4) as pool:
        for kind, res, wall, cpu in pool.imap_unordered(_run, tasks, chunksize=1):
            for name, df in res.items():
                frames.setdefault(name, []).append(df)
            c = cost.setdefault(kind, [0, 0.0, 0.0])
            c[0] += 1
            c[1] += wall
            c[2] += cpu
    for name, lst in frames.items():
        df = pd.concat(lst, ignore_index=True)
        sort = [c for c in ("id", "cell_id", "scenario", "cell", "model", "ref", "pair", "rep", "n_cases", "study",
                            "design", "n_readers", "sentinel_n", "w_drift", "test", "quantity") if c in df.columns]
        df = df.sort_values(sort, kind="stable").reset_index(drop=True)
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"wrote {out / (name + '.parquet')} rows={len(df)}", flush=True)
    cost_df = pd.DataFrame([dict(part=k, n_tasks=v[0], wall_s_sum=v[1], cpu_s_sum=v[2]) for k, v in cost.items()])
    cost_df["wall_s_total_job"] = time.perf_counter() - t0
    cost_df["workers"] = a.workers
    cost_df["pilot"] = pilot
    cost_df.to_csv(out / "run_cost.csv", index=False)
    print(cost_df.to_string(index=False))
    if "repro" in frames:
        r = pd.concat(frames["repro"])
        print("reproduction: rows", len(r), "max abs diff", r.abs_diff.max(), "unmatched", int(r.stored.isna().sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
