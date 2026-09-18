#!/usr/bin/env python
"""Amendment 2, E5b: closed-form check of apparent MSE and decomposition of reference error.

Regenerates the draws of selected E5b grid cells with the unchanged library (duomaxsim 0.1.0)
and the SAME seeds as the stored run (SeedSequence(20260920, spawn_key=(5, cell)); one child for
the inherited-AI calibration, then one child per 100 validation studies of 200 patients), in the
same draw order as experiments.run_E5. For the base cell the full 2,000 studies are regenerated and
the stored parquet values are reproduced as an exactness check; other cells use the first
N_STUDIES_OTHER studies (a prefix of the stored draws).

Per patient, with T1 the true span, R_img the error-free composite (amendment 2) and r_j read j:
  e_img  = R_img - T1                         image-level error, shared by every reader
  e_rd_j = r_j - R_img = b_j + w_j            read-specific error: reader bias b_j + remainder w_j
  single read reference:  e_ref = e_img + e_rd_1
  mean of two reads:      e_ref = e_img + (e_rd_1 + e_rd_2) / 2
  AI:  e_AI = lambda * e_img + sigma_AI z  (shared AI; lambda 0 = independent AI), or
       a + (b - 1) T1 + sigma_AI z (inherited AI).
Identity (exact in any sample, population moments, ddof 0):
  apparent MSE = true MSE + [Var(e_ref) + bias_ref^2] - 2 [Cov(e_AI, e_ref) + bias_AI bias_ref].
Because every validation study has n = 200, the pooled mean of d^2 equals the mean over studies of
the per-study MSE, which is the quantity the stored per-study metrics average.

Outputs (results/2026-09-18_amend2/analysis/):
  E5bE6b_decomp_reference_error.csv   moments of e_ref by reference and cell
  E5bE6b_decomp_identity.csv          identity check by AI, lambda, sigma_AI, reference and cell
  E5bE6b_decomp_reproduce_stored.csv  recomputed vs stored parquet values (base cell)
  E5bE6b_decomp_inter_reader.csv      read-specific SD vs E6b inter-reader SD
No parquet file is modified. Runtime about 1 to 2 CPU-min on the login node.
Usage: python code/10_amend2_E5b_decomposition.py
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "lib"))
from duomaxsim import config as C  # noqa: E402
from duomaxsim.experiments import (_agreement, _chunks, _gen, _n_chunks, image_composite,  # noqa: E402
                                   protocol_read)
from duomaxsim.model import AXES, draw_latent  # noqa: E402

RES = os.path.join(ROOT, "results", "2026-09-18_amend2")
OUT = os.path.join(RES, "analysis")
CFG = os.path.join(ROOT, "code", "configs", "amend2_2026-09-18.yaml")
BASE_CELL = 9                 # beat CV 0.15, view overestimation on, lambda 0
OTHER_CELLS = [0, 3, 6, 12, 15]   # lambda 0 cells; lambda does not enter the draws
N_STUDIES_OTHER = 500
LAMBDAS = [0.0, 0.5, 1.0]


def regenerate(cfg, cell, n_studies):
    """Replicates run_E5's draw order; returns per-patient arrays shaped (S, n_val)."""
    p = C.cell_params(cfg, "E5", cell)
    ss = C.cell_seed(cfg, "E5", cell)
    ax = AXES.index(p["study_axis"])
    nv = int(p["n_val"])
    R = int(p["n_readers"])
    spc = max(1, int(cfg["meta"]["chunk_size"]) // nv)
    n_full = int(cfg["experiments"]["E5"]["n_studies"])
    kids = ss.spawn(1 + _n_chunks(n_full, spc))       # same spawn count as the stored run
    rng = _gen(kids[0])
    lat = draw_latent(p, int(p["calib_n"]), rng)
    r = protocol_read(lat, p, rng, p["sigma_rb_mm"] * rng.standard_normal(lat.n))[:, ax]
    b_slope, a_int = np.polyfit(lat.S[:, ax], r, 1)
    keep = {k: [] for k in ("T1", "R_img", "r1", "r2", "r3", "b1", "b2", "z")}
    n_chunks_used = _n_chunks(n_studies, spc)
    for s_c, rng in _chunks(kids[1:1 + n_chunks_used], n_studies, spc):
        n = s_c * nv
        lat = draw_latent(p, n, rng)
        pool = p["sigma_rb_mm"] * rng.standard_normal((s_c, R))
        perm = np.argsort(rng.random((n, R)), axis=1)[:, :3]
        study = np.repeat(np.arange(s_c), nv)
        drift = p["drift_mm_per_100"] * np.tile(np.arange(nv), s_c) / 100.0
        reads, biases = [], []
        for j in range(3):
            bj = pool[study, perm[:, j]] + drift
            v, U = protocol_read(lat, p, rng, bj, return_u=True)
            reads.append(v[:, ax].reshape(s_c, nv))
            biases.append(bj.reshape(s_c, nv))
            if j == 0:
                U_r1 = U
        T1 = lat.S[:, ax].reshape(s_c, nv)
        z = rng.standard_normal((s_c, nv))
        R_img = image_composite(lat, p, U_r1)[:, ax].reshape(s_c, nv)
        for k, v in (("T1", T1), ("R_img", R_img), ("r1", reads[0]), ("r2", reads[1]), ("r3", reads[2]),
                     ("b1", biases[0]), ("b2", biases[1]), ("z", z)):
            keep[k].append(v)
    d = {k: np.concatenate(v, axis=0) for k, v in keep.items()}
    return p, d, float(a_int), float(b_slope)


def mom(x):
    x = np.ravel(x)
    return float(x.mean()), float(x.var())


def cov(x, y):
    x, y = np.ravel(x), np.ravel(y)
    return float(np.mean((x - x.mean()) * (y - y.mean())))


N_BLOCKS = 20   # block jackknife over groups of whole validation studies (studies are independent)


def ref_stats(ei, er, eb, ew):
    """Moments of the reference error e_ref = ei + er (image-level ei, read-specific er = eb + ew)."""
    e = ei + er
    m_e, v_e = mom(e)
    m_i, v_i = mom(ei)
    m_r, v_r = mom(er)
    Ee2 = float(np.mean(e ** 2))
    Eie = float(np.mean(ei * e))
    out = dict(bias_ref=m_e, var_ref=v_e, sd_ref=float(np.sqrt(v_e)), msq_ref=Ee2,
               bias_img=m_i, var_img=v_i, msq_img=float(np.mean(ei ** 2)),
               bias_read=m_r, var_read=v_r, two_cov_img_read=2 * cov(ei, er),
               var_reader_bias=mom(eb)[1], var_read_remainder=mom(ew)[1],
               two_cov_bias_remainder=2 * cov(eb, ew),
               share_img_of_var_ref=v_i / v_e if v_e > 0 else np.nan,
               share_img_of_msq_ref=float(np.mean(ei ** 2)) / Ee2,
               within_study_var_ref_mean=float(e.var(axis=1).mean()),
               within_study_var_img_mean=float(ei.var(axis=1).mean()),
               # lambda at which apparent MSE equals true MSE for the shared AI (z terms at expectation 0):
               # sig^2 + (1-l)^2 E[ei^2] + E[er^2] + 2(1-l)E[ei er] = sig^2 + l^2 E[ei^2]  =>  l* = E[e^2] / (2 E[ei e])
               lambda_star_mse=Ee2 / (2 * Eie) if Eie > 0 else np.nan)
    return out


def jackknife_se(fn, arrays, G):
    """Delete-one-block jackknife SE of every statistic returned by fn; blocks of whole studies."""
    S = arrays[0].shape[0]
    idx = np.array_split(np.arange(S), G)
    reps = []
    for g in range(G):
        keep = np.concatenate([idx[h] for h in range(G) if h != g])
        reps.append(fn(*[a[keep] for a in arrays]))
    out = {}
    for k in reps[0]:
        v = np.array([r[k] for r in reps], dtype=float)
        out[k] = float(np.sqrt((G - 1) / G * np.sum((v - v.mean()) ** 2)))
    return out


def within_sd_mean(x):
    """Mean over studies of the per-study SD (ddof 1), as bland_altman computes it, with MCSE."""
    s = x.std(axis=1, ddof=1)
    return float(s.mean()), float(s.std(ddof=1) / np.sqrt(s.size))


def analyse_cell(cfg, cell, n_studies):
    t0 = time.process_time()
    p, d, a_int, b_slope = regenerate(cfg, cell, n_studies)
    T1, R_img = d["T1"], d["R_img"]
    e_img = R_img - T1
    rd1, rd2 = d["r1"] - R_img, d["r2"] - R_img
    w1, w2 = rd1 - d["b1"], rd2 - d["b2"]
    keys = dict(cell=cell, p_beat_cv=p["beat_cv"], p_view_over=p["view_over"], n_studies=n_studies,
                n_patients=int(T1.size), seed=f"SeedSequence(20260920, spawn_key=(5, {cell}))")
    refs = {"single": (e_img, rd1, d["b1"], w1),
            "mean2": (e_img, 0.5 * (rd1 + rd2), 0.5 * (d["b1"] + d["b2"]), 0.5 * (w1 + w2)),
            "R_img": (e_img, np.zeros_like(e_img), np.zeros_like(e_img), np.zeros_like(e_img))}
    rows_ref = []
    for rn, (ei, er, eb, ew) in refs.items():
        full = ref_stats(ei, er, eb, ew)
        se = jackknife_se(ref_stats, (ei, er, eb, ew), N_BLOCKS)
        row = dict(**keys, ref=rn)
        for k_, v_ in full.items():
            row[k_] = v_
            row["mcse_" + k_] = se[k_]
        rows_ref.append(row)
    rows_id = []
    for sig in (1.0, 2.0, 3.0):
        ais = {("shared", lam): lam * e_img + sig * d["z"] for lam in LAMBDAS}
        ais[("inherited", np.nan)] = a_int + (b_slope - 1.0) * T1 + sig * d["z"]
        for (an, lam), e_ai in ais.items():
            m_ai, v_ai = mom(e_ai)
            mse_true = float(np.mean(e_ai ** 2))
            for rn, (ei, er, _, _) in refs.items():
                e_ref = ei + er
                dd = e_ai - e_ref
                mse_app = float(np.mean(dd ** 2))
                m_r, v_r = mom(e_ref)
                c = cov(e_ai, e_ref)
                closed = mse_true + (v_r + m_r ** 2) - 2 * (c + m_ai * m_r)
                row = dict(**keys, ai=an, lam=lam, sigma_ai_mm=sig, ref=rn,
                           mse_true=mse_true, mse_apparent=mse_app,
                           var_ref=v_r, bias_ref=m_r, cov_ai_ref=c, bias_ai=m_ai,
                           mse_apparent_closed_form=closed, identity_abs_error=abs(closed - mse_app),
                           mae_true=float(np.abs(e_ai).mean()), mae_apparent=float(np.abs(dd).mean()),
                           apparent_sd_within_mean=within_sd_mean(dd)[0],
                           apparent_sd_within_mcse=within_sd_mean(dd)[1])
                if an == "shared":
                    # closed form in lambda with the z cross terms set to their expectation 0
                    # (z independent of all image and read errors), e_ref = e_img + e_rd:
                    # E d^2 = sig^2 + (1-lam)^2 E[e_img^2] + E[e_rd^2] + 2 (1-lam) E[e_img e_rd]
                    # E e_AI^2 = sig^2 + lam^2 E[e_img^2]
                    Ei2 = float(np.mean(ei ** 2))
                    Er2 = float(np.mean(er ** 2))
                    Eir = float(np.mean(ei * er))
                    row["mse_apparent_lambda_form"] = sig ** 2 + (1 - lam) ** 2 * Ei2 + Er2 + 2 * (1 - lam) * Eir
                    row["mse_true_lambda_form"] = sig ** 2 + lam ** 2 * Ei2
                rows_id.append(row)
    # stored-value reproduction inputs and inter-reader comparison
    extra = dict(keys=keys, a_int=a_int, b_slope=b_slope, d=d, e_img=e_img, w1=w1, w2=w2,
                 cpu_s=time.process_time() - t0)
    return rows_ref, rows_id, extra


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = C.load_config(CFG)
    e5 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RES, "E5_cells_*.parquet")))],
                   ignore_index=True)
    e6 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RES, "E6_cells_*.parquet")))],
                   ignore_index=True)
    all_ref, all_id, rep, inter = [], [], [], []
    for cell, ns in [(BASE_CELL, 2000)] + [(c, N_STUDIES_OTHER) for c in OTHER_CELLS]:
        rr, ri, ex = analyse_cell(cfg, cell, ns)
        all_ref += rr
        all_id += ri
        k = ex["keys"]
        print(f"cell {cell}: {ns} studies, {ex['cpu_s']:.1f} CPU s", flush=True)
        d = ex["d"]
        # inter-reader: fixed-pair within-study SD of (r1 - r2) excluding reader-bias difference
        sd_w, se_w = within_sd_mean(ex["w1"] - ex["w2"])
        g6 = e6[(e6.p_beat_cv == k["p_beat_cv"]) & (e6.design == "gold") & (e6.n_double == 200)
                 & (e6.metric == "loa_width_mean") & (e6.p_ai_draft_sigma_mm == 2.0)]
        for _, r6 in g6.iterrows():
            inter.append(dict(**{kk: k[kk] for kk in ("cell", "p_beat_cv", "p_view_over", "n_studies")},
                              e5_within_sd_read_diff_excl_bias=sd_w, mcse_e5=se_w,
                              e6_cell=int(r6.cell), e6_sigma_rb_mm=r6.p_sigma_rb_mm,
                              e6_view_over=True, comparable=bool(k["p_view_over"]),
                              e6_within_sd_read_diff=r6.value / 3.92, mcse_e6=r6.mcse / 3.92,
                              diff_e5_minus_e6=sd_w - r6.value / 3.92,
                              mcse_diff=float(np.hypot(se_w, r6.mcse / 3.92))))
        if cell == BASE_CELL:
            st = e5[e5.cell == cell]
            T1 = d["T1"]
            recompute = {}
            ind = T1 + 2.0 * d["z"]
            for rn, ref in (("single", d["r1"]), ("mean2", 0.5 * (d["r1"] + d["r2"]))):
                a = _agreement(ind, ref)
                recompute[("independent", 2.0, rn, "apparent_mae")] = a["mae"].mean()
                recompute[("independent", 2.0, rn, "apparent_ba_bias")] = a["ba_bias"].mean()
                recompute[("independent", 2.0, rn, "apparent_icc")] = a["icc"].mean()
                inh = ex["a_int"] + ex["b_slope"] * T1 + 2.0 * d["z"]
                recompute[("inherited", 2.0, rn, "apparent_mae")] = _agreement(inh, ref)["mae"].mean()
            ref_rows = {("R_img", "ref_mae_vs_T1"): np.abs(d["R_img"] - T1).mean(axis=1).mean(),
                        ("R_img", "ref_ba_bias_vs_T1"): (d["R_img"] - T1).mean(axis=1).mean(),
                        ("single", "ref_mae_vs_T1"): np.abs(d["r1"] - T1).mean(axis=1).mean()}
            for (an, sig, rn, m), v in recompute.items():
                sv = st[(st.ai == an) & (st.sigma_ai_mm == sig) & (st.ref == rn) & (st.metric == m)].value.iloc[0]
                rep.append(dict(cell=cell, quantity=f"{an} sigma {sig:g} {rn} {m}", recomputed=v, stored=sv,
                                abs_diff=abs(v - sv)))
            for (rn, m), v in ref_rows.items():
                sv = st[(st.ai.isna()) & (st.ref == rn) & (st.metric == m)].value.iloc[0]
                rep.append(dict(cell=cell, quantity=f"reference {rn} {m}", recomputed=v, stored=sv,
                                abs_diff=abs(v - sv)))
            for m, v in (("inherited_intercept_mm", ex["a_int"]), ("inherited_slope", ex["b_slope"])):
                sv = st[(st.estimator == "calibration") & (st.metric == m)].value.iloc[0]
                rep.append(dict(cell=cell, quantity=m, recomputed=v, stored=sv, abs_diff=abs(v - sv)))
    ref_df = pd.DataFrame(all_ref)
    id_df = pd.DataFrame(all_id)
    rep_df = pd.DataFrame(rep)
    int_df = pd.DataFrame(inter)
    ref_df.to_csv(os.path.join(OUT, "E5bE6b_decomp_reference_error.csv"), index=False)
    id_df.to_csv(os.path.join(OUT, "E5bE6b_decomp_identity.csv"), index=False)
    rep_df.to_csv(os.path.join(OUT, "E5bE6b_decomp_reproduce_stored.csv"), index=False)
    int_df.to_csv(os.path.join(OUT, "E5bE6b_decomp_inter_reader.csv"), index=False)
    pd.set_option("display.width", 250)
    print(rep_df.to_string())
    print("max identity abs error:", id_df.identity_abs_error.max())
    print(ref_df[["cell", "p_beat_cv", "p_view_over", "ref", "bias_ref", "var_ref", "var_img", "var_read",
                  "two_cov_img_read", "share_img_of_var_ref", "lambda_star_mse", "mcse_lambda_star_mse"]].round(4).to_string())
    print(int_df.round(4).to_string())


if __name__ == "__main__":
    main()
