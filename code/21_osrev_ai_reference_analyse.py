#!/usr/bin/env python
"""Protocol amendment 3, package A3-5: tidy tables from the raw simulation output.

Reads results/2026-10-05_osrev/ai_reference/raw/*.parquet (written by
code/20_osrev_ai_reference_run.py) and the stored published results (read only), and writes csv
tables to results/2026-10-05_osrev/ai_reference/. No simulation; the only random numbers are the
pair-level bootstrap resamples for Monte Carlo standard errors of variance components
(SeedSequence(20261005, spawn_key=(75, 7, k))).

  python code/21_osrev_ai_reference_analyse.py [--dir results/2026-10-05_osrev/ai_reference]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "code" / "configs" / "osrev_ai_reference.yaml"
Z = 1.96


def mcse_mean(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


def prop(k, n):
    p = k / n
    return p, float(np.sqrt(p * (1 - p) / n))


def boot_gen(oc, k):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        entropy=oc["meta"]["seed_entropy"], spawn_key=(oc["meta"]["spawn_prefix"], oc["streams"]["bootstrap"], k))))


# ====================================================================== E5

def analyse_e5(raw: Path, out: Path, oc):
    st = pd.read_parquet(raw / "e5_study.parquet")
    ana = pd.read_parquet(raw / "e5_analytic.parquet")
    comp = pd.read_parquet(raw / "e5_components.parquet")
    mat = pd.read_parquet(raw / "e5_matching.parquet")
    mods = pd.read_parquet(raw / "e5_models.parquet")
    cal = pd.read_parquet(raw / "e5_calibration.parquet")

    comp.to_csv(out / "ai_reference_error_components.csv", index=False)
    cal.to_csv(out / "ai_inherited_calibration.csv", index=False)

    # ---- analytic check
    a = ana.copy()
    a["mse_z"] = a.mse_sim_minus_pred / a.mse_sim_minus_pred_mcse
    a["mae_sim_minus_normal"] = a.mae_sim - a.mae_normal_pred
    a["mae_sim_minus_normal_pct"] = 100 * a.mae_sim_minus_normal / a.mae_normal_pred
    a["loa_sim_minus_pred_within"] = a.loa_width_sim - a.loa_width_pred_within
    a["loa_sim_minus_pred_total"] = a.loa_width_sim - a.loa_width_pred_total
    a["var_within_sim_minus_pred"] = a.var_within_sim - a.var_within_pred
    # true (vs T1) columns alongside each apparent row
    tr = a[a.ref == "true"][["cell_id", "model", "mae_sim", "mae_sim_mcse", "loa_width_sim", "mse_sim"]].rename(
        columns={"mae_sim": "true_mae_sim", "mae_sim_mcse": "true_mae_sim_mcse", "loa_width_sim": "true_loa_width_sim",
                 "mse_sim": "true_mse_sim"})
    a = a.merge(tr, on=["cell_id", "model"], how="left")
    a.to_csv(out / "ai_analytic_check.csv", index=False)

    # ---- per-model summary with paired apparent-minus-true differences
    wide = {m: st[st.cell_id.notna()].pivot_table(index=["cell_id", "model", "study"], columns="ref", values=m)
            for m in ("mae", "bias", "sd")}
    refs = [r for r in ("true", "single", "adj_tol1", "adj_tol2", "adj_tol3", "mean2") if r in wide["mae"].columns]
    rows = []
    for (cid, model), g in wide["mae"].groupby(level=[0, 1]):
        gb = wide["bias"].loc[(cid, model)]
        gs = wide["sd"].loc[(cid, model)]
        g = g.droplevel([0, 1])
        info = mods[(mods.cell_id == cid) & (mods.model == model)].iloc[0].to_dict()
        for r in refs:
            mae, mae_se = mcse_mean(g[r])
            bias, bias_se = mcse_mean(gb[r])
            loa, loa_se = mcse_mean(2 * Z * gs[r])
            dm, dm_se = mcse_mean(g[r] - g["true"])
            dl, dl_se = mcse_mean(2 * Z * (gs[r] - gs["true"]))
            tm = float(g["true"].mean())
            rows.append(dict(**info, ref=r, n_studies=len(g), mae=mae, mae_mcse=mae_se, bias=bias, bias_mcse=bias_se,
                             loa_width=loa, loa_width_mcse=loa_se, true_mae=tm,
                             apparent_minus_true_mae=dm, apparent_minus_true_mae_mcse=dm_se,
                             apparent_minus_true_mae_pct=100 * dm / tm, apparent_minus_true_mae_pct_mcse=100 * dm_se / tm,
                             apparent_minus_true_loa_width=dl, apparent_minus_true_loa_width_mcse=dl_se))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "ai_model_summary.csv", index=False)

    # ---- matching
    ach = summ[(summ.ref == "true") & (summ.match == "eqmae")][["cell_id", "family", "noise", "sigma_own", "mae", "mae_mcse"]]
    ach = ach.rename(columns={"mae": "true_mae_achieved", "mae_mcse": "true_mae_achieved_mcse"})
    comp_true = summ[(summ.ref == "true") & (summ.model == "indep_sd2")][["cell_id", "mae", "mae_mcse"]].rename(
        columns={"mae": "true_mae_comparator", "mae_mcse": "true_mae_comparator_mcse"})
    m2 = mat.merge(ach, on=["cell_id", "family"], how="left").merge(comp_true, on="cell_id", how="left")
    m2.to_csv(out / "ai_matching.csv", index=False)

    # ---- ranking of matched pairs against the independent comparator (own-error SD 2 mm, noise z)
    rows = []
    for cid in mods.cell_id.unique():
        cmae, cb, cs = (wide[m].loc[(cid, "indep_sd2")] for m in ("mae", "bias", "sd"))
        for _, md in mods[(mods.cell_id == cid) & (mods.family != "independent")].iterrows():
            xm, xb, xs = (wide[m].loc[(cid, md.model)] for m in ("mae", "bias", "sd"))
            n = len(xm)
            for r in refs:
                dmae = (xm[r] - cmae[r]).to_numpy()
                dloa = (2 * Z * (xs[r] - cs[r])).to_numpy()
                dab = (xb[r].abs() - cb[r].abs()).to_numpy()
                k1, k2, k3 = int((dmae < 0).sum()), int((dloa < 0).sum()), int((dab < 0).sum())
                row = dict(cell_id=cid, candidate=md.model, family=md.family, lam=md.lam, match=md.match, noise=md.noise,
                           sigma_own_candidate=md.sigma_own, comparator="indep_sd2", sigma_own_comparator=2.0,
                           ref=r, n_studies=n)
                for nm, k, dd in (("mae", k1, dmae), ("loa_width", k2, dloa), ("abs_bias", k3, dab)):
                    p_, se_ = prop(k, n)
                    m_, ms_ = mcse_mean(dd)
                    row.update({f"n_candidate_better_{nm}": k, f"p_candidate_better_{nm}": p_,
                                f"p_candidate_better_{nm}_mcse": se_, f"diff_{nm}": m_, f"diff_{nm}_mcse": ms_})
                row.update(mae_candidate=float(xm[r].mean()), mae_comparator=float(cmae[r].mean()),
                           loa_width_candidate=float(2 * Z * xs[r].mean()), loa_width_comparator=float(2 * Z * cs[r].mean()),
                           bias_candidate=float(xb[r].mean()), bias_comparator=float(cb[r].mean()))
                rows.append(row)
    rank = pd.DataFrame(rows)
    tr = rank[rank.ref == "true"][["cell_id", "candidate", "mae_candidate", "mae_comparator", "diff_mae", "diff_mae_mcse",
                                   "loa_width_candidate", "loa_width_comparator", "p_candidate_better_mae"]].rename(
        columns={"mae_candidate": "true_mae_candidate", "mae_comparator": "true_mae_comparator",
                 "diff_mae": "true_mae_diff", "diff_mae_mcse": "true_mae_diff_mcse",
                 "loa_width_candidate": "true_loa_width_candidate", "loa_width_comparator": "true_loa_width_comparator",
                 "p_candidate_better_mae": "p_candidate_better_true_mae"})
    rank = rank.merge(tr, on=["cell_id", "candidate"], how="left")
    rank.to_csv(out / "ai_ranking_matched.csv", index=False)
    return summ, rank, comp, a


# ====================================================================== nested reader pairs

def _nested_stats(L, hw, theta_loo, theta_marg):
    """L, hw, theta_loo: (P, M). Per-pair summaries used by the bootstrap."""
    return dict(pair_mean=L.mean(1), pair_var=L.var(1, ddof=1),
                cover_cond=(np.abs(L - theta_loo) <= hw).mean(1), cover_marg=(np.abs(L - theta_marg) <= hw).mean(1),
                hw_mean=hw.mean(1), first=L[:, 0])


def _components(pm, pv, M):
    within = pv.mean()
    between = pm.var(ddof=1) - within / M
    return within, between


def analyse_nested(raw: Path, out: Path, oc):
    reps = pd.read_parquet(raw / "nested_reps.parquet")
    pairs = pd.read_parquet(raw / "nested_pairs.parquet")
    nc = oc["loa_nested"]
    nfull = int(nc["n_cases"])
    rows, targets = [], {}
    for si, sc in enumerate(nc["scenarios"]):
        g = reps[reps.scenario == sc["id"]]
        P = g.pair.nunique()
        M = g.rep.nunique()
        full = g[g.n_cases == nfull].sort_values(["pair", "rep"])
        bias = full.bias.to_numpy().reshape(P, M)
        sd = full.sd.to_numpy().reshape(P, M)
        s1 = nfull * bias
        s2 = (nfull - 1) * sd ** 2 + nfull * bias ** 2
        # leave-one-replicate-out conditional target: pair mean + 1.96 x pair SD from the other M - 1 samples
        N = (M - 1) * nfull
        t1 = s1.sum(1, keepdims=True) - s1
        t2 = s2.sum(1, keepdims=True) - s2
        theta_loo = t1 / N + Z * np.sqrt((t2 - t1 ** 2 / N) / (N - 1))
        # all-sample conditional limits and the marginal target
        Na = M * nfull
        pair_mean = s1.sum(1) / Na
        pair_var = (s2.sum(1) - s1.sum(1) ** 2 / Na) / (Na - 1)
        theta_pair = pair_mean + Z * np.sqrt(pair_var)
        within_d = float(pair_var.mean())
        between_d = float(pair_mean.var(ddof=1) - within_d / Na)
        total_d = within_d + between_d
        theta_marg = float(pair_mean.mean() + Z * np.sqrt(total_d))
        targets[sc["id"]] = dict(scenario=sc["id"], sigma_rb_mm=sc["sigma_rb_mm"], n_pairs=P, n_rep=M, n_cases=nfull,
                                 mean_diff=float(pair_mean.mean()), var_diff_within_pair=within_d,
                                 var_diff_between_pair_means=between_d, var_diff_between_nominal=2 * sc["sigma_rb_mm"] ** 2,
                                 var_diff_total=total_d, marginal_upper_loa=theta_marg,
                                 marginal_upper_loa_zero_mean=float(Z * np.sqrt(total_d)),
                                 sd_conditional_limit_across_pairs=float(theta_pair.std(ddof=1)),
                                 mean_conditional_limit=float(theta_pair.mean()),
                                 pair_var_min=float(pair_var.min()), pair_var_max=float(pair_var.max()),
                                 pair_var_sd=float(pair_var.std(ddof=1)))
        rng = boot_gen(oc, si)
        bidx = rng.integers(0, P, size=(int(nc["n_boot"]), P))
        for nn in nc["sizes"]:
            h = g[g.n_cases == nn].sort_values(["pair", "rep"])
            L = h.loa_hi.to_numpy().reshape(P, M)
            hw = h.ci_halfwidth.to_numpy().reshape(P, M)
            sdn = h.sd.to_numpy().reshape(P, M)
            q = _nested_stats(L, hw, theta_loo, theta_marg)
            within, between = _components(q["pair_mean"], q["pair_var"], M)
            # bootstrap over pairs
            bs = np.empty((bidx.shape[0], 6))
            for i, ix in enumerate(bidx):
                w_, b_ = _components(q["pair_mean"][ix], q["pair_var"][ix], M)
                bs[i] = (np.sqrt(w_), np.sqrt(max(b_, 0.0)), np.sqrt(max(w_ + b_, 0.0)), b_ / (w_ + b_),
                         q["first"][ix].std(ddof=1), w_ + b_)
            se = bs.std(axis=0, ddof=1)
            cc, cc_se = mcse_mean(q["cover_cond"])
            cm, cm_se = mcse_mean(q["cover_marg"])
            hwm, hwm_se = mcse_mean(q["hw_mean"])
            ba_se = np.sqrt(3 * sdn ** 2 / nn)                 # Bland-Altman SE of a limit
            bam, bam_se = mcse_mean(ba_se.mean(1))
            rows.append(dict(
                scenario=sc["id"], sigma_rb_mm=sc["sigma_rb_mm"], n_cases=nn, n_pairs=P, n_rep_per_pair=M,
                n_studies=P * M,
                sd_conditional=float(np.sqrt(within)), sd_conditional_mcse=float(se[0]),
                sd_between_pairs=float(np.sqrt(max(between, 0.0))), sd_between_pairs_mcse=float(se[1]),
                var_between_pairs=float(between),
                sd_marginal=float(np.sqrt(within + between)), sd_marginal_mcse=float(se[2]),
                share_between_of_marginal_var=float(between / (within + between)), share_between_mcse=float(se[3]),
                sd_one_sample_per_pair=float(q["first"].std(ddof=1)), sd_one_sample_per_pair_mcse=float(se[4]),
                empirical_95_range_marginal=float(2 * 1.959964 * np.sqrt(within + between)),
                empirical_95_range_conditional=float(2 * 1.959964 * np.sqrt(within)),
                bland_altman_se_mean=bam, bland_altman_se_mean_mcse=bam_se,
                ratio_conditional_sd_to_ba_se=float(np.sqrt(within) / bam),
                ci_width_mean=2 * hwm, ci_width_mean_mcse=2 * hwm_se,
                coverage_conditional=cc, coverage_conditional_mcse=cc_se,
                n_covered_conditional=int(round(cc * P * M)),
                coverage_marginal=cm, coverage_marginal_mcse=cm_se, n_covered_marginal=int(round(cm * P * M)),
                marginal_target_upper_loa=theta_marg))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "loa_conditional_vs_marginal.csv", index=False)
    tg = pd.DataFrame(list(targets.values()))
    tg.to_csv(out / "loa_nested_targets.csv", index=False)
    return tab, tg


# ====================================================================== multi-reader designs

def analyse_multi(raw: Path, out: Path, oc, tg):
    m = pd.read_parquet(raw / "multi.parquet")
    t = tg[tg.scenario == "rb075"].iloc[0]
    theta = float(t.marginal_upper_loa_zero_mean)
    sb = float(oc["loa_multireader"]["sigma_rb_mm"])
    sb_eff = float(np.sqrt(max(t.var_diff_between_pair_means, 0) / 2))
    rows = []
    for (design, R), g in m.groupby(["design", "n_readers"], sort=False):
        L = g.loa_marginal.to_numpy()
        n = len(g)
        err = L - theta
        mse = float(np.mean(err ** 2))
        mse_se = float(np.std(err ** 2, ddof=1) / np.sqrt(n))
        row = dict(design=design, n_readers=int(R), n_cases=int(g.n_cases.iloc[0]), total_reads=int(g.total_reads.iloc[0]),
                   n_reader_pairs=int(g.n_reader_pairs.iloc[0]), cases_per_pair_min=int(g.cases_per_pair_min.iloc[0]),
                   cases_per_pair_max=int(g.cases_per_pair_max.iloc[0]), n_studies=n,
                   target_marginal_upper_loa=theta,
                   loa_mean=float(L.mean()), loa_mean_mcse=float(L.std(ddof=1) / np.sqrt(n)),
                   loa_bias=float(err.mean()), loa_sd=float(L.std(ddof=1)),
                   loa_sd_mcse=float(L.std(ddof=1) / np.sqrt(2 * (n - 1))),
                   loa_rmse=float(np.sqrt(mse)), loa_rmse_mcse=float(mse_se / (2 * np.sqrt(mse))),
                   loa_p025=float(np.quantile(L, 0.025)), loa_p975=float(np.quantile(L, 0.975)),
                   loa_central95_width=float(np.quantile(L, 0.975) - np.quantile(L, 0.025)))
        if design != "conventional_two_reader":
            s = g.sd_b.to_numpy()
            e2 = (s - sb) ** 2
            k0 = int((g.tau2 <= 0).sum())
            row.update(sd_b_true_nominal=sb, sd_b_effective_nested=sb_eff,
                       sd_b_mean=float(s.mean()), sd_b_mean_mcse=float(s.std(ddof=1) / np.sqrt(n)),
                       sd_b_sd=float(s.std(ddof=1)), sd_b_rmse=float(np.sqrt(e2.mean())),
                       sd_b_rmse_mcse=float(e2.std(ddof=1) / np.sqrt(n) / (2 * np.sqrt(e2.mean()))),
                       tau2_mean=float(g.tau2.mean()), tau2_mean_mcse=float(g.tau2.std(ddof=1) / np.sqrt(n)),
                       tau2_sd=float(g.tau2.std(ddof=1)), n_tau2_nonpositive=k0, p_tau2_nonpositive=k0 / n,
                       sigw2_mean=float(g.sigw2.mean()), sigw2_sd=float(g.sigw2.std(ddof=1)))
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "loa_multireader_designs.csv", index=False)
    return tab


# ====================================================================== sentinel

def analyse_sentinel(raw: Path, out: Path, oc):
    s = pd.read_parquet(raw / "sentinel.parquet")
    shp = pd.read_parquet(raw / "sentinel_shape.parquet")
    s = s.merge(shp[["cell", "exkurt_dman", "skew_dman"]], on="cell", how="left")
    s["mci_lo"] = s.rate - 1.959964 * s.mcse
    s["mci_hi"] = s.rate + 1.959964 * s.mcse
    s.to_csv(out / "sentinel_size_power.csv", index=False)
    shp.to_csv(out / "sentinel_difference_distribution.csv", index=False)
    tests = ["f_onesided", "bf_onesided", "bf_twosided", "welch_mean", "perm_var_ratio_onesided"]
    lvl = float(oc["sentinel"]["level"])
    size = s[(s.w_drift == 0) & s.test.isin(tests)]
    rows = []
    for by, grp in (("all", [("all", size)]), ("sentinel_n", list(size.groupby("sentinel_n")))):
        for lab, g0 in grp:
            for tn in tests:
                g = g0[g0.test == tn]
                rows.append(dict(stratum=by, sentinel_n=lab, test=tn, n_combinations=len(g),
                                 n_studies_per_combination=int(g.n_studies.iloc[0]),
                                 size_min=g.rate.min(), size_median=g.rate.median(), size_max=g.rate.max(),
                                 size_pooled=g.n_reject.sum() / g.n_studies.sum(),
                                 n_reject_pooled=int(g.n_reject.sum()), n_studies_pooled=int(g.n_studies.sum()),
                                 n_point_above_level=int((g.rate > lvl).sum()),
                                 n_mci_above_level=int((g.mci_lo > lvl).sum()),
                                 n_mci_below_level=int((g.mci_hi < lvl).sum()),
                                 n_mci_includes_level=int(((g.mci_lo <= lvl) & (g.mci_hi >= lvl)).sum())))
    ssum = pd.DataFrame(rows)
    ssum.to_csv(out / "sentinel_size_summary.csv", index=False)
    pw = s[(s.w_drift > 0) & s.test.isin(tests)]
    psum = pw.groupby(["test", "sentinel_n", "w_drift"]).rate.agg(["min", "median", "max", "count"]).reset_index()
    psum = psum.rename(columns={"min": "power_min", "median": "power_median", "max": "power_max", "count": "n_cells"})
    psum.to_csv(out / "sentinel_power_summary.csv", index=False)
    base = s[(s.p_beat_cv == 0.15) & (s.p_sigma_rb_mm == 0.75) & (s.p_ai_draft_sigma_mm == 2.0)]
    base.to_csv(out / "sentinel_base_cell.csv", index=False)
    return s, ssum, psum, base


# ====================================================================== published versus new seeds

def _stored(res, exp):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / res / f"{exp}_cells_*.parquet")))],
                     ignore_index=True)


def published_vs_new(out, summ, rank, comp, nested, sent, ssum, base):
    rows = []

    def add(q, pub, pub_se, src, new, new_se, newsrc):
        z = (new - pub) / np.sqrt(pub_se ** 2 + new_se ** 2) if np.isfinite(pub_se) and np.isfinite(new_se) and (pub_se + new_se) > 0 else np.nan
        rows.append(dict(quantity=q, published=pub, published_mcse=pub_se, published_source=src, new_seed=new,
                         new_seed_mcse=new_se, new_source=newsrc, difference=new - pub, z=z))

    e5 = _stored("results/2026-09-18_amend2", "E5")
    c9 = e5[e5.cell == 9]
    dre = pd.read_csv(ROOT / "results/2026-09-18_amend2/analysis/E5bE6b_decomp_reference_error.csv")
    d9 = dre[(dre.cell == 9) & (dre.ref == "single")].iloc[0]
    cb = comp[(comp.cell_id == "base_cv15") & (comp.ref == "single")].iloc[0]
    add("Var(e_img), mm2", d9.var_img, d9.mcse_var_img, "E5bE6b_decomp_reference_error.csv cell 9", cb.var_img, cb.mcse_var_img,
        "ai_reference_error_components.csv:var_img")
    add("Var(e_ref) single read, mm2", d9.var_ref, d9.mcse_var_ref, "E5bE6b_decomp_reference_error.csv cell 9", cb.var_ref,
        cb.mcse_var_ref, "ai_reference_error_components.csv:var_ref")

    def lib(ai, ref, met, sig=2.0, df=c9):
        r = df[(df.ai == ai) & (df.sigma_ai_mm == sig) & (df.ref == ref) & (df.metric == met)].iloc[0]
        return float(r.value), float(r.mcse)

    sb = summ[summ.cell_id == "base_cv15"]

    def new(model, ref, col):
        r = sb[(sb.model == model) & (sb.ref == ref)].iloc[0]
        return float(r[col]), float(r[col + "_mcse"])

    tm, _ = lib("independent", "single", "true_mae")
    for ref in ("single", "mean2"):
        v, se = lib("independent", ref, "diff_mae")
        add(f"apparent minus true MAE, independent SD 2, {ref}, mm", v, se, "amend2 E5 cell 9", *new("indep_sd2", ref, "apparent_minus_true_mae"),
            "ai_model_summary.csv:apparent_minus_true_mae")
        add(f"apparent minus true MAE, independent SD 2, {ref}, %", 100 * v / tm, 100 * se / tm, "amend2 E5 cell 9",
            *new("indep_sd2", ref, "apparent_minus_true_mae_pct"), "ai_model_summary.csv:apparent_minus_true_mae_pct")
    hi, _ = lib("independent", "single", "apparent_loa_hi")
    lo, _ = lib("independent", "single", "apparent_loa_lo")
    thi, _ = lib("independent", "single", "true_loa_hi")
    tlo, _ = lib("independent", "single", "true_loa_lo")
    add("apparent minus true LoA width, independent SD 2, single, mm", (hi - lo) - (thi - tlo), np.nan, "amend2 E5 cell 9",
        *new("indep_sd2", "single", "apparent_minus_true_loa_width"), "ai_model_summary.csv:apparent_minus_true_loa_width")
    for sig in (1.0, 3.0):
        v, se = lib("independent", "single", "diff_mae", sig)
        t_, _ = lib("independent", "single", "true_mae", sig)
        add(f"apparent minus true MAE, independent SD {sig:g}, single, %", 100 * v / t_, 100 * se / t_, "amend2 E5 cell 9",
            *new(f"indep_sd{sig:g}", "single", "apparent_minus_true_mae_pct"), "ai_model_summary.csv:apparent_minus_true_mae_pct")
    v, se = lib("independent", "single", "apparent_ba_bias")
    add("apparent bias, independent SD 2, single, mm", v, se, "amend2 E5 cell 9", *new("indep_sd2", "single", "bias"),
        "ai_model_summary.csv:bias")
    for lam, cell in ((0.5, 10), (1.0, 11)):
        cc = e5[e5.cell == cell]
        for ref, met in (("single", "apparent_mae"), ("single", "true_mae")):
            v, se = lib("shared", ref, met, df=cc)
            add(f"{met} sharing lambda {lam:g}, SD 2, mm", v, se, f"amend2 E5 cell {cell}",
                *new(f"share_lam{lam:g}_eqsd_crn", "single" if met == "apparent_mae" else "true", "mae"), "ai_model_summary.csv:mae")
    rb = rank[(rank.cell_id == "base_cv15")]
    for ref in ("single", "mean2", "adj_tol1", "adj_tol2", "adj_tol3", "T1"):
        r = c9[(c9.ai == "inherited") & (c9.sigma_ai_mm == 2.0) & (c9.ref == ref) & (c9.metric == "p_lower_mae_than_independent")].iloc[0]
        nr = rb[(rb.candidate == "inherit_eqsd_crn") & (rb.ref == ("true" if ref == "T1" else ref))].iloc[0]
        add(f"P(inherited model has lower MAE than independent), {ref}", float(r.value), float(r.mcse), "amend2 E5 cell 9",
            float(nr.p_candidate_better_mae), float(nr.p_candidate_better_mae_mcse), "ai_ranking_matched.csv:p_candidate_better_mae")
    # E6
    for res, lab in (("results/2026-09-18_full", "main E6 cell 22"), ("results/2026-09-18_amend2", "amend2 E6 cell 22")):
        e6 = _stored(res, "E6")
        c = e6[(e6.cell == 22) & (e6.design == "gold")]
        for nn in (25, 200):
            r = c[(c.n_double == nn) & (c.metric == "loa_hi_empirical_sd")].iloc[0]
            nr = nested[(nested.scenario == "rb075") & (nested.n_cases == nn)].iloc[0]
            add(f"SD of upper LoA across studies, n = {nn}, mm ({lab})", float(r.value), float(r.mcse), lab,
                float(nr.sd_marginal), float(nr.sd_marginal_mcse), "loa_conditional_vs_marginal.csv:sd_marginal")
        r = c[(c.n_double == 200) & (c.metric == "loa_ci_width_each_mean")].iloc[0]
        nr = nested[(nested.scenario == "rb075") & (nested.n_cases == 200)].iloc[0]
        add(f"mean Bland-Altman CI width of a limit, n = 200, mm ({lab})", float(r.value), float(r.mcse), lab,
            float(nr.ci_width_mean), float(nr.ci_width_mean_mcse), "loa_conditional_vs_marginal.csv:ci_width_mean")
        sz = e6[(e6.metric == "power_reject_f_variance") & (e6.anchor_alpha == 0)]
        nf = ssum[(ssum.stratum == "all") & (ssum.test == "f_onesided")].iloc[0]
        add(f"F test: combinations (of 216) with size above 0.05 by MCI ({lab})",
            float(((sz.value - 1.959964 * sz.mcse) > 0.05).sum()), np.nan, lab, float(nf.n_mci_above_level), np.nan,
            "sentinel_size_summary.csv:n_mci_above_level")
        add(f"F test: median size ({lab})", float(sz.value.median()), np.nan, lab, float(nf.size_median), np.nan,
            "sentinel_size_summary.csv:size_median")
        if "power_reject_bf_onesided" in set(e6.metric):
            bz = e6[(e6.metric == "power_reject_bf_onesided") & (e6.anchor_alpha == 0)]
            nb = ssum[(ssum.stratum == "all") & (ssum.test == "bf_onesided")].iloc[0]
            add("Brown-Forsythe one-sided: minimum size", float(bz.value.min()), np.nan, lab, float(nb.size_min), np.nan,
                "sentinel_size_summary.csv:size_min")
            add("Brown-Forsythe one-sided: maximum size", float(bz.value.max()), np.nan, lab, float(nb.size_max), np.nan,
                "sentinel_size_summary.csv:size_max")
            c22 = e6[e6.cell == 22]
            for ns, w in ((50, 0.3), (100, 0.1), (50, 0.1)):
                r = c22[(c22.sentinel_n == ns) & (c22.anchor_alpha == w) & (c22.metric == "power_reject_bf_onesided")].iloc[0]
                nr = base[(base.sentinel_n == ns) & (base.w_drift == w) & (base.test == "bf_onesided")].iloc[0]
                add(f"Brown-Forsythe one-sided rejection rate, n = {ns}, drift {w:g}, base cell", float(r.value), float(r.mcse), lab,
                    float(nr.rate), float(nr.mcse), "sentinel_base_cell.csv:rate")
    df = pd.DataFrame(rows)
    df.to_csv(out / "published_vs_new_seed.csv", index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "results" / "2026-10-05_osrev" / "ai_reference"))
    a = ap.parse_args(argv)
    out = Path(a.dir)
    raw = out / "raw"
    with open(CFG_PATH) as fh:
        oc = yaml.safe_load(fh)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    if (raw / "repro.parquet").exists():
        rp = pd.read_parquet(raw / "repro.parquet")
        rp.to_csv(out / "reproduction_check.csv", index=False)
        ok = rp[rp.stored.notna()]
        print("reproduction:", ok.groupby("id").abs_diff.agg(["count", "max"]).to_string())
    summ, rank, comp, ana = analyse_e5(raw, out, oc)
    nested, tg = analyse_nested(raw, out, oc)
    multi = analyse_multi(raw, out, oc, tg)
    sent, ssum, psum, base = analyse_sentinel(raw, out, oc)
    print(nested.to_string())
    print(multi.to_string())
    print(ssum[ssum.stratum == "all"].to_string())
    try:
        pv = published_vs_new(out, summ, rank, comp, nested, sent, ssum, base)
        print(pv.to_string())
    except Exception as exc:  # pilot runs lack the full grid
        print("published_vs_new skipped:", repr(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
