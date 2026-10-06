#!/usr/bin/env python
"""Amendment 3, package A3-4 (triggers and review): derived tables.

Reads results/2026-10-05_osrev/triggers/raw/*.csv (written by
code/20_osrev_triggers_run.py) and the stored pre-specified and amendment-2
results, and writes tidy csv tables to results/2026-10-05_osrev/triggers/.
No simulation is run here except the cluster bootstrap, which resamples the
stored per-patient pattern counts (seed SeedSequence(20261005, spawn_key=(74, 9))).
Login node, well under one CPU-minute.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from duomaxsim import osrev_triggers as T  # noqa: E402

SPEC = yaml.safe_load(open(ROOT / "code" / "configs" / "osrev_triggers.yaml"))
Z = T.Z95
MEASURES = ("prevalence", "fired", "sensitivity", "false_positive_rate", "specificity", "ppv", "npv")
PUBLISHED = {  # manuscript v05, Results, 'Triggers' paragraph (AP axis, four views, base case)
    ("prevalence", 0.0): 0.069, ("sensitivity", 5.0): 0.088, ("false_positive_rate", 5.0): 0.0029,
    ("ppv", 5.0): 0.697, ("sensitivity", 3.0): 0.216, ("false_positive_rate", 3.0): 0.022,
}


def hist_from_rows(g: pd.DataFrame) -> np.ndarray:
    m = int((g.tp_pairs + g.fp_pairs + g.fn_pairs + g.tn_pairs).iloc[0])
    H = np.zeros((m + 1,) * 3, dtype=np.int64)
    np.add.at(H, (g.tp_pairs.to_numpy(), g.fp_pairs.to_numpy(), g.fn_pairs.to_numpy()), g.n_patients.to_numpy())
    return H


def cohort_label(cohort, rep):
    return "published" if cohort == "published" else f"new_rep{int(rep)}"


# =========================================================================== pair level

def pair_tables(raw: Path, out: Path):
    pat = pd.read_csv(raw / "A_pair_patterns.csv")
    n_boot = int(SPEC["triggers"]["bootstrap_resamples"])
    rng = T.gen(np.random.SeedSequence(int(SPEC["meta"]["seed_entropy"]), spawn_key=(int(SPEC["meta"]["spawn_prefix"]), 9)))
    hists = {}
    for (cohort, rep, axis, tau, dfn), g in pat.groupby(["cohort", "rep", "axis", "tau_mm", "definition"], sort=True):
        hists[(cohort_label(cohort, rep), axis, float(tau), dfn)] = hist_from_rows(g)
    reps = sorted({k[0] for k in hists if k[0].startswith("new_rep")}, key=lambda s: int(s[7:]))
    for (_, axis, tau, dfn) in [k for k in hists if k[0] == reps[0]]:
        hists[("new_pooled", axis, tau, dfn)] = sum(hists[(r, axis, tau, dfn)] for r in reps)
    rows, counts = [], []
    for (coh, axis, tau, dfn), H in sorted(hists.items()):
        r = T.rates_from_patterns(H)
        c = r["counts"]
        base = dict(cohort=coh, axis=axis, tau_mm=tau, definition=dfn)
        counts.append(dict(**base, fired_error=c["tp"], fired_no_error=c["fp"], not_fired_error=c["fn"],
                           not_fired_no_error=c["tn"], n_pairs=c["tp"] + c["fp"] + c["fn"] + c["tn"],
                           n_patients=c["n_patients"], pairs_per_patient=c["pairs_per_patient"],
                           n_patients_with_fired_pair=int(H.sum() - H[0, 0, :].sum()),
                           n_patients_with_error_pair=int(H.sum() - H[0, :, 0].sum())))
        boot = T.bootstrap_from_patterns(H, n_boot, rng) if not coh.startswith("new_rep") or coh == reps[0] else None
        for m in MEASURES:
            q = r[m]
            row = dict(**base, measure=m, value=q["value"], numerator=q["num"], denominator=q["den"],
                       se_iid=q["se_iid"], se_conservative_sqrt3=q["se_sqrt_m"], se_cluster=q["se_cluster"],
                       design_effect=q["deff"],
                       lo95_cluster=q["value"] - Z * q["se_cluster"], hi95_cluster=q["value"] + Z * q["se_cluster"],
                       lo95_conservative=q["value"] - Z * q["se_sqrt_m"], hi95_conservative=q["value"] + Z * q["se_sqrt_m"],
                       lo95_iid=q["value"] - Z * q["se_iid"], hi95_iid=q["value"] + Z * q["se_iid"])
            if boot is not None:
                row.update(se_bootstrap=boot[m][0], lo95_bootstrap=boot[m][1], hi95_bootstrap=boot[m][2],
                           bootstrap_resamples=n_boot)
            rows.append(row)
    rates = pd.DataFrame(rows)
    counts = pd.DataFrame(counts)
    # replicate-to-replicate SD of each rate: an empirical check of the standard errors
    rr = rates[rates.cohort.isin(reps)].groupby(["axis", "tau_mm", "definition", "measure"])["value"].agg(
        replicate_mean="mean", replicate_sd=lambda v: v.std(ddof=1), n_replicates="count").reset_index()
    rates = rates.merge(rr, on=["axis", "tau_mm", "definition", "measure"], how="left")
    keep = rates.cohort.isin(["published", "new_pooled", reps[0]])
    rates[keep].to_csv(out / "triggers_pair_rates.csv", index=False)
    counts[counts.cohort.isin(["published", "new_pooled", reps[0]])].to_csv(out / "triggers_pair_2x2_counts.csv", index=False)
    rates[rates.cohort.isin(reps + ["published"]) & (rates.definition == "D0_published") & rates.tau_mm.isin([3.0, 5.0])][
        ["cohort", "axis", "tau_mm", "measure", "value", "numerator", "denominator", "se_cluster"]].to_csv(
        out / "triggers_replicates.csv", index=False)
    return rates, counts, reps


def consistency_table(rates, counts, out: Path):
    """Mutual consistency of the published pair-level numbers (AP axis, published cohort)."""
    rows = []
    for tau in (3.0, 5.0):
        c = counts[(counts.cohort == "published") & (counts.axis == "AP") & (counts.tau_mm == tau)
                   & (counts.definition == "D0_published")].iloc[0]
        tp, fp, fn, tn = (int(c[k]) for k in ("fired_error", "fired_no_error", "not_fired_error", "not_fired_no_error"))
        r = T.rates_2x2(tp, fp, fn, tn)
        pv, se, fpr = r["prevalence"][0], r["sensitivity"][0], r["false_positive_rate"][0]
        bayes = pv * se / (pv * se + (1 - pv) * fpr)
        pub_p = PUBLISHED[("prevalence", 0.0)]
        pub_se, pub_fp = PUBLISHED[("sensitivity", tau)], PUBLISHED[("false_positive_rate", tau)]
        f = lambda a, b, cc: a * b / (a * b + (1 - a) * cc)
        hp, hs = 0.0005, 0.0005
        hf = 0.00005 if tau == 5.0 else 0.0005
        rows.append(dict(
            cohort="published", axis="AP", tau_mm=tau, fired_error=tp, fired_no_error=fp, not_fired_error=fn,
            not_fired_no_error=tn, n_pairs=tp + fp + fn + tn,
            prevalence=pv, sensitivity=se, false_positive_rate=fpr, ppv_from_counts=r["ppv"][0], npv_from_counts=r["npv"][0],
            ppv_from_unrounded_rates=bayes, abs_diff_counts_vs_rates=abs(bayes - r["ppv"][0]),
            published_prevalence=pub_p, published_sensitivity=pub_se, published_false_positive_rate=pub_fp,
            ppv_from_published_rounded_rates=f(pub_p, pub_se, pub_fp),
            ppv_min_within_rounding=f(pub_p - hp, pub_se - hs, pub_fp + hf),
            ppv_max_within_rounding=f(pub_p + hp, pub_se + hs, pub_fp - hf),
            published_ppv=PUBLISHED.get(("ppv", tau), 0.424)))
    pd.DataFrame(rows).to_csv(out / "triggers_pair_consistency.csv", index=False)


def interval_table(rates, out: Path):
    q = rates[(rates.definition == "D0_published") & rates.cohort.isin(["published", "new_pooled"])
              & rates.tau_mm.isin([3.0, 5.0]) & rates.measure.isin(["ppv", "sensitivity", "false_positive_rate", "prevalence", "npv"])]
    rows = []
    for _, r in q.iterrows():
        for method, se, lo, hi in (
                ("binomial, pairs independent", r.se_iid, r.lo95_iid, r.hi95_iid),
                ("'conservative' of manuscript v05: binomial SE x sqrt(3)", r.se_conservative_sqrt3,
                 r.lo95_conservative, r.hi95_conservative),
                ("patient-level linearization (delta method)", r.se_cluster, r.lo95_cluster, r.hi95_cluster),
                ("patient-level (cluster) bootstrap, percentile", r.se_bootstrap, r.lo95_bootstrap, r.hi95_bootstrap)):
            rows.append(dict(cohort=r.cohort, axis=r.axis, tau_mm=r.tau_mm, measure=r.measure, value=r.value,
                             numerator=r.numerator, denominator=r.denominator, method=method, se=se, lo95=lo, hi95=hi,
                             replicate_sd_10_new_cohorts=r.replicate_sd))
    pd.DataFrame(rows).to_csv(out / "triggers_interval_methods.csv", index=False)


# =========================================================================== patient level

def _pool_new(df, keys, sums):
    new = df[df.cohort == "new"]
    pooled = new.groupby(keys, as_index=False)[sums].sum()
    pooled["cohort"] = "new_pooled"
    one = pd.concat([df[df.cohort == "published"], new[new.rep == 0]]).copy()
    one["cohort"] = [cohort_label(c, r) for c, r in zip(one.cohort, one.rep)]
    per_rep = new.copy()
    per_rep["cohort"] = [cohort_label(c, r) for c, r in zip(per_rep.cohort, per_rep.rep)]
    return pd.concat([one[keys + sums + ["cohort"]], pooled[keys + sums + ["cohort"]]], ignore_index=True), per_rep


def patient_table(raw: Path, out: Path):
    df = pd.read_csv(raw / "A_patient_2x2.csv")
    keys = ["scope", "tau_mm", "truth"]
    t, per_rep = _pool_new(df, keys, ["tp", "fp", "fn", "tn", "tp_matched", "n_patients"])
    rows = []
    for _, r in t.iterrows():
        q = T.rates_2x2(int(r.tp), int(r.fp), int(r.fn), int(r.tn))
        row = dict(cohort=r.cohort, scope=r.scope, tau_mm=r.tau_mm, truth=r.truth,
                   fired_error=int(r.tp), fired_no_error=int(r.fp), not_fired_error=int(r.fn),
                   not_fired_no_error=int(r.tn), n_patients=int(r.n_patients))
        for m in MEASURES:
            row[m] = q[m][0]
            row[m + "_mcse"] = q[m][1]
        if r.tp_matched >= 0:
            v, se = T._prop(int(r.tp_matched), int(r.tp + r.fn))
            row.update(fired_in_a_pair_with_the_error=int(r.tp_matched), sensitivity_pair_matched=v,
                       sensitivity_pair_matched_mcse=se)
        rows.append(row)
    pd.DataFrame(rows).sort_values(["cohort", "scope", "truth", "tau_mm"]).to_csv(out / "triggers_patient_2x2.csv", index=False)
    return pd.DataFrame(rows)


def partition_tables(raw: Path, out: Path):
    df = pd.read_csv(raw / "A_pair_partition.csv")
    keys = ["axis", "tau_mm", "error_kind", "class_id", "class_label"]
    sums = ["n_pairs", "n_fired", "n_agree_within_tau", "n_patients_any", "n_patients_any_fired_any", "n_patients"]
    t, per_rep = _pool_new(df, keys, sums)
    m = int(SPEC["triggers"].get("pairs_per_patient", 3))
    t["n_pairs_total"] = t["n_patients"] * m
    t["prop_pairs"] = t.n_pairs / t.n_pairs_total
    t["prop_pairs_se_iid"] = np.sqrt(t.prop_pairs * (1 - t.prop_pairs) / t.n_pairs_total)
    t["prop_pairs_se_upper_sqrt3"] = t.prop_pairs_se_iid * np.sqrt(m)
    t["prop_patients_any"] = t.n_patients_any / t.n_patients
    t["prop_patients_any_mcse"] = np.sqrt(t.prop_patients_any * (1 - t.prop_patients_any) / t.n_patients)
    t["fired_given_class"] = t.n_fired / t.n_pairs.where(t.n_pairs > 0)
    t["fired_given_class_se_iid"] = np.sqrt(t.fired_given_class * (1 - t.fired_given_class) / t.n_pairs.where(t.n_pairs > 0))
    t["agree_within_tau_given_class"] = t.n_agree_within_tau / t.n_pairs.where(t.n_pairs > 0)
    # replicate-based MCSE (respects clustering) for the pooled new cohorts
    per_rep = per_rep.assign(pp=per_rep.n_pairs / (per_rep.n_patients * m),
                             fg=per_rep.n_fired / per_rep.n_pairs.where(per_rep.n_pairs > 0))
    rs = per_rep.groupby(keys).agg(prop_pairs_replicate_se=("pp", lambda v: v.std(ddof=1) / np.sqrt(v.size)),
                                   fired_given_class_replicate_se=("fg", lambda v: v.std(ddof=1) / np.sqrt(v.count()))).reset_index()
    rs["cohort"] = "new_pooled"
    t = t.merge(rs, on=keys + ["cohort"], how="left")
    t.sort_values(["cohort", "error_kind", "axis", "tau_mm", "class_id"]).to_csv(out / "triggers_pair_partition.csv", index=False)

    ex = pd.read_csv(raw / "A_exam_offsets.csv")
    keys = ["axis", "tau_mm", "error_kind"]
    sums = ["all_low", "all_low_fired", "all_high", "all_high_fired", "any_view_wrong", "any_view_wrong_fired",
            "fired_any", "n_patients"]
    e, _ = _pool_new(ex, keys, sums)
    for a in ("all_low", "all_high", "any_view_wrong", "fired_any"):
        e["prop_" + a] = e[a] / e.n_patients
        e["prop_" + a + "_mcse"] = np.sqrt(e["prop_" + a] * (1 - e["prop_" + a]) / e.n_patients)
    for a in ("all_low", "all_high", "any_view_wrong"):
        d = e[a].where(e[a] > 0)
        e["fired_given_" + a] = e[a + "_fired"] / d
        e["fired_given_" + a + "_mcse"] = np.sqrt(e["fired_given_" + a] * (1 - e["fired_given_" + a]) / d)
    e.sort_values(["cohort", "error_kind", "axis", "tau_mm"]).to_csv(out / "triggers_exam_offsets.csv", index=False)

    ins = pd.read_csv(raw / "A_instrument.csv")
    i, _ = _pool_new(ins.assign(dummy=0), ["dummy"], ["n_g_shift_gt_thr_ap", "n_g_shift_gt_thr_sl", "n_patients"])
    i = i.drop(columns="dummy")
    for a in ("ap", "sl"):
        i[f"prop_g_shift_gt_2mm_{a}"] = i[f"n_g_shift_gt_thr_{a}"] / i.n_patients
        i[f"prop_g_shift_gt_2mm_{a}_mcse"] = np.sqrt(i[f"prop_g_shift_gt_2mm_{a}"] * (1 - i[f"prop_g_shift_gt_2mm_{a}"]) / i.n_patients)
    i.to_csv(out / "triggers_instrument_shift.csv", index=False)


def definition_summary(rates, counts, patient, out: Path):
    """One row per error definition, trigger threshold, axis and cohort: pairs and patients."""
    rows = []
    for (coh, axis, tau, dfn), c in counts.groupby(["cohort", "axis", "tau_mm", "definition"]):
        if coh not in ("published", "new_pooled"):
            continue
        c = c.iloc[0]
        r = rates[(rates.cohort == coh) & (rates.axis == axis) & (rates.tau_mm == tau) & (rates.definition == dfn)].set_index("measure")
        p = patient[(patient.cohort == coh) & (patient.scope == axis) & (patient.tau_mm == tau)
                    & (patient.truth == "any_pair:" + dfn)].iloc[0]
        row = dict(cohort=coh, axis=axis, tau_mm=tau, definition=dfn,
                   n_pairs=int(c.n_pairs), n_error_pairs=int(c.fired_error + c.not_fired_error),
                   n_fired_pairs=int(c.fired_error + c.fired_no_error), n_fired_error_pairs=int(c.fired_error))
        for m in ("prevalence", "sensitivity", "false_positive_rate", "ppv", "npv"):
            row["pair_" + m] = r.loc[m, "value"]
            row["pair_" + m + "_se_cluster"] = r.loc[m, "se_cluster"]
        row.update(n_patients=int(p.n_patients), n_patients_with_error=int(p.fired_error + p.not_fired_error),
                   n_patients_fired=int(p.fired_error + p.fired_no_error), n_patients_fired_with_error=int(p.fired_error))
        for m in ("prevalence", "sensitivity", "false_positive_rate", "ppv", "npv"):
            row["patient_" + m] = p[m]
            row["patient_" + m + "_mcse"] = p[m + "_mcse"]
        row["patient_sensitivity_pair_matched"] = p.get("sensitivity_pair_matched", np.nan)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "triggers_error_definitions.csv", index=False)


def trigger_reproduction(rates, patient, raw: Path, out: Path):
    """Stored E4 cell 193 values against the rerun of the published cohort."""
    cell = int(SPEC["triggers"]["published"]["cell"])
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "results/2026-09-18_full/E4_*.parquet")))])
    d = d[d.cell == cell]
    ax = pd.read_csv(raw / "A_axis_diff.csv")
    ax = ax[ax.cohort == "published"]
    R = rates[(rates.cohort == "published") & (rates.definition == "D0_published")]
    rows = []

    def stored(metric, axis, **kw):
        q = d[(d.metric == metric) & (d.axis == axis)]
        for k, v in kw.items():
            q = q[q[k] == v]
        assert q.value.nunique() == 1, (metric, axis, kw)
        return float(q.value.iloc[0]), int(q.n.iloc[0])

    def add(label, metric, axis, new, n_new, **kw):
        v, n = stored(metric, axis, **kw)
        rows.append(dict(quantity=label, axis=axis, stored_metric=metric, stored_value=v, stored_n=n,
                         rerun_value=new, rerun_n=n_new, abs_diff=abs(v - new), identical=bool(v == new)))

    for axis in ("AP", "SL"):
        g = lambda m, tau: R[(R.axis == axis) & (R.tau_mm == tau) & (R.measure == m)].iloc[0]
        add("pairs with a true view error > 2 mm", "p_error_event_pair", axis, g("prevalence", 5.0).value, g("prevalence", 5.0).denominator)
        for tau, mt, kw in ((5.0, "adj", dict(t_adj=5.0)), (3.0, "trigger", dict(t_warn=3.0))):
            add(f"false-positive rate, {tau:g} mm", f"false_alarm_{mt}_pair", axis, g("false_positive_rate", tau).value,
                g("false_positive_rate", tau).denominator, **kw)
            add(f"1 - sensitivity, {tau:g} mm", f"miss_{mt}_pair", axis, 1.0 - g("sensitivity", tau).value,
                g("sensitivity", tau).denominator, **kw)
            p = patient[(patient.cohort == "published") & (patient.scope == axis) & (patient.tau_mm == tau)
                        & (patient.truth == "any_pair:D0_published")].iloc[0]
            add(f"patients with any trigger, {tau:g} mm", "p_any_adj" if mt == "adj" else "p_any_trigger", axis,
                p.fired, int(p.n_patients), **kw)
    for tau, metric, kw in ((3.0, "p_anchor_axis_diff_ge_twarn", dict(t_warn=3.0)), (5.0, "p_anchor_axis_diff_ge_tadj", dict(t_adj=5.0))):
        a = ax[(ax.quantity == "measured:A1_anchor") & (ax.tau_mm == tau)].iloc[0]
        add(f"measured anchor |AP - SL| >= {tau:g} mm", metric, "AP-SL", a.k / a.n_patients, int(a.n_patients), **kw)
    a = ax[(ax.quantity == "true") & (ax.tau_mm == 3.0)].iloc[0]
    add("true |AP - SL| >= 3 mm (this cell; the manuscript pools 144 cells)", "p_true_axis_diff_ge_twarn", "AP-SL",
        a.k / a.n_patients, int(a.n_patients), t_warn=3.0)
    pd.DataFrame(rows).to_csv(out / "triggers_reproduction_E4_cell193.csv", index=False)


# =========================================================================== review

def review_tables(raw: Path, out: Path):
    met = pd.read_csv(raw / "D_review_metrics.csv")
    par = pd.read_csv(raw / "D_review_paired.csv")
    proc = pd.read_csv(raw / "D_review_process.csv")
    idx = ["run", "scenario", "overestimation", "axis", "rule", "s_det", "f_rej"]
    met.to_csv(out / "review_metrics_long.csv", index=False)
    par.to_csv(out / "review_paired_long.csv", index=False)

    def wide(df, valcol, secol, suffix=""):
        v = df.pivot_table(index=idx, columns="metric", values=valcol)
        s = df.pivot_table(index=idx, columns="metric", values=secol)
        v.columns = [f"{c}{suffix}" for c in v.columns]
        s.columns = [f"{c}{suffix}_mcse" for c in s.columns]
        return pd.concat([v, s], axis=1)

    for c in ("s_det", "f_rej"):
        met[c] = met[c].fillna(-1.0)
        par[c] = par[c].fillna(-1.0)
    w = wide(met, "value", "mcse")
    cnt = met[met.metric.str.startswith(("sensitivity", "specificity"))].pivot_table(index=idx, columns="metric", values=["k", "n"])
    cnt.columns = [f"{m}_{'events_correct' if a == 'k' else 'denominator'}" for a, m in cnt.columns]
    w3 = wide(par[par.minus == "A3"], "difference", "mcse", "_minus_largest_view_mean")
    wb = wide(par[par.minus == "A4|0.8|0.1"], "difference", "mcse", "_minus_published_review")
    W = pd.concat([w, cnt, w3, wb], axis=1).reset_index()
    W["s_det"] = W["s_det"].replace(-1.0, np.nan)
    W["f_rej"] = W["f_rej"].replace(-1.0, np.nan)
    W["reviewer"] = np.select(
        [W.rule != "A4", (W.s_det == 0) & (W.f_rej == 0), W.s_det == W.f_rej, (W.s_det == 0.8) & (W.f_rej == 0.1),
         W.s_det < W.f_rej],
        ["not applicable", "no effective review", "uninformative (equal probabilities)", "published assumption",
         "worse than uninformative"], "informative")
    order = [c for c in W.columns if c not in idx + ["reviewer"]]
    W = W[idx + ["reviewer"] + sorted(order)]
    W.to_csv(out / "review_grid_wide.csv", index=False)

    proc = proc.copy()
    proc["prop_other_views_reviewed"] = proc.n_reviewed / proc.n_other_views
    proc["prop_reviewed_truly_overestimated"] = proc.n_reviewed_over / proc.n_reviewed.where(proc.n_reviewed > 0)
    proc["prop_patients_reviewed"] = proc.n_patients_reviewed / proc.n_patients
    proc["prop_patients_value_changed"] = proc.n_patients_value_changed / proc.n_patients
    proc["prop_patients_value_changed_mcse"] = np.sqrt(proc.prop_patients_value_changed * (1 - proc.prop_patients_value_changed) / proc.n_patients)
    proc["prop_exclusions_valid"] = proc.n_excluded_valid / proc.n_excluded.where(proc.n_excluded > 0)
    proc.to_csv(out / "review_process.csv", index=False)

    # dependence of the published advantage of review on the assumed probabilities
    A = W[(W.run == "new") & (W.rule == "A4")]
    rows = []
    for (sc, ov, axis), g in A.groupby(["scenario", "overestimation", "axis"]):
        a3 = W[(W.run == "new") & (W.rule == "A3") & (W.scenario == sc) & (W.overestimation == ov) & (W.axis == axis)].iloc[0]
        pub = g[(g.s_det == 0.8) & (g.f_rej == 0.1)].iloc[0]
        row = dict(scenario=sc, overestimation=ov, axis=axis,
                   largest_view_mean_bias_mm=a3.bias_mm, largest_view_mean_rmse_mm=a3.rmse_mm, largest_view_mean_mae_mm=a3.mae_mm,
                   published_review_bias_mm=pub.bias_mm, published_review_rmse_mm=pub.rmse_mm, published_review_mae_mm=pub.mae_mm)
        for m in ("bias_mm", "rmse_mm", "mae_mm", "sensitivity_13mm", "specificity_13mm"):
            col = f"{m}_minus_largest_view_mean"
            row[f"published_review_{m}_difference"] = pub[col]
            row[f"published_review_{m}_difference_mcse"] = pub[col + "_mcse"]
            row[f"grid_min_{m}_difference"] = g[col].min()
            row[f"grid_max_{m}_difference"] = g[col].max()
        d, se = g["rmse_mm_minus_largest_view_mean"], g["rmse_mm_minus_largest_view_mean_mcse"]
        better = g[(d + Z * se) < 0]
        worse = g[(d - Z * se) > 0]
        row["n_grid_points"] = len(g)
        row["n_rmse_lower_than_largest_view_mean"] = len(better)
        row["n_rmse_higher_than_largest_view_mean"] = len(worse)
        row["rmse_higher_at"] = "; ".join(f"({s:g}, {f:g})" for s, f in zip(worse.s_det, worse.f_rej))
        un = g[(g.s_det == g.f_rej) & (g.s_det > 0)]
        row["uninformative_rmse_difference_min"] = un["rmse_mm_minus_largest_view_mean"].min()
        row["uninformative_rmse_difference_max"] = un["rmse_mm_minus_largest_view_mean"].max()
        # share of the published RMSE and bias advantage retained by an uninformative reviewer at 0.1 and by s_det 0.5
        for lab, s, f in (("uninformative_0.1", 0.1, 0.1), ("uninformative_0.5", 0.5, 0.5), ("s0.5_f0.1", 0.5, 0.1),
                          ("s0.2_f0.1", 0.2, 0.1), ("s0_f0.1", 0.0, 0.1), ("s1_f0", 1.0, 0.0), ("s0.8_f0.3", 0.8, 0.3),
                          ("s0.8_f0.5", 0.8, 0.5), ("s0.8_f0", 0.8, 0.0)):
            q = g[(g.s_det == s) & (g.f_rej == f)].iloc[0]
            row[f"rmse_difference_{lab}"] = q["rmse_mm_minus_largest_view_mean"]
            row[f"rmse_difference_{lab}_mcse"] = q["rmse_mm_minus_largest_view_mean_mcse"]
            row[f"bias_difference_{lab}"] = q["bias_mm_minus_largest_view_mean"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "review_dependence_summary.csv", index=False)
    return W


def review_reproduction(W, out: Path):
    rows = []
    stores = {"base": ("results/2026-09-18_full", 703), "long_axis_8pct": ("results/2026-09-18_full", 1855),
              "extreme_accurate": ("results/2026-09-18_amend2", 36), "extreme_inaccurate": ("results/2026-09-18_amend2", 828)}
    cache = {}
    for sc, (res, cell) in stores.items():
        if res not in cache:
            cache[res] = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / res / "E1_*.parquet")))])
        d = cache[res]
        d = d[(d.cell == cell) & (d.estimand == "T1")]
        for axis in ("AP", "SL"):
            for rule, s, f in (("A1", None, None), ("A3", None, None), ("A4", 0.8, 0.1), ("A4", 0.0, 0.0), ("A4", 1.0, 0.3)):
                q = d[(d.estimator == rule) & (d.axis == axis)]
                w = W[(W.run == "reproduce") & (W.scenario == sc) & (W.axis == axis) & (W.rule == rule)]
                if rule == "A4":
                    q = q[(q.s_det == s) & (q.f_rej == f)]
                    w = w[(w.s_det == s) & (w.f_rej == f)]
                w = w.iloc[0]
                for m in ("bias_mm", "rmse_mm"):
                    sv = float(q[q.metric == m].value.iloc[0])
                    rows.append(dict(scenario=sc, stored_results=res, experiment="E1", cell=cell, axis=axis, rule=rule,
                                     s_det=s, f_rej=f, metric=m, stored_value=sv, rerun_value=w[m],
                                     abs_diff=abs(sv - w[m])))
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "results/2026-09-18_full/E3_*.parquet")))])
    d = d[(d.cell == 29) & (d.cutoff_mm == 13.0)]
    for axis in ("AP", "SL"):
        for rule, s, f in (("A1", None, None), ("A3", None, None), ("A4", 0.8, 0.1)):
            w = W[(W.run == "reproduce") & (W.scenario == "base_E3") & (W.axis == axis) & (W.rule == rule)]
            if rule == "A4":
                w = w[(w.s_det == s) & (w.f_rej == f)]
            w = w.iloc[0]
            for m, col in (("sensitivity", "sensitivity_13mm"), ("specificity", "specificity_13mm")):
                sv = float(d[(d.estimator == rule) & (d.axis == axis) & (d.metric == m)].value.iloc[0])
                rows.append(dict(scenario="base", stored_results="results/2026-09-18_full", experiment="E3", cell=29,
                                 axis=axis, rule=rule, s_det=s, f_rej=f, metric=col, stored_value=sv, rerun_value=w[col],
                                 abs_diff=abs(sv - w[col])))
    pd.DataFrame(rows).to_csv(out / "review_reproduction.csv", index=False)


# =========================================================================== axes

def axes_table(raw: Path, out: Path):
    c = pd.read_csv(raw / "E_axis_counts.csv")
    s = pd.read_csv(raw / "E_axis_summary.csv")
    c["value"] = c.k / c.n_patients
    c["mcse"] = np.sqrt(c.value * (1 - c.value) / c.n_patients)
    keep = ["scenario", "quantity", "tau_mm", "k", "n_patients", "value", "mcse", "analytic"]
    c[keep].to_csv(out / "axes_long.csv", index=False)
    rows = []
    for _, r in s.iterrows():
        q = c[c.scenario == r.scenario]
        row = {k: r[k] for k in ("scenario", "family", "target_mean_diff_mm", "variability", "r_mean", "r_lower", "r_upper",
                                 "r_concentration", "mu_log", "sd_log", "value", "n_patients", "mean_true_diff_mm",
                                 "mean_true_diff_mcse", "mean_abs_true_diff_mm", "mean_abs_true_diff_mcse",
                                 "mean_abs_anchor_measured_diff_mm", "mean_abs_anchor_measured_diff_mcse",
                                 "mean_ratio", "sd_ratio", "n_sl_gt_ap", "corr_log", "corr_mm")}
        row["constant_ratio"] = row.pop("value")

        def g(quantity, tau):
            x = q[(q.quantity == quantity) & (q.tau_mm == tau)].iloc[0]
            return x

        for tau in (3.0, 5.0):
            for lab, quantity in (("true", "true"), ("measured_anchor", "measured:A1_anchor"),
                                  ("measured_largest", "measured:A3_largest"), ("measured_reviewed", "measured:A4_reviewed")):
                x = g(quantity, tau)
                row[f"{lab}_ge{tau:g}_k"] = int(x.k)
                row[f"{lab}_ge{tau:g}"] = x.value
                row[f"{lab}_ge{tau:g}_mcse"] = x.mcse
                if lab == "true":
                    row[f"true_ge{tau:g}_analytic"] = x.analytic
            for axis in ("AP", "SL"):
                x = g(f"any_fired:{axis}", tau)
                row[f"patients_any_trigger_{axis}_{tau:g}mm_k"] = int(x.k)
                row[f"patients_any_trigger_{axis}_{tau:g}mm"] = x.value
                row[f"patients_any_trigger_{axis}_{tau:g}mm_mcse"] = x.mcse
                tp, fp, fn, tn = (int(g(f"pair_{k}:{axis}", tau).k) for k in ("tp", "fp", "fn", "tn"))
                rr = T.rates_2x2(tp, fp, fn, tn)
                for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
                    row[f"pair_{k}_{axis}_{tau:g}mm"] = v
                for m in ("prevalence", "sensitivity", "false_positive_rate", "ppv"):
                    row[f"pair_{m}_{axis}_{tau:g}mm"] = rr[m][0]
                    row[f"pair_{m}_{axis}_{tau:g}mm_se_iid"] = rr[m][1]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "axes_scenarios.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "2026-10-05_osrev" / "triggers"))
    a = ap.parse_args()
    out = Path(a.dir)
    raw = out / "raw"
    rates, counts, reps = pair_tables(raw, out)
    consistency_table(rates, counts, out)
    interval_table(rates, out)
    patient = patient_table(raw, out)
    partition_tables(raw, out)
    definition_summary(rates, counts, patient, out)
    trigger_reproduction(rates, patient, raw, out)
    W = review_tables(raw, out)
    review_reproduction(W, out)
    axes_table(raw, out)
    print("wrote", sorted(p.name for p in out.glob("*.csv")))


if __name__ == "__main__":
    main()
