#!/usr/bin/env python
"""Protocol amendment 3, package A3-1 (beat rules): derived tables.

Reads the csv files written by code/20_osrev_beats_run.py and writes the tables cited in
notes/scratch/2026-10-05-findings-beats.md (prefix beats_table_). No simulation here.

  python code/21_osrev_beats_tables.py [--dir results/2026-10-05_osrev/beats]

Conventions: MCSE = Monte Carlo standard error; 95% Monte Carlo interval = estimate +- 1.96 MCSE.
Contrasts within a condition are paired (common random numbers). Contrasts between conditions
(for example a sensitivity level against the reference) come from independently seeded
conditions, so their MCSE is sqrt(mcse_1^2 + mcse_2^2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
Z = 1.959963984540054


def ci(df, est, se, name):
    df[name + "_lo95"] = df[est] - Z * df[se]
    df[name + "_hi95"] = df[est] + Z * df[se]
    return df


def equal_budget(arms, pairs, rules):
    """One row per condition, axis, estimand, N and budget B."""
    a = arms[arms["grid"] == "budget"]
    p = pairs[pairs["grid"] == "budget"]
    r = rules[rules["grid"] == "budget"]
    keys = ["condition", "rhythm", "beat_cv", "axis", "estimand"]
    m = ["bias_mm", "bias_mcse", "rmse_mm", "rmse_mcse", "mae_mm", "mae_mcse"]
    rows = []
    for _, rr in r.iterrows():
        N, B = int(rr["N"]), int(rr["budget"])
        for tn in ("T1", "Tv"):
            sel = dict(condition=rr["condition"], axis=rr["axis"], estimand=tn)

            def arm(name):
                s = a[(a["condition"] == sel["condition"]) & (a["axis"] == sel["axis"])
                      & (a["estimand"] == tn) & (a["arm"] == name)]
                assert len(s) == 1, (name, sel, len(s))
                return s.iloc[0]

            def pair(x, y):
                if x == y:
                    return None
                s = p[(p["condition"] == sel["condition"]) & (p["axis"] == sel["axis"])
                      & (p["estimand"] == tn) & (p["arm_a"] == x) & (p["arm_b"] == y)]
                assert len(s) == 1, (x, y, sel, len(s))
                return s.iloc[0]

            w, pb, pn = arm(f"win{N}_b{B}"), arm(f"plain_{B}"), arm(f"plain_{N}")
            row = dict(condition=rr["condition"], rhythm=rr["rhythm"], beat_cv=rr["beat_cv"], axis=rr["axis"],
                       estimand=tn, N=N, W=rr["W"], budget=B, n_views=int(rr["n_views"]),
                       n_met_first_N=int(rr["n_met_first_N"]), n_accepted_within_budget=int(rr["n_accepted"]),
                       n_limited=int(rr["n_limited"]), mean_beats_acquired_window=rr["mean_beats_acquired"],
                       mcse_beats_acquired_window=rr["mcse_beats_acquired"])
            for lab, s in (("window", w), ("plain_all_B", pb), ("plain_first_N", pn)):
                for k in m:
                    row[f"{lab}_{k}"] = s[k]
            for lab, (x, y) in (("window_minus_plain_all_B", (f"win{N}_b{B}", f"plain_{B}")),
                                ("window_minus_plain_first_N", (f"win{N}_b{B}", f"plain_{N}")),
                                ("plain_all_B_minus_plain_first_N", (f"plain_{B}", f"plain_{N}"))):
                s = pair(x, y)
                for k in ("d_bias_mm", "d_bias_mcse", "d_rmse_mm", "d_rmse_mcse", "d_mae_mm", "d_mae_mcse",
                          "d_relbias_pct", "d_relbias_mcse"):
                    row[f"{lab}_{k}"] = 0.0 if s is None else s[k]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(keys[:1] + ["axis", "estimand", "N", "budget"]).reset_index(drop=True)


def equal_expected(arms, pairs, rules):
    """Window rule at the 30-beat cap against plain means with the same number of beats."""
    rows = []
    for (g, c, ax, tn), a in arms.groupby(["grid", "condition", "axis", "estimand"]):
        if g not in ("budget", "sens", "check_K3"):
            continue
        p = pairs[(pairs["grid"] == g) & (pairs["condition"] == c) & (pairs["axis"] == ax) & (pairs["estimand"] == tn)]
        r = rules[(rules["grid"] == g) & (rules["condition"] == c) & (rules["axis"] == ax)]
        A = a.set_index("arm")
        for N in (3, 5):
            top = f"win{N}_b30"
            ru = r[r["arm"] == top].iloc[0]
            mu = ru["mean_beats_acquired"]
            lo, hi = int(np.floor(mu)), int(np.ceil(mu))
            row = dict(grid=g, condition=c, axis=ax, estimand=tn, N=N, n_views=int(ru["n_views"]),
                       mean_beats_acquired_window=mu, mcse_beats_acquired_window=ru["mcse_beats_acquired"],
                       window_rmse_mm=A.loc[top, "rmse_mm"], window_rmse_mcse=A.loc[top, "rmse_mcse"],
                       window_mae_mm=A.loc[top, "mae_mm"], window_bias_mm=A.loc[top, "bias_mm"])
            for b in sorted({lo, hi}):
                row[f"plain_floor_or_ceil_{'lo' if b == lo else 'hi'}_n"] = b
                row[f"plain_floor_or_ceil_{'lo' if b == lo else 'hi'}_rmse_mm"] = A.loc[f"plain_{b}", "rmse_mm"]
            for lab, (x, y) in (("selection_window_minus_same_acquired", (top, f"sameacq{N}")),
                                ("acquisition_same_acquired_minus_plain_first_N", (f"sameacq{N}", f"plain_{N}")),
                                ("window_minus_plain_same_count_distribution", (top, f"permacq{N}")),
                                ("plain_same_count_distribution_minus_plain_first_N", (f"permacq{N}", f"plain_{N}")),
                                ("total_window_minus_plain_first_N", (top, f"plain_{N}"))):
                s = p[(p["arm_a"] == x) & (p["arm_b"] == y)].iloc[0]
                for k in ("d_bias_mm", "d_bias_mcse", "d_rmse_mm", "d_rmse_mcse", "d_mae_mm", "d_mae_mcse",
                          "d_relbias_pct", "d_relbias_mcse"):
                    row[f"{lab}_{k}"] = s[k]
            rows.append(row)
    return pd.DataFrame(rows)


def published_cells(repro):
    """Published E2 values (stored parquet) in wide form, keyed by cell settings."""
    r = repro.copy()
    r["window"] = r["window"].fillna(0.0)
    return r


def rmse_difference(repro, boot, pairs):
    """Published independent-arm interval next to the paired intervals of amendment 3."""
    r = published_cells(repro)
    rows = []
    for rs, cond in (("sinus", "sinus_cv15"), ("AF_rr0.2", "af_cv15")):
        for ax in ("AP", "SL"):
            for tn in ("T1", "Tv"):
                for N in (3, 5):
                    s = r[(r["rhythm_state"] == rs) & (r["axis"] == ax) & (r["estimand"] == tn)
                          & (r["N_beats"] == N) & (r["data_mode"] == "prospective") & (r["estimator"] == "A1")
                          & (r["subset"] == "all") & (r["metric"] == "rmse_mm")]
                    w = s[s["window"] > 0].iloc[0]
                    nw = s[s["window"] == 0].iloc[0]
                    d = w["value_published"] - nw["value_published"]
                    se = float(np.hypot(w["mcse_published"], nw["mcse_published"]))
                    b = boot[(boot["condition"] == cond) & (boot["axis"] == ax) & (boot["estimand"] == tn)
                             & (boot["arm_a"] == f"win{N}_b30")].iloc[0]
                    rows.append(dict(
                        rhythm_state=rs, axis=ax, estimand=tn, N=N, W=0.15,
                        published_rmse_window_mm=w["value_published"], published_rmse_no_window_mm=nw["value_published"],
                        published_cells=f"{int(nw['cell'])},{int(w['cell'])}",
                        published_arms_share_random_numbers=False,
                        published_diff_mm=d, published_mcse_indep=se,
                        published_lo95=d - Z * se, published_hi95=d + Z * se,
                        a3_condition=cond, a3_n=int(b["n"]), a3_diff_mm=b["d_rmse_mm"],
                        a3_mcse_paired_delta=b["mcse_paired_delta"],
                        a3_paired_lo95=b["d_rmse_mm"] - Z * b["mcse_paired_delta"],
                        a3_paired_hi95=b["d_rmse_mm"] + Z * b["mcse_paired_delta"],
                        a3_boot_n=int(b["n_boot"]), a3_boot_se=b["boot_se"],
                        a3_boot_lo95=b["boot_lo95"], a3_boot_hi95=b["boot_hi95"],
                        a3_mcse_if_treated_as_independent=b["mcse_indep_formula"],
                        a3_corr_squared_errors=b["corr_sq_err"],
                        a3_minus_published_mm=b["d_rmse_mm"] - d,
                        a3_minus_published_z=(b["d_rmse_mm"] - d) / np.hypot(se, b["mcse_paired_delta"])))
    return pd.DataFrame(rows)


def sensitivity(arms, pairs, rules):
    a = arms[arms["grid"] == "sens"]
    p = pairs[pairs["grid"] == "sens"]
    r = rules[rules["grid"] == "sens"]
    plain_n = [1, 2, 3, 4, 5, 6, 7, 8, 10, 13]
    rows = []
    desc = ["condition", "family", "rhythm", "beat_ar1_rho", "resp_amp", "resp_mode", "cal_noise_mode",
            "total_beat_cv"]
    for (c, ax), ac in a.groupby(["condition", "axis"]):
        d0 = ac.iloc[0]
        row = {k: d0[k] for k in desc}
        row["axis"] = ax
        rc = r[(r["condition"] == c) & (r["axis"] == ax)].set_index("arm")
        row["n_views"] = int(rc.loc["win3_b30", "n_views"])
        row["mean_caliper_sd_mm"] = rc.loc["win3_b30", "mean_caliper_sd_mm"]
        for N in (3, 5):
            top = rc.loc[f"win{N}_b30"]
            n = top["n_views"]
            k = top["n_met_first_N"]
            row[f"n_met_first_{N}"] = int(k)
            row[f"pct_met_first_{N}"] = 100.0 * k / n
            row[f"pct_met_first_{N}_mcse"] = 100.0 * np.sqrt(k / n * (1 - k / n) / n)
            row[f"mean_beats_acquired_N{N}"] = top["mean_beats_acquired"]
            row[f"mean_beats_acquired_N{N}_mcse"] = top["mcse_beats_acquired"]
            row[f"n_limited_at_30_N{N}"] = int(top["n_limited"])
        for tn in ("T1", "Tv"):
            A = ac[ac["estimand"] == tn].set_index("arm")
            for k in plain_n:
                row[f"{tn}_plain_{k}_rmse_mm"] = A.loc[f"plain_{k}", "rmse_mm"]
                row[f"{tn}_plain_{k}_rmse_mcse"] = A.loc[f"plain_{k}", "rmse_mcse"]
            pc = p[(p["condition"] == c) & (p["axis"] == ax) & (p["estimand"] == tn)]
            for N in (3, 5):
                s = pc[(pc["arm_a"] == f"win{N}_b30") & (pc["arm_b"] == f"plain_{N}")].iloc[0]
                row[f"{tn}_win{N}_rmse_mm"] = A.loc[f"win{N}_b30", "rmse_mm"]
                for k in ("d_rmse_mm", "d_rmse_mcse", "d_mae_mm", "d_mae_mcse", "d_relbias_pct", "d_relbias_mcse",
                          "d_bias_mm", "d_bias_mcse"):
                    row[f"{tn}_win{N}_minus_plain{N}_{k}"] = s[k]
                # benefit of averaging: paired RMSE change from N beats to 13 beats and from 1 to N... (independent MCSE not needed)
            for k in (3, 5, 13):
                # RMSE reduction from one beat to k beats (same patients); MCSE from the paired table is not
                # available for plain_1, so the conservative independent formula is used
                row[f"{tn}_rmse_gain_1_to_{k}_mm"] = A.loc["plain_1", "rmse_mm"] - A.loc[f"plain_{k}", "rmse_mm"]
                row[f"{tn}_rmse_gain_1_to_{k}_mcse_conservative"] = float(
                    np.hypot(A.loc["plain_1", "rmse_mcse"], A.loc[f"plain_{k}", "rmse_mcse"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    # contrasts with the reference condition of the same rhythm (independent seeds)
    add = []
    for i, row in out.iterrows():
        rh = row["condition"].split("_")[0]
        ref = out[(out["condition"] == f"{rh}_ref") & (out["axis"] == row["axis"])].iloc[0]
        d = {}
        for col, se in (("T1_win3_minus_plain3_d_rmse_mm", "T1_win3_minus_plain3_d_rmse_mcse"),
                        ("Tv_win3_minus_plain3_d_rmse_mm", "Tv_win3_minus_plain3_d_rmse_mcse"),
                        ("Tv_win3_minus_plain3_d_relbias_pct", "Tv_win3_minus_plain3_d_relbias_mcse"),
                        ("pct_met_first_3", "pct_met_first_3_mcse"), ("pct_met_first_5", "pct_met_first_5_mcse"),
                        ("T1_rmse_gain_1_to_3_mm", "T1_rmse_gain_1_to_3_mcse_conservative"),
                        ("Tv_rmse_gain_1_to_3_mm", "Tv_rmse_gain_1_to_3_mcse_conservative")):
            d[col + "_vs_ref"] = row[col] - ref[col]
            d[col + "_vs_ref_mcse"] = float(np.hypot(row[se], ref[se]))
        add.append(d)
    return pd.concat([out, pd.DataFrame(add)], axis=1)


def rmse_by_n(arms):
    """Long table: RMSE, MAE and bias of the plain mean of the first n beats (and ratio to one beat)."""
    a = arms[arms["grid"].isin(["budget", "sens"]) & arms["arm"].str.startswith("plain_")].copy()
    a["n_beats"] = a["arm"].str.replace("plain_", "").astype(int)
    one = a[a["n_beats"] == 1][["grid", "condition", "axis", "estimand", "rmse_mm"]].rename(
        columns={"rmse_mm": "rmse_one_beat_mm"})
    a = a.merge(one, on=["grid", "condition", "axis", "estimand"])
    a["rmse_ratio_to_one_beat"] = a["rmse_mm"] / a["rmse_one_beat_mm"]
    cols = ["grid", "condition", "family", "rhythm", "beat_cv", "beat_ar1_rho", "resp_amp", "resp_mode",
            "cal_noise_mode", "total_beat_cv", "axis", "estimand", "n_beats", "n", "bias_mm", "bias_mcse",
            "rmse_mm", "rmse_mcse", "mae_mm", "mae_mcse", "relbias_pct", "relbias_mcse", "rmse_ratio_to_one_beat"]
    return a[cols].sort_values(["grid", "condition", "axis", "estimand", "n_beats"]).reset_index(drop=True)


def limited_denominators(repro, retro):
    """Limited-sampling counts: published cells (count = stored proportion x n) and amendment 3."""
    r = published_cells(repro)
    s = r[(r["metric"] == "p_limited") & (r["data_mode"] == "retrospective")]
    rows = []
    for _, x in s.iterrows():
        n = int(x["n_published"])
        k = x["value_published"] * n
        assert abs(k - round(k)) < 1e-6, k
        rows.append(dict(source="published E2 cell (seed entropy 20260918)", cell=int(x["cell"]),
                         rhythm_state=x["rhythm_state"], axis=x["axis"], N=int(x["N_beats"]),
                         window=("none" if x["window"] == 0 else f"+-{x['window']:.2f}"),
                         n_limited=int(round(k)), n_views=n, pct_limited=100.0 * k / n,
                         mcse_pct=100.0 * x["mcse_published"]))
    for _, x in retro.iterrows():
        for win, col in (("none", "n_limited_no_window"), (f"+-{x['W']:.2f}", "n_limited_window")):
            k, n = int(x[col]), int(x["n_views"])
            rows.append(dict(source="amendment 3 (seed entropy 20261005)", cell=np.nan,
                             rhythm_state=x["condition"], axis=x["axis"], N=int(x["N"]), window=win,
                             n_limited=k, n_views=n, pct_limited=100.0 * k / n,
                             mcse_pct=100.0 * np.sqrt(k / n * (1 - k / n) / n),
                             n_fewer_than_N_available=int(x["n_available_lt_N"]),
                             n_enough_beats_window_not_met=(int(x["n_limited_window_available_ge_N"])
                                                            if win != "none" else 0),
                             n_views_with_at_least_N_available=int(x["n_views_available_ge_N"])))
    return pd.DataFrame(rows)


def published_check(repro, arms, pairs, rules):
    """Manuscript beat-rule numbers: stored value, bit-identical re-run, and the amendment-3
    estimate from new seeds (z = difference / combined MCSE)."""
    r = published_cells(repro)
    rows = []

    def pub(rs, N, w, mode, metric, est="rule", subset=None, estimand=None, axis="AP"):
        s = r[(r["rhythm_state"] == rs) & (r["N_beats"] == N) & (np.isclose(r["window"], w))
              & (r["data_mode"] == mode) & (r["metric"] == metric) & (r["estimator"] == est) & (r["axis"] == axis)]
        if subset is not None:
            s = s[s["subset"] == subset]
        if estimand is not None:
            s = s[s["estimand"] == estimand]
        assert len(s) == 1, (rs, N, w, mode, metric, len(s))
        return s.iloc[0]

    def add(label, ms_value, x, new=None, new_se=None, scale=1.0):
        row = dict(quantity=label, manuscript_value=ms_value, stored_value=scale * x["value_published"],
                   stored_mcse=scale * x["mcse_published"], n=int(x["n_published"]), cell=int(x["cell"]),
                   rerun_value=scale * x["value_rerun"], rerun_identical=bool(x["identical"]),
                   a3_value=new, a3_mcse=new_se)
        if new is not None:
            row["a3_minus_stored_z"] = (new - row["stored_value"]) / np.hypot(new_se, row["stored_mcse"])
        rows.append(row)

    def rule(cond, arm, col):
        s = rules[(rules["condition"] == cond) & (rules["axis"] == "AP") & (rules["arm"] == arm)].iloc[0]
        if col == "met":
            pr = s["n_met_first_N"] / s["n_views"]
            return 100 * pr, 100 * np.sqrt(pr * (1 - pr) / s["n_views"])
        return s["mean_beats_acquired"], s["mcse_beats_acquired"]

    def arm(cond, name, tn, col):
        s = arms[(arms["condition"] == cond) & (arms["axis"] == "AP") & (arms["arm"] == name)
                 & (arms["estimand"] == tn) & (arms["grid"] == "budget")].iloc[0]
        return s[col + ("_mm" if col in ("rmse", "bias", "mae") else "_pct")], s[col + "_mcse"]

    for rs, cond, m3, m5 in (("sinus", "sinus_cv15", "41.8", "15.2"), ("AF_rr0.2", "af_cv15", "37.9", "12.6")):
        add(f"{rs}: first 3 beats met +-15% (%)", m3, pub(rs, 3, 0.15, "prospective", "p_window_met_first_N"),
            *rule(cond, "win3_b30", "met"), scale=100.0)
        add(f"{rs}: first 5 beats met +-15% (%)", m5, pub(rs, 5, 0.15, "prospective", "p_window_met_first_N"),
            *rule(cond, "win5_b30", "met"), scale=100.0)
    add("sinus: mean beats acquired, N 3", "3.90",
        pub("sinus", 3, 0.15, "prospective", "beats_acquired_mean", subset="all"), *rule("sinus_cv15", "win3_b30", "used"))
    add("sinus: mean beats acquired, N 5", "7.12",
        pub("sinus", 5, 0.15, "prospective", "beats_acquired_mean", subset="all"), *rule("sinus_cv15", "win5_b30", "used"))
    add("sinus: RMSE vs T1, 3 beats, no window (mm)", "1.94",
        pub("sinus", 3, 0.0, "prospective", "rmse_mm", "A1", "all", "T1"), *arm("sinus_cv15", "plain_3", "T1", "rmse"))
    add("sinus: RMSE vs T1, 3 beats, +-15% (mm)", "2.04",
        pub("sinus", 3, 0.15, "prospective", "rmse_mm", "A1", "all", "T1"), *arm("sinus_cv15", "win3_b30", "T1", "rmse"))
    add("AF: RMSE vs T1, 3 beats, no window (mm)", "",
        pub("AF_rr0.2", 3, 0.0, "prospective", "rmse_mm", "A1", "all", "T1"), *arm("af_cv15", "plain_3", "T1", "rmse"))
    add("AF: RMSE vs T1, 3 beats, +-15% (mm)", "",
        pub("AF_rr0.2", 3, 0.15, "prospective", "rmse_mm", "A1", "all", "T1"), *arm("af_cv15", "win3_b30", "T1", "rmse"))
    add("sinus: relative bias vs Tv, 3 beats, no window (%)", "",
        pub("sinus", 3, 0.0, "prospective", "relbias_pct", "A1", "all", "Tv"), *arm("sinus_cv15", "plain_3", "Tv", "relbias"))
    add("sinus: relative bias vs Tv, 3 beats, +-15% (%)", "",
        pub("sinus", 3, 0.15, "prospective", "relbias_pct", "A1", "all", "Tv"), *arm("sinus_cv15", "win3_b30", "Tv", "relbias"))
    for N, w, ms in ((3, 0.0, "22.4"), (3, 0.15, "40.2"), (5, 0.15, "84.7"), (5, 0.0, "")):
        add(f"sinus retrospective: limited sampling, N {N}, window {w} (%)", ms,
            pub("sinus", N, w, "retrospective", "p_limited"), scale=100.0)
    out = pd.DataFrame(rows)
    # derived published contrasts
    def contrast(label, ms, a, b, new, new_se):
        d = a["value_published"] - b["value_published"]
        se = float(np.hypot(a["mcse_published"], b["mcse_published"]))
        return dict(quantity=label, manuscript_value=ms, stored_value=d, stored_mcse=se, n=int(a["n_published"]),
                    cell=np.nan, rerun_value=a["value_rerun"] - b["value_rerun"],
                    rerun_identical=bool(a["identical"] and b["identical"]), a3_value=new, a3_mcse=new_se,
                    a3_minus_stored_z=(new - d) / np.hypot(new_se, se))

    def pr(cond, tn, col):
        s = pairs[(pairs["grid"] == "budget") & (pairs["condition"] == cond) & (pairs["axis"] == "AP")
                  & (pairs["estimand"] == tn) & (pairs["arm_a"] == "win3_b30") & (pairs["arm_b"] == "plain_3")].iloc[0]
        return s[col], s[col.replace("_mm", "").replace("_pct", "") + "_mcse"]

    extra = [
        contrast("sinus: RMSE difference vs T1, window minus none (mm)", "0.10 (0.08 to 0.12)",
                 pub("sinus", 3, 0.15, "prospective", "rmse_mm", "A1", "all", "T1"),
                 pub("sinus", 3, 0.0, "prospective", "rmse_mm", "A1", "all", "T1"), *pr("sinus_cv15", "T1", "d_rmse_mm")),
        contrast("AF: RMSE difference vs T1, window minus none (mm)", "0.12",
                 pub("AF_rr0.2", 3, 0.15, "prospective", "rmse_mm", "A1", "all", "T1"),
                 pub("AF_rr0.2", 3, 0.0, "prospective", "rmse_mm", "A1", "all", "T1"), *pr("af_cv15", "T1", "d_rmse_mm")),
        contrast("sinus: upward shift of relative bias vs Tv (% points)", "0.71 (0.58 to 0.85)",
                 pub("sinus", 3, 0.15, "prospective", "relbias_pct", "A1", "all", "Tv"),
                 pub("sinus", 3, 0.0, "prospective", "relbias_pct", "A1", "all", "Tv"), *pr("sinus_cv15", "Tv", "d_relbias_pct")),
    ]
    return pd.concat([out, pd.DataFrame(extra)], ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "results" / "2026-10-05_osrev" / "beats"))
    a = ap.parse_args(argv)
    d = Path(a.dir)
    arms = pd.read_csv(d / "beats_arm_errors.csv")
    pairs = pd.read_csv(d / "beats_paired_contrasts.csv")
    rules = pd.read_csv(d / "beats_window_rule_counts.csv")
    boot = pd.read_csv(d / "beats_rmse_difference_intervals.csv")
    retro = pd.read_csv(d / "beats_retro_limited_denominators.csv")
    repro = pd.read_csv(d / "beats_reproduction_E2_cells.csv")
    out = {
        "beats_table_equal_budget.csv": equal_budget(arms, pairs, rules),
        "beats_table_equal_expected_beats.csv": equal_expected(arms, pairs, rules),
        "beats_table_rmse_difference_published_vs_paired.csv": rmse_difference(repro, boot, pairs),
        "beats_table_sensitivity.csv": sensitivity(arms, pairs, rules),
        "beats_table_rmse_by_n_beats.csv": rmse_by_n(arms),
        "beats_table_limited_denominators.csv": limited_denominators(repro, retro),
        "beats_table_published_numbers_check.csv": published_check(repro, arms, pairs, rules),
    }
    for name, df in out.items():
        df.to_csv(d / name, index=False)
        print(f"wrote {name} rows={len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
