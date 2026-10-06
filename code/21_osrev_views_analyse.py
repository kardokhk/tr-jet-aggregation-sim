#!/usr/bin/env python
"""Amendment 3, package A3-2 (view rules): analysis tables.

Input : results/2026-10-05_osrev/views/sim_{metrics,jack,decomp,cond}.parquet
        (code/20_osrev_views_run.py, Slurm job code/slurm/osrev_views.sh) and, read only, the
        stored E1 parquet files of results/2026-09-18_amend2 and results/2026-09-18_full.
Output: results/2026-10-05_osrev/views/views_*.csv (see README.md there).

Conventions. Error = measured minus true span T1 on the stated axis (AP = true maximal span).
Rules A1 anchor-view mean, A2 mean across views, A3 largest view mean, A4 largest view mean
after review (s_det 0.8, f_rej 0.1, 3 and 5 mm thresholds). Scenario = one of the 12 view-accuracy
combinations (mean anchor underestimation 2.4, 4.8, 10, 20% x mean long-axis underestimation 8,
20, 40%). Condition = one of the 72 combinations of K (2, 3, 4), beat CV (5, 15, 30%),
overestimation (off, on), window (none, +-15%) and median true AP span (10, 13 mm); each
condition holds all 12 scenarios (864 cells). 95% MCI = estimate +- 1.96 MCSE.

Decision criteria (per condition and axis, candidates A1 to A4), M_r(s) = metric of rule r in
scenario s:
  worst-case absolute bias   argmin_r max_s |bias_r(s)|     (the "minimax" of the manuscript and
                                                             of code/08_analyse_amend2_e1e3.py)
  worst-case RMSE            argmin_r max_s RMSE_r(s)
  worst-case MAE             argmin_r max_s MAE_r(s)
  mean RMSE                  argmin_r mean_s RMSE_r(s)      (equal weight on the 12 scenarios)
  worst-case q95|e|          argmin_r max_s q95(|e|)_r(s)
  mean q95|e|                argmin_r mean_s q95(|e|)_r(s)
  worst-case RMSE regret     argmin_r max_s [RMSE_r(s) - min_r' RMSE_r'(s)]
Gap = runner-up minus winner. Gap MCSE: delete-a-group jackknife of the paired difference when
both worst cases fall in the same cell (rules share random numbers within a cell), otherwise
sqrt(mcse_w^2 + mcse_r^2) (cells are independently seeded); for mean criteria the per-scenario
paired jackknife SEs are combined as sqrt(sum se^2) / 12. "clear" = gap > 1.96 x gap MCSE. The
MCSE of a maximum is that of the cell attaining it and ignores uncertainty in which cell attains it.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "2026-10-05_osrev" / "views"
Z = 1.959964
RULES = ["A1", "A2", "A3", "A4"]
RULE_NAME = {"A1": "Anchor-view mean", "A2": "Mean across views", "A3": "Largest view mean",
             "A4": "Largest view mean after review"}
COND = ["S_median_mm", "K", "beat_cv", "view_over", "window"]
ANCHOR_PCT = {0.0: 0.0, 0.03: 2.4, 0.06: 4.8, 0.125: 10.0, 0.25: 20.0}
LONG_PCT = {0.0: 0.0, 0.10: 8.0, 0.25: 20.0, 0.50: 40.0}
BASE = dict(S_median_mm=10.0, K=3, beat_cv=0.15, view_over=True, window=0.0)
PARAMS = ["u_anchor", "u_long", "anchor_mean_pct", "long_mean_pct", "scenario", "K", "beat_cv", "view_over",
          "view_under", "window", "S_median_mm", "rhythm", "rr_cv_af"]
ID = ["family", "cell", "seed_entropy", "seed_spawn_key", "n_patients"]


def label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["window"] = df["window"].fillna(0.0)          # 0 = no window
    if "rr_cv_af" in df:
        df["rr_cv_af"] = df["rr_cv_af"].fillna(0.0)  # 0 = not applicable (sinus rhythm)
    df["anchor_mean_pct"] = df["u_anchor"].map(ANCHOR_PCT)
    df["long_mean_pct"] = df["u_long"].map(LONG_PCT)
    assert df.anchor_mean_pct.notna().all() and df.long_mean_pct.notna().all()
    df["scenario"] = df.anchor_mean_pct.map("{:g}".format) + "/" + df.long_mean_pct.map("{:g}".format)
    df.loc[~df.view_under.astype(bool), "scenario"] = "0/0"
    return df


def is_base(df, **change):
    c = {**BASE, **change}
    m = np.ones(len(df), bool)
    for k, v in c.items():
        m &= np.isclose(df[k].astype(float), float(v))
    return m


def wide(d: pd.DataFrame, idx: list[str]) -> pd.DataFrame:
    v = d.pivot(index=idx, columns="metric", values="value")
    s = d.pivot(index=idx, columns="metric", values="mcse")
    s.columns = [c + "_mcse" for c in s.columns]
    c = d[d.metric.str.startswith("p_abs")].pivot(index=idx, columns="metric", values="count")
    c.columns = [c_.replace("p_abs", "count_abs") for c_ in c.columns]
    n = d.groupby(idx, dropna=False)["n"].first().rename("n")
    w = pd.concat([v, s, c, n], axis=1)
    order = []
    for m in d.metric.unique():
        order += [m, m + "_mcse"] + ([m.replace("p_abs", "count_abs")] if m.startswith("p_abs") else [])
    assert len(w) == len(v) == len(n)
    return w[order + ["n"]].reset_index()


def stored(res: str) -> pd.DataFrame:
    st = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "results" / res / "E1_cells_*.parquet")))],
                   ignore_index=True)
    st = st[(st.estimand == "T1") | st.estimand.isna()]
    st = st[(st.estimator != "A4") | ((st.s_det == 0.8) & (st.f_rej == 0.1))]
    return st.assign(rule=st.estimator, metric="acc_" + st.metric)[["cell", "rule", "axis", "metric", "value", "mcse"]]


def jack_se(a: np.ndarray) -> float:
    G = a.size
    return float(np.sqrt((G - 1) / G * ((a - a.mean()) ** 2).sum()))


def main():
    met = label(pd.read_parquet(OUT / "sim_metrics.parquet"))
    jack = pd.read_parquet(OUT / "sim_jack.parquet")
    dec = label(pd.read_parquet(OUT / "sim_decomp.parquet"))
    cond = label(pd.read_parquet(OUT / "sim_cond.parquet"))

    # ------------------------------------------------------------ 0. exact reproduction of stored cells
    rep = []
    for fam, res in (("repro_e1b", "2026-09-18_amend2"), ("repro_main", "2026-09-18_full")):
        a = met[(met.family == fam) & met.metric.str.startswith("acc_")]
        j = a.merge(stored(res), on=["cell", "rule", "axis", "metric"], suffixes=("", "_stored"), how="left")
        for mname, g in j.groupby("metric"):
            has = g.value_stored.notna()
            rep.append(dict(family=fam, stored_run=res, metric=mname.replace("acc_", ""),
                            n_values_resimulated=len(g), n_values_in_stored_run=int(has.sum()),
                            n_value_bit_identical=int((g.value[has] == g.value_stored[has]).sum()),
                            n_mcse_bit_identical=int((g.mcse[has] == g.mcse_stored[has]).sum()),
                            max_abs_diff_value=float((g.value[has] - g.value_stored[has]).abs().max()) if has.any() else np.nan))
    # per-patient summaries (this analysis) against the library accumulators: floating-point only
    pp = met[met.family.isin(["repro_e1b", "repro_main"]) & met.metric.isin(["bias_mm", "rmse_mm", "sd_err_mm"])]
    ac = met[met.metric.isin(["acc_bias_mm", "acc_rmse_mm", "acc_sd_err_mm"])].assign(
        metric=lambda d: d.metric.str.replace("acc_", ""))
    j = pp.merge(ac[["family", "cell", "rule", "axis", "metric", "value"]], on=["family", "cell", "rule", "axis", "metric"],
                 suffixes=("", "_acc"))
    for (fam, mname), g in j.groupby(["family", "metric"]):
        rep.append(dict(family=fam, stored_run="per-patient arrays vs library accumulator", metric=mname,
                        n_values_resimulated=len(g), n_values_in_stored_run=len(g),
                        n_value_bit_identical=int((g.value == g.value_acc).sum()), n_mcse_bit_identical=np.nan,
                        max_abs_diff_value=float((g.value - g.value_acc).abs().max())))
    rep = pd.DataFrame(rep)
    rep.to_csv(OUT / "views_repro_check.csv", index=False)
    print(rep.to_string())

    # ------------------------------------------------------------ (a) all cells, wide
    core = met[(met.arm == "full") & met.rule.isin(RULES) & ~met.metric.str.startswith(("acc_", "p_sel"))
               & met.family.isin(["repro_e1b", "repro_main", "zero_under"])]
    idx = ID + PARAMS + ["rule", "axis"]
    W = wide(core, idx)
    W.insert(W.columns.get_loc("rule") + 1, "rule_name", W.rule.map(RULE_NAME))
    W.to_csv(OUT / "views_a_all_cells_wide.csv", index=False)
    e1b = W[W.family == "repro_e1b"]
    zu = W[W.family == "zero_under"]
    s12 = e1b[is_base(e1b)].sort_values(["axis", "u_anchor", "u_long", "rule"])
    assert len(s12) == 12 * 4 * 2
    s12.to_csv(OUT / "views_a_12scenarios.csv", index=False)

    variants = [("base family (K 3, beat CV 15%, overestimation on, no window, median span 10 mm)", {}),
                ("overestimation off", {"view_over": False}), ("beat CV 30%", {"beat_cv": 0.30}),
                ("beat CV 5%", {"beat_cv": 0.05}), ("two views", {"K": 2}), ("four views", {"K": 4}),
                ("window +-15%", {"window": 0.15}), ("median span 13 mm", {"S_median_mm": 13.0})]
    add, rng = [], []
    mcols = ["bias_mm", "sd_err_mm", "rmse_mm", "mae_mm", "q2.5_err_mm", "q97.5_err_mm", "q95_abs_mm",
             "p_abs_gt1mm", "p_abs_gt2mm", "p_abs_gt3mm", "ratio_q95abs_rmse"]
    for lab, ch in variants:
        g = e1b[is_base(e1b, **ch)]
        assert len(g) == 96
        add.append(g.assign(variant=lab, view_accuracy_set="12 scenarios"))
        for (ax, r), h in g.groupby(["axis", "rule"]):
            row = dict(variant=lab, view_accuracy_set="12 scenarios", axis=ax, rule=r, rule_name=RULE_NAME[r], n_cells=len(h))
            for m in mcols:
                i0, i1 = h[m].idxmin(), h[m].idxmax()
                row.update({f"{m}_min": h.at[i0, m], f"{m}_min_mcse": h.at[i0, m + "_mcse"], f"{m}_min_scenario": h.at[i0, "scenario"],
                            f"{m}_max": h.at[i1, m], f"{m}_max_mcse": h.at[i1, m + "_mcse"], f"{m}_max_scenario": h.at[i1, "scenario"]})
            rng.append(row)
    for vo in (True, False):
        for lab, ch in (("", {}), (", beat CV 30%", {"beat_cv": 0.30}), (", beat CV 5%", {"beat_cv": 0.05}),
                        (", two views", {"K": 2}), (", four views", {"K": 4}), (", window +-15%", {"window": 0.15})):
            g = zu[is_base(zu, view_over=vo, **ch)]
            assert len(g) == 8
            add.append(g.assign(variant=f"zero underestimation in all views, overestimation {'on' if vo else 'off'}{lab}",
                                view_accuracy_set="single cell (new seed)"))
    add = pd.concat(add, ignore_index=True)
    add = add[["variant", "view_accuracy_set"] + [c for c in add.columns if c not in ("variant", "view_accuracy_set")]]
    add.to_csv(OUT / "views_a_added_scenarios.csv", index=False)
    pd.DataFrame(rng).to_csv(OUT / "views_a_variant_ranges.csv", index=False)

    # ------------------------------------------------------------ (b) decision analysis
    J = jack[jack.family.isin(["repro_e1b", "zero_under"]) & (jack.arm == "full")]
    J = {k: g.sort_values("group").value.to_numpy() for k, g in J.groupby(["family", "cell", "axis", "rule", "metric"])}

    def paired_se(fc, ax, m, r_w, r_r):
        a, b = J[(*fc, ax, r_r, m)], J[(*fc, ax, r_w, m)]
        if m == "bias_mm":
            a, b = np.abs(a), np.abs(b)
        return jack_se(a - b)

    OBJ = {"worst_abs_bias": ("bias_mm", "max"), "worst_rmse": ("rmse_mm", "max"), "worst_mae": ("mae_mm", "max"),
           "mean_rmse": ("rmse_mm", "mean"), "worst_q95abs": ("q95_abs_mm", "max"), "mean_q95abs": ("q95_abs_mm", "mean"),
           "worst_rmse_regret": ("rmse_mm", "regret")}

    def decide(g, kd, n_scen):
        """Worst-case rows and decision rows for one condition and axis (g: n_scen x 4 rule rows)."""
        assert len(g) == 4 * n_scen and g.scenario.nunique() == n_scen, (len(g), kd)
        wc_, dec_ = [], []
        best_rmse = g.groupby("scenario").rmse_mm.transform("min")
        g = g.assign(abs_bias=g.bias_mm.abs(), rmse_regret=g.rmse_mm - best_rmse)
        table = {}
        for obj, (m, how) in OBJ.items():
            for r, h in g.groupby("rule"):
                col = "abs_bias" if m == "bias_mm" else ("rmse_regret" if how == "regret" else m)
                if how in ("max", "regret"):
                    i = h[col].idxmax()
                    table[(obj, r)] = dict(value=h.at[i, col], mcse=h.at[i, m + "_mcse"],
                                           cell=(h.at[i, "family"], int(h.at[i, "cell"])), scenario=h.at[i, "scenario"])
                else:
                    table[(obj, r)] = dict(value=h[col].mean(), mcse=float(np.sqrt((h[m + "_mcse"] ** 2).sum()) / len(h)),
                                           cell=None, scenario=f"mean of {n_scen}")
        for r in RULES:
            row = dict(kd, n_scenarios=n_scen, rule=r, rule_name=RULE_NAME[r])
            for obj in OBJ:
                t = table[(obj, r)]
                row.update({obj: t["value"], obj + "_mcse": t["mcse"], obj + "_scenario": t["scenario"]})
            wc_.append(row)
        for obj, (m, how) in OBJ.items():
            rk = sorted(RULES, key=lambda r: table[(obj, r)]["value"])
            w, ru = rk[0], rk[1]
            tw, tr = table[(obj, w)], table[(obj, ru)]
            gap = tr["value"] - tw["value"]
            if how == "mean":
                cells = g[g.rule == w][["family", "cell"]].drop_duplicates().itertuples(index=False)
                ses = [paired_se((c.family, int(c.cell)), kd["axis"], m, w, ru) for c in cells]
                se = float(np.sqrt(sum(x * x for x in ses)) / len(ses))
                meth = "paired jackknife per scenario"
            elif how == "max" and tw["cell"] == tr["cell"]:
                se = paired_se(tw["cell"], kd["axis"], m, w, ru)
                meth = "paired jackknife (same cell)"
            else:
                se = float(np.hypot(tw["mcse"], tr["mcse"]))
                meth = "independent cells" if tw["cell"] != tr["cell"] else "independent (conservative)"
            dec_.append(dict(kd, n_scenarios=n_scen, objective=obj, winner=w, winner_value=tw["value"], winner_mcse=tw["mcse"],
                             winner_scenario=tw["scenario"], runnerup=ru, runnerup_value=tr["value"],
                             runnerup_scenario=tr["scenario"], gap=gap, gap_mcse=se, gap_mcse_method=meth,
                             clear=bool(gap > Z * se), ranking=" < ".join(rk),
                             **{f"value_{r}": table[(obj, r)]["value"] for r in RULES}))
        return wc_, dec_

    wc, decs, decs13 = [], [], []
    for key, g in e1b.groupby(COND + ["axis"]):
        kd = dict(zip(COND + ["axis"], key))
        w_, d_ = decide(g, kd, 12)
        wc += w_
        decs += d_
        # 13 scenarios: the 12 plus zero underestimation in all views (available at median span 10 mm)
        z13 = zu[(zu.axis == kd["axis"]) & is_base(zu, **{k: kd[k] for k in COND})]
        if len(z13):
            decs13 += decide(pd.concat([g, z13], ignore_index=True), kd, 13)[1]
    decs13 = pd.DataFrame(decs13)
    assert len(decs13) == 36 * 2 * len(OBJ)
    decs13.to_csv(OUT / "views_b_decision_13scenarios.csv", index=False)
    t13 = []
    for (ax, obj), h in decs13.groupby(["axis", "objective"]):
        row = dict(axis=ax, objective=obj, scope="36 conditions (median span 10 mm), 13 scenarios", n_conditions=len(h))
        for r in RULES:
            row[f"n_win_{r}"] = int((h.winner == r).sum())
            row[f"n_win_clear_{r}"] = int(((h.winner == r) & h.clear).sum())
        row["n_win_largest_view_rule"] = row["n_win_A3"] + row["n_win_A4"]
        row["n_not_clear"] = int((~h.clear).sum())
        t13.append(row)
    wc, decs = pd.DataFrame(wc), pd.DataFrame(decs)
    assert len(decs) == 72 * 2 * len(OBJ)
    decs["is_base_condition"] = is_base(decs)
    wc.to_csv(OUT / "views_b_worstcase.csv", index=False)
    decs.to_csv(OUT / "views_b_decision.csv", index=False)
    decs["winner_type"] = decs.winner.map({"A1": "anchor", "A2": "mean", "A3": "largest", "A4": "largest"})
    tal = []
    for (ax, obj), g in decs.groupby(["axis", "objective"]):
        for scope, h in [("72 conditions", g)] + [(f"beat CV {int(round(cv * 100))}% (24 conditions)", g[np.isclose(g.beat_cv, cv)])
                                                  for cv in (0.05, 0.15, 0.30)] + [("base condition (12 scenarios)", g[g.is_base_condition])]:
            row = dict(axis=ax, objective=obj, scope=scope, n_conditions=len(h))
            for r in RULES:
                row[f"n_win_{r}"] = int((h.winner == r).sum())
                row[f"n_win_clear_{r}"] = int(((h.winner == r) & h.clear).sum())
            row["n_win_largest_view_rule"] = row["n_win_A3"] + row["n_win_A4"]
            row["n_not_clear"] = int((~h.clear).sum())
            tal.append(row)
    tal = pd.concat([pd.DataFrame(tal), pd.DataFrame(t13)], ignore_index=True)
    tal.to_csv(OUT / "views_b_tally.csv", index=False)
    # agreement with the stored minimax table (worst-case |bias| and RMSE, A1 to A4)
    mm = pd.read_csv(ROOT / "results" / "2026-09-18_amend2" / "analysis" / "E1bE3b_e1_minimax.csv")
    mm = mm[mm.estimand == "T1"]
    chk = decs[decs.objective.isin(["worst_abs_bias", "worst_rmse"])].merge(mm, on=COND + ["axis"])
    a = chk[chk.objective == "worst_abs_bias"]
    b = chk[chk.objective == "worst_rmse"]
    print("stored minimax winners reproduced: |bias|", int((a.winner == a.maxabs_a1a4_winner).sum()), "of", len(a),
          "max|dv|", float((a.winner_value - a.maxabs_a1a4_value).abs().max()),
          "; RMSE", int((b.winner == b.rmse_a1a4_winner).sum()), "of", len(b),
          "max|dv|", float((b.winner_value - b.rmse_a1a4_value).abs().max()))

    # ------------------------------------------------------------ (c) Table 1 and S2 settings
    rows = []
    main_ = W[W.family == "repro_main"]
    for _, r in main_.iterrows():
        rows.append(dict(r, setting=f"pre-specified E1 ({'base case' if r.u_long == 0.25 else 'low scenario'})",
                         source="results/2026-09-18_full, re-simulated with the original seed"))
    for win in (0.15, 0.0):
        for a_, L in ((0.06, 0.25), (0.03, 0.10), (0.06, 0.10)):
            g = e1b[is_base(e1b, window=win) & np.isclose(e1b.u_anchor, a_) & np.isclose(e1b.u_long, L)]
            assert len(g) == 8
            for _, r in g.iterrows():
                rows.append(dict(r, setting="amendment 2 (E1b), independent seed",
                                 source="results/2026-09-18_amend2, re-simulated with the original seed"))
    c = pd.DataFrame(rows)
    c["window_label"] = np.where(c.window > 0, "+-15%", "none")
    for m in ("bias_mm", "rmse_mm", "mae_mm"):
        c[m + "_mci_lo"] = c[m] - Z * c[m + "_mcse"]
        c[m + "_mci_hi"] = c[m] + Z * c[m + "_mcse"]
    keep = ["setting", "source", "family", "cell", "seed_entropy", "seed_spawn_key", "anchor_mean_pct", "long_mean_pct",
            "window_label", "K", "beat_cv", "view_over", "S_median_mm", "axis", "rule", "rule_name", "n",
            "bias_mm", "bias_mm_mcse", "bias_mm_mci_lo", "bias_mm_mci_hi", "rmse_mm", "rmse_mm_mcse", "rmse_mm_mci_lo",
            "rmse_mm_mci_hi", "mae_mm", "mae_mm_mcse", "mae_mm_mci_lo", "mae_mm_mci_hi", "sd_err_mm", "sd_err_mm_mcse",
            "q2.5_err_mm", "q2.5_err_mm_mcse", "q97.5_err_mm", "q97.5_err_mm_mcse", "q95_abs_mm", "q95_abs_mm_mcse",
            "ratio_q95abs_rmse", "ratio_q95abs_rmse_mcse"]
    c = c[keep]
    pm = c[(c.family == "repro_main") & (c.axis == "AP")]
    c["is_min_rmse_of_published_pair_AP"] = False
    c["is_max_rmse_of_published_pair_AP"] = False
    c.loc[pm.rmse_mm.idxmin(), "is_min_rmse_of_published_pair_AP"] = True
    c.loc[pm.rmse_mm.idxmax(), "is_max_rmse_of_published_pair_AP"] = True
    c.to_csv(OUT / "views_c_table1_s2.csv", index=False)

    # ------------------------------------------------------------ (d) decomposition
    selm = met[(met.family == "decomp") & met.metric.str.startswith("p_sel")]
    sw = selm.pivot_table(index=["cell", "axis"], columns=["arm", "metric"], values="count", aggfunc="first")
    sw.columns = [f"count_{m[2:]}_{a}" for a, m in sw.columns]
    d = dec.merge(sw.reset_index(), on=["cell", "axis"]).drop(columns=["cpu_s"])
    d["offset_only_exceeds_noise_only"] = d.excess_zero_noise - d.excess_equal_views > 0
    d["full_minus_noise_only"] = d.offset_last
    front = ID + ["scenario", "anchor_mean_pct", "long_mean_pct", "u_anchor", "u_long", "K", "view_over", "beat_cv", "window",
                  "S_median_mm", "axis", "contrast"]
    d = d[front + [c_ for c_ in d.columns if c_ not in front and c_ not in PARAMS]]
    d.to_csv(OUT / "views_d_decomposition.csv", index=False)
    st = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "results/2026-09-18_amend2/E1_cells_*.parquet")))])
    st = label(st.rename(columns={c_: c_[2:] for c_ in st.columns if c_.startswith("p_")}).assign(view_under=True))
    st = st[(st.metric == "selection_inflation_mm") & np.isclose(st.beat_cv, 0.15) & (st.window == 0) & (st.S_median_mm == 10)]
    ck = d[d.contrast == "A3-A1"].merge(st[["scenario", "K", "view_over", "axis", "value", "mcse"]],
                                        on=["scenario", "K", "view_over", "axis"])
    ck = ck.assign(stored_inflation=ck.value, stored_inflation_mcse=ck.mcse, diff=ck.excess_full - ck.value,
                   z=(ck.excess_full - ck.value) / np.hypot(ck.excess_full_mcse, ck.mcse))
    ck[["scenario", "K", "view_over", "axis", "excess_full", "excess_full_mcse", "stored_inflation", "stored_inflation_mcse",
        "diff", "z"]].to_csv(OUT / "views_d_check_published.csv", index=False)
    assert len(ck) == 144
    print("decomposition full arm vs stored inflation: max |z|", float(ck.z.abs().max()), "n |z|>1.96", int((ck.z.abs() > 1.96).sum()), "of", len(ck))

    # ------------------------------------------------------------ (e) conditional quantiles
    cw_v = cond.pivot(index=ID + PARAMS + ["rule", "axis", "span_bin", "n"], columns="metric", values="value")
    cw_s = cond.pivot(index=ID + PARAMS + ["rule", "axis", "span_bin", "n"], columns="metric", values="mcse")
    cw_s.columns = [c_ + "_mcse" for c_ in cw_s.columns]
    cw = pd.concat([cw_v, cw_s], axis=1)
    order = [x for m in cond.metric.unique() for x in (m, m + "_mcse")]
    cw = cw[order].reset_index()
    cw["rule_name"] = cw.rule.map(RULE_NAME)
    cw["window_label"] = np.where(cw.window > 0, "+-15%", "none")
    cw["half_width_central95_mm"] = (cw["q97.5_err_mm"] - cw["q2.5_err_mm"]) / 2
    cw["bin_order"] = cw.span_bin.map({"0 to 5": 0, "5 to 7": 1, "7 to 10": 2, "10 to 13": 3, "13 to 16": 4, ">= 16": 5, "all": 6})
    cw = cw.sort_values(["axis", "cell", "rule", "bin_order"]).drop(columns="bin_order")
    cw.to_csv(OUT / "views_e_conditional_quantiles.csv", index=False)
    # ratio q95|e| / RMSE: all cells of the grid and the conditional family
    ra = W[["family", "cell", "scenario", "K", "beat_cv", "view_over", "view_under", "window", "S_median_mm", "rule", "axis",
            "rmse_mm", "q95_abs_mm", "ratio_q95abs_rmse", "ratio_q95abs_rmse_mcse", "q2.5_err_mm", "q97.5_err_mm"]]
    ra.to_csv(OUT / "views_e_ratio_all_cells.csv", index=False)
    rs = []
    for (ax, r), g in ra[ra.family == "repro_e1b"].groupby(["axis", "rule"]):
        i0, i1 = g.ratio_q95abs_rmse.idxmin(), g.ratio_q95abs_rmse.idxmax()
        rs.append(dict(axis=ax, rule=r, n_cells=len(g), ratio_min=g.at[i0, "ratio_q95abs_rmse"], ratio_min_mcse=g.at[i0, "ratio_q95abs_rmse_mcse"],
                       ratio_median=g.ratio_q95abs_rmse.median(), ratio_max=g.at[i1, "ratio_q95abs_rmse"],
                       ratio_max_mcse=g.at[i1, "ratio_q95abs_rmse_mcse"],
                       n_cells_q95abs_gt_rmse=int((g.q95_abs_mm > g.rmse_mm).sum()),
                       n_cells_q95abs_gt_2rmse=int((g.q95_abs_mm > 2 * g.rmse_mm).sum())))
    pd.DataFrame(rs).to_csv(OUT / "views_e_ratio_summary.csv", index=False)
    print("wrote", sorted(p.name for p in OUT.glob("views_*.csv")))


if __name__ == "__main__":
    main()
