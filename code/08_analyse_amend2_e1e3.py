#!/usr/bin/env python
"""Amendment 2 analysis of E1b (view-accuracy grid) and E3b (misclassification over view accuracy).

Label: amendment 2, secondary to E1 to E6 (protocol notes/protocol-2026-09-18-v01.md).
Input : results/2026-09-18_amend2/E1_cells_*.parquet, E3_cells_*.parquet (read only;
        config code/configs/amend2_2026-09-18.yaml, master seed 20260920).
Output: results/2026-09-18_amend2/analysis/E1bE3b_*.csv

View-accuracy grid ("12 combinations"): anchor half-normal scale a in {0.03, 0.06, 0.125, 0.25}
(mean anchor underestimation 0.798 a = 2.4, 4.8, 10, 20%) x long-axis scale L in {0.10, 0.25, 0.50}
(mean underestimation 8, 20, 40% in views 2 and 3; view 4 at 1.2 L).
Base otherwise (this family): K 3, beat CV 0.15, overestimation on, no acceptance window,
median true AP span 10 mm; rho 0.3, N 3, sinus, prospective (fixed). A4 base variant s_det 0.8,
f_rej 0.1, t_warn 3 mm, t_adj 5 mm. Estimand T1 unless stated.
A7 = offset-corrected mean (known view offsets); not an oracle or bound; excluded from minimax
because it needs the view offsets, which are unknown in practice.

Tables
  E1bE3b_e1_long.csv           tidy E1b rows (A4 base variant), wide by metric, both axes/estimands
  E1bE3b_e1_inflation.csv      selection inflation E[A3 - A1] per cell, both axes
  E1bE3b_e1_worstcase.csv      per condition x axis x estimand x estimator: min, max, range of
                               bias, max |bias|, max RMSE over the 12 view-accuracy combinations,
                               with the combination attaining each and its MCSE
  E1bE3b_e1_minimax.csv        per condition x axis x estimand: estimator minimising max |bias|
                               and max RMSE (among A1-A6 and among A1-A4), runner-up, gap, and
                               whether the gap exceeds 1.96 x its MCSE
  E1bE3b_e1_sensitivity.csv    base condition vs one-factor changes (K, beat CV, window,
                               median span, overestimation) for A1-A4, worst-case summaries
  E1bE3b_e1_consistency.csv    amendment cell equal to the main-run base cell, vs main run
  E1bE3b_e3_long.csv           tidy E3b rows, wide by metric
  E1bE3b_e3_ranges.csv         per axis x cutoff x estimator: min/max sensitivity and
                               specificity over the 12 combinations
95% Monte Carlo interval (MCI) = estimate +- 1.96 MCSE. Cells are independently seeded, so the
MCSE of a difference between two cells is sqrt(mcse1^2 + mcse2^2). Within a cell estimators share
random numbers, so that formula is conservative for within-cell estimator differences.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "2026-09-18_amend2"
MAIN = ROOT / "results" / "2026-09-18_full" / "analysis"
OUT = RES / "analysis"
Z = 1.959964
EST = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
DEPLOY6 = ["A1", "A2", "A3", "A4", "A5", "A6"]
DEPLOY4 = ["A1", "A2", "A3", "A4"]
COND = ["K", "beat_cv", "view_over", "window", "S_median_mm"]
ANCHOR_PCT = {0.03: 2.4, 0.06: 4.8, 0.125: 10.0, 0.25: 20.0}
LONG_PCT = {0.10: 8.0, 0.25: 20.0, 0.50: 40.0}
BASE = dict(K=3, beat_cv=0.15, view_over=True, window=0.0, S_median_mm=10.0)  # window 0 = none


def load(exp: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RES / f"{exp}_cells_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.rename(columns={c: c[2:] for c in df.columns if c.startswith("p_")})
    # mean underestimation = sqrt(2/pi) x scale = 0.798 x scale, rounded as in the paper's labels
    df["anchor_mean_pct"] = df["u_anchor"].map(ANCHOR_PCT)
    df["long_mean_pct"] = df["u_long"].map(LONG_PCT)
    assert df["anchor_mean_pct"].notna().all() and df["long_mean_pct"].notna().all()
    df["combo"] = df["anchor_mean_pct"].map(lambda v: f"{v:g}") + "/" + \
        df["long_mean_pct"].map(lambda v: f"{v:g}")
    if "window" in df:
        df["window"] = df["window"].fillna(0.0)   # NaN = no window
    return df


def wide(d: pd.DataFrame, idx: list[str]) -> pd.DataFrame:
    w = d.pivot_table(index=idx, columns="metric", values=["value", "mcse"], aggfunc="first")
    w.columns = [m.replace("_mm", "") + ("" if v == "value" else "_mcse") for v, m in w.columns]
    return w.reset_index()


def worst(g: pd.DataFrame) -> pd.Series:
    assert len(g) == 12, len(g)
    i_min, i_max = g["bias"].idxmin(), g["bias"].idxmax()
    ab = g["bias"].abs()
    i_ab, i_r = ab.idxmax(), g["rmse"].idxmax()
    ab_sorted = ab.sort_values(ascending=False)
    second_ab = ab_sorted.index[1]
    return pd.Series(dict(
        bias_min=g.at[i_min, "bias"], bias_min_mcse=g.at[i_min, "bias_mcse"], bias_min_combo=g.at[i_min, "combo"],
        bias_max=g.at[i_max, "bias"], bias_max_mcse=g.at[i_max, "bias_mcse"], bias_max_combo=g.at[i_max, "combo"],
        bias_range=g.at[i_max, "bias"] - g.at[i_min, "bias"],
        bias_range_mcse=np.hypot(g.at[i_max, "bias_mcse"], g.at[i_min, "bias_mcse"]),
        maxabs_bias=ab[i_ab], maxabs_bias_mcse=g.at[i_ab, "bias_mcse"], maxabs_bias_combo=g.at[i_ab, "combo"],
        maxabs_argmax_clear=bool(ab[i_ab] - ab[second_ab] >
                                 Z * np.hypot(g.at[i_ab, "bias_mcse"], g.at[second_ab, "bias_mcse"])),
        rmse_min=g["rmse"].min(),
        rmse_max=g.at[i_r, "rmse"], rmse_max_mcse=g.at[i_r, "rmse_mcse"], rmse_max_combo=g.at[i_r, "combo"],
        mean_abs_bias=ab.mean(),
    ))


def minimax(g: pd.DataFrame, cand: list[str], col: str, tag: str) -> dict:
    h = g[g.estimator.isin(cand)].sort_values(col)
    w, r = h.iloc[0], h.iloc[1]
    gap = r[col] - w[col]
    se = np.hypot(w[col + "_mcse"], r[col + "_mcse"])
    return {f"{tag}_winner": w.estimator, f"{tag}_value": w[col], f"{tag}_mcse": w[col + "_mcse"],
            f"{tag}_runnerup": r.estimator, f"{tag}_runnerup_value": r[col],
            f"{tag}_gap": gap, f"{tag}_gap_mcse": se, f"{tag}_clear": bool(gap > Z * se)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------ E1b
    e1 = load("E1")
    keep = (e1.estimator != "A4") | ((e1.s_det == 0.8) & (e1.f_rej == 0.1))
    e1 = e1[keep & e1.estimator.isin(EST)]
    idx = ["cell", "u_anchor", "u_long", "anchor_mean_pct", "long_mean_pct", "combo"] + COND + \
        ["axis", "estimand", "estimator"]
    lw = wide(e1, idx)
    assert len(lw) == 864 * 2 * 2 * 7, len(lw)
    lw.to_csv(OUT / "E1bE3b_e1_long.csv", index=False)

    inf = load("E1")
    inf = inf[inf.metric == "selection_inflation_mm"]
    inf = inf[["cell", "u_anchor", "u_long", "anchor_mean_pct", "long_mean_pct", "combo"] + COND +
              ["axis", "value", "mcse", "n"]].rename(columns={"value": "inflation", "mcse": "inflation_mcse"})
    assert len(inf) == 864 * 2
    inf.to_csv(OUT / "E1bE3b_e1_inflation.csv", index=False)

    wc = lw.groupby(COND + ["axis", "estimand", "estimator"]).apply(worst, include_groups=False).reset_index()
    wc.to_csv(OUT / "E1bE3b_e1_worstcase.csv", index=False)

    mm = []
    for key, g in wc.groupby(COND + ["axis", "estimand"]):
        row = dict(zip(COND + ["axis", "estimand"], key))
        g = g.rename(columns={"maxabs_bias_mcse": "maxabs_mcse", "rmse_max_mcse": "rmse_max_mcse"})
        g = g.assign(maxabs=g["maxabs_bias"])
        for cand, lab in ((DEPLOY6, "a1a6"), (DEPLOY4, "a1a4")):
            row.update(minimax(g, cand, "maxabs", f"maxabs_{lab}"))
            row.update(minimax(g, cand, "rmse_max", f"rmse_{lab}"))
        mm.append(row)
    mm = pd.DataFrame(mm)
    mm.to_csv(OUT / "E1bE3b_e1_minimax.csv", index=False)

    # one-factor sensitivity around the base condition (AP and SL, T1, A1-A4 plus A7 for context)
    changes = [("base", {}), ("K 2", {"K": 2}), ("K 4", {"K": 4}),
               ("beat CV 5%", {"beat_cv": 0.05}), ("beat CV 30%", {"beat_cv": 0.30}),
               ("window 0.15", {"window": 0.15}), ("median span 13 mm", {"S_median_mm": 13.0}),
               ("overestimation off", {"view_over": False})]
    sens = []
    for lab, ch in changes:
        c = {**BASE, **ch}
        m = np.ones(len(wc), bool)
        for k, v in c.items():
            m &= np.isclose(wc[k].astype(float), float(v))
        s = wc[m & (wc.estimand == "T1")].copy()
        s.insert(0, "scenario", lab)
        sens.append(s)
        mi = np.ones(len(inf), bool)
        for k, v in c.items():
            mi &= np.isclose(inf[k].astype(float), float(v))
        ii = inf[mi]
        for ax, gi in ii.groupby("axis"):
            sens.append(pd.DataFrame([dict(scenario=lab, axis=ax, estimand="T1", estimator="A3-A1",
                                           infl_min=gi.inflation.min(), infl_max=gi.inflation.max(),
                                           infl_max_mcse=gi.loc[gi.inflation.idxmax(), "inflation_mcse"],
                                           infl_min_combo=gi.loc[gi.inflation.idxmin(), "combo"],
                                           infl_max_combo=gi.loc[gi.inflation.idxmax(), "combo"], **c)]))
    sens = pd.concat(sens, ignore_index=True)
    sens.to_csv(OUT / "E1bE3b_e1_sensitivity.csv", index=False)

    # consistency with main run base cell (anchor 0.06, long 0.25, K3, CV .15, over on, window .15, S 10)
    cons = lw[(lw.u_anchor == 0.06) & (lw.u_long == 0.25) & (lw.K == 3) & (lw.beat_cv == 0.15) &
              lw.view_over & (lw.window == 0.15) & (lw.S_median_mm == 10.0) & (lw.estimand == "T1")]
    try:
        hd = pd.read_csv(MAIN / "E1E3_e1_headline.csv")
        hd = hd[(hd.u == "base") & (hd.estimand == "T1")][["axis", "estimator", "bias", "bias_mcse", "rmse", "rmse_mcse"]]
        cons = cons[["axis", "estimator", "bias", "bias_mcse", "rmse", "rmse_mcse"]].merge(
            hd, on=["axis", "estimator"], suffixes=("_amend2", "_main"))
        cons["bias_diff"] = cons.bias_amend2 - cons.bias_main
        cons["bias_diff_z"] = cons.bias_diff / np.hypot(cons.bias_mcse_amend2, cons.bias_mcse_main)
        cons["rmse_diff"] = cons.rmse_amend2 - cons.rmse_main
        cons["rmse_diff_z"] = cons.rmse_diff / np.hypot(cons.rmse_mcse_amend2, cons.rmse_mcse_main)
    except (FileNotFoundError, KeyError) as exc:
        print("consistency check skipped:", exc)
    cons.to_csv(OUT / "E1bE3b_e1_consistency.csv", index=False)

    # ------------------------------------------------------------------ E3b
    e3 = load("E3")
    e3w = wide(e3, ["cell", "u_anchor", "u_long", "anchor_mean_pct", "long_mean_pct", "combo",
                    "axis", "estimand", "cutoff_mm", "estimator"])
    assert len(e3w) == 12 * 2 * 3 * 7
    e3w.to_csv(OUT / "E1bE3b_e3_long.csv", index=False)
    rg = []
    for (ax, cut, est), g in e3w.groupby(["axis", "cutoff_mm", "estimator"]):
        r = dict(axis=ax, cutoff_mm=cut, estimator=est, prevalence_min=g.prevalence.min(),
                 prevalence_max=g.prevalence.max())
        for m in ("sensitivity", "specificity", "ppv"):
            i0, i1 = g[m].idxmin(), g[m].idxmax()
            r.update({f"{m}_min": g.at[i0, m], f"{m}_min_mcse": g.at[i0, m + "_mcse"],
                      f"{m}_min_combo": g.at[i0, "combo"],
                      f"{m}_max": g.at[i1, m], f"{m}_max_mcse": g.at[i1, m + "_mcse"],
                      f"{m}_max_combo": g.at[i1, "combo"], f"{m}_range": g.at[i1, m] - g.at[i0, m]})
        rg.append(r)
    pd.DataFrame(rg).to_csv(OUT / "E1bE3b_e3_ranges.csv", index=False)
    print("wrote", sorted(p.name for p in OUT.glob("E1bE3b_*.csv")))


if __name__ == "__main__":
    main()
