#!/usr/bin/env python
"""Amendment 2 tables for E5b (AI sharing image-level error) and E6b (Brown-Forsythe sentinel test).

Reads results/2026-09-18_amend2/E5_cells_*.parquet and E6_cells_*.parquet (config
code/configs/amend2_2026-09-18.yaml, master seed 20260920, duomaxsim 0.1.0; E5 = E5b, E6 = E6b)
and the decomposition tables written by code/10_amend2_E5b_decomposition.py. Read-only on the parquet
files. No random numbers. Writes results/2026-09-18_amend2/analysis/E5bE6b_*.csv.

Conventions
- MCI = 95% Monte Carlo interval, estimate +- 1.959964 x MCSE.
- E5b: 2,000 validation studies of n = 200 patients per cell; AP axis; every read is rule A4
  (s_det 0.8, f_rej 0.1, t_warn 3 mm, t_adj 5 mm). lambda is a grid factor (cells are not on common
  random numbers across lambda); within a cell the shared, independent and inherited AIs share z,
  images and reads. LoA width MCSE is the conservative bound MCSE(hi) + MCSE(lo).
- E6b: the anchoring fraction is called w here (the protocol's "alpha"), to keep alpha for the test
  level (0.05). Size = rejection rate at w = 0. Tests: f_variance (one-sided F, sentinel more
  variable), welch_mean (two-sided Welch t), bf (two-sided Brown-Forsythe), bf_onesided
  (Brown-Forsythe, sentinel more dispersed), either_unadjusted (F or Welch).
- Size validity of a test at (cell, n): "strict" = the 95% MCI of its size includes or lies below 0.05
  (lower limit <= 0.05); "liberal" = size <= 0.075 (Bradley 1978 liberal bound 1.5 x alpha).
Usage: python code/11_amend2_E5bE6b_tables.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "2026-09-18_amend2")
OUT = os.path.join(RES, "analysis")
Z = 1.959964
LEVEL = 0.05
POWER = 0.80
E5_BASE = dict(p_beat_cv=0.15, p_view_over=True)
E6_BASE = dict(p_beat_cv=0.15, p_sigma_rb_mm=0.75, p_ai_draft_sigma_mm=2.0)


def load(exp):
    files = sorted(glob.glob(os.path.join(RES, f"{exp}_cells_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert df["n_studies"].nunique() == 1 and df["n_studies"].iloc[0] == 2000
    assert df.cell.nunique() == {"E5": 18, "E6": 54}[exp]
    return df


def mci(df, v="value", se="mcse"):
    df["mci_lo"] = df[v] - Z * df[se]
    df["mci_hi"] = df[v] + Z * df[se]
    return df


# ------------------------------------------------------------------ E5b

def e5_tables(df):
    keys = ["cell", "p_beat_cv", "p_view_over", "p_ai_shared_lambda"]
    rq = df[df.ai.isna() & df.estimator.eq("A4")][keys + ["ref", "metric", "value", "mcse", "n"]].copy()
    rq = mci(rq.sort_values(keys + ["ref", "metric"]).reset_index(drop=True))
    cal = df[df.estimator == "calibration"].pivot_table(index=keys, columns="metric", values="value").reset_index()
    cal.columns.name = None
    # agreement: apparent / true / diff per (cell, ai, sigma, ref, m)
    a = df[df.ai.notna() & df.metric.str.match(r"^(apparent|true|diff)_")].copy()
    a["kind"] = a.metric.str.split("_", n=1).str[0]
    a["m"] = a.metric.str.split("_", n=1).str[1]
    idx = keys + ["ai", "sigma_ai_mm", "ref", "m"]
    w = a.pivot_table(index=idx, columns="kind", values=["value", "mcse"]).reset_index()
    w.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in w.columns]
    w = w.rename(columns={"value_apparent": "apparent", "value_true": "true", "value_diff": "diff"})
    lw = []
    for _, g in w[w.m.isin(["loa_lo", "loa_hi"])].groupby(keys + ["ai", "sigma_ai_mm", "ref"]):
        hi = g[g.m == "loa_hi"].iloc[0]
        lo = g[g.m == "loa_lo"].iloc[0]
        r = hi.copy()
        r["m"] = "loa_width"
        for c in ("apparent", "true", "diff"):
            r[c] = hi[c] - lo[c]
            r["mcse_" + c] = hi["mcse_" + c] + lo["mcse_" + c]   # conservative bound
        lw.append(r)
    w = pd.concat([w, pd.DataFrame(lw)], ignore_index=True)
    w["rel_diff_pct"] = 100 * w["diff"] / w["true"].where(w.m.isin(["mae", "loa_width"]))
    w["diff_mci_lo"] = w["diff"] - Z * w["mcse_diff"]
    w["diff_mci_hi"] = w["diff"] + Z * w["mcse_diff"]
    w = w.sort_values(idx).reset_index(drop=True)
    # paired comparisons with the independent AI
    pr = df[df.comparator.eq("independent")][keys + ["ai", "sigma_ai_mm", "ref", "metric", "value", "mcse", "n"]]
    pr = mci(pr.copy()).sort_values(keys + ["ai", "sigma_ai_mm", "ref", "metric"]).reset_index(drop=True)
    pr["note"] = np.where((pr.ai == "shared") & (pr.p_ai_shared_lambda == 0),
                          "lambda 0: shared AI identical to independent AI (difference exactly 0)", "")
    return rq, cal, w, pr


def e5_lambda_summary(w):
    """Shared AI vs lambda (and independent AI in the same cell): MAE, LoA width, ICC, BA bias."""
    s = w[w.ai.isin(["shared", "independent"]) & w.m.isin(["mae", "loa_width", "icc", "ba_bias"])].copy()
    s = s[["cell", "p_beat_cv", "p_view_over", "p_ai_shared_lambda", "ai", "sigma_ai_mm", "ref", "m",
           "true", "mcse_true", "apparent", "mcse_apparent", "diff", "mcse_diff", "rel_diff_pct"]]
    return s.reset_index(drop=True)


def e5_independent_consistency(w):
    """The independent AI is the same model in every lambda cell; spread across the 3 cells is MC noise."""
    g = w[(w.ai == "independent") & w.m.isin(["mae", "icc", "ba_bias"])]
    out = g.groupby(["p_beat_cv", "p_view_over", "sigma_ai_mm", "ref", "m"]).agg(
        apparent_min=("apparent", "min"), apparent_max=("apparent", "max"),
        mcse_apparent_max=("mcse_apparent", "max")).reset_index()
    out["range_over_max_mcse"] = (out.apparent_max - out.apparent_min) / out.mcse_apparent_max
    return out


def decomposition_summary(e6):
    """Reference-error decomposition (script 10) joined with the E6b inter-reader SD at the E5 base cell."""
    ref = pd.read_csv(os.path.join(OUT, "E5bE6b_decomp_reference_error.csv"))
    inter = pd.read_csv(os.path.join(OUT, "E5bE6b_decomp_inter_reader.csv"))
    rows = []
    for _, r in ref[ref.ref.isin(["single", "mean2"])].iterrows():
        row = r.to_dict()
        m = inter[(inter.cell == r.cell) & inter.comparable & (inter.e6_sigma_rb_mm == 0.75)]
        if len(m):
            sd6, se6 = float(m.e6_within_sd_read_diff.iloc[0]), float(m.mcse_e6.iloc[0])
            # per-read variance visible to a fixed-pair inter-reader study (within-pair SD^2 / 2) plus the
            # reader-bias variance that differences between pairs would reveal (sigma_rb^2 = 0.5625 mm^2)
            vis = sd6 ** 2 / 2 + 0.75 ** 2
            if r.ref == "mean2":
                vis = sd6 ** 2 / 4 + 0.75 ** 2 / 2
            row.update(e6_within_pair_sd=sd6, mcse_e6_within_pair_sd=se6,
                       inter_reader_visible_var=vis,
                       inter_reader_visible_share_of_var_ref=vis / r.var_ref,
                       mcse_visible_share=np.hypot(sd6 * se6 / (r.var_ref if r.ref == "single" else 2 * r.var_ref),
                                                   vis * r.mcse_var_ref / r.var_ref ** 2))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E6b

def e6_tables(df):
    keys = ["cell", "p_beat_cv", "p_sigma_rb_mm", "p_ai_draft_sigma_mm"]
    d = df[df.design != "sentinel"]
    prec = d.pivot_table(index=keys + ["design", "n_double", "overlap_frac"], columns="metric",
                         values=["value", "mcse"], dropna=False).dropna(how="all").reset_index()
    prec.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in prec.columns]
    prec = prec.rename(columns=lambda c: c.replace("value_", ""))
    prec["icc_emp95_range"] = 2 * Z * prec["icc_empirical_sd"]
    prec["loa_hi_emp95_range"] = 2 * Z * prec["loa_hi_empirical_sd"]
    prec["n_double"] = prec["n_double"].astype(int)
    s = df[df.design == "sentinel"][keys + ["sentinel_n", "anchor_alpha", "metric", "value", "mcse", "n"]].copy()
    s = s.rename(columns={"anchor_alpha": "w_anchor"})
    s["test"] = s.metric.str.replace("power_reject_", "", regex=False)
    s["sentinel_n"] = s["sentinel_n"].astype(int)
    s = mci(s).sort_values(keys + ["test", "sentinel_n", "w_anchor"]).reset_index(drop=True)
    size = s[s.w_anchor == 0].copy()
    size["size_valid_strict"] = size.mci_lo <= LEVEL
    size["size_valid_liberal"] = size.value <= 1.5 * LEVEL
    size["nominal_outside_mci"] = (size.mci_lo > LEVEL) | (size.mci_hi < LEVEL)
    size["above_nominal_mci"] = size.mci_lo > LEVEL
    size["below_nominal_mci"] = size.mci_hi < LEVEL
    return prec, s, size


def size_summary(size):
    return size.groupby(["test", "sentinel_n"]).agg(
        n_combos=("value", "size"), min_size=("value", "min"), median_size=("value", "median"),
        max_size=("value", "max"), n_outside_mci=("nominal_outside_mci", "sum"),
        n_above=("above_nominal_mci", "sum"), n_below=("below_nominal_mci", "sum"),
        n_strict_valid=("size_valid_strict", "sum"), n_liberal_valid=("size_valid_liberal", "sum")).reset_index()


def size_summary_by_test(size):
    return size.groupby("test").agg(
        n_combos=("value", "size"), min_size=("value", "min"), median_size=("value", "median"),
        max_size=("value", "max"), n_outside_mci=("nominal_outside_mci", "sum"),
        n_above=("above_nominal_mci", "sum"), n_below=("below_nominal_mci", "sum"),
        n_strict_valid=("size_valid_strict", "sum"), n_liberal_valid=("size_valid_liberal", "sum")).reset_index()


def min_n_table(s, size):
    """Per cell, test and w > 0: smallest sentinel n with power >= 0.80, under three size rules."""
    keys = ["cell", "p_beat_cv", "p_sigma_rb_mm", "p_ai_draft_sigma_mm"]
    sz = size[keys + ["test", "sentinel_n", "value", "mcse", "size_valid_strict", "size_valid_liberal"]].rename(
        columns={"value": "size", "mcse": "mcse_size"})
    pw = s[s.w_anchor > 0].merge(sz, on=keys + ["test", "sentinel_n"])
    rows = []
    for (k, g) in pw.groupby(keys + ["test", "w_anchor"]):
        g = g.sort_values("sentinel_n")
        row = dict(zip(keys + ["test", "w_anchor"], k))
        for rule, ok in (("any", np.ones(len(g), bool)), ("strict", g.size_valid_strict.to_numpy()),
                         ("liberal", g.size_valid_liberal.to_numpy())):
            hit = g[(g.value.to_numpy() >= POWER) & ok]
            if len(hit):
                h = hit.iloc[0]
                row[f"min_n_{rule}"] = int(h.sentinel_n)
                row[f"power_at_min_n_{rule}"] = h.value
                row[f"mcse_power_{rule}"] = h.mcse
                row[f"power_mci_lo_{rule}"] = h.mci_lo
                row[f"size_at_min_n_{rule}"] = h["size"]
            else:
                row[f"min_n_{rule}"] = np.nan
        row["max_power_tested"] = g.value.max()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    e5 = load("E5")
    e6 = load("E6")
    rq, cal, w, pr = e5_tables(e5)
    rq.to_csv(os.path.join(OUT, "E5bE6b_E5b_reference_quality.csv"), index=False)
    cal.to_csv(os.path.join(OUT, "E5bE6b_E5b_inherited_calibration.csv"), index=False)
    w.to_csv(os.path.join(OUT, "E5bE6b_E5b_ai_agreement.csv"), index=False)
    pr.to_csv(os.path.join(OUT, "E5bE6b_E5b_paired.csv"), index=False)
    e5_lambda_summary(w).to_csv(os.path.join(OUT, "E5bE6b_E5b_lambda_summary.csv"), index=False)
    e5_independent_consistency(w).to_csv(os.path.join(OUT, "E5bE6b_E5b_independent_consistency.csv"),
                                         index=False)
    prec, s, size = e6_tables(e6)
    prec.to_csv(os.path.join(OUT, "E5bE6b_E6b_precision.csv"), index=False)
    s.to_csv(os.path.join(OUT, "E5bE6b_E6b_sentinel_power.csv"), index=False)
    size.to_csv(os.path.join(OUT, "E5bE6b_E6b_size_check.csv"), index=False)
    ss = size_summary(size)
    ss.to_csv(os.path.join(OUT, "E5bE6b_E6b_size_summary.csv"), index=False)
    st = size_summary_by_test(size)
    st.to_csv(os.path.join(OUT, "E5bE6b_E6b_size_summary_by_test.csv"), index=False)
    mn = min_n_table(s, size)
    mn.to_csv(os.path.join(OUT, "E5bE6b_E6b_min_n.csv"), index=False)
    # distribution of min n across the 54 cells
    dist = []
    for (t, wa), g in mn.groupby(["test", "w_anchor"]):
        for rule in ("any", "strict", "liberal"):
            vc = g[f"min_n_{rule}"].fillna(0).astype(int).value_counts().to_dict()
            dist.append(dict(test=t, w_anchor=wa, rule=rule, n_cells=len(g),
                             **{f"cells_min_n_{k if k else 'none'}": vc.get(k, 0) for k in (10, 25, 50, 100, 0)}))
    pd.DataFrame(dist).to_csv(os.path.join(OUT, "E5bE6b_E6b_min_n_distribution.csv"), index=False)
    dec = decomposition_summary(e6)
    dec.to_csv(os.path.join(OUT, "E5bE6b_E5b_decomposition_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(st.to_string())
    print(ss.to_string())
    b = mn
    for k, v in E6_BASE.items():
        b = b[b[k] == v]
    print(b[["test", "w_anchor", "min_n_any", "min_n_strict", "min_n_liberal", "power_at_min_n_strict",
             "mcse_power_strict", "size_at_min_n_strict", "max_power_tested"]].to_string())
    print(pd.DataFrame(dist).to_string())


if __name__ == "__main__":
    main()
