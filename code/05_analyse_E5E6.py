"""Analyse E5 (reference design -> apparent AI performance) and E6 (reference-programme sizing).

Reads results/2026-09-18_full/E5_cells_*.parquet and E6_cells_*.parquet (long format, written by
code/01_run_experiment.py with duomaxsim 0.1.0) and writes tables to
results/2026-09-18_full/analysis/E5E6_*.csv. Read-only on the parquet files. No random numbers.

Conventions (see code/lib/duomaxsim/experiments.py run_E5, run_E6):
- E5: every read uses rule A4 (base s_det 0.8, f_rej 0.1, t_warn 3, t_adj 5, scrutiny 'triggered'),
  K = 3, AP axis, n_val = 200 patients per validation study, 2,000 studies per cell. Per-study
  metrics are averaged over studies; mcse = SD across studies / sqrt(2000).
  LoA width is derived as mean(loa_hi) - mean(loa_lo); its MCSE is not stored, so we report the
  conservative bound mcse_hi + mcse_lo.
- E6: gold set read by two fixed readers per study; overlap design of a 500-case production set is
  a gold set of size round(500 f). Overlap sizes 50 and 100 reuse the gold draws of n = 50 and 100
  (identical designs). Sentinel: manual (n_s) vs AI-assisted (500 - n_s) cases from one reader.
Usage: python code/05_analyse_E5E6.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "2026-09-18_full")
OUT = os.path.join(RES, "analysis")
Z = 1.959964

E5_BASE = dict(p_beat_cv=0.15, p_view_over=True)
E6_BASE = dict(p_beat_cv=0.15, p_sigma_rb_mm=0.75, p_ai_draft_sigma_mm=2.0)


def load(exp):
    files = sorted(glob.glob(os.path.join(RES, f"{exp}_cells_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert df["n_studies"].nunique() == 1
    return df


def sel(df, d):
    m = np.ones(len(df), bool)
    for k, v in d.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def e5_tables(df):
    keys = ["cell", "p_beat_cv", "p_view_over"]
    # reference quality
    rq = df[df.ai.isna() & (df.estimator != "calibration")][keys + ["ref", "metric", "value", "mcse", "n"]]
    rq = rq.sort_values(keys + ["ref", "metric"])
    # calibration of inherited-bias AI
    cal = df[df.estimator == "calibration"].pivot_table(index=keys, columns="metric", values="value").reset_index()
    cal.columns.name = None
    cal["implied_offset_at_10mm"] = cal.inherited_intercept_mm + (cal.inherited_slope - 1) * 10.0
    cal["implied_offset_at_5mm"] = cal.inherited_intercept_mm + (cal.inherited_slope - 1) * 5.0
    cal["implied_offset_at_20mm"] = cal.inherited_intercept_mm + (cal.inherited_slope - 1) * 20.0
    # AI agreement: wide per (cell, ai, sigma, ref)
    a = df[df.ai.notna()].copy()
    a["kind"] = a.metric.str.split("_", n=1).str[0]
    a["m"] = a.metric.str.split("_", n=1).str[1]
    idx = keys + ["ai", "sigma_ai_mm", "ref", "m"]
    w = a.pivot_table(index=idx, columns="kind", values=["value", "mcse"]).reset_index()
    w.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in w.columns]
    w = w.rename(columns={"value_apparent": "apparent", "value_true": "true", "value_diff": "diff",
                          "mcse_apparent": "mcse_apparent", "mcse_true": "mcse_true", "mcse_diff": "mcse_diff"})
    # derived LoA width
    lw = []
    for (_, g) in w[w.m.isin(["loa_lo", "loa_hi"])].groupby(keys + ["ai", "sigma_ai_mm", "ref"]):
        hi = g[g.m == "loa_hi"].iloc[0]
        lo = g[g.m == "loa_lo"].iloc[0]
        r = hi.copy()
        r["m"] = "loa_width"
        for c in ("apparent", "true", "diff"):
            r[c] = hi[c] - lo[c]
            r["mcse_" + c] = hi["mcse_" + c] + lo["mcse_" + c]  # conservative bound
        lw.append(r)
    w = pd.concat([w, pd.DataFrame(lw)], ignore_index=True)
    w["rel_diff_pct"] = 100 * w["diff"] / w["true"].where(w.m.isin(["mae", "loa_width"]))
    w = w.sort_values(keys + ["ai", "sigma_ai_mm", "ref", "m"]).reset_index(drop=True)
    return rq, cal, w


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
    s["test"] = s.metric.str.replace("power_reject_", "", regex=False)
    s["sentinel_n"] = s["sentinel_n"].astype(int)
    size = s[s.anchor_alpha == 0].copy()
    size["lo95_mc"] = size.value - Z * size.mcse
    size["hi95_mc"] = size.value + Z * size.mcse
    size["nominal_outside_mc95"] = (size.lo95_mc > 0.05) | (size.hi95_mc < 0.05)
    return prec, s, size


def main():
    os.makedirs(OUT, exist_ok=True)
    e5 = load("E5")
    e6 = load("E6")
    rq, cal, w = e5_tables(e5)
    rq.to_csv(os.path.join(OUT, "E5E6_E5_reference_quality.csv"), index=False)
    cal.to_csv(os.path.join(OUT, "E5E6_E5_inherited_calibration.csv"), index=False)
    w.to_csv(os.path.join(OUT, "E5E6_E5_ai_agreement.csv"), index=False)
    prec, sent, size = e6_tables(e6)
    prec.to_csv(os.path.join(OUT, "E5E6_E6_precision.csv"), index=False)
    sent.to_csv(os.path.join(OUT, "E5E6_E6_sentinel_power.csv"), index=False)
    size.to_csv(os.path.join(OUT, "E5E6_E6_size_check.csv"), index=False)

    # summaries across cells (for text)
    sz = size.groupby(["test", "sentinel_n"]).agg(
        min_size=("value", "min"), median_size=("value", "median"), max_size=("value", "max"),
        n_cells=("value", "size"), n_outside=("nominal_outside_mc95", "sum")).reset_index()
    cols = ["test", "sentinel_n", "p_beat_cv", "p_sigma_rb_mm", "p_ai_draft_sigma_mm", "value", "mcse"]
    worst = size.loc[size.groupby("test").value.idxmax(), cols]
    sz.to_csv(os.path.join(OUT, "E5E6_E6_size_summary.csv"), index=False)
    worst.to_csv(os.path.join(OUT, "E5E6_E6_size_worst.csv"), index=False)
    # F-test size by grid factor
    fsz = size[size.test == "f_variance"].groupby(["p_beat_cv", "p_ai_draft_sigma_mm"]).value.mean().unstack()
    fsz.to_csv(os.path.join(OUT, "E5E6_E6_fsize_by_beatcv_aisigma.csv"))
    # E5 sensitivity across cells: apparent minus true MAE for independent AI, sigma 2
    sens = w[(w.m.isin(["mae", "icc", "loa_width"])) & (w.sigma_ai_mm == 2.0)][
        ["p_beat_cv", "p_view_over", "ai", "ref", "m", "apparent", "true", "diff", "mcse_diff"]]
    sens.to_csv(os.path.join(OUT, "E5E6_E5_sensitivity_sigma2.csv"), index=False)
    print("wrote", sorted(os.listdir(OUT)))
    print(sz.to_string())
    print(worst.to_string())
    print(fsz.round(3).to_string())


if __name__ == "__main__":
    main()
