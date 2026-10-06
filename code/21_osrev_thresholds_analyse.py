#!/usr/bin/env python
"""Protocol amendment 3, package A3-3 (thresholds): tidy tables from the count tables.

Input : results/2026-10-05_osrev/thresholds/raw/{published,replicates}_*.parquet
        (code/20_osrev_thresholds_run.py), results/2026-09-18_full/E3_cells_*.parquet (read only).
Output: results/2026-10-05_osrev/thresholds/*.csv (login node, a few CPU-seconds).

  thresholds_true_span_distribution.csv   true-span distribution as implemented (closed form or quadrature),
                                          with the simulated proportions at or above each cut-off
  thresholds_true_span_density.csv        density on a 0.1 mm grid and simulated histogram (0.25 mm bins)
  thresholds_2x2.csv                      full 2 by 2 tables, predictive values, reclassification vs anchor-view mean
  thresholds_near_band.csv                false positives and false negatives among patients within 1 and 2 mm of a cut-off
  thresholds_by_distance.csv              misclassification probability by distance of the true span from the cut-off
  thresholds_reproduction.csv             published E3 values against the rebuilt and the amendment-3 values
  thresholds_replicate_check.csv          between-replicate SD against the binomial Monte Carlo SE

Samples. "published_seed": the 100,000 patients of the stored E3 cell, rebuilt with its original seed
(the numbers of the manuscript). "amend3_pooled": 100 new replicates of 100,000 patients pooled
(10,000,000 per scenario; entropy 20261005, spawn key (73, 1, replicate), common random numbers across
scenarios). 95% Monte Carlo interval = estimate -+ 1.959964 MCSE (binomial MCSE for proportions,
paired multinomial MCSE for reclassification). Cut-offs are illustrative values on the simulated
span scale, not clinical eligibility criteria.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from duomaxsim import config as C  # noqa: E402
from duomaxsim import osrev_thresholds as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "2026-10-05_osrev" / "thresholds"
RAW = OUT / "raw"
CFG_PATH = ROOT / "code" / "configs" / "osrev_thresholds.yaml"
Z = T.Z95
RULE_LABEL = {"A1": "Anchor-view mean", "A2": "Mean across views", "A3": "Largest view mean",
              "A4": "Largest view mean after review"}
CNT = ["tp", "fp", "fn", "tn"]
REC = ["ev_up", "ev_dn", "ne_up", "ne_dn"]


def load(table: str, oc: dict) -> pd.DataFrame:
    """Counts summed over replicates, one row per sample x scenario x keys."""
    pub = pd.read_parquet(RAW / f"published_{table}.parquet")
    rep = pd.read_parquet(RAW / f"replicates_{table}.parquet")
    rep["sample"] = "amend3_pooled"
    df = pd.concat([pub, rep], ignore_index=True)
    keys = [c for c in df.columns if c in ("sample", "scenario", "axis", "cutoff_mm", "rule", "half_width_mm",
                                           "d_lo_mm", "d_hi_mm", "lo_mm", "hi_mm")]
    vals = [c for c in df.columns if c in ("n", "n_pos", "n_mis", *CNT, *REC)]
    g = df.groupby(keys, sort=False, as_index=False)[vals].sum()
    nrep = df.groupby(["sample", "scenario"], sort=False)["replicate"].nunique().rename("n_replicates").reset_index()
    g = g.merge(nrep, on=["sample", "scenario"])
    g["scenario_label"] = g["scenario"].map({k: v["label"] for k, v in oc["scenarios"].items()})
    return g


def add_ci(df, name, est, se, scale=1.0):
    df[name] = scale * est
    df[name + "_mcse"] = scale * se
    df[name + "_lo95"] = scale * (est - Z * se)
    df[name + "_hi95"] = scale * (est + Z * se)


def main():
    with open(CFG_PATH) as fh:
        oc = yaml.safe_load(fh)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    # ------------------------------------------------------------------ 2 by 2 tables
    o = load("overall", oc)
    o["rule_label"] = o["rule"].map(RULE_LABEL)
    for c in CNT:
        o[c + "_per_100000"] = 1e5 * o[c] / o["n"]
        o[c + "_per_1000"] = 1e3 * o[c] / o["n"]
    m = T.classification_metrics(o.tp, o.fp, o.fn, o.tn)
    for k, v in m.items():
        o[k] = np.asarray(v)
    n_ev, n_ne = o.tp + o.fn, o.tn + o.fp
    r = T.reclassification_metrics(o.ev_up, o.ev_dn, o.ne_up, o.ne_dn, n_ev, n_ne)
    for k in ("nri_events", "nri_nonevents", "nri"):
        add_ci(o, k + "_vs_A1", np.asarray(r[k]), np.asarray(r[k + "_mcse"]))
    add_ci(o, "net_correct_per_1000_vs_A1", np.asarray(r["net_correct"]), np.asarray(r["net_correct_mcse"]), 1e3)
    o["moved_to_correct_side_per_1000_vs_A1"] = 1e3 * (o.ev_up + o.ne_dn) / o["n"]
    o["moved_to_wrong_side_per_1000_vs_A1"] = 1e3 * (o.ev_dn + o.ne_up) / o["n"]
    first = ["sample", "scenario", "scenario_label", "axis", "cutoff_mm", "rule", "rule_label", "n_replicates", "n"]
    o = o[first + [c for c in o.columns if c not in first]]
    o.to_csv(OUT / "thresholds_2x2.csv", index=False)

    # ------------------------------------------------------------------ near-threshold bands
    b = load("band", oc).rename(columns={"n": "n_band"})
    b["rule_label"] = b["rule"].map(RULE_LABEL)
    tot = o[["sample", "scenario", "axis", "cutoff_mm", "rule", "n", "fp", "fn"]].rename(
        columns={"n": "n_all", "fp": "fp_all", "fn": "fn_all"})
    b = b.merge(tot, on=["sample", "scenario", "axis", "cutoff_mm", "rule"])
    b["band_per_1000_all"] = 1e3 * b.n_band / b.n_all
    for k, num in (("fp", b.fp), ("fn", b.fn), ("misclassified", b.fp + b.fn)):
        pr, se = T._prop(num, b.n_band)
        add_ci(b, f"{k}_per_1000_band", pr, se, 1e3)
    # conditional on the side of the cut-off inside the band
    pr, se = T._prop(b.fp, b.fp + b.tn)
    add_ci(b, "fp_per_1000_band_nonevents", pr, se, 1e3)
    pr, se = T._prop(b.fn, b.fn + b.tp)
    add_ci(b, "fn_per_1000_band_events", pr, se, 1e3)
    b["n_band_nonevents"] = b.fp + b.tn
    b["n_band_events"] = b.fn + b.tp
    b["fp_band_per_1000_all"] = 1e3 * b.fp / b.n_all
    b["fn_band_per_1000_all"] = 1e3 * b.fn / b.n_all
    pr, se = T._prop(b.fp, b.fp_all)
    add_ci(b, "share_of_all_fp_in_band", pr, se)
    pr, se = T._prop(b.fn, b.fn_all)
    add_ci(b, "share_of_all_fn_in_band", pr, se)
    first = ["sample", "scenario", "scenario_label", "axis", "cutoff_mm", "half_width_mm", "rule", "rule_label",
             "n_replicates", "n_all", "n_band", "n_band_nonevents", "n_band_events"]
    b = b[first + [c for c in b.columns if c not in first]]
    b.to_csv(OUT / "thresholds_near_band.csv", index=False)

    # ------------------------------------------------------------------ by distance
    d = load("bins", oc)
    d["rule_label"] = d["rule"].map(RULE_LABEL)
    d["error_type"] = np.where(d.d_lo_mm < 0, "false_positive", "false_negative")
    pr, se = T._prop(d.n_mis, d.n)
    add_ci(d, "p_misclassified", pr, se)
    d["misclassified_per_1000_bin"] = 1e3 * d["p_misclassified"]
    pr, se = T._prop(d.n_pos, d.n)
    add_ci(d, "p_positive", pr, se)
    first = ["sample", "scenario", "scenario_label", "axis", "cutoff_mm", "rule", "rule_label", "d_lo_mm", "d_hi_mm",
             "error_type", "n_replicates", "n", "n_pos", "n_mis"]
    d = d[first + [c for c in d.columns if c not in first]]
    d.to_csv(OUT / "thresholds_by_distance.csv", index=False)

    # ------------------------------------------------------------------ true-span distribution
    base_cfg = C.load_config(ROOT / "code" / "configs" / "base.yaml")
    p10 = C.cell_params(base_cfg, "E3", 29)
    p13 = dict(p10, S_median_mm=13.0)
    cuts = [float(c) for c in oc["analysis"]["cutoffs_mm"]]
    kind, lo, hi, a_, b_ = T._ratio_spec(p10)
    rows = []
    emp_q = pd.read_csv(RAW / "published_base_true_span_empirical.csv").set_index("quantile")
    for pop, p, sid in (("base (median AP span 10 mm)", p10, "base"),
                        ("sensitivity (median AP span 13 mm)", p13, "base_median13")):
        for ax in T.AXES:
            row = {"population": pop, "scenario": sid}
            row.update(T.span_distribution(p, ax, cuts))
            row["family"] = ("lognormal(ln median, log SD)" if ax == "AP" else
                             "AP span x r, r = r_lower + (r_upper - r_lower) x Beta(a, b), independent of AP span")
            row.update(r_lower=lo, r_upper=hi, r_beta_a=a_, r_beta_b=b_, r_mean=float(p["r_mean"]),
                       method="closed form" if ax == "AP" else "Gauss-Legendre quadrature over r (400 nodes)")
            s = o[(o["sample"] == "amend3_pooled") & (o.scenario == sid) & (o.axis == ax) & (o.rule == "A1")]
            for c in cuts:
                t = s[s.cutoff_mm == c].iloc[0]
                k, n = int(t.tp + t.fn), int(t.n)
                row[f"sim_n_ge_{c:g}mm"] = k
                row[f"sim_prop_ge_{c:g}mm"] = k / n
                row[f"sim_prop_ge_{c:g}mm_mcse"] = np.sqrt(k / n * (1 - k / n) / n)
            row["sim_n"] = int(s.n.iloc[0])
            if sid == "base":   # empirical quantiles of the 100,000 published base-case patients
                for q, nm in (("0.025", "p2_5"), ("0.25", "p25"), ("0.5", "median"), ("0.75", "p75"),
                              ("0.975", "p97_5"), ("mean", "mean"), ("sd", "sd")):
                    row[f"published_seed_empirical_{nm}_mm"] = float(emp_q.loc[q, ax + "_mm"])
            rows.append(row)
    dist = pd.DataFrame(rows)
    dist.to_csv(OUT / "thresholds_true_span_distribution.csv", index=False)

    x = np.round(np.arange(0.0, 40.0001, 0.1), 1)
    h = load("hist", oc)
    h = h[(h["sample"] == "amend3_pooled") & h.scenario.isin(["base", "base_median13"])]
    dens = []
    for pop, p, sid in (("median10", p10, "base"), ("median13", p13, "base_median13")):
        for ax in T.AXES:
            dens.append(pd.DataFrame({"kind": "density", "population": pop, "axis": ax, "x_mm": x,
                                      "lo_mm": np.nan, "hi_mm": np.nan, "n": np.nan,
                                      "density_per_mm": T.span_pdf(p, ax, x), "cdf": T.span_cdf(p, ax, x)}))
            hh = h[(h.scenario == sid) & (h.axis == ax)].sort_values("lo_mm")
            ntot = hh.n.sum()
            hh = hh[np.isfinite(hh.hi_mm) & (hh.hi_mm <= 40.0)]
            dens.append(pd.DataFrame({"kind": "histogram", "population": pop, "axis": ax,
                                      "x_mm": (hh.lo_mm + hh.hi_mm) / 2, "lo_mm": hh.lo_mm, "hi_mm": hh.hi_mm,
                                      "n": hh.n, "density_per_mm": hh.n / ntot / (hh.hi_mm - hh.lo_mm),
                                      "cdf": np.cumsum(hh.n) / ntot}))
    dens = pd.concat(dens, ignore_index=True)
    dens.to_csv(OUT / "thresholds_true_span_density.csv", index=False)

    # ------------------------------------------------------------------ reproduction of published values
    files = sorted(glob.glob(str(ROOT / "results" / "2026-09-18_full" / "E3_cells_*.parquet")))
    st = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    st = st[st.cell.isin([29, 77]) & st.estimator.isin(list(RULE_LABEL)) &
            st.metric.isin(["sensitivity", "specificity", "ppv", "prevalence"])]
    st["scenario"] = st.cell.map({29: "base", 77: "low8"})
    st = st.rename(columns={"estimator": "rule", "value": "stored_value", "mcse": "stored_mcse", "n": "stored_n"})
    long = o.melt(id_vars=["sample", "scenario", "axis", "cutoff_mm", "rule"],
                  value_vars=["sensitivity", "specificity", "ppv", "prevalence"], var_name="metric")
    se = o.melt(id_vars=["sample", "scenario", "axis", "cutoff_mm", "rule"],
                value_vars=[v + "_mcse" for v in ("sensitivity", "specificity", "ppv", "prevalence")],
                var_name="metric", value_name="mcse")
    long["mcse"] = se["mcse"].to_numpy()
    key = ["scenario", "axis", "cutoff_mm", "rule", "metric"]
    rp = st[key + ["cell", "stored_value", "stored_mcse", "stored_n"]]
    rp = rp.merge(long[long["sample"] == "published_seed"][key + ["value"]].rename(columns={"value": "rebuilt_value"}),
                  on=key)
    rp = rp.merge(long[long["sample"] == "amend3_pooled"][key + ["value", "mcse"]].rename(
        columns={"value": "amend3_pooled_value", "mcse": "amend3_pooled_mcse"}), on=key)
    rp["rebuilt_identical"] = rp.stored_value.to_numpy() == rp.rebuilt_value.to_numpy()
    rp["stored_minus_pooled"] = rp.stored_value - rp.amend3_pooled_value
    rp["z_stored_vs_pooled"] = rp.stored_minus_pooled / np.sqrt(rp.stored_mcse ** 2 + rp.amend3_pooled_mcse ** 2)
    rp.to_csv(OUT / "thresholds_reproduction.csv", index=False)

    # ------------------------------------------------------------------ between-replicate SD vs binomial MCSE
    rep = pd.read_parquet(RAW / "replicates_overall.parquet")
    mm = T.classification_metrics(rep.tp, rep.fp, rep.fn, rep.tn)
    for k in ("sensitivity", "specificity", "ppv", "npv"):
        rep[k] = np.asarray(mm[k])
        rep[k + "_mcse"] = np.asarray(mm[k + "_mcse"])
    agg = {}
    for k in ("sensitivity", "specificity", "ppv", "npv"):
        agg[k + "_mean"] = (k, "mean")
        agg[k + "_sd_between_replicates"] = (k, lambda v: v.std(ddof=1))
        agg[k + "_binomial_mcse_mean"] = (k + "_mcse", "mean")
    rc = rep.groupby(["scenario", "axis", "cutoff_mm", "rule"], sort=False).agg(n_replicates=("replicate", "nunique"),
                                                                                **agg).reset_index()
    for k in ("sensitivity", "specificity", "ppv", "npv"):
        rc[k + "_sd_over_mcse"] = rc[k + "_sd_between_replicates"] / rc[k + "_binomial_mcse_mean"]
    rc.to_csv(OUT / "thresholds_replicate_check.csv", index=False)

    # ------------------------------------------------------------------ console summary
    f = lambda v: f"{100 * v:.1f}"
    print("true-span distribution\n", dist[["population", "axis", "mean_mm", "sd_mm", "p2_5_mm", "p25_mm", "median_mm",
                                             "p75_mm", "p97_5_mm"] + [f"prop_ge_{c:g}mm" for c in cuts]
                                            + [f"sim_prop_ge_{c:g}mm" for c in cuts]].round(4).to_string())
    s = o[(o["sample"] == "published_seed") & (o.axis == "AP")]
    print("\npublished-seed AP: per 1,000 and metrics")
    print(s[["scenario", "cutoff_mm", "rule", "tp", "fp", "fn", "tn", "sensitivity", "specificity", "ppv", "npv",
             "nri_vs_A1", "net_correct_per_1000_vs_A1"]].round(4).to_string())
    print("\nreproduction: all rebuilt identical:", bool(rp.rebuilt_identical.all()),
          "| max |z| stored vs pooled:", float(rp.z_stored_vs_pooled.abs().max()))
    t = rp[(rp.scenario == "base") & (rp.axis == "AP") & (rp.cutoff_mm == 13) & rp.metric.isin(["sensitivity", "specificity"])]
    print(t[["rule", "metric", "stored_value", "rebuilt_value", "amend3_pooled_value", "z_stored_vs_pooled"]].to_string())
    print("\nSD/MCSE ratio range:", rc[[c for c in rc.columns if c.endswith("_sd_over_mcse")]].stack().describe().round(3).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
