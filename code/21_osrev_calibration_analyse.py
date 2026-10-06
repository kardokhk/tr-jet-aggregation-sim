#!/usr/bin/env python
"""Amendment 3, package A3-6 (calibration): summary tables from the merged simulation output.

  python code/21_osrev_calibration_analyse.py --out results/2026-10-05_osrev/calibration

Reads orifice_fit.csv, geom_fit.csv, geom_checks.csv, geom_underestimation.csv, geom_bootstrap.csv,
libmatch.csv, viewrules_metrics.csv, repro_e1b_resimulated.csv (written by
code/20_osrev_calibration_run.py) and the stored E1b cells in results/2026-09-18_amend2.
Writes singh_reproduction.csv, repro_published_check.csv, geom_bootstrap_summary.csv,
calibration_summary.csv, calibration_variant_range.csv, identifiability_ellipse.csv,
viewrules_ranges.csv and viewrules_table_ap.csv. Deterministic; no
random numbers.

  python code/21_osrev_calibration_analyse.py --out results/2026-10-05_osrev/calibration --followup

Follow-up of 2026-10-06 (stages C and D of the runner). Leaves every table above untouched, reads
them together with followup_sd_*.csv, followup_matched_values.json and
followup_viewrules_metrics.csv, and writes followup_checks.csv, followup_sd_sensitivity.csv,
followup_dispersion.csv, followup_viewrules_ranges.csv, followup_viewrules_table_ap.csv and
calibration_table_s25.csv (the tidy source of Supplementary Table S25).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from duomaxsim import osrev_calibration as K  # noqa: E402

PAPER_LONG_AXIS_MEANS = (0.08, 0.20, 0.40)       # manuscript: mean long-axis underestimation scenarios
PUBLISHED_RANGE = {"A3": (-0.77, 0.92), "A2": (-3.16, -0.28), "A1": (-1.93, 0.02)}   # manuscript v05, AP bias, mm


def singh_reproduction():
    S = K.SINGH
    dmax, dmin, davg = S["d3_max"][0], S["d3_min"][0], S["d3_avg"][0]
    rows = []

    def add(statement, view, num, den, status, note=""):
        rows.append(dict(statement=statement, view=view, numerator_mm=num, denominator_mm=den,
                         percent=100.0 * num / den, status=status, note=note))

    for v in K.VIEWS:
        add("biplane below 3D average (Table 3 mean difference / mean 3D average)", v, S["ba_biplane_vs_avg"][v][0],
            davg, "percentage reported by the authors (24, 19, 25); reproduced")
        add("biplane below 3D average (difference of Table 2 means)", v, davg - S["views"][v]["biplane"][0], davg,
            "derived")
        add("single plane below 3D maximum (Table 3 mean difference / mean 3D maximum)", v, S["ba_single_vs_max"][v][0],
            dmax, "derived (mm difference reported, percentage not)")
        add("single plane below 3D minimum (Table 3 mean difference / mean 3D minimum)", v, S["ba_single_vs_min"][v][0],
            dmin, "derived")
        add("orthogonal plane below 3D maximum (difference of Table 2 means)", v, dmax - S["views"][v]["orthogonal"][0],
            dmax, "derived; no paired statistic published")
        add("orthogonal plane below 3D minimum (difference of Table 2 means)", v, dmin - S["views"][v]["orthogonal"][0],
            dmin, "derived; no paired statistic published")
        p, o = S["views"][v]["primary"][0], S["views"][v]["orthogonal"][0]
        f = np.sqrt((1 / dmax ** 2 + 1 / dmin ** 2) / (1 / p ** 2 + 1 / o ** 2))
        rows.append(dict(statement="linear factor by which the 2D pair is shorter than any orthogonal pair of central "
                                   "chords of the mean ellipse", view=v, numerator_mm=np.nan, denominator_mm=np.nan,
                         percent=100.0 * (1 - f), status="derived (means only)", note=f"factor {f:.4f}"))
    add("three-view mean, biplane below 3D average", "all", np.mean([S["ba_biplane_vs_avg"][v][0] for v in K.VIEWS]),
        davg, "derived")
    add("minor below major diameter (ellipticity alone)", "3D", dmax - dmin, dmax, "derived")
    return pd.DataFrame(rows)


def repro_check(out: Path, oc):
    # round_trip: the default pandas float parser is not exact in the last digit
    new = pd.read_csv(out / "repro_e1b_resimulated.csv", float_precision="round_trip")
    st = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "results/2026-09-18_amend2/E1_cells_*.parquet"))))
    st = st[st.cell.isin(new.cell.unique()) & (st.estimand == "T1") & st.estimator.isin(K.RULES)]
    st = st[(st.estimator != "A4") | ((st.s_det == 0.8) & (st.f_rej == 0.1))]
    m = new.merge(st[["cell", "estimator", "axis", "metric", "value", "mcse"]].rename(
        columns={"estimator": "rule", "value": "stored_value", "mcse": "stored_mcse"}),
        on=["cell", "rule", "axis", "metric"], how="left", validate="one_to_one")
    assert m.stored_value.notna().all()
    rows = []
    for (rule, axis, metric), g in m.groupby(["rule", "axis", "metric"]):
        rows.append(dict(rule=rule, axis=axis, metric=metric, n_cells=len(g),
                         stored_min=g.stored_value.min(), stored_max=g.stored_value.max(),
                         resimulated_min=g.value.min(), resimulated_max=g.value.max(),
                         max_abs_diff_value=float(np.abs(g.value - g.stored_value).max()),
                         max_abs_diff_mcse=float(np.abs(g.mcse - g.stored_mcse).max()),
                         n_bit_identical=int(((g.value == g.stored_value) & (g.mcse == g.stored_mcse)).sum()),
                         manuscript_min=PUBLISHED_RANGE.get(rule, (np.nan,) * 2)[0] if (axis, metric) == ("AP", "bias_mm") else np.nan,
                         manuscript_max=PUBLISHED_RANGE.get(rule, (np.nan,) * 2)[1] if (axis, metric) == ("AP", "bias_mm") else np.nan))
    return pd.DataFrame(rows)


def boot_summary(out: Path):
    b = pd.read_csv(out / "geom_bootstrap.csv")
    err = b[b.quantity == "fit_error"]
    b = b[b.quantity != "fit_error"]
    conv = b[b.quantity == "converged"].set_index(["variant", "replicate", "view"]).value
    b = b.join(conv.rename("conv"), on=["variant", "replicate", "view"])
    rows = []
    for (var, view, q), g in b.groupby(["variant", "view", "quantity"]):
        ok = g[g.conv == 1.0].value.to_numpy()
        rows.append(dict(variant=var, view=view, quantity=q, replicates=len(g), converged=int(len(ok)),
                         fit_errors=int((err.variant == var).sum()),
                         mean=float(ok.mean()) if len(ok) else np.nan, sd=float(ok.std(ddof=1)) if len(ok) > 1 else np.nan,
                         q2_5=float(np.percentile(ok, 2.5)) if len(ok) else np.nan,
                         q50=float(np.percentile(ok, 50)) if len(ok) else np.nan,
                         q97_5=float(np.percentile(ok, 97.5)) if len(ok) else np.nan,
                         all_replicates_q2_5=float(np.percentile(g.value, 2.5)),
                         all_replicates_q97_5=float(np.percentile(g.value, 97.5))))
    return pd.DataFrame(rows)


def calibration_summary(out: Path, bs: pd.DataFrame):
    u = pd.read_csv(out / "geom_underestimation.csv")
    f = pd.read_csv(out / "geom_fit.csv").set_index(["variant", "view"])
    want = [  # label, view, plane, reference, component, bootstrap quantity
        ("AP axis, inflow primary plane vs 3D maximum (rotation and offset)", "inflow", "primary", "3D maximal diameter", "total", "primary_vs_max_total"),
        ("AP axis, inflow primary plane, rotation component", "inflow", "primary", "3D maximal diameter", "rotation", "primary_vs_max_rotation"),
        ("AP axis, inflow primary plane, off-centre component (plane on axis)", "inflow", "primary", "central chord in plane direction", "off_centre", "primary_off_centre"),
        ("AP axis, 4CH orthogonal plane vs 3D maximum", "4CH", "orthogonal", "3D maximal diameter", "total", "orthogonal_vs_max_total"),
        ("AP axis, mBC orthogonal plane vs 3D maximum", "mBC", "orthogonal", "3D maximal diameter", "total", "orthogonal_vs_max_total"),
        ("AP axis, 4CH orthogonal plane, off-centre component", "4CH", "orthogonal", "central chord in plane direction", "off_centre", "orthogonal_off_centre"),
        ("AP axis, mBC orthogonal plane, off-centre component", "mBC", "orthogonal", "central chord in plane direction", "off_centre", "orthogonal_off_centre"),
        ("SL axis, 4CH primary plane vs 3D minimum (rotation and offset)", "4CH", "primary", "3D minimal diameter", "total", "primary_vs_min_total"),
        ("SL axis, 4CH primary plane, off-centre component (plane on axis)", "4CH", "primary", "central chord in plane direction", "off_centre", "primary_off_centre"),
        ("SL axis, mBC primary plane vs 3D minimum (rotation and offset)", "mBC", "primary", "3D minimal diameter", "total", "primary_vs_min_total"),
        ("SL axis, mBC primary plane, off-centre component (plane on axis)", "mBC", "primary", "central chord in plane direction", "off_centre", "primary_off_centre"),
        ("SL axis, inflow orthogonal plane vs 3D minimum", "inflow", "orthogonal", "3D minimal diameter", "total", None),
        ("cross-plane maximum of the inflow view vs 3D maximum", "inflow", "cross-plane maximum", "3D maximal diameter", "total", None),
        ("cross-plane maximum of the 4CH view vs 3D maximum", "4CH", "cross-plane maximum", "3D maximal diameter", "total", None),
        ("published estimand: inflow biplane vs 3D average", "inflow", "biplane", "3D average of maximal and minimal diameter", "total", "biplane_vs_3davg_ratio_of_means"),
        ("published estimand: 4CH biplane vs 3D average", "4CH", "biplane", "3D average of maximal and minimal diameter", "total", "biplane_vs_3davg_ratio_of_means"),
        ("published estimand: mBC biplane vs 3D average", "mBC", "biplane", "3D average of maximal and minimal diameter", "total", "biplane_vs_3davg_ratio_of_means"),
    ]
    bsi = bs.set_index(["variant", "view", "quantity"]) if len(bs) else None
    rows = []
    for var in u.variant.unique():
        for lab, view, plane, ref, comp, bq in want:
            r = u[(u.variant == var) & (u["view"] == view) & (u.plane == plane) & (u.reference == ref) & (u.component == comp)]
            assert len(r) == 1, (var, lab)
            r = r.iloc[0]
            row = dict(variant=var, quantity=lab, view=view, plane=plane, reference=ref, component=comp,
                       fit_converged=bool(f.loc[(var, view), "converged"]),
                       mean_fraction=r["mean"], mean_mcse=r["mean_mcse"], sd_between_patients=r["sd"], sd_mcse=r["sd_mcse"],
                       q2_5=r["q2.5"], q25=r["q25"], q50=r["q50"], q75=r["q75"], q97_5=r["q97.5"],
                       ratio_of_means=r["ratio_of_means"], ratio_of_means_mcse=r["ratio_of_means_mcse"],
                       count_negative=int(r["count_negative"]), n=int(r["n"]),
                       halfnormal_scale_same_mean=r["mean"] / K.HALF_NORMAL_MEAN)
            key = (var, view, bq)
            if bq is not None and bsi is not None and key in bsi.index:
                b = bsi.loc[key]
                row.update(target_sampling_q2_5=b["all_replicates_q2_5"], target_sampling_q97_5=b["all_replicates_q97_5"],
                           target_sampling_converged_only_q2_5=b["q2_5"], target_sampling_converged_only_q97_5=b["q97_5"],
                           bootstrap_converged=int(b["converged"]), bootstrap_replicates=int(b["replicates"]))
            rows.append(row)
    return pd.DataFrame(rows)


def variant_range(cs: pd.DataFrame) -> pd.DataFrame:
    """Smallest and largest mean fraction across the geometry variants whose fit reproduced the
    published means for that view (non-converged fits and the random-orientation variant excluded)."""
    ok = cs[cs.fit_converged]
    rows = []
    for q, g in ok.groupby("quantity", sort=False):
        i0, i1 = g.mean_fraction.idxmin(), g.mean_fraction.idxmax()
        rows.append(dict(quantity=q, n_variants=len(g), variants=";".join(g.variant),
                         min_mean_fraction=g.mean_fraction[i0], min_mcse=g.mean_mcse[i0], min_variant=g.variant[i0],
                         max_mean_fraction=g.mean_fraction[i1], max_mcse=g.mean_mcse[i1], max_variant=g.variant[i1],
                         min_sd_between_patients=g.sd_between_patients.min(),
                         max_sd_between_patients=g.sd_between_patients.max(),
                         excluded_variants=";".join(sorted(set(cs.variant) - set(g.variant)))))
    return pd.DataFrame(rows)


def identifiability_table() -> pd.DataFrame:
    """Deterministic illustration for the mean published ellipse (16.82 x 8.99 mm): quantities along
    an anatomical axis at angle phi to the major axis, and the cross-plane maximum of a central
    orthogonal plane pair whose primary plane is at angle phi to the major axis."""
    dmax, dmin = K.SINGH["d3_max"][0], K.SINGH["d3_min"][0]
    rows = []
    for deg in (0, 10, 20, 30, 45, 60, 75, 90):
        phi = np.radians(deg)
        c_ax, c_perp = float(K.central_chord(dmax, dmin, phi)), float(K.central_chord(dmax, dmin, phi + np.pi / 2))
        rows.append(dict(angle_to_major_axis_deg=deg, d3_max_mm=dmax, d3_min_mm=dmin,
                         longest_chord_along_axis_mm=c_ax, projected_extent_along_axis_mm=float(K.feret_width(dmax, dmin, phi)),
                         longest_chord_along_perpendicular_axis_mm=c_perp,
                         cross_plane_maximum_central_pair_mm=max(c_ax, c_perp),
                         biplane_average_central_pair_mm=0.5 * (c_ax + c_perp),
                         longest_chord_below_d3_max_fraction=1 - c_ax / dmax,
                         cross_plane_maximum_below_d3_max_fraction=1 - max(c_ax, c_perp) / dmax,
                         biplane_average_below_d3_average_fraction=1 - (c_ax + c_perp) / (dmax + dmin)))
    return pd.DataFrame(rows)


def viewrules_tables(out: Path):
    v = pd.read_csv(out / "viewrules_metrics.csv")
    rows = []
    pub = v[v.family == "published"]
    for (sc, fam, axis, rule, metric), g in list(v.groupby(["scenario", "family", "axis", "rule", "metric"], sort=False)) + \
            [(("published_12_scenarios_new_seed", "published", a, r, m), gg)
             for (a, r, m), gg in pub.groupby(["axis", "rule", "metric"], sort=False)]:
        i0, i1 = g.value.idxmin(), g.value.idxmax()
        ref = pub[(pub.axis == axis) & (pub.rule == rule) & (pub.metric == metric)].value
        rows.append(dict(scenario=sc, family=fam, axis=axis, rule=rule, metric=metric, n_anchor_levels=len(g),
                         long_axis=g[f"long_{axis}"].iloc[0] if fam != "published" or sc != "published_12_scenarios_new_seed" else "0.10, 0.25, 0.50",
                         mean_u_long_min=g.mean_u_long.min(), mean_u_long_max=g.mean_u_long.max(),
                         min_value=g.value.min(), min_mcse=g.mcse[i0], min_at_u_anchor=g.u_anchor[i0],
                         max_value=g.value.max(), max_mcse=g.mcse[i1], max_at_u_anchor=g.u_anchor[i1],
                         published12_new_seed_min=ref.min(), published12_new_seed_max=ref.max(),
                         within_published12_range=bool((g.value.min() >= ref.min()) and (g.value.max() <= ref.max()))))
    ranges = pd.DataFrame(rows)
    ap = v[(v.axis == "AP") & v.metric.isin(["bias_mm", "rmse_mm", "mae_mm"])].copy()
    wide = ap.pivot_table(index=["scenario_index", "scenario", "family", "u_anchor", "long_AP", "mean_u_anchor", "mean_u_long", "n"],
                          columns=["rule", "metric"], values=["value", "mcse", "diff_vs_base", "diff_vs_base_mcse"])
    wide.columns = [f"{r}_{m}_{w}" if w != "value" else f"{r}_{m}" for w, r, m in wide.columns]
    wide = wide.reset_index().sort_values("scenario_index")
    order = ["scenario_index", "scenario", "family", "u_anchor", "long_AP", "mean_u_anchor", "mean_u_long", "n"]
    order += sorted(c for c in wide.columns if c not in order)
    return ranges, wide[order]


# ------------------------------------------------------------------ follow-up 2026-10-06

RULE_NAMES = {"A1": "anchor-view mean", "A2": "mean across views", "A3": "largest view mean",
              "A4": "largest view mean after review"}
STAGE_B_SHAPES = ("r1_halfnormal", "r1_chord", "r2_halfnormal", "r2_chord")
HN_0510 = ["lib_biplane_inflow", "lib_biplane_mean_three_views", "lib_biplane_mBC", "geom_ind15_r1_halfnormal",
           "geom_ind15_r2_halfnormal", "geom_mid15_r1_halfnormal", "geom_mid15_r2_halfnormal"]
VARS = ("ind15", "mid15", "ind00")
SD707 = ("ind15_sd707", "mid15_sd707", "ind00_sd707")


def _g(shape, variants=VARS):
    return [f"geom_{v}_{shape}" for v in variants]


# group, variant label, scenarios, description
GROUPS = [
    ("halfnormal_calibrated_run_2026-10-05", "all", HN_0510,
     "half-normal shape at calibrated means, the 28 scenarios of 2026-10-05 (mean underestimation 8.3% to 29.7%)"),
    ("halfnormal_lowest_calibrated", "all", ["geom_ind00_r2_halfnormal"],
     "half-normal shape at the lowest calibrated mean (off-centre component, variant ind00)"),
    ("halfnormal_calibrated_all", "all", HN_0510 + ["geom_ind00_r1_halfnormal", "geom_ind00_r2_halfnormal"],
     "half-normal shape at calibrated means, all values including the lowest (36 scenarios)"),
    ("patient_geometry_reading1", "all", _g("r1_patient"),
     "patient-level geometric shape, rotation counted as error (variants ind15, mid15, ind00)"),
    ("patient_geometry_reading2", "all", _g("r2_patient"),
     "patient-level geometric shape, off-centre placement only (variants ind15, mid15, ind00); identical to the chord-distributed offset"),
    ("patient_geometry_both_readings", "all", _g("r1_patient") + _g("r2_patient"),
     "patient-level geometric shape, both readings (24 scenarios)"),
    ("all_calibrated_primary", "all", HN_0510 + ["geom_ind00_r1_halfnormal", "geom_ind00_r2_halfnormal"] + _g("r1_patient") + _g("r2_patient"),
     "half-normal and patient-level geometric shapes at all calibrated values (60 scenarios)"),
    ("constant_rotation_chord_reading1", "all", _g("r1_chord"),
     "constant rotation plus chord-distributed offset (between-patient SD about half the calibrated one; not a calibrated shape)"),
    ("patient_geometry_reading1_independent_offsets", "all", _g("r1_patient_indoff"),
     "sensitivity: patient-level geometric shape, reading 1, offsets independent between views"),
    ("patient_geometry_reading2_independent_offsets", "all", _g("r2_patient_indoff"),
     "sensitivity: patient-level geometric shape, reading 2, offsets independent between views"),
    ("patient_geometry_reading1_span_linked_ellipticity", "all", _g("r1_patient_spanq"),
     "sensitivity: patient-level geometric shape, reading 1, ellipticity linked to the true AP span"),
    ("sd707_halfnormal", "all", _g("r1_halfnormal", SD707) + _g("r2_halfnormal", SD707),
     "SD of the 3D maximal diameter 7.07 mm: half-normal shape at the recalibrated means (24 scenarios)"),
    ("sd707_patient_geometry_reading1", "all", _g("r1_patient", SD707),
     "SD of the 3D maximal diameter 7.07 mm: patient-level geometric shape, reading 1"),
    ("sd707_patient_geometry_reading2", "all", _g("r2_patient", SD707),
     "SD of the 3D maximal diameter 7.07 mm: patient-level geometric shape, reading 2"),
    ("sd707_all", "all", _g("r1_halfnormal", SD707) + _g("r2_halfnormal", SD707) + _g("r1_patient", SD707) + _g("r2_patient", SD707),
     "SD of the 3D maximal diameter 7.07 mm: half-normal and patient-level geometric shapes (48 scenarios)"),
]
GROUPS += [(f"patient_geometry_reading{r}", v, [f"geom_{v}_r{r}_patient"],
            f"patient-level geometric shape, reading {r}, variant {v}") for r in (1, 2) for v in VARS]


def followup_checks(out: Path, v0: pd.DataFrame, fv: pd.DataFrame, fm: dict) -> pd.DataFrame:
    rows = []
    key = ["scenario", "u_anchor", "axis", "rule", "metric"]
    cols = ["value", "mcse", "diff_vs_base", "diff_vs_base_mcse", "mean_u_anchor", "mean_u_long"]
    r0 = fv[fv.replicate == 0]
    # (a) stage B scenarios re-simulated in stage D on the same stream
    old = v0[v0.family == "geometric_match"]
    m = old.merge(r0[r0["shape"].isin(STAGE_B_SHAPES)], on=key, suffixes=("_stored", "_new"), validate="one_to_one")
    same = np.ones(len(m), bool)
    dmax = 0.0
    for c in cols:
        a, b = m[c + "_stored"].to_numpy(), m[c + "_new"].to_numpy()
        same &= (a == b) | (np.isnan(a) & np.isnan(b))
        dmax = max(dmax, float(np.nanmax(np.abs(a - b))))
    rows.append(dict(check="stage B geometric scenarios (ind15, mid15) re-simulated in stage D: stored and new csv values identical",
                     n_expected=len(old), n_compared=len(m), n_identical=int(same.sum()), max_abs_diff=dmax,
                     passed=bool(len(m) == len(old) and same.all())))
    # (b) reading 2 with latent offsets equals the chord-distributed offset
    a = fv[fv["shape"] == "r2_chord"].set_index(["replicate", "variant", "u_anchor", "axis", "rule", "metric"])[cols[:4]]
    b = fv[fv["shape"] == "r2_patient"].set_index(["replicate", "variant", "u_anchor", "axis", "rule", "metric"])[cols[:4]]
    b = b.loc[a.index]
    eq = ((a.to_numpy() == b.to_numpy()) | (np.isnan(a.to_numpy()) & np.isnan(b.to_numpy()))).all(axis=1)
    rows.append(dict(check="patient-level draw, reading 2, latent offsets equals the chord-distributed offset scenario",
                     n_expected=len(a), n_compared=len(b), n_identical=int(eq.sum()),
                     max_abs_diff=float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))), passed=bool(eq.all())))
    # (c) stage C with the stage A reading of the confidence interval reproduces stage A
    su = pd.read_csv(out / "followup_sd_underestimation.csv")
    gu = pd.read_csv(out / "geom_underestimation.csv")
    k2 = ["variant", "view", "plane", "reference", "reading", "component"]
    num = [c for c in gu.columns if c not in k2]
    lo = su[su.sd_max_side == "lower"]
    m = gu[gu.variant.isin(lo.variant.unique())].merge(lo, on=k2, suffixes=("_stored", "_new"), validate="one_to_one")
    same = np.ones(len(m), bool)
    dmax = 0.0
    for c in num:
        x, y = m[c + "_stored"].to_numpy(float), m[c + "_new"].to_numpy(float)
        same &= (x == y) | (np.isnan(x) & np.isnan(y))
        dmax = max(dmax, float(np.nanmax(np.abs(x - y))))
    rows.append(dict(check="stage C, lower half-width, reproduces geom_underestimation.csv of stage A (ind15, mid15, ind00)",
                     n_expected=int(gu.variant.isin(lo.variant.unique()).sum()), n_compared=len(m),
                     n_identical=int(same.sum()), max_abs_diff=dmax, passed=bool(same.all() and len(m) > 0)))
    # (d) anchor-view mean identical across all scenarios of a replicate at each anchor level
    a1 = fv[(fv.rule == "A1")].groupby(["replicate", "u_anchor", "axis", "metric"]).value.agg(lambda x: x.max() - x.min())
    rows.append(dict(check="anchor-view mean identical across scenarios within replicate and anchor level (common random numbers)",
                     n_expected=len(a1), n_compared=len(a1), n_identical=int((a1 == 0).sum()),
                     max_abs_diff=float(a1.max()), passed=bool((a1 == 0).all())))
    # (e) realized between-patient mean and SD of the patient-level fraction against the calibration
    for var, g in fm["geom"].items():
        for shape, mk, sk in (("r1_patient", "ap_total", "ap_total_sd"), ("r2_patient", "ap_off_centre", "ap_off_centre_sd")):
            r = r0[(r0.variant == var) & (r0["shape"] == shape) & (r0.axis == "AP")].iloc[0]
            d = max(abs(r.mean_u_long - g[mk]), abs(r.sd_u_long - g[sk]))
            rows.append(dict(check=f"{var} {shape}: realized mean and SD of the AP long-axis fraction ({r.mean_u_long:.4f}, {r.sd_u_long:.4f}) "
                                   f"against the calibration ({g[mk]:.4f}, {g[sk]:.4f}); tolerance 0.003",
                             n_expected=2, n_compared=2, n_identical=int(d == 0), max_abs_diff=float(d), passed=bool(d < 0.003)))
    return pd.DataFrame(rows)


def followup_sd_table(out: Path) -> pd.DataFrame:
    u = pd.read_csv(out / "followup_sd_underestimation.csv")
    f = pd.read_csv(out / "followup_sd_fit.csv")
    o = pd.read_csv(out / "followup_sd_orifice.csv")
    cc = "central chord in plane direction"
    want = [("inflow primary plane vs 3D maximum, reading 1 (rotation and offset)", "inflow", "primary", "3D maximal diameter", "total"),
            ("inflow primary plane, rotation component", "inflow", "primary", "3D maximal diameter", "rotation"),
            ("inflow primary plane, off-centre component (reading 2)", "inflow", "primary", cc, "off_centre"),
            ("4CH orthogonal plane vs 3D maximum, reading 1", "4CH", "orthogonal", "3D maximal diameter", "total"),
            ("mBC orthogonal plane vs 3D maximum, reading 1", "mBC", "orthogonal", "3D maximal diameter", "total"),
            ("4CH primary plane vs 3D minimum (SL axis), reading 1", "4CH", "primary", "3D minimal diameter", "total"),
            ("4CH primary plane, off-centre component (SL axis, reading 2)", "4CH", "primary", cc, "off_centre"),
            ("inflow biplane vs 3D average (published estimand, per-patient mean)", "inflow", "biplane", "3D average of maximal and minimal diameter", "total")]
    rows = []
    for (var, side), uu in u.groupby(["variant", "sd_max_side"], sort=False):
        oo = o[(o.variant == var) & (o.sd_max_side == side)].set_index("quantity").fitted_quadrature
        ff = f[(f.variant == var) & (f.sd_max_side == side)].set_index("view")
        for lab, view, plane, ref, comp in want:
            r = uu[(uu["view"] == view) & (uu.plane == plane) & (uu.reference == ref) & (uu.component == comp)]
            assert len(r) == 1, (var, side, lab)
            r = r.iloc[0]
            rows.append(dict(variant=var, sd_max_side=side, sd_max_mm=float(r.sd_max_mm), quantity=lab, view=view,
                             mean_fraction=r["mean"], mean_mcse=r["mean_mcse"], sd_between_patients=r["sd"],
                             ratio_of_means=r["ratio_of_means"], n=int(r["n"]),
                             fit_converged=bool(ff.loc[view, "converged"]), theta0_deg=float(ff.loc[view, "theta0_deg"]),
                             offset_scale=float(ff.loc[view, "value"]),
                             orifice_max_abs_residual_mm=float(oo["max_abs_residual"]),
                             implied_corr_dmax_dmin=float(oo["corr"]), implied_mean_sphericity_index=float(oo["mean_sphericity"]),
                             published_sphericity_index="2.04 (95% CI 1.73 to 2.35)"))
    return pd.DataFrame(rows)


def followup_ranges(v0: pd.DataFrame, fv: pd.DataFrame, rc: pd.DataFrame, stored_cells: pd.DataFrame):
    keep = ["scenario", "u_anchor", "axis", "rule", "metric", "value", "mcse", "mean_u_long"]
    stored = rc.set_index(["rule", "axis", "metric"])
    rows = []

    def add(group, var, desc, rep_, g, n_expected):
        for (axis, rule, metric), x in g.groupby(["axis", "rule", "metric"], sort=False):
            i0, i1 = x.value.idxmin(), x.value.idxmax()
            if (rule, axis, metric) in stored.index:       # the stored cells carry no MAE
                lo, hi = float(stored.loc[(rule, axis, metric), "stored_min"]), float(stored.loc[(rule, axis, metric), "stored_max"])
            else:
                lo = hi = np.nan
            rows.append(dict(group=group, variant=var, description=desc, replicate=rep_, axis=axis, rule=rule,
                             rule_name=RULE_NAMES[rule], metric=metric, n_scenarios=len(x), n_scenarios_expected=n_expected,
                             mean_u_long_min=x.mean_u_long.min(), mean_u_long_max=x.mean_u_long.max(),
                             sd_u_long_min=x.sd_u_long.min() if "sd_u_long" in x else np.nan,
                             sd_u_long_max=x.sd_u_long.max() if "sd_u_long" in x else np.nan,
                             min_value=x.value[i0], min_mcse=x.mcse[i0], min_at=f"{x.scenario[i0]} @ anchor {x.u_anchor[i0]:g}",
                             max_value=x.value[i1], max_mcse=x.mcse[i1], max_at=f"{x.scenario[i1]} @ anchor {x.u_anchor[i1]:g}",
                             largest_mcse=x.mcse.max(), stored12_min=lo, stored12_max=hi,
                             below_stored12_min_by=max(lo - x.value[i0], 0.0) if lo == lo else np.nan,
                             above_stored12_max_by=max(x.value[i1] - hi, 0.0) if hi == hi else np.nan,
                             bracketed_by_stored12=bool(x.value[i0] >= lo and x.value[i1] <= hi) if lo == lo else ""))

    sc = stored_cells.rename(columns={"cell": "scenario"}).assign(mean_u_long=np.nan)
    sc["scenario"] = "stored_cell_" + sc.scenario.astype(str)
    add("published_12_stored", "all", "the 12 view-accuracy scenarios as stored (results/2026-09-18_amend2), re-simulated bit-identically",
        -1, sc.reset_index(drop=True), 12)
    pub = v0[v0.family == "published"].reset_index(drop=True)
    add("published_12_new_seed", "all", "the 12 view-accuracy scenarios on the stage B stream (76, 2, 0)", 0, pub, 12)
    for rep_ in sorted(fv.replicate.unique()):
        pool = fv[fv.replicate == rep_][keep + ["sd_u_long"]]
        if rep_ == 0:
            extra = v0[~v0.scenario.isin(pool.scenario.unique())][keep]
            pool = pd.concat([pool, extra], ignore_index=True)
        have = set(pool.scenario.unique())
        for group, var, scen, desc in GROUPS:
            if not set(scen) <= have:
                continue
            g = pool[pool.scenario.isin(scen)].reset_index(drop=True)
            add(group, var, desc, rep_, g, 4 * len(scen))
    return pd.DataFrame(rows)


def followup_dispersion(fv: pd.DataFrame, fm: dict) -> pd.DataFrame:
    r = fv[(fv.rule == "A3") & (fv.metric == "bias_mm") & (fv.u_anchor == 0.06)]
    rows = []
    for (rep_, var, shape, axis), x in r.groupby(["replicate", "variant", "shape", "axis"], sort=False):
        x = x.iloc[0]
        g = fm["geom"][var]
        cal = {"r1": ("total", "total_sd"), "r2": ("off_centre", "off_centre_sd")}[shape[:2]]
        pre = "ap_" if axis == "AP" else "sl_"
        rows.append(dict(replicate=rep_, variant=var, shape=shape, axis=axis, mean_u_long=x.mean_u_long, sd_u_long=x.sd_u_long,
                         corr_u_between_long_axis_views=x.corr_u_long_views,
                         calibrated_mean=g.get(pre + cal[0], np.nan), calibrated_sd=g.get(pre + cal[1], np.nan), n=int(x.n)))
    return pd.DataFrame(rows)


def _mm(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "−")


def table_s25(out: Path, ranges: pd.DataFrame, sdt: pd.DataFrame) -> pd.DataFrame:
    """Tidy source of Supplementary Table S25. Percentages for fractions, mm for errors; `display`
    is the formatted cell (true minus sign); `interval_type` says what lower and upper are."""
    rows = []

    def add(part, item, condition, est=np.nan, lo=np.nan, hi=np.nan, mcse=np.nan, unit="%", n="", itype="", display="",
            bracket="", source="", note=""):
        rows.append(dict(part=part, item=item, condition=condition, estimate=est, lower=lo, upper=hi, mcse=mcse, unit=unit,
                         n=n, interval_type=itype, bracketed_by_12_scenarios=bracket, display=display, source=source, note=note))

    # A. published comparison
    sr = pd.read_csv(out / "singh_reproduction.csv")
    A = "A. Published comparison (Singh et al. 2026; 30 patients), reproduced from Tables 2 and 3"
    for r in sr.itertuples():
        if r.statement.startswith("linear factor") or "difference of Table 2 means)" in r.statement and r.statement.startswith("biplane"):
            continue
        if r.statement.startswith("orthogonal plane below 3D minimum"):
            continue
        add(A, r.statement, r.view, est=r.percent, unit="%", n=30, display=f"{r.percent:.1f}".replace("-", "−"),
            source="singh_reproduction.csv: percent",
            note=f"{r.numerator_mm:.2f} / {r.denominator_mm:.2f} mm; {r.status}")
    # B. corresponding quantity in the simulation model
    lm = pd.read_csv(out / "libmatch.csv")
    B = "B. Corresponding quantity in the simulation model (biplane width below the mean of the true AP and SL spans)"
    for r in lm[lm.kind == "grid"].itertuples():
        add(B, "biplane shortfall at long-axis scale", f"scale {r.L:g}; overestimation {'on' if r.view_over else 'off'}",
            est=100 * r.biplane_shortfall, mcse=100 * r.biplane_shortfall_mcse, n=int(r.n),
            display=f"{100 * r.biplane_shortfall:.1f} ({100 * r.biplane_shortfall_mcse:.2f})", source="libmatch.csv: biplane_shortfall",
            note=f"mean long-axis underestimation {100 * r.mean_u:.1f}%")
    for r in lm[lm.kind == "solved"].itertuples():
        add(B, "long-axis scale reproducing the published percentage",
            f"target {100 * r.target:.1f}% ({r.target_label}); overestimation {'on' if r.view_over else 'off'}",
            est=r.L, mcse=r.L_mcse, unit="scale", n=int(r.n), display=f"{r.L:.3f} ({r.L_mcse:.4f})", source="libmatch.csv: L, L_mcse",
            note=f"mean long-axis underestimation {100 * r.mean_u:.1f}%")
    # C. elliptical-orifice calibration
    cs = pd.read_csv(out / "calibration_summary.csv")
    vr = pd.read_csv(out / "calibration_variant_range.csv").set_index("quantity")
    C_ = "C. Elliptical-orifice calibration: mean fractional underestimation (SD of the 3D maximal diameter 5.73 mm)"
    for r in cs[cs.variant == "ind15"].itertuples():
        add(C_, r.quantity, "variant ind15 (primary); interval: target-sampling", est=100 * r.mean_fraction,
            lo=100 * r.target_sampling_q2_5, hi=100 * r.target_sampling_q97_5, mcse=100 * r.mean_mcse, n=int(r.n),
            itype="target-sampling interval (500 redraws of the published means)" if pd.notna(r.target_sampling_q2_5) else "",
            display=f"{100 * r.mean_fraction:.1f} ({100 * r.mean_mcse:.2f})".replace("-", "−"),
            source="calibration_summary.csv: mean_fraction, mean_mcse, target_sampling_q2_5, target_sampling_q97_5",
            note=f"between-patient SD {100 * r.sd_between_patients:.1f}%")
        if r.quantity in vr.index:
            q = vr.loc[r.quantity]
            add(C_, r.quantity, f"range of means over {int(q.n_variants)} fitted variants", lo=100 * q.min_mean_fraction,
                hi=100 * q.max_mean_fraction, mcse=100 * max(q.min_mcse, q.max_mcse), n=int(r.n), itype="range across variants",
                display=f"{100 * q.min_mean_fraction:.1f} to {100 * q.max_mean_fraction:.1f}".replace("-", "−"),
                source="calibration_variant_range.csv: min_mean_fraction, max_mean_fraction",
                note=f"smallest in {q.min_variant}, largest in {q.max_variant}")
    C2 = "C2. Sensitivity to the reading of the misprinted confidence interval of the 3D maximal diameter (14.68 to 19.96 about 16.82 mm)"
    side_lab = {"lower": "lower half-width (main choice)", "both": "half the full width", "upper": "upper half-width"}
    for r in sdt.itertuples():
        add(C2, r.quantity, f"SD {r.sd_max_mm:.2f} mm ({side_lab[r.sd_max_side]}); variant {r.variant}", est=100 * r.mean_fraction,
            mcse=100 * r.mean_mcse, n=int(r.n), display=f"{100 * r.mean_fraction:.1f} ({100 * r.mean_mcse:.2f})".replace("-", "−"),
            source="followup_sd_sensitivity.csv: mean_fraction, mean_mcse",
            note=f"between-patient SD {100 * r.sd_between_patients:.1f}%; implied correlation of maximal and minimal diameter "
                 f"{r.implied_corr_dmax_dmin:.2f}; implied mean sphericity index {r.implied_mean_sphericity_index:.2f} "
                 f"(published {r.published_sphericity_index}); fit converged: {r.fit_converged}")
    # D. view rules
    D = "D. View rules, AP axis: range across scenarios (four anchor levels; 100,000 virtual patients per scenario; common random numbers)"
    order = ["published_12_stored", "halfnormal_calibrated_run_2026-10-05", "halfnormal_lowest_calibrated", "halfnormal_calibrated_all",
             "patient_geometry_reading1", "patient_geometry_reading2", "patient_geometry_both_readings", "all_calibrated_primary",
             "constant_rotation_chord_reading1", "patient_geometry_reading1_independent_offsets",
             "patient_geometry_reading2_independent_offsets", "patient_geometry_reading1_span_linked_ellipticity",
             "sd707_halfnormal", "sd707_patient_geometry_reading1", "sd707_patient_geometry_reading2", "sd707_all"]
    rr = ranges[(ranges.axis == "AP") & ranges.metric.isin(["bias_mm", "rmse_mm"]) & ranges.replicate.isin([-1, 0])]
    for grp in order:
        for var in ["all"] + (list(VARS) if grp in ("patient_geometry_reading1", "patient_geometry_reading2") else []):
            for metric in ("bias_mm", "rmse_mm"):
                for rule in ("A3", "A4", "A2", "A1"):
                    x = rr[(rr.group == grp) & (rr.variant == var) & (rr.metric == metric) & (rr.rule == rule)]
                    assert len(x) == 1, (grp, var, metric, rule, len(x))
                    x = x.iloc[0]
                    mu = "" if np.isnan(x.mean_u_long_min) else \
                        f"mean long-axis underestimation {100 * x.mean_u_long_min:.1f}% to {100 * x.mean_u_long_max:.1f}%; "
                    add(D, f"{RULE_NAMES[rule]}, {'bias' if metric == 'bias_mm' else 'RMSE'}", x.description,
                        lo=x.min_value, hi=x.max_value, mcse=x.largest_mcse, unit="mm", n=f"{int(x.n_scenarios)} scenarios x 100,000",
                        itype="range across scenarios", bracket="reference" if grp == "published_12_stored" else str(bool(x.bracketed_by_stored12)),
                        display=f"{_mm(x.min_value)} to {_mm(x.max_value)}",
                        source="repro_published_check.csv: stored_min, stored_max" if grp == "published_12_stored"
                        else f"followup_viewrules_ranges.csv: group {grp}, variant {var}, replicate 0",
                        note=f"{mu}minimum at {x.min_at}; maximum at {x.max_at}; 12-scenario range {_mm(x.stored12_min)} to {_mm(x.stored12_max)}"
                             + (f"; below it by {x.below_stored12_min_by:.3f} mm" if x.below_stored12_min_by > 0 else "")
                             + (f"; above it by {x.above_stored12_max_by:.3f} mm" if x.above_stored12_max_by > 0 else ""))
    t = pd.DataFrame(rows)
    # `core`: the rows suggested for the printed table; the others are supporting detail
    core_d = ("the 12 view-accuracy scenarios as stored", "half-normal shape at calibrated means, all values",
              "patient-level geometric shape, rotation counted as error (variants", "patient-level geometric shape, off-centre placement only (variants",
              "SD of the 3D maximal diameter 7.07 mm: half-normal and patient-level")
    t["core"] = (
        t.part.str.startswith("A.")
        | (t.part.str.startswith("B.") & t.condition.str.contains("overestimation on") & ~t.condition.str.contains("scale 0.1|scale 0.5|4CH"))
        | (t.part.str.startswith("C.") & t.item.str.contains("inflow primary plane"))
        | (t.part.str.startswith("C2.") & t.item.str.contains("inflow primary plane") & t.condition.str.contains("SD 7.07"))
        | (t.part.str.startswith("D.") & t.item.str.endswith("bias") & t.condition.str.startswith(core_d)))
    return t


def followup(out: Path):
    ff = "%.10g"
    v0 = pd.read_csv(out / "viewrules_metrics.csv")
    fv = pd.read_csv(out / "followup_viewrules_metrics.csv")
    rc = pd.read_csv(out / "repro_published_check.csv", float_precision="round_trip")
    cells = pd.read_csv(out / "repro_e1b_resimulated.csv", float_precision="round_trip")
    with open(out / "followup_matched_values.json") as fh:
        fm = json.load(fh)
    chk = followup_checks(out, v0, fv, fm)
    chk.to_csv(out / "followup_checks.csv", index=False, float_format=ff)
    sdt = followup_sd_table(out)
    sdt.to_csv(out / "followup_sd_sensitivity.csv", index=False, float_format=ff)
    followup_dispersion(fv, fm).to_csv(out / "followup_dispersion.csv", index=False, float_format=ff)
    ranges = followup_ranges(v0, fv, rc, cells)
    ranges.to_csv(out / "followup_viewrules_ranges.csv", index=False, float_format=ff)
    ap = fv[(fv.axis == "AP") & fv.metric.isin(["bias_mm", "rmse_mm", "mae_mm"])]
    idx = ["replicate", "scenario_index", "scenario", "variant", "shape", "u_anchor", "long_AP", "mean_u_long", "sd_u_long",
           "corr_u_long_views", "n"]
    wide = ap.pivot_table(index=idx, columns=["rule", "metric"], values=["value", "mcse", "diff_vs_base", "diff_vs_base_mcse"])
    wide.columns = [f"{r}_{m}_{w}" if w != "value" else f"{r}_{m}" for w, r, m in wide.columns]
    wide = wide.reset_index().sort_values(["replicate", "scenario_index"])
    wide[idx + sorted(c for c in wide.columns if c not in idx)].to_csv(out / "followup_viewrules_table_ap.csv", index=False, float_format=ff)
    table_s25(out, ranges, sdt).to_csv(out / "calibration_table_s25.csv", index=False, float_format="%.6g")
    print(chk[["check", "n_compared", "n_identical", "max_abs_diff", "passed"]].to_string())
    if not chk.passed.all():
        print("FOLLOW-UP CHECK FAILED")
        return 1
    print("wrote followup_checks.csv followup_sd_sensitivity.csv followup_dispersion.csv followup_viewrules_ranges.csv "
          "followup_viewrules_table_ap.csv calibration_table_s25.csv")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results" / "2026-10-05_osrev" / "calibration"))
    ap.add_argument("--followup", action="store_true", help="write only the follow-up tables of 2026-10-06")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.followup:
        return followup(out)
    with open(ROOT / "code" / "configs" / "osrev_calibration.yaml") as fh:
        oc = yaml.safe_load(fh)
    ff = "%.10g"
    singh_reproduction().to_csv(out / "singh_reproduction.csv", index=False, float_format=ff)
    rc = repro_check(out, oc)
    rc.to_csv(out / "repro_published_check.csv", index=False, float_format="%.17g")
    bs = boot_summary(out)
    bs.to_csv(out / "geom_bootstrap_summary.csv", index=False, float_format=ff)
    cs = calibration_summary(out, bs)
    cs.to_csv(out / "calibration_summary.csv", index=False, float_format=ff)
    variant_range(cs).to_csv(out / "calibration_variant_range.csv", index=False, float_format=ff)
    identifiability_table().to_csv(out / "identifiability_ellipse.csv", index=False, float_format=ff)
    ranges, wide = viewrules_tables(out)
    ranges.to_csv(out / "viewrules_ranges.csv", index=False, float_format=ff)
    wide.to_csv(out / "viewrules_table_ap.csv", index=False, float_format=ff)
    tot = int(rc.n_cells.sum())
    print(f"reproduction of stored E1b cells: {int(rc.n_bit_identical.sum())} of {tot} rule x axis x metric x cell values bit-identical")
    print("wrote singh_reproduction.csv repro_published_check.csv geom_bootstrap_summary.csv calibration_summary.csv "
          "viewrules_ranges.csv viewrules_table_ap.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
