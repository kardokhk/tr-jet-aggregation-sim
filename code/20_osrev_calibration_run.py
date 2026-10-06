#!/usr/bin/env python
"""Amendment 3, package A3-6 (calibration against a matched estimand): simulation runner.

  python code/20_osrev_calibration_run.py --list A            # print the stage A task ids
  python code/20_osrev_calibration_run.py --task geom:mid15 --out DIR
  python code/20_osrev_calibration_run.py --merge A --out DIR # csv tables + matched_values.json
  python code/20_osrev_calibration_run.py --list B            # stage B needs DIR/matched_values.json
  python code/20_osrev_calibration_run.py --task rules:0:8 --out DIR
  python code/20_osrev_calibration_run.py --merge B --out DIR

Stage A: geometric calibration (geom:<variant>), bootstrap over the sampling error of the
published means (boot:<variant>:<first>:<last>), library-model matched estimand (lib).
Stage B: view rules under matched view accuracy (rules:<first>:<last>) and exact reproduction of
the 12 stored E1b cells with their original seeds (repro:<first>:<last>).
Config: code/configs/osrev_calibration.yaml. Library: code/lib/duomaxsim/osrev_calibration.py.
Seeds: SeedSequence(entropy 20261005, spawn_key (76, family, index)); families 0 fitting sample,
1 evaluation sample, 2 view rules (one seed for all scenarios), 3 bootstrap target draws,
4 library-model match.

Follow-up of 2026-10-06 (audit items 5.2 to 5.4), config section `followup`:
Stage C: recalibration under each reading of the misprinted confidence interval of the 3D maximal
diameter (sdsens:<variant>:<side>); needs nothing from stages A and B except matched_values.json.
Stage D: view rules with the patient-level geometric draw (frules:<replicate>:<first>:<last>);
needs DIR/followup_matched_values.json from --merge C.
  python code/20_osrev_calibration_run.py --list C ; --task sdsens:ind15:both ; --merge C
  python code/20_osrev_calibration_run.py --list D ; --task frules:0:0:2 ; --merge D
Seeds of stage D: replicate 0 uses the stage B stream (76, 2, 0) for everything the library draws
(common random numbers with the 56 stored scenarios) and (76, 5, 0) for the additional geometric
normals; replicate 1 is an independent repetition on (76, 5, 1) and (76, 5, 2).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from duomaxsim import config as C  # noqa: E402
from duomaxsim import osrev_calibration as K  # noqa: E402

CFG_PATH = ROOT / "code" / "configs" / "osrev_calibration.yaml"
BOOT_ORDER = [("d3_max", None), ("d3_min", None)] + [(v, pl) for v in K.VIEWS for pl in ("primary", "orthogonal")]


def load():
    with open(CFG_PATH) as fh:
        oc = yaml.safe_load(fh)
    return oc, C.load_config(ROOT / oc["meta"]["parameters_from"])


def seed(oc, *key):
    return np.random.SeedSequence(entropy=int(oc["meta"]["master_seed"]),
                                  spawn_key=(int(oc["meta"]["spawn_prefix"]),) + tuple(int(k) for k in key))


def normals(oc, family, n):
    """Five standard-normal vectors (orifice 1, orifice 2, plane angle, offset 1, offset 2)."""
    return np.random.Generator(np.random.PCG64(seed(oc, family, 0))).standard_normal((5, n))


def variant_kw(oc, name):
    v = oc["geometry"]["variants"][name]
    return dict(mechanism=v["mechanism"], offset_rule=v["offset_rule"], orientation=v["orientation"],
                sigma_theta=float(np.radians(v["sigma_theta_deg"])),
                delta_max=float(v.get("delta_max", oc["geometry"]["delta_max"])))


def view_targets(s=K.SINGH):
    return {v: (s["views"][v]["primary"][0], s["views"][v]["orthogonal"][0]) for v in K.VIEWS}


def fit_all(z, kw, otg, vtg):
    par, mo, res = K.fit_orifice(otg)
    dmax, dmin = K.draw_orifice(par, z[0], z[1])
    fits = {v: K.fit_view(dmax, dmin, z[2], z[3], z[4], vtg[v][0], vtg[v][1], **kw) for v in K.VIEWS}
    return par, mo, res, fits


def jack(fn, arrays, G=50):
    """Value and delete-a-group jackknife SE of fn(*arrays) (consecutive blocks of patients)."""
    n = arrays[0].size
    full = fn(*arrays)
    edges = np.linspace(0, n, G + 1).astype(np.int64)
    reps = np.array([fn(*[np.concatenate([a[:edges[g]], a[edges[g + 1]:]]) for a in arrays]) for g in range(G)])
    return float(full), float(np.sqrt((G - 1) / G * ((reps - reps.mean()) ** 2).sum()))


# ------------------------------------------------------------------ stage A tasks

def task_geom(oc, name, sd_side="lower"):
    """sd_side: which half of the printed 95% CI of the 3D maximal diameter gives its SD
    ('lower', the stage A choice; 'both' or 'upper' only in the follow-up stage C)."""
    g = oc["geometry"]
    kw = variant_kw(oc, name)
    otg, vtg = K.orifice_targets(), view_targets()
    if sd_side != "lower":
        otg["sd_max"] = K.sd_from_ci(*K.SINGH["d3_max"], K.SINGH["n"], side=sd_side)
    zf = normals(oc, 0, int(g["n_fit"]))
    par, mo, res, fits = fit_all(zf, kw, otg, vtg)
    ze = normals(oc, 1, int(g["n_eval"]))
    dmax, dmin = K.draw_orifice(par, ze[0], ze[1])
    n = dmax.size
    S = K.SINGH
    N = S["n"]
    orif = [dict(variant=name, quantity=k, target=otg.get(k, np.nan), fitted_quadrature=mo[k]) for k in mo]
    for k, x in (("mean_max", dmax), ("mean_min", dmin), ("mean_avg", 0.5 * (dmax + dmin)),
                 ("mean_sphericity", dmax / dmin)):
        m, se = K._mean_se(x)
        next(r for r in orif if r["quantity"] == k).update(eval_sample=m, eval_mcse=se, n=n)
    for nm, val in zip(("par_m", "par_s", "par_mq", "par_sq", "par_rho"), par):
        orif.append(dict(variant=name, quantity=nm, fitted_quadrature=float(val)))
    orif.append(dict(variant=name, quantity="max_abs_residual", fitted_quadrature=res))
    fit_rows, check_rows, under_rows = [], [], []
    for v in K.VIEWS:
        f = fits[v]
        sim = K.simulate_view(dmax, dmin, ze[2], ze[3], ze[4], 0.0 if np.isnan(f["theta0"]) else f["theta0"],
                              f["par"], **kw)
        row = dict(variant=name, view=v, **{k: kw[k] for k in ("mechanism", "offset_rule", "orientation")},
                   sigma_theta_deg=float(np.degrees(kw["sigma_theta"])),
                   theta0_deg=float(np.degrees(f["theta0"])),
                   parameter="scale_factor" if kw["mechanism"] == "scale" else "offset_scale", value=f["par"],
                   fit_resid_primary_mm=f["resid_primary"], fit_resid_orthogonal_mm=f["resid_orthogonal"],
                   converged=f["converged"], n_fit=int(g["n_fit"]), n_eval=n)
        for pl in ("primary", "orthogonal", "biplane"):
            m, se = K._mean_se(sim[pl])
            row.update({f"{pl}_model_mm": m, f"{pl}_mcse": se, f"{pl}_published_mm": S["views"][v][pl][0]})
        row["count_offset_at_cap"] = int((sim["delta1"] >= kw["delta_max"]).sum())
        fit_rows.append(row)
        # checks against published quantities that were not fitted
        d3avg = 0.5 * (dmax + dmin)
        sdf = lambda x: x.std(ddof=1)  # noqa: E731
        corr = lambda a, b: np.corrcoef(a, b)[0, 1]  # noqa: E731
        ba = lambda t: (t[2] - t[1]) / 3.92  # noqa: E731
        checks = [
            ("sd_primary_mm", sdf, (sim["primary"],), K.sd_from_ci(*S["views"][v]["primary"], N)),
            ("sd_orthogonal_mm", sdf, (sim["orthogonal"],), K.sd_from_ci(*S["views"][v]["orthogonal"], N)),
            ("sd_biplane_mm", sdf, (sim["biplane"],), K.sd_from_ci(*S["views"][v]["biplane"], N)),
            ("mean_sphericity_2d", np.mean, (np.maximum(sim["primary"], sim["orthogonal"])
                                             / np.minimum(sim["primary"], sim["orthogonal"]),),
             S["views"][v]["sphericity"][0]),
            ("sd_diff_3davg_minus_biplane_mm", sdf, (d3avg - sim["biplane"],), ba(S["ba_biplane_vs_avg"][v])),
            ("sd_diff_3dmax_minus_primary_mm", sdf, (dmax - sim["primary"],), ba(S["ba_single_vs_max"][v])),
            ("sd_diff_3dmin_minus_primary_mm", sdf, (dmin - sim["primary"],), ba(S["ba_single_vs_min"][v])),
            ("pearson_r_biplane_3davg", corr, (sim["biplane"], d3avg), S["r_biplane_vs_avg"][v]),
            ("mean_diff_3davg_minus_biplane_mm", np.mean, (d3avg - sim["biplane"],), S["ba_biplane_vs_avg"][v][0]),
            ("mean_diff_3dmax_minus_primary_mm", np.mean, (dmax - sim["primary"],), S["ba_single_vs_max"][v][0]),
            ("mean_diff_3dmin_minus_primary_mm", np.mean, (dmin - sim["primary"],), S["ba_single_vs_min"][v][0]),
        ]
        for nm, fn, arrs, pub in checks:
            val, se = jack(fn, arrs)
            check_rows.append(dict(variant=name, view=v, quantity=nm, model=val, mcse=se, published=float(pub),
                                   fitted=False, n=n))
        for r in K.underestimation_table(dmax, dmin, sim):
            under_rows.append(dict(variant=name, view=v, **r))
    return dict(orifice=orif, geom_fit=fit_rows, geom_checks=check_rows, geom_under=under_rows)


def task_boot(oc, name, first, last):
    g = oc["geometry"]
    kw = variant_kw(oc, name)
    nb = int(g["bootstrap"]["n_fit_boot"])
    z = normals(oc, 0, int(g["n_fit"]))[:, :nb]
    S, N = K.SINGH, K.SINGH["n"]
    base_o, rows = K.orifice_targets(), []
    for rep in range(first, last):
        zz = np.random.Generator(np.random.PCG64(seed(oc, 3, rep))).standard_normal(len(BOOT_ORDER))
        otg, vtg = dict(base_o), {v: list(t) for v, t in view_targets().items()}
        for (a, pl), zi in zip(BOOT_ORDER, zz):
            if pl is None:
                se = K.se_from_ci(*S[a], N, side="lower" if a == "d3_max" else "both")
                otg["mean_max" if a == "d3_max" else "mean_min"] = S[a][0] + se * zi
            else:
                vtg[a][0 if pl == "primary" else 1] = S["views"][a][pl][0] + K.se_from_ci(*S["views"][a][pl], N) * zi
        try:
            par, mo, res, fits = fit_all(z, kw, otg, vtg)
        except Exception as exc:  # report, never drop silently
            rows.append(dict(variant=name, replicate=rep, view="", quantity="fit_error", value=np.nan, note=repr(exc)))
            continue
        dmax, dmin = K.draw_orifice(par, z[0], z[1])
        for v in K.VIEWS:
            f = fits[v]
            sim = K.simulate_view(dmax, dmin, z[2], z[3], z[4], f["theta0"], f["par"], **kw)
            q = {
                "target_mean_max": otg["mean_max"], "target_mean_min": otg["mean_min"],
                "target_primary": vtg[v][0], "target_orthogonal": vtg[v][1],
                "orifice_max_abs_residual": res, "theta0_deg": np.degrees(f["theta0"]), "par": f["par"],
                "converged": float(f["converged"]),
                "resid_abs_max_mm": max(abs(f["resid_primary"]), abs(f["resid_orthogonal"])),
                "primary_vs_max_total": np.mean(1 - sim["primary"] / dmax),
                "primary_vs_max_rotation": np.mean(1 - sim["c0_primary"] / dmax),
                "primary_off_centre": np.mean(1 - sim["primary"] / sim["c0_primary"]),
                "primary_vs_min_total": np.mean(1 - sim["primary"] / dmin),
                "orthogonal_vs_max_total": np.mean(1 - sim["orthogonal"] / dmax),
                "orthogonal_off_centre": np.mean(1 - sim["orthogonal"] / sim["c0_orthogonal"]),
                "biplane_vs_3davg_ratio_of_means": 1 - sim["biplane"].mean() / (0.5 * (dmax + dmin)).mean(),
            }
            rows += [dict(variant=name, replicate=rep, view=v, quantity=k, value=float(val), note="")
                     for k, val in q.items()]
    return dict(geom_bootstrap=rows)


def rules_base(oc, cfg):
    p = C.base_params(cfg)
    p.update(oc["view_rules"]["fixed"])
    return p


def task_lib(oc, cfg):
    lm = oc["library_match"]
    rows = []
    for vo in lm["view_over"]:
        p = dict(rules_base(oc, cfg), view_over=bool(vo))
        ss = seed(oc, 4, 0)
        for L in lm["grid_L"]:
            rows.append(dict(kind="grid", target_label="", view_over=bool(vo),
                             **K.library_biplane_shortfall(p, float(L), int(lm["n"]), ss)))
        for lab, tg in lm["targets"].items():
            rows.append(dict(kind="solved", target_label=lab, view_over=bool(vo),
                             **K.solve_library_scale(p, float(tg), int(lm["n"]), ss)))
    return dict(libmatch=rows)


# ------------------------------------------------------------------ stage B

def u_scale(a, L):
    return [a, L, L, round(1.2 * L, 10)]


def geom_defs(a, g):
    """The four stage B scenario definitions of one geometry variant at anchor scale a:
    half-normal with the mean of reading 1 or 2, constant rotation plus chord-distributed offset
    (reading 1), chord-distributed offset (reading 2). Returns (name, overrides, AP label, SL label)."""
    hm = K.HALF_NORMAL_MEAN
    chord_link = ["halfnormal", "chord", "chord", "chord"]
    dm = float(g["delta_max"])
    Lap1, Lsl1 = g["ap_total"] / hm, max(g["sl_total"], 0.0) / hm
    Lap2, Lsl2 = g["ap_off_centre"] / hm, g["sl_off_centre"] / hm
    s_ax = [[a] + [g["ap_s"]] * 3, [a] + [g["sl_s"]] * 3]
    return [
        ("r1_halfnormal", {"u_scale_axis": [u_scale(a, Lap1), u_scale(a, Lsl1)]},
         f"half-normal scale {Lap1:.4f}", f"half-normal scale {Lsl1:.4f}"),
        ("r1_chord", {"u_scale_axis": s_ax, "u_link": chord_link, "u_delta_max": dm,
                      "u_shift_axis": [[0.0] + [g["ap_rotation"]] * 3, [0.0] + [g["sl_rotation"]] * 3]},
         f"chord link, shift {g['ap_rotation']:.4f}, offset scale {g['ap_s']:.4f}",
         f"chord link, shift {g['sl_rotation']:.4f}, offset scale {g['sl_s']:.4f}"),
        ("r2_halfnormal", {"u_scale_axis": [u_scale(a, Lap2), u_scale(a, Lsl2)]},
         f"half-normal scale {Lap2:.4f}", f"half-normal scale {Lsl2:.4f}"),
        ("r2_chord", {"u_scale_axis": s_ax, "u_link": chord_link, "u_delta_max": dm},
         f"chord link, offset scale {g['ap_s']:.4f}", f"chord link, offset scale {g['sl_s']:.4f}"),
    ]


def patient_defs(a, g):
    """Follow-up scenario definitions with the patient-level geometric draw (key u_geom)."""
    def ug(reading, **kw):
        return {"u_scale": u_scale(a, 0.0), "u_geom": dict(
            reading=reading, offset_scale=[g["ap_s"], g["sl_s"]], delta_max=float(g["delta_max"]),
            theta0_deg=[g["ap_theta0_deg"], g["sl_theta0_deg"]], sigma_theta_deg=float(g["sigma_theta_deg"]),
            orifice=[g["par_mq"], g["par_sq"], g["par_rho"]], **kw)}

    lab = (f"patient-level geometry: plane angle {g['ap_theta0_deg']:.1f} deg (SD {g['sigma_theta_deg']:g}), "
           f"offset scale {g['ap_s']:.4f}")
    lsl = (f"patient-level geometry: plane angle {g['sl_theta0_deg']:.1f} deg (SD {g['sigma_theta_deg']:g}), "
           f"offset scale {g['sl_s']:.4f}")
    r2a, r2s = f"patient-level offset only, offset scale {g['ap_s']:.4f}", f"patient-level offset only, offset scale {g['sl_s']:.4f}"
    return [
        ("r1_patient", ug("rotation_and_offset"), lab, lsl),
        ("r1_patient_indoff", ug("rotation_and_offset", offset_source="independent"), lab + ", independent offsets", lsl + ", independent offsets"),
        ("r1_patient_spanq", ug("rotation_and_offset", ellipticity="span_linked"), lab + ", span-linked ellipticity", lsl + ", span-linked ellipticity"),
        ("r2_patient", ug("offset_only"), r2a, r2s),
        ("r2_patient_indoff", ug("offset_only", offset_source="independent"), r2a + ", independent offsets", r2s + ", independent offsets"),
    ]


def build_followup_scenarios(oc, fmatched):
    sc = []
    for a in oc["view_rules"]["u_anchor"]:
        for var in oc["followup"]["scenario_variants"]:
            g = fmatched["geom"][var]
            for nm, over, la, ls in geom_defs(a, g) + patient_defs(a, g):
                sc.append(dict(scenario=f"geom_{var}_{nm}", family="geometric_match", variant=var, shape=nm,
                               u_anchor=a, over=over, long_AP=la, long_SL=ls))
    return sc


def build_scenarios(oc, matched):
    """Published 12 scenarios (plain u_scale, no extension keys) and the matched scenarios."""
    vr = oc["view_rules"]
    sc = []
    for a in vr["u_anchor"]:
        for L in vr["u_long_published"]:
            sc.append(dict(scenario=f"published_L{L:g}", family="published", u_anchor=a, over={"u_scale": u_scale(a, L)},
                           long_AP=f"half-normal scale {L:g}", long_SL=f"half-normal scale {L:g}"))
        for lab in ("inflow", "mean_three_views", "mBC"):
            L = matched["lib_L"][lab]
            sc.append(dict(scenario=f"lib_biplane_{lab}", family="library_match", u_anchor=a,
                           over={"u_scale": u_scale(a, L)},
                           long_AP=f"half-normal scale {L:.4f}", long_SL=f"half-normal scale {L:.4f}"))
        for var in oc["geometry"]["scenario_variants"]:
            for nm, over, la, ls in geom_defs(a, matched["geom"][var]):
                sc.append(dict(scenario=f"geom_{var}_{nm}", family="geometric_match", u_anchor=a, over=over,
                               long_AP=la, long_SL=ls))
    return sc


def task_rules(oc, cfg, matched, first, last):
    vr = oc["view_rules"]
    n, chunk = int(oc["meta"]["n_patients"]), int(oc["meta"]["chunk_size"])
    base = rules_base(oc, cfg)
    ss = seed(oc, 2, 0)
    b = vr["base_scenario"]
    ref = K.simulate_rules(dict(base, u_scale=u_scale(b["u_anchor"], b["u_long"])), ss, n, chunk)
    rows = []
    for i, s in enumerate(build_scenarios(oc, matched)[first:last], start=first):
        sim = K.simulate_rules(dict(base, **s["over"]), ss, n, chunk)
        meta = dict(scenario_index=i, scenario=s["scenario"], family=s["family"], u_anchor=s["u_anchor"],
                    long_AP=s["long_AP"], long_SL=s["long_SL"], seed_entropy=str(ss.entropy),
                    seed_spawn_key=",".join(map(str, ss.spawn_key)), n=n)
        for d, ax in enumerate(K.AXES):
            mu = dict(mean_u_anchor=float(sim["mean_u"][d, 0]), mean_u_long=float(sim["mean_u"][d, 1]))
            for r in K.RULES:
                e = sim["est"][r][:, d] - sim["S"][:, d]
                e0 = ref["est"][r][:, d] - ref["S"][:, d]
                pc = K.paired_contrast(e, e0)
                for m, (val, se) in K.error_metrics(e).items():
                    dv, dse = pc.get(m, (np.nan, np.nan))
                    rows.append(dict(**meta, axis=ax, rule=r, metric=m, value=val, mcse=se,
                                     diff_vs_base=dv, diff_vs_base_mcse=dse, **mu))
    return dict(viewrules=rows)


def task_sdsens(oc, name, side):
    """Stage C: the stage A geometric calibration of one variant with the SD of the 3D maximal
    diameter taken from the lower half-width (stage A choice, repeated as a check), half the full
    width, or the upper half-width of its printed confidence interval. Same fitting and
    evaluation samples as stage A."""
    res = task_geom(oc, name, sd_side=side)
    sd = K.sd_from_ci(*K.SINGH["d3_max"], K.SINGH["n"], side=side)
    return {"f_" + k: [dict(sd_max_side=side, sd_max_mm=sd, **r) for r in rows] for k, rows in res.items()}


def task_frules(oc, cfg, fmatched, rep_, first, last):
    """Stage D: view rules for follow-up scenarios [first, last) of replicate rep_."""
    vr, fu = oc["view_rules"], oc["followup"]
    n, chunk = int(oc["meta"]["n_patients"]), int(oc["meta"]["chunk_size"])
    base = rules_base(oc, cfg)
    ss, ssg = seed(oc, *fu["seeds"][rep_]["main"]), seed(oc, *fu["seeds"][rep_]["geometry"])
    b = vr["base_scenario"]
    ref = K.simulate_rules(dict(base, u_scale=u_scale(b["u_anchor"], b["u_long"])), ss, n, chunk)
    rows = []
    for i, s in enumerate(build_followup_scenarios(oc, fmatched)[first:last], start=first):
        sim = K.simulate_rules(dict(base, **s["over"]), ss, n, chunk, ssg)
        meta = dict(replicate=rep_, scenario_index=i, scenario=s["scenario"], family=s["family"], variant=s["variant"],
                    shape=s["shape"], u_anchor=s["u_anchor"], long_AP=s["long_AP"], long_SL=s["long_SL"],
                    seed_entropy=str(ss.entropy), seed_spawn_key=",".join(map(str, ss.spawn_key)),
                    seed_geometry_spawn_key=",".join(map(str, ssg.spawn_key)) if "u_geom" in s["over"] else "", n=n)
        for d, ax in enumerate(K.AXES):
            mu = dict(mean_u_anchor=float(sim["mean_u"][d, 0]), mean_u_long=float(sim["mean_u"][d, 1]),
                      sd_u_long=float(sim["sd_u"][d, 1]), corr_u_long_views=float(sim["corr_u_long"][d]))
            for r in K.RULES:
                e = sim["est"][r][:, d] - sim["S"][:, d]
                e0 = ref["est"][r][:, d] - ref["S"][:, d]
                pc = K.paired_contrast(e, e0)
                for m, (val, se) in K.error_metrics(e).items():
                    dv, dse = pc.get(m, (np.nan, np.nan))
                    rows.append(dict(**meta, axis=ax, rule=r, metric=m, value=val, mcse=se,
                                     diff_vs_base=dv, diff_vs_base_mcse=dse, **mu))
    return dict(f_viewrules=rows)


def followup_matched_values(out: Path, oc):
    """Values passed from stage C to stage D, read from the merged csv tables. For the stage A
    reading of the confidence interval (lower half-width) the scenario values of the variants
    already used in stage B are taken from matched_values.json, so that the re-simulated stage B
    scenarios are bit-identical; the stage C values must equal them."""
    fu = oc["followup"]
    u = pd.read_csv(out / "followup_sd_underestimation.csv")
    f = pd.read_csv(out / "followup_sd_fit.csv")
    o = pd.read_csv(out / "followup_sd_orifice.csv")
    with open(out / "matched_values.json") as fh:
        old = json.load(fh)["geom"]
    cc = "central chord in plane direction"
    geom = {}
    for key, spec in fu["scenario_variants"].items():
        pv, side = spec["variant"], spec["sd_side"]
        uu, ff, oo = (t[(t.variant == pv) & (t.sd_max_side == side)] for t in (u, f, o))

        def get(view, ref, comp, col="mean"):
            r = uu[(uu["view"] == view) & (uu.plane == "primary") & (uu.reference == ref) & (uu.component == comp)]
            assert len(r) == 1, (key, view, ref, comp)
            return float(r[col].iloc[0])

        fv = lambda v, col: float(ff[ff["view"] == v][col].iloc[0])  # noqa: E731
        ov = lambda q: float(oo[oo.quantity == q]["fitted_quadrature"].iloc[0])  # noqa: E731
        g = dict(delta_max=variant_kw(oc, pv)["delta_max"],
                 ap_view="inflow", ap_total=get("inflow", "3D maximal diameter", "total"),
                 ap_rotation=get("inflow", "3D maximal diameter", "rotation"),
                 ap_off_centre=get("inflow", cc, "off_centre"), ap_s=fv("inflow", "value"),
                 sl_view="4CH", sl_total=get("4CH", "3D minimal diameter", "total"),
                 sl_rotation=get("4CH", "3D minimal diameter", "rotation"),
                 sl_off_centre=get("4CH", cc, "off_centre"), sl_s=fv("4CH", "value"))
        if side == "lower" and pv in old:
            assert g == old[pv], f"stage C does not reproduce matched_values.json for {pv}"
        g.update(variant=pv, sd_max_side=side, sd_max_mm=float(uu.sd_max_mm.iloc[0]),
                 ap_theta0_deg=fv("inflow", "theta0_deg"), sl_theta0_deg=fv("4CH", "theta0_deg"),
                 sigma_theta_deg=fv("inflow", "sigma_theta_deg"),
                 par_mq=ov("par_mq"), par_sq=ov("par_sq"), par_rho=ov("par_rho"),
                 orifice_max_abs_residual=ov("max_abs_residual"),
                 ap_total_sd=get("inflow", "3D maximal diameter", "total", "sd"),
                 ap_off_centre_sd=get("inflow", cc, "off_centre", "sd"),
                 fit_converged=bool(ff[ff["view"] == "inflow"]["converged"].iloc[0]) and bool(ff[ff["view"] == "4CH"]["converged"].iloc[0]))
        geom[key] = g
    return dict(geom=geom)


def repro_cells(oc):
    vr = oc["view_rules"]
    c0 = C.load_config(ROOT / vr["repro_config"])
    out = []
    for ci in range(len(C.experiment_cells(c0, "E1"))):
        p = C.cell_params(c0, "E1", ci)
        if all((p[k] == v) for k, v in vr["repro_select"].items()):
            out.append((ci, p, C.cell_seed(c0, "E1", ci), int(c0["meta"]["chunk_size"])))
    return out


def task_repro(oc, first, last):
    rows = []
    for ci, p, ss, chunk in repro_cells(oc)[first:last]:
        sim = K.simulate_rules(p, ss, int(oc["meta"]["n_patients"]), chunk)
        for (r, d), acc in sim["acc"].items():
            for m, (val, se) in acc.summary().items():
                rows.append(dict(cell=ci, u_anchor=p["u_anchor"], u_long=p["u_long"], rule=r, axis=K.AXES[d],
                                 metric=m, value=float(val), mcse=float(se), n=acc.n,
                                 seed_entropy=str(ss.entropy), seed_spawn_key=",".join(map(str, ss.spawn_key))))
    return dict(repro=rows)


# ------------------------------------------------------------------ task list, merge, CLI

def list_tasks(oc, stage, out=None):
    g = oc["geometry"]
    if stage == "C":
        fu = oc["followup"]
        return [f"sdsens:{v}:{sd}" for v in fu["sd_variants"] for sd in fu["sd_sides"]]
    if stage == "D":
        fu = oc["followup"]
        n_sc = len(oc["view_rules"]["u_anchor"]) * len(fu["scenario_variants"]) * 9
        return [f"frules:{r}:{i}:{min(i + 2, n_sc)}" for r in range(len(fu["seeds"])) for i in range(0, n_sc, 2)]
    if stage == "A":
        t = [f"geom:{v}" for v in g["variants"]]
        b = g["bootstrap"]
        for v in b["variants"]:
            t += [f"boot:{v}:{i}:{min(i + b['per_task'], b['replicates'])}" for i in range(0, b["replicates"], b["per_task"])]
        return t + ["lib"]
    n_sc = 4 * (3 + 3 + 4 * len(g["scenario_variants"]))
    return [f"rules:{i}:{min(i + 2, n_sc)}" for i in range(0, n_sc, 2)] + [f"repro:{i}:{i + 2}" for i in range(0, 12, 2)]


def matched_values(out: Path, oc):
    """Values passed from stage A to stage B, read from the merged csv tables (primary variant)."""
    u = pd.read_csv(out / "geom_underestimation.csv")
    f = pd.read_csv(out / "geom_fit.csv")
    lm = pd.read_csv(out / "libmatch.csv")
    cc = "central chord in plane direction"
    geom = {}
    for pv in oc["geometry"]["scenario_variants"]:
        def get(view, ref, comp):
            r = u[(u.variant == pv) & (u["view"] == view) & (u.plane == "primary") & (u.reference == ref) & (u.component == comp)]
            assert len(r) == 1, (view, ref, comp)
            return float(r["mean"].iloc[0])

        par = lambda v: float(f[(f.variant == pv) & (f["view"] == v)]["value"].iloc[0])  # noqa: E731
        geom[pv] = dict(delta_max=variant_kw(oc, pv)["delta_max"],
                        ap_view="inflow", ap_total=get("inflow", "3D maximal diameter", "total"),
                        ap_rotation=get("inflow", "3D maximal diameter", "rotation"),
                        ap_off_centre=get("inflow", cc, "off_centre"), ap_s=par("inflow"),
                        sl_view="4CH", sl_total=get("4CH", "3D minimal diameter", "total"),
                        sl_rotation=get("4CH", "3D minimal diameter", "rotation"),
                        sl_off_centre=get("4CH", cc, "off_centre"), sl_s=par("4CH"))
    s = lm[(lm.kind == "solved") & (lm.view_over == True)]  # noqa: E712
    return dict(geom=geom, lib_L={r.target_label: float(r.L) for r in s.itertuples()})


def merge(out: Path, stage: str, oc):
    parts = sorted((out / "parts").glob(f"{stage}_*.pkl"))
    want = list_tasks(oc, stage)
    if len(parts) != len(want):
        raise SystemExit(f"stage {stage}: {len(parts)} part files, {len(want)} tasks expected")
    tabs = {}
    for f in parts:
        with open(f, "rb") as fh:
            for k, rows in pickle.load(fh).items():
                tabs.setdefault(k, []).extend(rows)
    names = {"orifice": "orifice_fit.csv", "geom_fit": "geom_fit.csv", "geom_checks": "geom_checks.csv",
             "geom_under": "geom_underestimation.csv", "geom_bootstrap": "geom_bootstrap.csv",
             "libmatch": "libmatch.csv", "viewrules": "viewrules_metrics.csv", "repro": "repro_e1b_resimulated.csv",
             "f_orifice": "followup_sd_orifice.csv", "f_geom_fit": "followup_sd_fit.csv",
             "f_geom_checks": "followup_sd_checks.csv", "f_geom_under": "followup_sd_underestimation.csv",
             "f_viewrules": "followup_viewrules_metrics.csv"}
    sort = {"geom_bootstrap": ["variant", "replicate", "view"], "viewrules": ["scenario_index", "axis", "rule"],
            "repro": ["cell", "rule", "axis"], "f_viewrules": ["replicate", "scenario_index", "axis", "rule"],
            "f_orifice": ["variant", "sd_max_side"], "f_geom_fit": ["variant", "sd_max_side"],
            "f_geom_checks": ["variant", "sd_max_side"], "f_geom_under": ["variant", "sd_max_side"]}
    for k, rows in tabs.items():
        df = pd.DataFrame(rows)
        if k in sort:
            df = df.sort_values(sort[k], kind="stable")
        # the re-simulated stored cells are compared bit for bit with the stored parquet values
        df.to_csv(out / names[k], index=False, float_format="%.17g" if k == "repro" else "%.10g")
        print(f"wrote {names[k]} rows={len(df)}")
    if stage == "A":
        with open(out / "matched_values.json", "w") as fh:
            json.dump(matched_values(out, oc), fh, indent=1)
        print("wrote matched_values.json")
    if stage == "C":
        with open(out / "followup_matched_values.json", "w") as fh:
            json.dump(followup_matched_values(out, oc), fh, indent=1)
        print("wrote followup_matched_values.json")
    for f in parts:
        f.unlink()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", choices=["A", "B", "C", "D"])
    ap.add_argument("--task")
    ap.add_argument("--merge", choices=["A", "B", "C", "D"])
    ap.add_argument("--out", default=str(ROOT / "results" / "2026-10-05_osrev" / "calibration"))
    a = ap.parse_args(argv)
    oc, cfg = load()
    out = Path(a.out)
    if a.list:
        print("\n".join(list_tasks(oc, a.list)))
        return 0
    if a.merge:
        merge(out, a.merge, oc)
        return 0
    if not a.task:
        ap.error("one of --list, --task, --merge is required")
    t0 = time.process_time()
    f = a.task.split(":")
    if f[0] == "geom":
        res, stage = task_geom(oc, f[1]), "A"
    elif f[0] == "boot":
        res, stage = task_boot(oc, f[1], int(f[2]), int(f[3])), "A"
    elif f[0] == "lib":
        res, stage = task_lib(oc, cfg), "A"
    elif f[0] == "rules":
        with open(out / "matched_values.json") as fh:
            matched = json.load(fh)
        res, stage = task_rules(oc, cfg, matched, int(f[1]), int(f[2])), "B"
    elif f[0] == "repro":
        res, stage = task_repro(oc, int(f[1]), int(f[2])), "B"
    elif f[0] == "sdsens":
        res, stage = task_sdsens(oc, f[1], f[2]), "C"
    elif f[0] == "frules":
        with open(out / "followup_matched_values.json") as fh:
            fmatched = json.load(fh)
        res, stage = task_frules(oc, cfg, fmatched, int(f[1]), int(f[2]), int(f[3])), "D"
    else:
        ap.error(f"unknown task {a.task}")
    (out / "parts").mkdir(parents=True, exist_ok=True)
    with open(out / "parts" / f"{stage}_{a.task.replace(':', '-')}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    print(f"task {a.task} done cpu_s={time.process_time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
