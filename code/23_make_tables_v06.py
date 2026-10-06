"""Build main Table 1 and Supplementary Tables S1 to S6 and S8 to S27 for manuscript v06 as markdown.

Modelled on code/15_make_tables_v04.py, which is imported for its formatting helpers and for the bodies of
Supplementary Tables S3 and S4 (numbers unchanged, wording corrected by asserted replacements) and is not edited.
Supplementary Table S7 (ADEMP reporting checklist) is appended at build time from
submission/source/ademp-checklist.md and is not produced here.

Every number in the output is read from a result file under results/2026-09-18_* or
results/2026-10-05_osrev/* (or from the two configuration files for the parameter table). Text that describes
sources, chronology and verification (Supplementary Tables S1, S5, S6, S8, S9, S26) is encoded here from the notes
named in each function; where such a row quotes a simulation number, the number is read from the csv file, and
where it quotes a published value held in a result file, it is asserted against it. Every value of the map of
numbers (Supplementary Table S27) is read from its result file, and the values quoted in the main text are
asserted there, so that the build stops if a result file, a selection or the main text value changes. Test counts
are read from the job logs and job times from the stored accounting listing (17 jobs).

Output: drafts/tables-2026-10-05-v06.md (generated; do not hand edit) and
        results/2026-10-05_osrev/tables/table_cells_v06.csv (one row per number printed in Table 1 and in the
        new tables that were registered through reg(); used by notes/scratch/2026-10-06-tables-check.py).

Run from the project root (login node; no random numbers; deterministic):
  /project/home/p201509/envs/duomax-sim/bin/python code/23_make_tables_v06.py

95% Monte Carlo interval (MCI) = estimate +/- 1.96 MCSE unless the source csv stores the limits.
Rounding: half away from zero on the shortest decimal representation (same rule as 15_make_tables_v04.py).
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tables_v04", ROOT / "code" / "15_make_tables_v04.py")
v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4)
f, ci, ci_lohi, md_table, sel, one, rel, words = v4.f, v4.ci, v4.ci_lohi, v4.md_table, v4.sel, v4.one, v4.rel, v4.words

FULL, AM2 = v4.FULL, v4.AM2
OS = ROOT / "results" / "2026-10-05_osrev"
OUT = ROOT / "drafts" / "tables-2026-10-05-v06.md"
DERIVED = OS / "tables"
SCRIPT = "code/23_make_tables_v06.py"
Z = 1.96
MINUS = v4.MINUS
RULES = [("A1", "Anchor-view mean"), ("A2", "Mean across views"), ("A3", "Largest view mean"),
         ("A4", "Largest view mean after review")]
RULE = dict(RULES)
ABBR_MC = "MCI, Monte Carlo interval; MCSE, Monte Carlo standard error"

CELLS = []  # (table, label, file, value, decimals, printed)


def reg(table, label, path, value, d, scale=1.0):
    """Format a number and record where it came from. Returns the printed string."""
    s = f(scale * float(value), d)
    CELLS.append((table, label, rel(path) if isinstance(path, Path) else str(path), float(value), d, scale, s))
    return s


def n_(x):
    """Integer with comma thousands."""
    return f"{int(round(float(x))):,}"


def pc(k, n, d=1):
    return f(100.0 * k / n, d) + "%"


def kn(k, n, d=1):
    """count of denominator (percentage)."""
    return f"{n_(k)} of {n_(n)} ({pc(k, n, d)})"


def mci(est, mcse, d):
    return f"{f(est, d)} ({f(est - Z * mcse, d)} to {f(est + Z * mcse, d)})"


def em(est, mcse, d, dm=None):
    """estimate (MCSE)."""
    return f"{f(est, d)} ({f(mcse, dm if dm is not None else d + 1)})"


def rng(lo, hi, d):
    return f"{f(lo, d)} to {f(hi, d)}"


def panel(letter, text):
    return f"**Panel {letter}.** {text}"


def bold_panels(text):
    """v04 wrote panel labels in italics; v06 uses bold caption labels only."""
    return re.sub(r"(?m)^\*(Panel [A-Z])\. (.*)\*$", r"**\1.** \2", text)


def swap(text, pairs):
    """Ordered, asserted replacements: every old string must be present."""
    for old, new in pairs:
        assert old in text, f"replacement source not found: {old!r}"
        text = text.replace(old, new)
    return text


def swap_all(text, pairs):
    """Ordered, asserted replacements of every occurrence: each old string must be present at least once."""
    for old, new in pairs:
        assert old in text, f"replacement source not found: {old!r}"
        text = text.replace(old, new)
    return text


def rd(path, **kw):
    assert path.exists(), path
    return pd.read_csv(path, **kw)


def scen(s):
    """'4.8/20' -> '4.8% / 20%'."""
    a, l = str(s).split("/")
    return f"{a}% / {l}%"


# ====================================================================== main Table 1
E1_BASE = v4.E1_BASE
T1_FOOT_MAX = 75


def table1_main():
    src_h, src_w, src_e3 = FULL / "E1E3_e1_headline.csv", AM2 / "E1bE3b_e1_worstcase.csv", FULL / "E1E3_e3_base.csv"
    src_c = OS / "views" / "views_c_table1_s2.csv"
    src_r = OS / "views" / "views_a_variant_ranges.csv"
    h = sel(rd(src_h), **E1_BASE, u="base", axis="AP", estimand="T1")
    w = sel(rd(src_w), **v4.E1B_WORST)
    e3 = sel(rd(src_e3), **v4.E3_BASE)
    c = sel(rd(src_c), cell=703, axis="AP")
    assert set(c.seed_entropy) == {20260918} and set(c.window_label) == {"+-15%"}
    r = rd(src_r)
    r = r[r.variant.str.startswith("base family") & (r.axis == "AP")]
    rows = []
    for code, name in RULES:
        rh, rw, re3, rc, rr = one(h, estimator=code), one(w, estimator=code), one(e3, estimator=code), \
            one(c, rule=code), one(r, rule=code)
        # the stored-seed reproduction of cell 703 must equal the published row
        assert abs(rc.bias_mm - rh.bias) < 1e-9 and abs(rc.rmse_mm - rh.rmse) < 1e-9, code
        assert abs(rr.rmse_mm_max - rw.rmse_max) < 1e-9 and abs(rr.bias_mm_min - rw.bias_min) < 1e-9, code
        T = "Table 1"
        bias = (f"{reg(T, code + ' bias', src_h, rh.bias, 2)} ({reg(T, code + ' bias lo', src_h, rh.bias_lo95, 2)} to "
                f"{reg(T, code + ' bias hi', src_h, rh.bias_hi95, 2)})")
        rows.append([name, bias, reg(T, code + " rmse", src_h, rh.rmse, 2), reg(T, code + " mae", src_c, rc.mae_mm, 2),
                     reg(T, code + " q95abs", src_c, rc.q95_abs_mm, 2),
                     f"{reg(T, code + ' bias min', src_w, rw.bias_min, 2)} to {reg(T, code + ' bias max', src_w, rw.bias_max, 2)}",
                     f"{reg(T, code + ' rmse min', src_r, rr.rmse_mm_min, 2)} to {reg(T, code + ' rmse max', src_r, rr.rmse_mm_max, 2)}",
                     f"{reg(T, code + ' sens', src_e3, re3.sensitivity, 1, 100)} / {reg(T, code + ' spec', src_e3, re3.specificity, 1, 100)}"])
    assert [x[2] for x in rows] == ["2.04", "2.20", "2.11", "2.01"], [x[2] for x in rows]
    hdr = ["Rule", "Bias, base case (mm, 95% MCI)", "RMSE, base case (mm)", "MAE, base case (mm)",
           "95th percentile of absolute error, base case (mm)", "Bias range, 12 scenarios (mm)",
           "RMSE range, 12 scenarios (mm)", "Sensitivity / specificity at 13 mm (%)"]
    title = "**Table 1. Error of the four rules for combining views.**"
    foot = ("Base case: three views, three beats per view, ±15% window, 15% beat-to-beat variation, anchor "
            "underestimation 4.8%, long-axis underestimation 20%, overestimation present, sinus rhythm, prospective "
            "acquisition. 12 scenarios: no window. Bias: mean of measured minus true span. MCI, Monte Carlo interval; "
            "RMSE, root-mean-square error; MAE, mean absolute error. The 13 mm cut-off is illustrative. 100,000 "
            "simulated patients per condition. MAE, 95th percentile and RMSE range: post hoc summaries of the same "
            "patients.")
    nfoot = words(foot)
    assert nfoot <= T1_FOOT_MAX, f"Table 1 footnote has {nfoot} words (limit {T1_FOOT_MAX})"
    prov = ("<!-- Provenance, Table 1 (not part of the table). Generated by " + SCRIPT + ". Bias (95% MCI, stored "
            f"limits) and RMSE: {rel(src_h)}, rows u = base, view_over True, K 3, beat_cv 0.15, rho 0.3, N_beats 3, "
            "axis AP, estimand T1 (E1 cell 703, seed 20260918 spawn key (1, 703)). MAE and 95th percentile of "
            f"absolute error: {rel(src_c)}, rows cell 703, axis AP, columns mae_mm and q95_abs_mm; this is the "
            "stored-seed reproduction of the same 100,000 patients (bias and RMSE asserted equal to the published "
            f"row to 1e-9). Bias range: {rel(src_w)} (E1b, second amendment), K 3, beat_cv 0.15, view_over True, "
            f"window 0, S_median_mm 10, axis AP, estimand T1, columns bias_min and bias_max. RMSE range: {rel(src_r)}, "
            "variant 'base family', axis AP, columns rmse_mm_min and rmse_mm_max (stored-seed reproduction of the "
            "same 12 E1b cells; maximum asserted equal to rmse_max of the worst-case file). Sensitivity and "
            f"specificity: {rel(src_e3)} (E3), u base, axis AP, cutoff_mm 13. Footnote length {nfoot} words. -->")
    text = "\n\n".join([title, md_table(hdr, rows, ["l"] + ["r"] * 7), foot])
    return text + "\n\n" + prov, text


# ====================================================================== Supplementary Table S1 (parameters)
def tableS1(cfg, cfg2):
    mean_pct = v4.mean_pct
    P = cfg["parameters"]
    v = lambda k: P[k]["value"]
    g = lambda k: P[k].get("grid")
    e1 = cfg["experiments"]["E1"]["vary"][0]["levels"]
    u_base, u_low = e1[0]["u_scale"], e1[1]["u_scale"]
    combos = cfg2["experiments"]["E1"]["vary"][0]["levels"]
    u_anch = sorted({c["u_anchor"] for c in combos})
    u_long = sorted({c["u_long"] for c in combos})
    a2_levels = {d["name"]: d["levels"] for d in cfg2["experiments"]["E1"]["vary"][1:] if isinstance(d, dict)}
    spans = a2_levels["S_median_mm"]
    lam = cfg2["parameters"]["ai_shared_lambda"]["grid"]
    lst = lambda xs, fn: ", ".join(fn(x) for x in xs)
    pcg = lambda x: f"{100 * x:g}%"
    # published values quoted in the source column, asserted against the stored reproduction of the source
    sr = rd(OS / "calibration" / "singh_reproduction.csv")
    ell = one(sr, statement="minor below major diameter (ellipticity alone)")
    ratio_singh = f(1 - ell.percent / 100, 2)
    assert ratio_singh == "0.53"
    IND, ASS, DES = "informed by an indirect source", "assumed", "design factor"
    rows = []
    add = lambda *a: rows.append(list(a))
    NV = "not varied"
    low = "low-underestimation scenario (E1, E3, E4)"
    add("Case mix", "Median true AP span (mm)", f"{v('S_median_mm'):g}", NV, lst(spans, lambda x: f"{x:g}") + " (E1b)",
        "[@pascalxtr2020]: vena contracta width in a treated cohort, median 9.5 mm, view not stated; "
        "[@fcvm2024renal]", IND)
    add("Case mix", "Log-SD of true AP span", f"{v('S_log_sd'):.2f}", NV, NV,
        "Derived from the interquartile range in [@pascalxtr2020]", IND + " (derived)")
    add("Geometry", "Mean SL/AP ratio", f"{v('r_mean'):.2f}", lst(g("r_mean"), lambda x: f"{x:.2f}") + " (E4)", NV,
        f"Matched to the mean AP minus SL difference of [@song2011]; {ratio_singh} derived from mean "
        "three-dimensional diameters in [@singh2026]; 0.8 and 0.9 assumed", IND + " (one moment matched)")
    add("Geometry", "Support of the SL/AP ratio", f"{v('r_lower'):.1f} to {v('r_upper'):.1f}", NV, NV, "none", ASS)
    add("Geometry", "Beta concentration of the SL/AP ratio", f"{v('r_concentration'):g}", NV, NV, "none", ASS)
    add("Views", "Underestimation scale, anchor view (mean)", f"{u_base[0]:g} ({mean_pct(u_base[0])})",
        f"{u_low[0]:g} ({mean_pct(u_low[0])}), {low}",
        lst(u_anch, lambda x: f"{x:g} ({mean_pct(x)})") + " (E1b, E3b)",
        "none; no source measured a transgastric view", ASS)
    add("Views", "Underestimation scale, views 2 and 3 (mean)", f"{u_base[1]:g} ({mean_pct(u_base[1])})",
        f"{u_low[1]:g} ({mean_pct(u_low[1])}), {low}",
        lst(u_long, lambda x: f"{x:g} ({mean_pct(x)})") + " (E1b, E3b)",
        "none for the jet span; [@singh2026] measured vena contracta width against three-dimensional diameters "
        "(Supplementary Table S25)", ASS)
    add("Views", "Underestimation scale, view 4 (mean)", f"{u_base[3]:g} ({mean_pct(u_base[3])})",
        f"{u_low[3]:g} ({mean_pct(u_low[3])}), {low}", "1.2 times the scale of views 2 and 3", "none", ASS)
    add("Views", "Underestimation cap", f"{v('u_max'):.1f}", NV, NV, "none", ASS)
    add("Views", "Probability of overestimation, views 1 to 4", lst(v("p_over"), lambda x: f"{x:.2f}"),
        "on, off (E1, E3, E5)", "on, off (E1b, E5b)", "none; [@mascherbauer2005] describes a different mechanism", ASS)
    add("Views", "Overestimation magnitude, median", pcg(v("o_median")), NV, NV, "none", ASS)
    add("Views", "Overestimation magnitude, log-SD", f"{v('o_log_sd'):.1f}", NV, NV, "none", ASS)
    add("Views", "Correlation of error drivers between views", f"{v('rho'):.1f}",
        lst(g("rho"), lambda x: f"{x:.1f}") + " (E1)", NV, "none", ASS)
    add("Beats", "Beat-to-beat CV of true span", pcg(v("beat_cv")), lst(g("beat_cv"), pcg) + " (all)",
        lst(a2_levels["beat_cv"], pcg) + " (E1b); " + lst([0.05, 0.15, 0.30], pcg) + " (E5b)",
        "[@moraldo2013]: measured CV of mitral PISA distance, 15.5%; [@wong1987]: jet area, about 7% to 11% as a "
        "linear dimension (derived)", ASS + ", upper range of an indirect source")
    add("Beats", "RR CV, sinus rhythm", pcg(v("rr_cv_sinus")), NV, NV, "none", ASS)
    add("Beats", "RR CV, AF", pcg(v("rr_cv_af")), lst(g("rr_cv_af"), pcg) + " (E2)", NV, "none", ASS)
    add("Beats", "Slope of log span on log preceding RR", f"{v('beta_rr'):.1f}", NV, NV,
        "none; direction only from an aortic source [@sumida2003]", ASS)
    add("Beats", "Additional beat-to-beat CV in AF (percentage points)", f"{100 * v('af_extra_cv'):g}", NV, NV, "none", ASS)
    add("Beats", "Stored beats, retrospective (zero-truncated Poisson rate; cap)",
        f"{v('avail_lambda'):g} (mean 4.07); {v('max_beats_retro')}", "prospective, retrospective (E2)", NV, "none", ASS)
    add("Beats", "Beat cap, prospective", f"{v('max_beats_prosp')}", NV, NV, "none", ASS)
    add("Reader", "Caliper SD per beat (mm)", f"{v('sigma_cal_mm'):.1f}", NV, NV,
        "0.69 mm derived from transthoracic inter-observer limits [@hauptmann2026]; about 1.2 mm derived from "
        "intra-observer agreement in 10 patients [@singh2026]", IND + " (derived)")
    add("Reader", "SD of reader offset (mm)", f"{v('sigma_rb_mm'):g}",
        lst(g("sigma_rb_mm"), lambda x: f"{x:g}") + " (E6)", lst(g("sigma_rb_mm"), lambda x: f"{x:g}") + " (E6b)",
        "0.75 mm derived from agreement statistics in [@singh2026] (SD from 30 patients, ICC from 10); 2.0 mm "
        "[@alexander2022]", IND + " (derived)")
    add("Machine settings", "SD of the machine-setting factor (log scale), per examination", f"{v('g_log_sd'):.2f}", NV, NV,
        "[@fan1994]: jet area", IND)
    add("AI model", "SD of AI error (mm)", f"{v('sigma_ai_mm'):g}",
        lst(g("sigma_ai_mm"), lambda x: f"{x:g}") + " (E5)", lst(g("sigma_ai_mm"), lambda x: f"{x:g}") + " (E5b)",
        "none", ASS)
    add("AI model", "Share of image-level error reproduced by the AI model, λ", "0", NV,
        lst(lam, lambda x: f"{x:g}") + " (E5b)", "none", ASS)
    add("AI model", "SD of AI draft error (mm)", f"{v('ai_draft_sigma_mm'):g}",
        lst(g("ai_draft_sigma_mm"), lambda x: f"{x:g}") + " (E6)",
        lst(g("ai_draft_sigma_mm"), lambda x: f"{x:g}") + " (E6b)", "none", ASS)
    add("AI model", "Bias of AI draft (mm)", f"{v('ai_draft_bias_mm'):g}", NV, NV, "none", ASS)
    n_dgm = len(rows)
    n_ass = sum(r[6].startswith(ASS) for r in rows)
    n_ind = sum(r[6].startswith(IND) for r in rows)
    assert n_ass + n_ind == n_dgm
    add("Rules", "Views per axis", f"{v('K')}", "1, 2, 3, 4 (E1, E3); 4 (E4)", "2, 3, 4 (E1b)", "initial protocol", DES)
    add("Rules", "Beats per view", f"{v('N_beats')}", lst(g("N_beats"), str) + " (E1, E2, E4)", "3",
        "initial protocol", DES)
    add("Rules", "Beat-consistency window", "±" + pcg(v("window")),
        "none, " + lst([x for x in g("window") if x], lambda x: "±" + pcg(x)) + " (E2)", "none, ±15% (E1b)",
        "initial protocol", DES)
    add("Rules", "Largest view mean after review: warning and adjudication thresholds (mm)",
        f"{v('t_warn'):g} and {v('t_adj'):g}", "warning 2, 3, 4; adjudication 4, 5, 6 (E4)", NV, "initial protocol", DES)
    add("Rules", "Largest view mean after review: probability of excluding an overestimated view; of excluding a "
        "valid view", f"{v('s_det'):g}; {v('f_rej'):g}", "0, 0.5, 0.8, 1.0; 0, 0.1, 0.3 (E1)", NV,
        "none; reviewer performance has no empirical source", DES + ", values assumed")
    add("Rules", "Adjudication tolerance between two reads (mm)", f"{v('adj_tol_mm'):g}",
        lst(g("adj_tol_mm"), lambda x: f"{x:g}") + " (E5)", "1, 2, 3 (E5b)", "initial protocol", DES)
    title = ("**Supplementary Table S1. Parameters of the data-generating mechanism.** No source measured the "
             f"colour-Doppler jet span, so no parameter was calibrated to it: {n_ind} of {n_dgm} parameters were "
             f"informed by an indirect source and {n_ass} were assumed.")
    body = md_table(["Component", "Parameter", "Base case", "Values varied, planned analyses and first amendment "
                     "(experiment)", "Values varied, second amendment (experiment)", "Source", "Status"], rows,
                    align=["l"] * 7)
    foot = ("Status: informed by an indirect source, value or range taken from a related quantity through a stated "
            "assumption (Supplementary Table S9); assumed, no source constrains the value, or the cited source "
            "supports plausibility only; design factor, a setting of the measurement rule and not of the "
            "data-generating mechanism. Derived, computed by us from published summary statistics. Underestimation "
            "is half-normal, so its mean is 0.798 times the scale. The low-underestimation scenario (anchor 2.4%, "
            "long axis 8%) was added by the first amendment; the second amendment crossed mean anchor "
            "underestimation of 2.4%, 4.8%, 10% and 20% with mean long-axis underestimation of 8%, 20% and 40% to "
            "give the 12 view-accuracy scenarios. Values varied in the third amendment are stated with "
            "Supplementary Tables S10 to S25. Not a Monte Carlo table: no n per condition, seed or MCSE applies. "
            f"Sources: code/configs/base.yaml ({cfg['meta']['config_version']}), "
            f"code/configs/amend2_2026-09-18.yaml ({cfg2['meta']['config_version']}), and for the value "
            f"{ratio_singh} results/2026-10-05_osrev/calibration/singh_reproduction.csv; generated by {SCRIPT}. AF, "
            "atrial fibrillation; AI, artificial intelligence; AP, anteroposterior; CV, coefficient of variation; "
            "ICC, intraclass correlation; PISA, proximal isovelocity surface area; RR, interval between successive "
            "R waves; SL, septolateral.")
    return "\n\n".join([title, body, foot]), (n_ind, n_ass, n_dgm)


# ====================================================================== Supplementary Table S2
EST = {"A1": "Anchor-view mean", "A2": "Mean across views", "A3": "Largest view mean",
       "A4": "Largest view mean after review", "A5": "Median across views", "A6": "Index beat",
       "A7": "Offset-corrected mean, benchmark"}


def tableS2():
    src_a = FULL / "E1E3_e1_headline.csv"
    h = sel(rd(src_a), **E1_BASE)
    h = h[h.estimand == "T1"]
    assert len(h) == 2 * 7 * 2
    hdrA = ["Rule", "AP bias (mm, 95% MCI)", "AP relative bias (%)", "AP RMSE (mm, 95% MCI)",
            "SL bias (mm, 95% MCI)", "SL relative bias (%)", "SL RMSE (mm, 95% MCI)"]
    blocks = []
    for u, lab in [("base", "Base case: anchor underestimation 4.8%, long-axis underestimation 20%"),
                   ("low", "Low-underestimation scenario: anchor 2.4%, long axis 8% (first amendment)")]:
        rows = []
        for e in EST:
            cells = [EST[e]]
            for ax in ["AP", "SL"]:
                r = one(h, u=u, estimator=e, axis=ax)
                cells += [ci_lohi(r.bias, r.bias_lo95, r.bias_hi95, 2), f(r.relbias_pct, 1),
                          ci_lohi(r.rmse, r.rmse_lo95, r.rmse_hi95, 2)]
            rows.append(cells)
        blocks.append((lab, md_table(hdrA, rows)))
    # RMSE ranges of the four view rules, for the caption
    r4 = h[h.estimator.isin(RULE) & (h.axis == "AP")]
    rb, rl = r4[r4.u == "base"].rmse, r4[r4.u == "low"].rmse
    cap_rng = (reg("S2", "rmse min base", src_a, rb.min(), 2), reg("S2", "rmse max base", src_a, rb.max(), 2),
               reg("S2", "rmse min both", src_a, min(rb.min(), rl.min()), 2),
               reg("S2", "rmse max both", src_a, max(rb.max(), rl.max()), 2))
    assert cap_rng == ("2.01", "2.20", "1.70", "2.34"), cap_rng
    # third block: long axis changed alone (second-amendment run, cell 254)
    src_c = OS / "views" / "views_c_table1_s2.csv"
    c = sel(rd(src_c), cell=254)
    assert set(c.anchor_mean_pct) == {4.8} and set(c.long_mean_pct) == {8.0} and set(c.window_label) == {"+-15%"}
    rowsC = []
    for code in RULE:
        cells = [EST[code]]
        for ax in ["AP", "SL"]:
            r = one(c, rule=code, axis=ax)
            cells += [f"{reg('S2', f'254 {code} {ax} bias', src_c, r.bias_mm, 2)} ({f(r.bias_mm_mci_lo, 2)} to {f(r.bias_mm_mci_hi, 2)})",
                      f"{reg('S2', f'254 {code} {ax} rmse', src_c, r.rmse_mm, 2)} ({f(r.rmse_mm_mci_lo, 2)} to {f(r.rmse_mm_mci_hi, 2)})"]
        rowsC.append(cells)
    hdrC = ["Rule", "AP bias (mm, 95% MCI)", "AP RMSE (mm, 95% MCI)", "SL bias (mm, 95% MCI)", "SL RMSE (mm, 95% MCI)"]

    src_b = AM2 / "E1bE3b_e1_worstcase.csv"
    w = sel(rd(src_b), K=3, beat_cv=0.15, view_over=True, window=0.0, S_median_mm=10.0, estimand="T1")
    rowsB, mcB = [], []
    for ax in ["AP", "SL"]:
        for e in ["A1", "A2", "A3", "A4", "A5", "A6"]:
            r = one(w, axis=ax, estimator=e)
            rowsB.append([ax, EST[e], f"{ci(r.bias_min, r.bias_min_mcse, 2)} at {v4.combo_lab(r.bias_min_combo)}",
                          f"{ci(r.bias_max, r.bias_max_mcse, 2)} at {v4.combo_lab(r.bias_max_combo)}",
                          ci(r.maxabs_bias, r.maxabs_bias_mcse, 2),
                          f"{ci(r.rmse_max, r.rmse_max_mcse, 2)} at {v4.combo_lab(r.rmse_max_combo)}"])
            mcB += [r.bias_min_mcse, r.bias_max_mcse, r.maxabs_bias_mcse, r.rmse_max_mcse]
    a7 = one(w, axis="AP", estimator="A7")
    hdrB = ["Axis", "Rule", "Lowest bias (mm, 95% MCI) at anchor / long-axis underestimation",
            "Highest bias (mm, 95% MCI) at anchor / long-axis underestimation", "Largest absolute bias (mm, 95% MCI)",
            "Largest RMSE (mm, 95% MCI) at anchor / long-axis underestimation"]
    mx = lambda col: h[col].max()
    title = ("**Supplementary Table S2. Bias and root-mean-square error of each rule against the true maximal span.** "
             f"The RMSE of the four view rules was {cap_rng[0]} to {cap_rng[1]} mm in the base case and "
             f"{cap_rng[2]} to {cap_rng[3]} mm across the base case and the low-underestimation scenario; across "
             "the 12 view-accuracy scenarios the largest-view rules had the smallest worst-case absolute bias.")
    foot = ("Panels A and B, planned experiment E1 (panel B, first amendment): three views, three beats per view, "
            "±15% beat-consistency window, beat-to-beat variation 15%, overestimation present, correlation between "
            "views 0.3. Panel C: the same condition with only the long-axis underestimation lowered, from the "
            "second-amendment run (independent seed, cell 254). Panel D, second amendment (E1b): the same condition "
            "without a window; lowest and highest bias, largest absolute bias and largest RMSE of each rule over "
            "the 12 scenarios (mean anchor underestimation 2.4%, 4.8%, 10% or 20% by mean long-axis underestimation "
            "8%, 20% or 40%), written anchor / long axis. Bias is the mean of estimate minus true maximal span; "
            "relative bias is the mean of per-patient ratios (estimate / truth) minus 1. The offset-corrected mean "
            "divides each view mean by its true view offset; it cannot be used in practice, is not a lower "
            f"bound on error, and is omitted from panel D (its AP bias ranged from {f(a7.bias_min, 2)} to "
            f"{f(a7.bias_max, 2)} mm). n = 100,000 simulated patients per condition; seeds 20260918 (panels A and "
            f"B) and 20260920 (panels C and D). Largest MCSE: panels A and B, bias {mx('bias_mcse'):.4f} mm, relative "
            f"bias {mx('relbias_pct_mcse'):.3f} percentage points, RMSE {mx('rmse_mcse'):.4f} mm; panel D "
            f"{max(mcB):.4f} mm. 95% MCI = estimate ± 1.96 MCSE. Panel D extremes are bounds over this grid, driven "
            f"by its corners, and not over clinical practice. Sources: {rel(src_a)} (panels A and B), {rel(src_c)} "
            f"(panel C) and {rel(src_b)} (panel D); generated by {SCRIPT}. AP, anteroposterior; {ABBR_MC}; RMSE, "
            "root-mean-square error; SL, septolateral.")
    return "\n\n".join([title, panel("A", blocks[0][0]), blocks[0][1], panel("B", blocks[1][0]), blocks[1][1],
                        panel("C", "Long-axis underestimation lowered alone: anchor 4.8%, long axis 8%"),
                        md_table(hdrC, rowsC),
                        panel("D", "Extremes across the 12 view-accuracy scenarios (second amendment, no window)"),
                        md_table(hdrB, rowsB, ["l", "l", "r", "r", "r", "r"]), foot])


# ====================================================================== Supplementary Tables S3 and S4 (v04 bodies)
def tableS3():
    t = bold_panels(v4.table3())
    t = swap(t, [
        ("Reference quality and apparent AI performance in the base cell.", "Reference quality and apparent AI "
         "performance in the base case."),
        ("an independent AI model appeared", "an AI model with independent error appeared"),
        ("Base cell: beat CV 15%", "Base case: beat-to-beat variation 15%"),
        ("validation studies of 200 patients", "evaluation studies of 200 patients"),
        ("Pre-specified and amendment 2 rows come from independent runs (master seeds 20260918 and 20260920)",
         "Planned and second-amendment rows come from independent runs (seeds 20260918 and 20260920)"),
        ("comparison within each simulated study, amendment 2 run", "comparison within each simulated study, "
         "second-amendment run, in which the two models carried identical residual errors (with independent "
         "residual errors the proportions were lower; Supplementary Table S22)"),
        ("n = 2,000 simulated validation studies per cell", "n = 2,000 simulated evaluation studies per condition"),
        ("Decomposition of reference error (amendment 2)", "Decomposition of reference error (second amendment)"),
        (SCRIPT.replace("23_make_tables_v06", "15_make_tables_v04"), SCRIPT),
    ])
    t = t.replace("| pre-specified |", "| planned |").replace("| amendment 2 |", "| second amendment |")
    t = t.replace("The LoA width MCSE is the conservative bound", "The LoA width MCSE is the upper bound")
    assert "pre-specified" not in t.lower() and "amendment 2" not in t and "validation" not in t, "S3 wording"
    # terms of the main text and the supplement in place of the notation of the earlier tables
    INH = "inheriting the systematic error of a single read"
    t = swap_all(t, [
        ("a model sharing the readers' image-level error appeared more accurate",
         "a model reproducing the image error appeared more accurate"),
        ("| Bias vs T1 (mm, 95% MCI) | MAE vs T1 (mm, 95% MCI) |",
         "| Bias against the true maximal span (mm, 95% MCI) | MAE against the true maximal span (mm, 95% MCI) |"),
        ("| Image composite R_img (no caliper error or reader bias) |",
         "| Error-free read of the same images, R_img (no caliper error or reader offset) |"),
        ("| AI model | σ_AI (mm) |", "| AI model | SD of the model's own error (mm) |"),
        ("| T1 (true) |", "| True maximal span |"),
        ("| Unbiased |", "| Independent error |"),
        ("| Inherited bias |", "| Inheriting the systematic error of a single read |"),
        ("| Shared error, λ = 0 (unbiased) |", "| Reproducing image error, λ = 0 |"),
        ("| Shared error, λ = 0.5 |", "| Reproducing image error, λ = 0.5 |"),
        ("| Shared error, λ = 1 |", "| Reproducing image error, λ = 1 |"),
        ("Paired comparison of the inherited-bias and unbiased AI (σ_AI 2 mm)",
         f"Paired comparison of the model {INH} and the model with independent error (own-error SD 2 mm)"),
        ("| Studies in which the inherited-bias AI had lower MAE (%, 95% MCI) | Mean paired MAE difference, "
         "inherited minus unbiased (mm, 95% MCI) |",
         "| Studies in which the inheriting model had the lower MAE (%, 95% MCI) | Mean paired MAE difference, "
         "inheriting model minus model with independent error (mm, 95% MCI) |"),
        ("| Image-level component shared by all readers, Var(e_img) (mm²) |",
         "| Image error, shared by all readers, Var(e_img) (mm²) |"),
        ("| Of Var(e_rd): reader-bias variance (mm²) |", "| Of Var(e_rd): reader-offset variance (mm²) |"),
        ("AP axis, K = 3, every read by the largest view mean after review (A4) (detection 0.8, false rejection "
         "0.1, thresholds 3 and 5 mm),",
         "AP axis, three views, every read by the largest view mean after review (probability of excluding an "
         "overestimated view 0.8 and a valid view 0.1; warning and adjudication thresholds 3 and 5 mm),"),
        ("Panel A: bias is reference minus T1;", "Panel A: bias is reference minus the true maximal span (T1);"),
        ("Panel B: bias is AI minus comparator; the unbiased AI had normal error with SD σ_AI around T1; the "
         "inherited-bias AI added the systematic component of one A4 read regressed on T1 in 20,000 separate "
         "patients; the shared-error AI was T1 + λ(R_img − T1) + independent error, and at λ = 0 equals the "
         "unbiased AI.",
         "Panel B: bias is AI model minus comparator; the model with independent error had normal error of the "
         f"stated SD around T1; the model {INH} added the systematic component of one read regressed on T1 in "
         "20,000 separate patients; the model reproducing image error was T1 + λ(R_img − T1) + independent error, "
         "and at λ = 0 equals the model with independent error."),
        ("in which the two models carried identical residual errors (with independent residual errors the "
         "proportions were lower; Supplementary Table S22), λ = 0 cell, σ_AI 2 mm.",
         "in which the two models carried identical own errors (with independent own errors the proportions were "
         "lower; Supplementary Table S22), λ = 0, own-error SD 2 mm."),
        ("R_img, largest view mean after review (A4) computed from the images without caliper error or reader "
         "bias.",
         "R_img, largest view mean after review computed from the images without caliper error or reader offset; "
         "T1, true maximal span."),
    ])
    for bad in ["σ_AI", "nbiased", "nherited", "(A4)", "A4 read", "K = 3", "hared error", "shared-error",
                "image-level", "residual error", " vs ", "reader-bias"]:
        assert bad not in t, f"S3 remnant {bad!r}"
    return t


def tableS4():
    s1, s2 = rd(FULL / "E5E6_E6_size_check.csv"), rd(AM2 / "E5bE6b_E6b_size_check.csv")
    fz = s1[s1.test == "f_variance"]
    assert len(fz) == 216
    n_above = int(((fz.value - Z * fz.mcse) > 0.05).sum())
    n_point = int((fz.value > 0.05).sum())
    bf = s2[s2.test == "bf_onesided"]
    n_bf_below = int(((bf.value + Z * bf.mcse) < 0.05).sum())
    n_bf_above = int(((bf.value - Z * bf.mcse) > 0.05).sum())
    s_n = reg("S4", "F above by MCI", FULL / "E5E6_E6_size_check.csv", n_above, 0)
    assert s_n == "176", s_n
    bf_lo, bf_hi = reg("S4", "BF min size", AM2 / "E5bE6b_E6b_size_check.csv", bf.value.min(), 3), \
        reg("S4", "BF max size", AM2 / "E5bE6b_E6b_size_check.csv", bf.value.max(), 3)
    t = bold_panels(v4.table4())
    t = swap(t, [
        ("the pre-specified F test exceeded its nominal size, and the one-sided Brown-Forsythe test held it and "
         "reached power 0.80 with 50 sentinel cases at w = 0.3.",
         f"the 95% MCI of the false-positive rate of the planned F test lay entirely above 0.05 in {s_n} of 216 "
         f"combinations (point estimate above 0.05 in {n_point}), and the one-sided Brown-Forsythe test was "
         f"conservative (false-positive rate {bf_lo} to {bf_hi}; 95% MCI entirely below 0.05 in {n_bf_below} and "
         f"entirely above it in {n_bf_above} of 216 combinations) and reached power 0.80 with 50 sentinel cases "
         "at w = 0.3."),
        ("F test on variance, one-sided (pre-specified test)", "F test on variance, one-sided (planned test)"),
        ("Panel A, pre-specified run (E6): beat CV 15%", "Panel A, planned run (E6): beat-to-beat variation 15%"),
        ("a combination is above or below 0.05 when its 95% MCI excludes 0.05", "a combination is counted as above "
         "or below 0.05 when its 95% MCI lies entirely above or entirely below 0.05"),
        ("Panels C and D, amendment 2 run (E6b), base cell", "Panels C and D, second-amendment run (E6b), base case"),
        ("values in parentheses in panel C are the pre-specified F test", "values in parentheses in panel C are the "
         "planned F test"),
        ("n = 2,000 simulated studies per cell", "n = 2,000 simulated studies per condition; seeds 20260918 "
         "(planned run) and 20260920 (second-amendment run). The sentinel subset and the assisted remainder are "
         "different cases read by the same reader (unpaired design); the endpoint is the variance of read minus "
         "AI draft, the alternative is a smaller variance in assisted reads, and the null is w = 0"),
        (SCRIPT.replace("23_make_tables_v06", "15_make_tables_v04"), SCRIPT),
    ])
    t = t.replace("| pre-specified |", "| planned |").replace("| amendment 2 |", "| second amendment |")
    t = t.replace("base cell", "base case").replace("54 cells", "54 conditions").replace("number of cells", "number of conditions")
    t = t.replace("in that cell", "in that condition").replace("over 54 cells", "over 54 conditions")
    assert "pre-specified" not in t.lower() and "amendment 2" not in t, "S4 wording"
    t = swap_all(t, [
        ("performance of sentinel tests for anchoring.**", "performance of sentinel tests for drift towards the AI "
         "proposal.**"),
        ("the one-sided Brown-Forsythe test was conservative (false-positive rate",
         "the one-sided Brown-Forsythe test was conservative or near nominal (false-positive rate"),
        ("with 50 sentinel cases at w = 0.3.", "with 50 sentinel cases at a drift of 0.3 (w, the fraction of the way "
         "towards the AI proposal)."),
        ("| σ_rb (mm) |", "| SD of reader offset (mm) |"),
        ("Empirical size of sentinel tests at w = 0", "False-positive rate of the sentinel tests without drift (w = 0)"),
        ("| Minimum size | Median size | Maximum size |", "| Lowest | Median | Highest |"),
        ("| Anchoring fraction w |", "| Drift towards the AI proposal, w |"),
        ("| 0.0 (size) |", "| 0 (false-positive rate) |"),
        ("| not size-valid |", "| false-positive rate not controlled |"),
        ("gold sets read by two readers per simulated study", "double-read sets read by two readers per simulated "
         "study"),
        ("(beat CV × σ_rb × AI draft SD)", "(beat-to-beat variation × SD of reader offset × SD of the AI proposal)"),
        ("beat CV 15%, σ_rb 0.75 mm, AI draft bias 1 mm and SD 2 mm", "beat-to-beat variation 15%, SD of reader "
         "offset 0.75 mm, AI proposal bias 1 mm and SD 2 mm"),
        ("of a manual-only sentinel subset of n cases", "of a sentinel subset of n cases read without AI assistance"),
        ("whose size exceeded 0.05, so its power is not size-calibrated", "whose false-positive rate exceeded 0.05, "
         "so its power is not comparable at the nominal level"),
        ("for a test whose size MCI did not lie above 0.05 in that condition", "for a test whose false-positive "
         "rate had a 95% MCI not entirely above 0.05 in that condition"),
        ("panel B size MCSE", "panel B false-positive rate MCSE"),
        ("; σ_rb, reader bias SD.", "."),
    ])
    for bad in ["σ_rb", "nchoring", "gold", "beat CV", "manual-only", "size-", " size ", "(size)"]:
        assert bad not in t, f"S4 remnant {bad!r}"
    return t


# ====================================================================== Supplementary Table S5 (face validity)
def tableS5():
    src = AM2 / "face_table.csv"
    blob = " ".join(rd(src).astype(str).values.ravel())

    def chk(*strings):
        for x in strings:
            assert x in blob, f"S5 value {x!r} not found in {src.name}"

    # numbers are those of v04 Supplementary Table S5, asserted present in face_table.csv
    chk("width 5.50 mm", "4.47 to 6.65", "3.18 mm", "3.88 mm", "width 3.82 mm", "1.000")
    chk("0.937", "0.813 to 0.974", "0.935 (0.894 to 0.961)")
    chk("0.925", "0.626 to 0.989", "0.865 to 0.944", "0.258")
    chk("-1.67 mm", "-15.4%", "-3.65 mm", "-33.7%")
    chk("-1.33 mm", "-14.9%", "-2.95 mm", "-33.2%", "19% to 25%")
    chk("20.8%", "14.9%", "17.5%", "15.3%", "15.5%")
    chk("19.1%", "14% to 22%", "7% to 11%")
    chk("median 8.32", "6.99 to 9.88", "median 9.73", "8.36 to 11.35", "median 10.32", "8.86 to 11.99",
        "median 6.44", "4.91 to 8.01", "9.5 [7.2 to 12.3]")
    # like-axis and cross-axis single-plane comparisons of Singh et al., from the stored reproduction
    src_s = OS / "calibration" / "singh_reproduction.csv"
    sr = rd(src_s)
    sp = sr[sr.statement.str.startswith("single plane below 3D maximum")].set_index("view").percent
    like = reg("S5", "Singh inflow vs 3D max", src_s, sp["inflow"], 1)
    cross = sorted([sp["4CH"], sp["mBC"]])
    cross_s = f"{reg('S5', 'Singh cross lo', src_s, cross[0], 1)}% and {reg('S5', 'Singh cross hi', src_s, cross[1], 1)}%"
    ell = reg("S5", "ellipticity alone", src_s, one(sr, statement="minor below major diameter (ellipticity alone)").percent, 1)
    bp = sr[sr.statement.str.startswith("biplane below 3D average (Table 3")].percent
    bi = f"{reg('S5', 'Singh biplane lo', src_s, bp.min(), 1)}% to {reg('S5', 'Singh biplane hi', src_s, bp.max(), 1)}%"
    m = lambda x: x.replace("-", MINUS)
    SET, NOT = "check against the source of the parameter", "comparison with a source not used for the parameter"
    singh_pub = (f"{like}% below the three-dimensional maximal diameter in the plane on the same axis (inflow; "
                 f"derived from reported means) [@singh2026]; {cross_s} in the other two planes are comparisons "
                 f"across axes (ellipticity alone {ell}%)")
    rows = [
        ["Inter-reader 95% LoA width, single beat, long-axis AP, n = 50 (mm)",
         "5.50 (95% of studies 4.47 to 6.65); three-beat mean 3.18; biplane single beat 3.88",
         "3.82 [@hauptmann2026]", "vena contracta width, transthoracic, core laboratory", SET + " (caliper SD)",
         "Wider than published in 100% of studies, by construction: same-beat reads differ only by caliper error, "
         "so the width equals 2 × 1.96 × √2 × caliper SD and compares 1.0 mm with the derived 0.69 mm"],
        ["Inter-reader ICC(A,1), single beat, n = 50", "0.937 (95% of studies 0.813 to 0.974)",
         "0.935 (0.894 to 0.961) [@hauptmann2026]", "vena contracta width, transthoracic", SET,
         "Similar only because a larger per-read error is offset by a wider simulated case mix; not a sharp check"],
        ["Inter-reader ICC, biplane, n = 10", "0.925 (95% of studies 0.626 to 0.989); 25.8% of studies ≤ 0.865",
         "0.865 to 0.944 [@singh2026]", "vena contracta width, TEE, 10 patients", SET + " (reader offset SD)",
         "Compatible, but n = 10 is compatible with a wide range of error models"],
        ["Single long-axis view three-beat mean minus true maximal span, long-axis underestimation 20% (base case)",
         m("-1.67 mm; -15.4% of mean truth"), singh_pub,
         "vena contracta width against a three-dimensional diameter; anaesthetized surgical patients", NOT,
         "About half the published difference on the same axis; the long-axis value is an assumption"],
        ["Same, long-axis underestimation 40% (second amendment)", m("-3.65 mm; -33.7% of mean truth"),
         f"{like}% on the same axis [@singh2026]", "as above", NOT,
         "Close to the published difference on the same axis; above every calibrated point estimate "
         "(Supplementary Table S25)"],
        ["Biplane average minus mean of true AP and SL spans, long-axis underestimation 20% (base case)",
         m("-1.33 mm; -14.9% of mean truth"),
         f"{bi} below the average of three-dimensional maximal and minimal diameters [@singh2026]",
         "vena contracta width, biplane against three-dimensional average", NOT,
         "Below the published range: overestimation, the lognormal beat mean and the machine-setting factor offset part "
         "of the 20% mean underestimation"],
        ["Same, long-axis underestimation 40% (second amendment)", m("-2.95 mm; -33.2% of mean truth"),
         f"{bi} [@singh2026]", "as above", NOT, "Above the published range"],
        ["Beat-to-beat CV of measured values, 5 beats, long-axis AP, series with mean ≥ 3 mm",
         "20.8% at beat-to-beat variation 15% (true spans 14.9%); 17.5% at 10%; 15.3% at 5%",
         "15.5% [@moraldo2013]", "PISA distance, mitral, transthoracic", SET + " (beat-to-beat variation)",
         "About 5 points above published: the base value equals a measured CV and caliper error is added to it, so "
         "measurement variability is counted twice; depends on the post hoc restriction to series with mean ≥ 3 mm"],
        ["Mean beat-to-beat CV of measured values", "19.1% in the base case; 13.4% at beat-to-beat variation 5%",
         "14% to 22% for jet area [@wong1987]; about 7% to 11% as a linear dimension (derived)",
         "jet area, three valves pooled", SET, "Above the linear equivalent at every value simulated"],
        ["Measured AP span, median of cohorts of 44 (95% range), long-axis three-beat mean, long-axis underestimation "
         "20% (mm)", "8.32 (6.99 to 9.88)", "9.5 (IQR 7.2 to 12.3) [@pascalxtr2020]",
         "vena contracta width, TEE, view not stated", SET + " (case mix)", "Published median inside the simulated range"],
        ["Same, anchor three-beat mean (mm)", "9.73 (8.36 to 11.35)", "9.5 [@pascalxtr2020]", "as above", SET, "Inside"],
        ["Same, largest view mean after review (mm)", "10.32 (8.86 to 11.99)", "9.5 [@pascalxtr2020]", "as above",
         SET, "Inside"],
        ["Same, long-axis three-beat mean, long-axis underestimation 40% (mm)", "6.44 (4.91 to 8.01)",
         "9.5 [@pascalxtr2020]", "as above", SET, "Outside, if the published width came from a single long-axis view"],
    ]
    cs25 = rd(OS / "calibration" / "calibration_table_s25.csv")
    b25s = reg("S5", "base-case biplane shortfall, one beat, no window", OS / "calibration" / "calibration_table_s25.csv",
               one(cs25[cs25.part.str.startswith("B.")], condition="scale 0.25; overestimation on").estimate, 1)
    assert b25s == "16.0", b25s
    hdr = ["Quantity", "Simulated", "Published", "Published quantity", "Role", "Reading"]
    title = ("**Supplementary Table S5. Simulated values in the base case against published measurements.** No "
             "published quantity was the colour-Doppler jet span, most comparisons used the sources that informed "
             "the parameters, and the simulated reader and beat variability exceeded the published values.")
    foot = ("Simulated values from the comparison with published values of the second amendment "
            "(code/12_face_validity.py; seed 20260920; 2,000 simulated reader studies for rows 1 to 3, re-simulated "
            "patients for rows 4 to 13; MCSE of the mean differences in rows 4 to 7 at most 0.013 mm and 0.11 "
            "percentage points; remaining MCSE in the source file). Mean anchor underestimation 4.8% throughout. "
            "Percentages in rows 4 to 7 are ratios of means (mean difference divided by mean truth), not the mean of "
            "per-patient ratios used for relative bias in Supplementary Table S2. Role: agreement with a source "
            "that informed the parameter is expected and is neither a calibration of the jet span nor independent "
            "validation. Published percentages for [@singh2026] were derived from reported means. Rows 4 to 7 use the "
            "three-beat view mean with the ±15% window; computed from one beat without a window in the "
            f"third-amendment calibration, the base-case value of row 6 was {b25s}% (Supplementary Table S25). Rows 1 "
            "and 2 use one reader pair per study with reader offsets drawn per study. "
            f"Sources: {rel(src)} and, for the values of [@singh2026], {rel(src_s)}; generated by {SCRIPT}. AP, "
            "anteroposterior; CV, coefficient of variation; ICC, intraclass correlation; IQR, interquartile range; "
            "LoA, limits of agreement; PISA, proximal isovelocity surface area; SL, septolateral; TEE, "
            "transoesophageal echocardiography.")
    return "\n\n".join([title, md_table(hdr, rows, ["l"] * 6), foot])


# ====================================================================== Supplementary Table S6 (names and codes)
def tableS6():
    rows = [
        ["Rule", "Anchor-view mean", "A1", "Mean of the accepted beats in the anchor view only"],
        ["Rule", "Mean across views", "A2", "Average of the view means of all views"],
        ["Rule", "Largest view mean", "A3", "Highest view mean, without review"],
        ["Rule", "Largest view mean after review", "A4",
         "Highest view mean after any view exceeding the anchor by 3 mm or more is reviewed and, if judged "
         "artefactual, excluded"],
        ["Rule", "Largest-view rules", "A3 and A4", "The two rules above taken together"],
        ["Rule", "Median across views", "A5", "Median of the view means"],
        ["Rule", "Index beat", "A6",
         "Single anchor-view beat whose preceding and pre-preceding RR intervals are most nearly equal"],
        ["Rule", "Offset-corrected mean (benchmark)", "A7",
         "Mean of the view means after dividing each by its true view offset; needs simulated truth and cannot be "
         "used in practice"],
        ["Reference value", "True maximal span", "T1", "True maximal jet span on the axis measured"],
        ["Reference value", "Anchor-view median span", "T2",
         "Median beat span in the anchor view, including that view's under- or overestimation, before the "
         "machine-setting factor and caliper error"],
        ["Scenario", "Base case", "E1 cell 703; u = base",
         "Three views, three beats per view, ±15% beat-consistency window, 15% beat-to-beat variation, anchor "
         "underestimation 4.8%, long-axis underestimation 20%, overestimation present, sinus rhythm, prospective "
         "acquisition"],
        ["Scenario", "Low-underestimation scenario", "E1 cell 1855; u = low",
         "Base case with anchor underestimation 2.4% and long-axis underestimation 8% (first amendment)"],
        ["Scenario", "The 12 view-accuracy scenarios", "E1b; u_combo",
         "Anchor underestimation 2.4%, 4.8%, 10% or 20% by long-axis underestimation 8%, 20% or 40%, no window "
         "(second amendment)"],
        ["Scenario", "Condition; cell", "cell",
         "A condition is one combination of design factors; within the second-amendment grid, 72 conditions each "
         "contain the 12 scenarios, giving 864 cells"],
        ["Analysis", "Beat-window analysis", "E2", "Planned (initial protocol)"],
        ["Analysis", "View-rule analysis", "E1", "Planned; low-underestimation scenario by the first amendment"],
        ["Analysis", "View-accuracy scenarios", "E1b", "Second amendment"],
        ["Analysis", "Cut-off analysis", "E3 and E3b", "E3 planned; E3b second amendment"],
        ["Analysis", "Cross-view trigger analysis", "E4", "Planned"],
        ["Analysis", "AI evaluation analysis", "E5", "Planned"],
        ["Analysis", "AI model reproducing image error", "E5b", "Second amendment"],
        ["Analysis", "Reference-set analysis", "E6 and E6b", "E6 planned; E6b second amendment"],
        ["Analysis", "Sensitivity analyses of the third amendment", "A3-1 to A3-6",
         "Beat rules (A3-1), view rules (A3-2), cut-offs (A3-3), triggers and review (A3-4), AI evaluation and "
         "reference sets (A3-5), calibration (A3-6); post hoc; Supplementary Sections S8.1 to S8.6"],
        ["Term", "Cut-off", "cutoff_mm", "Illustrative classification value of 7, 10 or 13 mm; not a clinical "
         "eligibility criterion"],
        ["Term", "Threshold", "t_warn, t_adj; tau", "Cross-view trigger value (3 mm warning, 5 mm adjudication) or a "
         "published clinical threshold"],
        ["Symbol", "Share of image error reproduced by the AI model", "λ",
         "Fraction of the image-level error that the AI model reproduces"],
        ["Symbol", "Drift towards the AI draft", "w",
         "Fraction of the way an assisted reader moves towards the AI draft"],
    ]
    hdr = ["Type", "Name in the main text and figures", "Code in the protocol, code and result files", "Meaning"]
    title = ("**Supplementary Table S6. Names used in the main text and figures and the corresponding codes in the "
             "protocol, analysis code and result files.**")
    foot = ("AI, artificial intelligence; RR, interval between successive R waves.")
    return "\n\n".join([title, md_table(hdr, rows, ["l"] * 4), foot])


# ====================================================================== S12 to S15 (view rules, third amendment)
VW = OS / "views"
FOOT_V = ("n = 100,000 simulated patients per condition; AP axis; error is measured minus true maximal span. ")


def tableS12():
    s12, sadd, srng = VW / "views_a_12scenarios.csv", VW / "views_a_added_scenarios.csv", VW / "views_a_variant_ranges.csv"
    a = rd(s12)
    a = a[a.axis == "AP"]
    assert len(a) == 48 and a.n.eq(100000).all()
    order = sorted(a.scenario.unique(), key=lambda x: tuple(float(y) for y in x.split("/")))
    rowsA, mc = [], {k: 0.0 for k in ["bias", "rmse", "mae", "q95"]}
    for sc in order:
        for code, name in RULES:
            r = one(a, scenario=sc, rule=code)
            T = "S12"
            rowsA.append([scen(sc), name, reg(T, f"{sc} {code} bias", s12, r.bias_mm, 2),
                          reg(T, f"{sc} {code} rmse", s12, r.rmse_mm, 2), reg(T, f"{sc} {code} mae", s12, r.mae_mm, 2),
                          reg(T, f"{sc} {code} q95abs", s12, r.q95_abs_mm, 2),
                          f"{f(r['q2.5_err_mm'], 2)} to {f(r['q97.5_err_mm'], 2)}", n_(r.count_abs_gt2mm)])
            for k, col in [("bias", "bias_mm_mcse"), ("rmse", "rmse_mm_mcse"), ("mae", "mae_mm_mcse"), ("q95", "q95_abs_mm_mcse")]:
                mc[k] = max(mc[k], r[col])
    hdrA = ["Anchor / long-axis underestimation", "Rule", "Bias (mm)", "RMSE (mm)", "MAE (mm)",
            "95th percentile of absolute error (mm)", "Central 95% of errors (mm)", "Patients with absolute error > 2 mm"]
    v = rd(srng)
    v = v[v.axis == "AP"]
    labs = [("base family", "12 scenarios as in Table 1"), ("window +-15%", "±15% window"),
            ("overestimation off", "No overestimation"), ("beat CV 5%", "Beat-to-beat variation 5%"),
            ("beat CV 30%", "Beat-to-beat variation 30%"), ("two views", "Two views"), ("four views", "Four views"),
            ("median span 13 mm", "Median true span 13 mm")]
    rowsB = []
    for key, lab in labs:
        vv = v[v.variant.str.startswith(key)]
        assert len(vv) == 4 and vv.n_cells.eq(12).all(), key
        for code, name in RULES:
            r = one(vv, rule=code)
            rowsB.append([lab, name, rng(r.bias_mm_min, r.bias_mm_max, 2), rng(r.rmse_mm_min, r.rmse_mm_max, 2),
                          rng(r.mae_mm_min, r.mae_mm_max, 2), rng(r.q95_abs_mm_min, r.q95_abs_mm_max, 2)])
    hdrB = ["Setting", "Rule", "Bias (mm)", "RMSE (mm)", "MAE (mm)", "95th percentile of absolute error (mm)"]
    z = rd(sadd)
    z = z[(z.family == "zero_under") & (z.axis == "AP") & (z.K == 3) & (z.beat_cv == 0.15) & (z.window == 0)]
    rowsC = []
    for over, lab in [(True, "present"), (False, "absent")]:
        for code, name in RULES:
            r = one(z, view_over=over, rule=code)
            rowsC.append([lab, name, mci(r.bias_mm, r.bias_mm_mcse, 2), reg("S12", f"zero {over} {code} rmse", sadd, r.rmse_mm, 2),
                          f(r.mae_mm, 2), f(r.q95_abs_mm, 2)])
    hdrC = ["Overestimation", "Rule", "Bias (mm, 95% MCI)", "RMSE (mm)", "MAE (mm)", "95th percentile of absolute error (mm)"]
    b = v[v.variant.str.startswith("base family")].set_index("rule")
    title = ("**Supplementary Table S12. Error distribution of the four view rules across the 12 view-accuracy "
             "scenarios and added settings.** Across the 12 scenarios the 95th percentile of absolute error of the "
             f"largest view mean was {f(b.loc['A3', 'q95_abs_mm_min'], 2)} to {f(b.loc['A3', 'q95_abs_mm_max'], 2)} mm "
             f"and that of the mean across views {f(b.loc['A2', 'q95_abs_mm_min'], 2)} to "
             f"{f(b.loc['A2', 'q95_abs_mm_max'], 2)} mm; with no underestimation in any view the largest view mean "
             f"overestimated by {f(one(z, view_over=True, rule='A3').bias_mm, 2)} mm.")
    foot = ("Third amendment (post hoc). Panel A: the 12 scenarios of the second amendment (three views, three beats "
            "per view, no window, beat-to-beat variation 15%, overestimation present, median true span 10 mm), "
            "recomputed per patient from the stored seeds (20260920, spawn key (1, cell)); bias and RMSE equal the "
            "stored second-amendment values. Panel B: lowest and highest value over the 12 scenarios when one "
            "setting was changed. Panel C: no underestimation in any view (new seed 20261005, spawn key (72, 1, "
            "cell); single conditions). " + FOOT_V +
            f"Largest MCSE in panel A: bias {mc['bias']:.3f} mm, RMSE {mc['rmse']:.3f} mm, MAE {mc['mae']:.3f} mm, "
            f"95th percentile {mc['q95']:.3f} mm (percentile MCSE by delete-a-group jackknife, approximate). "
            f"Sources: {rel(s12)} (panel A), {rel(srng)} (panel B), {rel(sadd)} (panel C); generated by {SCRIPT}. "
            f"AP, anteroposterior; MAE, mean absolute error; {ABBR_MC}; RMSE, root-mean-square error.")
    return "\n\n".join([title, panel("A", "The 12 view-accuracy scenarios"), md_table(hdrA, rowsA, ["l", "l"] + ["r"] * 6),
                        panel("B", "Range over the 12 scenarios when one setting was changed"),
                        md_table(hdrB, rowsB, ["l", "l"] + ["r"] * 4),
                        panel("C", "No underestimation in any view"), md_table(hdrC, rowsC, ["l", "l"] + ["r"] * 4), foot])


def tableS13():
    st, sw = VW / "views_b_tally.csv", VW / "views_b_worstcase.csv"
    t = rd(st)
    t = t[t.axis == "AP"]
    obj = [("worst_abs_bias", "Largest absolute bias"), ("worst_rmse", "Largest RMSE"), ("worst_mae", "Largest MAE"),
           ("worst_q95abs", "Largest 95th percentile of absolute error"), ("mean_rmse", "Mean RMSE over the 12 scenarios")]
    scopes = [("72 conditions", "All 72"), ("beat CV 5%", "Beat-to-beat variation 5% (24)"),
              ("beat CV 15%", "Beat-to-beat variation 15% (24)"), ("beat CV 30%", "Beat-to-beat variation 30% (24)")]
    rowsA = []
    for o, olab in obj:
        for skey, slab in scopes:
            r = t[(t.objective == o) & t.scope.str.startswith(skey)]
            assert len(r) == 1, (o, skey)
            r = r.iloc[0]
            T = "S13"
            rowsA.append([olab, slab] + [reg(T, f"{o} {skey} {c}", st, r[c], 0) for c in
                                         ["n_win_A1", "n_win_A2", "n_win_A3", "n_win_A4", "n_win_largest_view_rule", "n_not_clear"]])
    hdrA = ["Criterion (smallest value preferred)", "Conditions", "Anchor-view mean", "Mean across views",
            "Largest view mean", "Largest view mean after review", "Either largest-view rule", "Not clear"]
    w = sel(rd(sw), S_median_mm=10.0, K=3, beat_cv=0.15, view_over=True, window=0.0, axis="AP")
    assert len(w) == 4
    rowsB = []
    for code, name in RULES:
        r = one(w, rule=code)
        rowsB.append([name] + [f"{reg('S13', f'base {code} {c}', sw, r[c], 2)} at {scen(r[c + '_scenario'])}"
                               for c in ["worst_abs_bias", "worst_rmse", "worst_mae", "worst_q95abs"]]
                     + [f(r.mean_rmse, 2)])
    hdrB = ["Rule", "Largest absolute bias (mm)", "Largest RMSE (mm)", "Largest MAE (mm)",
            "Largest 95th percentile of absolute error (mm)", "Mean RMSE (mm)"]
    rowsC = []
    for o, olab in obj:
        r = t[(t.objective == o) & t.scope.str.startswith("36 conditions")]
        assert len(r) == 1
        r = r.iloc[0]
        rowsC.append([olab] + [f"{int(r[c])}" for c in ["n_win_A1", "n_win_A2", "n_win_A3", "n_win_A4",
                                                         "n_win_largest_view_rule", "n_not_clear"]])
    hdrC = ["Criterion", "Anchor-view mean", "Mean across views", "Largest view mean",
            "Largest view mean after review", "Either largest-view rule", "Not clear"]
    g = lambda o, c: int(t[(t.objective == o) & t.scope.str.startswith("72 conditions")].iloc[0][c])
    g30 = lambda o, c: int(t[(t.objective == o) & t.scope.str.startswith("beat CV 30%")].iloc[0][c])
    assert g("worst_rmse", "n_win_largest_view_rule") == g("worst_mae", "n_win_largest_view_rule") == \
        g("worst_q95abs", "n_win_largest_view_rule") == 72
    title = ("**Supplementary Table S13. Preferred rule by worst-case criterion.** A largest-view rule had the "
             "smallest worst-case RMSE, MAE and 95th percentile of absolute error in 72 of 72 conditions and the "
             f"smallest worst-case absolute bias in {g('worst_abs_bias', 'n_win_largest_view_rule')}; the mean "
             f"across views had the smallest mean RMSE in {g('mean_rmse', 'n_win_A2')} conditions, "
             f"{'all' if g30('mean_rmse', 'n_win_A2') == g('mean_rmse', 'n_win_A2') else g30('mean_rmse', 'n_win_A2')} of them at 30% beat-to-beat variation.")
    foot = ("Third amendment (post hoc). For each of 72 conditions (two, three or four views; "
            "beat-to-beat variation 5%, 15% or 30%; overestimation present or absent; no window or ±15% window; "
            "median true span 10 or 13 mm) the worst case of a rule is its largest value over the 12 view-accuracy "
            "scenarios, and the preferred rule is the one with the smallest worst case; no scenario weights and no "
            "regret were used. The 72 conditions by 12 scenarios are the 864 cells of the second-amendment grid. "
            "Panel A: number of conditions in which each rule was preferred; not clear, the gap to the runner-up "
            "was within 1.96 MCSE (paired jackknife within a cell; independent combination when the two worst cases "
            "fell in different cells). Panel B: worst-case values in the condition of Table 1 (three views, 15%, "
            "overestimation present, no window, 10 mm), with the scenario attaining each. Panel C: the 36 "
            "conditions with median span 10 mm after adding the setting without underestimation as a 13th scenario "
            "(single conditions, new seed). " + FOOT_V + "Seeds 20260920 (stored, 864 cells) and 20261005 (spawn "
            f"key (72, 1, cell)). Sources: {rel(st)} (panels A and C), {rel(sw)} (panel B); generated by {SCRIPT}. "
            f"MAE, mean absolute error; {ABBR_MC}; RMSE, root-mean-square error.")
    return "\n\n".join([title, panel("A", "Number of conditions in which each rule was preferred (12 scenarios)"),
                        md_table(hdrA, rowsA, ["l", "l"] + ["r"] * 6),
                        panel("B", "Worst-case values in the condition of Table 1"), md_table(hdrB, rowsB),
                        panel("C", "Number of conditions (of 36) with a 13th scenario without underestimation"),
                        md_table(hdrC, rowsC), foot])


def tableS14():
    sd = VW / "views_d_decomposition.csv"
    d = sel(rd(sd), axis="AP", K=3, view_over=True, contrast="A3-A1")
    assert len(d) == 12 and d.window.eq(0).all() and d.n_patients.eq(100000).all()
    d = d.assign(_a=d.anchor_mean_pct, _l=d.long_mean_pct).sort_values(["_a", "_l"])
    rows, mc = [], 0.0
    for _, r in d.iterrows():
        T, s = "S14", r.scenario
        rows.append([scen(s), reg(T, f"{s} full", sd, r.excess_full, 2), reg(T, f"{s} variability only", sd, r.excess_equal_views, 2),
                     reg(T, f"{s} offsets only", sd, r.excess_zero_noise, 2), reg(T, f"{s} interaction", sd, r.interaction, 2),
                     n_(r.count_sel_nonanchor_full), n_(r.count_sel_overestimated_full)])
        mc = max(mc, r.excess_full_mcse, r.excess_equal_views_mcse, r.excess_zero_noise_mcse, r.interaction_mcse)
    n_off = int((d.excess_zero_noise > d.excess_equal_views).sum())
    n_below = int((d.excess_full < d.excess_equal_views).sum())
    hdr = ["Anchor / long-axis underestimation", "Excess, full model (mm)", "Within-view variability only (mm)",
           "Differences between views only (mm)", "Interaction (mm)", "Patients whose largest view was not the anchor",
           "Patients whose largest view carried overestimation"]
    src2 = AM2 / "E1bE3b_e1_sensitivity.csv"
    e2 = rd(src2)
    e2 = e2[(e2.scenario == "base") & (e2.axis == "AP") & e2.infl_min.notna()]
    assert len(e2) == 1, len(e2)
    e2 = e2.iloc[0]
    stored = (reg("S14", "stored excess min", src2, e2.infl_min, 2), reg("S14", "stored excess max", src2, e2.infl_max, 2))
    assert stored == ("0.31", "2.48"), stored
    title = ("**Supplementary Table S14. Decomposition of the excess of the largest view mean over the anchor-view "
             f"mean.** The excess was {rng(d.excess_full.min(), d.excess_full.max(), 2)} mm ({stored[0]} to "
             f"{stored[1]} mm in the stored second-amendment output, the values of the main text); within-view variability "
             f"alone gave {rng(d.excess_equal_views.min(), d.excess_equal_views.max(), 2)} mm and differences "
             f"between views alone {rng(d.excess_zero_noise.min(), d.excess_zero_noise.max(), 2)} mm, the two did "
             f"not add up (interaction {rng(d.interaction.min(), d.interaction.max(), 2)} mm), and differences "
             f"between views gave the larger excess in {n_off} of 12 scenarios.")
    foot = ("Third amendment (post hoc). Excess: mean of the largest view mean minus the anchor-view mean, three "
            "views, three beats per view, no window, beat-to-beat variation 15%, overestimation present. Four arms "
            "were computed on the same simulated patients: the full model; within-view variability only (every view "
            "given the anchor view's under- and overestimation); differences between views only (beat-to-beat "
            "variation, caliper error and reader offset removed); and neither (excess exactly 0). Interaction = "
            "full minus the two single-source arms. The within-view variability arm is dominated by simulated "
            "beat-to-beat variation and is not measurement error alone. The full excess was below the "
            f"variability-only excess in {n_below} of 12 scenarios. The decomposition depends on this definition "
            f"of equal views. Patient counts are of 100,000, full model. n = 100,000 simulated patients per "
            f"scenario; seed 20261005, spawn key (72, 4, .); paired MCSE at most {mc:.3f} mm. Sources: {rel(sd)} "
            f"and, for the stored second-amendment range, {rel(src2)}; "
            f"generated by {SCRIPT}. {ABBR_MC}.")
    return "\n\n".join([title, md_table(hdr, rows, ["l"] + ["r"] * 6), foot])


def tableS15():
    sq, sr_ = VW / "views_e_conditional_quantiles.csv", VW / "views_e_ratio_summary.csv"
    q = sel(rd(sq), scenario="4.8/20", window=0.15, rhythm="sinus", axis="AP")
    bins = ["0 to 5", "5 to 7", "7 to 10", "10 to 13", "13 to 16", ">= 16", "all"]
    blab = {"0 to 5": "< 5", "5 to 7": "5 to < 7", "7 to 10": "7 to < 10", "10 to 13": "10 to < 13",
            "13 to 16": "13 to < 16", ">= 16": "≥ 16", "all": "All"}
    rows, mcq = [], 0.0
    for b in bins:
        for code, name in RULES:
            r = one(q, span_bin=b, rule=code)
            T = "S15"
            rows.append([blab[b], n_(r.n), name, f(r.bias_mm, 2), reg(T, f"{b} {code} rmse", sq, r.rmse_mm, 2),
                         f"{reg(T, f'{b} {code} q2.5', sq, r['q2.5_err_mm'], 2)} to {reg(T, f'{b} {code} q97.5', sq, r['q97.5_err_mm'], 2)}",
                         f(r.q95_abs_mm, 2)])
            mcq = max(mcq, r["q2.5_err_mm_mcse"], r["q97.5_err_mm_mcse"], r.q95_abs_mm_mcse)
    hdr = ["True span (mm)", "Patients", "Rule", "Bias (mm)", "RMSE (mm)", "Central 95% of errors (mm)",
           "95th percentile of absolute error (mm)"]
    rs = rd(sr_)
    rs = rs[rs.axis == "AP"]
    rowsB = [[RULE[r.rule], f"{int(r.n_cells)}", reg("S15", f"ratio min {r.rule}", sr_, r.ratio_min, 2), f(r.ratio_median, 2),
              reg("S15", f"ratio max {r.rule}", sr_, r.ratio_max, 2), f"{int(r.n_cells_q95abs_gt_2rmse)}"] for _, r in rs.iterrows()]
    hdrB = ["Rule", "Cells", "Lowest ratio", "Median ratio", "Highest ratio", "Cells with ratio above 2"]
    # the same condition in atrial fibrillation (stored with the package, spawn key (72, 5, 1))
    qa = sel(rd(sq), scenario="4.8/20", window=0.15, rhythm="AF", axis="AP")
    assert len(qa) == 28 and set(qa.rr_cv_af) == {0.2} and set(qa.seed_spawn_key) == {"72,5,1"}
    rowsC = []
    for b in bins:
        for code, name in RULES:
            r = one(qa, span_bin=b, rule=code)
            rowsC.append([blab[b], n_(r.n), name, f(r.bias_mm, 2), reg("S15", f"AF {b} {code} rmse", sq, r.rmse_mm, 2),
                          f"{reg('S15', f'AF {b} {code} q2.5', sq, r['q2.5_err_mm'], 2)} to {reg('S15', f'AF {b} {code} q97.5', sq, r['q97.5_err_mm'], 2)}",
                          f(r.q95_abs_mm, 2)])
            mcq = max(mcq, r["q2.5_err_mm_mcse"], r["q97.5_err_mm_mcse"], r.q95_abs_mm_mcse)
    a3 = lambda b: one(q, span_bin=b, rule="A3")
    title = ("**Supplementary Table S15. Error quantiles conditional on true span.** In the base case the central "
             f"95% of errors of the largest view mean ran from {f(a3('all')['q2.5_err_mm'], 2)} to "
             f"{f(a3('all')['q97.5_err_mm'], 2)} mm overall, from {f(a3('5 to 7')['q2.5_err_mm'], 2)} to "
             f"{f(a3('5 to 7')['q97.5_err_mm'], 2)} mm for true spans of 5 to 7 mm and from "
             f"{f(a3('>= 16')['q2.5_err_mm'], 2)} to {f(a3('>= 16')['q97.5_err_mm'], 2)} mm for 16 mm or more; the "
             f"95th percentile of absolute error was {f(rs.ratio_min.min(), 2)} to {f(rs.ratio_max.max(), 2)} times "
             "the RMSE.")
    foot = ("Third amendment (post hoc). Panel A: base case (three views, three beats per view, ±15% window, "
            "beat-to-beat variation 15%, anchor underestimation 4.8%, long-axis underestimation 20%, sinus rhythm), "
            "new seed 20261005, spawn key (72, 5, 0), n = 100,000 simulated patients; AP axis; error is measured "
            "minus true maximal span. The quantiles are conditional on the true span, which a reader does not "
            "know; quantiles conditional on the measured span were not computed, so the intervals are not margins "
            f"for an individual patient. Percentile MCSE (delete-a-group jackknife, approximate) at most {mcq:.2f} "
            "mm. Panel B: ratio of the 95th percentile of absolute error to RMSE over the 864 cells of the "
            "second-amendment grid (stored seeds 20260920). Panel C: as panel A in atrial fibrillation (RR-interval "
            f"CV 20%; spawn key (72, 5, 1)). Sources: {rel(sq)} (panels A and C), {rel(sr_)} (panel B); "
            f"generated by {SCRIPT}. AP, anteroposterior; {ABBR_MC}; RMSE, root-mean-square error.")
    return "\n\n".join([title, panel("A", "Base case, by true span"), md_table(hdr, rows, ["l", "r", "l", "r", "r", "r", "r"]),
                        panel("B", "Ratio of the 95th percentile of absolute error to RMSE, 864 cells"),
                        md_table(hdrB, rowsB),
                        panel("C", "Base case in atrial fibrillation, by true span"),
                        md_table(hdr, rowsC, ["l", "r", "l", "r", "r", "r", "r"]), foot])



# ====================================================================== S10 and S11 (beat rules, third amendment)
BT = OS / "beats"


def tableS10():
    sp, se, sb, sl = (BT / "beats_table_rmse_difference_published_vs_paired.csv", BT / "beats_table_equal_expected_beats.csv",
                      BT / "beats_table_equal_budget.csv", BT / "beats_table_limited_denominators.csv")
    p = sel(rd(sp), axis="AP", estimand="T1", N=3)
    e = sel(rd(se), grid="budget", axis="AP", estimand="T1", N=3)
    T = "S10"
    cols = [("sinus", "sinus_cv15", "Sinus rhythm"), ("AF_rr0.2", "af_cv15", "Atrial fibrillation")]
    A = {k: [] for k in range(6)}
    for rs, cond, _ in cols:
        rp, re_ = one(p, rhythm_state=rs), one(e, condition=cond)
        assert rp.a3_condition == cond and not rp.published_arms_share_random_numbers
        A[0].append(f"{reg(T, cond + ' planned diff', sp, rp.published_diff_mm, 2)} ({reg(T, cond + ' planned lo', sp, rp.published_lo95, 2)} to {reg(T, cond + ' planned hi', sp, rp.published_hi95, 2)})")
        A[1].append(f"{reg(T, cond + ' paired diff', sp, rp.a3_diff_mm, 2)} ({reg(T, cond + ' paired lo', sp, rp.a3_paired_lo95, 2)} to {reg(T, cond + ' paired hi', sp, rp.a3_paired_hi95, 2)})")
        for k, pre in [(2, "selection_window_minus_same_acquired"), (3, "window_minus_plain_same_count_distribution"),
                       (4, "acquisition_same_acquired_minus_plain_first_N")]:
            d, m = re_[pre + "_d_rmse_mm"], re_[pre + "_d_rmse_mcse"]
            A[k].append(f"{reg(T, f'{cond} {pre}', se, d, 2)} ({f(d - Z * m, 2)} to {f(d + Z * m, 2)})")
        A[5].append(f"{reg(T, cond + ' beats acquired', se, re_.mean_beats_acquired_window, 2)}")
    labA = ["Planned estimate: windowed mean minus plain mean of three beats, independently simulated conditions",
            "Same simulated beats: windowed mean minus plain mean of the first three beats",
            "Same acquired beats: windowed mean minus plain mean of all beats the windowed rule acquired",
            "Same number of beats: windowed mean minus plain mean with the same distribution of beat counts",
            "Acquisition: plain mean of all acquired beats minus plain mean of the first three beats",
            "Mean number of beats acquired by the windowed rule"]
    rowsA = [[labA[k]] + A[k] for k in range(6)]
    hdrA = ["Contrast (RMSE difference, mm, 95% MCI)"] + [c[2] for c in cols]
    b = sel(rd(sb), condition="sinus_cv15", axis="AP", estimand="T1", N=3)
    rowsB = []
    for B in [4, 5, 6, 8, 10, 13]:
        r = one(b, budget=B)
        rowsB.append([f"{B}", kn(r.n_accepted_within_budget, r.n_views), reg(T, f"B{B} beats", sb, r.mean_beats_acquired_window, 2),
                      f(r.window_rmse_mm, 2), f(r.plain_all_B_rmse_mm, 2),
                      f"{reg(T, f'B{B} win-all', sb, r.window_minus_plain_all_B_d_rmse_mm, 3)} ({f(r.window_minus_plain_all_B_d_rmse_mcse, 3)})",
                      em(r.window_minus_plain_first_N_d_rmse_mm, r.window_minus_plain_first_N_d_rmse_mcse, 3, 3)])
    hdrB = ["Largest number of beats allowed", "Views with three accepted beats", "Mean beats used by the windowed rule",
            "RMSE, windowed mean (mm)", "RMSE, plain mean of all beats allowed (mm)",
            "Windowed minus plain mean of all beats allowed (mm, MCSE)", "Windowed minus plain mean of the first three (mm, MCSE)"]
    l = rd(sl)
    l = l[l.axis == "AP"]
    rowsC = []
    for _, r in l.iterrows():
        pub = r.source.startswith("published")
        rh = "Atrial fibrillation" if ("AF" in r.rhythm_state or "af" in r.rhythm_state) else "Sinus rhythm"
        key = f"{'pub' if pub else 'a3'} {rh[:2]} N{r.N} {r.window}"
        rowsC.append(["Planned run" if pub else "Third amendment", rh, f"{int(r.N)}", "none" if r.window == "none" else "±15%",
                      f"{reg(T, key + ' limited', sl, r.n_limited, 0) and n_(r.n_limited)} of {n_(r.n_views)} ({pc(r.n_limited, r.n_views)})",
                      "not stored" if pub else n_(r.n_fewer_than_N_available),
                      "not stored" if pub else (f"{n_(r.n_enough_beats_window_not_met)} of {n_(r.n_views_with_at_least_N_available)} "
                                                f"({pc(r.n_enough_beats_window_not_met, r.n_views_with_at_least_N_available)})")])
    hdrC = ["Run", "Rhythm", "Beats required", "Window", "Views with limited sampling", "Views with fewer stored beats than required",
            "Views with enough stored beats and no qualifying set"]
    rs_, re_s = one(p, rhythm_state="sinus"), one(e, condition="sinus_cv15")
    title = ("**Supplementary Table S10. Beat rules compared on matched beats, and denominators of limited sampling.** "
             f"On the same simulated beats the windowed mean had an RMSE {f(rs_.a3_diff_mm, 2)} mm higher than the "
             f"plain mean of three beats, and {f(re_s.selection_window_minus_same_acquired_d_rmse_mm, 2)} mm higher "
             "than the plain mean of all the beats that the windowed rule had acquired; in retrospective data "
             "limited sampling without a window reflected the number of stored beats alone.")
    foot = ("Panel A, first row: planned experiment E2 (seed 20260918; windowed and plain conditions simulated with "
            "independent random-number streams, so the MCSE of the difference is the root sum of squares of the two "
            "MCSE). All other rows and panel B: third amendment (post hoc), seed 20261005, spawn key (71, grid, "
            "condition, chunk, stream); all rules of a condition were applied to the same simulated beats, reader "
            "offsets and caliper errors, and the MCSE of each RMSE difference is the delta-method MCSE of the paired "
            "squared errors. Anchor view, AP axis, three accepted beats, ±15% window, beat-to-beat variation 15%, "
            "prospective acquisition of at most 30 beats; error against the true maximal span; n = 100,000 simulated "
            "views per condition. The windowed rule stopped at the first set of three beats within the window; in "
            "panel B the plain reader averaged every beat allowed, so that comparison is not at an equal number of "
            "beats. Panel C: retrospective data (stored beats zero-truncated Poisson with rate 4, cap 15, independent "
            "of rhythm and span by assumption); limited sampling, no set of the required number of beats met the "
            "rule; the planned run stored the first count only, and its four rows per rhythm come from four "
            "independently simulated sets of 100,000 views, whereas each third-amendment rhythm uses one set. "
            f"Sources: {rel(sp)} (panel A, rows 1 and 2), {rel(se)} (panel A, rows 3 to 6), {rel(sb)} (panel B), "
            f"{rel(sl)} (panel C); generated by {SCRIPT}. AP, anteroposterior; {ABBR_MC}; RMSE, root-mean-square error.")
    return "\n\n".join([title, panel("A", "Effect of the ±15% window on RMSE, three accepted beats"),
                        md_table(hdrA, rowsA), panel("B", "Fixed largest number of beats, sinus rhythm"),
                        md_table(hdrB, rowsB, ["r"] * 7),
                        panel("C", "Limited sampling in retrospective data"),
                        md_table(hdrC, rowsC, ["l", "l", "r", "l", "r", "r", "r"]), foot])


def tableS11():
    ss = BT / "beats_table_sensitivity.csv"
    s = rd(ss)
    both = s.T1_win3_minus_plain3_d_rmse_mm
    n44, n44_up = len(s), int((both - Z * s.T1_win3_minus_plain3_d_rmse_mcse > 0).sum())
    assert n44 == 44 and set(s.axis) == {"AP", "SL"} and (both > 0).all() and n44_up == 44
    s = s[s.axis == "AP"]
    assert s.n_views.eq(100000).all()
    labs = [("ref", "Reference (independent beats, fixed-size caliper error)"),
            ("ar1_0.3", "Serial correlation 0.3"), ("ar1_0.6", "Serial correlation 0.6"), ("ar1_0.9", "Serial correlation 0.9"),
            ("resp_add_0.05", "Respiratory component 5%, added"), ("resp_add_0.10", "Respiratory component 10%, added"),
            ("resp_pres_0.05", "Respiratory component 5%, total variation kept"),
            ("resp_pres_0.10", "Respiratory component 10%, total variation kept"),
            ("cal_prop", "Caliper error proportional to span"), ("cal_mixed", "Caliper error part fixed, part proportional"),
            ("combined", "Serial correlation 0.6, respiratory component 10% (added) and caliper error part fixed, part "
             "proportional")]
    hdr = ["Setting", "First three beats within ±15% (of 100,000)", "RMSE, one beat (mm)", "RMSE, plain mean of three beats (mm)",
           "Window effect, three beats (mm, 95% MCI)", "Window effect, five beats (mm, 95% MCI)",
           "Shift in relative bias, three beats (percentage points, 95% MCI)"]
    out, n3_hi, n5_hi, n3_lo = [], 0, 0, 0
    T = "S11"
    for rh, rlab in [("sinus", "Sinus rhythm"), ("af", "Atrial fibrillation")]:
        rows = []
        for key, lab in labs:
            r = one(s, condition=f"{rh}_{key}")
            d3, m3 = r.T1_win3_minus_plain3_d_rmse_mm, r.T1_win3_minus_plain3_d_rmse_mcse
            d5, m5 = r.T1_win5_minus_plain5_d_rmse_mm, r.T1_win5_minus_plain5_d_rmse_mcse
            rb, mb = r.Tv_win3_minus_plain3_d_relbias_pct, r.Tv_win3_minus_plain3_d_relbias_mcse
            n3_hi += d3 - Z * m3 > 0
            n3_lo += d3 + Z * m3 < 0
            n5_hi += d5 - Z * m5 > 0
            rows.append([lab, reg(T, f"{rh} {key} met3", ss, r.n_met_first_3, 0) and n_(r.n_met_first_3),
                         f(r.T1_plain_1_rmse_mm, 2), reg(T, f"{rh} {key} plain3 rmse", ss, r.T1_plain_3_rmse_mm, 2),
                         f"{reg(T, f'{rh} {key} win3 effect', ss, d3, 3)} ({f(d3 - Z * m3, 3)} to {f(d3 + Z * m3, 3)})",
                         mci(d5, m5, 3), f"{reg(T, f'{rh} {key} relbias shift', ss, rb, 2)} ({f(rb - Z * mb, 2)} to {f(rb + Z * mb, 2)})"])
        out += [panel("A" if rh == "sinus" else "B", rlab), md_table(hdr, rows)]
    g = lambda c, col: one(s, condition=c)[col]
    title = ("**Supplementary Table S11. Sensitivity of the beat-rule results to serial correlation, a respiratory "
             "component and the form of caliper error.** The window did not lower RMSE in any setting; with serial "
             f"correlation of 0.6 between consecutive beats the RMSE of a three-beat mean rose from "
             f"{f(g('sinus_ref', 'T1_plain_3_rmse_mm'), 2)} to {f(g('sinus_ar1_0.6', 'T1_plain_3_rmse_mm'), 2)} mm and "
             f"the window effect fell from {f(g('sinus_ref', 'T1_win3_minus_plain3_d_rmse_mm'), 2)} to "
             f"{f(g('sinus_ar1_0.6', 'T1_win3_minus_plain3_d_rmse_mm'), 2)} mm, and the direction of the shift in "
             "relative bias depended on the form of caliper error.")
    assert n3_lo == 0
    foot = ("Third amendment (post hoc); none of the added settings has an empirical source. Anchor view, AP axis, "
            "beat-to-beat variation 15%, prospective acquisition; n = 100,000 simulated views per setting; seed "
            "20261005, spawn key (71, 2, condition, chunk, stream). Within a setting all rules were applied to the "
            "same simulated beats. Window effect: RMSE of the windowed mean minus RMSE of the plain mean of the same "
            "number of first beats, against the true maximal span, with the delta-method MCSE of the paired squared "
            "errors. Shift in relative bias: windowed minus plain, against the anchor-view median span. "
            "Serial correlation: first-order autoregressive beat-to-beat variation with the stated correlation "
            "between consecutive beats. Respiratory component: sinusoidal variation of the stated amplitude, either "
            "added to the beat-to-beat variation or with the total variation kept at 15%. Caliper error: SD 1.0 mm "
            "in the reference, proportional to span, or a mixture of the two. The 95% MCI of the three-beat window "
            f"effect lay above zero in {n3_hi} of 22 settings and that of the five-beat effect in {n5_hi} of 22; "
            "neither lay below zero in any setting. With the SL axis, which is not shown, the three-beat window "
            f"effect was above zero in {reg('S11', 'three-beat effect above zero, both axes', ss, n44_up, 0)} of {n44} "
            f"comparisons (22 settings on two axes). Source: {rel(ss)}; generated by {SCRIPT}. AP, anteroposterior; "
            f"{ABBR_MC}; RMSE, root-mean-square error; SL, septolateral.")
    return "\n\n".join([title] + out + [foot])



# ====================================================================== S16 and S17 (cut-offs, third amendment)
TH = OS / "thresholds"


def tableS16():
    s2 = TH / "thresholds_2x2.csv"
    d = rd(s2)
    d = d[(d["sample"] == "published_seed") & (d.axis == "AP")]
    T = "S16"

    def block(scenario):
        b = d[d.scenario == scenario]
        assert len(b) == 12 and b.n.eq(100000).all()
        rows = []
        for c in [7.0, 10.0, 13.0]:
            for code, name in RULES:
                r = one(b, cutoff_mm=c, rule=code)
                assert r.tp + r.fp + r.fn + r.tn == r.n
                k = f"{scenario} {c:g} {code}"
                se = lambda x: 1000 * (x / r.n * (1 - x / r.n) / r.n) ** 0.5
                rows.append([f"{c:g}", name, reg(T, k + " tp", s2, r.tp, 0) and n_(r.tp), reg(T, k + " fp", s2, r.fp, 0) and n_(r.fp),
                             reg(T, k + " fn", s2, r.fn, 0) and n_(r.fn), reg(T, k + " tn", s2, r.tn, 0) and n_(r.tn),
                             f"{f(100 * r.sensitivity, 1)} / {f(100 * r.specificity, 1)}",
                             f"{f(100 * r.ppv, 1)} / {f(100 * r.npv, 1)}",
                             f"{f(r.fp_per_1000, 1)} ({f(se(r.fp), 2)}) / {f(r.fn_per_1000, 1)} ({f(se(r.fn), 2)})"])
        return rows
    hdr = ["Cut-off (mm)", "Rule", "True positive", "False positive", "False negative", "True negative",
           "Sensitivity / specificity (%)", "PPV / NPV (%)", "False positive / false negative per 1,000 patients (MCSE)"]
    rowsC = []
    b = d[d.scenario == "base"]
    for c in [7.0, 10.0, 13.0]:
        for code in ["A2", "A3", "A4"]:
            r = one(b, cutoff_mm=c, rule=code)
            rowsC.append([f"{c:g}", RULE[code],
                          f"{reg(T, f'youden {c:g} {code}', s2, r.nri_vs_A1, 1, 100)} ({f(100 * r.nri_vs_A1_lo95, 1)} to {f(100 * r.nri_vs_A1_hi95, 1)})",
                          f"{reg(T, f'netcorrect {c:g} {code}', s2, r.net_correct_per_1000_vs_A1, 1)} ({f(r.net_correct_per_1000_vs_A1_lo95, 1)} to {f(r.net_correct_per_1000_vs_A1_hi95, 1)})"])
    hdrC = ["Cut-off (mm)", "Rule", "Difference in Youden index (percentage points, 95% MCI)",
            "Difference in correctly classified patients per 1,000 (95% MCI)"]
    b13 = b[b.cutoff_mm == 13.0]
    tot = (b13.fp_per_1000 + b13.fn_per_1000)
    b7 = b[b.cutoff_mm == 7.0]
    tot7 = (b7.fp_per_1000 + b7.fn_per_1000)
    a3 = lambda c: one(b, cutoff_mm=c, rule="A3")
    title = ("**Supplementary Table S16. Two-by-two classification tables at the illustrative cut-offs of 7, 10 and "
             f"13 mm.** At 13 mm in the base case the four rules misclassified {f(tot.min(), 0)} to {f(tot.max(), 0)} per "
             "1,000 patients and differed mainly in the split between false positives and false negatives, whereas "
             f"at 7 mm they misclassified {f(tot7.min(), 0)} to {f(tot7.max(), 0)} per 1,000; "
             "against the anchor-view mean, the largest view mean changed the Youden index and the number of "
             f"correctly classified patients in opposite directions at 7 mm ({f(100 * a3(7.0).nri_vs_A1, 1)} "
             f"percentage points and {f(a3(7.0).net_correct_per_1000_vs_A1, 1)} per 1,000) and at 13 mm "
             f"({f(100 * a3(13.0).nri_vs_A1, 1)} and {f(a3(13.0).net_correct_per_1000_vs_A1, 1)}).")
    foot = ("Third amendment (post hoc) tabulation of the planned cut-off analysis (E3): the 100,000 simulated "
            "patients of each planned condition were rebuilt from the stored seeds (20260918, spawn keys (3, 29) "
            "and (3, 77)), and sensitivity and specificity equal the stored values. Event: true AP span at or above "
            "the cut-off; positive: reported value at or above the cut-off. The cut-offs are illustrative values on "
            "the simulated span scale and are not clinical eligibility criteria; 81.4%, 50.0% and 25.6% of true AP "
            "spans equalled or exceeded 7, 10 and 13 mm (Supplementary Figure S6). Three views, three beats per "
            "view, ±15% window, beat-to-beat variation 15%. MCSE of counts per 1,000 is binomial. Panel C: paired "
            "differences against the anchor-view mean on the same patients; the Youden index (sensitivity plus "
            "specificity minus 1) weights events and non-events equally, whereas the count of correctly classified "
            "patients weights them by prevalence. n = 100,000 simulated patients per condition. "
            f"Source: {rel(s2)} (sample published_seed); generated by {SCRIPT}. AP, anteroposterior; {ABBR_MC}; NPV, "
            "negative predictive value; PPV, positive predictive value.")
    al = ["r", "l"] + ["r"] * 7
    return "\n\n".join([title, panel("A", "Base case"), md_table(hdr, block("base"), al),
                        panel("B", "Low-underestimation scenario (anchor 2.4%, long axis 8%)"), md_table(hdr, block("low8"), al),
                        panel("C", "Base case: each rule against the anchor-view mean"), md_table(hdrC, rowsC, ["r", "l", "r", "r"]), foot])


def tableS17():
    sb, sd = TH / "thresholds_near_band.csv", TH / "thresholds_by_distance.csv"
    b = rd(sb)
    b = b[(b["sample"] == "amend3_pooled") & (b.scenario == "base") & (b.axis == "AP")]
    assert len(b) == 24 and b.n_all.eq(10_000_000).all()
    T, rows, mc = "S17", [], 0.0
    for c in [7.0, 10.0, 13.0]:
        for hw in [1.0, 2.0]:
            for code, name in RULES:
                r = one(b, cutoff_mm=c, half_width_mm=hw, rule=code)
                k = f"{c:g} {hw:g} {code}"
                rows.append([f"{c:g}", f"{hw:g}", n_(r.n_band), name,
                             reg(T, k + " mis/1000 band", sb, r.misclassified_per_1000_band, 1),
                             f"{f(r.fp_per_1000_band, 1)} / {f(r.fn_per_1000_band, 1)}",
                             f"{reg(T, k + ' fp/1000 all', sb, r.fp_band_per_1000_all, 1)} / {reg(T, k + ' fn/1000 all', sb, r.fn_band_per_1000_all, 1)}",
                             f"{f(100 * r.share_of_all_fp_in_band, 1)} / {f(100 * r.share_of_all_fn_in_band, 1)}"])
                mc = max(mc, r.misclassified_per_1000_band_mcse)
    hdr = ["Cut-off (mm)", "True span within (mm)", "Patients in band (of 10,000,000)", "Rule",
           "Misclassified per 1,000 patients in band", "False positive / false negative per 1,000 patients in band",
           "False positive / false negative in band per 1,000 of all patients",
           "Share of all false positives / false negatives lying in band (%)"]
    d = rd(sd)
    d = d[(d["sample"] == "amend3_pooled") & (d.scenario == "base") & (d.axis == "AP") & (d.cutoff_mm == 13.0)]
    rowsB = []
    for lo in [-5.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.5]:
        et = "false_positive" if lo < 0 else "false_negative"
        x = d[(d.d_lo_mm == lo) & (d.error_type == et)]
        assert len(x) == 4, (lo, len(x))
        hi = x.d_hi_mm.iloc[0]
        cells = [f"{f(lo, 1)} to {f(hi, 1)}", "false positive" if lo < 0 else "false negative", n_(x.n.iloc[0])]
        for code in RULE:
            r = one(x, rule=code)
            cells.append(reg(T, f"dist {lo:g} {code}", sd, r.p_misclassified, 1, 100))
        rowsB.append(cells)
    hdrB = ["True span minus 13 mm (mm)", "Error possible", "Patients (of 10,000,000)"] + [f"{n} (%)" for n in RULE.values()]
    w1 = b[b.half_width_mm == 1.0].misclassified_per_1000_band
    w13 = b[(b.half_width_mm == 1.0) & (b.cutoff_mm == 13.0)].misclassified_per_1000_band
    title = ("**Supplementary Table S17. Misclassification near each cut-off.** Of simulated patients whose true span "
             f"lay within 1 mm of a cut-off, {f(w1.min(), 0)} to {f(w1.max(), 0)} per 1,000 were misclassified under "
             f"every rule ({f(w13.min(), 0)} to {f(w13.max(), 0)} at 13 mm), and misclassification was not confined "
             "to a 2 mm margin.")
    foot = ("Third amendment (post hoc). Base case (three views, three beats per view, ±15% window, beat-to-beat "
            "variation 15%, anchor underestimation 4.8%, long-axis underestimation 20%), AP axis; pooled sample of "
            "n = 10,000,000 simulated patients (100 replicates of 100,000; seed 20261005, spawn key (73, 1, "
            "replicate); the four rules share patients). The cut-offs are illustrative and are not clinical "
            "eligibility criteria. Panel A gives both denominators: patients in the band, and all patients. Binomial "
            f"MCSE of the misclassified count per 1,000 in band at most {mc:.2f}. Panel B: percentage of patients "
            "in each 0.5 mm interval of true span classified on the wrong side of the 13 mm cut-off; a patient "
            "below the cut-off can only be a false positive and a patient at or above it only a false negative. "
            f"Sources: {rel(sb)} (panel A), {rel(sd)} (panel B), sample amend3_pooled; generated by {SCRIPT}. AP, "
            f"anteroposterior; {ABBR_MC}.")
    return "\n\n".join([title, panel("A", "Bands of 1 and 2 mm around each cut-off"),
                        md_table(hdr, rows, ["r", "r", "r", "l", "r", "r", "r", "r"]),
                        panel("B", "By distance of the true span from the 13 mm cut-off"),
                        md_table(hdrB, rowsB, ["l", "l", "r", "r", "r", "r", "r"]), foot])


# ====================================================================== S18 to S21 (triggers and review)
TR = OS / "triggers"


def tableS18():
    sc, sr_, sp, si = (TR / "triggers_pair_2x2_counts.csv", TR / "triggers_pair_rates.csv",
                       TR / "triggers_patient_2x2.csv", TR / "triggers_interval_methods.csv")
    c = sel(rd(sc), axis="AP", definition="D0_published")
    r = sel(rd(sr_), axis="AP", definition="D0_published")
    T = "S18"

    def rate(coh, tau, m):
        x = one(r, cohort=coh, tau_mm=tau, measure=m)
        return (f"{reg(T, f'{coh} {tau:g} {m}', sr_, x.value, 1 if m != 'false_positive_rate' else 2, 100)}% "
                f"({n_(x.numerator)} of {n_(x.denominator)}; "
                f"{f(100 * x.lo95_cluster, 1 if m != 'false_positive_rate' else 2)}% to "
                f"{f(100 * x.hi95_cluster, 1 if m != 'false_positive_rate' else 2)}%)")
    rowsA = []
    for coh, clab in [("published", "Planned run (n = 100,000 patients)"), ("new_pooled", "Replication (n = 1,000,000 patients)")]:
        for tau in [3.0, 5.0]:
            x = one(c, cohort=coh, tau_mm=tau)
            k = f"{coh} {tau:g}"
            rowsA.append([clab, f"{tau:g}", reg(T, k + " TP", sc, x.fired_error, 0) and n_(x.fired_error),
                          reg(T, k + " FP", sc, x.fired_no_error, 0) and n_(x.fired_no_error),
                          n_(x.not_fired_error), n_(x.not_fired_no_error),
                          rate(coh, tau, "sensitivity"), rate(coh, tau, "false_positive_rate"), rate(coh, tau, "ppv")])
    hdrA = ["Sample", "Trigger threshold (mm)", "Fired, error", "Fired, no error", "Not fired, error", "Not fired, no error",
            "Sensitivity (count; 95% MCI)", "False-positive rate (count; 95% MCI)", "PPV (count; 95% MCI)"]
    p = sel(rd(sp), scope="AP", cohort="published")
    truths = [("any_pair:D0_published", "View error above 2 mm in at least one view pair"),
              ("reported_wrong:A3_largest", "Largest view mean wrong by more than 2 mm"),
              ("reported_wrong:A1_anchor", "Anchor-view mean wrong by more than 2 mm")]
    rowsB = []
    for tr_, tl in truths:
        for tau in [3.0, 5.0]:
            x = one(p, truth=tr_, tau_mm=tau)
            ne, nn = x.fired_error + x.not_fired_error, x.fired_no_error + x.not_fired_no_error
            k = f"patient {tr_} {tau:g}"
            rowsB.append([tl, f"{tau:g}", reg(T, k + " TP", sp, x.fired_error, 0) and n_(x.fired_error), n_(x.fired_no_error),
                          n_(x.not_fired_error), n_(x.not_fired_no_error),
                          f"{reg(T, k + ' sens', sp, x.sensitivity, 1, 100)}% ({n_(x.fired_error)} of {n_(ne)}; "
                          f"{f(100 * (x.sensitivity - Z * x.sensitivity_mcse), 1)}% to {f(100 * (x.sensitivity + Z * x.sensitivity_mcse), 1)}%)",
                          f"{f(100 * x.false_positive_rate, 2)}% ({n_(x.fired_no_error)} of {n_(nn)})",
                          f"{f(100 * x.ppv, 1)}% ({f(100 * (x.ppv - Z * x.ppv_mcse), 1)}% to {f(100 * (x.ppv + Z * x.ppv_mcse), 1)}%)"])
    hdrB = ["Patient-level error", "Trigger threshold (mm)", "Fired, error", "Fired, no error", "Not fired, error",
            "Not fired, no error", "Sensitivity (count; 95% MCI)", "False-positive rate (count)", "PPV (95% MCI)"]
    im = sel(rd(si), axis="AP", tau_mm=5.0, measure="ppv")
    mlab = [("binomial, pairs independent", "Binomial, view pairs treated as independent"),
            ("'conservative'", "Binomial standard error multiplied by √3 (upper bound: three pairs per patient as one observation)"),
            ("patient-level linearization", "Patient-clustered linearization (reported)"),
            ("patient-level (cluster) bootstrap", "Patient-level bootstrap, percentile (10,000 resamples)")]
    rowsC = []
    for coh, clab in [("published", "Planned run"), ("new_pooled", "Replication")]:
        for key, lab in mlab:
            x = im[(im.cohort == coh) & im.method.str.startswith(key)]
            assert len(x) == 1, (coh, key)
            x = x.iloc[0]
            rowsC.append([clab, lab, f"{f(100 * x.value, 1)}% ({n_(x.numerator)} of {n_(x.denominator)})", f(100 * x.se, 2),
                          f"{reg(T, f'{coh} {key} lo', si, x.lo95, 1, 100)}% to {reg(T, f'{coh} {key} hi', si, x.hi95, 1, 100)}%"])
    hdrC = ["Sample", "Interval method", "PPV at 5 mm", "Standard error (percentage points)", "95% interval"]
    pv, pn = one(r, cohort="published", tau_mm=5.0, measure="ppv"), one(r, cohort="new_pooled", tau_mm=5.0, measure="ppv")
    ps = one(p, truth="any_pair:D0_published", tau_mm=5.0)
    title = ("**Supplementary Table S18. Contingency tables of the cross-view triggers at the level of the view pair "
             "and of the patient, and interval methods for the positive predictive value.** The 5 mm trigger fired "
             f"in {pc(one(c, cohort='published', tau_mm=5.0).fired_error, one(r, cohort='published', tau_mm=5.0, measure='sensitivity').denominator)} "
             f"of view pairs with an error above 2 mm and in {f(100 * ps.sensitivity, 1)}% of patients with such a "
             f"pair; its PPV was {f(100 * pv.value, 1)}% ({n_(pv.numerator)} of {n_(pv.denominator)} triggered pairs; "
             f"95% MCI {f(100 * pv.lo95_cluster, 1)}% to {f(100 * pv.hi95_cluster, 1)}%) in the planned run and "
             f"{f(100 * pn.value, 1)}% ({f(100 * pn.lo95_cluster, 1)}% to {f(100 * pn.hi95_cluster, 1)}%) in "
             "1,000,000 new patients.")
    foot = ("Planned experiment E4, base case with four views (three pairs of the anchor view with another view per "
            "patient), AP axis, three beats per view, beat-to-beat variation 15%; the planned sample was rebuilt "
            "from its stored seed (20260918, spawn key (4, 193)) and all stored values were reproduced; replication "
            "and interval methods are third amendment (post hoc; seed 20261005, spawn keys (74, 1, 0 to 9), "
            "bootstrap (74, 9)). A trigger fired when the other view mean exceeded the anchor-view mean by at least "
            "the threshold (one-sided). Pair-level error (panel A): systematic overestimation of the other view or "
            "underestimation of the anchor view by more than 2 mm. All pair-level rates come from the one table of "
            "counts shown, and their 95% MCI uses the patient-clustered standard error (linearization). Panel B: the "
            "unit is the patient (trigger fired in any pair; binomial MCSE); a reported value wrong by more than "
            "2 mm includes beat-to-beat, caliper and reader error. Panel C compares interval methods for the same "
            f"proportion. Sources: {rel(sc)} and {rel(sr_)} (panel A), {rel(sp)} (panel B), {rel(si)} (panel C); "
            f"generated by {SCRIPT}. AP, anteroposterior; {ABBR_MC}; PPV, positive predictive value.")
    return "\n\n".join([title, panel("A", "View-pair level"), md_table(hdrA, rowsA, ["l", "r"] + ["r"] * 7),
                        panel("B", "Patient level, planned run (n = 100,000)"), md_table(hdrB, rowsB, ["l", "r"] + ["r"] * 7),
                        panel("C", "Interval methods for the PPV of the 5 mm trigger"),
                        md_table(hdrC, rowsC, ["l", "l", "r", "r", "r"]), foot])


def tableS19():
    se_, so = TR / "triggers_error_definitions.csv", TR / "triggers_exam_offsets.csv"
    e = sel(rd(se_), cohort="published", axis="AP")
    defs = [("D0_published", "Planned definition: other view overestimated or anchor view underestimated by more than 2 mm"),
            ("D0a_other_over", "Other view overestimated by more than 2 mm"),
            ("D0b_anchor_under", "Anchor view underestimated by more than 2 mm"),
            ("D1_plus_other_under", "Planned definition, or other view underestimated by more than 2 mm"),
            ("D2_net_either_view", "Either view wrong by more than 2 mm in either direction, plane-related error only"),
            ("D3_total_either_view", "Either view wrong by more than 2 mm in either direction, with the machine-setting factor"),
            ("D5_shared_total", "Both views wrong by more than 2 mm in the same direction, with the machine-setting factor"),
            ("D5n_shared_net", "Both views wrong by more than 2 mm in the same direction, plane-related error only")]
    T, rows = "S19", []
    for key, lab in defs:
        x3, x5 = one(e, definition=key, tau_mm=3.0), one(e, definition=key, tau_mm=5.0)
        assert x5.n_pairs == 300000 and x5.n_patients == 100000
        rows.append([lab, f"{reg(T, key + ' error pairs', se_, x5.n_error_pairs, 0) and n_(x5.n_error_pairs)} ({pc(x5.n_error_pairs, x5.n_pairs)})",
                     f"{n_(x3.n_fired_error_pairs)} ({pc(x3.n_fired_error_pairs, x3.n_error_pairs)})",
                     f"{reg(T, key + ' fired error pairs 5mm', se_, x5.n_fired_error_pairs, 0) and n_(x5.n_fired_error_pairs)} ({pc(x5.n_fired_error_pairs, x5.n_error_pairs)})",
                     f"{n_(x5.n_patients_with_error)} ({pc(x5.n_patients_with_error, x5.n_patients)})",
                     f"{n_(x5.n_patients_fired_with_error)} ({pc(x5.n_patients_fired_with_error, x5.n_patients_with_error)})"])
    hdr = ["Definition of error", "View pairs with the error (of 300,000)", "Of these, 3 mm trigger fired",
           "Of these, 5 mm trigger fired", "Patients with the error in any pair (of 100,000)", "Of these, 5 mm trigger fired in any pair"]
    o = sel(rd(so), cohort="published", axis="AP", tau_mm=5.0)
    rowsB = []
    for kind, lab in [("total", "With the machine-setting factor"), ("net", "Plane-related error only")]:
        x = one(o, error_kind=kind)
        rowsB.append([lab, f"{reg(T, kind + ' all low', so, x.all_low, 0) and n_(x.all_low)} ({pc(x.all_low, x.n_patients, 2)})",
                      f"{n_(x.all_low_fired)} ({pc(x.all_low_fired, x.all_low)})",
                      f"{n_(x.all_high)} ({pc(x.all_high, x.n_patients, 2)})",
                      f"{n_(x.any_view_wrong)} ({pc(x.any_view_wrong, x.n_patients)})",
                      f"{n_(x.any_view_wrong_fired)} ({pc(x.any_view_wrong_fired, x.any_view_wrong)})"])
    hdrB = ["Error counted", "Patients with every view too low by more than 2 mm (of 100,000)", "Of these, 5 mm trigger fired",
            "Patients with every view too high by more than 2 mm", "Patients with any view wrong by more than 2 mm",
            "Of these, 5 mm trigger fired"]
    d1, d5, d5n = (one(e, definition=k, tau_mm=5.0) for k in ["D1_plus_other_under", "D5_shared_total", "D5n_shared_net"])
    ot = one(o, error_kind="total")
    title = ("**Supplementary Table S19. Trigger performance under wider definitions of view error and for errors "
             "shared by views.** When underestimation of the other view also counted as an error, "
             f"{pc(d1.n_error_pairs, d1.n_pairs)} of view pairs carried an error and the 5 mm trigger fired in "
             f"{pc(d1.n_fired_error_pairs, d1.n_error_pairs)} of them; both views were wrong in the same direction in "
             f"{pc(d5.n_error_pairs, d5.n_pairs)} of pairs ({pc(d5n.n_error_pairs, d5n.n_pairs)} without the "
             f"machine-setting factor), and every view was too low in {pc(ot.all_low, ot.n_patients)} of patients, in "
             f"{pc(ot.all_low_fired, ot.all_low)} of whom the trigger fired.")
    foot = ("Third amendment (post hoc) tabulation of the planned sample of E4 (base case with four views, AP axis; "
            "n = 100,000 simulated patients, 300,000 view pairs; stored seed 20260918, spawn key (4, 193)). The "
            "trigger is one-sided (other view above the anchor view), so it cannot signal a view that reads low and "
            "cannot detect an error shared by both views. Errors are systematic view errors in millimetres at the "
            "patient's span; the machine-setting factor is a multiplicative factor common to all views of an examination "
            "(log-scale SD 0.10, assumed), and shared errors arise mostly from it. Counts are exact; percentages "
            f"have the denominator stated in the column heading or the preceding column. Sources: {rel(se_)} "
            f"(panel A), {rel(so)} (panel B); generated by {SCRIPT}. AP, anteroposterior.")
    return "\n\n".join([title, panel("A", "View-pair and patient level by definition of error"), md_table(hdr, rows),
                        panel("B", "Errors shared by all views of a patient"), md_table(hdrB, rowsB), foot])


def tableS20():
    sg, sd_ = TR / "review_grid_wide.csv", TR / "review_dependence_summary.csv"
    g = rd(sg)
    g = g[(g.run == "new") & (g.axis == "AP") & (g.rule == "A4")]
    scen_lab = [("base", "Base case (4.8% / 20%, ±15% window)"),
                ("long_axis_8pct", "Low-underestimation scenario (2.4% / 8%, ±15% window)"),
                ("extreme_inaccurate", "Least accurate of the 12 scenarios (20% / 40%, no window)")]
    T = "S20"
    S, F = [0.0, 0.2, 0.5, 0.8, 1.0], [0.0, 0.1, 0.3, 0.5]
    out, mc = [], 0.0
    for sk, sl in scen_lab:
        x = g[(g.scenario == sk) & (g.overestimation == "base")]
        rows = []
        for s_ in S:
            cells = [f"{s_:g}"]
            for f_ in F:
                r = one(x, s_det=s_, f_rej=f_)
                cells.append(reg(T, f"{sk} s{s_:g} f{f_:g}", sg, r.rmse_mm_minus_largest_view_mean, 3))
                mc = max(mc, r.rmse_mm_minus_largest_view_mean_mcse)
            rows.append(cells)
        out.append((sl, md_table(["Probability of excluding an overestimated view"] +
                                 [f"Probability of excluding a valid view {f_:g}" for f_ in F], rows, ["r"] * 5)))
    d = rd(sd_)
    d = d[d.axis == "AP"]
    olab = {"base": "as in the base case", "off": "absent", "doubled": "doubled"}
    slab = dict(scen_lab + [("extreme_accurate", "Most accurate of the 12 scenarios (2.4% / 8%, no window)")])
    rowsD = []
    for _, r in d.iterrows():
        assert r.scenario in slab, r.scenario
        rowsD.append([slab[r.scenario], olab[r.overestimation], f(r.largest_view_mean_rmse_mm, 2),
                      mci(r.published_review_rmse_mm_difference, r.published_review_rmse_mm_difference_mcse, 3),
                      rng(r.grid_min_rmse_mm_difference, r.grid_max_rmse_mm_difference, 3),
                      f"{int(r.n_rmse_lower_than_largest_view_mean)} / {int(r.n_rmse_higher_than_largest_view_mean)}",
                      mci(r["rmse_difference_uninformative_0.5"], r["rmse_difference_uninformative_0.5_mcse"], 3)])
    hdrD = ["View accuracy (anchor / long axis)", "Overestimation", "RMSE, largest view mean (mm)",
            "Assumed reviewer (0.8, 0.1) minus largest view mean (mm, 95% MCI)", "Range over the 25 grid points (mm)",
            "Grid points with lower / higher RMSE", "Reviewer excluding half of all reviewed views (mm, 95% MCI)"]
    b = one(d, scenario="base", overestimation="base")
    w = one(d, scenario="extreme_inaccurate", overestimation="base")
    hh = sel(rd(FULL / "E1E3_e1_headline.csv"), **E1_BASE, u="base", axis="AP", estimand="T1")
    t1_gain = float(f(one(hh, estimator="A3").rmse, 2)) - float(f(one(hh, estimator="A4").rmse, 2))
    t1_gain = reg("S20", "Table 1 RMSE difference, A3 minus A4 (rounded values)", FULL / "E1E3_e1_headline.csv", t1_gain, 2)
    assert t1_gain == "0.10", t1_gain
    title = ("**Supplementary Table S20. Dependence of the largest view mean after review on assumed reviewer "
             f"performance.** In the base case review lowered RMSE by {f(-b.published_review_rmse_mm_difference, 2)} mm "
             f"in the sample of this analysis ({t1_gain} mm in the planned run, Table 1) "
             "at the assumed reviewer probabilities, approximately in proportion to the probability of excluding an "
             "overestimated view and with little dependence on the probability of excluding a valid view; in the "
             "least accurate "
             f"scenario review raised RMSE by {f(w.published_review_rmse_mm_difference, 2)} mm.")
    foot = ("Third amendment (post hoc). The reviewer probabilities have no empirical source; the planned analyses "
            "assumed 0.8 and 0.1. Entries of panels A to C are the RMSE of the largest view mean after review minus "
            "the RMSE of the largest view mean without review (mm; negative, review lowers RMSE), paired on the same "
            "simulated patients; three views, three beats per view, beat-to-beat variation 15%, overestimation as in "
            "the base case, AP axis; anchor / long-axis underestimation and window as stated for each panel; n = 100,000 simulated patients per condition; seed "
            "20261005, spawn key (74, 4), the same for every condition, so that true spans, view errors and the "
            "machine-setting factor are shared between conditions (beat, caliper and reader errors are shared only "
            f"between conditions with the same window setting). Paired MCSE at most {mc:.3f} mm. With both "
            "probabilities 0 the rule equals the largest view mean. Panel D: grid of 25 pairs of probabilities; a "
            "grid point counts as lower or higher when the 95% MCI of the paired difference excludes zero. "
            f"Sources: {rel(sg)} (panels A to C), {rel(sd_)} (panel D); generated by {SCRIPT}. AP, anteroposterior; "
            f"{ABBR_MC}; RMSE, root-mean-square error.")
    body = []
    for letter, (sl, tb) in zip("ABC", out):
        body += [panel(letter, sl), tb]
    return "\n\n".join([title] + body + [panel("D", "Summary by view accuracy and overestimation"),
                                         md_table(hdrD, rowsD, ["l", "l", "r", "r", "r", "r", "r"]), foot])


def tableS21():
    sa = TR / "axes_scenarios.csv"
    a = rd(sa)
    assert a.n_patients.eq(100000).all() and len(a) == 23
    T = "S21"
    var = {"coded": "concentration as in the base case", "high_variability": "more variable ratio",
           "low_variability": "less variable ratio", "high_correlation": "high correlation",
           "mid_correlation": "intermediate correlation", "low_correlation": "low correlation"}

    def lab(r):
        if r.scenario == "coded_base":
            return "Base case (mean SL/AP ratio 0.64)"
        if r.scenario == "coded_r053":
            return "Mean SL/AP ratio 0.53"
        if r.scenario == "circular_exact":
            return "Circular (SL equal to AP)"
        if r.scenario == "near_circular_beta_r090":
            return "Near-circular, ratio model, mean ratio 0.90"
        if r.scenario == "near_circular_lognormal":
            return "Near-circular, bivariate lognormal"
        fam = "Ratio model" if r.family == "beta" else "Bivariate lognormal"
        return f"{fam}, mean difference {r.target_mean_diff_mm:g} mm, {var[r.variability]}"
    order = ["coded_base", "coded_r053"] + [s for s in a.scenario if s.startswith("beta_diff")] + \
        [s for s in a.scenario if s.startswith("lognormal_diff")] + ["near_circular_beta_r090", "near_circular_lognormal", "circular_exact"]
    assert sorted(order) == sorted(a.scenario)
    rows = []
    for sc_ in order:
        r = one(a, scenario=sc_)
        rows.append([lab(r), em(r.mean_true_diff_mm, r.mean_true_diff_mcse, 2, 3), f(r.corr_mm, 2),
                     f"{reg(T, sc_ + ' true ge3', sa, r.true_ge3_k, 0) and n_(r.true_ge3_k)} ({pc(r.true_ge3_k, r.n_patients)})",
                     f"{reg(T, sc_ + ' measured ge3', sa, r.measured_anchor_ge3_k, 0) and n_(r.measured_anchor_ge3_k)} ({pc(r.measured_anchor_ge3_k, r.n_patients)})",
                     f"{n_(r.measured_largest_ge3_k)} ({pc(r.measured_largest_ge3_k, r.n_patients)})"])
    hdr = ["Joint distribution of true AP and SL span", "Mean true AP minus SL (mm, MCSE)", "Correlation of true AP and SL",
           "True AP and SL differ by ≥ 3 mm (of 100,000)", "Anchor-view means differ by ≥ 3 mm", "Largest view means differ by ≥ 3 mm"]
    grp = lambda pre: a[a.scenario.str.contains(pre)]
    r2, r6 = grp("diff2_"), grp("diff6_")
    nc = a[a.scenario.str.contains("circular")]
    base = one(a, scenario="coded_base")
    src_p = FULL / "E4_ellipticity_truth.csv"
    pl = one(rd(src_p), r_mean=0.64, tau=3.0, n=14400000)
    planned = reg(T, "planned run, true AP and SL differ by 3 mm or more", src_p, pl.value, 1, 100)
    assert planned == "62.4", planned
    r26 = a[a.target_mean_diff_mm.notna()]
    assert len(r26) == 18 and set(r26.target_mean_diff_mm) == {2.0, 3.9, 6.0}
    title = ("**Supplementary Table S21. Differences between the AP and SL spans under alternative joint "
             f"distributions.** True AP and SL spans differed by 3 mm or more in {pc(base.true_ge3_k, base.n_patients)} "
             f"of patients in the base case of this sample ({planned}% in the planned run, the value of the main "
             f"text) and in {reg(T, 'true ge3 min, mean difference 2 to 6 mm', sa, r26.true_ge3.min(), 0, 100)}% to "
             f"{reg(T, 'true ge3 max, mean difference 2 to 6 mm', sa, r26.true_ge3.max(), 0, 100)}% when the mean "
             f"difference between the axes was set to 2 to 6 mm ({f(100 * r2.true_ge3.min(), 0)}% to "
             f"{f(100 * r2.true_ge3.max(), 0)}% at 2 mm and {f(100 * r6.true_ge3.min(), 0)}% to "
             f"{f(100 * r6.true_ge3.max(), 0)}% at 6 mm); with circular or near-circular orifices the anchor-view means "
             f"still differed by 3 mm or more in {f(100 * nc.measured_anchor_ge3.min(), 0)}% to "
             f"{f(100 * nc.measured_anchor_ge3.max(), 0)}% of patients through measurement error.")
    foot = ("Third amendment (post hoc). The joint distributions other than the base case are assumed, not "
            "calibrated. Four views, three beats per view, beat-to-beat variation 15%; n = 100,000 simulated "
            "patients per distribution, all on the same patients and measurement errors (seed 20261005, spawn keys "
            "(74, 5, 0) and (74, 5, 1)); AP-axis results were identical in all 23 distributions. Ratio model: SL "
            "span = AP span multiplied by a beta-distributed ratio between 0.4 and 1, as in the base case. Binomial "
            f"MCSE of each percentage at most 0.16 percentage points. The planned run (experiment E4; 14,400,000 "
            f"simulated patients pooled over its conditions) gave {planned}% for the base-case distribution. Sources: "
            f"{rel(sa)} and, for the planned run, {rel(src_p)}; generated by {SCRIPT}. AP, "
            f"anteroposterior; {ABBR_MC}; SL, septolateral.")
    return "\n\n".join([title, md_table(hdr, rows), foot])



# ====================================================================== S22 to S24 (AI evaluation and reference sets)
AI = OS / "ai_reference"
REFLAB = {"single": "Single read", "mean2": "Mean of two reads", "adj_tol1": "Adjudicated, tolerance 1 mm",
          "adj_tol2": "Adjudicated, tolerance 2 mm", "adj_tol3": "Adjudicated, tolerance 3 mm"}


def tableS22():
    sr_, sc = AI / "ai_ranking_matched.csv", AI / "ai_reference_error_components.csv"
    r = sel(rd(sr_), cell_id="base_cv15")
    assert r.n_studies.eq(2000).all()
    T = "S22"
    fam = [("share_lam0.25", "Reproduces 25% of the image error"), ("share_lam0.5", "Reproduces 50%"),
           ("share_lam0.75", "Reproduces 75%"), ("share_lam1", "Reproduces 100%"),
           ("inherit", "Inherits the systematic error of a single read")]
    rowsA = []
    for mt, ml in [("eqmae", "Equal true MAE"), ("eqsd", "Equal SD of own error")]:
        for fk, fl in fam:
            x = one(r, family=fk, match=mt, noise="ind", ref="single")
            k = f"{fk} {mt}"
            rowsA.append([ml, fl, f"{reg(T, k + ' n better', sr_, x.n_candidate_better_mae, 0) and n_(x.n_candidate_better_mae)} ({pc(x.n_candidate_better_mae, x.n_studies)})",
                          f"{reg(T, k + ' diff mae', sr_, x.diff_mae, 2)} ({f(x.diff_mae - Z * x.diff_mae_mcse, 2)} to {f(x.diff_mae + Z * x.diff_mae_mcse, 2)})",
                          mci(x.true_mae_diff, x.true_mae_diff_mcse, 2), f(x.sigma_own_candidate, 2)])
    hdrA = ["Matching", "Candidate model", "Studies in which the candidate had the lower apparent MAE (of 2,000)",
            "Apparent MAE, candidate minus comparator (mm, 95% MCI)", "True MAE, candidate minus comparator (mm, 95% MCI)",
            "SD of the candidate's own error (mm)"]
    rowsB = []
    for ref, rl in REFLAB.items():
        cells = [rl]
        for mt, nz in [("eqsd", "ind"), ("eqmae", "ind"), ("eqsd", "crn")]:
            x = one(r, family="inherit", match=mt, noise=nz, ref=ref)
            cells.append(f"{reg(T, f'inherit {mt} {nz} {ref}', sr_, x.n_candidate_better_mae, 0) and n_(x.n_candidate_better_mae)} ({pc(x.n_candidate_better_mae, x.n_studies)})")
        rowsB.append(cells)
    hdrB = ["Reference design", "Equal SD of own error, independent own errors", "Equal true MAE, independent own errors",
            "Equal SD of own error, identical own errors"]
    c = sel(rd(sc), cell_id="base_cv15", ref="single")
    assert len(c) == 1
    c = c.iloc[0]
    sh = r[r.family.str.startswith("share") & (r.noise == "ind") & (r.ref == "single")]
    e1, e2 = sh[sh.match == "eqmae"], sh[sh.match == "eqsd"]
    title = ("**Supplementary Table S22. Ranking of AI models matched for true MAE or for the SD of their own error.** "
             "Against a single-read reference, a model reproducing 25% to 100% of the image error had a lower apparent "
             f"MAE than a comparator of equal true MAE in {n_(e1.n_candidate_better_mae.min())} to "
             f"{n_(e1.n_candidate_better_mae.max())} of 2,000 studies, and than a comparator with the same SD of own "
             f"error in {n_(e2.n_candidate_better_mae.min())} to {n_(e2.n_candidate_better_mae.max())}, although its "
             f"true MAE was then {f(e2.true_mae_diff.min(), 2)} to {f(e2.true_mae_diff.max(), 2)} mm higher.")
    foot = ("Third amendment (post hoc). Base case of the AI evaluation analysis (beat-to-beat variation 15%, "
            "overestimation present, AP axis, three views, every read by the largest view mean after review), "
            "evaluation studies of 200 patients; n = 2,000 simulated studies; seed 20261005, spawn key (75, 1, 0) "
            "(matching constants from 400,000 separate patients, spawn key (75, 2, 0)). Comparator: model with "
            "independent normal error of SD 2 mm about the true span. Candidates: true span plus the stated share of "
            "the image-level error common to all readers, or plus the systematic component of a single read "
            "(one fitted calibration line), plus own error. Equal true MAE: the SD of the candidate's own error was "
            "chosen so that its MAE against the true span equalled the comparator's; equal SD of own error: both "
            "models had own error of SD 2 mm. Own errors of candidate and comparator were independent except in the "
            "last column of panel B, which repeats the design of the second amendment (Supplementary Table S3, "
            "panel C). Differences are paired within study. Variance of the image-level error "
            f"{reg(T, 'var img', sc, c.var_img, 2) if 'var_img' in c.index else ''} mm² and of the single-read "
            f"reference error {reg(T, 'var ref', sc, c.var_ref, 2) if 'var_ref' in c.index else ''} mm². "
            f"Sources: {rel(sr_)}, {rel(sc)}; generated by {SCRIPT}. AI, artificial intelligence; AP, anteroposterior; "
            f"MAE, mean absolute error; {ABBR_MC}.")
    return "\n\n".join([title, panel("A", "Single-read reference, independent own errors"),
                        md_table(hdrA, rowsA, ["l", "l", "r", "r", "r", "r"]),
                        panel("B", "Model inheriting the systematic error of a single read: studies (of 2,000) in which it had the lower apparent MAE"),
                        md_table(hdrB, rowsB), foot])


def tableS23():
    sl, sm = AI / "loa_conditional_vs_marginal.csv", AI / "loa_multireader_designs.csv"
    l = rd(sl)
    T = "S23"
    hdr = ["Double-read cases", "SD for a fixed reader pair (mm, MCSE)", "SD between reader pairs (mm, MCSE)",
           "SD across reader pairs and case sets (mm, MCSE)", "Mean width of the nominal 95% CI (mm)",
           "CI covered the pair's own limit (of sets)", "CI covered the limit of a random pair (of sets)"]

    def block(scn):
        rows = []
        for _, x in l[l.scenario == scn].sort_values("n_cases").iterrows():
            k = f"{scn} {int(x.n_cases)}"
            rows.append([f"{int(x.n_cases)}", f"{reg(T, k + ' sd cond', sl, x.sd_conditional, 2)} ({f(x.sd_conditional_mcse, 3)})",
                         f"{reg(T, k + ' sd between', sl, x.sd_between_pairs, 2)} ({f(x.sd_between_pairs_mcse, 3)})",
                         f"{reg(T, k + ' sd marg', sl, x.sd_marginal, 2)} ({f(x.sd_marginal_mcse, 3)})", f(x.ci_width_mean, 2),
                         f"{reg(T, k + ' cov cond', sl, x.n_covered_conditional, 0) and n_(x.n_covered_conditional)} of {n_(x.n_studies)} ({pc(x.n_covered_conditional, x.n_studies)})",
                         f"{n_(x.n_covered_marginal)} of {n_(x.n_studies)} ({pc(x.n_covered_marginal, x.n_studies)})"])
        return rows
    m = rd(sm)
    dl = {"conventional_two_reader": "Two readers, usual limit from their differences",
          "crossed": "Variance components, every reader reads every case",
          "rotating": "Variance components, rotating reader pairs"}
    m = m.assign(_o=m.design.map({"conventional_two_reader": 0, "crossed": 1, "rotating": 2})).sort_values(["_o", "n_readers"])
    rowsC = []
    for _, x in m.iterrows():
        k = f"{x.design} {int(x.n_readers)}"
        rowsC.append([dl[x.design], f"{int(x.n_readers)}", f"{int(x.n_cases)}", f"{int(x.total_reads)}",
                      mci(x.loa_mean, x.loa_mean_mcse, 2), em(x.loa_sd, x.loa_sd_mcse, 2, 3),
                      f"{reg(T, k + ' rmse', sm, x.loa_rmse, 2)} ({f(x.loa_rmse_mcse, 3)})",
                      "not estimated" if pd.isna(x.sd_b_mean) else em(x.sd_b_mean, x.sd_b_sd, 2, 2)])
    hdrC = ["Design", "Readers", "Cases", "Reads", "Mean estimated upper limit (mm, 95% MCI)", "SD of the estimate (mm, MCSE)",
            "RMSE against the limit for a random pair (mm, MCSE)", "Estimated SD of reader offset (mm; SD across studies)"]
    a = l[l.scenario == "rb075"].set_index("n_cases")
    conv = one(m, design="conventional_two_reader")
    six = one(m, design="crossed", n_readers=6)
    title = ("**Supplementary Table S23. Precision of inter-reader limits of agreement for a fixed reader pair, across "
             "reader pairs and with more readers.** For a fixed pair the SD of the upper limit fell from "
             f"{f(a.loc[25, 'sd_conditional'], 2)} mm with 25 cases to {f(a.loc[200, 'sd_conditional'], 2)} mm with "
             f"200, whereas the SD between pairs was {f(a.sd_between_pairs.min(), 2)} to "
             f"{f(a.sd_between_pairs.max(), 2)} mm at every size; the nominal confidence interval covered the pair's "
             f"own limit in {pc(a.loc[200, 'n_covered_conditional'], a.loc[200, 'n_studies'], 0)} of sets at 200 "
             f"cases, and at 400 reads the RMSE against the limit for a random pair was {f(conv.loa_rmse, 2)} mm "
             f"with two readers and {f(six.loa_rmse, 2)} mm with six.")
    foot = ("Third amendment (post hoc). Two estimands: the limit of agreement of one fixed pair of readers "
            "(conditional on the pair), and the limit for a pair drawn at random from the reader population "
            "(marginal). Nested simulation: reader pairs with offsets drawn once (1,000 pairs with offset SD 0.75 mm; "
            "200 pairs with offset SD 0), each with 50 independent sets of double-read cases; base case, AP axis; "
            "seed 20261005, spawn keys (75, stream, .). SD across reader pairs and case sets is the quantity "
            "reported by the planned analysis (Supplementary Table S4, panel A). The nominal CI is the Bland-Altman "
            "interval; it undercovered because inter-reader differences were not normally distributed in this "
            "model, and no alternative interval was evaluated. The limit for a random pair used as the target "
            f"({f(l.marginal_target_upper_loa.iloc[0], 2)} mm in panels A and B, {f(m.target_marginal_upper_loa.iloc[0], 2)} "
            "mm in panel C) was itself estimated from one draw of 1,000 pairs and carries Monte Carlo error of about "
            "0.03 mm. Panel C: n = 2,000 simulated studies per design, about 400 reads each; variance-component "
            f"designs estimate the between-reader and within-reader variances by moments. Sources: {rel(sl)} (panels "
            f"A and B), {rel(sm)} (panel C); generated by {SCRIPT}. AP, anteroposterior; CI, confidence interval; "
            f"{ABBR_MC}; RMSE, root-mean-square error.")
    al = ["r"] * 7
    return "\n\n".join([title, panel("A", "Reader offset SD 0.75 mm (50,000 sets per size)"), md_table(hdr, block("rb075"), al),
                        panel("B", "No reader offset (10,000 sets per size)"), md_table(hdr, block("rb000"), al),
                        panel("C", "Designs with about 400 reads"), md_table(hdrC, rowsC, ["l"] + ["r"] * 7), foot])


def tableS24():
    ss, sb = AI / "sentinel_size_summary.csv", AI / "sentinel_base_cell.csv"
    z = rd(ss)
    tl = [("f_onesided", "F test on the variance, one-sided (planned)"),
          ("welch_mean", "Welch t test on the mean, two-sided (planned)"),
          ("bf_onesided", "Brown-Forsythe test, one-sided (second amendment)"),
          ("bf_twosided", "Brown-Forsythe test, two-sided (second amendment)"),
          ("perm_var_ratio_onesided", "Permutation test of the variance ratio, one-sided (third amendment)")]
    T = "S24"
    rowsA = []
    for st, sn in [("all", "all")] + [("sentinel_n", str(n)) for n in [10, 25, 50, 100]]:
        for tk, tn in tl:
            if st != "all" and tk in ("welch_mean", "bf_twosided"):
                continue
            x = z[(z.stratum == st) & (z.sentinel_n.astype(str) == sn) & (z.test == tk)]
            assert len(x) == 1, (st, sn, tk)
            x = x.iloc[0]
            k = f"{tk} {sn}"
            rowsA.append(["All four sizes" if st == "all" else sn, tn, f"{int(x.n_combinations)}",
                          f"{reg(T, k + ' min', ss, x.size_min, 3)} / {reg(T, k + ' median', ss, x.size_median, 3)} / {reg(T, k + ' max', ss, x.size_max, 3)}",
                          f"{f(x.size_pooled, 3)} ({n_(x.n_reject_pooled)} of {n_(x.n_studies_pooled)})",
                          reg(T, k + ' n above', ss, x.n_mci_above_level, 0), reg(T, k + ' n below', ss, x.n_mci_below_level, 0)])
    hdrA = ["Sentinel cases", "Test", "Combinations", "False-positive rate: lowest / median / highest",
            "Pooled false-positive rate (count)", "Combinations with 95% MCI entirely above 0.05",
            "Combinations with 95% MCI entirely below 0.05"]
    b = rd(sb)
    assert b.n_studies.eq(2000).all() and set(b.cell) == {22}
    rowsB = []
    for w in [0.0, 0.1, 0.2, 0.3, 0.5]:
        for tk, tn in [("bf_onesided", "Brown-Forsythe, one-sided"), ("perm_var_ratio_onesided", "Permutation, one-sided"),
                       ("f_onesided", "F test, one-sided")]:
            cells = [f"{w:g}" + (" (no drift)" if w == 0 else ""), tn]
            for n in [10, 25, 50, 100]:
                x = one(b, w_drift=w, test=tk, sentinel_n=n)
                cells.append(f"{reg(T, f'base {tk} w{w:g} n{n}', sb, x.n_reject, 0) and n_(x.n_reject)} ({f(x.rate, 3)})")
            rowsB.append(cells)
    hdrB = ["Drift towards the AI draft, w", "Test"] + [f"{n} sentinel cases" for n in [10, 25, 50, 100]]
    g = lambda tk: z[(z.stratum == "all") & (z.test == tk)].iloc[0]
    pb = one(b, w_drift=0.3, test="bf_onesided", sentinel_n=50)
    pp = one(b, w_drift=0.3, test="perm_var_ratio_onesided", sentinel_n=50)
    title = ("**Supplementary Table S24. False-positive rate and power of the sentinel tests.** The 95% MCI of the "
             f"false-positive rate of the F test lay entirely above 0.05 in {int(g('f_onesided').n_mci_above_level)} "
             "of 216 combinations; the one-sided Brown-Forsythe test was conservative (false-positive rate "
             f"{f(g('bf_onesided').size_min, 3)} to {f(g('bf_onesided').size_max, 3)}, entirely below 0.05 in "
             f"{int(g('bf_onesided').n_mci_below_level)}) and a permutation test was close to nominal (median "
             f"{f(g('perm_var_ratio_onesided').size_median, 3)}); with 50 sentinel cases and a drift of 0.3 they "
             f"rejected in {n_(pb.n_reject)} and {n_(pp.n_reject)} of 2,000 studies.")
    foot = ("Third amendment (post hoc), new seed 20261005, spawn key (75, stream, .); the planned and "
            "second-amendment estimates are in Supplementary Table S4 (176 of 216 combinations for the F test in "
            "the planned run). Design: within a production set of 500 cases read by one reader, a sentinel subset "
            "was read without AI assistance and the remainder with it; the two samples are different cases "
            "(unpaired). Endpoint: variance of read minus AI draft; one-sided alternative, smaller variance in "
            "assisted reads; null hypothesis, no drift (w = 0); level 0.05. Panel A: 216 combinations = 54 "
            "conditions (beat-to-beat variation 5%, 15% or 30%; reader offset SD 0, 0.75 or 2 mm; AI draft SD and "
            "bias as in Supplementary Table S1) by four sentinel sizes; n = 2,000 simulated studies per combination; "
            "permutation test with 999 permutations. The false-positive rate of the F test rose with the excess "
            "kurtosis of the differences. Panel B: base case (beat-to-beat variation 15%, reader offset SD 0.75 mm, "
            "AI draft bias 1 mm and SD 2 mm), number of 2,000 studies rejecting (proportion); binomial MCSE at most "
            f"{b[b.mcse.notna()].mcse.max():.3f}. The power of the F test is not at a controlled false-positive rate. "
            f"Sources: {rel(ss)} (panel A), {rel(sb)} (panel B); generated by {SCRIPT}. AI, artificial intelligence; "
            f"{ABBR_MC}.")
    return "\n\n".join([title, panel("A", "False-positive rate without drift"), md_table(hdrA, rowsA, ["l", "l", "r", "r", "r", "r", "r"]),
                        panel("B", "Rejections in the base case, by drift and number of sentinel cases"),
                        md_table(hdrB, rowsB, ["l", "l", "r", "r", "r", "r"]), foot])


# ====================================================================== S25 (calibration against Singh et al.)
CA = OS / "calibration"


def tableS25():
    src = CA / "calibration_table_s25.csv"
    if not src.exists():
        raise SystemExit("calibration_table_s25.csv is missing: build S25 from calibration_summary.csv, libmatch.csv, "
                         "viewrules_ranges.csv and singh_reproduction.csv instead")
    d = rd(src)
    T = "S25"
    part = lambda pre: d[d.part.str.startswith(pre)]
    A = part("A.")
    vw = [("4CH", "Four-chamber"), ("inflow", "Inflow-outflow"), ("mBC", "Modified bicaval")]
    itemsA = [("biplane below 3D average", "Biplane width below the average of 3D maximal and minimal diameters (reported)"),
              ("single plane below 3D maximum", "Primary-plane width below the 3D maximal diameter (derived)"),
              ("single plane below 3D minimum", "Primary-plane width below the 3D minimal diameter (derived)"),
              ("orthogonal plane below 3D maximum", "Orthogonal-plane width below the 3D maximal diameter (derived, no paired statistic)")]
    rowsA = []
    for key, lab in itemsA:
        x = A[A.item.str.startswith(key)]
        rowsA.append([lab] + [reg(T, f"A {key} {v}", src, one(x, condition=v).estimate, 1) for v, _ in vw])
    tv = A[A.item.str.startswith("three-view mean")].iloc[0]
    el = A[A.item.str.startswith("minor below major")].iloc[0]
    hdrA = ["Published comparison (%)"] + [v[1] for v in vw]
    B = part("B.")
    rowsB = []
    for sc_ in ["0.1", "0.25", "0.5"]:
        cells = [f"Biplane shortfall at long-axis scale {float(sc_):.2f} (%, MCSE)"]
        for ov in ["on", "off"]:
            x = one(B, condition=f"scale {sc_}; overestimation {ov}")
            cells.append(f"{reg(T, f'B shortfall {sc_} {ov}', src, x.estimate, 1)} ({f(x.mcse, 2)})")
        rowsB.append(cells)
    sc = B[B.item.str.startswith("long-axis scale reproducing")]
    for _, x in sc[sc.condition.str.endswith("overestimation on")].iterrows():
        y = one(sc, condition=x.condition.replace("overestimation on", "overestimation off"))
        tg = x.condition.split(";")[0].replace("target ", "").replace("mean_three_views", "three-view mean").replace("4CH", "four-chamber").replace("mBC", "modified bicaval")
        rowsB.append([f"Long-axis scale reproducing {tg} (MCSE)", f"{reg(T, 'B scale ' + tg, src, x.estimate, 3)} ({f(x.mcse, 4)})",
                      f"{f(y.estimate, 3)} ({f(y.mcse, 4)})"])
    hdrB = ["Quantity in the simulation model", "Overestimation present", "Overestimation absent"]
    C = part("C. ")
    itemsC = [("AP axis, inflow primary plane vs 3D maximum", "AP axis: inflow primary plane against the maximal diameter (rotation and off-centre placement)"),
              ("AP axis, inflow primary plane, rotation component", "of which rotation of the plane"),
              ("AP axis, inflow primary plane, off-centre component", "of which off-centre placement (plane on the axis)"),
              ("AP axis, 4CH orthogonal plane vs 3D maximum", "AP axis: four-chamber orthogonal plane against the maximal diameter"),
              ("AP axis, mBC orthogonal plane vs 3D maximum", "AP axis: modified bicaval orthogonal plane against the maximal diameter"),
              ("SL axis, 4CH primary plane vs 3D minimum", "SL axis: four-chamber primary plane against the minimal diameter"),
              ("SL axis, mBC primary plane vs 3D minimum", "SL axis: modified bicaval primary plane against the minimal diameter"),
              ("cross-plane maximum of the inflow view", "Larger of the two planes of the inflow view against the maximal diameter"),
              ("cross-plane maximum of the 4CH view", "Larger of the two planes of the four-chamber view against the maximal diameter")]
    rowsC = []
    for key, lab in itemsC:
        x = C[C.item.str.startswith(key)]
        assert len(x) == 2, (key, len(x))
        pr = x[x.condition.str.startswith("variant ind15")].iloc[0]
        rg = x[x.condition.str.startswith("range of means")].iloc[0]
        rowsC.append([lab, f"{reg(T, 'C ' + key, src, pr.estimate, 1)} ({f(pr.mcse, 2)})",
                      "not computed" if pd.isna(pr.lower) else rng(pr.lower, pr.upper, 1),
                      rng(rg.lower, rg.upper, 1)])
    hdrC = ["Mean fractional underestimation (%)", "Primary variant (MCSE)", "Target-sampling interval, primary variant",
            "Range of means over fitted variants"]
    C2 = part("C2")
    sds = [("SD 5.73 mm", "5.73 (lower half-width; main reading)"), ("SD 7.07 mm", "7.07 (half the full width)"),
           ("SD 8.41 mm", "8.41 (upper half-width)")]
    rowsD = []
    for it in C2.item.unique():
        lab = (it.replace("4CH", "four-chamber").replace("mBC", "modified bicaval").replace(" vs ", " against ")
               .replace(", reading 1 (rotation and offset)", " (rotation and off-centre placement)")
               .replace(", reading 1", " (rotation counted)").replace("SL axis, reading 2", "SL axis, plane on the axis")
               .replace("(reading 2)", "(plane on the axis)").replace("3D maximum", "the 3D maximal diameter")
               .replace("3D minimum", "the 3D minimal diameter").replace("3D average", "the 3D average")
               .replace("published estimand", "published comparison"))
        cells = [lab[0].upper() + lab[1:]]
        for sk, _ in sds:
            x = C2[(C2.item == it) & C2.condition.str.startswith(sk) & C2.condition.str.endswith("variant ind15")]
            assert len(x) == 1, (it, sk)
            cells.append(f"{reg(T, f'C2 {it[:30]} {sk}', src, x.iloc[0].estimate, 1)} ({f(x.iloc[0].mcse, 2)})")
        rowsD.append(cells)
    hdrD = ["Mean fractional underestimation (%, MCSE), primary variant"] + [f"SD {v[1].split(' ', 1)[0]} mm {v[1].split(' ', 1)[1]}" for v in sds]
    D = part("D.")
    conds = [("the 12 view-accuracy scenarios", "The 12 view-accuracy scenarios (Table 1)"),
             ("half-normal shape at calibrated means, the 28", "Main error distribution at calibrated means of 8.3% to 29.7%"),
             ("half-normal shape at the lowest", "Main error distribution at the lowest calibrated mean, 6.2%"),
             ("half-normal shape at calibrated means, all", "Main error distribution at all calibrated means, 6.2% to 29.7%"),
             ("patient-level geometric shape, rotation counted", "Geometric error drawn per patient, rotation counted as error"),
             ("patient-level geometric shape, off-centre", "Geometric error drawn per patient, off-centre placement only"),
             ("patient-level geometric shape, both readings", "Geometric error drawn per patient, both readings"),
             ("half-normal and patient-level geometric shapes at all", "All calibrated values, both error distributions"),
             ("constant rotation plus chord", "Constant rotation with chord-distributed offset (not a calibrated distribution)"),
             ("sensitivity: patient-level geometric shape, reading 1, offsets", "Sensitivity: rotation counted, offsets independent between views"),
             ("sensitivity: patient-level geometric shape, reading 2, offsets", "Sensitivity: off-centre only, offsets independent between views"),
             ("sensitivity: patient-level geometric shape, reading 1, ellipticity", "Sensitivity: rotation counted, ellipticity linked to true AP span"),
             ("SD of the 3D maximal diameter 7.07 mm: half-normal shape", "SD of the 3D maximal diameter 7.07 mm: main error distribution"),
             ("SD of the 3D maximal diameter 7.07 mm: half-normal and", "SD of the 3D maximal diameter 7.07 mm: both error distributions")]
    rl = [("anchor-view mean", "A1"), ("mean across views", "A2"), ("largest view mean", "A3"), ("largest view mean after review", "A4")]
    rowsE, mcE = [], 0.0
    for key, lab in conds:
        cells = [lab]
        nsc = None
        for item, code in rl:
            x = D[(D.item == item + ", bias") & D.condition.str.startswith(key)]
            assert len(x) == 1, (key, item, len(x))
            x = x.iloc[0]
            nsc = x.n.split(" ")[0]
            flag = "" if str(x.bracketed_by_12_scenarios) in ("True", "reference") else " †"
            cells.append(f"{reg(T, f'E {key[:40]} {code} lo', src, x.lower, 2)} to {reg(T, f'E {key[:40]} {code} hi', src, x.upper, 2)}{flag}")
            mcE = max(mcE, x.mcse)
        cells.insert(1, nsc)
        rowsE.append(cells)
    hdrE = ["Setting", "Scenarios", "Anchor-view mean", "Mean across views", "Largest view mean", "Largest view mean after review"]
    gE = lambda key, item: D[(D.item == item) & D.condition.str.startswith(key)].iloc[0]
    b25 = one(B, condition="scale 0.25; overestimation on")
    c1 = C[C.item.str.startswith("AP axis, inflow primary plane vs 3D maximum") & C.condition.str.startswith("variant ind15")].iloc[0]
    c2 = C[C.item.str.startswith("AP axis, inflow primary plane, off-centre") & C.condition.str.startswith("range")].iloc[0]
    h28, hall = gE(conds[1][0], "largest view mean, bias"), gE(conds[7][0], "largest view mean, bias")
    h36 = gE(conds[3][0], "largest view mean, bias")
    assert h36.n.split(" ")[0] == "36" and hall.n.split(" ")[0] == "60"
    ref = gE(conds[0][0], "largest view mean, bias")
    bi = A[A.item.str.startswith("biplane below 3D average")].estimate
    # follow-up settings of 6 October 2026: which scenarios make up the 36, and the SL axis where rotation counted
    src_f = CA / "followup_viewrules_metrics.csv"
    fm = rd(src_f)
    fm = fm[(fm.replicate == 0) & (fm.rule == "A3") & (fm.metric == "bias_mm") & fm.variant.isin(["ind15", "mid15", "ind00"])]
    hn00 = fm[(fm["shape"].isin(["r1_halfnormal", "r2_halfnormal"])) & (fm.variant == "ind00") & (fm.axis == "AP")]
    assert len(hn00) == 8 and int(h28.n.split(" ")[0]) + len(hn00) == 36
    m00 = sorted(hn00.groupby("shape").mean_u_long.first())
    add00 = (reg(T, "follow-up half-normal mean, off-centre only", src_f, m00[0], 1, 100),
             reg(T, "follow-up half-normal mean, rotation counted", src_f, m00[1], 1, 100))
    assert add00 == ("6.2", "29.1"), add00
    sl = fm[(fm.axis == "SL") & fm["shape"].isin(["r1_halfnormal", "r1_patient"])]
    assert len(sl) == 24 and sl.n.eq(100000).all()
    sl_h, sl_p = sl[sl["shape"] == "r1_halfnormal"], sl[sl["shape"] == "r1_patient"]
    sl_txt = (f"{reg(T, 'SL bias min, rotation counted, main distribution', src_f, sl_h.value.min(), 2)} to "
              f"{reg(T, 'SL bias max, rotation counted, main distribution', src_f, sl_h.value.max(), 2)} mm with the "
              "main error distribution and "
              f"{reg(T, 'SL bias min, rotation counted, per patient', src_f, sl_p.value.min(), 2)} to "
              f"{reg(T, 'SL bias max, rotation counted, per patient', src_f, sl_p.value.max(), 2)} mm with the "
              "geometric error drawn per patient")
    sl_all = (reg(T, "SL bias min, rotation counted", src_f, sl.value.min(), 2), reg(T, "SL bias max, rotation counted", src_f, sl.value.max(), 2))
    assert sl_all == ("0.73", "1.95"), sl_all
    sl_u = rng(100 * sl.mean_u_long.min(), 100 * sl.mean_u_long.max(), 1)
    title = ("**Supplementary Table S25. Calibration of long-axis underestimation against Singh et al.** On the "
             f"published comparison the base case gave a biplane shortfall of {f(b25.estimate, 1)}% against a "
             f"published {f(bi.min(), 1)}% to {f(bi.max(), 1)}%; an elliptical-orifice model implied a mean long-axis "
             f"underestimation of {f(c1.estimate, 1)}% if plane rotation counts as error and {f(c2.lower, 1)}% to "
             f"{f(c2.upper, 1)}% for off-centre placement alone; at calibrated values the bias of the largest view "
             f"mean ranged from {f(h36.lower, 2)} to {f(h36.upper, 2)} mm with the main error distribution (36 "
             f"scenarios) and from {f(hall.lower, 2)} to {f(hall.upper, 2)} mm over all 60 calibrated scenarios, against "
             f"{f(ref.lower, 2)} to {f(ref.upper, 2)} mm in the 12 scenarios.")
    foot = ("Third amendment (post hoc). The source [@singh2026] measured vena contracta width in 30 anaesthetized "
            "surgical patients in three mid-oesophageal views and compared biplane widths with the average of "
            "three-dimensional (3D) maximal and minimal diameters; it reported means with 95% confidence intervals, "
            "contains no transgastric view, and its quantity is not the jet span, so the calibration is geometric "
            "and conditional and the accuracy of the anchor view remains an assumption. Panel A: percentages "
            f"recomputed from the published means (three-view mean of the biplane comparison {f(tv.estimate, 1)}%; "
            f"minimal below maximal diameter, the difference expected from ellipticity alone, {f(el.estimate, 1)}%); "
            "comparisons of a plane near one axis with the diameter on the other axis are not underestimation on "
            "an axis. Panel B: biplane width of a long-axis view against the mean of the true AP and SL spans in the "
            "simulation model, n = 100,000 simulated patients. Panel C: single elliptical orifice fitted to the "
            "published means (primary variant: plane offsets independent between planes, plane-angle SD 15°; other "
            "variants differ in offset rule, angle spread and cap), n = 100,000; the target-sampling interval is the "
            "2.5th to 97.5th percentile over 500 redraws of the published means and is not a Monte Carlo interval. "
            "Panel D: the published confidence interval of the 3D maximal diameter (14.68 to 19.96 about 16.82 mm) is "
            "asymmetric, and the SD derived from it depends on the reading. Panel E: range of bias (mm) across "
            "scenarios of 100,000 simulated patients each (four levels of anchor underestimation; three views, three "
            "beats per view, no window; all scenarios on the same random numbers); † range not contained in the "
            f"range of the 12 view-accuracy scenarios; MCSE of each bias at most {mcE:.3f} mm. The 36 scenarios with "
            "the main error distribution are the 28 run on 5 October 2026 and eight added on 6 October 2026 (four "
            f"at a mean of {add00[0]}% and four at {add00[1]}%). On the SL axis, in the 24 scenarios in which "
            f"rotation of the plane counted as error (mean underestimation {f(100 * sl.mean_u_long.min(), 1)}% to {f(100 * sl.mean_u_long.max(), 1)}%), the bias of the largest view "
            f"mean was {sl_all[0]} to {sl_all[1]} mm ({sl_txt}; MCSE at most {sl.mcse.max():.3f} mm). Seed 20261005, "
            f"spawn key (76, family, index); stored cells with seed 20260920. Sources: {rel(src)} (assembled by "
            f"code/21_osrev_calibration_analyse.py from the csv files named in its source column) and {rel(src_f)} "
            "(composition of the 36 scenarios; SL axis); generated by "
            f"{SCRIPT}. AP, anteroposterior; {ABBR_MC}; SL, septolateral.")
    return "\n\n".join([title, panel("A", "Published comparisons, recomputed from reported means (n = 30)"), md_table(hdrA, rowsA),
                        panel("B", "Corresponding quantity in the simulation model"), md_table(hdrB, rowsB),
                        panel("C", "Elliptical-orifice calibration"), md_table(hdrC, rowsC),
                        panel("D", "Sensitivity to the reading of the published interval of the 3D maximal diameter"), md_table(hdrD, rowsD),
                        panel("E", "Bias of the view rules at calibrated values, AP axis (mm)"), md_table(hdrE, rowsE, ["l"] + ["r"] * 5), foot])



# ====================================================================== S8, S9, S26, S27 (documentation tables)
def _passed(jobid):
    """Number of tests reported as passed in the log of a test job (logs/duomax-pytest-<job>.out)."""
    p = ROOT / "logs" / f"duomax-pytest-{jobid}.out"
    assert p.exists(), p
    txt = p.read_text()
    mm = re.search(r"(\d+) passed", txt)
    assert mm and "failed" not in txt and "error" not in txt.lower() and "exit status: 0" in txt, p
    return int(mm.group(1))


def _sacct():
    j = rd(DERIVED / "slurm_jobs_sacct_2026-10-06.csv", sep="|")
    return j.set_index("JobID")


def tableS8():
    """Chronology. Rows and evidence classes follow notes/scratch/2026-10-05-evidence-chronology.md, section 2
    (items 2, 4, 6, 7, 10, 12, 15, 17, 19 to 23, 25); job times are read from the stored sacct listing, which holds
    the 17 jobs of Supplementary Methods S6."""
    j = _sacct()
    assert len(j) == 17, len(j)
    AF_, CA_, PA_ = "authors' files only", "cluster accounting", "public archive timestamp"
    used = []

    def job(i, lab):
        r = j.loc[i]
        d, t0 = r.Start.split("T")
        used.append(i)
        return [f"{d} {t0} to {r.End.split('T')[1]}",
                f"{lab} (job {i}; {int(r.CPUTimeRAW) / 3600:.1f} CPU-hours; {r.State.split(' ')[0].lower()})",
                "cluster accounting record", CA_]
    rows = [
        ["2026-09-18 13:56", "Initial protocol written (estimands, error model, rules, six experiments, number of "
         "simulated patients)", "local session record of the file write; file unchanged until the amendments were appended", AF_],
        ["2026-09-18 14:19 to 14:22", "Timing pilot on the login node with provisional parameters", "run manifests", AF_],
        ["2026-09-18 14:34", "Parameter values fixed; low-underestimation scenario implemented in the configuration "
         "(first amendment, implementation)", "file time; checksum recorded by the compute node at 14:40 and equal to "
         "the archived configuration", AF_ + "; content fixed by the public archive of 2026-09-19"],
        job(5223287, "Full run of the six planned experiments"),
        ["2026-09-18 14:50 to 14:56", "Analysis scripts of the planned experiments written and first run", "file times", AF_],
        job(5223386, "Exploratory sensitivity run (post hoc)"),
        ["2026-09-18 15:34", "Texts of the first and second amendments appended to the protocol", "local session record", AF_],
        job(5223669, "Second-amendment run"),
        ["2026-09-18 16:46", "Correction note: the text of the first amendment post-dated the full run", "local session record", AF_],
        ["2026-09-18 17:51", "First commit of the code and results to version control (commit e650519, version 0.3.1)",
         "commit date in the repository history (17:51:57 CEST), which is set by the committing computer", AF_],
        ["2026-09-18 17:53", "Code and results archived, version 0.3.1", "repository record creation time (15:53:42 UTC)", PA_],
        ["2026-09-19 19:18", "Commit 57371d3 (version 0.4.0; presentation changes, results unchanged)",
         "commit date in the repository history (19:18:30 CEST); push recorded by the public repository at 19:18:48 "
         "CEST (17:18:48 UTC)", AF_ + " for the commit date; " + PA_ + " for the push"],
        ["2026-09-19 19:19", "Archive version 0.4.0 (presentation changes; results unchanged)",
         "repository record creation time (17:19:29 UTC); checksum equal to the local archive", PA_],
        ["2026-10-05 17:55", "Third amendment written, before its analyses were coded or run",
         "file time recorded on 5 October 2026 in the authors' chronology note; the file was extended on 6 October "
         "2026 with the deviation note", AF_],
        job(5301518, "Rerun of the second implementations; paired replicates of the window effect"),
        job(5301599, "Third amendment, view rules"),
        job(5301601, "Third amendment, cut-offs"),
        job(5301653, "Third amendment, beat rules"),
        job(5301795, "Third amendment, triggers and review"),
        job(5301846, "Third amendment, calibration; superseded by job 5301947, no output kept"),
        job(5301894, "Third amendment, AI evaluation and reference sets; stopped early, no output kept"),
        job(5301941, "Third amendment, AI evaluation and reference sets"),
        job(5301947, "Third amendment, calibration"),
        job(5306852, f"Automated tests ({_passed(5306852)} tests, all passed)"),
        job(5306853, "Patient-clustered standard errors for Figure 4"),
        job(5306860, "Third amendment, calibration, follow-up settings; the analysis step stopped, no output kept"),
        job(5306863, "Third amendment, calibration, follow-up settings"),
        job(5306870, f"Automated tests ({_passed(5306870)} tests, all passed)"),
    ]
    assert sorted(used) == sorted(j.index), "every job of the accounting listing has one row"
    assert [r[0][:16] for r in rows] == sorted(r[0][:16] for r in rows), "chronological order"
    hours = lambda ids: sum(int(j.loc[i].CPUTimeRAW) for i in ids) / 3600
    h_sep, h_oct = hours([i for i in j.index if i < 5300000]), hours([i for i in j.index if i > 5300000])
    hdr = ["Date and time (CEST)", "Event", "Evidence", "Evidence class"]
    title = ("**Supplementary Table S8. Chronology of the protocol, its amendments and the simulation runs, with the "
             "class of evidence for each date.** Job times are held in the cluster accounting records and archive "
             "times in a public repository, whereas the timing of the protocol and amendment texts rests on the "
             "authors' files only.")
    foot = ("Evidence classes: cluster accounting, the scheduler's accounting database of the computing centre, which "
            "records when a job was submitted, started and ended and not what it ran; public archive timestamp, "
            "the record creation time, push time and file checksum held by the public repository; authors' files "
            "only, file times, commit dates, run manifests and local session records, which are internally "
            "consistent and can be altered. The protocol was not registered and its file is not published. The "
            "table lists all 17 batch jobs of Supplementary Methods S6, including two that failed and one that was "
            "superseded. CPU-hours are accounted CPU-hours as in Supplementary Methods S6: the elapsed time "
            "multiplied by the 256 hardware threads of the node (128 physical cores); in physical core-hours they "
            f"are half as large. The three jobs of 18 September 2026 used {h_sep:.1f} CPU-hours and the 14 jobs of 5 "
            f"and 6 October 2026 {h_oct:.1f} CPU-hours. The audit of the calibration follow-up settings on 6 October "
            "2026 ran on the login node and has no job (Supplementary Table S26). Not a Monte Carlo table. Sources: "
            f"{rel(DERIVED / 'slurm_jobs_sacct_2026-10-06.csv')} (accounting listing retrieved on 6 October 2026) "
            f"and the authors' chronology note; generated by {SCRIPT}. CEST, Central European Summer Time.")
    return "\n\n".join([title, md_table(hdr, rows, ["l"] * 4), foot])


def tableS9():
    """Transfer of indirect sources: notes/scratch/2026-10-05-evidence-source-variability.md, section 5, with two
    rows for Singh et al. from notes/scratch/2026-10-05-evidence-source-singh.md, section 4.4."""
    sr = rd(OS / "calibration" / "singh_reproduction.csv")
    bp = sr[sr.statement.str.startswith("biplane below 3D average (Table 3")].percent
    IND, ASS = "informed by an indirect source", "assumed"
    rows = [
        ["[@wong1987]: 50 patients, five consecutive beats, one observer", "Colour jet area; aortic, mitral and tricuspid jets pooled",
         "CV 14% to 22%", "Beat-to-beat variation of true span (range only; not used for the base case)",
         "Length proportional to the square root of area, giving about 7% to 11%; jet area behaves as the span",
         "Beat-to-beat", IND + " (range only)"],
        ["[@moraldo2013]: 11 patients, sinus rhythm, one sonographer", "PISA distance; mitral; transthoracic",
         "Beat-to-beat CV 15.5%; in vitro frame-to-frame floor 9%", "Beat-to-beat variation of true span, base case 15%",
         "Equal relative variability of PISA distance and jet span; mitral as tricuspid; transthoracic as "
         "transoesophageal. The reported CV includes measurement variability, and caliper error is added separately",
         "Beat-to-beat", ASS + ", upper range of an indirect source"],
        ["[@hauptmann2026]: 50 patients, two blinded core-laboratory observers", "Vena contracta width; tricuspid; transthoracic",
         "Inter-observer 95% limits −1.78 to 2.04 mm", "Caliper SD per beat, 1.0 mm",
         "Per-reading SD = width of the limits / (3.92 × √2) = 0.69 mm; transthoracic as transoesophageal; width as "
         "span; error independent of jet size", "Within-reader measurement error per beat", IND + " (derived)"],
        ["[@singh2026]: 10 of 30 patients read twice", "Vena contracta width; tricuspid; transoesophageal biplane",
         "Intra- and inter-observer ICC with between-patient SD from 30 patients", "SD of reader offset, 0.75 mm; caliper SD (about 1.2 mm)",
         "Reader-offset variance = inter-observer minus intra-observer error variance; ICC form assumed; width as span",
         "Between-reader offset", IND + " (derived)"],
        ["[@song2011]: 52 patients, sinus rhythm", "Vena contracta width on two axes; functional tricuspid regurgitation; "
         "three-dimensional transthoracic colour", "AP minus SL 3.9 mm (SD 3.7 mm)", "Mean SL/AP ratio, 0.64",
         "Same case mix as the simulation; ratio independent of size; axis widths as jet spans; only the mean difference matched",
         "Between-patient geometry (not an error term)", IND + " (one moment matched)"],
        ["[@pascalxtr2020]: 44 matched patients, single centre", "Vena contracta width; tricuspid; transoesophageal, view not stated",
         "Median 9.5 mm (interquartile range 7.2 to 12.3)", "Median true AP span 10 mm; log-SD 0.40",
         "Lognormal shape; width distribution as the distribution of true AP span; observed spread is true spread",
         "Between-patient case mix", IND],
        ["[@fan1994]", "Colour jet area under varied machine settings", "Not transferred as a value",
         "SD of the machine-setting factor, 0.10 on the log scale", "Area-based dependence on settings applies to a linear span",
         "Between-examination (shared by all views)", IND],
        ["[@singh2026]: 30 anaesthetized surgical patients, mid-oesophageal views", "Vena contracta width; biplane against the average of "
         "three-dimensional maximal and minimal diameters", f"Biplane widths {f(bp.min(), 0)}% to {f(bp.max(), 0)}% below the three-dimensional average",
         "Long-axis underestimation (20% in the base case)", "None adopted: different quantity and comparator, no transgastric "
         "view; examined post hoc in Supplementary Table S25", "View-level systematic error", ASS],
        ["[@mascherbauer2005]: in vitro, two systems", "Colour vena contracta diameter against the anatomical orifice",
         "Overestimation of 45% to 60% against the orifice", "None quantitatively; cited beside the overestimation term",
         "None valid: colour blooming is present in every measurement and is part of the colour-jet span", "View-level systematic error", ASS],
    ]
    assert f(bp.min(), 0) == "19" and f(bp.max(), 0) == "25"
    hdr = ["Source", "Source quantity", "Reported value", "Parameter informed", "Assumption for transfer", "Variance component", "Status"]
    title = ("**Supplementary Table S9. Indirect sources of the parameters and the assumptions needed to transfer them "
             "to the colour-Doppler jet span.** No source measured the jet span, so none of these transfers is a "
             "calibration of it.")
    foot = ("Status: informed by an indirect source, value or range taken from a related quantity through the stated "
            "assumption; assumed, no source constrains the value or the cited source supports plausibility only. "
            "Derived, computed by us from published summary statistics. Full texts were read for [@moraldo2013], "
            "[@pascalxtr2020], [@hauptmann2026] and [@singh2026]; abstracts only for [@wong1987], [@song2011] and "
            "[@mascherbauer2005]. Parameters without any source are listed as assumed in Supplementary Table S1. Not "
            f"a Monte Carlo table. The percentages of [@singh2026] are read from "
            f"results/2026-10-05_osrev/calibration/singh_reproduction.csv; generated by {SCRIPT}. AP, anteroposterior; "
            "CV, coefficient of variation; ICC, intraclass correlation; PISA, proximal isovelocity surface area; SL, septolateral.")
    return "\n\n".join([title, md_table(hdr, rows, ["l"] * 7), foot])


def tableS26():
    """Second implementations (results/2026-10-05_osrev/verification) and audits of the third-amendment packages
    (section '## Audit (independent)' of notes/scratch/2026-10-05-findings-*.md)."""
    VF = OS / "verification"
    s1, s2 = VF / "verif_e1_indep_vs_library.csv", VF / "verif_amend2_indep_vs_stored.csv"
    e1, a2 = rd(s1), rd(s2)
    T = "S26"
    z1 = e1.z.abs()
    a2z = a2[a2.z.notna() & np.isfinite(a2.z)]
    e5 = a2z[(a2z.part == "E5b") & ~a2z.quantity.str.contains("proportion|studies in which", case=False)]
    e6 = a2z[a2z.part != "E5b"]
    e6_all = a2[a2.part != "E5b"]
    rowsA = [
        ["Core model and view rules (anchor-view mean, largest view mean, largest view mean after review)", "separate AI coding agent",
         "none (protocol text and configuration notes only; self-reported)", "provisional values of the pilot, not the final ones",
         f"{reg(T, 'E1 comparisons', s1, len(e1), 0)} (6 conditions; bias and RMSE; 200 distinct values)",
         reg(T, "E1 max z", s1, z1.max(), 2), f"3; {int((z1 < 3).sum())} within 3, {int((z1 < 2).sum())} within 2", "none"],
        ["Beat window (one view, base case)", "AI coding agent that audited the beat-window analysis", "none (no library import)",
         "base case", "5 (20,000 patients; exhaustive search over beat subsets)", "about 1.9", "3; all within 2", "none"],
        ["Second-amendment analyses: AI model reproducing image error", "separate AI coding agent",
         "imported the library's data-generating and rule functions; did not read the second-amendment code (self-reported)",
         "second-amendment configuration", f"{reg(T, 'E5b comparisons', s2, len(e5), 0)} (plus 4 proportions at or near 1)",
         reg(T, "E5b max z", s2, e5.z.abs().max(), 2), f"3; {int((e5.z.abs() < 2).sum())} within 2", "none"],
        ["Second-amendment analyses: Brown-Forsythe test", "same agent", "as above", "second-amendment configuration",
         f"{reg(T, 'E6b comparisons', s2, len(e6_all), 0)} rejection rates; *P* values against a reference library to 1.0 × 10⁻¹⁵",
         reg(T, "E6b max z", s2, e6.z.abs().max(), 2),
         f"3; {int((e6.z.abs() < 2).sum())} within 2" + (f", {len(e6_all) - len(e6)} with both rates equal and no standard error" if len(e6_all) > len(e6) else ""), "none"],
    ]
    hdrA = ["Part reimplemented", "Author", "Access to library code", "Parameters", "Comparisons", "Largest standardized difference",
            "Tolerance and result", "Discrepancies leading to a code change"]
    pc_ = rd(DERIVED / "pytest_collect_2026-10-06.csv", header=None, names=["file", "n"]).set_index("file").n
    old = int(pc_[["tests/test_amend2.py", "tests/test_edge_cases.py", "tests/test_known_answers.py", "tests/test_regression.py"]].sum())
    assert old == 58
    new = {k.replace("tests/test_osrev_", "").replace(".py", ""): int(v) for k, v in pc_.items() if "osrev" in k}
    n_early, n_all = _passed(5306852), _passed(5306870)
    assert n_all == old + sum(new.values()) == 211 and n_early == 199, (n_early, n_all)
    # audit of the calibration follow-up of 6 October 2026: section '## Audit of the follow-up (independent)' of
    # notes/scratch/2026-10-05-findings-calibration.md and notes/scratch/2026-10-06-audit-calibration-followup.out
    fu = (ROOT / "notes" / "scratch" / "2026-10-05-findings-calibration.md").read_text()
    fu = fu[fu.index("## Audit of the follow-up (independent)"):]
    for s_ in ["43 passed (42 and 1)", "384 of 384 within 3 combined standard errors, largest |z| 1.97",
               "No error was found in the follow-up code or in any stored follow-up number",
               "16 values, all equal to the stored csv to its 10 digits", "spawn key (977, k)", "0.730 mm"]:
        assert s_ in fu, f"audit record does not contain {s_!r}"
    aud = [
        ["Beat rules (S8.1)", f"{new['beats']}", "1,646 of 1,646", "187; none beyond 3 MCSE (11 beyond 2)", "no error; four suggested sentences reworded"],
        ["View rules (S8.2)", f"{new['views']}", "572 of 572 (14 to rounding)", "111; 110 within 3 MCSE (one at 3.58, not confirmed on eight replicates)",
         "no error; within-view variability arm is mostly beat-to-beat variation; percentile MCSE approximate"],
        ["Cut-offs (S8.3)", f"{new['thresholds']}", "281 of 281 (1 to rounding)", "846; 1 beyond 3 MCSE at 1,000,000 patients per scenario (11 at 100,000, in three correlated clusters)",
         "no error; claims restricted to the cut-off and scenario they hold for"],
        ["Triggers and review (S8.4)", f"{new['triggers']}", "108 of 108", "96; all within 3 MCSE (largest 2.28)",
         "no error; shared errors attributed to the machine-setting factor; dependence on reviewer probabilities limited to the base case"],
        ["AI evaluation and reference sets (S8.5)", f"{new['ai_reference']}", "330 of 330 (6 to rounding)",
         "151; 150 within 3 MCSE (one explained by the single fitted calibration line of the inheriting model)",
         "no error; sentinel calibration was run as an unpaired permutation test"],
        ["Calibration (S8.6), runs of 5 October 2026", f"{new['calibration']} (including follow-up tests)", "all reported values; 78 of 78 source values found in the article",
         "155; 154 within 3 MCSE (one at 3.04, agreeing to 3 × 10⁻⁵ on the same sample)",
         "no error; one scenario was not a calibrated distribution and was replaced by the follow-up settings of 6 October 2026"],
        ["Calibration (S8.6), follow-up settings of 6 October 2026", f"{new['calibration'] + int(pc_['tests/test_regression.py'])} (calibration tests and the regression fixture)",
         "all reported values of the follow-up; 16 of 16 values of a rerun of the package equal to the stored values to 10 digits",
         "384 bias values over 48 scenarios; all within 3 standard errors (largest 1.97)",
         "no error in the code or in a stored value; the range on the SL axis was taken over all variants, including the "
         "one without spread of the plane angle (0.73 to 1.95 mm), and the calibration was described as conditional on "
         "whether rotation of the plane counts as error"],
    ]
    assert new["calibration"] + int(pc_["tests/test_regression.py"]) == 43
    hdrC = ["Package (Supplementary Section)", "Automated tests", "Reported values recomputed from stored files",
            "Comparisons of a separately written simulator", "Outcome"]
    title = ("**Supplementary Table S26. Second implementations and separate audits used for software verification.** "
             "No discrepancy led to a change of the simulation code; the second implementations and audits were "
             "written by AI coding agents and are software verification, not independent scientific review.")
    foot = ("Panel A: standardized difference = (second implementation minus library) / root sum of squares of the two "
            "MCSE; the comparisons within a condition share simulated patients and are not independent confirmations. "
            "Not covered by a second implementation: the mean across views, the median, index-beat and "
            "offset-corrected rules, atrial fibrillation, retrospective data, and the planned experiments E3 to E6. "
            "What each agent could read is self-reported and cannot be verified from the files. All three were rerun "
            "on 5 October 2026 (job 5301518). Panel B: each audit was made by a separate AI coding agent that had "
            "written neither the package nor its tests but had read the package code, so its simulator is a second "
            "implementation and not an independent reading of the specification; the simulators used seed 20261005 "
            "with spawn-key prefixes 971 to 977. The follow-up settings of the calibration package (job 5306863) "
            "were audited separately on 6 October 2026, on the login node: the audit found no error in the "
            "follow-up code or in a stored value, and both ends of the range of the largest view mean on the AP "
            "axis (Supplementary Table S25, panel E) were reproduced by the second implementation; target-sampling "
            "refits were not rerun, and a difference in bias below about 0.02 mm would not have been detected. "
            f"Automated tests: {old} for the library of the planned and second-amendment analyses and "
            f"{sum(new.values())} for the third-amendment packages; all {n_all} passed on 6 October 2026 in one job "
            f"(job 5306870), after {n_early} had passed earlier that morning (job 5306852) before "
            f"{n_all - n_early} calibration tests were added, at commit 57371d3 with the third-amendment files as "
            f"additions not yet committed. Sources: {rel(s1)}, {rel(s2)}, tests/independent/indep_check_e2.out, "
            f"{rel(DERIVED / 'pytest_collect_2026-10-06.csv')}, logs/duomax-pytest-5306852.out, "
            "logs/duomax-pytest-5306870.out and the audit records of each package; generated by "
            f"{SCRIPT}. AI, artificial intelligence; AP, anteroposterior; {ABBR_MC}; SL, septolateral.")
    return "\n\n".join([title, panel("A", "Second implementations of the planned and second-amendment analyses"),
                        md_table(hdrA, rowsA, ["l"] * 8), panel("B", "Separate audits of the third-amendment analyses"),
                        md_table(hdrC, aud, ["l"] * 5), foot]), (old, new, n_early, n_all)


def tableS27():
    """Map from the numbers of the main text (Results, Limitations and the Methods values that are results) and
    of the supplement to result files and scripts. Every value is read here from the named file; the values of
    the main text are asserted at the end of the function, so that a change in a result file or in a selection
    stops the build. Rows or parts of rows marked 'supplement' are not quoted in the main text."""
    Fd, Ad = "results/2026-09-18_full/analysis/", "results/2026-09-18_amend2/analysis/"
    T = "S27"
    rowsA = []
    PCT = lambda x, d=1: f(100 * float(x), d) + "%"

    def ma(label, value, files, column, script):
        for fp in files.split("; "):
            assert (ROOT / fp).exists(), fp
        for sp_ in script.split("; "):
            assert (ROOT / sp_).exists(), sp_
        rowsA.append([f"{label}: {value}", files, column, script])
    s2, s14, s4, s56, s8_, s11 = ("code/03_analyse_e2.py", "code/02_analyse_e1e3.py", "code/04_analyse_E4.py",
                                  "code/05_analyse_E5E6.py", "code/08_analyse_amend2_e1e3.py", "code/11_amend2_E5bE6b_tables.py")
    E2 = dict(p_beat_cv=0.15, axis="AP")
    wm = sel(rd(ROOT / (Fd + "E2_window_met.csv")), **E2, W=0.15, p_data_mode="prospective")
    g2 = lambda d, rs, N: float(one(d, rhythm_state=rs, p_N_beats=N).value)
    ma("First three and first five beats within the ±15% window: sinus rhythm; atrial fibrillation (RR-interval CV 20%)",
       f"{PCT(g2(wm, 'sinus', 3))} and {PCT(g2(wm, 'sinus', 5))}; {PCT(g2(wm, 'AF_rr0.2', 3))} and {PCT(g2(wm, 'AF_rr0.2', 5))}",
       Fd + "E2_window_met.csv", "value; beat_cv 0.15, W 0.15, prospective, AP; sinus and AF_rr0.2, N 3 and 5", s2)
    ba = sel(rd(ROOT / (Fd + "E2_beats_acquired.csv")), **E2, W=0.15, p_data_mode="prospective")
    ma("Mean number of beats acquired under a three-beat and a five-beat rule", f"{f(g2(ba, 'sinus', 3), 2)} and {f(g2(ba, 'sinus', 5), 2)}",
       Fd + "E2_beats_acquired.csv", "value; sinus, beat_cv 0.15, W 0.15, prospective, AP, N 3 and 5", s2)
    li = sel(rd(ROOT / (Fd + "E2_limited.csv")), **E2, rhythm_state="sinus", p_data_mode="retrospective")
    l0 = one(li, W=0.0, p_N_beats=3)
    ma("Retrospective views with fewer than three stored beats; limited sampling with the window, three and five beats",
       f"{kn(round(l0.value * l0.n), l0.n)}; {PCT(one(li, W=0.15, p_N_beats=3).value)} and {PCT(one(li, W=0.15, p_N_beats=5).value)}",
       Fd + "E2_limited.csv", "value, n; sinus, beat_cv 0.15, retrospective, AP; N 3, W 0; N 3 and 5, W 0.15", s2)
    er = sel(rd(ROOT / (Fd + "E2_error.csv")), **E2, rhythm_state="sinus", p_N_beats=3, p_data_mode="prospective", estimator="A1",
             estimand="T1", metric="rmse_mm", subset="all")
    we = one(sel(rd(ROOT / (Fd + "E2_window_effect.csv")), **E2, rhythm_state="sinus", p_N_beats=3, p_data_mode="prospective",
                 estimator="A1", estimand="T1", metric="rmse_mm"), W=0.15)
    ma("RMSE of the three-beat anchor-view mean without and with the window (mm); difference (mm, 95% MCI)",
       f"{f(one(er, W=0.0).value, 2)} and {f(one(er, W=0.15).value, 2)}; {f(we['diff'], 2)} mm ({f(we.lo95, 2)} to {f(we.hi95, 2)})",
       Fd + "E2_error.csv; " + Fd + "E2_window_effect.csv", "sinus, beat_cv 0.15, N 3, prospective, AP, A1, T1, rmse_mm; W 0 and 0.15; diff, lo95, hi95", s2)
    h = sel(rd(FULL / "E1E3_e1_headline.csv"), **E1_BASE, axis="AP", estimand="T1")
    hb = lambda u, e, c="bias": float(one(h, u=u, estimator=e)[c])
    r4 = lambda u: h[(h.u == u) & h.estimator.isin(RULE)].rmse
    rb, rl = r4("base"), r4("low")
    gain = float(f(hb("base", "A3", "rmse"), 2)) - float(f(hb("base", "A4", "rmse"), 2))
    ma("Bias of the largest view mean and of the mean across views, base case; low-underestimation scenario (mm)",
       f"{f(hb('base', 'A3'), 2)} and {f(hb('base', 'A2'), 2)}; {f(hb('low', 'A3'), 2)} and {f(hb('low', 'A2'), 2)}",
       Fd + "E1E3_e1_headline.csv", "bias; u base and low, three views, three beats, beat_cv 0.15, AP, T1; estimators A3 and A2", s14)
    ma("RMSE of the four view rules, base case; base case and low-underestimation scenario (mm); lowering of RMSE by review, base case (Table 1)",
       f"{rng(rb.min(), rb.max(), 2)}; {rng(min(rb.min(), rl.min()), max(rb.max(), rl.max()), 2)}; {f(gain, 2)} mm "
       f"({f(hb('base', 'A3', 'rmse'), 2)} against {f(hb('base', 'A4', 'rmse'), 2)})",
       Fd + "E1E3_e1_headline.csv", "rmse; u base and low, AP, T1; estimators A1 to A4", s14)
    w = sel(rd(AM2 / "E1bE3b_e1_worstcase.csv"), **v4.E1B_WORST)
    wr = lambda e: rng(one(w, estimator=e).bias_min, one(w, estimator=e).bias_max, 2)
    ma("Bias range over the 12 view-accuracy scenarios: largest view mean; mean across views; anchor-view mean (mm)",
       f"{wr('A3')}; {wr('A2')}; {wr('A1')}", Ad + "E1bE3b_e1_worstcase.csv",
       "bias_min, bias_max; K 3, beat_cv 0.15, view_over True, window 0, S_median_mm 10, AP, T1; estimators A3, A2, A1", s8_)
    e2 = rd(AM2 / "E1bE3b_e1_sensitivity.csv")
    e2 = e2[(e2.scenario == "base") & (e2.axis == "AP") & e2.infl_min.notna()].iloc[0]
    ma("Excess of the largest view mean over the anchor-view mean across the 12 scenarios, stored second-amendment values (mm)",
       rng(e2.infl_min, e2.infl_max, 2), Ad + "E1bE3b_e1_sensitivity.csv", "infl_min, infl_max; scenario base, AP, A3 minus A1", s8_)
    e3 = sel(rd(FULL / "E1E3_e3_base.csv"), **v4.E3_BASE)
    e3g = lambda e, c: float(one(e3, estimator=e)[c])
    ma("Sensitivity and specificity at 13 mm: mean across views; largest view mean; difference in sensitivity (percentage points)",
       f"{PCT(e3g('A2', 'sensitivity'))} and {PCT(e3g('A2', 'specificity'))}; {PCT(e3g('A3', 'sensitivity'))} and {PCT(e3g('A3', 'specificity'))}; "
       f"{f(100 * (e3g('A3', 'sensitivity') - e3g('A2', 'sensitivity')), 0)}",
       Fd + "E1E3_e3_base.csv", "sensitivity, specificity; u base, AP, cutoff_mm 13; estimators A2 and A3", s14)
    op = sel(rd(ROOT / (Fd + "E4_operating.csv")), cell=193, axis="AP")
    o5, o3 = one(op, tau=5.0), one(op, tau=3.0)
    at = one(sel(rd(ROOT / (Fd + "E4_any_trigger.csv")), cell=193, axis="AP", metric="p_any_adj"), tau=5.0)
    n_fp5 = int(o5.n_fired_pairs - o5.n_fired_event)
    ma("Patients with a 5 mm adjudication trigger, four views", PCT(at.value), Fd + "E4_any_trigger.csv", "value; cell 193, AP, p_any_adj, tau 5", s4)
    ma("View pairs with a true view error above 2 mm; 5 mm trigger fired in pairs with and without such an error; PPV; "
       "3 mm trigger, sensitivity and false-positive rate (supplement)",
       f"{kn(o5.n_event_pairs, o5.n_pairs)}; {kn(o5.n_fired_event, o5.n_event_pairs)} and "
       f"{n_(n_fp5)} of {n_(o5.n_nonevent_pairs)} ({pc(n_fp5, o5.n_nonevent_pairs, 2)}); "
       f"{PCT(o5.ppv)} ({n_(o5.n_fired_event)} of {n_(o5.n_fired_pairs)}); {PCT(o3.hit)} and {PCT(o3.false_alarm)}",
       Fd + "E4_operating.csv", "cell 193, AP, tau 5 and 3; n_event_pairs, n_pairs, n_fired_event, n_fired_pairs, n_nonevent_pairs, ppv, hit, false_alarm", s4)
    pb = sel(rd(ROOT / (Fd + "E4_ppv_base.csv")), axis="AP", tau=5.0, N_beats=3)
    ma("PPV of the 5 mm trigger at a beat-to-beat variation of 5% and of 30%", f"{PCT(one(pb, beat_cv=0.05).ppv)} and {PCT(one(pb, beat_cv=0.30).ppv)}",
       Fd + "E4_ppv_base.csv", "ppv; AP, tau 5, N_beats 3; beat_cv 0.05 and 0.30", s4)
    el = one(rd(ROOT / (Fd + "E4_ellipticity_truth.csv")), r_mean=0.64, tau=3.0, n=14400000)
    ma("Patients whose true AP and SL spans differed by 3 mm or more, planned run", PCT(el.value), Fd + "E4_ellipticity_truth.csv",
       "value; r_mean 0.64, tau 3, pooled conditions of E4", s4)
    rq = sel(rd(ROOT / (Fd + "E5E6_E5_reference_quality.csv")), cell=5)
    rqg = lambda ref, m_: float(one(rq, ref=ref, metric=m_).value)
    adj = [rqg(k, "ref_mae_vs_T1") for k in ["adj_tol1", "adj_tol2", "adj_tol3"]]
    ma("Bias of a single read; MAE of a single read and of the mean of two reads; MAE of adjudicated references (supplement) (mm)",
       f"{f(rqg('single', 'ref_ba_bias_vs_T1'), 2)}; {f(rqg('single', 'ref_mae_vs_T1'), 2)} and {f(rqg('mean2', 'ref_mae_vs_T1'), 2)}; {rng(min(adj), max(adj), 2)}",
       Fd + "E5E6_E5_reference_quality.csv", "value; cell 5; ref single, mean2, adj_tol1 to adj_tol3; ref_ba_bias_vs_T1, ref_mae_vs_T1", s56)
    aa = sel(rd(ROOT / (Fd + "E5E6_E5_ai_agreement.csv")), cell=5, ai="independent", sigma_ai_mm=2.0)
    aag = lambda ref, m_: one(aa, ref=ref, m=m_)
    ma("Apparent minus true MAE of the model with independent error (SD 2 mm): against a single read; against the mean of two "
       "reads; widening of the limits of agreement against a single read (mm)",
       f"{f(aag('single', 'mae')['diff'], 2)} mm ({f(aag('single', 'mae').rel_diff_pct, 0)}%); {f(aag('mean2', 'mae')['diff'], 2)} mm "
       f"({f(aag('mean2', 'mae').rel_diff_pct, 0)}%); {f(aag('single', 'loa_width')['diff'], 2)}",
       Fd + "E5E6_E5_ai_agreement.csv", "diff, rel_diff_pct; cell 5, independent, sigma_ai_mm 2; m mae and loa_width; ref single and mean2", s56)
    dc = one(rd(AM2 / "E5bE6b_E5b_decomposition_summary.csv"), cell=9, ref="single")
    ma("Variance of the image error and of the total error of a single read (mm²)", f"{f(dc.var_img, 2)} and {f(dc.var_ref, 2)}",
       Ad + "E5bE6b_E5b_decomposition_summary.csv", "var_img, var_ref; cell 9, single", "code/10_amend2_E5b_decomposition.py; " + s11)
    pp = sel(rd(AM2 / "E5bE6b_E5b_paired.csv"), cell=9, ai="inherited", sigma_ai_mm=2.0, metric="p_lower_mae_than_independent")
    pp = pp[pp.ref != "T1"].value
    ma("Studies in which the model inheriting the systematic error of a single read had the lower apparent MAE, identical own "
       "errors, five reference designs (supplement)", f"{f(100 * pp.min(), 0)}% to {f(100 * pp.max(), 0)}%",
       Ad + "E5bE6b_E5b_paired.csv", "value; cell 9, inherited, sigma_ai_mm 2, p_lower_mae_than_independent; all reference designs", s11)
    ab = sel(rd(AM2 / "E5bE6b_E5b_ai_agreement.csv"), ai="shared", sigma_ai_mm=2.0, ref="single", m="mae")
    a0, a1 = one(ab, cell=9), one(ab, cell=11)
    assert a0.p_ai_shared_lambda == 0.0 and a1.p_ai_shared_lambda == 1.0
    ma("Apparent MAE against a single read, and true MAE, as the share of image error reproduced rose from none to all (mm)",
       f"{f(a0.apparent, 2)} to {f(a1.apparent, 2)}; {f(a0['true'], 2)} to {f(a1['true'], 2)}",
       Ad + "E5bE6b_E5b_ai_agreement.csv", "apparent, true; shared, sigma_ai_mm 2, single, mae; cells 9 and 11", s11)
    pr = sel(rd(ROOT / (Fd + "E5E6_E6_precision.csv")), cell=22, design="gold")
    ma("SD of the upper limit of agreement across reader pairs, 25 and 200 double-read cases, planned analysis; mean width of the "
       "reported interval at 200 cases (mm)",
       f"{f(one(pr, n_double=25).loa_hi_empirical_sd, 2)} and {f(one(pr, n_double=200).loa_hi_empirical_sd, 2)}; {f(one(pr, n_double=200).loa_ci_width_each_mean, 2)}",
       Fd + "E5E6_E6_precision.csv", "loa_hi_empirical_sd, loa_ci_width_each_mean; cell 22, n_double 25 and 200", s56)
    fz = rd(ROOT / (Fd + "E5E6_E6_size_check.csv"))
    fz = fz[fz.test == "f_variance"]
    ma("F test: combinations with the 95% MCI of the false-positive rate above 0.05; median false-positive rate (supplement)",
       f"{int(((fz.value - Z * fz.mcse) > 0.05).sum())} of {len(fz)}; {f(fz.value.median(), 3)}",
       Fd + "E5E6_E6_size_check.csv", "value, mcse; test f_variance", s56)
    bfz = rd(AM2 / "E5bE6b_E6b_size_check.csv")
    bfz = bfz[bfz.test == "bf_onesided"].value
    pw = sel(rd(AM2 / "E5bE6b_E6b_sentinel_power.csv"), cell=22, test="bf_onesided")
    pwg = lambda n, w_: float(one(pw, sentinel_n=n, w_anchor=w_).value)
    ma("Brown-Forsythe test, one-sided: range of the false-positive rate; power with 50 sentinel cases for a drift of 30% and of "
       "10%; power with 100 cases for a drift of 10%",
       f"{rng(bfz.min(), bfz.max(), 3)}; {f(pwg(50, 0.3), 2)} and {f(pwg(50, 0.1), 2)}; {f(pwg(100, 0.1), 2)}",
       Ad + "E5bE6b_E6b_size_check.csv; " + Ad + "E5bE6b_E6b_sentinel_power.csv", "value; test bf_onesided; cell 22, sentinel_n 50 and 100, w_anchor 0.3 and 0.1", s11)
    R = "results/2026-10-05_osrev/"
    B = []

    def m(label, fn, column, script, value):
        assert (ROOT / (R + fn)).exists(), fn
        for sp_ in script.split("; "):
            assert (ROOT / sp_).exists(), sp_
        B.append([f"{label}: {value}", R + fn, column, script])
    sb, sv, st_, sg, sa, sc_ = ("code/21_osrev_beats_tables.py", "code/21_osrev_views_analyse.py", "code/21_osrev_thresholds_analyse.py",
                               "code/21_osrev_triggers_tables.py", "code/21_osrev_ai_reference_analyse.py", "code/21_osrev_calibration_analyse.py")
    p = one(sel(rd(BT / "beats_table_rmse_difference_published_vs_paired.csv"), axis="AP", estimand="T1", N=3), rhythm_state="sinus")
    m("Window effect on the same simulated beats (mm, 95% MCI)", "beats/beats_table_rmse_difference_published_vs_paired.csv",
      "sinus, AP, T1, N 3; a3_diff_mm, a3_paired_lo95, a3_paired_hi95", sb, f"{f(p.a3_diff_mm, 2)} ({f(p.a3_paired_lo95, 2)} to {f(p.a3_paired_hi95, 2)})")
    e = one(sel(rd(BT / "beats_table_equal_expected_beats.csv"), grid="budget", axis="AP", estimand="T1", N=3), condition="sinus_cv15")
    d_, m_ = e.selection_window_minus_same_acquired_d_rmse_mm, e.selection_window_minus_same_acquired_d_rmse_mcse
    m("Windowed mean minus plain mean of the same acquired beats (mm, 95% MCI)", "beats/beats_table_equal_expected_beats.csv",
      "grid budget, sinus_cv15, AP, T1, N 3; selection_window_minus_same_acquired_d_rmse_mm, _mcse", sb, mci(d_, m_, 2))
    bb = sel(rd(BT / "beats_table_equal_budget.csv"), condition="sinus_cv15", axis="AP", estimand="T1", N=3)
    m("Windowed minus plain mean of all beats allowed, 4 and 13 beats (supplement) (mm)", "beats/beats_table_equal_budget.csv",
      "sinus_cv15, AP, T1, N 3, budget 4 and 13; window_minus_plain_all_B_d_rmse_mm", sb,
      f"{f(one(bb, budget=4).window_minus_plain_all_B_d_rmse_mm, 2)} and {f(one(bb, budget=13).window_minus_plain_all_B_d_rmse_mm, 2)}")
    l = rd(BT / "beats_table_limited_denominators.csv")
    l3 = l[(l.axis == "AP") & l.source.str.startswith("amendment 3") & (l.rhythm_state == "retro_sinus_cv15") & (l.N == 3) & (l.window != "none")].iloc[0]
    m("Retrospective views with at least three stored beats and no qualifying set (supplement)", "beats/beats_table_limited_denominators.csv",
      "rows with seed entropy 20261005, retro_sinus_cv15, AP, N 3, window ±0.15; n_enough_beats_window_not_met, n_views_with_at_least_N_available", sb,
      kn(l3.n_enough_beats_window_not_met, l3.n_views_with_at_least_N_available))
    ss_ = rd(BT / "beats_table_sensitivity.csv")
    n_low = int((ss_.T1_win3_minus_plain3_d_rmse_mm < 0).sum())
    m("Three-beat sensitivity comparisons with prospective acquisition in which the window lowered RMSE (22 settings on two axes)",
      "beats/beats_table_sensitivity.csv", "all rows, AP and SL; T1_win3_minus_plain3_d_rmse_mm below zero", sb, f"{n_low} of {len(ss_)}")
    ss_ = ss_[ss_.axis == "AP"].set_index("condition")
    m("RMSE of a three-beat mean at serial correlation 0 and 0.6 (supplement); window effect at correlation 0 and 0.6 (mm)", "beats/beats_table_sensitivity.csv",
      "sinus_ref, sinus_ar1_0.6, AP; T1_plain_3_rmse_mm, T1_win3_minus_plain3_d_rmse_mm", sb,
      f"{f(ss_.loc['sinus_ref', 'T1_plain_3_rmse_mm'], 2)} and {f(ss_.loc['sinus_ar1_0.6', 'T1_plain_3_rmse_mm'], 2)}; "
      f"{f(ss_.loc['sinus_ref', 'T1_win3_minus_plain3_d_rmse_mm'], 2)} and {f(ss_.loc['sinus_ar1_0.6', 'T1_win3_minus_plain3_d_rmse_mm'], 2)}")
    c = sel(rd(VW / "views_c_table1_s2.csv"), cell=703, axis="AP").set_index("rule")
    m("MAE and 95th percentile of absolute error, base case, four rules (mm)", "views/views_c_table1_s2.csv", "cell 703, AP; mae_mm, q95_abs_mm", sv,
      "; ".join(f"{f(c.loc[k, 'mae_mm'], 2)} and {f(c.loc[k, 'q95_abs_mm'], 2)}" for k in RULE))
    rt = rd(VW / "views_e_ratio_summary.csv")
    rt = rt[rt.axis == "AP"]
    assert len(rt) == 4 and rt.n_cells.eq(864).all()
    m("Ratio of the 95th percentile of absolute error to the RMSE, four rules, 864 cells of the view-accuracy grid", "views/views_e_ratio_summary.csv",
      "AP, rules A1 to A4; ratio_min, ratio_max", sv, rng(rt.ratio_min.min(), rt.ratio_max.max(), 2))
    vr = rd(VW / "views_a_variant_ranges.csv")
    vr = vr[vr.variant.str.startswith("base family") & (vr.axis == "AP")].set_index("rule")
    m("RMSE range over the 12 scenarios, four rules (Table 1) (mm)", "views/views_a_variant_ranges.csv", "variant base family, AP; rmse_mm_min, rmse_mm_max", sv,
      "; ".join(rng(vr.loc[k, "rmse_mm_min"], vr.loc[k, "rmse_mm_max"], 2) for k in RULE))
    t = rd(VW / "views_b_tally.csv")
    t = t[(t.axis == "AP") & t.scope.str.startswith("72 conditions")].set_index("objective")
    t30 = rd(VW / "views_b_tally.csv")
    t30 = t30[(t30.axis == "AP") & (t30.objective == "worst_abs_bias") & t30.scope.str.startswith("beat CV 30%")].iloc[0]
    n_a1 = int(t.loc["worst_abs_bias", "n_win_A1"])
    assert n_a1 == int(t30.n_win_A1) and n_a1 + int(t.loc["worst_abs_bias", "n_win_largest_view_rule"]) == 72
    m("Conditions (of 72) in which a largest-view rule was preferred: worst-case RMSE, MAE, 95th percentile of absolute error; "
      "worst-case absolute bias (anchor-view mean in the others, all at a beat-to-beat variation of 30%)", "views/views_b_tally.csv",
      "AP, 72 conditions and beat-to-beat variation 30%; n_win_largest_view_rule, n_win_A1", sv,
      ", ".join(str(int(t.loc[o, "n_win_largest_view_rule"])) for o in ["worst_rmse", "worst_mae", "worst_q95abs"]) +
      f"; {int(t.loc['worst_abs_bias', 'n_win_largest_view_rule'])} ({n_a1})")
    zu = rd(VW / "views_a_added_scenarios.csv")
    zu = zu[(zu.family == "zero_under") & (zu.axis == "AP") & (zu.K == 3) & (zu.beat_cv == 0.15) & (zu.window == 0) & (zu.view_over == True)].set_index("rule")
    assert len(zu) == 4
    m("No underestimation in any view (three views, beat-to-beat variation 15%, no window): bias of the largest view mean and of "
      "the mean across views (mm); their RMSE (supplement) (mm)", "views/views_a_added_scenarios.csv",
      "family zero_under, AP, K 3, beat_cv 0.15, window 0, view_over True; rules A3 and A2; bias_mm, rmse_mm", sv,
      f"{f(zu.loc['A3', 'bias_mm'], 2)} and {f(zu.loc['A2', 'bias_mm'], 2)}; {f(zu.loc['A3', 'rmse_mm'], 2)} and {f(zu.loc['A2', 'rmse_mm'], 2)}")
    dd = sel(rd(VW / "views_d_decomposition.csv"), axis="AP", K=3, view_over=True, contrast="A3-A1")
    m("Excess of the largest view mean over the anchor-view mean in this sample (supplement); with within-view variability only; "
      "with differences between views only (mm)", "views/views_d_decomposition.csv",
      "AP, K 3, view_over True, A3-A1; excess_full, excess_equal_views, excess_zero_noise", sv,
      f"{rng(dd.excess_full.min(), dd.excess_full.max(), 2)}; {rng(dd.excess_equal_views.min(), dd.excess_equal_views.max(), 2)}; {rng(dd.excess_zero_noise.min(), dd.excess_zero_noise.max(), 2)}")
    q = one(sel(rd(VW / "views_e_conditional_quantiles.csv"), scenario="4.8/20", window=0.15, rhythm="sinus", axis="AP", rule="A3"), span_bin="all")
    m("Central 95% of errors, largest view mean, base case (supplement) (mm)", "views/views_e_conditional_quantiles.csv",
      "scenario 4.8/20, window 0.15, sinus, AP, A3, span_bin all; q2.5_err_mm, q97.5_err_mm", sv, rng(q["q2.5_err_mm"], q["q97.5_err_mm"], 2))
    x2 = rd(TH / "thresholds_2x2.csv")
    x2 = x2[(x2["sample"] == "published_seed") & (x2.scenario == "base") & (x2.axis == "AP") & (x2.cutoff_mm == 13.0)].set_index("rule")
    m("False positives and false negatives per 1,000 at 13 mm, four rules", "thresholds/thresholds_2x2.csv",
      "published_seed, base, AP, cutoff_mm 13; fp_per_1000, fn_per_1000", st_, "; ".join(f"{f(x2.loc[k, 'fp_per_1000'], 0)} and {f(x2.loc[k, 'fn_per_1000'], 0)}" for k in RULE))
    nb = rd(TH / "thresholds_near_band.csv")
    nb = nb[(nb["sample"] == "amend3_pooled") & (nb.scenario == "base") & (nb.axis == "AP") & (nb.half_width_mm == 1.0)]
    m("Misclassified per 1,000 patients within 1 mm of a cut-off, 10,000,000 patients: all cut-offs (supplement); 13 mm", "thresholds/thresholds_near_band.csv",
      "amend3_pooled, base, AP, half_width_mm 1; misclassified_per_1000_band", st_,
      f"{rng(nb.misclassified_per_1000_band.min(), nb.misclassified_per_1000_band.max(), 0)}; "
      f"{rng(nb[nb.cutoff_mm == 13.0].misclassified_per_1000_band.min(), nb[nb.cutoff_mm == 13.0].misclassified_per_1000_band.max(), 0)}")
    ts = rd(TH / "thresholds_true_span_distribution.csv")
    ta = one(ts, scenario="base", axis="AP")
    m("True AP spans at or above 7, 10 and 13 mm", "thresholds/thresholds_true_span_distribution.csv", "base, AP; prop_ge_7mm, prop_ge_10mm, prop_ge_13mm", st_,
      f"{f(100 * ta.prop_ge_7mm, 1)}%, {f(100 * ta.prop_ge_10mm, 1)}%, {f(100 * ta.prop_ge_13mm, 1)}%")
    pr_ = sel(rd(TR / "triggers_pair_rates.csv"), axis="AP", definition="D0_published", tau_mm=5.0, measure="ppv")
    for coh, lab in [("published", "PPV of the 5 mm trigger, planned run (patient-clustered 95% MCI)"), ("new_pooled", "PPV of the 5 mm trigger in 1,000,000 further patients")]:
        x = one(pr_, cohort=coh)
        m(lab, "triggers/triggers_pair_rates.csv", f"cohort {coh}, AP, D0_published, tau 5, ppv; value, numerator, denominator, lo95_cluster, hi95_cluster", sg,
          f"{f(100 * x.value, 1)}% ({n_(x.numerator)} of {n_(x.denominator)}; {f(100 * x.lo95_cluster, 1)}% to {f(100 * x.hi95_cluster, 1)}%)")
    pt = one(sel(rd(TR / "triggers_patient_2x2.csv"), scope="AP", cohort="published", tau_mm=5.0), truth="any_pair:D0_published")
    m("Patient-level sensitivity of the 5 mm trigger", "triggers/triggers_patient_2x2.csv", "published, AP, tau 5, any_pair:D0_published; fired_error, not_fired_error", sg,
      kn(pt.fired_error, pt.fired_error + pt.not_fired_error))
    ed = sel(rd(TR / "triggers_error_definitions.csv"), cohort="published", axis="AP", tau_mm=5.0)
    d1 = one(ed, definition="D1_plus_other_under")
    m("View pairs with an error under the wider definition; 5 mm trigger fired", "triggers/triggers_error_definitions.csv",
      "published, AP, tau 5, D1_plus_other_under; n_error_pairs, n_fired_error_pairs", sg,
      f"{kn(d1.n_error_pairs, d1.n_pairs)}; {kn(d1.n_fired_error_pairs, d1.n_error_pairs)}")
    axs = rd(TR / "axes_scenarios.csv")
    axs = axs[axs.target_mean_diff_mm.notna()]
    assert len(axs) == 18 and set(axs.target_mean_diff_mm) == {2.0, 3.9, 6.0}
    m("Patients whose true AP and SL spans differed by 3 mm or more when the mean difference between the axes was set to 2 to 6 mm (18 joint distributions)",
      "triggers/axes_scenarios.csv", "rows with target_mean_diff_mm 2, 3.9 and 6; true_ge3", sg, f"{f(100 * axs.true_ge3.min(), 0)}% to {f(100 * axs.true_ge3.max(), 0)}%")
    rv = rd(TR / "review_dependence_summary.csv")
    rv = rv[(rv.axis == "AP") & (rv.overestimation == "base")].set_index("scenario")
    m("RMSE with review minus RMSE without review, sample of the reviewer grid: base case (supplement; 0.10 mm lower in the planned run, "
      "panel A); least accurate of the 12 scenarios (mm, MCSE)", "triggers/review_dependence_summary.csv",
      "AP, overestimation base, scenarios base and extreme_inaccurate; published_review_rmse_mm_difference, _mcse", sg,
      f"{em(rv.loc['base', 'published_review_rmse_mm_difference'], rv.loc['base', 'published_review_rmse_mm_difference_mcse'], 2, 3)}; "
      f"{em(rv.loc['extreme_inaccurate', 'published_review_rmse_mm_difference'], rv.loc['extreme_inaccurate', 'published_review_rmse_mm_difference_mcse'], 2, 3)}")
    rk = sel(rd(AI / "ai_ranking_matched.csv"), cell_id="base_cv15", noise="ind", ref="single")
    sh = rk[rk.family.str.startswith("share")]
    kq, ks = sh[sh.match == "eqmae"].n_candidate_better_mae, sh[sh.match == "eqsd"].n_candidate_better_mae
    assert len(kq) == 4 and len(ks) == 4 and sh.n_studies.eq(2000).all()
    m("Studies (of 2,000) in which a model reproducing a quarter to all of the image error had the lower apparent MAE: against a model "
      "of equal true MAE; of equal SD of own error; excess of its true MAE in the second comparison (mm)", "ai_reference/ai_ranking_matched.csv",
      "base_cv15, noise ind, ref single, families share_lam0.25 to share_lam1; n_candidate_better_mae, true_mae_diff", sa,
      f"{n_(kq.min())} to {n_(kq.max())} ({pc(kq.min(), 2000, 0)} to {pc(kq.max(), 2000, 0)}); "
      f"{n_(ks.min())} to {n_(ks.max())} ({pc(ks.min(), 2000, 0)} to {pc(ks.max(), 2000, 0)}); "
      f"{rng(sh[sh.match == 'eqsd'].true_mae_diff.min(), sh[sh.match == 'eqsd'].true_mae_diff.max(), 2)}")
    ih = sel(rd(AI / "ai_ranking_matched.csv"), cell_id="base_cv15", family="inherit", match="eqsd", noise="ind")
    ih = ih[ih.ref != "true"]
    m("Studies (of 2,000) in which the model inheriting the systematic error of a single read had the lower apparent MAE, independent "
      "own errors of equal SD, five reference designs (supplement)", "ai_reference/ai_ranking_matched.csv",
      "base_cv15, family inherit, match eqsd, noise ind; n_candidate_better_mae", sa,
      f"{n_(ih.n_candidate_better_mae.min())} to {n_(ih.n_candidate_better_mae.max())} ({pc(ih.n_candidate_better_mae.min(), 2000, 0)} to {pc(ih.n_candidate_better_mae.max(), 2000, 0)})")
    lo = rd(AI / "loa_conditional_vs_marginal.csv")
    lo = lo[lo.scenario == "rb075"].set_index("n_cases")
    m("SD of the upper limit of agreement, 25 and 200 double-read cases, nested simulation: for a fixed reader pair; between reader "
      "pairs; across pairs and case sets (mm)", "ai_reference/loa_conditional_vs_marginal.csv",
      "rb075, n_cases 25 and 200; sd_conditional, sd_between_pairs, sd_marginal", sa,
      f"{f(lo.loc[25, 'sd_conditional'], 2)} and {f(lo.loc[200, 'sd_conditional'], 2)}; {f(lo.loc[25, 'sd_between_pairs'], 2)} and "
      f"{f(lo.loc[200, 'sd_between_pairs'], 2)}; {f(lo.loc[25, 'sd_marginal'], 2)} and {f(lo.loc[200, 'sd_marginal'], 2)}")
    m("Sets (of 50,000) in which the reported interval at 200 cases covered the pair's own limit; the limit of a randomly chosen pair",
      "ai_reference/loa_conditional_vs_marginal.csv", "rb075, n_cases 200; n_covered_conditional, n_covered_marginal, n_studies", sa,
      f"{kn(lo.loc[200, 'n_covered_conditional'], lo.loc[200, 'n_studies'], 0)}; {kn(lo.loc[200, 'n_covered_marginal'], lo.loc[200, 'n_studies'], 0)}")
    mr = rd(AI / "loa_multireader_designs.csv")
    m("RMSE of the limit for a random pair: two readers; six readers (supplement) (mm)", "ai_reference/loa_multireader_designs.csv",
      "designs conventional_two_reader and crossed with n_readers 6; loa_rmse", sa,
      f"{f(one(mr, design='conventional_two_reader').loa_rmse, 2)}; {f(one(mr, design='crossed', n_readers=6).loa_rmse, 2)}")
    sz = rd(AI / "sentinel_size_summary.csv")
    sz = sz[sz.stratum == "all"].set_index("test")
    m("False-positive rate of the sentinel tests, third-amendment sample: F test, combinations with the 95% MCI above 0.05 (supplement); "
      "Brown-Forsythe test, range (supplement); permutation test, median", "ai_reference/sentinel_size_summary.csv",
      "stratum all; n_mci_above_level, size_min, size_max, size_median", sa,
      f"{int(sz.loc['f_onesided', 'n_mci_above_level'])} of 216; {f(sz.loc['bf_onesided', 'size_min'], 3)} to {f(sz.loc['bf_onesided', 'size_max'], 3)}; {f(sz.loc['perm_var_ratio_onesided', 'size_median'], 3)}")
    cs = rd(CA / "calibration_table_s25.csv")
    b25 = one(cs[cs.part.str.startswith("B.")], condition="scale 0.25; overestimation on")
    m("Biplane shortfall of the base case on the published comparison (supplement) (%, MCSE)", "calibration/calibration_table_s25.csv",
      "part B, scale 0.25, overestimation on; estimate, mcse (from libmatch.csv)", sc_, em(b25.estimate, b25.mcse, 1, 2))
    Cp = cs[cs.part.str.startswith("C. ")]
    cr = Cp[Cp.item.str.startswith("AP axis, inflow primary plane vs 3D maximum") & Cp.condition.str.startswith("range of means")].iloc[0]
    co = Cp[Cp.item.str.startswith("AP axis, inflow primary plane, off-centre") & Cp.condition.str.startswith("range of means")].iloc[0]
    lo_all, hi_all = min(cr.lower, co.lower), max(cr.upper, co.upper)
    m("Mean long-axis underestimation implied by the calibration, range over fitted variants: rotation of the plane counted as error; "
      "off-centre placement alone; both readings, rounded as in the main text", "calibration/calibration_table_s25.csv",
      "part C, AP axis, inflow primary plane against the 3D maximal diameter and its off-centre component, rows 'range of means'; lower, upper", sc_,
      f"{f(cr.lower, 1)}% to {f(cr.upper, 1)}%; {f(co.lower, 1)}% to {f(co.upper, 1)}%; {f(lo_all, 0)}% to {f(hi_all, 0)}%")
    Dp = cs[cs.part.str.startswith("D.") & (cs.item == "largest view mean, bias")]
    for key, lab in [("half-normal shape at calibrated means, the 28", "Bias of the largest view mean at calibrated means, main error distribution, 28 scenarios (supplement) (mm)"),
                     ("half-normal and patient-level geometric shapes at all", "Bias of the largest view mean on the AP axis over all 60 calibrated scenarios (mm)")]:
        x = Dp[Dp.condition.str.startswith(key)].iloc[0]
        m(lab, "calibration/calibration_table_s25.csv", f"part D, largest view mean, bias, condition beginning '{key}'; lower, upper", sc_, rng(x.lower, x.upper, 2))
    fm = rd(CA / "followup_viewrules_metrics.csv")
    fm = fm[(fm.replicate == 0) & (fm.rule == "A3") & (fm.metric == "bias_mm") & (fm.axis == "SL") & fm.variant.isin(["ind15", "mid15", "ind00"])
            & fm["shape"].isin(["r1_halfnormal", "r1_patient"])]
    assert len(fm) == 24
    m("Bias of the largest view mean on the SL axis where the calibration suggested little underestimation (rotation of the plane counted "
      "as error; 24 scenarios) (mm)", "calibration/followup_viewrules_metrics.csv",
      "replicate 0, rule A3, bias_mm, SL, variants ind15, mid15 and ind00, shapes r1_halfnormal and r1_patient; value", sc_,
      rng(fm.value.min(), fm.value.max(), 2))
    w4 = one(sel(rd(AM2 / "E1bE3b_e1_worstcase.csv"), K=4, beat_cv=0.30, view_over=True, window=0.0, S_median_mm=10.0,
                 axis="AP", estimand="T1", estimator="A3"))
    B.append([f"Largest bias of the largest view mean over the 12 scenarios with four views and a beat variation of 30% (mm): {w4.bias_max:.2f}",
              "results/2026-09-18_amend2/analysis/E1bE3b_e1_worstcase.csv",
              "K 4, beat_cv 0.30, view_over True, window 0, S_median_mm 10, AP, T1, A3; bias_max", "code/08_analyse_amend2_e1e3.py"])
    # every number of the main text Results (and the Methods values that are results) must be in this table
    flat = " | ".join(r[0] for r in rowsA + B)
    MAIN = ["41.8% and 15.2%; 37.9% and 12.6%", "3.90 and 7.12", "22,440 of 100,000 (22.4%); 40.2% and 84.7%",
            "1.94 and 2.04; 0.10 mm (0.08 to 0.12)", "0.10 (0.09 to 0.11)", "0.17 (0.16 to 0.17)", "0.11 and 0.03", "0 of 44",
            "0.54 and −1.18; 1.10 and −0.22", "2.01 to 2.20; 1.70 to 2.34; 0.10 mm (2.11 against 2.01)", "and 4.39;", "and 4.00",
            "1.84 to 2.07", "−0.77 to 0.92; −3.16 to −0.28; −1.93 to 0.02", "72, 72, 72; 62 (10)", "1.53 and 0.40; 2.59 and 1.72",
            "6% to 30%", "−0.84 to 1.10", "0.73 to 1.95", ": 2.36", "0.31 to 2.48", "0.84 to 0.96; 0.11 to 2.06",
            "64.4% and 97.6%; 87.1% and 89.6%; 23", "18 and 92; 77 and 33", "402 to 415", "25.6%", ": 2.4%",
            "20,762 of 300,000 (6.9%); 1,835 of 20,762 (8.8%) and 799 of 279,238 (0.29%); 69.7% (1,835 of 2,634)",
            "69.7% (1,835 of 2,634; 67.9% to 71.5%)", "67.7% (", "99.1% and 30.5%", "1,845 of 15,344 (12.0%)",
            "140,353 of 300,000 (46.8%); 1,905 of 140,353 (1.4%)", ": 62.4%", "14% to 93%", "0.07 (0.004)",
            "0.40; 1.50 and 1.34", "3.07 and 3.88", "0.63 mm (39%); 0.52 mm (33%); 3.05", "2.23 to 1.89; 1.60 to 2.08",
            "1,766 to 2,000 (88% to 100%)", "(84% to 98%); 0.04 to 0.50", "0.68 and 0.27; 1.08 and 1.09; 1.28 and 1.12",
            "1.25 and 1.07; 0.65", "39,485 of 50,000 (79%)", "of 50,000 (20%)", "176 of 216", "0.024 to 0.057; 0.91 and 0.22; 0.32",
            "; 0.051"]
    for s_ in MAIN:
        assert s_ in flat, f"main-text number not mapped in Supplementary Table S27: {s_!r}"
    hdr = ["Number in the main text or supplement", "Result file", "Rows and columns", "Analysis script"]
    title = ("**Supplementary Table S27. Map from the numbers in the main text and supplement to the result files and "
             "the scripts that produced them.** Every number in the Results of the main text has a row; numbers "
             "marked supplement are quoted in the supplement only.")
    foot = ("The value after each label is read from the named file when this table is generated. Panel A: planned, "
            "first-amendment and second-amendment numbers; simulation outputs were written by "
            "code/01_run_experiment.py with code/configs/base.yaml or code/configs/amend2_2026-09-18.yaml, and the "
            "named script wrote the listed file; these files are those of the public archive (version 0.4.0). Panel "
            "B: third-amendment numbers (post hoc); the simulation script of each package is "
            "code/20_osrev_<package>_run.py. Numbers of the main text that are assumptions or design settings are in "
            "Supplementary Table S1, and those of Table 1 are also mapped in the note under that table. Base case unless stated: AP axis, three views, three beats per view, "
            "beat-to-beat variation 15%, sinus rhythm, prospective acquisition, error against the true maximal span "
            "(T1). Main Table 1 and all supplementary tables are generated by " + SCRIPT + ", which records one row "
            f"per printed number in {rel(DERIVED / 'table_cells_v06.csv')}. AF, atrial fibrillation; AP, "
            f"anteroposterior; MAE, mean absolute error; {ABBR_MC}; PPV, positive predictive value; RMSE, "
            "root-mean-square error; SL, septolateral.")
    return "\n\n".join([title, panel("A", "Planned, first-amendment and second-amendment analyses"), md_table(hdr, rowsA, ["l"] * 4),
                        panel("B", "Third-amendment analyses (post hoc)"), md_table(hdr, B, ["l"] * 4), foot])


# ====================================================================== main
FORBIDDEN = ["–", "—", "pre-specified", "prespecified", "amendment 2", "amendment 3", "26151", "AI validation",
             "independent reimplementation", "without access to the rule code", "acceptance window", "septal-lateral",
             "conservative 95% MCI", "“", "”", "’", "31% to 49%", "base scenario", "low scenario",
             "instrument factor", "to confirm", "gold set", "unbiased", "σ_", "AI draft", "independent audit",
             "3-beat", "size-valid", "anchoring", "image-level", "(A1)", "(A2)", "(A3)", "(A4)", "(A5)", "(A6)",
             "(A7)", "post hoc exploratory", "Supplementary Methods S7", "Supplementary Methods S8",
             "Supplementary Methods S9", "P values"]


def main():
    cfg = yaml.safe_load(v4.CFG.read_text())
    cfg2 = yaml.safe_load(v4.CFG2.read_text())
    t1_full, t1_text = table1_main()
    s1, s1n = tableS1(cfg, cfg2)
    s26, tests = tableS26()
    supp = [s1, tableS2(), tableS3(), tableS4(), tableS5(), tableS6(),
            "<!-- Supplementary Table S7 (ADEMP reporting checklist) is appended at build time from "
            "submission/source/ademp-checklist.md and is not generated here. -->",
            tableS8(), tableS9(), tableS10(), tableS11(), tableS12(), tableS13(), tableS14(), tableS15(), tableS16(),
            tableS17(), tableS18(), tableS19(), tableS20(), tableS21(), tableS22(), tableS23(), tableS24(), tableS25(),
            s26, tableS27()]
    doc = ["# Tables, manuscript v06",
           f"<!-- Generated by {SCRIPT} on the login node from results/2026-09-18_full/analysis, "
           "results/2026-09-18_amend2/analysis, results/2026-10-05_osrev and code/configs. Do not edit by hand; rerun "
           "the script. Citation keys ([@key]) follow the manuscript. Main text: Table 1. Supplement: Supplementary "
           "Tables S1 to S6 and S8 to S27; S7 is appended at build time. -->",
           "## Main table", t1_full, "## Supplementary tables"] + supp
    text = "\n\n".join(doc) + "\n"
    text = text.replace("reader bias", "reader offset").replace("Reader bias", "Reader offset")
    # one term each, as in the main text: AI proposal, image error
    text = text.replace("AI draft", "AI proposal").replace("image-level error", "image error")
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    low = body.lower()
    for bad in FORBIDDEN:
        assert bad.lower() not in low, f"forbidden term {bad!r} in output"
    assert "nan" not in re.sub(r"[A-Za-z]nan|nan[a-z]", "", low), "NaN printed"
    assert not re.search(r"(?<![\w(])\+\d", body), "plus sign before a number"
    nums = re.findall(r"\*\*Supplementary Table S(\d+)\.", body)
    assert [int(x) for x in nums] == [1, 2, 3, 4, 5, 6] + list(range(8, 28)), nums
    OUT.write_text(text, encoding="utf-8")
    DERIVED.mkdir(exist_ok=True)
    cells = pd.DataFrame(CELLS, columns=["table", "label", "file", "value", "decimals", "scale", "printed"])
    cells.to_csv(DERIVED / "table_cells_v06.csv", index=False)
    t1_words = sum(words(line) for line in t1_text.splitlines() if not set(line.strip()) <= set("|-: "))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text.split())} words; Table 1 with footnote {t1_words} words; "
          f"S1 status counts {s1n}; tests {tests}; {len(cells)} registered cells)")


if __name__ == "__main__":
    main()
