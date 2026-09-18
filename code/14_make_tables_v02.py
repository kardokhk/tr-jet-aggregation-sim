"""Build manuscript Tables 1 to 4 and Supplementary Table S1 for draft v02 as markdown.

Supersedes code/07_make_tables.py (v01 tables) for drafts/manuscript-2026-09-18-v02.md. 07 is left
unchanged so that drafts/tables-2026-09-18-v01.md stays reproducible.

Inputs (read only):
  code/configs/base.yaml, code/configs/amend2_2026-09-18.yaml       parameter values and grids
  notes/parameter-table-2026-09-18.md                               construct match and sources (encoded in table1())
  results/2026-09-18_full/analysis/E1E3_e1_headline.csv              Table 2A
  results/2026-09-18_amend2/analysis/E1bE3b_e1_worstcase.csv         Table 2B
  results/2026-09-18_full/analysis/E5E6_E5_reference_quality.csv     Table 3A
  results/2026-09-18_full/analysis/E5E6_E5_ai_agreement.csv          Table 3B (unbiased, inherited)
  results/2026-09-18_amend2/analysis/E5bE6b_E5b_reference_quality.csv  Table 3A (R_img row)
  results/2026-09-18_amend2/analysis/E5bE6b_E5b_ai_agreement.csv     Table 3B (shared-error AI)
  results/2026-09-18_amend2/analysis/E5bE6b_E5b_paired.csv           Table 3C
  results/2026-09-18_amend2/analysis/E5bE6b_E5b_decomposition_summary.csv  Table 3D
  results/2026-09-18_full/analysis/E5E6_E6_precision.csv             Table 4A
  results/2026-09-18_full/analysis/E5E6_E6_size_check.csv            Table 4B (pre-specified run)
  results/2026-09-18_amend2/analysis/E5bE6b_E6b_size_check.csv       Table 4B (amendment 2 run)
  results/2026-09-18_amend2/analysis/E5bE6b_E6b_sentinel_power.csv   Table 4C
  results/2026-09-18_amend2/analysis/E5bE6b_E6b_min_n.csv            Table 4D
  results/2026-09-18_amend2/analysis/E5bE6b_E6b_min_n_distribution.csv  Table 4D
  results/2026-09-18_amend2/analysis/face_table.csv                  Table S1 (strings checked, not retyped)
  notes/scratch/2026-09-18-findings-face-validity.md                 audit corrections applied in tableS1()

Output: drafts/tables-2026-09-18-v02.md (generated; do not hand edit).

Run from the project root (login node; no random numbers):
  /project/home/p201509/envs/duomax-sim/bin/python code/14_make_tables_v02.py

95% Monte Carlo interval (MCI) = estimate +/- 1.96 MCSE unless the source CSV stores the limits.
Rounding: half away from zero on the shortest decimal representation (same rule as 07).
"""
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results" / "2026-09-18_full" / "analysis"
AM2 = ROOT / "results" / "2026-09-18_amend2" / "analysis"
CFG = ROOT / "code" / "configs" / "base.yaml"
CFG2 = ROOT / "code" / "configs" / "amend2_2026-09-18.yaml"
OUT = ROOT / "drafts" / "tables-2026-09-18-v02.md"
SCRIPT = "code/14_make_tables_v02.py"
Z = 1.96
MINUS = "−"


# ---------------------------------------------------------------- formatting
def f(x, d):
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{q:.{d}f}"
    if s.startswith("-"):
        s = MINUS + s[1:]
    if s == MINUS + f"{0:.{d}f}":
        s = s[1:]
    return s


def ci(est, mcse, d):
    return f"{f(est, d)} ({f(est - Z * mcse, d)} to {f(est + Z * mcse, d)})"


def ci_lohi(est, lo, hi, d):
    return f"{f(est, d)} ({f(lo, d)} to {f(hi, d)})"


def pct(x, d=0):
    return f(100 * x, d) + "%"


def md_table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def sel(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m]


def one(df, **kw):
    r = sel(df, **kw)
    assert len(r) == 1, (kw, len(r))
    return r.iloc[0]


def rel(p):
    return str(p.relative_to(ROOT))


def mean_pct(scale):
    # mean of a half-normal with scale s is s * sqrt(2/pi) = 0.798 s
    # two significant figures, as in the manuscript (2.4%, 4.8%, 10%, 20%, 40%)
    m = 100 * scale * (2 / 3.141592653589793) ** 0.5
    s = f"{m:.1f}" if m < 10 else f"{m:.0f}"
    return (s[:-2] if s.endswith(".0") else s) + "%"


# ---------------------------------------------------------------- Table 1
def table1(cfg, cfg2):
    P = cfg["parameters"]
    v = lambda k: P[k]["value"]
    g = lambda k: P[k].get("grid")
    e1 = cfg["experiments"]["E1"]["vary"][0]["levels"]
    u_base, u_low = e1[0]["u_scale"], e1[1]["u_scale"]
    combos = cfg2["experiments"]["E1"]["vary"][0]["levels"]
    u_anch = sorted({c["u_anchor"] for c in combos})
    u_long = sorted({c["u_long"] for c in combos})
    assert all(abs(c["u_scale"][3] - 1.2 * c["u_long"]) < 1e-9 for c in combos)
    a2_levels = {d["name"]: d["levels"] for d in cfg2["experiments"]["E1"]["vary"][1:] if isinstance(d, dict)}
    spans = a2_levels["S_median_mm"]
    assert spans == [10.0, 13.0], spans
    P2 = cfg2["parameters"]
    lam = P2["ai_shared_lambda"]["grid"]
    mm = lambda x: f"{x:g} mm"
    lst = lambda xs, fn: ", ".join(fn(x) for x in xs)
    pc = lambda x: f"{100 * x:g}%"
    rows = []

    def add(group, label, base, pre, am2, construct, source):
        rows.append([group, label, base, pre, am2, construct, source])

    NV = "not varied"
    add("Case mix", "Median true AP span, S_AP (mm)", f"{v('S_median_mm'):g}", NV, lst(spans, lambda x: f"{x:g}") + " (E1b)",
        "close", "[@pascalxtr2020; @fcvm2024renal]")
    add("Case mix", "Log-SD of true AP span", f"{v('S_log_sd'):.2f}", NV, NV, "close",
        "[@pascalxtr2020] (derived from IQR)")
    add("Geometry", "Mean SL/AP ratio, r", f"{v('r_mean'):.2f}", lst(g("r_mean"), lambda x: f"{x:.2f}") + " (E4)",
        NV, "close", "[@song2011]; 0.53 [@singh2026]; 0.8 and 0.9 assumed")
    add("Geometry", "Support of r", f"{v('r_lower'):.1f} to {v('r_upper'):.1f}", NV, NV, "none", "assumed")
    add("Geometry", "Beta concentration of r", f"{v('r_concentration'):g}", NV, NV, "none", "assumed")
    add("Views", "Underestimation scale, anchor view (mean)", f"{u_base[0]:g} ({mean_pct(u_base[0])})",
        f"{u_low[0]:g} ({mean_pct(u_low[0])}), low scenario (E1, E3, E4)",
        lst(u_anch, lambda x: f"{x:g} ({mean_pct(x)})") + " (E1b, E3b)", "none", "assumed")
    add("Views", "Underestimation scale, views 2 and 3 (mean)", f"{u_base[1]:g} ({mean_pct(u_base[1])})",
        f"{u_low[1]:g} ({mean_pct(u_low[1])}), low scenario (E1, E3, E4)",
        lst(u_long, lambda x: f"{x:g} ({mean_pct(x)})") + " (E1b, E3b)", "close", "[@singh2026]")
    add("Views", "Underestimation scale, view 4 (mean)", f"{u_base[3]:g} ({mean_pct(u_base[3])})",
        f"{u_low[3]:g} ({mean_pct(u_low[3])}), low scenario (E1, E3, E4)", "1.2 times the views 2 and 3 scale", "none",
        "assumed")
    add("Views", "Underestimation cap", f"{v('u_max'):.1f}", NV, NV, "none", "assumed")
    add("Views", "Probability of overestimation, views 1 to 4", lst(v("p_over"), lambda x: f"{x:.2f}"),
        "on, off (E1, E3, E5)", "on, off (E1b, E5b)", "none",
        "assumed (mechanism only [@mascherbauer2005])")
    add("Views", "Overestimation magnitude, median", pc(v("o_median")), NV, NV, "none", "assumed")
    add("Views", "Overestimation magnitude, log-SD", f"{v('o_log_sd'):.1f}", NV, NV, "none", "assumed")
    add("Views", "Inter-view correlation of error drivers, ρ", f"{v('rho'):.1f}",
        lst(g("rho"), lambda x: f"{x:.1f}") + " (E1)", NV, "none", "assumed")
    add("Beats", "Beat-to-beat CV", pc(v("beat_cv")), lst(g("beat_cv"), pc) + " (all)",
        lst(a2_levels["beat_cv"], pc) + " (E1b); " + lst([0.05, 0.15, 0.30], pc) + " (E5b)", "close",
        "[@moraldo2013; @wong1987]")
    add("Beats", "RR CV, sinus rhythm", pc(v("rr_cv_sinus")), NV, NV, "none", "assumed")
    add("Beats", "RR CV, AF", pc(v("rr_cv_af")), lst(g("rr_cv_af"), pc) + " (E2)", NV, "none", "assumed")
    add("Beats", "Slope of log span on log preceding RR", f"{v('beta_rr'):.1f}", NV, NV, "none",
        "assumed (direction only [@sumida2003])")
    add("Beats", "Additional beat CV in AF (percentage points)", f"{100 * v('af_extra_cv'):g}", NV, NV, "none",
        "assumed")
    add("Beats", "Available beats, retrospective (zero-truncated Poisson rate; cap)",
        f"{v('avail_lambda'):g} (mean 4.07); {v('max_beats_retro')}", "prospective, retrospective (E2)", NV,
        "none", "assumed")
    add("Beats", "Beat cap, prospective", f"{v('max_beats_prosp')}", NV, NV, "none", "assumed")
    add("Reader", "Caliper SD per beat (mm)", f"{v('sigma_cal_mm'):.1f}", NV, NV, "close",
        "[@singh2026; @hauptmann2026]")
    add("Reader", "Reader bias SD, σ_rb (mm)", f"{v('sigma_rb_mm'):g}",
        lst(g("sigma_rb_mm"), lambda x: f"{x:g}") + " (E6)", lst(g("sigma_rb_mm"), lambda x: f"{x:g}") + " (E6b)",
        "close", "0.75 [@singh2026]; 2.0 [@alexander2022]")
    add("Instrument", "Instrument factor SD (log), per examination", f"{v('g_log_sd'):.2f}", NV, NV, "proxy",
        "[@fan1994]")
    add("AI model", "AI error SD, σ_AI (mm)", f"{v('sigma_ai_mm'):g}", lst(g("sigma_ai_mm"), lambda x: f"{x:g}") + " (E5)",
        lst(g("sigma_ai_mm"), lambda x: f"{x:g}") + " (E5b)", "none", "assumed")
    add("AI model", "Shared image-level error weight, λ", "0", NV, lst(lam, lambda x: f"{x:g}") + " (E5b)", "none",
        "assumed")
    add("AI model", "AI draft error SD (mm)", f"{v('ai_draft_sigma_mm'):g}",
        lst(g("ai_draft_sigma_mm"), lambda x: f"{x:g}") + " (E6)", lst(g("ai_draft_sigma_mm"), lambda x: f"{x:g}") + " (E6b)",
        "none", "assumed")
    add("AI model", "AI draft bias (mm)", f"{v('ai_draft_bias_mm'):g}", NV, NV, "none", "assumed")
    n_dgm = len(rows)
    n_assumed = sum(r[6].startswith("assumed") for r in rows)
    # design factors of the measurement rules (not parameters of the data-generating mechanism)
    add("Rules (design)", "Views per axis, K", f"{v('K')}", "1, 2, 3, 4 (E1, E3); 4 (E4)", "2, 3, 4 (E1b)", "design",
        "protocol")
    add("Rules (design)", "Beats per view, N", f"{v('N_beats')}", lst(g("N_beats"), str) + " (E1, E2, E4)", "3",
        "design", "protocol")
    add("Rules (design)", "Beat-consistency window, W", "±" + pc(v("window")),
        "none, " + lst([x for x in g("window") if x], lambda x: "±" + pc(x)) + " (E2)", "none, ±15% (E1b)",
        "design", "protocol")
    add("Rules (design)", "A4 warning and adjudication thresholds (mm)", f"{v('t_warn'):g} and {v('t_adj'):g}",
        "warning 2, 3, 4; adjudication 4, 5, 6 (E4)", NV, "design", "protocol")
    add("Rules (design)", "A4 detection of true overestimation; false rejection",
        f"{v('s_det'):g}; {v('f_rej'):g}", "0, 0.5, 0.8, 1.0; 0, 0.1, 0.3 (E1)", NV, "design", "protocol")
    add("Rules (design)", "Adjudication tolerance between two reads (mm)", f"{v('adj_tol_mm'):g}",
        lst(g("adj_tol_mm"), lambda x: f"{x:g}") + " (E5)", "1, 2, 3 (E5b)", "design", "protocol")

    title = (f"**Table 1. Parameters of the data-generating mechanism.** No source measured the TEE colour-jet "
             f"span directly, and {n_assumed} of {n_dgm} parameters of the data-generating mechanism had no "
             f"empirical source.")
    body = md_table(["Component", "Parameter", "Base value", "Values varied, pre-specified (experiment)",
                     "Values varied, amendment 2 (experiment)", "Construct match", "Source"], rows,
                    align=["l"] * 7)
    foot = ("Construct match: close, vena contracta width or coaptation gap; proxy, jet area or another "
            "area-based measure; none, no empirical source; design, a factor of the measurement rule rather than "
            "of the data-generating mechanism. Underestimation u_v is half-normal, so its mean is 0.798 times the "
            "scale; the anchor and long-axis grids of amendment 2 (E1b, E3b) were crossed to give 12 view-accuracy "
            "combinations of mean anchor underestimation 2.4%, 4.8%, 10% and 20% with mean long-axis "
            "underestimation 8%, 20% and 40%. The overestimation probability has a weak in-vitro proxy for the "
            "mechanism only, and the RR slope a direction-only aortic source, so both are counted as assumed. The "
            "long-axis scale is also consistent with a transgastric versus mid-oesophageal coaptation-gap "
            "difference in the bRIGHT registry (doi:10.1016/j.echo.2023.12.002), which draft v02 does not cite. "
            "Not a Monte Carlo table: no n per cell or MCSE applies. Sources: code/configs/base.yaml "
            f"({cfg['meta']['config_version']}), code/configs/amend2_2026-09-18.yaml "
            f"({cfg2['meta']['config_version']}) and notes/parameter-table-2026-09-18.md; generated by {SCRIPT}. "
            "AF, atrial fibrillation; AP, anteroposterior; CV, coefficient of variation; IQR, interquartile range; "
            "RR, interval between successive R waves; SL, septolateral; TEE, transoesophageal echocardiography.")
    return "\n\n".join([title, body, foot])


# ---------------------------------------------------------------- Table 2
EST = {"A1": "A1, anchor-view mean", "A2": "A2, mean of view means", "A3": "A3, maximum of view means",
       "A4": "A4, composite rule", "A5": "A5, median of view means", "A6": "A6, index beat",
       "A7": "A7, offset-corrected mean"}
E1_BASE = dict(K=3, beat_cv=0.15, rho=0.3, N_beats=3, view_over=True)


def combo_lab(c):
    a, l = c.split("/")
    return f"{a}% / {l}%"


def table2():
    src_a = FULL / "E1E3_e1_headline.csv"
    h = sel(pd.read_csv(src_a), **E1_BASE)
    h = h[h.estimand == "T1"]
    assert len(h) == 2 * 7 * 2, len(h)
    rowsA = []
    for u, lab in [("base", "Long-axis underestimation 20% (base scenario)"),
                   ("low", "Long-axis underestimation 8% (low scenario)")]:
        rowsA.append([f"*{lab}*", "", "", "", "", "", ""])
        for e in EST:
            cells = [EST[e]]
            for ax in ["AP", "SL"]:
                r = one(h, u=u, estimator=e, axis=ax)
                cells += [ci_lohi(r.bias, r.bias_lo95, r.bias_hi95, 2), f(r.relbias_pct, 1),
                          ci_lohi(r.rmse, r.rmse_lo95, r.rmse_hi95, 2)]
            rowsA.append(cells)
    hdrA = ["Estimator", "AP bias (mm, 95% MCI)", "AP relative bias (%)", "AP RMSE (mm, 95% MCI)",
            "SL bias (mm, 95% MCI)", "SL relative bias (%)", "SL RMSE (mm, 95% MCI)"]

    src_b = AM2 / "E1bE3b_e1_worstcase.csv"
    w = sel(pd.read_csv(src_b), K=3, beat_cv=0.15, view_over=True, window=0.0, S_median_mm=10.0, estimand="T1")
    rowsB = []
    mcB = []
    for ax in ["AP", "SL"]:
        rowsB.append([f"*{ax} axis*", "", "", "", ""])
        for e in ["A1", "A2", "A3", "A4", "A5", "A6"]:
            r = one(w, axis=ax, estimator=e)
            rowsB.append([EST[e],
                          f"{ci(r.bias_min, r.bias_min_mcse, 2)} at {combo_lab(r.bias_min_combo)}",
                          f"{ci(r.bias_max, r.bias_max_mcse, 2)} at {combo_lab(r.bias_max_combo)}",
                          ci(r.maxabs_bias, r.maxabs_bias_mcse, 2),
                          f"{ci(r.rmse_max, r.rmse_max_mcse, 2)} at {combo_lab(r.rmse_max_combo)}"])
            mcB += [r.bias_min_mcse, r.bias_max_mcse, r.maxabs_bias_mcse, r.rmse_max_mcse]
    a7 = one(w, axis="AP", estimator="A7")
    hdrB = ["Estimator", "Lowest bias (mm, 95% MCI) at anchor / long-axis underestimation",
            "Highest bias (mm, 95% MCI) at anchor / long-axis underestimation", "Maximum |bias| (mm, 95% MCI)",
            "Maximum RMSE (mm, 95% MCI) at anchor / long-axis underestimation"]
    mx = lambda c: h[c].max()
    title = ("**Table 2. Estimator bias and root-mean-square error against the true maximal span T1.** In the "
             "pre-specified base case the maximum-type rules (A3, A4) overestimated and the mean of view means "
             "(A2) underestimated T1; across 12 combinations of anchor and long-axis underestimation, A3 and A4 "
             "had the smallest maximum absolute bias.")
    foot = ("Panel A, pre-specified E1 base case: K = 3 views, N = 3 beats per view, ±15% beat-consistency window, "
            "beat CV 15%, overestimation present, inter-view correlation 0.3, mean anchor underestimation 4.8% "
            "(base) or 2.4% (low scenario). Panel B, amendment 2 (E1b): the same condition but without a window "
            "and with median true AP span 10 mm; each estimator's lowest and highest bias, maximum absolute bias "
            "and maximum RMSE are taken over the 12 combinations of mean anchor underestimation (2.4%, 4.8%, 10%, "
            "20%) and mean long-axis underestimation (8%, 20%, 40%), written anchor / long-axis. A7 is excluded from "
            f"panel B because it needs the true view offsets; its AP bias ranged from {f(a7.bias_min, 2)} to "
            f"{f(a7.bias_max, 2)} mm. Bias is the mean of estimate minus T1; relative bias is the mean of "
            "per-patient ratios (estimate / T1) minus 1. A7, offset-corrected mean, divides each view mean by its "
            "true offset (1 − u_v + o_v); it cannot be deployed and is not a lower bound on error. n = 100,000 "
            "simulated patients per cell. Maximum MCSE, panel A: bias "
            f"{mx('bias_mcse'):.4f} mm, relative bias {mx('relbias_pct_mcse'):.3f} percentage points, RMSE "
            f"{mx('rmse_mcse'):.4f} mm; panel B: {max(mcB):.4f} mm. The 95% MCI is estimate ± 1.96 MCSE (panel A "
            "limits as stored in the source CSV). Panel B extremes are bounds over this grid, driven by its corners, "
            f"not over clinical practice. Sources: {rel(src_a)} (panel A) and {rel(src_b)} (panel B); generated by "
            f"{SCRIPT}. AP, anteroposterior; CV, coefficient of variation; MCI, Monte Carlo interval; MCSE, Monte "
            "Carlo standard error; RMSE, root-mean-square error; SL, septolateral.")
    return "\n\n".join([title, "*Panel A. Pre-specified base case (E1)*", md_table(hdrA, rowsA),
                        "*Panel B. Worst case across 12 view-accuracy combinations (E1b, amendment 2)*",
                        md_table(hdrB, rowsB, align=["l", "r", "r", "r", "r"]), foot])


# ---------------------------------------------------------------- Table 3
REF = {"single": "Single read", "mean2": "Mean of two reads", "adj_tol1": "Adjudicated, tolerance 1 mm",
       "adj_tol2": "Adjudicated, tolerance 2 mm", "adj_tol3": "Adjudicated, tolerance 3 mm"}
E5_BASE = dict(p_beat_cv=0.15, p_view_over=True)


def table3():
    s_rq, s_ag = FULL / "E5E6_E5_reference_quality.csv", FULL / "E5E6_E5_ai_agreement.csv"
    s_rq2, s_ag2 = AM2 / "E5bE6b_E5b_reference_quality.csv", AM2 / "E5bE6b_E5b_ai_agreement.csv"
    s_pr, s_dc = AM2 / "E5bE6b_E5b_paired.csv", AM2 / "E5bE6b_E5b_decomposition_summary.csv"
    rq = sel(pd.read_csv(s_rq), **E5_BASE)
    assert rq.n.eq(2000).all()
    ag = sel(pd.read_csv(s_ag), **E5_BASE)
    rq2 = sel(pd.read_csv(s_rq2), **E5_BASE, p_ai_shared_lambda=0.0)
    ag2 = sel(pd.read_csv(s_ag2), **E5_BASE)
    assert set(ag2.cell) == {9, 10, 11}, set(ag2.cell)

    # Panel A
    rowsA, mcA = [], []
    for ref, lab in REF.items():
        g = rq[rq.ref == ref].set_index("metric")
        b, m = g.loc["ref_ba_bias_vs_T1"], g.loc["ref_mae_vs_T1"]
        padj = ci(100 * g.loc["p_adjudicated"].value, 100 * g.loc["p_adjudicated"].mcse, 1) \
            if "p_adjudicated" in g.index else "not applicable"
        rowsA.append([lab, "pre-specified", ci(b.value, b.mcse, 2), ci(m.value, m.mcse, 2), padj])
        mcA += [b.mcse, m.mcse]
    g = rq2[rq2.ref == "R_img"].set_index("metric")
    b, m = g.loc["ref_ba_bias_vs_T1"], g.loc["ref_mae_vs_T1"]
    rowsA.append(["Image composite R_img (no caliper error or reader bias)", "amendment 2",
                  ci(b.value, b.mcse, 2), ci(m.value, m.mcse, 2), "not applicable"])
    mcA += [b.mcse, m.mcse]
    hdrA = ["Reference design", "Run", "Bias vs T1 (mm, 95% MCI)", "MAE vs T1 (mm, 95% MCI)",
            "Adjudicated (%, 95% MCI)"]

    # Panel B
    comps = [("true", "single", "T1 (true)"), ("apparent", "single", "Single read"),
             ("apparent", "mean2", "Mean of two reads")]
    metrics = [("mae", 2), ("loa_width", 2), ("ba_bias", 2), ("icc", 3)]
    rowsB, mcB = [], {m: 0.0 for m, _ in metrics}

    def block(df, ai, sg, model_lab, run_lab, extra=None):
        for kind, ref, clab in comps:
            cells = [model_lab, f"{sg:g}", run_lab, clab]
            for m, d in metrics:
                r = one(df, ai=ai, sigma_ai_mm=sg, ref=ref, m=m, **(extra or {}))
                cells.append(ci(r[kind], r["mcse_" + kind], d))
                mcB[m] = max(mcB[m], r["mcse_" + kind])
            rowsB.append(cells)

    for sg in [1.0, 2.0, 3.0]:
        block(ag, "independent", sg, "Unbiased", "pre-specified")
    block(ag, "inherited", 2.0, "Inherited bias", "pre-specified")
    block(ag2, "independent", 2.0, "Shared error, λ = 0 (unbiased)", "amendment 2", dict(p_ai_shared_lambda=0.0))
    block(ag2, "shared", 2.0, "Shared error, λ = 0.5", "amendment 2", dict(p_ai_shared_lambda=0.5))
    block(ag2, "shared", 2.0, "Shared error, λ = 1", "amendment 2", dict(p_ai_shared_lambda=1.0))
    hdrB = ["AI model", "σ_AI (mm)", "Run", "Compared with", "MAE (mm, 95% MCI)", "95% LoA width (mm, 95% MCI)",
            "Bland-Altman bias (mm, 95% MCI)", "ICC(A,1) (95% MCI)"]

    # Panel C: paired per-study comparison, inherited-bias AI vs unbiased AI, sigma 2 mm, lambda-0 cell (cell 9)
    pr = sel(pd.read_csv(s_pr), **E5_BASE, p_ai_shared_lambda=0.0, ai="inherited", sigma_ai_mm=2.0)
    assert pr.n.eq(2000).all() and set(pr.cell) == {9}
    rowsC, mcC = [], []
    for ref, lab in list(REF.items()) + [("T1", "T1 (true)")]:
        p = one(pr, ref=ref, metric="p_lower_mae_than_independent")
        d = one(pr, ref=ref, metric="paired_mae_diff_mean")
        rowsC.append([lab, ci(100 * p.value, 100 * p.mcse, 1), ci(d.value, d.mcse, 3)])
        mcC.append(100 * p.mcse)
    hdrC = ["Compared with", "Studies in which the inherited-bias AI had lower MAE (%, 95% MCI)",
            "Mean paired MAE difference, inherited minus unbiased (mm, 95% MCI)"]

    # Panel D: decomposition of the reference error, cell 9
    dc = sel(pd.read_csv(s_dc), **E5_BASE)
    assert set(dc.cell) == {9} and dc.n_studies.eq(2000).all()
    rowsD = []
    items = [("var_ref", "Total reference error variance, Var(e_ref)", "mm²"),
             ("var_img", "Image-level component shared by all readers, Var(e_img)", "mm²"),
             ("var_read", "Read-specific component, Var(e_rd)", "mm²"),
             ("two_cov_img_read", "Twice their covariance, 2 Cov(e_img, e_rd)", "mm²"),
             ("var_reader_bias", "Of Var(e_rd): reader-bias variance", "mm²"),
             ("share_img_of_var_ref", "Ratio Var(e_img) / Var(e_ref)", "ratio")]
    mcD = []
    for col, lab, unit in items:
        cells = [lab + (f" ({unit})" if unit != "ratio" else "")]
        for ref in ["single", "mean2"]:
            r = one(dc, ref=ref)
            mc = r.get("mcse_" + col)
            d = 3 if unit == "ratio" else 2
            if pd.notna(mc):
                cells.append(ci(r[col], mc, d))
                mcD.append((col, mc))
            else:
                cells.append(f(r[col], d))
        rowsD.append(cells)
    hdrD = ["Quantity", "Single read (95% MCI)", "Mean of two reads (95% MCI)"]
    no_mc = [c for c, _, _ in items if pd.isna(one(dc, ref="single").get("mcse_" + c))]

    title = ("**Table 3. Reference quality and apparent AI performance in the base cell.** A second read or "
             "adjudication lowered the error of the reference but not its bias, an independent AI model appeared "
             "less accurate than it was, and a model sharing the readers' image-level error appeared more "
             "accurate than it was.")
    foot = ("Base cell: beat CV 15%, view overestimation present, AP axis, K = 3, every read by rule A4 "
            "(detection 0.8, false rejection 0.1, thresholds 3 and 5 mm), validation studies of 200 patients read "
            "by three of four readers. Panel A: bias is reference minus T1; an adjudicated reference used the "
            "third reader's read when the first two differed by more than the tolerance. Panel B: bias is AI minus "
            "comparator; the unbiased AI had normal error with SD σ_AI around T1; the inherited-bias AI added the "
            "systematic component of one A4 read regressed on T1 in 20,000 separate patients; the shared-error "
            "AI was T1 + λ(R_img − T1) + independent error, and at λ = 0 equals the unbiased AI. Pre-specified and "
            "amendment 2 rows come from independent runs (master seeds 20260918 and 20260920). Panel C: paired "
            "comparison within each simulated study, amendment 2 run, λ = 0 cell, σ_AI 2 mm. Panel D: e_ref = "
            "e_img + e_rd, where e_img = R_img − T1 and e_rd = read − R_img; because the covariance is negative, "
            "the components do not partition the total and the ratio is not a share; MCSE by block jackknife over "
            "20 blocks of studies" + (f" (not stored for {', '.join(no_mc)})" if no_mc else "") + ". "
            "n = 2,000 simulated validation studies per cell; values are means across studies. Maximum MCSE: "
            f"panel A {max(mcA):.4f} mm (adjudicated share below 0.21 percentage points); panel B MAE "
            f"{mcB['mae']:.4f} mm, LoA width {mcB['loa_width']:.4f} mm, bias {mcB['ba_bias']:.4f} mm, ICC "
            f"{mcB['icc']:.4f}; panel C {max(mcC):.2f} percentage points; panel D "
            f"{max(v for _, v in mcD):.4f}. The LoA width MCSE is the conservative bound MCSE(upper) + "
            "MCSE(lower), because the covariance of the limits was not stored. 95% MCI = estimate ± 1.96 MCSE. "
            f"Sources: {rel(s_rq)} and {rel(s_rq2)} (panel A); {rel(s_ag)} and {rel(s_ag2)} (panel B); {rel(s_pr)} "
            f"(panel C); {rel(s_dc)} (panel D); generated by {SCRIPT}. ICC(A,1), two-way absolute-agreement "
            "single-measure intraclass correlation; LoA, limits of agreement; MAE, mean absolute error; MCI, Monte "
            "Carlo interval; MCSE, Monte Carlo standard error; R_img, A4 composite computed from the images without "
            "caliper error or reader bias.")
    return "\n\n".join([title, "*Panel A. Reference quality by design*", md_table(hdrA, rowsA, ["l", "l", "r", "r", "r"]),
                        "*Panel B. True and apparent agreement of simulated AI models*",
                        md_table(hdrB, rowsB, ["l", "r", "l", "l", "r", "r", "r", "r"]),
                        "*Panel C. Paired comparison of the inherited-bias and unbiased AI (σ_AI 2 mm)*",
                        md_table(hdrC, rowsC, ["l", "r", "r"]),
                        "*Panel D. Decomposition of reference error (amendment 2)*",
                        md_table(hdrD, rowsD, ["l", "r", "r"]), foot])


# ---------------------------------------------------------------- Table 4
TESTS = {"f_variance": "F test on variance, one-sided (pre-specified test)",
         "welch_mean": "Welch t test on mean, two-sided",
         "bf_onesided": "Brown-Forsythe test, one-sided",
         "bf": "Brown-Forsythe test, two-sided"}


def table4():
    s_pr = FULL / "E5E6_E6_precision.csv"
    s_sz1, s_sz2 = FULL / "E5E6_E6_size_check.csv", AM2 / "E5bE6b_E6b_size_check.csv"
    s_sp, s_mn, s_md = AM2 / "E5bE6b_E6b_sentinel_power.csv", AM2 / "E5bE6b_E6b_min_n.csv", \
        AM2 / "E5bE6b_E6b_min_n_distribution.csv"

    # Panel A: precision, pre-specified run, gold sets
    pr = sel(pd.read_csv(s_pr), p_beat_cv=0.15, p_ai_draft_sigma_mm=2.0, design="gold")
    cols = [("icc_mean", 3), ("icc_empirical_sd", 3), ("icc_ci_width_mean", 3),
            ("loa_hi_empirical_sd", 2), ("loa_ci_width_each_mean", 2)]
    rowsA, mcA = [], {}
    for srb in [0.0, 0.75]:
        for _, r in pr[pr.p_sigma_rb_mm == srb].sort_values("n_double").iterrows():
            cells = [f"{srb:g}", f"{int(r.n_double)}"]
            for c, d in cols:
                cells.append(ci(r[c], r["mcse_" + c], d))
                mcA[c] = max(mcA.get(c, 0), r["mcse_" + c])
            cells.append(f(r.loa_hi_emp95_range, 2))
            rowsA.append(cells)
    hdrA = ["σ_rb (mm)", "Double-read cases, n", "ICC mean (95% MCI)", "ICC empirical SD (95% MCI)",
            "ICC nominal 95% CI width (95% MCI)", "Upper LoA empirical SD (mm, 95% MCI)",
            "Upper LoA nominal 95% CI width (mm, 95% MCI)", "Upper LoA empirical 95% range (mm)"]

    # Panel B: size at w = 0
    sz1, sz2 = pd.read_csv(s_sz1), pd.read_csv(s_sz2)
    rowsB = []
    for run, df, tests in [("pre-specified", sz1, ["f_variance", "welch_mean"]),
                           ("amendment 2", sz2, ["f_variance", "welch_mean", "bf_onesided", "bf"])]:
        for t in tests:
            d = df[df.test == t]
            assert len(d) == 216, (run, t, len(d))
            lo = d.value - Z * d.mcse
            hi = d.value + Z * d.mcse
            rowsB.append([TESTS[t], run, f(d.value.min(), 3), f(d.value.median(), 3), f(d.value.max(), 3),
                          f"{int((lo > 0.05).sum())}", f"{int((hi < 0.05).sum())}"])
    hdrB = ["Test", "Run", "Minimum size", "Median size", "Maximum size", "Combinations above 0.05 (of 216)",
            "Combinations below 0.05 (of 216)"]

    # Panel C: power at base, BF one-sided with F in parentheses
    sp = sel(pd.read_csv(s_sp), p_beat_cv=0.15, p_sigma_rb_mm=0.75, p_ai_draft_sigma_mm=2.0)
    assert set(sp.cell) == {22} and sp.n.eq(2000).all()
    ns = sorted(sp.sentinel_n.unique())
    ws = sorted(sp.w_anchor.unique())
    rowsC = []
    for w in ws:
        cells = [f"{w:.1f}" + (" (size)" if w == 0 else "")]
        for n in ns:
            b = one(sp, test="bf_onesided", sentinel_n=n, w_anchor=w)
            fv = one(sp, test="f_variance", sentinel_n=n, w_anchor=w)
            cells.append(f"{f(b.value, 3)} (F {f(fv.value, 3)})")
        rowsC.append(cells)
    mcC = sp[sp.test.isin(["bf_onesided", "f_variance"])].mcse.max()
    welch = one(sp, test="welch_mean", sentinel_n=100, w_anchor=0.5)
    hdrC = ["Anchoring fraction w"] + [f"Sentinel n = {n}" for n in ns]

    # Panel D: smallest n with power >= 0.80 (strict size rule), base cell and across 54 cells
    mn = sel(pd.read_csv(s_mn), p_beat_cv=0.15, p_sigma_rb_mm=0.75, p_ai_draft_sigma_mm=2.0)
    md = pd.read_csv(s_md)
    md = md[md.rule == "strict"]
    rowsD = []
    for w in [0.1, 0.2, 0.3, 0.5]:
        cells = [f"{w:.1f}"]
        for t in ["bf_onesided", "bf"]:
            r = one(mn, test=t, w_anchor=w)
            if pd.isna(r.min_n_strict):
                cells.append(f"none; maximum power {f(r.max_power_tested, 3)}")
            else:
                cells.append(f"{int(r.min_n_strict)}; power {ci(r.power_at_min_n_strict, r.mcse_power_strict, 3)}")
        r = one(md, test="bf_onesided", w_anchor=w)
        parts = [f"{int(r[c])} at n = {c.split('_')[-1]}" for c in
                 ["cells_min_n_10", "cells_min_n_25", "cells_min_n_50", "cells_min_n_100"] if r[c] > 0]
        if r.cells_min_n_none > 0:
            parts.append(f"{int(r.cells_min_n_none)} not reached")
        cells.append("; ".join(parts))
        rF = one(mn, test="f_variance", w_anchor=w)
        cells.append("not size-valid" if pd.isna(rF.min_n_strict) else f"{int(rF.min_n_strict)}")
        rowsD.append(cells)
    hdrD = ["Anchoring fraction w", "Brown-Forsythe one-sided, base cell: smallest n; power (95% MCI)",
            "Brown-Forsythe two-sided, base cell: smallest n; power (95% MCI)",
            "Brown-Forsythe one-sided, 54 cells: number of cells by smallest n", "F test, base cell"]

    title = ("**Table 4. Precision of double-read agreement statistics and performance of sentinel tests for "
             "anchoring.** With reader bias present, more double-read cases barely improved the precision of the "
             "upper limit of agreement; the pre-specified F test exceeded its nominal size, and the one-sided "
             "Brown-Forsythe test held it and reached power 0.80 with 50 sentinel cases at w = 0.3.")
    foot = ("Panel A, pre-specified run (E6): beat CV 15%, AI draft SD 2 mm, AP axis, gold sets read by two readers per "
            "simulated study (fixed within a study, drawn afresh for each study); empirical SD is the SD of the estimate across studies; nominal CI width "
            "is the mean width of the McGraw-Wong (ICC) or Bland-Altman (upper LoA) 95% CI; the empirical 95% range "
            "of the upper LoA is 3.92 times its empirical SD across studies. Panel B: empirical rejection "
            "rate at w = 0 over 54 cells (beat CV × σ_rb × AI draft SD) × 4 sentinel sizes = 216 combinations per "
            "test; a combination is above or below 0.05 when its 95% MCI excludes 0.05. Panels C and D, amendment 2 "
            "run (E6b), base cell: beat CV 15%, σ_rb 0.75 mm, AI draft bias 1 mm and SD 2 mm; rejection rate at "
            "level 0.05 of a manual-only sentinel subset of n cases against the AI-assisted remainder of a "
            "500-case production set, where assisted reads moved a fraction w of the way towards the AI draft; "
            "values in parentheses in panel C are the pre-specified F test, whose size exceeded 0.05, so its power "
            "is not size-calibrated; the two-sided Welch test on the mean reached power " + f(welch.value, 3) + " at n = 100 and w = 0.5. Panel D: smallest n in {10, 25, 50, 100} with power at least 0.80 for a test "
            "whose size MCI did not lie above 0.05 in that cell; the true requirement lies between the listed n "
            "and the next smaller grid value, and power 0.802 (w = 0.2) and 0.809 (w = 0.5) do not clearly exceed "
            "0.80. n = 2,000 simulated studies per cell. Maximum MCSE: panel A ICC mean "
            f"{mcA['icc_mean']:.4f}, ICC empirical SD {mcA['icc_empirical_sd']:.4f}, ICC CI width "
            f"{mcA['icc_ci_width_mean']:.4f}, upper LoA empirical SD {mcA['loa_hi_empirical_sd']:.4f} mm, upper "
            f"LoA CI width {mcA['loa_ci_width_each_mean']:.4f} mm; panel B size MCSE "
            f"{max(sz1.mcse.max(), sz2.mcse.max()):.4f}; panels C and D {mcC:.4f}. 95% MCI = estimate ± 1.96 MCSE. "
            f"Sources: {rel(s_pr)} (panel A); {rel(s_sz1)} and {rel(s_sz2)} (panel B); {rel(s_sp)} (panel C); "
            f"{rel(s_mn)} and {rel(s_md)} (panel D); generated by {SCRIPT}. CI, confidence interval; ICC, "
            "intraclass correlation ICC(A,1); LoA, limit of agreement; MCI, Monte Carlo interval; MCSE, Monte "
            "Carlo standard error; σ_rb, reader bias SD.")
    return "\n\n".join([title, "*Panel A. Precision of inter-reader agreement by number of double-read cases*",
                        md_table(hdrA, rowsA, ["r", "r", "r", "r", "r", "r", "r", "r"]),
                        "*Panel B. Empirical size of sentinel tests at w = 0*",
                        md_table(hdrB, rowsB, ["l", "l", "r", "r", "r", "r", "r"]),
                        "*Panel C. Power of the one-sided Brown-Forsythe test (F test in parentheses), base cell*",
                        md_table(hdrC, rowsC),
                        "*Panel D. Smallest sentinel set with power at least 0.80*",
                        md_table(hdrD, rowsD, ["l", "l", "l", "l", "l"]), foot])


# ---------------------------------------------------------------- Table S1
def tableS1():
    src = AM2 / "face_table.csv"
    ft = pd.read_csv(src)
    blob = " ".join(ft.astype(str).values.ravel())

    def chk(*strings):
        for s in strings:
            assert s in blob, f"S1 value {s!r} not found in {src.name}"

    # Each row: item, simulated, published (with key), construct, role, judgement.
    # Numbers are copied from face_table.csv and asserted present there (chk). Judgements apply the audit
    # corrections in notes/scratch/2026-09-18-findings-face-validity.md:
    #  (1) single-read LoA is wider than published by construction (same-beat reader difference = caliper
    #      error only; width = 2 x 1.96 x sqrt(2) x sigma_cal), so it tests sigma_cal 1.0 against 0.69 mm;
    #  (2) comparisons against sources used to set the parameter are calibration checks, not validation;
    #  (3) percentages in rows b are ratios of means (mean difference / mean T1), not the E1 relative bias
    #      (mean of per-patient ratios); beat CV headline restricted to series with mean >= 3 mm.
    chk("width 5.50 mm", "4.47 to 6.65", "3.18 mm", "3.88 mm", "width 3.82 mm", "1.000")
    chk("0.937", "0.813 to 0.974", "0.935 (0.894 to 0.961)")
    chk("0.925", "0.626 to 0.989", "0.865 to 0.944", "0.258")
    chk("-1.67 mm", "-15.4%", "-3.65 mm", "-33.7%", "31% to 49%")
    chk("-1.33 mm", "-14.9%", "-2.95 mm", "-33.2%", "19% to 25%")
    chk("20.8%", "14.9%", "17.5%", "15.3%", "15.5%")
    chk("19.1%", "14% to 22%", "7% to 11%")
    chk("median 8.32", "6.99 to 9.88", "median 9.73", "8.36 to 11.35", "median 10.32", "8.86 to 11.99",
        "median 6.44", "4.91 to 8.01", "9.5 [7.2 to 12.3]")
    m = lambda s: s.replace("-", MINUS)
    rows = [
        ["Inter-reader 95% LoA width, single beat, long-axis AP, n = 50 (mm)",
         "5.50 (95% of studies 4.47 to 6.65); 3-beat mean 3.18; biplane single beat 3.88",
         "3.82 [@hauptmann2026] (TTE, core laboratory)", "close", "calibration check (σ_cal)",
         "Wider than published in 100% of studies, by construction: same-beat reads differ only by caliper error, "
         "so the width equals 2 × 1.96 × √2 × σ_cal and tests σ_cal 1.0 mm against about 0.69 mm"],
        ["Inter-reader ICC(A,1), single beat, n = 50", "0.937 (95% of studies 0.813 to 0.974)",
         "0.935 (0.894 to 0.961) [@hauptmann2026]", "close", "calibration check",
         "Matches only because a larger per-read error is offset by a wider simulated case mix; not a sharp check"],
        ["Inter-reader ICC, biplane, n = 10", "0.925 (95% of studies 0.626 to 0.989); 25.8% of studies ≤ 0.865",
         "0.865 to 0.944 [@singh2026] (TEE)", "direct", "calibration check",
         "Consistent, but n = 10 is compatible with a wide range of error models"],
        ["Single long-axis view 3-beat mean minus T1, long-axis mean underestimation 20%",
         m("-1.67 mm; -15.4% of mean T1"), "31% to 49% below the 3D maximum [@singh2026] (derived)",
         "direct (3D reference imperfect)", "comparison not used for calibration (same study as the target)",
         "Underestimates about half as much as published"],
        ["Same, long-axis mean underestimation 40% (amendment 2)", m("-3.65 mm; -33.7% of mean T1"),
         "31% to 49% [@singh2026]", "direct (3D reference imperfect)",
         "comparison not used for calibration (same study as the target)",
         "At the lower end of the published range"],
        ["Biplane average minus mean of true AP and SL, long-axis 20%", m("-1.33 mm; -14.9% of mean truth"),
         "19% to 25% below the 3D average [@singh2026]", "direct", "calibration check (target of the long-axis scale)",
         "Misses its own calibration target: overestimation, the lognormal beat-mean bias and the instrument "
         "factor offset part of the 20% mean underestimation"],
        ["Same, long-axis 40% (amendment 2)", m("-2.95 mm; -33.2% of mean truth"), "19% to 25% [@singh2026]",
         "direct", "calibration check", "Overshoots; no single long-axis scale fits both Singh comparisons"],
        ["Beat-to-beat CV of measured values, 5 beats, long-axis AP, series with mean ≥ 3 mm",
         "20.8% at beat CV 15% (true spans 14.9%); 17.5% at 10%; 15.3% at 5%",
         "15.5% [@moraldo2013] (PISA distance)", "close", "calibration check (source of beat CV)",
         "Exceeds published by about 5 points because caliper error is added to beat variation; the headline "
         "depends on the post hoc restriction to series with mean ≥ 3 mm"],
        ["Mean beat CV of measured values vs jet-area CV", "19.1% at base; 13.4% at beat CV 5%",
         "14% to 22% jet area [@wong1987]; linear equivalent about 7% to 11% (derived)", "proxy (area)",
         "calibration check", "Above the linear equivalent at every grid value"],
        ["Measured AP span, median of cohorts of 44 (95% range), long-axis 3-beat mean, long-axis 20% (mm)",
         "8.32 (6.99 to 9.88)", "9.5 (IQR 7.2 to 12.3) [@pascalxtr2020] (view not stated)", "close",
         "calibration check (source of case mix)", "Published median inside the simulated range"],
        ["Same, anchor 3-beat mean (mm)", "9.73 (8.36 to 11.35)", "9.5 [@pascalxtr2020]", "close",
         "calibration check", "Inside"],
        ["Same, A4 composite (mm)", "10.32 (8.86 to 11.99)", "9.5 [@pascalxtr2020]", "close", "calibration check",
         "Inside"],
        ["Same, long-axis 3-beat mean, long-axis 40% (mm)", "6.44 (4.91 to 8.01)", "9.5 [@pascalxtr2020]", "close",
         "calibration check", "Outside, if the published width came from a single long-axis view"],
    ]
    hdr = ["Quantity", "Simulated", "Published", "Construct match", "Role", "Judgement"]
    title = ("**Supplementary Table S1. Face validity of the base scenario against published measurements.** "
             "Most comparisons re-used the sources that set the parameters and are calibration checks, and the "
             "simulated reader and beat noise exceeded published values.")
    foot = ("Simulated values from the amendment 2 face-validity analysis (code/12_face_validity.py; 2,000 "
            "simulated reader studies for rows 1 to 3, re-simulated patients for rows 4 to 13; MCSE of the mean "
            "differences in rows 4 to 7 at most 0.013 mm and 0.11 percentage points; see the source file for the "
            "remaining MCSE). Mean anchor underestimation 4.8% throughout. Percentages in rows 4 to 7 are ratios of "
            "means (mean difference divided by mean truth), not the mean of per-patient ratios used for relative "
            "bias in Table 2. Role: a calibration check compares against a source that was used to set the "
            "parameter, so agreement is expected and is not independent validation. Published percentages for "
            "Singh were derived from reported means. Values in rows 1 and 2 use one reader pair per study with "
            "reader bias drawn per study, unlike E1 to E4. Corrections from the audit "
            "(notes/scratch/2026-09-18-findings-face-validity.md) are applied in the Role and Judgement columns. "
            f"Source: {rel(src)}; generated by {SCRIPT}. 3D, three-dimensional; ICC, intraclass correlation; "
            "IQR, interquartile range; LoA, limits of agreement; PISA, proximal isovelocity surface area; TEE, "
            "transoesophageal echocardiography; TTE, transthoracic echocardiography.")
    return "\n\n".join([title, md_table(hdr, rows, ["l"] * 6), foot])


def main():
    cfg = yaml.safe_load(CFG.read_text())
    cfg2 = yaml.safe_load(CFG2.read_text())
    doc = ["# Tables, DUO-Max simulation manuscript v02",
           f"<!-- Generated by {SCRIPT} on the login node from results/2026-09-18_full/analysis, "
           "results/2026-09-18_amend2/analysis, code/configs and notes/parameter-table-2026-09-18.md. Do not edit "
           "by hand; rerun the script. Citation keys follow drafts/manuscript-2026-09-18-v02.md. -->",
           table1(cfg, cfg2), table2(), table3(), table4(), tableS1()]
    text = "\n\n".join(doc) + "\n"
    for bad in ("–", "—"):
        assert bad not in text, f"forbidden dash {bad!r} in output"
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text.split())} words)")


if __name__ == "__main__":
    main()
