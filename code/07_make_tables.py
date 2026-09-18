"""Build manuscript Tables 1 to 4 as markdown.

Inputs (read only):
  code/configs/base.yaml                               parameter values and grids
  notes/parameter-table-2026-09-18.md                  construct match and sources (encoded in table1())
  results/2026-09-18_full/analysis/E1E3_e1_headline.csv
  results/2026-09-18_full/analysis/E5E6_E5_reference_quality.csv
  results/2026-09-18_full/analysis/E5E6_E5_ai_agreement.csv
  results/2026-09-18_full/analysis/E5E6_E6_precision.csv
  results/2026-09-18_full/analysis/E5E6_E6_sentinel_power.csv

Output: drafts/tables-2026-09-18-v01.md (generated; do not hand edit).

Run from the project root:
  /project/home/p201509/envs/duomax-sim/bin/python code/07_make_tables.py

No random numbers. 95% Monte Carlo interval (MCI) = estimate +/- 1.96 MCSE.
"""
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "results" / "2026-09-18_full" / "analysis"
CFG = ROOT / "code" / "configs" / "base.yaml"
OUT = ROOT / "drafts" / "tables-2026-09-18-v01.md"
Z = 1.96
MINUS = "−"

# Base cells (protocol base case; notes/scratch/2026-09-18-findings-*.md)
E1_BASE = dict(K=3, beat_cv=0.15, rho=0.3, N_beats=3, view_over=True)
E5_BASE = dict(p_beat_cv=0.15, p_view_over=True)
E6_BASE = dict(p_beat_cv=0.15, p_sigma_rb_mm=0.75, p_ai_draft_sigma_mm=2.0)


# ---------------------------------------------------------------- formatting
def f(x, d):
    # Round half away from zero on the shortest decimal repr, so that values such as
    # 0.0715 (a proportion of 2,000 studies) print as 0.072, not by binary float rounding.
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{q:.{d}f}"
    if s.startswith("-"):
        s = MINUS + s[1:]
    if s in (MINUS + f"{0:.{d}f}",):
        s = s[1:]
    return s


def ci(est, mcse, d):
    return f"{f(est, d)} ({f(est - Z * mcse, d)} to {f(est + Z * mcse, d)})"


def ci_lohi(est, lo, hi, d):
    return f"{f(est, d)} ({f(lo, d)} to {f(hi, d)})"


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


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def listfmt(vals, fn):
    return ", ".join(fn(v) for v in vals)


# ---------------------------------------------------------------- Table 1
# Construct match and source keys from notes/parameter-table-2026-09-18.md,
# mapped to the citation keys used in drafts/manuscript-2026-09-18-v01.md.
# "assumed" where the parameter table gives no source or only a source that
# the draft does not cite (Mascherbauer 2005 in-vitro proxy; Sumida 2003).
def table1(cfg):
    P = cfg["parameters"]
    e1_levels = cfg["experiments"]["E1"]["vary"][0]["levels"]
    u_base, u_low = e1_levels[0]["u_scale"], e1_levels[1]["u_scale"]
    v = lambda k: P[k]["value"]
    g = lambda k: P[k].get("grid")
    mm = lambda x: f"{x:g} mm"
    rows = []

    def add(group, label, base, grid, construct, source):
        rows.append([group, label, base, grid, construct, source])

    add("Case mix", "Median true AP span (T1)", mm(v("S_median_mm")), "fixed", "close",
        "[@pascalxtr2020; @fcvm2024renal]")
    add("Case mix", "Log-SD of true AP span", f"{v('S_log_sd'):.2f}", "fixed", "close",
        "[@pascalxtr2020] (derived from IQR)")
    add("Geometry", "Mean SL/AP ratio", f"{v('r_mean'):.2f}",
        listfmt(g("r_mean"), lambda x: f"{x:.2f}") + " (E4)", "close", "[@song2011; @singh2026]")
    add("Geometry", "SL/AP ratio support", f"{v('r_lower'):.1f} to {v('r_upper'):.1f}", "fixed", "none", "assumed")
    add("Geometry", "SL/AP ratio beta concentration", f"{v('r_concentration'):g}", "fixed", "none", "assumed")
    add("Views", "Underestimation scale, anchor view", f"{u_base[0]:.2f}",
        f"{u_low[0]:.2f} (low scenario)", "none", "assumed")
    add("Views", "Underestimation scale, views 2 to 4", listfmt(u_base[1:], lambda x: f"{x:.2f}"),
        listfmt(u_low[1:], lambda x: f"{x:.2f}") + " (low scenario)", "close",
        "[@singh2026; @donal2024]")
    add("Views", "Underestimation cap", f"{v('u_max'):.1f}", "fixed", "none", "assumed")
    add("Views", "Probability of overestimation, views 1 to 4",
        listfmt(v("p_over"), lambda x: f"{x:.2f}"), "on, off", "none", "assumed")
    add("Views", "Overestimation magnitude, median", pct(v("o_median")), "fixed", "none", "assumed")
    add("Views", "Overestimation magnitude, log-SD", f"{v('o_log_sd'):.1f}", "fixed", "none", "assumed")
    add("Views", "Inter-view correlation of error drivers", f"{v('rho'):.1f}",
        listfmt(g("rho"), lambda x: f"{x:.1f}"), "none", "assumed")
    add("Beats", "Beat-to-beat CV", pct(v("beat_cv")),
        listfmt(g("beat_cv"), lambda x: pct(x)), "close", "[@moraldo2013; @wong1987]")
    add("Beats", "RR CV, sinus rhythm", pct(v("rr_cv_sinus")), "fixed", "none", "assumed")
    add("Beats", "RR CV, AF", pct(v("rr_cv_af")),
        listfmt(g("rr_cv_af"), lambda x: pct(x)) + " (E2)", "none", "assumed")
    add("Beats", "Slope of log span on log preceding RR", f"{v('beta_rr'):.1f}", "fixed", "none", "assumed")
    add("Beats", "Slope of log span on log RR ratio", f"{v('beta_drr'):.1f}", "fixed", "none", "assumed")
    add("Beats", "Additional beat CV in AF", pct(v("af_extra_cv")), "fixed", "none", "assumed")
    add("Beats", "Available beats, retrospective (truncated Poisson rate; cap)",
        f"{v('avail_lambda'):g}; {v('max_beats_retro')}", "fixed", "none", "assumed")
    add("Beats", "Beat cap, prospective", f"{v('max_beats_prosp')}", "fixed", "none", "assumed")
    add("Reader", "Caliper SD per beat", mm(v("sigma_cal_mm")), "fixed", "close",
        "[@hauptmann2026; @singh2026]")
    add("Reader", "Reader bias SD", mm(v("sigma_rb_mm")),
        listfmt(g("sigma_rb_mm"), lambda x: f"{x:g}") + " mm (E6)", "close",
        "[@singh2026; @alexander2022]")
    add("Instrument", "Instrument factor SD (log)", f"{v('g_log_sd'):.2f}", "fixed", "proxy", "[@fan1994]")
    add("AI model", "AI error SD (E5)", mm(v("sigma_ai_mm")),
        listfmt(g("sigma_ai_mm"), lambda x: f"{x:g}") + " mm", "none", "assumed")
    add("AI model", "AI draft error SD (E6)", mm(v("ai_draft_sigma_mm")),
        listfmt(g("ai_draft_sigma_mm"), lambda x: f"{x:g}") + " mm", "none", "assumed")
    add("AI model", "AI draft bias (E6)", mm(v("ai_draft_bias_mm")), "fixed", "none", "assumed")

    n_assumed = sum(r[5] == "assumed" for r in rows)
    title = (f"**Table 1. Parameters of the data-generating mechanism: no source measured TEE colour-jet "
             f"span directly, and {n_assumed} of {len(rows)} parameters had no empirical source.**")
    body = md_table(["Component", "Parameter", "Base value", "Grid", "Construct match", "Source"], rows,
                    align=["l", "l", "l", "l", "l", "l"])
    foot = ("Construct match: close, vena contracta width or coaptation gap; proxy, jet area or other "
            "area-based measure; none, no empirical source. Grid values were varied only in the experiments "
            "named in parentheses. The two underestimation scenarios were run in E1, E3 and E4; the inter-view "
            "correlation grid in E1; overestimation on and off in E1, E3 and E5; the beat CV grid in every "
            "experiment. The overestimation probability has a weak in-vitro proxy (Mascherbauer 2005, "
            "10.1016/j.echo.2005.03.021) and the RR slope a direction-only aortic source (Sumida 2003, "
            "10.1016/s0894-7317(03)00275-x); neither is cited in the draft, so both are marked assumed. "
            "Not a Monte Carlo table: no n per cell or MCSE applies. Source: code/configs/base.yaml "
            f"({cfg['meta']['config_version']}) and notes/parameter-table-2026-09-18.md. "
            "AF, atrial fibrillation; AP, anteroposterior; CV, coefficient of variation; RR, interval between "
            "successive R waves; SL, septolateral; TEE, transoesophageal echocardiography.")
    return "\n\n".join([title, body, foot])


# ---------------------------------------------------------------- Table 2
EST = {"A1": "A1, anchor-view mean", "A2": "A2, mean of view means", "A3": "A3, maximum of view means",
       "A4": "A4, composite adjudication rule", "A5": "A5, median of view means", "A6": "A6, index beat",
       "A7": "A7, oracle offset-corrected mean"}


def table2():
    src = "E1E3_e1_headline.csv"
    h = pd.read_csv(AN / src)
    h = sel(h, **E1_BASE)
    assert len(h) == 2 * 7 * 2 * 2, len(h)
    rows = []
    for u, lab in [("base", "Long-axis underestimation 20% (base scenario)"),
                   ("low", "Long-axis underestimation 8% (low scenario)")]:
        rows.append([f"**{lab}**", "", "", "", "", "", "", ""])
        for e in EST:
            cells = [EST[e]]
            for ax in ["AP", "SL"]:
                r = sel(h, u=u, estimator=e, axis=ax, estimand="T1").iloc[0]
                cells += [ci_lohi(r.bias, r.bias_lo95, r.bias_hi95, 2), f(r.relbias_pct, 1),
                          ci_lohi(r.rmse, r.rmse_lo95, r.rmse_hi95, 2)]
            r2 = sel(h, u=u, estimator=e, axis="AP", estimand="T2").iloc[0]
            cells.append(ci_lohi(r2.bias, r2.bias_lo95, r2.bias_hi95, 2))
            rows.append(cells)
    hdr = ["Estimator", "AP bias vs T1 (mm, 95% MCI)", "AP relative bias vs T1 (%)", "AP RMSE vs T1 (mm, 95% MCI)",
           "SL bias vs T1 (mm, 95% MCI)", "SL relative bias vs T1 (%)", "SL RMSE vs T1 (mm, 95% MCI)",
           "AP bias vs T2 (mm, 95% MCI)"]
    title = ("**Table 2. In the base case, the maximum-type rules (A3, A4) overestimated and the mean of view "
             "means (A2) underestimated the true maximal span, and RMSE differed little between multi-beat "
             "rules.**")
    mx = lambda c: h[c].max()
    foot = ("Base case: K = 3 views, N = 3 beats per view, beat CV 15%, overestimation present, inter-view "
            "correlation 0.3. Bias is the mean of estimate minus estimand; relative bias is the mean of "
            "per-patient ratios minus 1. T1, true maximal span; T2, expected anchor-view span under reference "
            "instrument settings. A7 uses the true view offsets and cannot be deployed. n = 100,000 simulated "
            f"patients per cell. Maximum MCSE across rows shown: bias {mx('bias_mcse'):.4f} mm, relative bias "
            f"{mx('relbias_pct_mcse'):.3f} percentage points, RMSE {mx('rmse_mcse'):.4f} mm. MCI, Monte Carlo "
            "interval (estimate plus or minus 1.96 MCSE); RMSE, root-mean-square error. "
            f"Source: results/2026-09-18_full/analysis/{src}.")
    return "\n\n".join([title, md_table(hdr, rows), foot])


# ---------------------------------------------------------------- Table 3
REF = {"single": "Single read", "mean2": "Mean of two reads", "adj_tol1": "Adjudicated, tolerance 1 mm",
       "adj_tol2": "Adjudicated, tolerance 2 mm", "adj_tol3": "Adjudicated, tolerance 3 mm"}


def table3():
    s_rq, s_ag = "E5E6_E5_reference_quality.csv", "E5E6_E5_ai_agreement.csv"
    rq = sel(pd.read_csv(AN / s_rq), **E5_BASE)
    ag = sel(pd.read_csv(AN / s_ag), **E5_BASE)
    assert rq.n.eq(2000).all()

    rowsA = []
    for ref, lab in REF.items():
        g = rq[rq.ref == ref].set_index("metric")
        b, m = g.loc["ref_ba_bias_vs_T1"], g.loc["ref_mae_vs_T1"]
        if "p_adjudicated" in g.index:
            p = g.loc["p_adjudicated"]
            padj = ci(100 * p.value, 100 * p.mcse, 1)
        else:
            padj = "not applicable"
        rowsA.append([lab, ci(b.value, b.mcse, 2), ci(m.value, m.mcse, 2), padj])
    hdrA = ["Reference design", "Bias vs T1 (mm, 95% MCI)", "MAE vs T1 (mm, 95% MCI)", "Adjudicated (%, 95% MCI)"]

    comps = [("true", None, "T1 (true)"), ("apparent", "single", "Single read"),
             ("apparent", "adj_tol2", "Adjudicated, 2 mm"), ("apparent", "mean2", "Mean of two reads")]
    metrics = [("mae", 2), ("loa_width", 2), ("icc", 3), ("ba_bias", 2)]
    rowsB = []
    shown = []
    for ai, ailab in [("independent", "Unbiased"), ("inherited", "Inherited bias")]:
        for sg in [1.0, 2.0, 3.0]:
            for kind, ref, clab in comps:
                refq = ref or "single"  # true value is identical across references
                cells = [ailab, f"{sg:g}", clab]
                for m, d in metrics:
                    r = sel(ag, ai=ai, sigma_ai_mm=sg, ref=refq, m=m).iloc[0]
                    cells.append(ci(r[kind], r["mcse_" + kind], d))
                    shown.append((m, r["mcse_" + kind]))
                rowsB.append(cells)
    hdrB = ["AI model", "σ_AI (mm)", "Compared with", "MAE (mm, 95% MCI)", "95% LoA width (mm, 95% MCI)",
            "ICC(A,1) (95% MCI)", "Bland-Altman bias (mm, 95% MCI)"]
    mx = {m: max(v for mm, v in shown if mm == m) for m, _ in metrics}
    mxA = rq.groupby("metric").mcse.max()

    title = ("**Table 3. A second read or adjudication lowered the error of the reference but not its bias, and "
             "a protocol-built reference inflated the apparent error of an AI model and hid a bias the model "
             "shared with the protocol.**")
    foot = ("Base cell: beat CV 15%, view overestimation present, AP axis, every read by rule A4 with K = 3. "
            "Panel A gives the quality of each reference against the true maximal span T1; bias is reference "
            "minus T1. Panel B gives the agreement of a simulated AI model with T1 (true) and with three "
            "reference designs (apparent); bias is AI minus comparator. The unbiased AI had normal error with SD "
            "σ_AI around T1; the inherited-bias AI added the systematic component of one A4 read regressed on "
            "T1 in 20,000 separate patients. n = 2,000 simulated validation studies of 200 patients per cell; "
            "values are means across studies. Maximum MCSE, panel A: bias "
            f"{mxA['ref_ba_bias_vs_T1']:.4f} mm, MAE {mxA['ref_mae_vs_T1']:.4f} mm, adjudicated "
            f"{100 * mxA['p_adjudicated']:.2f} percentage points; panel B: MAE {mx['mae']:.4f} mm, LoA width "
            f"{mx['loa_width']:.4f} mm, ICC {mx['icc']:.4f}, Bland-Altman bias {mx['ba_bias']:.4f} mm. The LoA "
            "width MCSE is the conservative bound MCSE(upper) plus MCSE(lower), because the covariance of the "
            "limits was not stored. ICC(A,1), two-way absolute-agreement single-measure intraclass correlation; "
            "LoA, limits of agreement; MAE, mean absolute error; MCI, Monte Carlo interval (estimate plus or "
            f"minus 1.96 MCSE). Sources: results/2026-09-18_full/analysis/{s_rq} (panel A) and {s_ag} (panel B).")
    return "\n\n".join([title, "*Panel A. Reference quality by design*", md_table(hdrA, rowsA),
                        "*Panel B. True and apparent agreement of an AI model*",
                        md_table(hdrB, rowsB, align=["l", "r", "l", "r", "r", "r", "r"]), foot])


# ---------------------------------------------------------------- Table 4
def table4():
    s_pr, s_sp = "E5E6_E6_precision.csv", "E5E6_E6_sentinel_power.csv"
    pr = pd.read_csv(AN / s_pr)
    pr = sel(pr, p_beat_cv=0.15, p_ai_draft_sigma_mm=2.0)
    rowsA, mcA = [], {}
    cols = [("icc_mean", 3), ("icc_empirical_sd", 3), ("icc_ci_width_mean", 3),
            ("loa_hi_empirical_sd", 2), ("loa_ci_width_each_mean", 2)]
    for srb in [0.0, 0.75]:
        g = pr[pr.p_sigma_rb_mm == srb]
        gold = g[g.design == "gold"]
        ov = g[(g.design == "overlap") & (~g.n_double.isin(gold.n_double))]
        for _, r in pd.concat([gold, ov]).sort_values("n_double").iterrows():
            design = "gold set" if r.design == "gold" else f"overlap {100 * r.overlap_frac:.0f}% of 500"
            cells = [f"{srb:g}", f"{int(r.n_double)} ({design})"]
            for c, d in cols:
                cells.append(ci(r[c], r["mcse_" + c], d))
                mcA[c] = max(mcA.get(c, 0), r["mcse_" + c])
            rowsA.append(cells)
    hdrA = ["σ_rb (mm)", "Double-read cases, n (design)", "ICC mean (95% MCI)", "ICC empirical SD (95% MCI)",
            "ICC nominal 95% CI width (95% MCI)", "Upper LoA empirical SD (mm, 95% MCI)",
            "Upper LoA nominal 95% CI width (mm, 95% MCI)"]

    sp = sel(pd.read_csv(AN / s_sp), **E6_BASE)
    assert sp.n.eq(2000).all()
    ns = sorted(sp.sentinel_n.unique())
    alphas = sorted(sp.anchor_alpha.unique())
    panels = []
    mcB = 0
    for test, lab in [("f_variance", "one-sided F test on the variance"), ("welch_mean", "two-sided Welch t test on the mean")]:
        rows = []
        for a in alphas:
            cells = [f"{a:.1f} (size)" if a == 0 else f"{a:.1f}"]
            for n in ns:
                r = sel(sp, test=test, sentinel_n=n, anchor_alpha=a).iloc[0]
                cells.append(ci(r.value, r.mcse, 3))
                mcB = max(mcB, r.mcse)
            rows.append(cells)
        panels.append((lab, md_table(["Anchoring fraction α"] + [f"n = {n} (95% MCI)" for n in ns], rows)))

    title = ("**Table 4. With reader bias present, more double-read cases improved the precision of the upper "
             "limit of agreement little, and a manual-only sentinel set detected anchoring through the variance "
             "test more readily than through the mean, although the variance test exceeded its nominal size.**")
    foot = ("Base cell: beat CV 15%, AI draft SD 2 mm, AP axis; panels B and C use reader bias SD σ_rb 0.75 mm. "
            "Panel A: two fixed readers per study; empirical SD is the SD of the estimate across studies; "
            "nominal CI width is the mean width of the McGraw-Wong (ICC) or Bland-Altman (upper LoA) 95% CI. "
            "Overlap designs double-read a fraction of a 500-case production set; overlap 10% and 20% are "
            "identical to gold sets of 50 and 100 cases and are not shown separately. Panels B and C: rejection "
            "rate at level 0.05 for a manual-only sentinel subset of n cases against the AI-assisted remainder "
            "of a 500-case production set, where assisted reads moved a fraction α towards an AI draft with "
            "bias 1 mm; α = 0 gives the empirical size. n = 2,000 simulated studies per cell. Maximum MCSE, "
            f"panel A: ICC mean {mcA['icc_mean']:.4f}, ICC empirical SD {mcA['icc_empirical_sd']:.4f}, ICC CI "
            f"width {mcA['icc_ci_width_mean']:.4f}, upper LoA empirical SD {mcA['loa_hi_empirical_sd']:.4f} mm, "
            f"upper LoA CI width {mcA['loa_ci_width_each_mean']:.4f} mm; panels B and C: {mcB:.4f}. "
            "CI, confidence interval; ICC, intraclass correlation ICC(A,1); LoA, limit of agreement; MCI, Monte "
            "Carlo interval (estimate plus or minus 1.96 MCSE). "
            f"Sources: results/2026-09-18_full/analysis/{s_pr} (panel A) and {s_sp} (panels B and C).")
    parts = [title, "*Panel A. Precision of inter-reader agreement by number of double-read cases*",
             md_table(hdrA, rowsA, align=["r", "l", "r", "r", "r", "r", "r"])]
    for (lab, tab), letter in zip(panels, "BC"):
        parts += [f"*Panel {letter}. Sentinel rejection rate, {lab}*", tab]
    parts.append(foot)
    return "\n\n".join(parts)


def main():
    cfg = yaml.safe_load(CFG.read_text())
    doc = ["# Tables, DUO-Max simulation manuscript",
           "<!-- Generated by code/07_make_tables.py from results/2026-09-18_full/analysis/*.csv, "
           "code/configs/base.yaml and notes/parameter-table-2026-09-18.md. Do not edit by hand; rerun the "
           "script. Citation keys follow drafts/manuscript-2026-09-18-v01.md. -->",
           table1(cfg), table2(), table3(), table4()]
    text = "\n\n".join(doc) + "\n"
    for bad in ("–", "—"):
        assert bad not in text, f"forbidden dash {bad!r} in output"
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text.split())} words)")


if __name__ == "__main__":
    main()
