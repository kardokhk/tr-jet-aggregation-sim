"""Figure 5 (final numbering, unchanged in v04; amendment 2 version; v04 relabelling 2026-09-19): reference design, AI sharing image-level error, reference-programme
precision and sentinel power (E5b, E6b).

Replaces the v01 figure (v01 script and outputs kept in notes/scratch/2026-09-18-fig6_reference_v1_script.py
and notes/scratch/2026-09-18-fig6-v1/).
Inputs: results/2026-09-18_amend2/analysis/E5bE6b_*.csv (written by code/11_amend2_E5bE6b_tables.py from
results/2026-09-18_amend2/E5_cells_*.parquet and E6_cells_*.parquet; config
code/configs/amend2_2026-09-18.yaml, master seed 20260920).
Outputs: figures/fig5_reference.{pdf,png,tif}, figures/fig5_reference_source.csv,
         figures/fig5_reference_grey.png, figures/fig5_reference_actualsize.png.
Base cells: E5b beat CV 0.15, view overestimation on (lambda 0 cell for panel a); E6b beat CV 0.15,
sigma_rb 0.75 mm, AI-draft SD 2 mm. All reads use rule A4 (s_det 0.8, f_rej 0.1, t_warn 3, t_adj 5).
Colour: no estimator is compared (every read is A4), so the per-estimator Okabe-Ito colours of
code/figures/estimator_style.py are deliberately not used. One meaning per colour:
viridis ramp = sigma_AI (a only); black/grey with marker shape = reference design (b only);
black = precision series in c (sigma_rb by marker fill); grey ramp = sentinel n (d only).
The anchoring fraction is written w throughout; the Greek letter for the test level is not reused for it.
All text is at least 8 pt (8.5 pt titles, 9 pt bold panel letters; EHJ-CVI 2 mm floor, v05 2026-09-19);
mathtext subscripts are avoided because they render below the floor.
"""
import os
import sys

sys.path.insert(0, "/home/users/u104629/.claude/academic/assets")
import figstyle  # noqa: E402  (sets Agg)
import figqa  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estimator_style as es  # noqa: E402  shared text sizes (8 pt floor)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AN = os.path.join(ROOT, "results", "2026-09-18_amend2", "analysis")
STEM = os.path.join(ROOT, "figures", "fig5_reference")
WIDTH_MM, HEIGHT_MM = 183, 170
Z = 1.959964

fam = figstyle.use_print_style()

es.use_journal_text()

# ---------------------------------------------------------------- data
agr = pd.read_csv(os.path.join(AN, "E5bE6b_E5b_ai_agreement.csv"))
agr = agr[(agr.p_beat_cv == 0.15) & (agr.p_view_over)]
w0 = agr[agr.p_ai_shared_lambda == 0.0]
# Adjudicated references: tolerance on the tick, one group label "Adjudicated" under the three ticks
# (six two-line "Adjudicated" ticks do not fit the panel width).
# v05 (8 pt): the tolerance ticks are stacked ("1" over "mm") like the other two-line ticks, because
# "1 mm 2 mm 3 mm" on one line touch at 8 pt; the group label moves to a third line.
REFS = [("true", "True\nspan"), ("single", "Single\nread"), ("adj_tol1", "1\nmm"),
        ("adj_tol2", "2\nmm"), ("adj_tol3", "3\nmm"), ("mean2", "Mean\nof 2")]
SIGMAS = [1.0, 2.0, 3.0]
cmap = plt.get_cmap("viridis")
SIG_COL = {1.0: cmap(0.0), 2.0: cmap(0.45), 3.0: cmap(0.78)}
AI_STYLE = {"independent": dict(ls="-", marker="o", mfc=None, dx=-0.09),
            "inherited": dict(ls="--", marker="s", mfc="white", dx=0.09)}

src = []


def e5_series(ai, sig, m):
    g = w0[(w0.ai == ai) & (w0.sigma_ai_mm == sig) & (w0.m == m)].set_index("ref")
    y, se = [], []
    for r, _ in REFS:
        if r == "true":
            y.append(g["true"].iloc[0]); se.append(g["mcse_true"].iloc[0])
        else:
            y.append(g.loc[r, "apparent"]); se.append(g.loc[r, "mcse_apparent"])
    return np.array(y), np.array(se)


prec = pd.read_csv(os.path.join(AN, "E5bE6b_E6b_precision.csv"))
prec = prec[(prec.p_beat_cv == 0.15) & (prec.p_ai_draft_sigma_mm == 2.0)]
sent = pd.read_csv(os.path.join(AN, "E5bE6b_E6b_sentinel_power.csv"))
sent = sent[(sent.p_beat_cv == 0.15) & (sent.p_sigma_rb_mm == 0.75) & (sent.p_ai_draft_sigma_mm == 2.0)]

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=figstyle.mm(WIDTH_MM, HEIGHT_MM), layout="constrained")
assert HEIGHT_MM <= figstyle.MAX_HEIGHT_MM
axd = fig.subplot_mosaic([["a1", "a2", "b"], ["c1", "c2", "d"], ["c1", "c2", "e"]],
                         height_ratios=[1.0, 0.74, 0.30], width_ratios=[1.1, 1.1, 1.0])
fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, wspace=0.04, hspace=0.06)

# ---------------------------------------------------------------- a: apparent vs true, lambda 0
x = np.arange(len(REFS))
for key, m, ylab in (("a1", "mae", "MAE vs reference (mm)"),
                     ("a2", "loa_width", "95% LoA width (mm)")):
    ax = axd[key]
    for sig in SIGMAS:
        for ai, st in AI_STYLE.items():
            y, se = e5_series(ai, sig, m)
            ax.plot(x + st["dx"], y, ls=st["ls"], marker=st["marker"], color=SIG_COL[sig],
                    mfc=st["mfc"] or SIG_COL[sig], mec=SIG_COL[sig], ms=3.5, lw=0.9)
            for (r, _), yi, si in zip(REFS, y, se):
                src.append(dict(panel=key, quantity=m, ai=ai, lam=0.0, sigma_ai_mm=sig, reference=r,
                                value=yi, mcse=si, n_studies=2000, n_patients_per_study=200))
    ax.axvline(0.5, color="0.6", lw=0.5, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in REFS])
    # group label for the adjudicated ticks (x 2 to 4), on the second tick-label line, with a bracket
    tr = ax.get_xaxis_transform()
    ax.annotate("Adjudicated", xy=(3, 0), xycoords=tr, xytext=(0, -26.0), textcoords="offset points",
                ha="center", va="top", fontsize=es.FS_MIN, annotation_clip=False)
    ax.plot([1.65, 4.35], [0, 0], transform=tr + ScaledTranslation(0, -25.0 / 72, fig.dpi_scale_trans),
            color="0.3", lw=0.5, clip_on=False)
    ax.set_xlim(-0.5, len(REFS) - 0.5)
    ax.set_ylabel(ylab)
    ax.set_xlabel("Reference", labelpad=13)   # clears the "Adjudicated" group label
    ax.set_ylim(0, None)
axd["a1"].set_ylim(0, 3.1)
axd["a1"].set_yticks(np.arange(0, 3.01, 0.5))
h_sig = [Line2D([], [], color=SIG_COL[s], lw=0.9, label=f"AI error SD {s:g} mm") for s in SIGMAS]
h_ai = [Line2D([], [], color="0.3", ls="-", marker="o", ms=3.5, lw=0.9, label="Unbiased AI"),
        Line2D([], [], color="0.3", ls="--", marker="s", mfc="white", ms=3.5, lw=0.9,
               label="Inherited-bias AI")]
axd["a1"].legend(handles=h_sig, loc="lower right", frameon=False, handlelength=1.5)
axd["a2"].legend(handles=h_ai, loc="lower right", frameon=False, handlelength=2.2)

# ---------------------------------------------------------------- b: shared AI vs lambda (sigma_AI 2)
ax = axd["b"]
LAMS = [0.0, 0.5, 1.0]
B_SERIES = [("T1", "True MAE (vs true span)", dict(color="0.0", ls="-", marker="o", mfc="0.0", lw=1.1)),
            ("single", "Apparent, single read", dict(color="0.0", ls="--", marker="s", mfc="white", lw=0.9)),
            ("adj_tol2", "Apparent, adjudicated 2 mm", dict(color="0.45", ls=(0, (1.2, 1.2)), marker="^", mfc="0.45",
                                                     lw=0.9)),
            ("mean2", "Apparent, mean of 2 reads", dict(color="0.45", ls=(0, (5.0, 1.6, 1.2, 1.6)), marker="D", mfc="white", lw=0.9))]
sh = agr[(agr.ai == "shared") & (agr.sigma_ai_mm == 2.0) & (agr.m == "mae")]
for ref, lab, st in B_SERIES:
    ys, ses = [], []
    for lam in LAMS:
        g = sh[sh.p_ai_shared_lambda == lam]
        if ref == "T1":
            r = g.iloc[0]; ys.append(r["true"]); ses.append(r["mcse_true"])
        else:
            r = g[g.ref == ref].iloc[0]; ys.append(r["apparent"]); ses.append(r["mcse_apparent"])
    ax.errorbar(LAMS, ys, yerr=Z * np.array(ses), color=st["color"], ls=st["ls"], marker=st["marker"],
                mfc=st["mfc"], mec=st["color"], ms=3.5, lw=st["lw"], elinewidth=0.6, capsize=0, label=lab)
    for lam, yi, si in zip(LAMS, ys, ses):
        src.append(dict(panel="b", quantity="mae_true" if ref == "T1" else "mae_apparent", ai="shared",
                        lam=lam, sigma_ai_mm=2.0, reference=ref, value=yi, mcse=si, n_studies=2000,
                        n_patients_per_study=200))
ax.set_xlabel("Share of image error reproduced by AI")
ax.set_ylabel("MAE of AI (mm)")
ax.set_xticks(LAMS, ["0%", "50%", "100%"])
ax.set_xlim(-0.08, 1.08)
ax.set_ylim(1.0, 2.4)
ax.legend(loc="lower left", frameon=False, handlelength=4.0, title="AI error SD 2 mm", alignment="left")

# ---------------------------------------------------------------- c: precision
RB = {0.0: dict(color="0.0", mfc="white"), 0.75: dict(color="0.0", mfc="0.0")}
for key, sdcol, nomcol, ylab in (("c1", "icc_empirical_sd", "icc_ci_width_mean", "ICC 95% interval width"),
                                 ("c2", "loa_hi_empirical_sd", "loa_ci_width_each_mean",
                                  "Upper LoA 95% interval width (mm)")):
    ax = axd[key]
    for rb, st in RB.items():
        g = prec[prec.p_sigma_rb_mm == rb]
        u = g.drop_duplicates("n_double").sort_values("n_double")
        emp = 2 * Z * u[sdcol]
        emp_se = 2 * Z * u["mcse_" + sdcol]
        ax.errorbar(u.n_double, emp, yerr=Z * emp_se, color=st["color"], lw=0.9, marker="o", ms=3.2,
                    mfc=st["mfc"], mec=st["color"], mew=0.8, capsize=0, elinewidth=0.6)
        ax.plot(u.n_double, u[nomcol], color=st["color"], lw=0.9, ls="--", marker="^", ms=3.2,
                mfc=st["mfc"], mec=st["color"], mew=0.8)
        ov = g[g.design == "overlap"].sort_values("n_double")
        ax.plot(ov.n_double, 2 * Z * ov[sdcol], ls="none", marker="D", ms=5.5, mfc="none",
                mec=st["color"], mew=0.6)
        for _, r in u.iterrows():
            src.append(dict(panel=key, quantity=sdcol.replace("_empirical_sd", "") + "_empirical_95range",
                            sigma_rb_mm=rb, n_double=int(r.n_double), value=2 * Z * r[sdcol],
                            mcse=2 * Z * r["mcse_" + sdcol], n_studies=2000))
            src.append(dict(panel=key, quantity=nomcol, sigma_rb_mm=rb, n_double=int(r.n_double),
                            value=r[nomcol], mcse=r["mcse_" + nomcol], n_studies=2000))
        for _, r in ov.iterrows():
            src.append(dict(panel=key, quantity="overlap_design", sigma_rb_mm=rb, n_double=int(r.n_double),
                            overlap_frac=r.overlap_frac, value=2 * Z * r[sdcol], n_studies=2000))
    ax.set_xlabel("Double-read cases (n)")
    ax.set_ylabel(ylab)
    ax.set_xticks([25, 50, 100, 150, 200, 250])
    ax.set_ylim(0, None)
    ax.set_xlim(0, 270)
h_c = [Line2D([], [], color="0.0", lw=0.9, marker="o", ms=3.2, label="Empirical range"),
       Line2D([], [], color="0.0", lw=0.9, ls="--", marker="^", ms=3.2, label="Nominal CI (mean)"),
       Line2D([], [], color="0.0", ls="none", marker="o", ms=3.2, label="Reader-bias SD 0.75 mm"),
       Line2D([], [], color="0.0", ls="none", marker="o", mfc="white", ms=3.2,
              label="Reader-bias SD 0 mm"),
       Line2D([], [], color="0.0", ls="none", marker="D", mfc="none", ms=5.5, mew=0.6,
              label="Overlap design, 500-case set")]
axd["c2"].set_ylim(0, 7.2)
# v05 (8 pt): the five-entry key no longer fits beside the n = 25 points, so the two line types are keyed
# in c1 (above the curves) and the three marker types in c2
axd["c1"].legend(handles=h_c[:2], loc="upper right", frameon=False, handlelength=2.2, borderaxespad=0.2)
axd["c2"].legend(handles=h_c[2:], loc="upper right", frameon=False, handlelength=1.8, borderaxespad=0.2)

# ---------------------------------------------------------------- d: sentinel power, BF one-sided (F faint)
# Upper axes: rejection rate against w. Lower strip (still panel d): empirical size at w = 0 on its own
# 0 to 0.10 scale, because on the 0 to 1 power axis the eight size points overplot.
NS = [10, 25, 50, 100]
NS_COL = {10: "#000000", 25: "#3d3d3d", 50: "#6e6e6e", 100: "#9a9a9a"}
NS_MK = {10: "o", 25: "s", 50: "^", 100: "D"}
LV_LS, PW_LS = (0, (6, 3)), (0, (1, 2))
ax = axd["d"]
axz = axd["e"]
for n in NS:
    g = sent[(sent.sentinel_n == n) & (sent.test == "f_variance")].sort_values("w_anchor")
    ax.plot(g.w_anchor, g.value, color=NS_COL[n], ls="-", lw=0.6, alpha=0.3, marker=NS_MK[n], ms=2.4,
            zorder=1)
    for _, r in g.iterrows():
        src.append(dict(panel="d", quantity="power_f_variance", sentinel_n=n, w_anchor=r.w_anchor,
                        value=r.value, mcse=r.mcse, n_studies=2000))
    z = g[g.w_anchor == 0].iloc[0]
    axz.errorbar([NS.index(n) + 0.15], [z.value], yerr=[Z * z.mcse], color=NS_COL[n], alpha=0.35,
                 marker=NS_MK[n], ms=2.6, ls="none", elinewidth=0.6, capsize=0)
    g = sent[(sent.sentinel_n == n) & (sent.test == "bf_onesided")].sort_values("w_anchor")
    ax.errorbar(g.w_anchor, g.value, yerr=Z * g.mcse, color=NS_COL[n], ls="-", marker=NS_MK[n],
                ms=3.2, lw=1.0, elinewidth=0.6, capsize=0, zorder=3)
    z = g[g.w_anchor == 0].iloc[0]
    axz.errorbar([NS.index(n) - 0.15], [z.value], yerr=[Z * z.mcse], color=NS_COL[n], marker=NS_MK[n],
                 ms=3.2, ls="none", elinewidth=0.6, capsize=0)
    for _, r in g.iterrows():
        src.append(dict(panel="d", quantity="power_bf_onesided", sentinel_n=n, w_anchor=r.w_anchor,
                        value=r.value, mcse=r.mcse, n_studies=2000))
ax.axhline(0.05, color="0.0", lw=0.5, ls=LV_LS, zorder=0)
ax.axhline(0.80, color="0.0", lw=0.5, ls=PW_LS, zorder=0)
ax.set_xlabel("Drift towards AI draft")
ax.set_ylabel("Rejection rate")
# head room above 1.0 holds the two keys, so no reference line or data passes through them
ax.set_ylim(0, 1.62)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.spines["left"].set_bounds(0, 1.0)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.5], ["0%", "10%", "20%", "30%", "50%"])
h_d = [Line2D([], [], color=NS_COL[n], marker=NS_MK[n], ms=3.2, lw=1.0, label=f"{n}") for n in NS]
h_t = [Line2D([], [], color="0.3", ls="-", lw=1.0, label="Brown-Forsythe"),
       Line2D([], [], color="0.3", ls="-", lw=0.6, alpha=0.3, label="F test (faint)"),
       Line2D([], [], color="0.0", ls=LV_LS, lw=0.5, label="Level 0.05"),
       Line2D([], [], color="0.0", ls=PW_LS, lw=0.5, label="Power 0.80")]
leg = ax.legend(handles=h_d, loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, handlelength=1.6,
                ncol=4, columnspacing=0.8, handletextpad=0.4, title="Sentinel n", alignment="left",
                borderaxespad=0.0)
ax.add_artist(leg)
ax.legend(handles=h_t, loc="upper left", bbox_to_anchor=(0.0, 0.83), frameon=False, handlelength=2.2,
          ncol=2, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)
axz.axhline(0.05, color="0.0", lw=0.5, ls=LV_LS, zorder=0)
axz.set_xticks(range(len(NS)))
axz.set_xticklabels([str(n) for n in NS])
axz.set_xlim(-0.5, len(NS) - 0.5)
axz.set_ylim(0, 0.10)
axz.set_yticks([0, 0.05, 0.10])
axz.set_xlabel("Sentinel n (size at no drift)")
axz.set_ylabel("Size")

figstyle.panel_labels([axd[k] for k in ("a1", "b", "c1", "d")], letters=["a", "b", "c", "d"],
                      dx=-0.28, dy=1.02, size=es.FS_LETTER)

fig.canvas.draw()
problems = figqa.report(fig, min_pt=es.FS_MIN)
paths = figstyle.save_all(fig, STEM)
pd.DataFrame(src).to_csv(STEM + "_source.csv", index=False)
grey, actual = figqa.greyscale_and_downscale(STEM + ".png", WIDTH_MM)
assert os.path.getmtime(STEM + ".png") > os.path.getmtime(os.path.abspath(__file__))
print("font:", fam)
print("saved:", paths, grey, actual)
print("figqa:", *problems, sep="\n  ")
