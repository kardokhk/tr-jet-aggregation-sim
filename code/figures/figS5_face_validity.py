"""Supplementary Figure S5 (final numbering): face validity of the generative model (Amendment 2).

(a) Inter-reader 95% LoA width in simulated reader studies of n = 50, by read
    type, vs Hauptmann 2026 (3.82 mm).
(b) Inter-reader ICC(A,1), n = 50 (AP width, biplane width) and n = 10
    (biplane), vs Hauptmann 2026 and Singh 2026.
(c) Single long-axis view mean minus T1 and biplane mean minus the mean of
    the true AP and SL spans (% of mean truth) vs long-axis underestimation
    scale L, vs Singh 2026 bands.
(d) CV of measured beat values (AP, long-axis view, 5 beats) vs the beat-CV
    parameter, with and without caliper error, series with a mean >= 3 mm, vs Moraldo 2013.
    Wong 1987 (14% to 22%) is a CV of jet AREA, a different construct from a linear-span CV, so it is
    not drawn on this axis (audit 2026-09-18); it is cited in the caption only.
(e) Median and IQR of AP spans (T1 and measured quantities) vs Sugiura 2021.

Input:  results/2026-09-18_amend2/analysis/face_*.csv (code/12_face_validity.py)
Output: figures/figS5_face_validity.{pdf,png,tif} and figures/figS5_face_validity_source.csv
Run:    /project/home/p201509/envs/duomax-sim/bin/python code/figures/figS5_face_validity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/users/u104629/.claude/academic/assets")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figqa  # noqa: E402
import estimator_style as es  # noqa: E402  shared text sizes (8 pt floor)
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from estimator_style import COLOR, MARKER  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "results" / "2026-09-18_amend2" / "analysis"
STEM = str(ROOT / "figures" / "figS5_face_validity")
WIDTH_MM, HEIGHT_MM = 183.0, 170.0

BAND = dict(facecolor="#d9d9d9", edgecolor="#8c8c8c", hatch="////", lw=0.5)
READS = ["single beat, same beat", "3-beat mean, same beats",
         "single beat, different beats", "3-beat mean, different beats"]
READ_LAB = ["1 beat,\nsame", "3 beats,\nsame", "1 beat,\ndifferent", "3 beats,\ndifferent"]
QTY = {"long-axis view, AP width": dict(color="#000000", marker="o", mfc="#000000", label="AP width, n = 50"),
       "long-axis view, biplane width": dict(color="#6e6e6e", marker="s", mfc="white", label="Biplane width, n = 50")}
SINGH_Q = dict(color="#6e6e6e", marker="^", mfc="#6e6e6e", label="Biplane width, n = 10")
BASE_A, BASE_L = 0.06, 0.25


def rng_bar(ax, x, y, lo, hi, **k):
    ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], ls="none", capsize=0, elinewidth=0.8,
                ms=3.2, mew=0.8, **k)


def main():
    fam = figstyle.use_print_style()
    es.use_journal_text()
    A = pd.read_csv(AN / "face_a_reader_agreement.csv")
    B = pd.read_csv(AN / "face_b_view_vs_T1.csv")
    C = pd.read_csv(AN / "face_c_beat_cv.csv")
    D = pd.read_csv(AN / "face_d_span_distribution.csv")
    src = []

    fig = plt.figure(figsize=figstyle.mm(WIDTH_MM, HEIGHT_MM), layout="constrained")
    fig.get_layout_engine().set(h_pad=2 / 72, w_pad=2 / 72, hspace=0.06, wspace=0.06)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.3, 0.9])
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    axe = fig.add_subplot(gs[2, :])

    # ---------------- (a) LoA width
    ha = A[A.design.str.startswith("Hauptmann")]
    axa.axhline(3.82, color="#000000", lw=0.8, ls=(0, (4, 2)))
    # v05 (8 pt): label above the line, left-aligned in the gap between the 3-beat and 1-beat-different points
    axa.text(0.25, 3.82 + 0.3, "Hauptmann 2026,\n3.82 mm", ha="left", va="bottom", fontsize=es.FS_MIN)
    for j, (q, st) in enumerate(QTY.items()):
        for i, rd in enumerate(READS):
            r = ha[(ha.quantity == q) & (ha.read == rd)].iloc[0]
            x = i + (-0.13 if j == 0 else 0.13)
            rng_bar(axa, x, r.loa_width_median, r.loa_width_p025, r.loa_width_p975,
                    color=st["color"], marker=st["marker"], mfc=st["mfc"], mec=st["color"])
            src.append(dict(panel="a", series=q, x=rd, y=r.loa_width_median, lo=r.loa_width_p025,
                            hi=r.loa_width_p975, what="median and 2.5-97.5 percentile of study LoA width (mm)"))
    axa.set_xticks(range(4), READ_LAB)
    axa.set_xlim(-0.5, 3.5)
    axa.set_ylim(0, 15)
    axa.set_ylabel("Inter-reader 95% LoA\nwidth (mm)")
    axa.set_xlabel("Beats per read; same or different beats for the two readers")
    axa.legend(handles=[Line2D([], [], ls="none", ms=3.2, mew=0.8, mec=s["color"], color=s["color"],
                               marker=s["marker"], mfc=s["mfc"], label=s["label"]) for s in QTY.values()],
               loc="upper left", frameon=False, handletextpad=0.3, labelspacing=0.3)

    # ---------------- (b) ICC
    axb.axhspan(0.894, 0.961, color="#d9d9d9", lw=0)
    axb.axhline(0.935, color="#000000", lw=0.8, ls=(0, (4, 2)))
    sg = A[A.design.str.startswith("Singh")]
    for j, (q, st) in enumerate(QTY.items()):
        for i, rd in enumerate(READS):
            r = ha[(ha.quantity == q) & (ha.read == rd)].iloc[0]
            x = i + (-0.2 if j == 0 else 0.0)
            rng_bar(axb, x, r.icc_median, r.icc_p025, r.icc_p975,
                    color=st["color"], marker=st["marker"], mfc=st["mfc"], mec=st["color"])
            src.append(dict(panel="b", series=q, x=rd, y=r.icc_median, lo=r.icc_p025, hi=r.icc_p975,
                            what="median and 2.5-97.5 percentile of study ICC(A,1)"))
    for i, rd in enumerate(READS):
        r = sg[sg.read == rd].iloc[0]
        rng_bar(axb, i + 0.2, r.icc_median, r.icc_p025, r.icc_p975, color=SINGH_Q["color"],
                marker=SINGH_Q["marker"], mfc=SINGH_Q["mfc"], mec=SINGH_Q["color"])
        src.append(dict(panel="b", series="long-axis view, biplane width, n=10", x=rd, y=r.icc_median,
                        lo=r.icc_p025, hi=r.icc_p975, what="median and 2.5-97.5 percentile of study ICC(A,1)"))
    # Singh 2026 inter-observer biplane ICC range 0.865 to 0.944: bracket at right edge
    axb.plot([3.72, 3.72], [0.865, 0.944], color="#000000", lw=2.4, solid_capstyle="butt", zorder=4)
    axb.text(3.72, 0.855, "Singh\n2026", ha="center", va="top", fontsize=es.FS_MIN)
    axb.set_xticks(range(4), READ_LAB)
    axb.set_xlim(-0.5, 3.95)
    axb.set_ylim(0.2, 1.0)
    axb.set_ylabel("Inter-reader ICC(A,1)")
    axb.set_xlabel("Beats per read; same or different beats for the two readers")
    axb.legend(handles=[Line2D([], [], ls="none", ms=3.2, mew=0.8, mec=s["color"], color=s["color"],
                               marker=s["marker"], mfc=s["mfc"], label=s["label"])
                        for s in list(QTY.values()) + [SINGH_Q]]
               + [Line2D([], [], color="#000000", lw=0.8, ls=(0, (4, 2)), label="Hauptmann 2026, 95% CI shaded"),
                  Line2D([], [], color="#000000", lw=2.4, label="Singh 2026 range (n = 10)")],
               loc="lower left", frameon=False, handletextpad=0.6, handlelength=2.2, ncol=2,
               columnspacing=1.0, labelspacing=0.25, borderaxespad=0.1)

    # ---------------- (c) view minus truth
    axc.axhspan(-49, -31, **BAND)
    axc.axhspan(-25, -19, facecolor="#f0f0f0", edgecolor="#8c8c8c", hatch="....", lw=0.5)
    # the bands are keyed in the legend (no text boxes over the bands, no labels crossing the lines)
    series_c = {"long-axis view mean, AP, vs T1 AP": dict(color="#000000", marker="o", mfc="#000000", ls="-",
                                                          label="Single view (AP) vs true maximal span"),
                "long-axis view biplane mean vs mean of T1 AP and SL": dict(color="#6e6e6e", marker="s",
                                                                           mfc="white", ls=(0, (4, 1.5)),
                                                                           label="Biplane vs mean of AP, SL")}
    for jit, (comp, st) in zip((-0.006, 0.006), series_c.items()):
        s = B[(B.comparison == comp) & (np.isclose(B.u_anchor, BASE_A))].sort_values("u_long")
        lab = st.pop("label")
        axc.plot(s.u_long + jit, s.pct_ratio_of_means, mec=st["color"], ms=3.2, mew=0.8, lw=0.9, label=lab, **st)
        st["label"] = lab
        for _, r in s.iterrows():
            src.append(dict(panel="c", series=comp, x=r.u_long, y=r.pct_ratio_of_means,
                            lo=r.pct_ratio_of_means - 1.96 * r.pct_ratio_of_means_mcse,
                            hi=r.pct_ratio_of_means + 1.96 * r.pct_ratio_of_means_mcse,
                            what="100*mean(view - truth)/mean(truth), u_anchor 0.06; lo/hi = 95% MCI"))
    axc.set_xticks([0.10, 0.25, 0.50], ["0.10\n(8%)", "0.25\n(20%)", "0.50\n(40%)"])
    axc.set_xlim(0.05, 0.53)
    # head room above 0 holds the legend, so it covers no data point or band
    axc.set_ylim(-52, 27)
    axc.set_yticks(np.arange(-50, 1, 10))
    axc.spines["left"].set_bounds(-52, 0)
    axc.axhline(0, color="#8c8c8c", lw=0.5)
    axc.set_xlabel("Long-axis underestimation scale L (mean underestimation)")
    axc.set_ylabel("View value minus truth\n(% of mean truth)")
    hc, lc = axc.get_legend_handles_labels()
    hc += [Patch(**BAND), Patch(facecolor="#f0f0f0", edgecolor="#8c8c8c", hatch="....", lw=0.5)]
    lc += ["Singh 2026: single plane vs 3D maximum", "Singh 2026: biplane vs 3D average"]
    axc.legend(hc, lc, loc="upper right", frameon=False, handletextpad=0.4, borderaxespad=0.1,
               handlelength=2.2, labelspacing=0.2)

    # ---------------- (d) beat CV
    axd.axhline(15.5, color="#000000", lw=0.8, ls=(0, (4, 2)))
    axd.text(31.5, 14.6, "Moraldo 2013, 15.5%", ha="right", va="top", fontsize=es.FS_MIN)
    cs = C[(C.axis == "AP") & (C.view == "long-axis view")]
    for srcname, st in (("measured (with caliper)", dict(color="#000000", marker="o", mfc="#000000", ls="-",
                                                        label="Measured beats (with caliper)")),
                        ("true beat spans (no caliper)", dict(color="#6e6e6e", marker="s", mfc="white",
                                                             ls=(0, (4, 1.5)), label="True beat spans"))):
        s = cs[cs.source == srcname].sort_values("beat_cv_param")
        lab = st.pop("label")
        axd.plot(100 * s.beat_cv_param, s.pooled_cv_ge3_pct, mec=st["color"], ms=3.2, mew=0.8, lw=0.9,
                 label=lab, **st)
        for _, r in s.iterrows():
            src.append(dict(panel="d", series=srcname, x=100 * r.beat_cv_param, y=r.pooled_cv_ge3_pct,
                            lo=r.pooled_cv_ge3_pct - 1.96 * r.pooled_cv_ge3_mcse,
                            hi=r.pooled_cv_ge3_pct + 1.96 * r.pooled_cv_ge3_mcse,
                            what="pooled CV sqrt(mean CV_i^2), 5 beats, AP long-axis view, series mean >= 3 mm; lo/hi 95% MCI"))
    axd.set_xticks([5, 10, 15, 20, 25, 30])
    axd.set_xlim(3, 32)
    axd.set_ylim(0, 36)
    axd.set_xlabel("Beat-to-beat CV parameter (%)")
    axd.set_ylabel("Pooled within-series CV (%),\nseries mean ≥ 3 mm")
    axd.legend(loc="upper left", frameon=False, handletextpad=0.3, bbox_to_anchor=(0, 0.97))

    # ---------------- (e) span distributions
    items = [("T1 AP (true span)", BASE_L, "True\nmaximal\nspan", "#000000", "D", "#000000"),
             ("anchor view 3-beat mean", BASE_L, "Anchor view\n3-beat mean", COLOR["A1"], MARKER["A1"], COLOR["A1"]),
             ("long-axis view single beat", BASE_L, "Long-axis\nview single\nbeat", "#6e6e6e", "v", "white"),
             ("long-axis view 3-beat mean", BASE_L, "Long-axis\nview 3-beat\nmean", "#6e6e6e", "s", "#6e6e6e"),
             ("A4 composite", BASE_L, "Largest view\nmean after\nreview", COLOR["A4"], MARKER["A4"], "white"),
             ("long-axis view 3-beat mean", 0.50, "Long-axis\nview 3-beat\nmean, L 0.50", "#6e6e6e", "s", "white"),
             ("A4 composite", 0.50, "Largest view\nmean after\nreview,\nL 0.50", COLOR["A4"], MARKER["A4"], "white")]
    x0 = len(items)
    axe.axhspan(7.2, 12.3, xmin=0, xmax=1, facecolor="#f0f0f0", lw=0)
    axe.axhline(9.5, color="#000000", lw=0.8, ls=(0, (4, 2)))
    axe.errorbar([x0], [9.5], yerr=[[2.3], [2.8]], ls="none", capsize=0, elinewidth=0.8, color="#000000",
                 marker="*", ms=8.0, mfc="#000000", mec="#000000")
    for i, (q, L, lab, col, mk, mf) in enumerate(items):
        r = D[(D.quantity == q) & np.isclose(D.u_anchor, BASE_A) & np.isclose(D.u_long, L)].iloc[0]
        axe.errorbar([i], [r["median"]], yerr=[[r["median"] - r.q25], [r.q75 - r["median"]]], ls="none",
                     capsize=0, elinewidth=0.8, color=col, marker=mk, ms=4.6, mew=1.2, mfc=mf, mec=col,
                     zorder=3)
        # 95% range of medians of simulated cohorts of n = 44 (thin grey bar, offset)
        axe.plot([i + 0.18, i + 0.18], [r.cohort44_median_p025, r.cohort44_median_p975], color="#8c8c8c",
                 lw=2.0, solid_capstyle="butt")
        src.append(dict(panel="e", series=f"{q}, L {L}", x=i, y=r["median"], lo=r.q25, hi=r.q75,
                        what="pooled median and IQR (mm), u_anchor 0.06; 88,000 patients"))
        src.append(dict(panel="e", series=f"{q}, L {L}, cohort n=44 median", x=i + 0.18, y=np.nan,
                        lo=r.cohort44_median_p025, hi=r.cohort44_median_p975,
                        what="2.5-97.5 percentile of medians of 2,000 simulated cohorts of 44"))
    src.append(dict(panel="e", series="Sugiura 2021 VC width", x=x0, y=9.5, lo=7.2, hi=12.3,
                    what="published median [IQR], n = 44"))
    axe.set_xticks(range(x0 + 1), [it[2] for it in items] + ["Sugiura\n2021 VC\nwidth,\nn = 44"])
    axe.set_xlim(-0.5, x0 + 0.5)
    axe.set_ylim(0, 16)
    axe.set_ylabel("AP span (mm)")
    axe.set_xlabel("Quantity (anchor scale 0.06; long-axis scale L 0.25 unless stated)")
    axe.legend(handles=[Line2D([], [], color="#000000", marker="|", ls="none", ms=9, mew=0.8,
                               label="Marker, median; thin bar, IQR"),
                        Line2D([], [], color="#8c8c8c", lw=2.0, label="95% range of medians, cohorts of 44")],
               loc="lower right", frameon=False, ncol=2, handletextpad=0.4)

    for ax in (axa, axb, axc, axd, axe):
        ax.grid(False)
    figstyle.panel_labels([axa, axb, axc, axd, axe], list("abcde"), dx=-0.17, dy=1.01, size=es.FS_LETTER)
    axe.texts[-1].set_x(-0.075)

    fig.canvas.draw()
    issues = figqa.report(fig, min_pt=es.FS_MIN)
    print("font:", fam)
    print("figqa:", issues if issues else "clean")
    paths = figstyle.save_all(fig, STEM)
    pd.DataFrame(src).to_csv(STEM + "_source.csv", index=False, float_format="%.6g")
    grey, actual = figqa.greyscale_and_downscale(STEM + ".png", WIDTH_MM)
    print(paths, grey, actual)
    print("png newer than script:", Path(STEM + ".png").stat().st_mtime > Path(__file__).stat().st_mtime)


if __name__ == "__main__":
    main()
