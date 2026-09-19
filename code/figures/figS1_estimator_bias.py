#!/usr/bin/env python
"""Supplementary Figure S1 (final numbering; formerly Figure 2): estimator bias and RMSE versus the true maximal span T1 (experiment E1).

Input : results/2026-09-18_full/analysis/E1E3_e1_base_slice.csv, E1E3_e1_a3_crossing.csv,
        E1E3_e1_a4_variants.csv (from code/02_analyse_e1e3.py).
Output: figures/figS1_estimator_bias.{pdf,png,tif}, figures/figS1_estimator_bias_source.csv.
AP axis, estimand T1, rho 0.3, N_beats 3, window 15%, sinus, prospective.
"""
from __future__ import annotations

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

from estimator_style import COLOR, EST, LABEL, kw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "results" / "2026-09-18_full" / "analysis"
STEM = ROOT / "figures" / "figS1_estimator_bias"
WIDTH_MM, HEIGHT_MM = 183.0, 170.0
AXIS = "AP"
SCEN = [("base", True), ("base", False), ("low", True), ("low", False)]
SCEN_TITLE = {("base", True): "Underestimation 20%\noverestimation on",
              ("base", False): "Underestimation 20%\noverestimation off",
              ("low", True): "Underestimation 8%\noverestimation on",
              ("low", False): "Underestimation 8%\noverestimation off"}
K_LS = {2: ":", 3: "--", 4: "-"}
K_MFC = {2: "white", 3: "white", 4: COLOR["A3"]}


def main():
    fam = figstyle.use_print_style()
    es.use_journal_text()
    bs = pd.read_csv(AN / "E1E3_e1_base_slice.csv")
    bs = bs[(bs.axis == AXIS) & (bs.estimand == "T1")]
    cr = pd.read_csv(AN / "E1E3_e1_a3_crossing.csv")
    cr = cr[cr.axis == AXIS]
    a4 = pd.read_csv(AN / "E1E3_e1_a4_variants.csv")
    a4 = a4[(a4.axis == AXIS) & (a4.estimand == "T1") & (a4.K == 3) & (a4.beat_cv == 0.15)]
    src = []

    mosaic = [["a", "b", "c", "d"], ["e", "f", "g", "h"], ["i", "j", "k", "l"]]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=figstyle.mm(WIDTH_MM, HEIGHT_MM),
                                  layout="constrained")
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.04, wspace=0.03)

    # ---- a-d: bias vs beat CV, K = 3, all estimators
    for key, (u, ov) in zip("abcd", SCEN):
        ax = axd[key]
        ax.axhline(0, color="0.6", lw=0.5, zorder=0)
        for e in EST:
            s = bs[(bs.u == u) & (bs.view_over == ov) & (bs.K == 3) & (bs.estimator == e)].sort_values("beat_cv")
            ax.plot(100 * s.beat_cv, s.bias, **kw(e))
            src.append(s.assign(panel=key))
        ax.set_title(SCEN_TITLE[(u, ov)], fontsize=es.FS_TITLE)
        ax.set_xticks([5, 10, 15, 20, 25, 30])
        ax.set_ylim(-1.9, 2.2)
        ax.set_xlabel("Beat-to-beat CV (%)")
        if key == "a":
            ax.set_ylabel("Bias vs true maximal span,\nK = 3 (mm)")
        else:
            ax.tick_params(labelleft=False)

    # ---- e-h: A3 bias vs beat CV by K, crossing marked
    end_labels = {}   # panel -> right-end K labels, pushed apart after layout (8 pt labels, v05)
    for key, (u, ov) in zip("efgh", SCEN):
        ax = axd[key]
        ax.axhline(0, color="0.6", lw=0.5, zorder=0)
        crossings = []
        for K in (2, 3, 4):
            s = bs[(bs.u == u) & (bs.view_over == ov) & (bs.K == K) & (bs.estimator == "A3")].sort_values("beat_cv")
            ax.plot(100 * s.beat_cv, s.bias, **kw("A3", ls=K_LS[K], mfc=K_MFC[K]))
            end_labels.setdefault(key, []).append(
                ax.annotate(f"K = {K}", (100 * s.beat_cv.iloc[-1], s.bias.iloc[-1]), xytext=(4.5, 0),
                            textcoords="offset points", va="center", ha="left", fontsize=es.FS_MIN,
                            color=COLOR["A3"]))
            src.append(s.assign(panel=key))
            c = cr[(cr.u == u) & (cr.view_over == ov) & (cr.K == K)].iloc[0]
            if c.status == "crossed":
                xc = 100 * c.cv_cross
                ax.plot(xc, 0, marker="|", ms=7, mew=1.2, color="black", zorder=5)
                ax.plot([xc, xc], [-0.07, -0.66], color="0.55", lw=0.5, zorder=0)
                crossings.append((xc, K))
        # crossing labels: centred under the leader; a label within 3 CV points of the previous crossing
        # is set to the right of its leader instead (at 8 pt the digits would touch)
        crossings.sort()
        for i, (xc, K) in enumerate(crossings):
            near_prev = i > 0 and xc - crossings[i - 1][0] < 3
            near_next = i + 1 < len(crossings) and crossings[i + 1][0] - xc < 3
            ha, dx = ("left", 1.5) if near_prev else ("center", 0)
            ax.annotate(f"{K}", (xc, -0.68), xytext=(dx, 0), textcoords="offset points", ha=ha, va="top",
                        fontsize=es.FS_MIN, color="black")
        s1 = bs[(bs.u == u) & (bs.view_over == ov) & (bs.K == 1) & (bs.estimator == "A1")].sort_values("beat_cv")
        ax.plot(100 * s1.beat_cv, s1.bias, color=COLOR["A1"], lw=0.7, ls="-", marker=None)
        end_labels[key].append(
            ax.annotate("K = 1", (100 * s1.beat_cv.iloc[-1], s1.bias.iloc[-1]), xytext=(4.5, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=es.FS_MIN,
                        color=COLOR["A1"]))
        src.append(s1.assign(panel=key))
        ax.set_xlim(3, 38)
        ax.set_xticks([5, 10, 20, 30])   # six ticks touch at 8 pt in these narrower data areas
        ax.set_ylim(-0.85, 2.7)
        ax.set_title(SCEN_TITLE[(u, ov)], fontsize=es.FS_TITLE)
        ax.set_xlabel("Beat-to-beat CV (%)")
        if key == "e":
            ax.set_ylabel("Largest view mean: bias vs\ntrue maximal span (mm)")
        else:
            ax.tick_params(labelleft=False)

    # ---- i: bias vs K at CV 15%, base scenario; j: RMSE vs K; k: RMSE vs CV (K 3)
    b0 = bs[(bs.u == "base") & (bs.view_over)]
    for key, metric, xvar, ylab in (("i", "bias", "K", "Bias vs true maximal span,\nCV 15% (mm)"),
                                    ("j", "rmse", "K", "RMSE vs true maximal span,\nCV 15% (mm)"),
                                    ("k", "rmse", "beat_cv", "RMSE vs true maximal span,\nK = 3 (mm)")):
        ax = axd[key]
        if metric == "bias":
            ax.axhline(0, color="0.6", lw=0.5, zorder=0)
        for e in EST:
            if xvar == "K":
                s = b0[(b0.beat_cv == 0.15) & (b0.estimator == e)].sort_values("K")
                x = s.K
            else:
                s = b0[(b0.K == 3) & (b0.estimator == e)].sort_values("beat_cv")
                x = 100 * s.beat_cv
            ax.plot(x, s[metric], **kw(e))
            src.append(s.assign(panel=key))
        if xvar == "K":
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xlabel("Number of views, K")
        else:
            ax.set_xticks([5, 10, 15, 20, 25, 30])
            ax.set_xlabel("Beat-to-beat CV (%)")
        ax.set_ylabel(ylab)
    axd["j"].set_ylim(1.5, 2.6)
    axd["k"].set_ylim(1.4, 4.2)
    for k in "ijkl":
        axd[k].set_title("Underestimation 20%\noverestimation on" if k != "l" else
                         "K = 3, CV 15%\noverestimation on", fontsize=es.FS_TITLE)

    # ---- l: A4 bias vs s_det by f_rej, base cell, both u scenarios (overestimation on)
    ax = axd["l"]
    ax.axhline(0, color="0.6", lw=0.5, zorder=0)
    FR_LS = {0.0: "-", 0.1: "--", 0.3: ":"}
    for u, mfc in (("base", COLOR["A4"]), ("low", "white")):
        for fr in (0.0, 0.1, 0.3):
            s = a4[(a4.u == u) & (a4.view_over) & (a4.f_rej == fr)].sort_values("s_det")
            ax.plot(s.s_det, s.bias, **kw("A4", ls=FR_LS[fr], mfc=mfc))
            src.append(s.assign(panel="l", estimator="A4"))
        # scenario label above the left end of its series (v05: frees the space under the lower series
        # for the f_rej key at 8 pt)
        s = a4[(a4.u == u) & (a4.view_over) & (a4.s_det == 0.0)]
        ax.annotate("Underestimation 20%" if u == "base" else "Underestimation 8%", (0.0, s.bias.max()),
                    xytext=(0, 5), textcoords="offset points", ha="left", va="bottom", fontsize=es.FS_MIN)
    ax.set_xticks([0, 0.5, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.5", "0.8", "1"])
    ax.set_xlabel("Detection sensitivity, $s_{det}$")
    ax.set_ylabel("Largest view mean after review:\nbias vs true maximal span (mm)")
    ax.set_ylim(0, 1.4)
    h = [Line2D([], [], color=COLOR["A4"], ls=FR_LS[f], lw=0.9) for f in (0.0, 0.1, 0.3)]
    # v05 (8 pt): the three-entry key no longer fits inside the 27 mm wide panel, so it sits in a strip under
    # the figure, right-aligned below panel l (the pink A4 lines appear in l only)
    fig.legend(h, ["$f_{rej}$ 0", "$f_{rej}$ 0.1", "$f_{rej}$ 0.3"], loc="outside lower right", ncol=3,
               frameon=False, handlelength=2.2, columnspacing=1.2, fontsize=es.FS_MIN)

    # shared estimator legend
    hs = [Line2D([], [], **kw(e)) for e in EST]
    fig.legend(hs, [LABEL[e] for e in EST], loc="outside upper center", ncol=4, frameon=False,
               handlelength=2.6, columnspacing=1.0, fontsize=es.FS_MIN)
    figstyle.panel_labels([axd[k] for k in "abcdefghijkl"], list("abcdefghijkl"), size=es.FS_LETTER)

    # push the right-end K labels apart to at least 1.15 line heights (display units, after layout)
    fig.canvas.draw()
    gap = 1.15 * es.FS_MIN
    for key, anns in end_labels.items():
        ax = axd[key]
        ys = [ax.transData.transform(a.xy)[1] * 72 / fig.dpi for a in anns]   # anchor y in points
        order = sorted(range(len(anns)), key=lambda i: -ys[i])
        pos = [ys[i] for i in order]
        for j in range(1, len(pos)):
            pos[j] = min(pos[j], pos[j - 1] - gap)
        for i, pj in zip(order, pos):
            anns[i].xyann = (4.5, pj - ys[i])
    paths = figstyle.save_all(fig, str(STEM))
    pd.concat(src, ignore_index=True).to_csv(str(STEM) + "_source.csv", index=False)
    fig.canvas.draw()
    for d in figqa.report(fig, min_pt=es.FS_MIN):
        print("QA:", d)
    print(figqa.greyscale_and_downscale(str(STEM) + ".png", WIDTH_MM))
    print("font", fam, paths)
    png = Path(str(STEM) + ".png")
    print("png newer than script:", png.stat().st_mtime > Path(__file__).stat().st_mtime)


if __name__ == "__main__":
    main()
