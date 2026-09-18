#!/usr/bin/env python
"""Figure 2 (final numbering; amendment 2): estimator bias and RMSE across view-accuracy scenarios (E1b).

Input : results/2026-09-18_amend2/analysis/E1bE3b_e1_long.csv, E1bE3b_e1_inflation.csv,
        E1bE3b_e1_worstcase.csv (from code/08_analyse_amend2_e1e3.py).
Output: figures/fig2_view_accuracy.{pdf,png,tif}, figures/fig2_view_accuracy_source.csv.
AP axis, estimand T1. Panels a-c at K 3 (a, b), beat CV 15% (a, b), overestimation on, no acceptance
window, median true AP span 10 mm; rho 0.3, N 3, sinus, prospective; A4 s_det 0.8, f_rej 0.1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/home/users/u104629/.claude/academic/assets")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figqa  # noqa: E402
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from estimator_style import COLOR, LABEL, LS, MARKER, kw, mfc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "results" / "2026-09-18_amend2" / "analysis"
STEM = ROOT / "figures" / "fig2_view_accuracy"
WIDTH_MM, HEIGHT_MM = 183.0, 160.0
AXIS = "AP"
EST4 = ["A1", "A2", "A3", "A4"]
LONG = [8.0, 20.0, 40.0]
ANCH = [2.4, 4.8, 10.0, 20.0]
K_LS = {2: ":", 3: "--", 4: "-"}
K_MARKER = {2: "o", 3: "s", 4: "^"}          # one marker shape per K (panel c)
K_MFC = {2: "white", 3: "0.55", 4: "black"}  # open, grey, black fill: fill darkens with K
K_BAR = {2: "0.80", 3: "0.62", 4: "0.42"}    # range-bar grey darkens with K (panel c)
REF_COMBO = (4.8, 20.0)   # anchor / long-axis mean underestimation of the main-run base case


def base_mask(d, K=3, cv=0.15):
    return ((d.K == K) & np.isclose(d.beat_cv, cv) & d.view_over & (d.window == 0) &
            (d.S_median_mm == 10.0) & (d.axis == AXIS))


def main():
    fam = figstyle.use_print_style()
    lw = pd.read_csv(AN / "E1bE3b_e1_long.csv")
    lw = lw[(lw.estimand == "T1")]
    inf = pd.read_csv(AN / "E1bE3b_e1_inflation.csv")
    wc = pd.read_csv(AN / "E1bE3b_e1_worstcase.csv")
    wc = wc[(wc.estimand == "T1")]
    src = []

    # nested grids so the y label of panel d does not widen the gap between a1 and a2
    # (nested gridspecs rather than subfigures, so that figqa.report sees every axes via fig.axes)
    fig = plt.figure(figsize=figstyle.mm(WIDTH_MM, HEIGHT_MM), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.03, wspace=0.03)
    outer = fig.add_gridspec(2, 1, height_ratios=[2, 1.12])
    gt = outer[0].subgridspec(2, 3)
    gb = outer[1].subgridspec(1, 2, width_ratios=[1, 2.1])
    axd = {f"{r}{j + 1}": fig.add_subplot(gt[i, j]) for i, r in enumerate("ab") for j in range(3)}
    axd["c"], axd["d"] = fig.add_subplot(gb[0, 0]), fig.add_subplot(gb[0, 1])

    # ---- a, b: bias and RMSE vs anchor underestimation, one panel per long-axis level
    base = lw[base_mask(lw)]
    for row, metric, ylab, ylim in (("a", "bias", "Bias vs T1 (mm)", (-3.4, 1.2)),
                                    ("b", "rmse", "RMSE vs T1 (mm)", (1.5, 4.3))):
        for j, L in enumerate(LONG, start=1):
            ax = axd[f"{row}{j}"]
            if metric == "bias":
                ax.axhline(0, color="0.6", lw=0.5, zorder=0)
            for e in EST4:
                s = base[(base.long_mean_pct == L) & (base.estimator == e)].sort_values("anchor_mean_pct")
                ax.plot(s.anchor_mean_pct, s[metric], **kw(e))
                src.append(s.assign(panel=row, quantity=metric))
            ax.set_xscale("log")
            ax.set_xticks(ANCH)
            ax.set_xticklabels(["2.4", "4.8", "10", "20"])
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.set_xlim(2.0, 24.0)
            ax.set_ylim(*ylim)
            ax.set_title(f"Long-axis views {L:g}% underestimated", fontsize=7)
            ax.set_xlabel("Anchor mean underestimation (%, log scale)")
            if j == 1:
                ax.set_ylabel(ylab)
            else:
                ax.tick_params(labelleft=False)

    # ---- c: selection inflation E[A3 - A1] vs beat CV by K
    ax = axd["c"]
    ib = inf[(inf.view_over) & (inf.window == 0) & (inf.S_median_mm == 10.0) & (inf.axis == AXIS)]
    off = {2: -1.4, 3: 0.0, 4: 1.4}
    khandles = []
    for K in (2, 3, 4):
        g = ib[ib.K == K]
        rng = g.groupby("beat_cv").inflation.agg(["min", "max"]).reset_index()
        x = 100 * rng.beat_cv + off[K]
        ax.vlines(x, rng["min"], rng["max"], color=K_BAR[K], lw=2.2, zorder=1)
        ref = g[(g.anchor_mean_pct == REF_COMBO[0]) & (g.long_mean_pct == REF_COMBO[1])].sort_values("beat_cv")
        ax.plot(100 * ref.beat_cv + off[K], ref.inflation, color="black", ls=K_LS[K],
                marker=K_MARKER[K], ms=3.4, mfc=K_MFC[K], mec="black", mew=0.8, lw=0.9, zorder=3)
        khandles.append((Rectangle((0, 0), 1, 1, fc=K_BAR[K], ec="none"),
                         Line2D([], [], color="black", ls=K_LS[K], marker=K_MARKER[K], ms=3.4,
                                mfc=K_MFC[K], mec="black", mew=0.8, lw=0.9)))
        src.append(g.assign(panel="c", estimator="A3-A1", quantity="inflation"))
    ax.legend(khandles, [f"K = {K}" for K in (2, 3, 4)], loc="upper left", frameon=False,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, handlelength=3.6,
              fontsize=6.5, borderaxespad=0.2)
    ax.set_xticks([5, 15, 30])
    ax.set_xlim(1, 34)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("Beat-to-beat CV (%)")
    ax.set_ylabel("Selection inflation E[A3 − A1] (mm)")
    ax.set_title("A3 minus A1, K = 2 to 4", fontsize=7)

    # ---- d: min-max bias over the 12 view-accuracy combinations
    ax = axd["d"]
    ax.axhline(0, color="0.6", lw=0.5, zorder=0)
    wb = wc[(wc.view_over) & (wc.window == 0) & (wc.S_median_mm == 10.0) & (wc.axis == AXIS)]
    CVS = [0.05, 0.15, 0.30]
    eoff = dict(zip(EST4, np.linspace(-0.27, 0.27, 4)))
    centres, labels = [], []
    for gi, (K, cv) in enumerate([(K, cv) for K in (2, 3, 4) for cv in CVS]):
        xc = gi + (gi // 3) * 0.6
        centres.append(xc)
        labels.append(f"{100 * cv:g}")
        for e in EST4:
            r = wb[(wb.K == K) & np.isclose(wb.beat_cv, cv) & (wb.estimator == e)].iloc[0]
            x = xc + eoff[e]
            # range bar with flat caps; no end markers (a triangle at a bar end reads as an arrow)
            ax.plot([x, x], [r.bias_min, r.bias_max], color=COLOR[e], ls=LS[e], lw=1.3,
                    solid_capstyle="butt", dash_capstyle="butt")
            ax.plot([x, x], [r.bias_min, r.bias_max], ls="none", marker="_", ms=4.0,
                    mec=COLOR[e], mew=1.1)
            src.append(pd.DataFrame([r]).assign(panel="d", quantity="bias_min_max"))
    ax.set_xticks(centres)
    ax.set_xticklabels(labels)
    ax.set_xlim(centres[0] - 0.55, centres[-1] + 0.55)
    ax.set_ylim(-4.0, 2.8)
    for k, K in enumerate((2, 3, 4)):
        ax.text(np.mean(centres[3 * k:3 * k + 3]), 2.7, f"K = {K}", ha="center", va="top", fontsize=7)
    for k in (1, 2):
        ax.axvline((centres[3 * k - 1] + centres[3 * k]) / 2, color="0.8", lw=0.5, zorder=0)
    ax.set_xlabel("Beat-to-beat CV (%)")
    ax.set_ylabel("Bias vs T1, range over 12 scenarios (mm)")
    ax.set_title("Minimum to maximum bias across view-accuracy scenarios", fontsize=7)

    hs = [Line2D([], [], **kw(e, lw=1.1)) for e in EST4]
    fig.legend(hs, [LABEL[e] for e in EST4], loc="outside upper center", ncol=4, frameon=False,
               handlelength=2.8, columnspacing=1.8, fontsize=6.5)
    figstyle.panel_labels([axd["a1"], axd["b1"], axd["c"], axd["d"]], list("abcd"))

    paths = figstyle.save_all(fig, str(STEM))
    pd.concat(src, ignore_index=True).to_csv(str(STEM) + "_source.csv", index=False)
    fig.canvas.draw()
    for d in figqa.report(fig):
        print("QA:", d)
    print(figqa.greyscale_and_downscale(str(STEM) + ".png", WIDTH_MM))
    print("font", fam, paths)
    png = Path(str(STEM) + ".png")
    print("png newer than script:", png.stat().st_mtime > Path(__file__).stat().st_mtime)


if __name__ == "__main__":
    main()
