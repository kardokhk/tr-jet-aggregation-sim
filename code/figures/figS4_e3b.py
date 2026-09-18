#!/usr/bin/env python
"""Supplementary Figure S4 (amendment 2): sensitivity and specificity at 7, 10 and 13 mm across the
12 view-accuracy scenarios (E3b), A1 to A4, AP and SL axes.

Input : results/2026-09-18_amend2/analysis/E1bE3b_e3_long.csv (from code/08_analyse_amend2_e1e3.py).
Output: figures/figS4_e3b.{pdf,png,tif}, figures/figS4_e3b_source.csv.
E3b cells: K 3, beat CV 15%, overestimation on, acceptance window 15%, median true AP span 10 mm,
rho 0.3, N 3, sinus, prospective; reference T1 on the same axis; A4 s_det 0.8, f_rej 0.1.
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
from matplotlib.lines import Line2D  # noqa: E402

from estimator_style import LABEL, kw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "results" / "2026-09-18_amend2" / "analysis"
STEM = ROOT / "figures" / "figS4_e3b"
WIDTH_MM, HEIGHT_MM = 183.0, 170.0
EST4 = ["A1", "A2", "A3", "A4"]
ANCH = [2.4, 4.8, 10.0, 20.0]
LONG = [8.0, 20.0, 40.0]
CUTS = [7.0, 10.0, 13.0]
ROWS = [("AP", "sensitivity"), ("AP", "specificity"), ("SL", "sensitivity"), ("SL", "specificity")]
YLIM = {"sensitivity": (10, 101)}
# Specificity: own y range per axis and cut-off so the 13 mm columns (95.7% to 99.9% on SL) are not
# compressed into a thin band; every specificity panel therefore carries its own tick labels.
SPEC_LIM = {("AP", 7.0): (60, 101, 10), ("AP", 10.0): (75, 100.5, 5), ("AP", 13.0): (85, 100.5, 5),
            ("SL", 7.0): (75, 100.5, 5), ("SL", 10.0): (88, 100.3, 4), ("SL", 13.0): (95, 100.2, 1)}


def xpos(a, L):
    return ANCH.index(a) * 3.8 + LONG.index(L)


def main():
    fam = figstyle.use_print_style()
    d = pd.read_csv(AN / "E1bE3b_e3_long.csv")
    d = d[d.estimand == "T1"]
    src = []
    fig, axs = plt.subplots(4, 3, figsize=figstyle.mm(WIDTH_MM, HEIGHT_MM), layout="constrained",
                            sharex=True)
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.03, wspace=0.03)
    ticks = [xpos(a, L) for a in ANCH for L in LONG]
    for i, (axis, metric) in enumerate(ROWS):
        for j, cut in enumerate(CUTS):
            ax = axs[i, j]
            for e in EST4:
                s = d[(d.axis == axis) & (d.cutoff_mm == cut) & (d.estimator == e)]
                for a in ANCH:
                    g = s[s.anchor_mean_pct == a].sort_values("long_mean_pct")
                    x = [xpos(a, L) for L in g.long_mean_pct]
                    ax.plot(x, 100 * g[metric], **kw(e, ms=2.6, lw=0.8))
                src.append(s.assign(panel=f"{axis} {metric} {cut:g} mm"))
            for a in ANCH[1:]:
                ax.axvline(xpos(a, 8.0) - 1.4, color="0.85", lw=0.5, zorder=0)
            if metric == "sensitivity":
                ax.set_ylim(*YLIM[metric])
                ax.set_yticks(np.arange(20, 101, 20))
            else:
                lo, hi, step = SPEC_LIM[(axis, cut)]
                vals = 100 * d[(d.axis == axis) & (d.cutoff_mm == cut) & d.estimator.isin(EST4)][metric]
                assert vals.min() >= lo and vals.max() <= hi, (axis, cut, vals.min(), vals.max())
                ax.set_ylim(lo, hi)
                ax.set_yticks(np.arange(lo, 100.01, step))
            ax.set_xlim(-0.6, xpos(20.0, 40.0) + 0.6)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{L:g}" for a in ANCH for L in LONG], fontsize=6)
            if i == 0:
                ax.set_title(f"Cut-off {cut:g} mm", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{axis} {metric} (%)")
            elif metric == "sensitivity":
                ax.tick_params(labelleft=False)
            if i == 0:
                for a in ANCH:
                    ax.text(xpos(a, 20.0), YLIM[metric][0] + 1, f"Anchor\n{a:g}%", ha="center", va="bottom",
                            fontsize=6, color="0.35")
            if i == 3:
                ax.set_xlabel("Long-axis mean underestimation (%)")
    hs = [Line2D([], [], **kw(e)) for e in EST4]
    fig.legend(hs, [LABEL[e] for e in EST4], loc="outside upper center", ncol=4, frameon=False,
               handlelength=2.8, columnspacing=1.8, fontsize=6.5)
    figstyle.panel_labels([axs[i, 0] for i in range(4)], list("abcd"))

    paths = figstyle.save_all(fig, str(STEM))
    pd.concat(src, ignore_index=True).to_csv(str(STEM) + "_source.csv", index=False)
    fig.canvas.draw()
    for q in figqa.report(fig):
        print("QA:", q)
    print(figqa.greyscale_and_downscale(str(STEM) + ".png", WIDTH_MM))
    print("font", fam, paths)
    png = Path(str(STEM) + ".png")
    print("png newer than script:", png.stat().st_mtime > Path(__file__).stat().st_mtime)


if __name__ == "__main__":
    main()
