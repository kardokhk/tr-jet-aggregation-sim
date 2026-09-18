#!/usr/bin/env python
"""Supplementary Figure S2 (final numbering; formerly Figure 3): threshold misclassification of "span >= c" by estimator (experiment E3).

Input : results/2026-09-18_full/analysis/E1E3_e3_base.csv (from code/02_analyse_e1e3.py).
Output: figures/figS2_misclassification.{pdf,png,tif}, figures/figS2_misclassification_source.csv.
Base case (K 3, beat CV 0.15, overestimation on, rho 0.3, N_beats 3), AP axis, estimand T1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/home/users/u104629/.claude/academic/assets")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figqa  # noqa: E402
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from estimator_style import COLOR, EST, LABEL, kw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "results" / "2026-09-18_full" / "analysis"
STEM = ROOT / "figures" / "figS2_misclassification"
WIDTH_MM, HEIGHT_MM = 183.0, 112.0
AXIS = "AP"
Z = 1.959964
DODGE = {e: (i - 3) * 0.28 for i, e in enumerate(EST)}
ROWS = [("base", "Underestimation 20%, overestimation on"),
        ("low", "Underestimation 8%, overestimation on")]
COLS = [("sensitivity", "Sensitivity (%)", (60, 100)),
        ("specificity", "Specificity (%)", (60, 100)),
        ("nri_vs_A1", "NRI vs A1 (percentage points)", (-12, 9))]


def main():
    fam = figstyle.use_print_style()
    w = pd.read_csv(AN / "E1E3_e3_base.csv")
    w = w[w.axis == AXIS]
    mosaic = [["a", "b", "c"], ["d", "e", "f"]]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=figstyle.mm(WIDTH_MM, HEIGHT_MM), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.05, wspace=0.04)
    src = []
    keys = iter("abcdef")
    for u, utitle in ROWS:
        for metric, ylab, ylim in COLS:
            k = next(keys)
            ax = axd[k]
            if metric.startswith("nri"):
                ax.axhline(0, color="0.6", lw=0.5, zorder=0)
            for e in EST:
                s = w[(w.u == u) & (w.estimator == e)].sort_values("cutoff_mm")
                y, se = 100 * s[metric], 100 * s[metric + "_mcse"]
                x = s.cutoff_mm + DODGE[e]
                # 95% MC interval as a capped bar drawn behind the marker. For A7 (x marker) a
                # white disc under the cross hides the bar inside the marker, so the bar never
                # runs through the cross and cannot read as an asterisk.
                ax.errorbar(x, y, yerr=Z * se, fmt="none", ecolor="0.55" if e == "A7" else COLOR[e],
                            elinewidth=0.6, capsize=1.3, capthick=0.6, zorder=2)
                if e == "A7":
                    ax.plot(x, y, ls="none", marker="o", ms=4.6, mfc="white", mec="none", zorder=3)
                ax.plot(x, y, **kw(e, ms=3.2, lw=0, ls="none"), zorder=4)
                src.append(s[["u", "axis", "cutoff_mm", "estimator", metric, metric + "_mcse"]]
                           .assign(panel=k, metric=metric))
            for c in (8.5, 11.5):
                ax.axvline(c, color="0.85", lw=0.5, zorder=0)
            ax.set_xticks([7, 10, 13])
            ax.set_xlim(5.5, 14.5)
            ax.set_ylim(*ylim)
            if metric.startswith("nri"):
                ax.set_yticks([-10, -5, 0, 5])
            ax.set_xlabel("Cut-off c (mm)")
            ax.set_ylabel(ylab)
            ax.set_title(utitle, fontsize=7)
    hs = [Line2D([], [], **kw(e, lw=0)) for e in EST]
    fig.legend(hs, [LABEL[e] for e in EST], loc="outside upper center", ncol=7, frameon=False,
               handlelength=1.2, columnspacing=1.2, fontsize=6.5)
    figstyle.panel_labels([axd[k] for k in "abcdef"], list("abcdef"))
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
