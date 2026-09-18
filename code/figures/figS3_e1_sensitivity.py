#!/usr/bin/env python
"""Supplementary Figure S3 (final numbering; formerly S1): E1 sensitivity to inter-view correlation, beats per view and A4 reader behaviour.

Input : results/2026-09-18_full/analysis/E1E3_e1_rho_N.csv, E1E3_e1_a4_variants.csv,
        E1E3_e1_base_slice.csv (from code/02_analyse_e1e3.py).
Output: figures/figS3_e1_sensitivity.{pdf,png,tif}, figures/figS3_e1_sensitivity_source.csv.
AP axis, estimand T1, K 3, overestimation on; beat CV 0.15 unless on the x axis.
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
STEM = ROOT / "figures" / "figS3_e1_sensitivity"
WIDTH_MM, HEIGHT_MM = 183.0, 120.0
AXIS = "AP"
UT = {"base": "Underestimation 20%", "low": "Underestimation 8%"}
SD_LS = {0.0: (0, (1.0, 1.2)), 0.5: (0, (4.0, 1.5, 1.0, 1.5)), 0.8: (0, (3.0, 1.5)), 1.0: "-"}


def main():
    fam = figstyle.use_print_style()
    rn = pd.read_csv(AN / "E1E3_e1_rho_N.csv")
    rn = rn[(rn.axis == AXIS) & (rn.estimand == "T1") & (rn.view_over)]
    a4 = pd.read_csv(AN / "E1E3_e1_a4_variants.csv")
    a4 = a4[(a4.axis == AXIS) & (a4.estimand == "T1") & (a4.K == 3) & (a4.view_over) & (a4.f_rej == 0.1)]
    bs = pd.read_csv(AN / "E1E3_e1_base_slice.csv")
    bs = bs[(bs.axis == AXIS) & (bs.estimand == "T1") & (bs.K == 3) & (bs.view_over)]
    src = []
    fig, axd = plt.subplot_mosaic([["a", "b", "c", "d"], ["e", "f", "g", "h"]],
                                  figsize=figstyle.mm(WIDTH_MM, HEIGHT_MM), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.05, wspace=0.03)

    panels = [("a", "base", "rho", "bias"), ("b", "low", "rho", "bias"),
              ("c", "base", "N_beats", "bias"), ("d", "low", "N_beats", "bias"),
              ("e", "base", "N_beats", "rmse"), ("f", "low", "N_beats", "rmse")]
    for k, u, xv, m in panels:
        ax = axd[k]
        if m == "bias":
            ax.axhline(0, color="0.6", lw=0.5, zorder=0)
        for e in EST:
            if xv == "rho":
                s = rn[(rn.u == u) & (rn.N_beats == 3) & (rn.estimator == e)].sort_values("rho")
            else:
                s = rn[(rn.u == u) & (rn.rho == 0.3) & (rn.estimator == e)].sort_values("N_beats")
            ax.plot(s[xv], s[m], **kw(e))
            src.append(s.assign(panel=k))
        if xv == "rho":
            ax.set_xticks([0, 0.3, 0.6, 0.9])
            ax.set_xlabel("Inter-view correlation, $\\rho$")
        else:
            ax.set_xticks([1, 3, 5, 7, 10, 13])
            ax.set_xlabel("Beats per view, N")
        ax.set_title(UT[u], fontsize=7)
        if m == "bias":
            ax.set_ylim(-1.5, 1.7)
            ax.set_ylabel("Bias vs T1 (mm)")
        else:
            ax.set_ylim(1.4, 3.0)
            ax.set_ylabel("RMSE vs T1 (mm)")
    for k in "bd":
        axd[k].set_ylabel("")
        axd[k].tick_params(labelleft=False)
    axd["f"].set_ylabel("")
    axd["f"].tick_params(labelleft=False)

    for k, u in (("g", "base"), ("h", "low")):
        ax = axd[k]
        ax.axhline(0, color="0.6", lw=0.5, zorder=0)
        s3 = bs[(bs.u == u) & (bs.estimator == "A3")].sort_values("beat_cv")
        ax.plot(100 * s3.beat_cv, s3.bias, **kw("A3", lw=0.7, ms=2.5))
        src.append(s3.assign(panel=k))
        ax.annotate("A3", (100 * s3.beat_cv.iloc[-1], s3.bias.iloc[-1]), xytext=(-4, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6, color=COLOR["A3"])
        for sd in (0.0, 0.5, 0.8, 1.0):
            s = a4[(a4.u == u) & (a4.s_det == sd)].sort_values("beat_cv")
            # every s_det level uses the standard open A4 diamond; s_det is carried by line style only
            ax.plot(100 * s.beat_cv, s.bias, **kw("A4", ls=SD_LS[sd], ms=2.5))
            src.append(s.assign(panel=k, estimator="A4"))
        ax.set_xticks([5, 10, 15, 20, 25, 30])
        ax.set_xlabel("Beat-to-beat CV (%)")
        ax.set_ylim(-0.2, 2.45)
        ax.set_title(UT[u] + ", $f_{rej}$ 0.1", fontsize=7)
        ax.set_ylabel("Bias vs T1 (mm)" if k == "g" else "")
        if k == "h":
            ax.tick_params(labelleft=False)
    hsd = [Line2D([], [], **kw("A4", ls=SD_LS[s], ms=2.5)) for s in (0.0, 0.5, 0.8, 1.0)]
    axd["g"].legend(hsd, [f"A4 $s_{{det}}$ {s:g}" for s in (0.0, 0.5, 0.8, 1.0)], loc="upper left",
                    frameon=False, fontsize=6, handlelength=2.6, borderaxespad=0.2)

    hs = [Line2D([], [], **kw(e)) for e in EST]
    fig.legend(hs, [LABEL[e] for e in EST], loc="outside upper center", ncol=4, frameon=False,
               handlelength=2.8, columnspacing=1.6, fontsize=6.5)
    figstyle.panel_labels([axd[k] for k in "abcdefgh"], list("abcdefgh"))
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
