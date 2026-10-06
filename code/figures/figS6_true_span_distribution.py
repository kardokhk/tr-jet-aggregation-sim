#!/usr/bin/env python
"""Supplementary Figure S6: distribution of the simulated true maximal span (amendment 3, A3-3).

Input : results/2026-10-05_osrev/thresholds/thresholds_true_span_density.csv
        (code/21_osrev_thresholds_analyse.py).
Output: figures/figS6_true_span_distribution.{pdf,png,tif}, figures/figS6_true_span_distribution_source.csv.
Curves: density as implemented (AP lognormal in closed form; SL by quadrature over the SL/AP ratio) for a
median AP span of 10 mm (base population) and 13 mm (sensitivity population). Grey bars: 10,000,000
simulated base-population patients in 0.5 mm bins. Vertical lines: the illustrative cut-offs 7, 10, 13 mm.
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
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "2026-10-05_osrev" / "thresholds" / "thresholds_true_span_density.csv"
STEM = ROOT / "figures" / "figS6_true_span_distribution"
WIDTH_MM, HEIGHT_MM = 89.0, 112.0
XMAX = 30.0
CUTS = (7.0, 10.0, 13.0)
POP = {"median10": dict(color="#0072B2", ls="-", label="Median AP span 10 mm (base)"),
       "median13": dict(color="#D55E00", ls=(0, (4.0, 1.5)), label="Median AP span 13 mm (sensitivity)")}
AXIS_NAME = {"AP": "Anteroposterior", "SL": "Septolateral"}


def main():
    fam = figstyle.use_print_style()
    es.use_journal_text()
    df = pd.read_csv(SRC)
    fig, axs = plt.subplots(2, 1, figsize=figstyle.mm(WIDTH_MM, HEIGHT_MM), layout="constrained", sharex=True)
    src = []
    for ax, axis in zip(axs, ("AP", "SL")):
        h = df[(df.kind == "histogram") & (df.population == "median10") & (df.axis == axis)].sort_values("lo_mm")
        h = h[h.hi_mm <= XMAX]
        # 0.25 mm bins merged in pairs to 0.5 mm bins
        lo = h.lo_mm.to_numpy()[0::2]
        n = h.n.to_numpy().reshape(-1, 2).sum(axis=1)
        ntot = float(h.n.iloc[0] / (h.density_per_mm.iloc[0] * 0.25)) if h.density_per_mm.iloc[0] > 0 else np.nan
        if not np.isfinite(ntot):
            k = int(np.argmax(h.n.to_numpy()))
            ntot = float(h.n.iloc[k] / (h.density_per_mm.iloc[k] * 0.25))
        dens = n / ntot / 0.5
        ax.bar(lo, dens, width=0.5, align="edge", color="0.80", edgecolor="none", zorder=1)
        src.append(pd.DataFrame({"panel": "a" if axis == "AP" else "b", "axis": axis, "kind": "histogram",
                                 "population": "median10", "x_mm": lo + 0.25, "lo_mm": lo, "hi_mm": lo + 0.5,
                                 "n": n, "n_total": ntot, "density_per_mm": dens}))
        for pop, st in POP.items():
            c = df[(df.kind == "density") & (df.population == pop) & (df.axis == axis) & (df.x_mm <= XMAX)]
            ax.plot(c.x_mm, c.density_per_mm, color=st["color"], ls=st["ls"], lw=1.1, zorder=3)
            src.append(pd.DataFrame({"panel": "a" if axis == "AP" else "b", "axis": axis, "kind": "density",
                                     "population": pop, "x_mm": c.x_mm, "density_per_mm": c.density_per_mm}))
        for cut in CUTS:
            ax.axvline(cut, color="0.25", lw=0.6, ls=(0, (1.0, 1.5)), zorder=2)
        ax.set_xlim(0, XMAX)
        ax.set_ylim(0, None)
        ax.set_ylabel("Density (per mm)")
        ax.set_title(f"{AXIS_NAME[axis]} axis", fontsize=es.FS_TITLE)
    axs[0].set_ylim(0, 0.115)
    axs[1].set_ylim(0, 0.175)
    axs[0].set_yticks([0, 0.05, 0.10])
    axs[1].set_yticks([0, 0.05, 0.10, 0.15])
    axs[1].set_xticks([0, 5, 7, 10, 13, 15, 20, 25, 30])
    axs[1].set_xticklabels(["0", "", "7", "10", "13", "", "20", "25", "30"])
    axs[1].set_xlabel("True maximal span (mm)")
    hs = [Line2D([], [], color=st["color"], ls=st["ls"], lw=1.1) for st in POP.values()]
    hs += [Patch(facecolor="0.80", edgecolor="none"), Line2D([], [], color="0.25", lw=0.6, ls=(0, (1.0, 1.5)))]
    labels = [st["label"] for st in POP.values()] + ["Simulated patients (base)", "Illustrative cut-offs"]
    fig.legend(hs, labels, loc="outside lower center", ncol=1, frameon=False, handlelength=2.0, fontsize=es.FS_MIN)
    figstyle.panel_labels(list(axs), ["a", "b"], dx=-0.17, dy=1.03, size=es.FS_LETTER)
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
