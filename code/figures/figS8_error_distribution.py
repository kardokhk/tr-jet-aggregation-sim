#!/usr/bin/env python
"""Supplementary Figure S8: error distribution of the four view rules
(third amendment, post hoc; package A3-2, Slurm job 5301599).

(a) 95th percentile of absolute error against RMSE, one point per rule and view-accuracy scenario
    (12 scenarios of the second amendment: three views, three beats, beat-to-beat variation 15%, no
    window). Lines: 95th percentile equal to RMSE, and to 1.96 x RMSE (normal error with no bias).
(b) Central 95% interval of error (2.5th to 97.5th percentile; marker, median) by band of true maximal
    span, base case (anchor underestimation 4.8%, long axis 20%, three views, three beats, +-15%
    window, sinus rhythm; cell 0 of family `cond`, seed SeedSequence(20261005, spawn_key=(72, 5, 0))).
    Grey band: plus or minus the largest overall RMSE of the four rules, to show that RMSE is not a 95%
    error bound and that the interval widens with span.

AP axis, 100,000 simulated patients per scenario; error = reported value minus true maximal span.
Percentile Monte Carlo standard errors are jackknife estimates (approximate, see the views audit);
they are in the source csv and are smaller than the markers in (a).

Input:  results/2026-10-05_osrev/views/views_a_12scenarios.csv, views_e_conditional_quantiles.csv
Output: figures/figS8_error_distribution.{pdf,png,tif}, figures/figS8_error_distribution_source.csv
Run:    /project/home/p201509/envs/duomax-sim/bin/python code/figures/figS8_error_distribution.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/users/u104629/.claude/academic/assets")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figqa  # noqa: E402
import estimator_style as es  # noqa: E402
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "2026-10-05_osrev" / "views"
STEM = str(ROOT / "figures" / "figS8_error_distribution")
WIDTH_MM, HEIGHT_MM = 183.0, 100.0
RULES = ["A1", "A2", "A3", "A4"]
BINS = ["0 to 5", "5 to 7", "7 to 10", "10 to 13", "13 to 16", ">= 16", "all"]
BIN_LAB = ["< 5", "5 to 7", "7 to 10", "10 to 13", "13 to 16", "≥ 16", "All"]


def main():
    fam = figstyle.use_print_style()
    es.use_journal_text()
    a = pd.read_csv(RES / "views_a_12scenarios.csv")
    a = a[a.axis == "AP"]
    q = pd.read_csv(RES / "views_e_conditional_quantiles.csv")
    q = q[(q.scenario == "4.8/20") & np.isclose(q.window, 0.15) & (q.rhythm == "sinus") & (q.axis == "AP")]
    assert len(a) == 48 and a.scenario.nunique() == 12 and len(q) == 28 and q.cell.nunique() == 1
    src = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figstyle.mm(WIDTH_MM, HEIGHT_MM), layout="constrained",
                                   width_ratios=[0.8, 1.2])
    fig.get_layout_engine().set(h_pad=3 / 72, w_pad=3 / 72, wspace=0.06)

    # ---- a
    xx = np.array([1.5, 4.3])
    ax1.plot(xx, xx, color="#8c8c8c", lw=0.6, ls=(0, (1, 1.2)))
    ax1.plot(xx, 1.96 * xx, color="#8c8c8c", lw=0.6, ls=(0, (4, 2)))
    ax1.text(4.25, 3.2, "equal to RMSE", ha="right", va="top", fontsize=es.FS_MIN, color="#4d4d4d")
    ax1.text(3.45, 1.96 * 3.45 + 0.35, "1.96 × RMSE", ha="right", va="bottom", fontsize=es.FS_MIN, color="#4d4d4d")
    for e in RULES:
        s = a[a.rule == e]
        assert len(s) == 12 and (s.n == 100000).all()
        ax1.plot(s.rmse_mm, s.q95_abs_mm, ls="none", marker=es.MARKER[e], ms=3.6, mfc=es.mfc(e), mec=es.COLOR[e],
                 mew=0.8, color=es.COLOR[e], zorder=3)
        for r in s.itertuples():
            src.append(dict(panel="a", rule=es.LABEL[e], scenario_anchor_long_pct=r.scenario, span_band="",
                            x_rmse_mm=r.rmse_mm, x_mcse=r.rmse_mm_mcse, y_q95_abs_err_mm=r.q95_abs_mm,
                            y_mcse=r.q95_abs_mm_mcse, ratio=r.ratio_q95abs_rmse, n=r.n,
                            table=f"views_a_12scenarios.csv cell {r.cell}"))
    ax1.set_xlim(1.5, 4.3)
    ax1.set_ylim(1.0, 8.6)
    ax1.set_xlabel("RMSE (mm)")
    ax1.set_ylabel("95th percentile of absolute error (mm)")

    # ---- b
    rmax = float(q[q.span_bin == "all"].rmse_mm.max())
    ax2.axhspan(-rmax, rmax, color="#E0E0E0", lw=0, zorder=0)
    ax2.axhline(0, color="#8c8c8c", lw=0.5, zorder=1)
    off = dict(zip(RULES, (-0.27, -0.09, 0.09, 0.27)))
    for e in RULES:
        for i, b in enumerate(BINS):
            r = q[(q.rule == e) & (q.span_bin == b)].iloc[0]
            x = i + off[e] + (0.25 if b == "all" else 0.0)
            ax2.plot([x, x], [r["q2.5_err_mm"], r["q97.5_err_mm"]], color=es.COLOR[e], lw=1.4,
                     solid_capstyle="butt", zorder=2)
            ax2.plot([x], [r.q50_err_mm], ls="none", marker=es.MARKER[e], ms=3.4, mfc=es.mfc(e), mec=es.COLOR[e],
                     mew=0.8, zorder=3)
            src.append(dict(panel="b", rule=es.LABEL[e], scenario_anchor_long_pct=r.scenario, span_band=b,
                            n=r.n, q2_5_err_mm=r["q2.5_err_mm"], q2_5_mcse=r["q2.5_err_mm_mcse"],
                            q50_err_mm=r.q50_err_mm, q97_5_err_mm=r["q97.5_err_mm"],
                            q97_5_mcse=r["q97.5_err_mm_mcse"], rmse_mm=r.rmse_mm, bias_mm=r.bias_mm,
                            y_q95_abs_err_mm=r.q95_abs_mm, table=f"views_e_conditional_quantiles.csv cell {r.cell}"))
    ax2.axvline(5.75, color="#8c8c8c", lw=0.5, ls=(0, (1, 1.2)))
    ax2.set_xticks(list(range(6)) + [6.25], BIN_LAB)
    ax2.set_xlim(-0.6, 6.85)
    ax2.set_ylim(-9, 9)
    ax2.set_yticks(range(-8, 9, 4))
    ax2.set_xlabel("True maximal span (mm)")
    ax2.set_ylabel("Error, reported minus true (mm)")
    h = [Line2D([], [], ls="none", marker=es.MARKER[e], ms=3.6, mfc=es.mfc(e), mec=es.COLOR[e], mew=0.8,
                color=es.COLOR[e], label=es.LABEL[e]) for e in RULES]
    fig.legend(handles=h, loc="outside upper center", ncol=4, frameon=False, handletextpad=0.3, columnspacing=1.6)
    ax2.legend(handles=[Patch(facecolor="#E0E0E0", edgecolor="none", label=f"± largest overall RMSE ({rmax:.2f} mm)"),
                        Line2D([], [], color="#4d4d4d", lw=1.4, marker="o", ms=3.4, mfc="#4d4d4d",
                               label="Median and central 95% of error")],
               loc="upper left", frameon=False, handlelength=1.8, borderaxespad=0.2)
    figstyle.panel_labels([ax1, ax2], ["a", "b"], dx=-0.14, dy=1.03, size=es.FS_LETTER)
    ax2.texts[-1].set_x(-0.10)

    fig.canvas.draw()
    issues = figqa.report(fig, min_pt=es.FS_MIN)
    paths = figstyle.save_all(fig, STEM)
    pd.DataFrame(src).to_csv(STEM + "_source.csv", index=False, float_format="%.6g")
    grey, actual = figqa.greyscale_and_downscale(STEM + ".png", WIDTH_MM)
    assert Path(STEM + ".png").stat().st_mtime > Path(__file__).stat().st_mtime
    print("font:", fam)
    print("figqa:", issues if issues else "clean")
    print(f"largest overall RMSE {rmax:.3f} mm")
    print(paths, grey, actual)


if __name__ == "__main__":
    main()
