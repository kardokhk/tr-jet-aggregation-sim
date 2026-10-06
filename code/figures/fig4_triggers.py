#!/usr/bin/env python
"""Figure 4 (final numbering; formerly Figure 5): cross-view discrepancy triggers (experiment E4).

Input: results/2026-09-18_full/analysis/E4_*.csv from code/04_analyse_E4.py, and
results/2026-10-05_osrev/fig4_clustered/fig4_clustered_se.csv from code/23_osrev_fig4_clustered_se.py
(Slurm job 5306853, 2026-10-06): patient-clustered standard errors of the pair-level rates in panels
c and d, from the same six E4 cells re-simulated with their original seeds (counts identical).
v06 (2026-10-06): the bars of panels c and d are 95% Monte Carlo intervals from these clustered
standard errors; up to v05 they were binomial standard errors multiplied by sqrt(3). No point
estimate changed.
v07 (2026-10-06, after review): plain-text panel titles (no mathtext subscripts), key title "Threshold", the
"CV 15%" label moved clear of the CV 30% series, the x axis of panel d described in words, and every axis
given a small margin beyond its first and last tick so that no marker is drawn across or outside an axis
(markers were formerly unclipped on the 0% and 100% lines). No plotted value changed. The figure is
greyscale by design (see the paragraph below).
Output: figures/fig4_triggers.{pdf,png,tif} and figures/fig4_triggers_source.csv.
Estimators are not plotted here. Six of the seven Okabe-Ito hues (and black) are bound to
estimators A1 to A7 paper-wide, so reusing any of them for a threshold would give one colour two
meanings; the remaining hue (yellow) cannot carry five levels. Thresholds (ordinal, mm) therefore
use a neutral grey ramp, redundantly encoded by a distinct marker per threshold, so every series
is separable in greyscale by construction. The same encoding is used in every panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/home/users/u104629/.claude/academic/assets")
import figqa  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimator_style as es  # noqa: E402  shared text sizes (8 pt floor)
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "results" / "2026-09-18_full" / "analysis"
CLUST = ROOT / "results" / "2026-10-05_osrev" / "fig4_clustered" / "fig4_clustered_se.csv"
STEM = ROOT / "figures" / "fig4_triggers"
WIDTH_MM, HEIGHT_MM = 183.0, 130.0
Z = 1.959964
AXIS = "AP"
BASE = dict(u="base", r_mean=0.64, N_beats=3)
TAUS = [2.0, 3.0, 4.0, 5.0, 6.0]
TCOL = {2.0: "#000000", 3.0: "#333333", 4.0: "#5c5c5c", 5.0: "#808080", 6.0: "#a3a3a3"}
TMARK = {2.0: "o", 3.0: "s", 4.0: "^", 5.0: "D", 6.0: "v"}
CV_LS = {0.05: "-", 0.15: "--", 0.30: ":"}


def pad(ax, lo, hi, frac=0.035):
    """y limits a little beyond lo and hi, with the axis line drawn from lo to hi only, so that a marker at
    lo or hi lies wholly inside the axes and is not clipped."""
    d = frac * (hi - lo)
    ax.set_ylim(lo - d, hi + d)
    ax.spines["left"].set_bounds(lo, hi)


def main():
    fam = figstyle.use_print_style()
    es.use_journal_text()
    anyt = pd.read_csv(AN / "E4_any_trigger.csv")
    oc = pd.read_csv(AN / "E4_operating.csv")
    truth = pd.read_csv(AN / "E4_ellipticity_truth.csv")
    ell = pd.read_csv(AN / "E4_ellipticity_cells.csv")
    src = []
    # patient-clustered standard errors, keyed by (cell, tau, measure); values checked against the stored ones
    cse = pd.read_csv(CLUST).set_index(["cell", "tau_mm", "measure"])

    def clustered(cell, tau, measure, value):
        r = cse.loc[(int(cell), float(tau), measure)]
        assert abs(r.value - value) < 1e-12, (cell, tau, measure, r.value, value)
        return float(r.se_patient_clustered)

    fig, axd = plt.subplot_mosaic([["a", "b", "c"], ["d", "e", "f"]],
                                  figsize=figstyle.mm(WIDTH_MM, HEIGHT_MM), layout="constrained")
    fig.get_layout_engine().set(h_pad=2 / 72, w_pad=2 / 72, hspace=0.05, wspace=0.05)

    # ---- a, b: per-patient probability of >= 1 trigger / adjudication, AP axis
    base = anyt[(anyt.u == BASE["u"]) & (anyt.r_mean == BASE["r_mean"])
                & (anyt.N_beats == BASE["N_beats"]) & (anyt.axis == AXIS)]
    # The per-patient rate at threshold tau is identical for warning and adjudication
    # (one-sided rule; checked in 04_analyse_E4.py), so tau 4 mm is drawn once, in a.
    for key, metric, taus, ylab in (("a", "p_any_trigger", [2.0, 3.0, 4.0],
                                     "Patients with ≥ 1\nwarning (%)"),
                                    ("b", "p_any_adj", [5.0, 6.0],
                                     "Patients with ≥ 1\nadjudication (%)")):
        ax = axd[key]
        for t in taus:
            for ve, ls, fill in ((True, "-", True), (False, "--", False)):
                s = base[(base.metric == metric) & (base.tau == t) & (base.view_errors == ve)].sort_values("beat_cv")
                ax.plot(100 * s.beat_cv, 100 * s.value, ls=ls, color=TCOL[t], marker=TMARK[t],
                        mfc=TCOL[t] if fill else "white", mec=TCOL[t], mew=0.8, lw=0.9)
                for r in s.itertuples():
                    src.append(dict(panel=key, axis=AXIS, series=f"{metric} tau={t:g} view_errors={ve}",
                                    x_beat_cv=r.beat_cv, y=r.value, mcse=r.mcse, n=r.n, table=f"E4_any_trigger.csv cell {r.cell}"))
        ax.set_xlabel("Beat-to-beat CV (%)")
        ax.set_ylabel(ylab)
        ax.set_xticks([5, 10, 15, 20, 25, 30])
        ymax = 50 if key == "a" else 20           # own scale per panel; see caption
        pad(ax, 0, ymax)
        ax.set_yticks(range(0, 51, 10) if key == "a" else range(0, 21, 5))
        ax.set_xlim(3, 32)
        ax.set_title("Warning threshold\n2 to 4 mm" if key == "a"
                     else "Adjudication threshold\n5 and 6 mm", fontsize=es.FS_TITLE)

    # ---- c: PPV of a trigger at threshold tau for a true error, AP, view errors on
    ax = axd["c"]
    s0 = oc[(oc.u == BASE["u"]) & oc.view_errors & (oc.r_mean == BASE["r_mean"])
            & (oc.N_beats == BASE["N_beats"]) & (oc.axis == AXIS)]
    for t in TAUS:
        s = s0[s0.tau == t].sort_values("beat_cv")
        se = np.array([clustered(r.cell, t, "ppv", r.ppv) for r in s.itertuples()])
        hw = Z * se
        lo = np.clip(s.ppv - hw, 0, 1)
        hi = np.clip(s.ppv + hw, 0, 1)  # a proportion cannot exceed 100%: clip the Wald bound
        ax.errorbar(100 * s.beat_cv, 100 * s.ppv, yerr=[100 * (s.ppv - lo), 100 * (hi - s.ppv)],
                    color=TCOL[t], marker=TMARK[t], ms=3, lw=0.9, elinewidth=0.6, capsize=1.2,
                    label=f"{t:g} mm")
        for r, se_r in zip(s.itertuples(), se):
            src.append(dict(panel="c", axis=AXIS, series=f"ppv tau={t:g}", x_beat_cv=r.beat_cv, y=r.ppv,
                            mcse=se_r, n=r.n_fired_pairs, table=f"E4_operating.csv cell {r.cell}"))
    ax.set_xlabel("Beat-to-beat CV (%)")
    ax.set_ylabel("PPV for a true error\n> 2 mm (%)")
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_xlim(3, 32)
    pad(ax, 0, 100)
    ax.set_title("Positive predictive value,\nview errors", fontsize=es.FS_TITLE)

    # ---- d: operating points (false-alarm vs hit rate per pair), AP, three beat CVs
    ax = axd["d"]
    for cv, ls in CV_LS.items():
        s = s0[np.isclose(s0.beat_cv, cv)].sort_values("tau")
        ax.plot(100 * s.false_alarm, 100 * s.hit, ls=ls, color="0.45", lw=0.8, zorder=1)
        for r in s.itertuples():
            ax.errorbar(100 * r.false_alarm, 100 * r.hit, yerr=100 * Z * clustered(r.cell, r.tau, "sensitivity", r.hit),
                        color=TCOL[r.tau], marker=TMARK[r.tau], ms=3, ls="none", elinewidth=0.6, capsize=0,
                        zorder=2)
            src.append(dict(panel="d", axis=AXIS, series=f"beat_cv={cv:g} tau={r.tau:g}", x_false_alarm=r.false_alarm,
                            x_mcse=clustered(r.cell, r.tau, "false_positive_rate", r.false_alarm), y=r.hit,
                            mcse=clustered(r.cell, r.tau, "sensitivity", r.hit), n=r.n_event_pairs,
                            n_nonevent=r.n_nonevent_pairs, table=f"E4_operating.csv cell {r.cell}"))
        # CV 5% above its 3 mm point, CV 15% left of its top point, CV 30% above its top point: each label
        # is clear of the neighbouring series
        top = s.sort_values("tau").iloc[1 if cv == 0.05 else 0]
        dx, dy, ha = {0.05: (-3, 5, "center"), 0.15: (-6, 2, "right"), 0.30: (2, 5, "center")}[cv]
        ax.annotate(f"CV {100 * cv:.0f}%", (100 * top.false_alarm, 100 * top.hit),
                    xytext=(dx, dy), textcoords="offset points", fontsize=es.FS_MIN,
                    ha=ha, va="bottom", color="0.2")
    ax.set_xlabel("False-positive rate per view pair (%;\nlinear to 0.5, logarithmic above)")
    ax.set_ylabel("Sensitivity per view pair (%)")
    ax.set_xscale("symlog", linthresh=0.5, linscale=0.6)
    ax.set_xlim(-0.12, 22)
    ax.set_xticks([0, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0", "0.5", "1", "2", "5", "10", "20"])
    ax.set_title("Operating points,\nview errors", fontsize=es.FS_TITLE)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, 42)
    ax.spines["bottom"].set_bounds(0, 20)

    # ---- e: true AP-SL difference exceeds tau, by r_mean
    ax = axd["e"]
    tr = truth[truth.tau.notna()]
    for t in TAUS:
        s = tr[(tr.tau == t) & (tr.source.str.startswith("E4" if t <= 4 else "supplementary"))].sort_values("r_mean")
        ax.plot(s.r_mean, 100 * s.value, color=TCOL[t], marker=TMARK[t], ms=3, lw=0.9, label=f"{t:g} mm")
        for r in s.itertuples():
            src.append(dict(panel="e", axis="AP-SL", series=f"true tau={t:g}", x_r_mean=r.r_mean, y=r.value,
                            mcse=r.mcse, n=r.n, table=f"E4_ellipticity_truth.csv ({r.source})"))
    ax.set_xlabel("Mean SL/AP ratio")
    ax.set_ylabel("True |AP − SL| ≥\nthreshold (%)")
    pad(ax, 0, 100)
    ax.set_xlim(0.51, 0.92)
    ax.spines["bottom"].set_bounds(0.53, 0.90)
    ax.set_xticks([0.53, 0.64, 0.8, 0.9])
    ax.set_title("True span difference\nbetween axes", fontsize=es.FS_TITLE)

    # ---- f: measured anchor-view AP-SL difference exceeds tau (base cell)
    ax = axd["f"]
    m = ell[(ell.quantity.str.startswith("measured")) & (ell.u == "base") & ell.view_errors
            & (ell.beat_cv == 0.15) & (ell.N_beats == 3)]
    for t in TAUS:
        s = m[m.tau == t].sort_values("r_mean")
        ax.plot(s.r_mean, 100 * s.value, color=TCOL[t], marker=TMARK[t], ms=3, lw=0.9, label=f"{t:g} mm")
        for r in s.itertuples():
            src.append(dict(panel="f", axis="AP-SL", series=f"measured anchor tau={t:g}", x_r_mean=r.r_mean,
                            y=r.value, mcse=r.mcse, n=r.n, table=f"E4_ellipticity_cells.csv cell {r.cell}"))
    ax.set_xlabel("Mean SL/AP ratio")
    ax.set_ylabel("Anchor-view |AP − SL|\n≥ threshold (%)")
    pad(ax, 0, 100)
    ax.set_xlim(0.51, 0.92)
    ax.spines["bottom"].set_bounds(0.53, 0.90)
    ax.set_xticks([0.53, 0.64, 0.8, 0.9])
    ax.set_title("Measured difference,\nanchor view", fontsize=es.FS_TITLE)

    # one shared key: threshold (grey level + marker, all panels) and scenario (a, b only)
    hk = [Line2D([], [], color=TCOL[t], marker=TMARK[t], ms=3, lw=0.9, label=f"{t:g} mm") for t in TAUS]
    hk += [Line2D([], [], color="0.3", ls="-", marker="o", ms=3, mfc="0.3", lw=0.9,
                  label="View errors (a, b)"),
           Line2D([], [], color="0.3", ls="--", marker="o", ms=3, mfc="white", lw=0.9,
                  label="No view errors (a, b)")]
    fig.legend(handles=hk, loc="outside upper center", ncol=7, frameon=False, handlelength=2.4,
               columnspacing=1.2, title="Threshold", title_fontsize=es.FS_MIN, fontsize=es.FS_MIN)
    figstyle.panel_labels([axd[k] for k in "abcdef"], list("abcdef"), dx=-0.25, dy=1.02, size=es.FS_LETTER)

    STEM.parent.mkdir(parents=True, exist_ok=True)
    probs = figqa.report(fig, min_pt=es.FS_MIN)
    paths = figstyle.save_all(fig, str(STEM))
    pd.DataFrame(src).to_csv(f"{STEM}_source.csv", index=False)
    grey, actual = figqa.greyscale_and_downscale(str(STEM) + ".png", WIDTH_MM)
    assert Path(str(STEM) + ".png").stat().st_mtime > Path(__file__).stat().st_mtime
    print("font:", fam)
    print("figqa:", probs if probs else "clean")
    print("wrote", paths, grey, actual)


if __name__ == "__main__":
    main()
