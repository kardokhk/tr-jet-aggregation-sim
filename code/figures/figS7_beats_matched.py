#!/usr/bin/env python
"""Supplementary Figure S7: beat rules at matched beats and under serial correlation
(third amendment, post hoc; package A3-1, Slurm job 5301653).

(a) RMSE against the true maximal span of three ways of using the beats of one anchor view, by
    beat-to-beat variation, in sinus rhythm and atrial fibrillation: the windowed mean (three accepted
    beats, +-15%, up to 30 beats), the plain mean of all beats that the windowed rule acquired for that
    view (same acquisition, no selection) and the plain mean of the first three beats. All three are
    computed on the same simulated beats (common random numbers within a condition).
(b) RMSE of the plain mean of the first n beats, n = 1 to 13, with lag-1 serial correlation of
    consecutive beats 0, 0.3, 0.6 and 0.9 (sinus rhythm, beat-to-beat variation 15%).
(c) Upper: views in which the first three beats met the +-15% window; lower: window penalty (RMSE of
    the windowed mean minus RMSE of the plain mean of the first three beats, paired), against serial
    correlation, sinus rhythm and atrial fibrillation.

AP axis, 100,000 simulated views per condition. Bars are 95% Monte Carlo intervals (1.96 x MCSE).
The arms are beat rules on one view, not the four view rules, so the per-rule Okabe-Ito colours are
not used: black and grey with distinct markers and dash patterns (window = open circle, dashed, as
in Figure 2 and the graphical abstract).

Input:  results/2026-10-05_osrev/beats/beats_arm_errors.csv, beats_table_rmse_by_n_beats.csv,
        beats_table_sensitivity.csv
Output: figures/figS7_beats_matched.{pdf,png,tif}, figures/figS7_beats_matched_source.csv
Run:    /project/home/p201509/envs/duomax-sim/bin/python code/figures/figS7_beats_matched.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/users/u104629/.claude/academic/assets")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figqa  # noqa: E402
import estimator_style as es  # noqa: E402
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "2026-10-05_osrev" / "beats"
STEM = str(ROOT / "figures" / "figS7_beats_matched")
WIDTH_MM, HEIGHT_MM = 183.0, 140.0
Z = 1.959964
ARMS = {"win3_b30": dict(label="Windowed mean (±15%)", color="#000000", marker="o", mfc="white",
                         ls=(0, (3, 1.5))),
        "plain_3": dict(label="Plain mean, first 3 beats", color="#000000", marker="o", mfc="#000000", ls="-"),
        "sameacq3": dict(label="Plain mean, same acquired beats", color="#6e6e6e", marker="s", mfc="#6e6e6e",
                         ls=(0, (1.0, 1.2)))}
RHO = {0.0: dict(color="#000000", marker="o", ls="-"), 0.3: dict(color="#4d4d4d", marker="s", ls=(0, (4, 1.5))),
       0.6: dict(color="#808080", marker="^", ls=(0, (4, 1.5, 1, 1.5))), 0.9: dict(color="#a6a6a6", marker="D", ls=(0, (1, 1.2)))}
RHY = {"sinus": dict(ls="-", mfc="#000000", label="Sinus rhythm"),
       "AF": dict(ls=(0, (3, 1.5)), mfc="white", label="Atrial fibrillation")}


def main():
    fam = figstyle.use_print_style()
    es.use_journal_text()
    arm = pd.read_csv(RES / "beats_arm_errors.csv")
    byn = pd.read_csv(RES / "beats_table_rmse_by_n_beats.csv")
    sens = pd.read_csv(RES / "beats_table_sensitivity.csv")
    src = []
    fig, axd = plt.subplot_mosaic([["a1", "a1", "a1", "a2", "a2", "a2"], ["b", "b", "b", "c1", "c1", "c1"],
                                   ["b", "b", "b", "c2", "c2", "c2"]],
                                  figsize=figstyle.mm(WIDTH_MM, HEIGHT_MM), layout="constrained",
                                  height_ratios=[1.0, 0.5, 0.5])
    fig.get_layout_engine().set(h_pad=3 / 72, w_pad=3 / 72, hspace=0.06, wspace=0.06)

    # ---- a: three arms by beat-to-beat variation, sinus and AF
    a = arm[(arm.grid == "budget") & (arm.axis == "AP") & (arm.estimand == "T1") & arm.arm.isin(ARMS)]
    for key, rh, title in (("a1", "sinus", "Sinus rhythm"), ("a2", "AF", "Atrial fibrillation")):
        ax = axd[key]
        for k, st in ARMS.items():
            s = a[(a.rhythm == rh) & (a.arm == k)].sort_values("beat_cv")
            assert len(s) == 3 and (s.n == 100000).all()
            ax.errorbar(100 * s.beat_cv, s.rmse_mm, yerr=Z * s.rmse_mcse, color=st["color"], marker=st["marker"],
                        mfc=st["mfc"], mec=st["color"], ls=st["ls"], ms=3.6, mew=0.8, lw=0.9, elinewidth=0.6,
                        capsize=0, clip_on=False)
            for r in s.itertuples():
                src.append(dict(panel="a", rhythm=rh, series=st["label"], arm=k, x_beat_cv_pct=100 * r.beat_cv,
                                y=r.rmse_mm, mcse=r.rmse_mcse, n=r.n, quantity="RMSE vs true maximal span (mm)",
                                table=f"beats_arm_errors.csv budget/{r.condition}"))
        ax.set_xticks([5, 15, 30])
        ax.set_xlim(2, 33)
        ax.set_ylim(1.5, 3.1)
        ax.set_yticks([1.5, 2.0, 2.5, 3.0])
        ax.set_xlabel("Beat-to-beat CV (%)")
        ax.set_ylabel("RMSE (mm)")
        ax.set_title(title, fontsize=es.FS_TITLE)
    axd["a1"].legend(handles=[Line2D([], [], color=st["color"], marker=st["marker"], mfc=st["mfc"], mec=st["color"],
                                     ls=st["ls"], ms=3.6, mew=0.8, lw=0.9, label=st["label"])
                              for st in ARMS.values()],
                     loc="upper left", frameon=False, handlelength=2.6, borderaxespad=0.2)

    # ---- b: RMSE of the plain mean of the first n beats under serial correlation (sinus, CV 15%)
    ax = axd["b"]
    cond = {0.0: "sinus_ref", 0.3: "sinus_ar1_0.3", 0.6: "sinus_ar1_0.6", 0.9: "sinus_ar1_0.9"}
    for rho, st in RHO.items():
        s = byn[(byn.grid == "sens") & (byn.condition == cond[rho]) & (byn.axis == "AP") & (byn.estimand == "T1")
                & (byn.n_beats <= 13)].sort_values("n_beats")
        assert len(s) == 13 and (s.n == 100000).all() and abs(s.beat_ar1_rho.iloc[0] - rho) < 1e-9
        ax.errorbar(s.n_beats, s.rmse_mm, yerr=Z * s.rmse_mcse, color=st["color"], marker=st["marker"], ls=st["ls"],
                    ms=3.2, mew=0.8, lw=0.9, elinewidth=0.6, capsize=0, label=f"{rho:g}")
        for r in s.itertuples():
            src.append(dict(panel="b", rhythm="sinus", series=f"serial correlation {rho:g}", x_n_beats=r.n_beats,
                            y=r.rmse_mm, mcse=r.rmse_mcse, n=r.n, quantity="RMSE vs true maximal span (mm)",
                            table=f"beats_table_rmse_by_n_beats.csv sens/{r.condition}"))
    ax.set_xticks([1, 3, 5, 7, 10, 13])
    ax.set_xlim(0.3, 13.7)
    ax.set_ylim(1.5, 2.7)
    ax.set_xlabel("Beats averaged (n)")
    ax.set_ylabel("RMSE (mm)")
    ax.legend(title="Serial correlation", loc="upper right", frameon=False, ncol=2, handlelength=2.6,
              columnspacing=1.0, borderaxespad=0.2)

    # ---- c: pass rate and window penalty against serial correlation
    s0 = sens[(sens.axis == "AP") & sens.family.isin(["reference", "ar1"])]
    for key, ycol, ecol, ylab, q in (
            ("c1", "pct_met_first_3", "pct_met_first_3_mcse", "First 3 beats\nmet window (%)",
             "views in which the first three beats met the window (%)"),
            ("c2", "T1_win3_minus_plain3_d_rmse_mm", "T1_win3_minus_plain3_d_rmse_mcse", "Window penalty\n(mm)",
             "RMSE windowed mean minus RMSE plain mean of first 3 beats, paired (mm)")):
        ax = axd[key]
        for rh, st in RHY.items():
            s = s0[s0.rhythm == rh].sort_values("beat_ar1_rho")
            assert len(s) == 4 and (s.n_views == 100000).all()
            ax.errorbar(s.beat_ar1_rho, s[ycol], yerr=Z * s[ecol], color="#000000", marker="o", mfc=st["mfc"],
                        mec="#000000", ls=st["ls"], ms=3.6, mew=0.8, lw=0.9, elinewidth=0.6, capsize=0,
                        label=st["label"], clip_on=False)
            for r in s.itertuples():
                src.append(dict(panel="c", rhythm=rh, series=st["label"], x_serial_correlation=r.beat_ar1_rho,
                                y=getattr(r, ycol), mcse=getattr(r, ecol), n=r.n_views, quantity=q,
                                count=r.n_met_first_3 if key == "c1" else None,
                                table=f"beats_table_sensitivity.csv {r.condition}"))
        ax.set_xticks([0, 0.3, 0.6, 0.9])
        ax.set_xlim(-0.08, 0.98)
        ax.set_ylabel(ylab)
    axd["c1"].set_ylim(0, 80)
    axd["c1"].set_yticks([0, 20, 40, 60, 80])
    axd["c1"].tick_params(labelbottom=False)
    axd["c1"].legend(loc="lower right", frameon=False, handlelength=2.6, borderaxespad=0.2)
    axd["c2"].axhline(0, color="#8c8c8c", lw=0.5)
    axd["c2"].set_ylim(-0.02, 0.16)
    axd["c2"].set_yticks([0, 0.05, 0.10, 0.15])
    axd["c2"].set_xlabel("Serial correlation of consecutive beats")
    fig.align_ylabels([axd["c1"], axd["c2"]])
    figstyle.panel_labels([axd["a1"], axd["b"], axd["c1"]], ["a", "b", "c"], dx=-0.14, dy=1.03, size=es.FS_LETTER)

    fig.canvas.draw()
    issues = figqa.report(fig, min_pt=es.FS_MIN)
    paths = figstyle.save_all(fig, STEM)
    pd.DataFrame(src).to_csv(STEM + "_source.csv", index=False, float_format="%.6g")
    grey, actual = figqa.greyscale_and_downscale(STEM + ".png", WIDTH_MM)
    assert Path(STEM + ".png").stat().st_mtime > Path(__file__).stat().st_mtime
    print("font:", fam)
    print("figqa:", issues if issues else "clean")
    print(paths, grey, actual)


if __name__ == "__main__":
    main()
