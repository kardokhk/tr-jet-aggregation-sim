"""Figure 2 (v04 numbering, 2026-09-19; Figure 3 in v03, Figure 4 in v01): beat-consistency windows (experiment E2).

Rows: (a) probability that a +-W window is met by the first N beats vs beat CV;
(b) expected beats acquired to satisfy the window (prospective);
(c) RMSE of the anchor-view mean (A1) and index beat (A6) vs T1 against N,
    with and without a +-15% window, sinus and AF;
(d) proportion flagged 'limited sampling' in retrospective data.
AP axis throughout. AF shown at RR CV 20% (base AF state).

Input: results/2026-09-18_full/analysis/E2_*.csv (from code/03_analyse_e2.py).
Output: figures/fig2_beat_rules.{pdf,png,tif} and figures/fig2_beat_rules_source.csv.
Run: /project/home/p201509/envs/duomax-sim/bin/python code/figures/fig2_beat_rules.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/users/u104629/.claude/academic/assets")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402  (sets Agg backend)
import figqa  # noqa: E402
import estimator_style as es  # noqa: E402  shared text sizes (8 pt floor)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "results" / "2026-09-18_full" / "analysis"
STEM = str(ROOT / "figures" / "fig2_beat_rules")
WIDTH_MM, HEIGHT_MM = 183.0, 170.0

from estimator_style import COLOR, MARKER  # noqa: E402  fixed estimator colours and markers

# Fixed estimator encoding shared across all figures of the paper (colour and marker from
# estimator_style). In (c) line style and marker fill encode the window, not the estimator.
EST = {e: dict(color=COLOR[e], marker=MARKER[e]) for e in ("A1", "A6")}
# Window levels: grey ramp (estimator-free panels) with a marker per level so every series
# is identifiable in greyscale; dotted black without markers for no window.
WCOL = {0.10: "#000000", 0.15: "#404040", 0.20: "#6e6e6e", 0.25: "#949494"}
WMK = {0.10: "o", 0.15: "s", 0.20: "^", 0.25: "D"}
MODE_T = {"prospective": "Prospective", "retrospective": "Retrospective"}
RHY = {"sinus": "-", "AF_rr0.2": "--"}
RHY_LABEL = {"sinus": "Sinus rhythm", "AF_rr0.2": "AF (RR CV 20%)"}
AXIS = "AP"
CV_TICKS = [5, 10, 15, 20, 25, 30]


def sel(t, **kw):
    for k, v in kw.items():
        t = t[t[k] == v]
    return t


def main():
    fam = figstyle.use_print_style()
    es.use_journal_text()
    met = pd.read_csv(AN / "E2_window_met.csv")
    beats = pd.read_csv(AN / "E2_beats_acquired.csv")
    lim = pd.read_csv(AN / "E2_limited.csv")
    err = pd.read_csv(AN / "E2_error.csv")

    fig, axs = figstyle.figure_mm(WIDTH_MM, HEIGHT_MM, nrows=4, ncols=2, layout="constrained")
    fig.get_layout_engine().set(h_pad=2 / 72, w_pad=2 / 72, hspace=0.04, wspace=0.04)
    src = []

    # ---- (a) window met by first N beats; (b) beats acquired; (d) limited (retro)
    specs = [
        (0, met, "prospective", "Window met by\nfirst N beats (%)", (0, 100)),
        (1, beats, "prospective", "Mean beats acquired\nper view (beats)", None),
        (3, lim, "retrospective", "Views flagged\n'limited sampling'\n(%)", (0, 100)),
    ]
    for row, tab, mode, ylab, ylim in specs:
        for col, N in enumerate([3, 5]):
            ax = axs[row, col]
            Ws = [0.0, 0.10, 0.15, 0.20, 0.25] if row != 0 else [0.10, 0.15, 0.20, 0.25]
            for W in Ws:
                for rs, ls in RHY.items():
                    d = sel(tab, rhythm_state=rs, p_N_beats=N, p_data_mode=mode, axis=AXIS, W=W)
                    d = d.sort_values("p_beat_cv")
                    if W == 0.0:
                        c, lsx = "#000000", ":"
                        if rs != "sinus":
                            continue  # no-window line identical for sinus and AF; draw once
                    else:
                        c, lsx = WCOL[W], ls
                    yv = d["value"] * (1 if row == 1 else 100)
                    if W == 0.0:
                        ax.plot(100 * d["p_beat_cv"], yv, color=c, ls=lsx, lw=1.0)
                    else:
                        ax.plot(100 * d["p_beat_cv"], yv, color=c, ls=lsx, lw=0.9, marker=WMK[W],
                                ms=2.6, mew=0.7, mec=c, mfc=c if rs == "sinus" else "white",
                                clip_on=False)
                    for _, r in d.iterrows():
                        src.append(dict(panel="abd"[[0, 1, 3].index(row)], column=f"N={N}", rhythm=rs,
                                        data_mode=mode, axis=AXIS, W=W, beat_cv=r["p_beat_cv"],
                                        N=N, estimator="rule", estimand="",
                                        quantity=tab.attrs.get("q", ["p_window_met_first_N",
                                                                    "beats_acquired_mean",
                                                                    "p_limited"][[0, 1, 3].index(row)]),
                                        value=r["value"], mcse=r["mcse"], n=r["n"]))
            ax.set_xticks(CV_TICKS)
            ax.set_xlim(4, 31)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xlabel("Beat-to-beat CV (%)")
            if col == 0:
                ax.set_ylabel(ylab)
            ax.set_title(f"{MODE_T[mode]}, N = {N}", fontsize=es.FS_TITLE, pad=2)
            if row == 1:
                # own scale per column so the N = 3 curves are not compressed; see caption
                ax.set_ylim(0, 6.5 if N == 3 else 13)
                ax.set_yticks([0, 2, 4, 6] if N == 3 else [0, 3, 6, 9, 12])

    # ---- (c) RMSE vs T1 against N, A1 and A6, with and without +-15% window
    for col, rs in enumerate(["sinus", "AF_rr0.2"]):
        ax = axs[2, col]
        for est in ["A1", "A6"]:
            for W, ls, fill in [(0.0, "-", True), (0.15, "--", False)]:
                d = sel(err, rhythm_state=rs, p_data_mode="prospective", axis=AXIS, W=W, estimator=est,
                        subset="all", estimand="T1", metric="rmse_mm", p_beat_cv=0.15).sort_values("p_N_beats")
                st = EST[est]
                ax.plot(d["p_N_beats"], d["value"], color=st["color"], ls=ls, lw=1.0, marker=st["marker"],
                        ms=4, mfc=st["color"] if fill else "white", mec=st["color"], mew=0.8)
                for _, r in d.iterrows():
                    src.append(dict(panel="c", column=rs, rhythm=rs, data_mode="prospective", axis=AXIS, W=W,
                                    beat_cv=0.15, N=r["p_N_beats"], estimator=est, estimand="T1",
                                    quantity="rmse_mm", value=r["value"], mcse=r["mcse"], n=r["n"]))
        ax.set_xticks([1, 3, 5, 7, 10, 13])
        ax.set_xlim(0, 14)
        ax.set_ylim(1.5, 3.0)
        ax.set_xlabel("Beats averaged per view, N")
        if col == 0:
            ax.set_ylabel("RMSE vs true maximal\nspan (mm)")
        ax.set_title("Prospective, " + RHY_LABEL[rs][0].lower() + RHY_LABEL[rs][1:] + ", beat CV 15%"
                     if rs == "sinus" else "Prospective, " + RHY_LABEL[rs] + ", beat CV 15%",
                     fontsize=es.FS_TITLE, pad=2)
    # Legend split by encoding (estimator = colour and marker; window = line style and marker fill), so the
    # entries stay short enough to sit between the curves at 8 pt (v05, 2026-09-19).
    hc = [Line2D([], [], color=EST["A1"]["color"], marker="o", ms=4, ls="-", label="Anchor-view mean"),
          Line2D([], [], color=EST["A6"]["color"], marker="P", ms=4, ls="-", label="Index beat"),
          Line2D([], [], color="#6e6e6e", marker="o", ms=4, ls="-", label="No window (filled)"),
          Line2D([], [], color="#6e6e6e", marker="o", ms=4, mfc="white", ls="--",
                 label="±15% window (open)")]
    # estimators keyed in the sinus panel, window in the AF panel, each in the gap between the curves
    for ax, hh in ((axs[2, 0], hc[:2]), (axs[2, 1], hc[2:])):
        ax.legend(handles=hh, loc="center right", bbox_to_anchor=(1.0, 0.45), ncol=1, frameon=False,
                  fontsize=es.FS_MIN, handlelength=2.6, borderaxespad=0.2, labelspacing=0.3)

    # ---- shared legend for rows a, b, d
    # Window levels as colour swatches, rhythm as line style in neutral mid-grey,
    # so that no legend glyph encodes two things at once.
    hw = [Line2D([], [], color=WCOL[w], ls="-", lw=0.9, marker=WMK[w], ms=3,
                 label=f"W ±{int(round(100 * w))}%") for w in WCOL]
    hw += [Line2D([], [], color="#000000", ls=":", lw=1.0, label="No window"),
           Line2D([], [], color="#6e6e6e", ls="-", lw=0.9, marker="o", ms=3, label="Sinus (filled)"),
           Line2D([], [], color="#6e6e6e", ls="--", lw=0.9, marker="o", ms=3, mfc="white",
                  label="AF, RR CV 20% (open)")]
    fig.legend(handles=hw, loc="outside upper center", ncol=7, frameon=False, fontsize=es.FS_MIN,
               title="Rows a, b, d: beat-consistency window W and rhythm", title_fontsize=es.FS_MIN, handlelength=2.6)

    figstyle.panel_labels([axs[0, 0], axs[1, 0], axs[2, 0], axs[3, 0]], list("abcd"), dx=-0.22, dy=1.02, size=es.FS_LETTER)

    fig.canvas.draw()
    issues = figqa.report(fig, min_pt=es.FS_MIN)
    print("font:", fam)
    print("figqa:", issues if issues else "clean")
    paths = figstyle.save_all(fig, STEM)
    print("saved:", paths)
    s = pd.DataFrame(src)
    s.to_csv(STEM + "_source.csv", index=False, float_format="%.6g")
    print("max MCSE by quantity:", s.groupby("quantity")["mcse"].max().to_dict())
    grey, actual = figqa.greyscale_and_downscale(STEM + ".png", WIDTH_MM)
    print(grey, actual)
    print("png newer than script:", Path(STEM + ".png").stat().st_mtime > Path(__file__).stat().st_mtime)


if __name__ == "__main__":
    main()
