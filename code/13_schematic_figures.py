"""Figure 1 (study design schematic) and the graphical abstract.

Family: schematic. Amendment 2 context (protocol notes/protocol-2026-09-18-v01.md).

Outputs
  figures/fig1_design.{pdf,png,tif}, figures/fig1_design_source.csv
  figures/graphical_abstract.{pdf,png,tif}, figures/graphical_abstract_source.csv
  results/2026-09-18_amend2/analysis/schematic_ga_numbers.csv   (every number drawn in the
      graphical abstract, with MCSE and source row)
  results/2026-09-18_amend2/analysis/schematic_e1b_base_ap_bias.csv (E1b slice behind panel a)

Inputs (read only)
  results/2026-09-18_amend2/E1_cells_*.parquet, E5_cells_*.parquet
  results/2026-09-18_full/analysis/E2_error.csv, E2_window_met.csv, E2_window_effect.csv,
      E4_operating.csv, E4_any_trigger.csv, E5E6_E5_ai_agreement.csv

Figure 1 is a drawing. Its beat values are illustrative (fixed numbers in this script, no random
numbers) and are written to the source CSV only so that the drawing is regenerable.
Every number in the graphical abstract is computed here from the result files and checked
against the values quoted in the task brief before drawing (assertions below).

Run: /project/home/p201509/envs/duomax-sim/bin/python code/13_schematic_figures.py [fig1|ga|all]
"""
import glob
import os
import sys

sys.path.insert(0, "/home/users/u104629/.claude/academic/assets")
import figstyle  # noqa: E402  (sets Agg)
import figqa  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "figures"))
import estimator_style as es  # noqa: E402

AM = os.path.join(ROOT, "results", "2026-09-18_amend2")
AN_FULL = os.path.join(ROOT, "results", "2026-09-18_full", "analysis")
AN_OUT = os.path.join(AM, "analysis")
FIG = os.path.join(ROOT, "figures")
os.makedirs(AN_OUT, exist_ok=True)

# Estimator names come from the shared style module (A7: "offset-corrected mean").
LABEL = dict(es.LABEL)

fam = figstyle.use_print_style()

# Type sizes (points). Three body sizes plus the panel letters.
FS_BODY, FS_HEAD, FS_LETTER = 6.5, 7.0, 8.0


# ============================================================================ helpers
def qa(fig, stem, width_mm):
    paths = figstyle.save_all(fig, stem)
    probs = figqa.report(fig)
    grey, actual = figqa.greyscale_and_downscale(stem + ".png", width_mm)
    script_mtime = os.path.getmtime(os.path.abspath(__file__))
    assert os.path.getmtime(stem + ".png") > script_mtime, "stale PNG"
    print(f"[{os.path.basename(stem)}] font family: {fam}")
    print(f"[{os.path.basename(stem)}] files: {paths + [grey, actual]}")
    print(f"[{os.path.basename(stem)}] figqa: {len(probs)} issue(s)")
    for p in probs:
        print("   ", p)
    return probs


def arrow(ax, p0, p1, style="-|>", lw=0.7, color="black", ms=6, ls="-", cs="arc3"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                        linestyle=ls, connectionstyle=cs, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def dim(ax, x0, x1, y, text=None, dy=0.0, color="black", fs=FS_BODY, va="bottom", lw=0.7):
    arrow(ax, (x0, y), (x1, y), style="<|-|>", lw=lw, color=color, ms=4)
    if text:
        ax.text((x0 + x1) / 2, y + dy, text, ha="center", va=va, fontsize=fs, color=color)


def box(ax, x, y, w, h, fc="white", ec="black", lw=0.6, r=0.6):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    return p


# ============================================================================ Figure 1
def fig1():
    W_MM, H_MM = 183, 125
    fig = plt.figure(figsize=figstyle.mm(W_MM, H_MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 0.80, 1.40], height_ratios=[1.0, 1.0])
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[:, 2])
    for ax in (ax_a1, ax_a2, ax_b, ax_c):
        ax.set_axis_off()
    src = []
    GREY = "#666666"

    # ---------------------------------------------------------------- a (top): orifice en face
    ax = ax_a1
    ax.set_xlim(-8.2, 8.2)
    ax.set_ylim(-6.3, 7.0)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.text(-8.1, 6.9, "a", ha="left", va="top", fontsize=FS_LETTER, fontweight="bold")
    ax.text(-6.9, 6.9, "View 1 (anchor): en face short axis", ha="left", va="top",
            fontsize=FS_HEAD, fontweight="bold")
    ax.text(-6.9, 5.7, "$T_1$ = largest true span over all planes, per axis", ha="left",
            va="top", fontsize=FS_BODY)
    A, B = 5.0, 3.2   # semi-axes: AP (horizontal) and SL (vertical), SL/AP = 0.64
    ax.add_patch(Ellipse((0, 0), 2 * A, 2 * B, fc="#E8E8E8", ec="black", lw=0.8))
    # view 3 plane through the centre along AP (behind the AP arrow)
    ax.plot([-7.4, 6.4], [0.0, 0.0], color="black", lw=0.6, ls=(0, (3, 1.5)), zorder=1)
    ax.text(6.6, -0.1, "view 3", ha="left", va="top", fontsize=FS_BODY)
    # true spans
    dim(ax, -A, A, 0.0, None, lw=0.9)
    ax.text(-2.4, -0.3, "true AP span", ha="center", va="top", fontsize=FS_BODY)
    arrow(ax, (0, -B), (0, B), style="<|-|>", lw=0.9, ms=4)
    ax.text(0.3, 1.5, "true SL\nspan", ha="left", va="center", fontsize=FS_BODY)
    # orientation: AP axis horizontal (anterior left, posterior right), SL axis vertical
    # (septal top, lateral bottom); 6 pt grey so the labels do not compete with the spans
    FS_OR, C_OR = 6.0, "#444444"
    ax.text(0.0, B + 0.2, "septal", ha="center", va="bottom", fontsize=FS_OR, color=C_OR)
    ax.text(0.0, -B - 0.2, "lateral", ha="center", va="top", fontsize=FS_OR, color=C_OR)
    ax.text(-A - 0.3, 0.3, "anterior", ha="right", va="bottom", fontsize=FS_OR, color=C_OR)
    ax.text(A + 0.3, 0.3, "posterior", ha="left", va="bottom", fontsize=FS_OR, color=C_OR)
    # view 2 plane: offset parallel plane -> shorter chord (underestimation)
    yoff = -2.3
    half = A * np.sqrt(1 - (yoff / B) ** 2)
    ax.plot([-7.4, 6.4], [yoff, yoff], color=GREY, lw=0.6, ls=(0, (3, 1.5)), zorder=1)
    ax.plot([-half, half], [yoff, yoff], color="black", lw=1.8, solid_capstyle="butt", zorder=2)
    ax.text(6.6, yoff, "view 2", ha="left", va="center", fontsize=FS_BODY)
    ax.text(-6.9, -4.3, "View 2 (long axis), oblique or offset plane: chord\n"
            "shorter than the AP span (underestimation $u$)", ha="left", va="top",
            fontsize=FS_BODY)

    # ---------------------------------------------------------------- a (bottom): long-axis side view
    ax = ax_a2
    ax.set_xlim(-8.2, 8.2)
    ax.set_ylim(-8.2, 7.0)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.text(-6.9, 6.9, "View 3 (long axis): side view of the jet", ha="left", va="top",
            fontsize=FS_HEAD, fontweight="bold")
    # Systolic tenting: leaflets hinge at the annulus and bow toward the ventricle; their tips
    # meet (coaptation) below the annular plane. The jet leaves the coaptation gap toward the atrium.
    gap, y_tip, y_ann, x_hinge = 1.5, -2.6, 0.0, 6.0
    ax.plot([-7.4, 7.4], [y_ann, y_ann], color=GREY, lw=0.6, ls=(0, (3, 1.5)), zorder=1)
    ax.text(7.4, y_ann + 0.25, "annular\nplane", ha="right", va="bottom", fontsize=FS_BODY,
            color="black", linespacing=1.0)
    tt = np.linspace(0, 1, 40)
    for sgn in (-1, 1):
        p0, p1, p2 = np.array([sgn * x_hinge, y_ann]), np.array([sgn * 3.4, y_tip + 0.05]), \
            np.array([sgn * gap / 2, y_tip])          # quadratic Bezier bowed toward the ventricle
        cur = ((1 - tt) ** 2)[:, None] * p0 + (2 * (1 - tt) * tt)[:, None] * p1 + (tt ** 2)[:, None] * p2
        ax.plot(cur[:, 0], cur[:, 1], color="black", lw=1.6, solid_capstyle="round", zorder=3)
        ax.plot([sgn * x_hinge], [y_ann], marker="o", ms=2.6, color="black", zorder=4)
    ax.text(6.3, -1.3, "leaflet", ha="center", va="top", fontsize=FS_BODY)
    jet = Polygon([(-gap / 2, y_tip), (gap / 2, y_tip), (3.4, 4.2), (-3.4, 4.2)], closed=True,
                  fc="#BFBFBF", ec="none", zorder=0)
    ax.add_patch(jet)
    ax.text(0, 4.35, "atrium", ha="center", va="bottom", fontsize=FS_BODY)
    ax.text(4.2, -3.3, "ventricle", ha="center", va="top", fontsize=FS_BODY)
    # proximal jet width: the jet neck just atrial to the leaflet tips (not the gap between them)
    y_neck = y_tip + 0.8
    w_neck = gap / 2 + (3.4 - gap / 2) * (y_neck - y_tip) / (4.2 - y_tip)
    dim(ax, -w_neck, w_neck, y_neck, None, lw=0.9)
    ax.text(-7.4, -3.3, "proximal jet at coaptation\nlevel: correct span", ha="left",
            va="top", fontsize=FS_BODY, linespacing=1.05)
    ax.plot([-2.2, -w_neck + 0.15], [-3.3, y_neck - 0.1], color="black", lw=0.5)   # leader
    y_hi = 2.7
    w_hi = gap / 2 + (3.4 - gap / 2) * (y_hi - y_tip) / (4.2 - y_tip)
    dim(ax, -w_hi, w_hi, y_hi, None, lw=0.9)
    ax.text(w_hi + 0.4, y_hi, "above the leaflet tips:\nwider (overestimation $o$)",
            ha="left", va="center", fontsize=FS_BODY)
    ax.text(-6.9, -5.9, "Expected view mean: $\\mu_v = S\\,(1 - u_v + o_v)$,\n"
            "$S$ the true span on that axis", ha="left", va="top", fontsize=FS_BODY)

    # ---------------------------------------------------------------- b: beats per view
    ax = ax_b
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30.0)
    ax.text(0.0, 29.85, "b", ha="left", va="top", fontsize=FS_LETTER, fontweight="bold")
    ax.text(0.9, 29.85, "Beats measured in each view", ha="left", va="top", fontsize=FS_HEAD,
            fontweight="bold")
    ax.text(0.0, 28.5, "Spans vary from beat to beat (beat CV);\neach reading adds caliper "
            "error (bars).\nThe first N beats are tested against\n±W of their own mean.",
            ha="left", va="top", fontsize=FS_BODY, linespacing=1.15)
    views = [
        ("View 1 (anchor)", [9.3, 10.1, 9.7], [], 20.3, "window met\nwith N = 3"),
        ("View 2", [8.9, 6.6, 8.5, 8.2], [1], 12.9, "beat 2 outside\nthe first-3 window:\nbeat 4 "
         "added,\n3 most consistent\nbeats kept"),
        ("View 3", [11.6, 12.4, 11.9], [], 5.5, "window met\nwith N = 3"),
    ]
    yscale = 1.2   # drawing units per mm of span
    Wf = 0.15
    for name, vals, dropped, y0, note in views:
        kept = [v for i, v in enumerate(vals) if i not in dropped]
        m = float(np.mean(kept))
        ax.add_patch(Rectangle((0.2, y0 - Wf * m * yscale), 4.6, 2 * Wf * m * yscale,
                               fc="#DDDDDD", ec="none", zorder=0))
        m3 = float(np.mean(vals[:3]))
        if dropped:
            # window of the first three beats (the test that failed), centred on their own mean
            assert any(abs(v - m3) > Wf * m3 for v in vals[:3]), "first-3 window must fail"
            ax.add_patch(Rectangle((0.3, y0 + (m3 - m) * yscale - Wf * m3 * yscale), 3.2,
                                   2 * Wf * m3 * yscale, fc="none", ec="black", lw=0.6,
                                   ls=(0, (2.5, 1.5)), zorder=1))
        else:
            assert all(abs(v - m3) <= Wf * m3 for v in vals[:3]), "first-3 window must be met"
        # coded rule: kept beats lie within +-W of their own mean
        assert all(abs(v - m) <= Wf * m for i, v in enumerate(vals) if i not in dropped)
        ax.plot([0.2, 4.8], [y0, y0], color="black", lw=0.8)
        for i, v in enumerate(vals):
            x = 0.9 + 1.1 * i
            y = y0 + (v - m) * yscale
            open_ = i in dropped
            ax.errorbar(x, y, yerr=0.8 * yscale, fmt="o", ms=3.4, lw=0.6, capsize=1.2,
                        mfc="white" if open_ else "black", mec="black", ecolor="black",
                        mew=0.8)
            src.append(dict(figure="fig1_design", panel="b", view=name, beat=i + 1,
                            span_mm=v, kept=not open_, view_mean_mm=round(m, 3),
                            first3_mean_mm=round(m3, 3), window=Wf,
                            note="illustrative values, not simulation output"))
        # label sits above the grey band but below the lowest whisker of the view above
        ax.text(0.0, y0 + 2.5, name, ha="left", va="bottom", fontsize=FS_BODY,
                fontweight="bold")
        ax.text(5.2, y0, note, ha="left", va="center", fontsize=FS_BODY, linespacing=1.15)
    ax.text(0.0, 0.0, "Line and grey band, mean of the kept beats\n±W; dashed box, ±W around the "
            "mean of\nthe first 3 beats; open circle, beat not\nkept. If the window is never met, "
            "the\nview is flagged as limited sampling.", ha="left", va="bottom", fontsize=FS_BODY, linespacing=1.15)

    # ---------------------------------------------------------------- c: estimators and outputs
    ax = ax_c
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.text(0.0, 99.8, "c", ha="left", va="top", fontsize=FS_LETTER, fontweight="bold")
    ax.text(4.5, 99.8, "Estimators combine the beats and view means", ha="left", va="top",
            fontsize=FS_HEAD, fontweight="bold")
    rows = [
        ("A1", "anchor mean", "mean of the anchor-view beats"),
        ("A2", "mean", "mean of the view means"),
        ("A3", "maximum", "largest view mean"),
        ("A4", "composite rule", "anchor mean; a verification view mean exceeding it by at "
         "least\n$t_{warn}$ raises a warning, by at least $t_{adj}$ an adjudication; the reader\n"
         "rejects a view judged artefactual, otherwise keeps the\nhighest mean"),
        ("A5", "median", "median of the view means"),
        ("A6", "index beat", "one anchor-view beat, RR ratio closest to 1"),
        ("A7", "offset-corrected mean",
         "each view mean divided by its known (1 − $u$ + $o$), then averaged"),
    ]
    y = 95.0
    x_l, bw = 0.5, 99.0
    LH = 2.35   # line height in axes units at 6.5 pt
    for e, name, desc in rows:
        nlines = desc.count("\n") + 1
        h = 1.4 + LH * (nlines + 1) + 0.6
        box(ax, x_l, y - h, bw, h, fc="white", ec=es.COLOR[e], lw=1.6, r=0.8)
        ax.plot([x_l + 2.6], [y - 0.8 - LH / 2], marker=es.MARKER[e], ms=4, color=es.COLOR[e],
                mfc=es.mfc(e), mec=es.COLOR[e], mew=0.9, ls="none")
        ax.text(x_l + 5.0, y - 0.7, f"{e} {name}", ha="left", va="top", fontsize=FS_BODY,
                fontweight="bold")
        ax.text(x_l + 5.0, y - 0.7 - LH, desc, ha="left", va="top", fontsize=FS_BODY,
                linespacing=1.15)
        y -= h + 0.9
    y_est_bottom = y + 0.9
    # outputs: two branches, each a column of boxes (x0, top, heading, body)
    ow = 47.0
    XL, XR = 0.5, 52.5

    def obox(x0, top, head, body):
        nl = body.count("\n") + 1
        h = 1.4 + LH * (nl + 1) + 0.6
        box(ax, x0, top - h, ow, h, fc="#F2F2F2", ec="black", lw=0.6, r=0.8)
        ax.text(x0 + 2.0, top - 0.7, head, ha="left", va="top", fontsize=FS_BODY,
                fontweight="bold")
        ax.text(x0 + 2.0, top - 0.7 - LH, body, ha="left", va="top", fontsize=FS_BODY,
                linespacing=1.15)
        return top - h

    y_arrow_top = y_est_bottom - 0.4
    top_out = y_est_bottom - 4.0
    b1 = obox(XL, top_out, "Comparison with the truth",
              "bias and RMSE against the\ntrue spans $T_1$ and $T_2$;\n"
              "classification at 7, 10\nand 13 mm")
    b2 = obox(XR, top_out, "Reference for AI validation",
              "A4 reads: single read,\nmean of two reads, or\nadjudicated read")
    for x0 in (XL, XR):
        arrow(ax, (x0 + ow / 2, y_arrow_top), (x0 + ow / 2, top_out), lw=0.7)
    top2 = min(b1, b2) - 3.5
    b3 = obox(XR, top2, "AI with known true error",
              "error independent of the\nimages, inheriting the\nprotocol bias, or sharing\n"
              "image error")
    b4 = obox(XL, top2, "Apparent and true agreement",
              "MAE, Bland-Altman bias\nand limits, ICC of the AI\nagainst the reference\n"
              "and against $T_1$")
    arrow(ax, (XR + ow / 2, b2), (XR + ow / 2, top2), lw=0.7)
    ymid = (top2 + b3) / 2
    arrow(ax, (XR, ymid), (XL + ow, ymid), lw=0.7)
    hbox = top2 - b3
    assert top2 - hbox > 0, top2 - hbox

    pd.DataFrame(src).to_csv(os.path.join(FIG, "fig1_design_source.csv"), index=False)
    stem = os.path.join(FIG, "fig1_design")
    qa(fig, stem, W_MM)
    plt.close(fig)

# ============================================================================ GA numbers
def ga_numbers():
    rows = []

    def add(name, value, mcse, source, note=""):
        rows.append(dict(name=name, value=value, mcse=mcse, source=source, note=note))

    # ---- E1b base slice, AP, T1 (amendment 2)
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(AM, "E1_cells_*.parquet")))])
    d = d.drop(columns=["p_u_scale"])
    s = d[(d.p_K == 3) & (d.p_beat_cv == 0.15) & (d.p_view_over) & (d.p_window.isna())
          & (d.p_S_median_mm == 10.0) & (d.axis == "AP") & (d.estimand == "T1")
          & (d.metric == "bias_mm") & d.estimator.isin(["A1", "A2", "A3", "A4", "A5", "A6", "A7"])]
    s = s[(s.estimator != "A4") | ((s.s_det == 0.8) & (s.f_rej == 0.1))]
    s = s[["cell", "p_u_anchor", "p_u_long", "estimator", "value", "mcse", "n"]].reset_index(drop=True)
    s["anchor_mean_under_pct"] = (0.798 * s.p_u_anchor * 100).round(1)
    s["long_mean_under_pct"] = (0.798 * s.p_u_long * 100).round(0)
    assert len(s) == 7 * 12, len(s)
    s.sort_values(["estimator", "p_u_anchor", "p_u_long"]).to_csv(
        os.path.join(AN_OUT, "schematic_e1b_base_ap_bias.csv"), index=False)
    brief = {"A1": (0.01, -1.93), "A2": (-0.28, -3.16), "A3": (0.92, -0.77), "A4": (0.76, -0.97)}
    for e in ["A1", "A2", "A3", "A4"]:
        g = s[s.estimator == e]
        imax, imin = g.value.idxmax(), g.value.idxmin()
        gx, gn = g.loc[[imax]].iloc[0], g.loc[[imin]].iloc[0]
        for lab, r in (("max", gx), ("min", gn)):
            add(f"E1b {e} AP bias vs T1, {lab} over 12 view-accuracy scenarios", r.value, r.mcse,
                f"amend2 E1 cell {int(r.cell)}, estimator {e}, AP, T1, bias_mm"
                + (", s_det 0.8, f_rej 0.1" if e == "A4" else ""),
                f"u_anchor {r.p_u_anchor}, u_long {r.p_u_long}")
        # brief check at 0.02 mm tolerance (rounding)
        assert abs(gx.value - brief[e][0]) < 0.02 and abs(gn.value - brief[e][1]) < 0.02, (
            e, gx.value, gn.value, brief[e])

    # ---- E2 main run: window and RMSE vs T1, A1, sinus, CV 0.15, prospective, AP
    e2 = pd.read_csv(os.path.join(AN_FULL, "E2_error.csv"))
    e2 = e2[(e2.rhythm_state == "sinus") & (e2.p_beat_cv == 0.15) & (e2.p_data_mode == "prospective")
            & (e2.axis == "AP") & (e2.estimator == "A1") & (e2.estimand == "T1")
            & (e2.metric == "rmse_mm") & (e2.subset == "all") & (e2.W.isin([0.0, 0.15]))]
    rmse = e2[["W", "p_N_beats", "value", "mcse"]].sort_values(["W", "p_N_beats"])
    for _, r in rmse.iterrows():
        add(f"E2 A1 RMSE vs T1, sinus, CV 15%, N {int(r.p_N_beats)}, W {r.W}", r.value, r.mcse,
            "full/analysis/E2_error.csv: sinus, 0.15, prospective, AP, A1, subset all, T1, rmse_mm")
    we = pd.read_csv(os.path.join(AN_FULL, "E2_window_effect.csv"))
    we = we[(we.rhythm_state == "sinus") & (we.p_beat_cv == 0.15) & (we.p_N_beats == 3)
            & (we.p_data_mode == "prospective") & (we.axis == "AP") & (we.estimator == "A1")
            & (we.estimand == "T1") & (we.metric == "rmse_mm") & (we.W == 0.15)].iloc[0]
    add("E2 window effect on A1 RMSE vs T1, N 3, W 0.15 (windowed minus none)", we["diff"],
        we.mcse_diff, "full/analysis/E2_window_effect.csv: sinus, 0.15, N 3, prospective, AP, A1, T1, rmse_mm, W 0.15",
        f"95% MCI {we.lo95:.3f} to {we.hi95:.3f}")
    wm = pd.read_csv(os.path.join(AN_FULL, "E2_window_met.csv"))
    wm = wm[(wm.rhythm_state == "sinus") & (wm.p_beat_cv == 0.15) & (wm.p_data_mode == "prospective")
            & (wm.axis == "AP") & (wm.p_N_beats == 3) & (wm.W == 0.15)].iloc[0]
    add("E2 P(window 15% met by first 3 beats), sinus, CV 15%, AP", wm.value, wm.mcse,
        "full/analysis/E2_window_met.csv: sinus, 0.15, W 0.15, N 3, prospective, AP")
    assert abs(wm.value - 0.418) < 0.001
    r3 = rmse.set_index(["W", "p_N_beats"]).value
    assert abs(r3[(0.0, 3)] - 1.942) < 0.001 and abs(r3[(0.15, 3)] - 2.041) < 0.001

    # ---- E4 main run: triggers, base cell 193 (view errors) and 49 (noise only), AP
    op = pd.read_csv(os.path.join(AN_FULL, "E4_operating.csv"))
    op = op[(op.cell == 193) & (op.axis == "AP")].set_index("tau")
    for tau in (3.0, 5.0):
        r = op.loc[tau]
        add(f"E4 hit rate for true error > 2 mm at threshold {tau:g} mm, AP (per pair)", r.hit,
            r.hit_mcse * np.sqrt(3), f"full/analysis/E4_operating.csv: cell 193, AP, tau {tau:g}",
            "MCSE conservative (pair SE x sqrt 3)")
        add(f"E4 false-alarm rate at threshold {tau:g} mm, AP (per pair)", r.false_alarm,
            r.false_alarm_mcse, f"full/analysis/E4_operating.csv: cell 193, AP, tau {tau:g}")
    at = pd.read_csv(os.path.join(AN_FULL, "E4_any_trigger.csv"))
    at = at[(at.axis == "AP")]
    for cell, lab in ((49, "no view errors"), (193, "view errors")):
        for metric, tau in (("p_any_trigger", 3.0), ("p_any_adj", 5.0)):
            r = at[(at.cell == cell) & (at.metric == metric) & (at.tau == tau)].iloc[0]
            add(f"E4 {metric} at {tau:g} mm, AP, {lab}", r.value, r.mcse,
                f"full/analysis/E4_any_trigger.csv: cell {cell}, AP, {metric}, tau {tau:g}")
    assert abs(op.loc[3.0].hit - 0.216) < 0.001 and abs(op.loc[5.0].hit - 0.088) < 0.001

    # ---- E5b (amendment 2): shared-image-error AI, lambda 1, sigma_AI 2, base (cell 11)
    e5 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(AM, "E5_cells_*.parquet")))])
    e5 = e5[(e5.p_beat_cv == 0.15) & (e5.p_view_over) & (e5.p_ai_shared_lambda == 1.0)]
    assert e5.cell.nunique() == 1
    cell5 = int(e5.cell.iloc[0])

    def g5(ai, ref, metric):
        r = e5[(e5.ai == ai) & (e5.sigma_ai_mm == 2.0) & (e5.ref == ref) & (e5.metric == metric)]
        assert len(r) == 1, (ai, ref, metric, len(r))
        return r.iloc[0]
    for ai in ("independent", "shared"):
        for metric in ("true_mae", "apparent_mae"):
            r = g5(ai, "mean2", metric)
            add(f"E5b {ai} AI {metric}, sigma_AI 2 mm, lambda 1, reference mean of 2 reads",
                r.value, r.mcse, f"amend2 E5 cell {cell5}, ai {ai}, ref mean2, {metric}")
    for ref in ("mean2", "single"):
        r = g5("shared", ref, "p_lower_mae_than_independent")
        add(f"E5b P(shared AI apparent MAE < independent AI), ref {ref}, lambda 1, sigma 2",
            r.value, r.mcse, f"amend2 E5 cell {cell5}, ai shared, ref {ref}, p_lower_mae_than_independent")
        r = g5("shared", ref, "paired_mae_diff_mean")
        add(f"E5b paired apparent MAE difference shared minus independent, ref {ref}",
            r.value, r.mcse, f"amend2 E5 cell {cell5}, ai shared, ref {ref}, paired_mae_diff_mean")
    r = g5("shared", "T1", "paired_mae_diff_mean")
    add("E5b paired true MAE difference shared minus independent (vs T1)", r.value, r.mcse,
        f"amend2 E5 cell {cell5}, ai shared, ref T1, paired_mae_diff_mean")

    # ---- E5 main run: inherited-bias AI bias transfer, cell 5, sigma 2, mean of 2 reads
    ag = pd.read_csv(os.path.join(AN_FULL, "E5E6_E5_ai_agreement.csv"))
    b = ag[(ag.cell == 5) & (ag.sigma_ai_mm == 2.0) & (ag.ref == "mean2") & (ag.m == "ba_bias")]
    for ai in ("independent", "inherited"):
        r = b[b.ai == ai].iloc[0]
        add(f"E5 {ai} AI Bland-Altman bias true (vs T1), sigma 2", r["true"], r.mcse_true,
            f"full/analysis/E5E6_E5_ai_agreement.csv: cell 5, {ai}, 2.0, mean2, ba_bias, true")
        add(f"E5 {ai} AI Bland-Altman bias apparent vs mean of 2 reads, sigma 2", r.apparent,
            r.mcse_apparent,
            f"full/analysis/E5E6_E5_ai_agreement.csv: cell 5, {ai}, 2.0, mean2, ba_bias, apparent")
    r = ag[(ag.cell == 5) & (ag.sigma_ai_mm == 2.0) & (ag.ref == "single") & (ag.m == "mae")
           & (ag.ai == "independent")].iloc[0]
    add("E5 independent AI MAE true, sigma 2", r["true"], r.mcse_true,
        "full/analysis/E5E6_E5_ai_agreement.csv: cell 5, independent, 2.0, single, mae, true")
    add("E5 independent AI MAE apparent vs single read, sigma 2", r.apparent, r.mcse_apparent,
        "full/analysis/E5E6_E5_ai_agreement.csv: cell 5, independent, 2.0, single, mae, apparent")
    add("E5 independent AI MAE relative inflation vs single read (%), sigma 2", r.rel_diff_pct,
        np.nan, "full/analysis/E5E6_E5_ai_agreement.csv: cell 5, independent, 2.0, single, mae, rel_diff_pct")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(AN_OUT, "schematic_ga_numbers.csv"), index=False)
    return out, s, rmse


# ============================================================================ Graphical abstract
def ga():
    nums, s, rmse = ga_numbers()
    N = nums.set_index("name")

    def v(name):
        return float(N.loc[name, "value"])

    W_MM, H_MM = 180, 110
    fig = plt.figure(figsize=figstyle.mm(W_MM, H_MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06, hspace=0.08)
    fig.suptitle("Composite multi-view echocardiographic reference rules for tricuspid "
                 "regurgitant jet span: simulation results", fontsize=8.5, fontweight="bold",
                 x=0.01, ha="left")
    gs = fig.add_gridspec(2, 2)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    src = []
    FS = 7.0

    def fmt(x):
        return f"{x:+.2f}".replace("-", "\u2212")
    TFS = 8.0

    # ---- panel a: bias range across view accuracy
    ax = axs[0]
    ests = ["A1", "A2", "A3", "A4"]
    names = {"A1": "Anchor\nmean", "A2": "Mean of\nviews", "A3": "Maximum\nof views",
             "A4": "Composite\nrule"}
    ax.axhspan(-1, 1, color="#E6E6E6", zorder=0, lw=0)
    ax.axhline(0, color="black", lw=0.5, zorder=1)
    for i, e in enumerate(ests):
        g = s[s.estimator == e]
        lo, hi = g.value.min(), g.value.max()
        ax.plot([i, i], [lo, hi], color=es.COLOR[e], lw=2.2, solid_capstyle="butt", zorder=2)
        xs = i + np.linspace(-0.13, 0.13, len(g))
        gg = g.sort_values(["p_u_anchor", "p_u_long"])
        ax.plot(xs, gg.value, ls="none", marker=es.MARKER[e], ms=3.0, mfc=es.mfc(e),
                mec=es.COLOR[e], mew=0.7, zorder=3)
        ax.text(i + 0.24, hi, fmt(hi), ha="left", va="center", fontsize=FS)
        ax.text(i + 0.24, lo, fmt(lo), ha="left", va="center", fontsize=FS)
        for _, r in gg.iterrows():
            src.append(dict(panel="a", estimator=e, u_anchor=r.p_u_anchor, u_long=r.p_u_long,
                            bias_mm=r.value, mcse=r.mcse, source_cell=int(r.cell)))
    ax.set_xticks(range(4), [names[e] for e in ests], fontsize=FS)
    ax.set_xlim(-0.5, 3.75)
    ax.set_ylim(-4.35, 1.6)
    ax.text(-0.45, -4.25, "Points left to right: anchor underestimation 2.4% to 20%,\n"
            "long-axis 8% to 40% within each; no beat window", ha="left", va="bottom",
            fontsize=6.5, color="#444444", linespacing=1.05)
    ax.set_yticks([-3, -2, -1, 0, 1])
    ax.tick_params(axis="y", labelsize=FS)
    ax.set_ylabel("Bias against true span (mm)", fontsize=FS)
    ax.set_title("a  Maximum-type rules stay within about ±1 mm across\n    unknown view "
                 "accuracy; mean-type rules reach −3 mm", fontsize=TFS, fontweight="bold")
    ax.text(-0.45, 1.05, "grey band ±1 mm", ha="left", va="bottom", fontsize=FS, color="#444444")

    # ---- panel b: beat window adds error
    ax = axs[1]
    for W, ls, mfc, lab in ((0.0, "-", "black", "no window"), (0.15, (0, (3, 1.5)), "white",
                                                                 "±15% window")):
        g = rmse[rmse.W == W]
        ax.plot(g.p_N_beats, g.value, ls=ls, color="black", marker="o", ms=3.2, mfc=mfc,
                mec="black", mew=0.8, lw=0.9)
        for _, r in g.iterrows():
            src.append(dict(panel="b", W=W, N_beats=int(r.p_N_beats), rmse_mm=r.value,
                            mcse=r.mcse))
    g0 = rmse[rmse.W == 0.0].set_index("p_N_beats").value
    g1 = rmse[rmse.W == 0.15].set_index("p_N_beats").value
    ax.text(13.4, g1[13] + 0.02, "±15% window", ha="left", va="bottom", fontsize=FS)
    ax.text(13.4, g0[13] - 0.02, "no window", ha="left", va="top", fontsize=FS)
    pmet = v("E2 P(window 15% met by first 3 beats), sinus, CV 15%, AP")
    ax.annotate(f"3 beats: {g0[3]:.2f} to {g1[3]:.2f} mm\nwindow met by the first 3 beats\n"
                f"in {100 * pmet:.0f}% of views", xy=(3, g1[3]), xytext=(5.2, 2.35),
                fontsize=FS, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="black", shrinkA=1, shrinkB=2))
    ax.set_xticks([1, 3, 5, 7, 10, 13])
    ax.set_xlim(0.5, 17.5)
    ax.set_ylim(1.5, 2.6)
    ax.spines["bottom"].set_bounds(1, 13)
    ax.tick_params(labelsize=FS)
    ax.set_xlabel("Beats averaged in the anchor view", fontsize=FS)
    ax.set_ylabel("Root-mean-square error (mm)", fontsize=FS)
    ax.set_title("b  A percentage beat window adds error\n    to the anchor-view mean",
                 fontsize=TFS, fontweight="bold")

    # ---- panel c: triggers detect few true errors
    ax = axs[2]
    h3 = v("E4 hit rate for true error > 2 mm at threshold 3 mm, AP (per pair)")
    h5 = v("E4 hit rate for true error > 2 mm at threshold 5 mm, AP (per pair)")
    n3 = v("E4 p_any_trigger at 3 mm, AP, no view errors")
    n5 = v("E4 p_any_adj at 5 mm, AP, no view errors")
    cats = ["True view errors > 2 mm\nthat trigger", "Patients triggered with\nno view error"]
    y = np.array([1.0, 0.0])
    bh = 0.34
    ax.barh(y + bh / 2, [100 * h3, 100 * n3], height=bh, color="#555555", ec="black", lw=0.5,
            label="warning, 3 mm")
    ax.barh(y - bh / 2, [100 * h5, 100 * n5], height=bh, color="white", ec="black", lw=0.5,
            hatch="//////", label="adjudication, 5 mm")
    for yy, val in ((1 + bh / 2, h3), (0 + bh / 2, n3), (1 - bh / 2, h5), (0 - bh / 2, n5)):
        ax.text(100 * val + 0.8, yy, f"{100 * val:.1f}%", ha="left", va="center", fontsize=FS)
    for yy, (a, b) in ((1, (h3, h5)), (0, (n3, n5))):
        pass
    src += [dict(panel="c", quantity="hit rate, true error > 2 mm", threshold_mm=3, value=h3),
            dict(panel="c", quantity="hit rate, true error > 2 mm", threshold_mm=5, value=h5),
            dict(panel="c", quantity="P(any trigger), no view errors", threshold_mm=3, value=n3),
            dict(panel="c", quantity="P(any adjudication), no view errors", threshold_mm=5, value=n5)]
    ax.set_yticks(y, cats, fontsize=FS)
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.6, 1.9)
    ax.tick_params(axis="x", labelsize=FS)
    ax.set_xlabel("Percentage of true errors or of patients (%)", fontsize=FS)
    ax.legend(loc="upper right", fontsize=FS, frameon=False, handlelength=1.2,
              bbox_to_anchor=(1.0, 1.02))
    ax.set_title("c  Fixed-millimetre cross-view triggers detect few true\n    errors and fire on "
                 "measurement noise", fontsize=TFS, fontweight="bold")

    # ---- panel d: reference hides/inflates AI error
    ax = axs[3]
    ti = v("E5b independent AI true_mae, sigma_AI 2 mm, lambda 1, reference mean of 2 reads")
    ai_ = v("E5b independent AI apparent_mae, sigma_AI 2 mm, lambda 1, reference mean of 2 reads")
    ts = v("E5b shared AI true_mae, sigma_AI 2 mm, lambda 1, reference mean of 2 reads")
    as_ = v("E5b shared AI apparent_mae, sigma_AI 2 mm, lambda 1, reference mean of 2 reads")
    pl = v("E5b P(shared AI apparent MAE < independent AI), ref mean2, lambda 1, sigma 2")
    rowsd = [(1.0, "AI with error\nindependent of\nthe images", ti, ai_),
             (0.0, "AI sharing the\nreference's\nimage error", ts, as_)]
    for yy, lab, t, a in rowsd:
        sgn = 1 if a > t else -1
        arrow(ax, (t + sgn * 0.03, yy), (a - sgn * 0.035, yy), style="-|>", lw=0.9, ms=7)
        ax.plot([t], [yy], marker="o", ms=4.5, color="black", ls="none", zorder=3)
        ax.plot([a], [yy], marker="s", ms=4.5, mfc="white", mec="black", mew=0.9, ls="none",
                zorder=3)
        ax.text(t, yy + 0.2, f"{t:.2f}", ha="center", va="bottom", fontsize=FS)
        ax.text(a, yy + 0.2, f"{a:.2f}", ha="center", va="bottom", fontsize=FS)
        src.append(dict(panel="d", ai=lab.replace("\n", " "), true_mae=t, apparent_mae=a))
    ax.plot([], [], marker="o", ms=4.5, color="black", ls="none", label="true (against the true span)")
    ax.plot([], [], marker="s", ms=4.5, mfc="white", mec="black", ls="none",
            label="apparent (against mean of 2 reads)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=1, fontsize=FS,
              frameon=False, handletextpad=0.3, borderaxespad=0.2)
    ax.text(1.395, 0.5, f"Ranking reversed in {100 * pl:.1f}% of validation studies",
            ha="left", va="center", fontsize=FS, style="normal")
    ax.set_yticks([1, 0], [r[1] for r in rowsd], fontsize=FS)
    ax.set_ylim(-0.75, 1.55)
    ax.set_xlim(1.35, 2.35)
    ax.set_xticks([1.4, 1.6, 1.8, 2.0, 2.2])
    ax.tick_params(axis="x", labelsize=FS)
    ax.set_xlabel("Mean absolute error of the AI (mm)", fontsize=FS)
    ax.set_title("d  Protocol-built references inflate\n    independent AI error, hide "
                 "shared error", fontsize=TFS, fontweight="bold")


    pd.DataFrame(src).to_csv(os.path.join(FIG, "graphical_abstract_source.csv"), index=False)
    stem = os.path.join(FIG, "graphical_abstract")
    qa(fig, stem, W_MM)
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("fig1", "all"):
        fig1()
    if what in ("ga", "all"):
        ga()
