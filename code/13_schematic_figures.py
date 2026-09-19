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

Run: /project/home/p201509/envs/duomax-sim/bin/python code/13_schematic_figures.py [fig1|fig1alt|ga|all]\n  fig1alt writes figures/fig1_design_alt.png only (alternative layout, not a deliverable).
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
from matplotlib.layout_engine import PlaceHolderLayoutEngine  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
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
def qa(fig, stem, width_mm, min_pt=6.0):
    paths = figstyle.save_all(fig, stem)
    probs = figqa.report(fig, min_pt=min_pt)
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
# v05 (2026-09-19): clean schematic. The legend carries the explanation; the figure carries at most
# about four words per label and no formulas. Drawn on a millimetre canvas: every sub-drawing is an
# axes placed in figure millimetres, with an exact drawing scale, so geometry does not depend on a
# layout engine. Two layouts: "row" (a | b | c, main output) and "stack" (a and b stacked left, c right,
# saved as fig1_design_alt.png only).
# v06 (2026-09-19): EHJ-CVI 2 mm text floor -> body 8 pt, panel titles 9 pt bold, letters 9 pt bold
F1_BODY, F1_HEAD, F1_LETTER = 8.0, 9.0, 9.0   # body text; panel titles and letters bold
C_LINE = "#8C8C8C"      # thin grey box outlines
C_FILL = "#F2F2F2"      # light box fill
C_OR = "#555555"        # orientation labels
R_BOX = 1.0             # corner radius (mm) of every box


class MMCanvas:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=figstyle.mm(W, H), layout="none")
        # rc turns constrained layout on, and matplotlib re-reads it at savefig when the engine is None;
        # a do-nothing engine keeps this absolute millimetre canvas untouched
        self.fig.set_layout_engine(PlaceHolderLayoutEngine(adjust_compatible=True,
                                                           colorbar_gridspec=True))
        # overlay axes in figure millimetres (y up), for panel headings
        self.ov = self.fig.add_axes([0, 0, 1, 1])
        self.ov.set_xlim(0, W)
        self.ov.set_ylim(0, H)
        self.ov.set_xticks([])
        self.ov.set_yticks([])
        self.ov.set_axis_off()
        self.ov.set_zorder(-1)

    def axes(self, x0, y0, w, h):
        """Axes at (x0, y0) mm from the lower-left corner, w x h mm."""
        return self.fig.add_axes([x0 / self.W, y0 / self.H, w / self.W, h / self.H])

    def drawing(self, x0, ytop, xlim, ylim, s):
        """Axes for a drawing at s mm per data unit, top-left corner at (x0, ytop) mm."""
        w, h = (xlim[1] - xlim[0]) * s, (ylim[1] - ylim[0]) * s
        ax = self.axes(x0, ytop - h, w, h)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        return ax

    def heading(self, x, ytop, letter, title):
        self.ov.text(x, ytop, letter, ha="left", va="top", fontsize=F1_LETTER, fontweight="bold")
        self.ov.text(x + 3.6, ytop, title, ha="left", va="top", fontsize=F1_HEAD,
                     fontweight="bold")


def draw_enface(ax):
    """Orifice en face: true AP and SL spans, orientation, one oblique plane with a shorter chord."""
    A, B = 5.0, 3.2   # semi-axes: AP (horizontal) and SL (vertical), SL/AP = 0.64
    ax.add_patch(Ellipse((0, 0), 2 * A, 2 * B, fc="#E8E8E8", ec="black", lw=0.8, zorder=1))
    arrow(ax, (-A, 0), (A, 0), style="<|-|>", lw=0.9, ms=4)
    arrow(ax, (0, -B), (0, B), style="<|-|>", lw=0.9, ms=4)
    ax.text(-2.5, 0.25, "AP", ha="center", va="bottom", fontsize=F1_BODY)
    ax.text(0.3, 1.6, "SL", ha="left", va="center", fontsize=F1_BODY)
    # orientation: AP horizontal (anterior left, posterior right), SL vertical (septal top,
    # lateral bottom); unchanged from v04, to be checked by the co-author
    ax.text(0.0, B + 0.25, "septal", ha="center", va="bottom", fontsize=F1_BODY, color=C_OR)
    ax.text(0.0, -B - 0.25, "lateral", ha="center", va="top", fontsize=F1_BODY, color=C_OR)
    ax.text(-A - 0.3, 0.0, "anterior", ha="right", va="center", fontsize=F1_BODY, color=C_OR)
    ax.text(A + 0.3, 0.0, "posterior", ha="left", va="center", fontsize=F1_BODY, color=C_OR)
    # oblique imaging plane y = m x + c: dashed trace, chord inside the orifice drawn heavy
    m, c = -0.20, -2.05
    qa_, qb, qc = 1 / A ** 2 + m ** 2 / B ** 2, 2 * m * c / B ** 2, c ** 2 / B ** 2 - 1
    xs = np.sort(np.roots([qa_, qb, qc]).real)
    chord = np.hypot(xs[1] - xs[0], m * (xs[1] - xs[0]))
    assert chord < 2 * A, chord          # the oblique chord is shorter than the true AP span
    xe = np.array([-5.9, 6.4])
    ax.plot(xe, m * xe + c, color="black", lw=0.6, ls=(0, (3, 1.5)), zorder=2)
    ax.plot(xs, m * xs + c, color="black", lw=1.8, solid_capstyle="butt", zorder=3)
    ax.text(2.6, -3.75, "oblique plane:\nshorter chord", ha="left", va="top",
            fontsize=F1_BODY, linespacing=1.1)


def draw_sideview(ax):
    """Long-axis side view of the jet: tented leaflets, funnel-shaped jet, two width markers.
    Geometry unchanged from v04."""
    gap, y_tip, y_ann, x_hinge = 1.5, -2.6, 0.0, 6.0
    ax.plot([-7.0, 7.0], [y_ann, y_ann], color=C_LINE, lw=0.5, ls=(0, (3, 1.5)), zorder=1)
    tt = np.linspace(0, 1, 40)
    for sgn in (-1, 1):
        p0, p1, p2 = np.array([sgn * x_hinge, y_ann]), np.array([sgn * 3.4, y_tip + 0.05]), \
            np.array([sgn * gap / 2, y_tip])          # quadratic Bezier bowed toward the ventricle
        cur = ((1 - tt) ** 2)[:, None] * p0 + (2 * (1 - tt) * tt)[:, None] * p1 + (tt ** 2)[:, None] * p2
        ax.plot(cur[:, 0], cur[:, 1], color="black", lw=1.6, solid_capstyle="round", zorder=3)
        ax.plot([sgn * x_hinge], [y_ann], marker="o", ms=2.6, color="black", zorder=4)
    ax.add_patch(Polygon([(-gap / 2, y_tip), (gap / 2, y_tip), (3.4, 4.2), (-3.4, 4.2)],
                         closed=True, fc="#BFBFBF", ec="none", zorder=0))
    ax.text(0, 4.4, "atrium", ha="center", va="bottom", fontsize=F1_BODY)
    ax.text(0, -3.0, "ventricle", ha="center", va="top", fontsize=F1_BODY)

    def width_at(y):
        return gap / 2 + (3.4 - gap / 2) * (y - y_tip) / (4.2 - y_tip)
    # measured span: the jet neck just atrial to the leaflet tips
    y_neck = y_tip + 0.8
    w = width_at(y_neck)
    arrow(ax, (-w, y_neck), (w, y_neck), style="<|-|>", lw=0.9, ms=4)
    ax.text(-4.3, 1.2, "measured\nspan", ha="right", va="center", fontsize=F1_BODY,
            linespacing=1.1)
    ax.plot([-4.2, -w - 0.1], [1.0, y_neck + 0.1], color="black", lw=0.5)     # leader
    # higher in the atrium: too wide
    y_hi = 2.7
    w = width_at(y_hi)
    arrow(ax, (-w, y_hi), (w, y_hi), style="<|-|>", lw=0.9, ms=4)
    ax.text(w + 0.4, y_hi, "above tips:\ntoo wide", ha="left", va="center", fontsize=F1_BODY,
            linespacing=1.1)


BEATS = [8.9, 6.6, 8.5, 8.2]     # illustrative view-2 spans (mm), beats 1 to 4
BEAT_OUT = 1                     # beat 2 (index 1) excluded
CAL_ERR = 0.8                    # illustrative caliper error bar (mm)
WIN = 0.15


def draw_beats(ax, src):
    kept = [v for i, v in enumerate(BEATS) if i != BEAT_OUT]
    m = float(np.mean(kept))
    lo, hi = m * (1 - WIN), m * (1 + WIN)
    assert all(lo <= v <= hi for v in kept) and not lo <= BEATS[BEAT_OUT] <= hi
    ax.axhspan(lo, hi, xmin=0.0, xmax=1.0, color="#E0E0E0", lw=0, zorder=0)
    ax.axhline(m, color="#7A7A7A", lw=0.6, zorder=1)
    for i, v in enumerate(BEATS):
        out = i == BEAT_OUT
        ax.errorbar(i + 1, v, yerr=CAL_ERR, fmt="o", ms=4.0, lw=0.6, capsize=1.5,
                    mfc="white" if out else "black", mec="black", ecolor="black", mew=0.8,
                    zorder=3)
        src.append(dict(figure="fig1_design", panel="b", view="View 2", beat=i + 1, span_mm=v,
                        caliper_error_bar_mm=CAL_ERR, kept=not out, kept_mean_mm=round(m, 3),
                        window=WIN, window_lo_mm=round(lo, 3), window_hi_mm=round(hi, 3),
                        note="illustrative values, not simulation output"))
    ax.text(4.45, hi + 0.12, "±15% window", ha="right", va="bottom", fontsize=F1_BODY)
    ax.text(2.2, BEATS[BEAT_OUT], "excluded", ha="left", va="center", fontsize=F1_BODY)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(5.0, 11.0)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([5, 7, 9, 11])
    ax.tick_params(labelsize=F1_BODY)
    ax.set_xlabel("Beat", fontsize=F1_BODY)
    ax.set_ylabel("Span (mm)", fontsize=F1_BODY)


def rbox(ax, x, y, w, h, text=None, fc=C_FILL, ec=C_LINE, lw=0.5, bold=False, tx=None, **k):
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=R_BOX)
    if text:
        ax.text(x + w / 2 if tx is None else tx, y + h / 2, text,
                ha="center" if tx is None else "left", va="center", fontsize=F1_BODY,
                fontweight="bold" if bold else "normal", linespacing=1.1, **k)


def draw_flow(ax, w, h):
    """Views -> four rules -> two outputs. Coordinates are millimetres inside the axes."""
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    AMS = 5.5                          # arrow head size
    gap_v = 5.0                        # vertical gap spanned by an arrow
    # views
    rw = w - 9.0                       # width of the rule container; the view row matches it
    bh, vg = 6.0, 2.0                  # v06: view gap 2.5 -> 2.0 mm so "Anchor view" clears its box at 8 pt
    vw = (rw - 2 * vg) / 3
    y = h - bh
    xs_v = [0.5 + i * (vw + vg) for i in range(3)]
    for x, lab in zip(xs_v, ["Anchor view", "View 2", "View 3"]):
        rbox(ax, x, y, vw, bh, lab)
    # rules: white rows inside one light container, so an arrow from the container means "each rule"
    pad = 1.6
    rh, rg = 6.0, 1.4
    top_rules = y - gap_v
    for x in xs_v:
        arrow(ax, (x + vw / 2, y), (x + vw / 2, top_rules), lw=0.6, ms=AMS)
    yr = top_rules - pad
    ymid = {}
    rows_y = []
    for e in ["A1", "A2", "A3", "A4"]:
        yr -= rh
        rows_y.append(yr)
        ymid[e] = yr + rh / 2
        yr -= rg
    bottom_rules = yr + rg - pad
    rbox(ax, 0.5, bottom_rules, rw, top_rules - bottom_rules)
    for e, yr in zip(["A1", "A2", "A3", "A4"], rows_y):
        rbox(ax, 0.5 + pad, yr, rw - 2 * pad, rh, fc="white")
        ax.plot([0.5 + pad + 3.5], [yr + rh / 2], marker=es.MARKER[e], ms=4.5, color=es.COLOR[e],
                mfc=es.mfc(e), mec=es.COLOR[e], mew=0.9, ls="none")
        ax.text(0.5 + pad + 7.0, yr + rh / 2, es.LABEL[e], ha="left", va="center",
                fontsize=F1_BODY)
    # outputs
    ow, oh = (w - 1.0 - 4.0) / 2, 9.5
    xl, xr = 0.5, 0.5 + ow + 4.0
    top_out = bottom_rules - gap_v
    rbox(ax, xl, top_out - oh, ow, oh, "Accuracy against\nthe true span")
    rbox(ax, xr, top_out - oh, ow, oh, "Reference for\nAI validation")
    arrow(ax, (xl + ow / 2, bottom_rules), (xl + ow / 2, top_out), lw=0.6, ms=AMS)
    # the reference is read with the largest view mean after review only: elbow from that row
    xe = 0.5 + rw + 4.0
    ax.plot([0.5 + rw - pad, xe, xe], [ymid["A4"], ymid["A4"], top_out + 0.01], color="black", lw=0.6)
    arrow(ax, (xe, top_out + 0.6), (xe, top_out), lw=0.6, ms=AMS)
    top_ai = top_out - oh - gap_v
    # v06: two lines at the height of the other output boxes (one line filled the box edge to edge at 8 pt)
    rbox(ax, xr, top_ai - oh, ow, oh, "Apparent vs true\nAI error")
    arrow(ax, (xr + ow / 2, top_out - oh), (xr + ow / 2, top_ai), lw=0.6, ms=AMS)
    return top_ai - oh          # lowest y used


def fig1(layout="row"):
    src = []
    if layout == "row":
        # v06 (8 pt text): 78 -> 82 mm tall, panel c starts 3 mm further left and 3 mm wider; drawings,
        # plot and flow unchanged in scale
        W_MM, H_MM = 183, 82
        cv = MMCanvas(W_MM, H_MM)
        top = H_MM - 1.0
        # a: 0-60 mm
        cv.heading(0.5, top, "a", "Sources of error")
        ax = cv.drawing(0.5, top - 6.5, (-9.6, 9.6), (-5.6, 4.5), 3.0)
        draw_enface(ax)
        ax = cv.drawing(3.0, top - 43.0, (-8.8, 8.8), (-4.0, 5.3), 3.0)
        draw_sideview(ax)
        # b: 64-107 mm
        cv.heading(64.0, top, "b", "Beat-consistency window")
        ax = cv.axes(75.0, 16.0, 31.0, 50.0)
        draw_beats(ax, src)
        # c: 112-183 mm
        cv.heading(112.0, top, "c", "View rules and outputs")
        ax = cv.axes(112.0, 1.5, 70.5, top - 7.0 - 1.5)
        low = draw_flow(ax, 70.5, top - 7.0 - 1.5)
        stem = os.path.join(FIG, "fig1_design")
    else:
        W_MM, H_MM = 183, 106
        cv = MMCanvas(W_MM, H_MM)
        top = H_MM - 1.0
        cv.heading(0.5, top, "a", "Sources of error")
        ax = cv.drawing(0.5, top - 6.0, (-9.6, 9.6), (-5.6, 4.5), 2.75)
        draw_enface(ax)
        ax = cv.drawing(55.0, top - 6.0, (-8.8, 8.8), (-4.0, 5.3), 2.75)
        draw_sideview(ax)
        cv.heading(0.5, top - 40.0, "b", "Beat-consistency window")
        ax = cv.axes(24.0, 12.0, 60.0, 46.0)
        draw_beats(ax, src)
        cv.heading(113.0, top, "c", "View rules and outputs")
        ax = cv.axes(113.0, 2.0, 69.5, top - 6.5 - 2.0)
        low = draw_flow(ax, 69.5, top - 6.5 - 2.0)
        stem = os.path.join(FIG, "fig1_design_alt")
    assert low >= 0, low
    fig = cv.fig
    if layout == "row":
        pd.DataFrame(src).to_csv(os.path.join(FIG, "fig1_design_source.csv"), index=False)
        qa(fig, stem, W_MM, min_pt=F1_BODY)
    else:
        fig.savefig(stem + ".png", dpi=600)
        probs = figqa.report(fig)
        print(f"[fig1_design_alt] figqa: {len(probs)} issue(s)")
        for p in probs:
            print("   ", p)
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
    """Graphical abstract, v05 (2026-09-19): 180 x 110 mm landscape. Calm 2 x 2 layout of the v02
    abstract at legible size, reached by removing content: body text 10 pt, tick labels 9 pt, panel
    titles 11 pt bold on one line, figure title 12 pt bold on one line. In-panel notes removed (the
    caption carries them); only the extreme values named in the brief are printed.
    Same four panels, data and source rows as v04. No plotted value changed.
    """
    nums, s, rmse = ga_numbers()
    N = nums.set_index("name")

    def v(name):
        return float(N.loc[name, "value"])

    W_MM, H_MM = 180, 110
    FS = 10.0     # axis labels, value labels, keys
    FT = 9.0      # tick labels
    TFS = 11.0    # panel titles, bold
    fig = plt.figure(figsize=figstyle.mm(W_MM, H_MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.10, hspace=0.12)
    fig.suptitle(GA_TITLE, fontsize=12, fontweight="bold", x=0.01, ha="left")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0])
    top = outer[0].subgridspec(1, 2, width_ratios=[1.2, 0.8])
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0])
    axs = [fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1]), fig.add_subplot(bot[0, 0]),
           fig.add_subplot(bot[0, 1])]
    src = []

    def fmt(x):
        return f"{x:+.2f}".replace("-", "−")

    def txt(ax, *a, **k):
        t = ax.text(*a, **k)
        t.set_in_layout(False)
        return t

    # ---- panel a: bias range across view accuracy
    ax = axs[0]
    ests = ["A1", "A2", "A3", "A4"]
    names = {"A1": "Anchor-view\nmean", "A2": "Mean across\nviews", "A3": "Largest\nview mean",
             "A4": "Largest view mean\nafter review"}
    ax.axhspan(-1, 1, color="#EBEBEB", zorder=0, lw=0)
    ax.axhline(0, color="black", lw=0.5, zorder=1)
    for i, e in enumerate(ests):
        g = s[s.estimator == e]
        lo, hi = g.value.min(), g.value.max()
        ax.plot([i, i], [lo, hi], color=es.COLOR[e], lw=2.2, solid_capstyle="butt", zorder=2)
        xs = i + np.linspace(-0.13, 0.13, len(g))
        gg = g.sort_values(["p_u_anchor", "p_u_long"])
        ax.plot(xs, gg.value, ls="none", marker=es.MARKER[e], ms=3.2, mfc=es.mfc(e),
                mec=es.COLOR[e], mew=0.7, zorder=3)
        if e == "A3":
            txt(ax, i + 0.2, hi, fmt(hi), ha="left", va="center", fontsize=FS)
            txt(ax, i + 0.2, lo, fmt(lo), ha="left", va="center", fontsize=FS)
        if e == "A2":
            txt(ax, i + 0.2, lo, fmt(lo), ha="left", va="center", fontsize=FS)
        for _, r in gg.iterrows():
            src.append(dict(panel="a", estimator=e, u_anchor=r.p_u_anchor, u_long=r.p_u_long,
                            bias_mm=r.value, mcse=r.mcse, source_cell=int(r.cell)))
    ax.set_xticks(range(4), [names[e] for e in ests], fontsize=FT)
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylim(-3.6, 1.5)
    ax.set_yticks([-3, -2, -1, 0, 1])
    ax.tick_params(axis="y", labelsize=FT)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Bias (mm)", fontsize=FS)
    ax.set_title(GA_TITLES["a"], fontsize=TFS, fontweight="bold", loc="left")

    # ---- panel b: beat window adds error
    ax = axs[1]
    for W, ls, mfc in ((0.0, "-", "black"), (0.15, (0, (3, 1.5)), "white")):
        g = rmse[rmse.W == W]
        ax.plot(g.p_N_beats, g.value, ls=ls, color="black", marker="o", ms=3.5, mfc=mfc,
                mec="black", mew=0.8, lw=0.9)
        for _, r in g.iterrows():
            src.append(dict(panel="b", W=W, N_beats=int(r.p_N_beats), rmse_mm=r.value,
                            mcse=r.mcse))
    g0 = rmse[rmse.W == 0.0].set_index("p_N_beats").value
    g1 = rmse[rmse.W == 0.15].set_index("p_N_beats").value
    xl_ = 8.3     # direct labels just above the dashed and below the solid line, between 7 and 10 beats
    y1 = np.interp(xl_, g1.index, g1.values)
    y0 = np.interp(xl_, g0.index, g0.values)
    txt(ax, xl_, y1 + 0.035, "±15% window", ha="left", va="bottom", fontsize=FS)
    txt(ax, xl_, y0 - 0.06, "no window", ha="left", va="top", fontsize=FS)
    ax.set_xticks([1, 3, 5, 7, 10, 13])
    ax.set_xlim(0.3, 13.7)
    ax.set_ylim(1.5, 2.6)
    ax.set_yticks([1.6, 2.0, 2.4])
    ax.tick_params(labelsize=FT)
    ax.set_xlabel("Beats averaged", fontsize=FS)
    ax.set_ylabel("RMSE (mm)", fontsize=FS)
    ax.set_title(GA_TITLES["b"], fontsize=TFS, fontweight="bold", loc="left")

    # ---- panel c: triggers detect few true errors
    ax = axs[2]
    h3 = v("E4 hit rate for true error > 2 mm at threshold 3 mm, AP (per pair)")
    h5 = v("E4 hit rate for true error > 2 mm at threshold 5 mm, AP (per pair)")
    n3 = v("E4 p_any_trigger at 3 mm, AP, no view errors")
    n5 = v("E4 p_any_adj at 5 mm, AP, no view errors")
    cats = ["Views with\nerror > 2 mm", "Patients without\nview error"]
    y = np.array([1.0, 0.0])
    bh = 0.34
    ax.barh(y + bh / 2, [100 * h3, 100 * n3], height=bh, color="#555555", ec="black", lw=0.5,
            label="3 mm warning")
    ax.barh(y - bh / 2, [100 * h5, 100 * n5], height=bh, color="white", ec="black", lw=0.5,
            hatch="//////", label="5 mm adjudication")
    for yy, val in ((1 + bh / 2, h3), (0 + bh / 2, n3), (1 - bh / 2, h5), (0 - bh / 2, n5)):
        txt(ax, 100 * val + 0.8, yy, f"{100 * val:.1f}%", ha="left", va="center", fontsize=FS)
    src += [dict(panel="c", quantity="hit rate, true error > 2 mm", threshold_mm=3, value=h3),
            dict(panel="c", quantity="hit rate, true error > 2 mm", threshold_mm=5, value=h5),
            dict(panel="c", quantity="P(any trigger), no view errors", threshold_mm=3, value=n3),
            dict(panel="c", quantity="P(any adjudication), no view errors", threshold_mm=5, value=n5)]
    ax.set_yticks(y, cats, fontsize=FT)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(-1.0, 1.45)
    ax.tick_params(axis="x", labelsize=FT)
    ax.set_xlabel("Triggered (%)", fontsize=FS)
    leg = ax.legend(loc="lower right", ncol=1, fontsize=FS, frameon=False, handlelength=1.4,
                    borderaxespad=0.0, labelspacing=0.3)
    leg.set_in_layout(False)
    ax.set_title(GA_TITLES["c"], fontsize=TFS, fontweight="bold", loc="left")

    # ---- panel d: reference hides/inflates AI error
    ax = axs[3]
    ti = v("E5b independent AI true_mae, sigma_AI 2 mm, lambda 1, reference mean of 2 reads")
    ai_ = v("E5b independent AI apparent_mae, sigma_AI 2 mm, lambda 1, reference mean of 2 reads")
    ts = v("E5b shared AI true_mae, sigma_AI 2 mm, lambda 1, reference mean of 2 reads")
    as_ = v("E5b shared AI apparent_mae, sigma_AI 2 mm, lambda 1, reference mean of 2 reads")
    rowsd = [(1.0, "Independent AI", ti, ai_), (0.0, "AI sharing\nimage error", ts, as_)]
    for yy, lab, t, a in rowsd:
        sgn = 1 if a > t else -1
        arrow(ax, (t + sgn * 0.03, yy), (a - sgn * 0.035, yy), style="-|>", lw=0.9, ms=7)
        ax.plot([t], [yy], marker="o", ms=5, color="black", ls="none", zorder=3)
        ax.plot([a], [yy], marker="s", ms=5, mfc="white", mec="black", mew=0.9, ls="none",
                zorder=3)
        txt(ax, t, yy + 0.16, f"{t:.2f}", ha="center", va="bottom", fontsize=FS)
        txt(ax, a, yy + 0.16, f"{a:.2f}", ha="center", va="bottom", fontsize=FS)
        src.append(dict(panel="d", ai=lab.replace("\n", " "), true_mae=t, apparent_mae=a))
    hd = [Line2D([], [], marker="o", ms=5, color="black", ls="none", label="true"),
          Line2D([], [], marker="s", ms=5, mfc="white", mec="black", ls="none", label="apparent")]
    leg = ax.legend(handles=hd, loc="lower right", ncol=2, fontsize=FS, frameon=False, handletextpad=0.2,
                    borderaxespad=0.0, columnspacing=1.0)
    leg.set_in_layout(False)
    ax.set_yticks([1, 0], [r[1] for r in rowsd], fontsize=FT)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.75, 1.55)
    ax.set_xlim(1.4, 2.3)
    ax.set_xticks([1.4, 1.6, 1.8, 2.0, 2.2])
    ax.tick_params(axis="x", labelsize=FT)
    ax.set_xlabel("AI mean absolute error (mm)", fontsize=FS)
    ax.set_title(GA_TITLES["d"], fontsize=TFS, fontweight="bold", loc="left")

    # titles start at the left edge of each column (the y-axis decorations), not of the axes box
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in axs:
        left = ax.yaxis.get_tightbbox(r).x0
        ax._left_title.set_x((left - ax.bbox.x0) / ax.bbox.width)   # the loc="left" heading

    pd.DataFrame(src).to_csv(os.path.join(FIG, "graphical_abstract_source.csv"), index=False)
    stem = os.path.join(FIG, "graphical_abstract")
    qa(fig, stem, W_MM)
    plt.close(fig)


# one line at 12 pt bold: "...: simulation results" measured 191 mm, so it is shortened (176 mm)
GA_TITLE = "Combining beats and views in tricuspid jet measurement: a simulation study"
GA_TITLES = {"a": "a  Largest view mean stays within ±1 mm",
             "b": "b  A ±15% beat window adds error",
             "c": "c  Millimetre triggers miss most view errors",
             "d": "d  References distort apparent AI error"}


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("fig1", "all"):
        fig1("row")
    if what in ("fig1alt", "all"):
        fig1("stack")
    if what in ("ga", "all"):
        ga()
