"""Fixed colour, marker, fill and line style per estimator, shared by every DUO-Max figure.

Colours are fixed Okabe-Ito hues. In greyscale they fall into two clusters of
similar grey level (A1, A6, A3 dark; A4, A5, A2 light), so within each cluster
every estimator has its own dash pattern and marker (A4 also an open marker)
to stay separable when printed in greyscale:
  A1 solid, filled circle        A6 dotted, filled plus
  A2 solid, filled square        A5 dash-dot, filled down-triangle
  A3 dash-dot-dot, filled up-triangle   A4 short dash, open diamond
  A7 black long dash, x (offset-corrected mean, benchmark)
"""
EST = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
COLOR = {"A1": "#0072B2", "A2": "#E69F00", "A3": "#D55E00", "A4": "#CC79A7",
         "A5": "#56B4E9", "A6": "#009E73", "A7": "#000000"}
MARKER = {"A1": "o", "A2": "s", "A3": "^", "A4": "D", "A5": "v", "A6": "P", "A7": "x"}
LS = {"A1": "-", "A2": "-",
      "A3": (0, (5.0, 1.2, 1.0, 1.2, 1.0, 1.2)),  # dash-dot-dot (separates A3 from A1 in greyscale)
      "A4": (0, (3.0, 1.5)),          # short dash
      "A5": (0, (4.0, 1.5, 1.0, 1.5)),  # dash-dot
      "A6": (0, (1.0, 1.2)),          # dotted
      "A7": (0, (6.0, 2.0))}          # long dash
OPEN = {"A4"}
# Plain names shown in every figure (v04, 2026-09-19); codes A1 to A7 stay internal and are mapped
# in Supplementary Table S6.
LABEL = {"A1": "Anchor-view mean", "A2": "Mean across views", "A3": "Largest view mean",
         "A4": "Largest view mean after review", "A5": "Median across views", "A6": "Index beat",
         "A7": "Offset-corrected mean (benchmark)"}


def mfc(e):
    return "white" if e in OPEN else COLOR[e]


def kw(e, ms=3.0, lw=0.9, **extra):
    d = dict(color=COLOR[e], marker=MARKER[e], ls=LS[e], lw=lw, ms=ms, mew=0.8,
             mec=COLOR[e], mfc=mfc(e))
    d.update(extra)
    return d


# ---- Text sizes (v05, 2026-09-19). EHJ-CVI requires figure text no smaller than 2 mm tall at
# print size; 7 pt Helvetica/Nimbus Sans capitals are about 1.8 mm, so 8 pt is the floor for every
# text element (body, tick labels, keys, annotations). Panel titles 8.5 pt, panel letters 9 pt bold.
FS_MIN = 8.0
FS_TITLE = 8.5
FS_LETTER = 9.0


def use_journal_text():
    """Raise the figstyle rcParams (7 pt body, 6 pt ticks) to the EHJ-CVI floor. Call after
    figstyle.use_print_style()."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": FS_MIN, "axes.labelsize": FS_MIN, "axes.titlesize": FS_TITLE,
                         "xtick.labelsize": FS_MIN, "ytick.labelsize": FS_MIN,
                         "legend.fontsize": FS_MIN, "legend.title_fontsize": FS_MIN,
                         "figure.titlesize": FS_TITLE})
    # mathtext shrinks sub- and superscripts to 70% (8 pt -> 5.6 pt, below the 2 mm floor). Subscripted
    # symbols (t_warn, f_rej, s_det) therefore render their subscripts at full size. SHRINK_FACTOR is a
    # private module constant read at parse time (checked in matplotlib 3.11.2).
    import matplotlib._mathtext as _mt
    _mt.SHRINK_FACTOR = 1.0
