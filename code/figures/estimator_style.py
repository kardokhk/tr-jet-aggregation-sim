"""Fixed colour, marker, fill and line style per estimator, shared by every DUO-Max figure.

Colours are fixed Okabe-Ito hues. In greyscale they fall into two clusters of
similar grey level (A1, A6, A3 dark; A4, A5, A2 light), so within each cluster
every estimator has its own dash pattern and marker (A4 also an open marker)
to stay separable when printed in greyscale:
  A1 solid, filled circle        A6 dotted, filled plus
  A2 solid, filled square        A5 dash-dot, filled down-triangle
  A3 dash-dot-dot, filled up-triangle   A4 short dash, open diamond
  A7 black long dash, x (offset-corrected mean)
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
LABEL = {"A1": "A1 anchor mean", "A2": "A2 mean of views", "A3": "A3 max of views",
         "A4": "A4 composite rule", "A5": "A5 median of views", "A6": "A6 index beat",
         "A7": "A7 offset-corrected mean"}


def mfc(e):
    return "white" if e in OPEN else COLOR[e]


def kw(e, ms=3.0, lw=0.9, **extra):
    d = dict(color=COLOR[e], marker=MARKER[e], ls=LS[e], lw=lw, ms=ms, mew=0.8,
             mec=COLOR[e], mfc=mfc(e))
    d.update(extra)
    return d
