#!/usr/bin/env python
"""Analysis of E1 (estimator bias and RMSE) and E3 (threshold misclassification).

Input : results/2026-09-18_full/E1_*.parquet, E3_*.parquet (read only).
Output: results/2026-09-18_full/analysis/E1E3_*.csv

Tables
  E1E3_e1_long.csv          tidy E1 rows, all cells, A4 base variant only for A4
  E1E3_e1_base_slice.csv    rho 0.3, N_beats 3; all u, view_over, K, beat_cv; wide by metric
  E1E3_e1_headline.csv      base case cell (K 3, beat CV 0.15, over on, rho 0.3, N 3), both u
  E1E3_e1_a3_crossing.csv   beat CV at which A3 bias vs T1 = 0 (linear interpolation, delta-method SE)
  E1E3_e1_a4_variants.csv   A4 s_det x f_rej at t_warn 3 / t_adj 5 (base cell and K x beat CV)
  E1E3_e1_rho_N.csv         rho and N_beats effects at K 3, beat CV 0.15
  E1E3_e1_max_mcse.csv      maximum MCSE per metric over the plotted slices
  E1E3_e3_long.csv          tidy E3 rows (with FPR = 1 - specificity, FNR = 1 - sensitivity)
  E1E3_e3_base.csv          base case (K 3, beat CV 0.15, over on), both u, wide by metric
  E1E3_e3_max_mcse.csv
Base case: K 3, beat CV 0.15, view_over True, rho 0.3, N_beats 3, window 0.15, sinus,
prospective, sigma_cal 1.0 mm, sigma_rb 0.75 mm, g_log_sd 0.10; A4 base variant s_det 0.8,
f_rej 0.1, t_warn 3 mm, t_adj 5 mm. 95% Monte Carlo interval = estimate +- 1.96 MCSE.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "2026-09-18_full"
OUT = RES / "analysis"
Z = 1.959964
KEEP = ["cell", "u", "p_K", "p_beat_cv", "p_view_over", "p_rho", "p_N_beats", "estimator",
        "axis", "estimand", "metric", "value", "mcse", "n", "s_det", "f_rej", "t_warn", "t_adj"]


def load(exp: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RES / f"{exp}_cells_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    s = df["p_u_scale"].astype(str)
    df["u"] = np.where(s.str.contains("0.25"), "base", np.where(s.str.contains("0.1"), "low", "?"))
    assert (df["u"] != "?").all()
    return df


def ren(df):
    return df.rename(columns={c: c[2:] for c in df.columns if c.startswith("p_")})


def crossing(g: pd.DataFrame, col="bias", se="bias_mcse"):
    """First sign change of bias along beat_cv; linear interpolation; delta-method SE."""
    g = g.sort_values("beat_cv")
    x, b, s = g["beat_cv"].to_numpy(), g[col].to_numpy(), g[se].to_numpy()
    for i in range(len(x) - 1):
        if b[i] <= 0 < b[i + 1] or b[i] >= 0 > b[i + 1]:
            dx, db = x[i + 1] - x[i], b[i + 1] - b[i]
            xs = x[i] - b[i] * dx / db
            sx = dx / db ** 2 * np.sqrt(b[i + 1] ** 2 * s[i] ** 2 + b[i] ** 2 * s[i + 1] ** 2)
            return pd.Series(dict(status="crossed", cv_cross=xs, cv_cross_mcse=sx,
                                  cv_lo=x[i], cv_hi=x[i + 1]))
    st = "positive_at_all_cv" if (b > 0).all() else ("negative_at_all_cv" if (b < 0).all() else "other")
    return pd.Series(dict(status=st, cv_cross=np.nan, cv_cross_mcse=np.nan, cv_lo=np.nan, cv_hi=np.nan))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------ E1
    e1 = load("E1")
    a4base = (e1.estimator != "A4") | ((e1.s_det == 0.8) & (e1.f_rej == 0.1))
    e1s = ren(e1[KEEP].copy())
    e1s = e1s[e1s.estimator != "rule"]
    std = e1s[a4base.loc[e1s.index]].copy()
    std.to_csv(OUT / "E1E3_e1_long.csv", index=False)

    def wide(d, idx):
        w = d.pivot_table(index=idx, columns="metric", values=["value", "mcse"], aggfunc="first")
        w.columns = [f"{m}" if v == "value" else f"{m}_mcse" for v, m in w.columns]
        w = w.rename(columns=lambda c: c.replace("_mm", "").replace("relbias_pct", "relbias_pct"))
        return w.reset_index()

    idx = ["u", "view_over", "K", "beat_cv", "rho", "N_beats", "estimator", "axis", "estimand"]
    base_slice = wide(std[(std.rho == 0.3) & (std.N_beats == 3)], idx)
    base_slice.to_csv(OUT / "E1E3_e1_base_slice.csv", index=False)

    head = base_slice[(base_slice.K == 3) & (base_slice.beat_cv == 0.15) & (base_slice.view_over)].copy()
    for m in ("bias", "rmse"):
        head[f"{m}_lo95"] = head[m] - Z * head[f"{m}_mcse"]
        head[f"{m}_hi95"] = head[m] + Z * head[f"{m}_mcse"]
    head.to_csv(OUT / "E1E3_e1_headline.csv", index=False)

    # A3 crossing: raw, and offset-adjusted (A3 minus A7 bias; exploratory)
    t1 = base_slice[base_slice.estimand == "T1"]
    a3 = t1[t1.estimator == "A3"].set_index(["u", "view_over", "K", "axis", "beat_cv"])
    a7 = t1[t1.estimator == "A7"].set_index(["u", "view_over", "K", "axis", "beat_cv"])
    adj = a3[["bias"]].copy()
    adj["bias_adj"] = a3["bias"] - a7["bias"]
    adj["bias_adj_mcse"] = np.sqrt(a3["bias_mcse"] ** 2 + a7["bias_mcse"] ** 2)  # conservative
    adj["bias_mcse"] = a3["bias_mcse"]
    adj = adj.reset_index()
    rows = []
    for key, g in adj.groupby(["u", "view_over", "K", "axis"]):
        r1 = crossing(g, "bias", "bias_mcse")
        r2 = crossing(g, "bias_adj", "bias_adj_mcse").add_prefix("adj_")
        rows.append(pd.concat([pd.Series(dict(zip(["u", "view_over", "K", "axis"], key))), r1, r2]))
    cross = pd.DataFrame(rows)
    cross["cv_cross_lo95"] = cross.cv_cross - Z * cross.cv_cross_mcse
    cross["cv_cross_hi95"] = cross.cv_cross + Z * cross.cv_cross_mcse
    cross.to_csv(OUT / "E1E3_e1_a3_crossing.csv", index=False)
    adj.to_csv(OUT / "E1E3_e1_a3_offset_adjusted.csv", index=False)

    # A4 variants (all 12 s_det x f_rej), rho 0.3, N 3
    a4 = e1s[(e1s.estimator == "A4") & (e1s.rho == 0.3) & (e1s.N_beats == 3)]
    a4w = wide(a4, ["u", "view_over", "K", "beat_cv", "s_det", "f_rej", "axis", "estimand"])
    a4w.to_csv(OUT / "E1E3_e1_a4_variants.csv", index=False)

    # rho and N_beats at K 3, beat CV 0.15 (both view_over levels)
    rn = std[(std.K == 3) & (std.beat_cv == 0.15)]
    rnw = wide(rn, ["u", "view_over", "rho", "N_beats", "estimator", "axis", "estimand"])
    rnw.to_csv(OUT / "E1E3_e1_rho_N.csv", index=False)

    mx = e1s.groupby(["metric", "axis"]).mcse.max().rename("max_mcse_all_cells").reset_index()
    sl = std[(std.rho == 0.3) & (std.N_beats == 3)].groupby(["metric", "axis"]).mcse.max()
    mx = mx.merge(sl.rename("max_mcse_base_slice").reset_index(), on=["metric", "axis"])
    mx.to_csv(OUT / "E1E3_e1_max_mcse.csv", index=False)

    # ------------------------------------------------------------------ E3
    e3 = ren(load("E3")[["cell", "u", "p_K", "p_beat_cv", "p_view_over", "estimator", "axis", "estimand",
                         "cutoff_mm", "metric", "value", "mcse", "n"]])
    extra = []
    for m, new in (("sensitivity", "fnr"), ("specificity", "fpr")):
        t = e3[e3.metric == m].copy()
        t["value"] = 1 - t["value"]
        t["metric"] = new
        extra.append(t)
    e3 = pd.concat([e3] + extra, ignore_index=True)
    e3.to_csv(OUT / "E1E3_e3_long.csv", index=False)
    b3 = e3[(e3.K == 3) & (e3.beat_cv == 0.15) & (e3.view_over)]
    w3 = b3.pivot_table(index=["u", "axis", "cutoff_mm", "estimator"], columns="metric",
                        values=["value", "mcse"], aggfunc="first")
    w3.columns = [m if v == "value" else f"{m}_mcse" for v, m in w3.columns]
    w3 = w3.reset_index()
    w3.to_csv(OUT / "E1E3_e3_base.csv", index=False)
    e3.groupby(["metric", "axis"]).mcse.max().rename("max_mcse").reset_index().to_csv(
        OUT / "E1E3_e3_max_mcse.csv", index=False)

    # console summary
    pd.set_option("display.width", 220)
    print(head[(head.axis == "AP") & (head.estimand == "T1")][
        ["u", "estimator", "bias", "bias_mcse", "rmse", "rmse_mcse", "relbias_pct"]].round(3).to_string())
    print(cross[cross.axis == "AP"].round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
