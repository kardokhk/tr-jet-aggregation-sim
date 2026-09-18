"""EXPLORATORY post hoc sensitivity analyses (config code/configs/sens_2026-09-18.yaml, job 5223386).

Writes paired comparisons of each variant against its reference level (same seed stream) to
results/2026-09-18_sens/analysis/. Base-case slice: K 3, beat CV 15%, overestimation on, AP, T1.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "2026-09-18_sens"
OUT = RES / "analysis"
OUT.mkdir(exist_ok=True)


def load(e):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(RES / f"{e}_*.parquet")))], ignore_index=True)


def ci(v, se):
    return v - 1.96 * se, v + 1.96 * se


# E1: estimator bias under mean-centred beat noise and alternative A4 scrutiny
e1 = load("E1")
e1["u_level"] = np.where(e1.p_u_scale.astype(str).str.contains("0.25"), "base", "low")
e1["variant"] = e1.p_beat_noise_centering.astype(str) + "/" + e1.p_a4_scrutiny.astype(str)
m = (e1.axis == "AP") & (e1.estimand == "T1") & (e1.metric.isin(["bias_mm", "rmse_mm"])) & (
    e1.s_det.isna() | ((e1.s_det == 0.8) & (e1.f_rej == 0.1)))
t1 = e1[m][["u_level", "variant", "p_K", "p_beat_cv", "p_view_over", "estimator", "metric", "value", "mcse", "n"]]
t1.to_csv(OUT / "sens_E1_bias_rmse.csv", index=False)
base = t1[(t1.p_K == 3) & (t1.p_beat_cv == 0.15) & (t1.p_view_over == True)]
base = base.assign(lo=lambda d: d.value - 1.96 * d.mcse, hi=lambda d: d.value + 1.96 * d.mcse)
base.to_csv(OUT / "sens_E1_base_slice.csv", index=False)

# E2: window effect on anchor-mean RMSE vs T1 under both centrings (sinus, prospective)
e2 = load("E2")
m2 = (e2.axis == "AP") & (e2.estimator == "A1") & (e2.metric == "rmse_mm") & (e2.estimand == "T1") & (e2.subset == "all")
t2 = e2[m2][["p_beat_noise_centering", "p_beat_cv", "p_window", "p_N_beats", "value", "mcse"]].copy()
t2["p_window"] = pd.to_numeric(t2.p_window)  # NaN = no window
t2.to_csv(OUT / "sens_E2_rmse.csv", index=False)
rows = []
for (c, cv, n), g in t2.groupby(["p_beat_noise_centering", "p_beat_cv", "p_N_beats"]):
    nw = g[g.p_window.isna()]
    w15 = g[np.isclose(g.p_window, 0.15)]
    if len(nw) == 1 and len(w15) == 1:
        d = float(w15.value.iloc[0] - nw.value.iloc[0])
        se = float(np.hypot(w15.mcse.iloc[0], nw.mcse.iloc[0]))  # conservative: ignores positive CRN covariance
        rows.append(dict(centering=c, beat_cv=cv, N=n, rmse_nowindow=float(nw.value.iloc[0]),
                         rmse_w15=float(w15.value.iloc[0]), diff=d, diff_lo=d - 1.96 * se, diff_hi=d + 1.96 * se))
pd.DataFrame(rows).to_csv(OUT / "sens_E2_window_effect.csv", index=False)

# E5: adjudication value (third read vs closest-pair mean): reference MAE vs T1
e5 = load("E5")
t5 = e5[e5.metric.isin(["ref_mae_vs_T1", "ref_ba_bias_vs_T1"])][
    ["p_adjudication_value", "p_beat_cv", "p_view_over", "ref", "metric", "value", "mcse"]]
t5.to_csv(OUT / "sens_E5_reference.csv", index=False)
print(base[base.metric == "bias_mm"].pivot_table(index="estimator", columns=["u_level", "variant"], values="value").round(3).to_string())
we = pd.DataFrame(rows)
print(we[(we.beat_cv == 0.15) & (we.N == 3)].round(3).to_string())
print(t5[(t5.p_beat_cv == 0.15) & (t5.p_view_over == True) & (t5.metric == "ref_mae_vs_T1")].round(3).to_string())
