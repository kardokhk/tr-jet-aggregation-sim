"""E2 analysis: beat acceptance windows, number of beats, AF, index beat,
retrospective versus prospective availability.

Reads results/2026-09-18_full/E2_*.parquet (read-only) and writes tables to
results/2026-09-18_full/analysis/E2_*.csv.

Conventions
- W = 0 in the tables means "no acceptance window" (p_window missing in parquet).
- rhythm_state: 'sinus', 'AF_rr0.1', 'AF_rr0.2', 'AF_rr0.3' (RR CV in AF).
- Differences between cells (window versus no window) come from independently
  seeded cells, so MCSE(diff) = sqrt(mcse1^2 + mcse2^2); 95% Monte Carlo
  interval = diff +- 1.96 MCSE(diff).
- Tv = g * mu_anchor (the anchor-view expected span under the study's
  instrument setting): the estimand a beat rule can at best recover.

Run: /project/home/p201509/envs/duomax-sim/bin/python code/03_analyse_e2.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "2026-09-18_full"
OUT = RES / "analysis"
KEYS = ["rhythm_state", "p_beat_cv", "W", "p_N_beats", "p_data_mode"]


def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(RES / "E2_cells_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert df["cell"].nunique() == 1440, df["cell"].nunique()
    df["rhythm_state"] = np.where(df["p_rhythm"] == "sinus", "sinus",
                                  "AF_rr" + df["p_rr_cv_af"].astype(str))
    df["W"] = df["p_window"].fillna(0.0).round(2)
    df["p_beat_cv"] = df["p_beat_cv"].round(2)
    return df


def rule_table(df, metric, subset=None):
    s = df[(df["metric"] == metric)]
    if subset is not None:
        s = s[s["subset"] == subset]
    cols = KEYS + ["axis", "value", "mcse", "n"]
    return s[cols].sort_values(KEYS + ["axis"]).reset_index(drop=True)


def error_table(df):
    s = df[df["estimator"].isin(["A1", "A6"])]
    s = s[s["metric"].isin(["bias_mm", "rmse_mm", "sd_err_mm", "relbias_pct"])]
    cols = KEYS + ["axis", "estimator", "subset", "estimand", "metric", "value", "mcse", "n"]
    return s[cols].sort_values(cols[:-3]).reset_index(drop=True)


def window_effect(err):
    """Windowed minus no-window value at the same N, CV, rhythm, mode, axis, estimator, estimand."""
    e = err[err["subset"] == "all"]
    k = ["rhythm_state", "p_beat_cv", "p_N_beats", "p_data_mode", "axis", "estimator", "estimand", "metric"]
    base = e[e["W"] == 0][k + ["value", "mcse"]].rename(columns={"value": "v_none", "mcse": "se_none"})
    w = e[e["W"] > 0][k + ["W", "value", "mcse"]].rename(columns={"value": "v_win", "mcse": "se_win"})
    m = w.merge(base, on=k, how="left")
    m["diff"] = m["v_win"] - m["v_none"]
    m["mcse_diff"] = np.sqrt(m["se_win"] ** 2 + m["se_none"] ** 2)
    m["lo95"] = m["diff"] - 1.96 * m["mcse_diff"]
    m["hi95"] = m["diff"] + 1.96 * m["mcse_diff"]
    m["ratio"] = m["v_win"] / m["v_none"]
    return m.sort_values(k + ["W"]).reset_index(drop=True)


def equivalent_n(err):
    """For each windowed rule at N beats (prospective, AP/SL, A1, all, Tv/T1 RMSE),
    the smallest no-window N on the grid whose RMSE is <= the windowed RMSE."""
    e = err[(err["subset"] == "all") & (err["estimator"] == "A1") & (err["metric"] == "rmse_mm")
            & (err["p_data_mode"] == "prospective")]
    rows = []
    for key, g in e.groupby(["rhythm_state", "p_beat_cv", "axis", "estimand"]):
        nowin = g[g["W"] == 0].set_index("p_N_beats")["value"].sort_index()
        for _, r in g[g["W"] > 0].iterrows():
            ok = nowin[nowin <= r["value"]]
            rows.append(dict(zip(["rhythm_state", "p_beat_cv", "axis", "estimand"], key),
                             W=r["W"], p_N_beats=r["p_N_beats"], rmse_win=r["value"],
                             smallest_nowin_N_as_good=int(ok.index.min()) if len(ok) else np.nan))
    return pd.DataFrame(rows).sort_values(["rhythm_state", "p_beat_cv", "axis", "estimand", "W", "p_N_beats"])


def headline(df, err, eff, met, beats, lim):
    """Named numbers for the Results section (AP axis unless stated)."""
    H = []

    def add(name, table, sel: dict, col="value", se_col="mcse"):
        t = table
        for c, v in sel.items():
            t = t[t[c] == v]
        assert len(t) == 1, (name, len(t))
        r = t.iloc[0]
        H.append(dict(name=name, table=table.attrs.get("file", ""),
                      selector="; ".join(f"{c}={v}" for c, v in sel.items()),
                      value=r[col], mcse=r[se_col],
                      lo95=r[col] - 1.96 * r[se_col], hi95=r[col] + 1.96 * r[se_col]))

    base = dict(p_beat_cv=0.15, p_data_mode="prospective", axis="AP")
    for rs in ["sinus", "AF_rr0.2"]:
        for N in [3, 5]:
            for W in [0.10, 0.15, 0.20, 0.25]:
                add(f"p_met_first_N {rs} N{N} W{W}", met, dict(base, rhythm_state=rs, p_N_beats=N, W=W))
        for N in [3, 5]:
            add(f"p_met_first_N SL {rs} N{N} W0.15", met,
                dict(base, axis="SL", rhythm_state=rs, p_N_beats=N, W=0.15))
            for W in [0.0, 0.15]:
                add(f"beats_acquired {rs} N{N} W{W}", beats, dict(base, rhythm_state=rs, p_N_beats=N, W=W))
        for N in [3, 5]:
            for est in ["T1", "Tv"]:
                for W in [0.0, 0.15]:
                    for m in ["rmse_mm", "bias_mm", "relbias_pct"]:
                        add(f"A1 {m} vs {est} {rs} N{N} W{W}", err,
                            dict(base, rhythm_state=rs, p_N_beats=N, W=W, estimator="A1",
                                 subset="all", estimand=est, metric=m))
                    add(f"A6 rmse_mm vs {est} {rs} N{N} W{W}", err,
                        dict(base, rhythm_state=rs, p_N_beats=N, W=W, estimator="A6",
                             subset="all", estimand=est, metric="rmse_mm"))
                for m in ["rmse_mm", "bias_mm", "relbias_pct"]:
                    add(f"window effect (W0.15 - none) A1 {m} vs {est} {rs} N{N}", eff,
                        dict(base, rhythm_state=rs, p_N_beats=N, W=0.15, estimator="A1",
                             estimand=est, metric=m), col="diff", se_col="mcse_diff")
    for rs in ["sinus", "AF_rr0.2"]:
        for N in [3, 5]:
            for W in [0.0, 0.15]:
                add(f"retro p_limited {rs} N{N} W{W}", lim,
                    dict(p_beat_cv=0.15, p_data_mode="retrospective", axis="AP",
                         rhythm_state=rs, p_N_beats=N, W=W))
    # SL window effect on Tv relative bias at base
    add("window effect (W0.15 - none) A1 relbias_pct vs Tv SL sinus N3", eff,
        dict(base, axis="SL", rhythm_state="sinus", p_N_beats=3, W=0.15, estimator="A1",
             estimand="Tv", metric="relbias_pct"), col="diff", se_col="mcse_diff")
    # retrospective, accepted vs limited subsets at base
    for sub in ["accepted", "limited"]:
        add(f"retro A1 relbias_pct vs Tv sinus N3 W0.15 subset={sub}", err,
            dict(p_beat_cv=0.15, p_data_mode="retrospective", axis="AP", rhythm_state="sinus",
                 p_N_beats=3, W=0.15, estimator="A1", subset=sub, estimand="Tv", metric="relbias_pct"))
    return pd.DataFrame(H)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    met = rule_table(df, "p_window_met_first_N")
    beats = rule_table(df, "beats_acquired_mean", subset="all")
    lim = rule_table(df, "p_limited")
    err = error_table(df)
    eff = window_effect(err)
    eqn = equivalent_n(err)
    outs = {"E2_window_met.csv": met, "E2_beats_acquired.csv": beats, "E2_limited.csv": lim,
            "E2_error.csv": err, "E2_window_effect.csv": eff, "E2_equivalent_n.csv": eqn}
    for f, t in outs.items():
        t.attrs["file"] = f"results/2026-09-18_full/analysis/{f}"
    H = headline(df, err, eff, met, beats, lim)
    outs["E2_headline.csv"] = H
    for f, t in outs.items():
        t.to_csv(OUT / f, index=False, float_format="%.6g")
        print(f"wrote {OUT / f} ({len(t)} rows)")
    # max MCSE for captions (all-patients subsets only)
    mx = {
        "p_met_first_N": met["mcse"].max(),
        "p_limited": lim["mcse"].max(),
        "beats_acquired": beats["mcse"].max(),
        "rmse_all_A1A6": err[(err.subset == "all") & (err.metric == "rmse_mm")]["mcse"].max(),
        "rmse_all_A1A6_T1_prosp": err[(err.subset == "all") & (err.metric == "rmse_mm") & (err.estimand == "T1")
                                      & (err.p_data_mode == "prospective")]["mcse"].max(),
    }
    pd.Series(mx, name="max_mcse").to_csv(OUT / "E2_max_mcse.csv")
    print(mx)


if __name__ == "__main__":
    main()
