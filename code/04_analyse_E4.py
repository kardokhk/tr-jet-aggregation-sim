#!/usr/bin/env python
"""E4 analysis: cross-view discrepancy triggers (protocol section 5, E4).

Reads results/2026-09-18_full/E4_*.parquet and writes
results/2026-09-18_full/analysis/E4_*.csv.

What the E4 metrics mean (code/lib/duomaxsim/experiments.py, run_E4; rules.cross_view_flags):
- K = 4 views, view 0 = anchor. Per axis d and verification view v in 1..3, the
  signed difference diff = mean_v - mean_anchor (after the beat rule).
  "trigger" = diff >= t_warn (warning or adjudication); "adj" = diff >= t_adj.
  The trigger is one-sided: a verification view LOWER than the anchor never fires.
- A true error is present for pair (anchor, v) on axis d when
  g*S_d*o_v > 2 mm (verification view overestimates) or g*S_d*u_anchor > 2 mm
  (anchor underestimates). The verification view's own underestimation u_v is not
  part of the event, so an overestimating view can be pulled back by u_v.
- *_pair metrics are pooled over the 3 verification views of each patient
  (n = 3 x 100,000 pairs per axis); the stored MCSE treats pairs as independent.
  A conservative MCSE multiplies it by sqrt(3) (full within-patient dependence).
- p_any_* are per patient and axis: any of the 3 pairs fires.
- p_anchor_axis_diff_*: |anchor AP mean - anchor SL mean| >= threshold (measured);
  p_true_axis_diff_ge_twarn: |S_AP - S_SL| >= t_warn (truth). These are the
  cross-axis (ellipticity) quantities and are not cross-view triggers.

Because trigger = diff >= t_warn and adj = diff >= t_adj, the per-pair metrics
depend on one threshold only; the analysis re-indexes them by a single
threshold tau in {2, 3, 4, 5, 6} mm and checks that tau = 4 agrees between the
t_warn = 4 and t_adj = 4 rows (same data, so they must be identical).

A supplementary exploratory computation of P(|S_AP - S_SL| >= tau) for tau = 5, 6
(not stored by E4) uses duomaxsim.model.draw_latent with a separate seed.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from duomaxsim import config as C  # noqa: E402
from duomaxsim.model import draw_latent  # noqa: E402

RES = ROOT / "results" / "2026-09-18_full"
OUT = RES / "analysis"
Z = 1.959964
BASE = dict(u="base", view_errors=True, r_mean=0.64, beat_cv=0.15, N_beats=3)
SUPP_SEED_KEY = 99          # spawn_key (99, i) off the master seed; exploratory only
SUPP_N, SUPP_CHUNK = 1_000_000, 100_000


def load() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(RES / "E4_*.parquet")))],
                   ignore_index=True)
    df["u"] = np.where(df["p_u_scale"].astype(str).str.contains("0.25"), "base", "low")
    df["view_errors"] = df["p_view_errors"].astype(str) == "True"
    for c in ("r_mean", "beat_cv"):
        df[c] = df["p_" + c].astype(float)
    df["N_beats"] = df["p_N_beats"].astype(int)
    assert df["cell"].nunique() == 576, df["cell"].nunique()
    return df


KEYS = ["u", "view_errors", "r_mean", "beat_cv", "N_beats", "cell"]


def per_threshold(df: pd.DataFrame, m_warn: str, m_adj: str, axes) -> pd.DataFrame:
    """Collapse a metric stored per (t_warn, t_adj) into one row per threshold tau."""
    a = df[(df.metric == m_warn) & df.axis.isin(axes)].copy()
    a["tau"] = a["t_warn"]
    b = df[(df.metric == m_adj) & df.axis.isin(axes)].copy()
    b["tau"] = b["t_adj"]
    ab = pd.concat([a, b])
    # invariance: identical across the partner threshold, and tau = 4 identical in both sources
    g = ab.groupby(KEYS + ["axis", "tau"])["value"]
    spread = (g.max() - g.min()).max()
    assert spread < 1e-12, f"{m_warn}/{m_adj}: value differs across partner threshold ({spread})"
    return ab.groupby(KEYS + ["axis", "tau"], as_index=False).first()[
        KEYS + ["axis", "tau", "value", "mcse", "n"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()

    # ---- 1. per-patient probability of any trigger / any adjudication ------------------
    rows = []
    for m, thr in (("p_any_trigger", "t_warn"), ("p_any_adj", "t_adj")):
        s = df[df.metric == m].copy()
        s["tau"] = s[thr]
        g = s.groupby(KEYS + ["axis", "tau"])["value"]
        assert (g.max() - g.min()).max() < 1e-12
        s = s.groupby(KEYS + ["axis", "tau"], as_index=False).first()
        s["rule"] = "any trigger (diff >= t_warn)" if m == "p_any_trigger" else "any adjudication (diff >= t_adj)"
        s["metric"] = m
        rows.append(s[KEYS + ["axis", "metric", "rule", "tau", "value", "mcse", "n"]])
    anyt = pd.concat(rows, ignore_index=True)
    anyt["lo95"] = anyt["value"] - Z * anyt["mcse"]
    anyt["hi95"] = anyt["value"] + Z * anyt["mcse"]
    anyt.to_csv(OUT / "E4_any_trigger.csv", index=False)

    # warning-only per pair (t_warn, t_adj), for completeness
    w = df[df.metric == "p_any_warn"][KEYS + ["axis", "t_warn", "t_adj", "value", "mcse", "n"]]
    w.to_csv(OUT / "E4_any_warn_by_pair.csv", index=False)

    # ---- 2. pair-level operating characteristics by threshold tau ------------------------
    fa = per_threshold(df, "false_alarm_trigger_pair", "false_alarm_adj_pair", ["AP", "SL"])
    fa = fa.rename(columns={"value": "false_alarm", "mcse": "false_alarm_mcse", "n": "n_nonevent_pairs"})
    ms = per_threshold(df, "miss_trigger_pair", "miss_adj_pair", ["AP", "SL"])
    ms = ms.rename(columns={"value": "miss", "mcse": "hit_mcse", "n": "n_event_pairs"})
    pe = df[df.metric == "p_error_event_pair"].groupby(KEYS + ["axis"], as_index=False).first()
    pe = pe[KEYS + ["axis", "value", "mcse", "n"]].rename(
        columns={"value": "prev_event", "mcse": "prev_event_mcse", "n": "n_pairs"})
    oc = fa.merge(ms, on=KEYS + ["axis", "tau"], how="left").merge(pe, on=KEYS + ["axis"], how="left")
    oc["hit"] = 1.0 - oc["miss"]
    # PPV = P(true error | fires) from pair counts
    a = (oc["hit"] * oc["n_event_pairs"]).fillna(0)
    b = oc["false_alarm"] * oc["n_nonevent_pairs"]
    oc["n_fired_pairs"] = np.rint(a + b).astype(int)
    oc["n_fired_event"] = np.rint(a).astype(int)
    oc["ppv"] = np.where(oc["n_fired_pairs"] > 0, oc["n_fired_event"] / oc["n_fired_pairs"].clip(lower=1), np.nan)
    oc["ppv_mcse"] = np.sqrt(oc["ppv"] * (1 - oc["ppv"]) / oc["n_fired_pairs"].clip(lower=1))
    oc["ppv_mcse_conservative"] = oc["ppv_mcse"] * np.sqrt(3)
    for c in ("false_alarm", "hit", "ppv"):
        se = oc[c + "_mcse"]
        oc[c + "_lo95"] = oc[c] - Z * se
        oc[c + "_hi95"] = oc[c] + Z * se
    oc["role"] = np.select([oc.tau.isin([2.0, 3.0]), oc.tau.isin([5.0, 6.0])],
                           ["t_warn only", "t_adj only"], "t_warn and t_adj")
    oc.to_csv(OUT / "E4_operating.csv", index=False)

    # PPV of the adjudication trigger at the base t_adj = 5 and the base cell, both axes
    q = oc[(oc.u == BASE["u"]) & oc.view_errors & (oc.r_mean == BASE["r_mean"])
           & (oc.N_beats == BASE["N_beats"])]
    q[["u", "view_errors", "r_mean", "beat_cv", "N_beats", "axis", "tau", "prev_event", "hit", "false_alarm",
       "ppv", "ppv_mcse", "ppv_mcse_conservative", "ppv_lo95", "ppv_hi95", "n_fired_pairs"]].to_csv(
        OUT / "E4_ppv_base.csv", index=False)

    # ---- 3. ellipticity: cross-axis differences in the anchor view ----------------------
    ax = df[df.axis == "AP-SL"]
    meas = per_threshold(ax, "p_anchor_axis_diff_ge_twarn", "p_anchor_axis_diff_ge_tadj", ["AP-SL"])
    meas["quantity"] = "measured anchor |AP-SL| >= tau"
    true = ax[ax.metric == "p_true_axis_diff_ge_twarn"].copy()
    true["tau"] = true["t_warn"]
    true = true.groupby(KEYS + ["tau"], as_index=False).first()
    true["quantity"] = "true |S_AP-S_SL| >= tau"
    ell = pd.concat([meas, true[KEYS + ["axis", "tau", "value", "mcse", "n", "quantity"]]], ignore_index=True)
    ell.to_csv(OUT / "E4_ellipticity_cells.csv", index=False)
    # truth does not depend on beat_cv, N, u or view_errors: pool over the 144 cells per r_mean
    tp = true.groupby(["r_mean", "tau"]).agg(k=("value", lambda v: np.sum(v * true.loc[v.index, "n"])),
                                            n=("n", "sum")).reset_index()
    tp["value"] = tp["k"] / tp["n"]
    tp["mcse"] = np.sqrt(tp["value"] * (1 - tp["value"]) / tp["n"])
    tp["source"] = "E4 cells pooled (144 cells x 1e5 patients)"
    # supplementary: tau = 5, 6 not stored by E4
    cfg = C.load_config(ROOT / "code" / "configs" / "base.yaml")
    sup = []
    for j, rm in enumerate(sorted(df.r_mean.unique())):
        p = C.cell_params(cfg, "E4", 0, {"r_mean": float(rm), "K": 1, "window": None, "N_beats": 1})
        ss = np.random.SeedSequence(cfg["meta"]["master_seed"], spawn_key=(SUPP_SEED_KEY, j))
        d = []
        for ch in ss.spawn(SUPP_N // SUPP_CHUNK):
            lat = draw_latent(p, SUPP_CHUNK, np.random.Generator(np.random.PCG64(ch)))
            d.append(np.abs(lat.S[:, 0] - lat.S[:, 1]))
        d = np.concatenate(d)
        for tau in (2.0, 3.0, 4.0, 5.0, 6.0):
            v = float((d >= tau).mean())
            sup.append(dict(r_mean=rm, tau=tau, value=v, mcse=np.sqrt(v * (1 - v) / d.size), n=d.size,
                            source=f"supplementary draw_latent, seed (20260918, spawn_key=({SUPP_SEED_KEY},{j}))"))
        sup.append(dict(r_mean=rm, tau=np.nan, value=float(np.mean(d)), mcse=float(np.std(d, ddof=1) / np.sqrt(d.size)),
                        n=d.size, source="mean |S_AP-S_SL| mm, supplementary draw"))
        sup.append(dict(r_mean=rm, tau=np.nan, value=float(np.median(d)), mcse=np.nan,
                        n=d.size, source="median |S_AP-S_SL| mm, supplementary draw"))
    sup = pd.DataFrame(sup)
    truth = pd.concat([tp[["r_mean", "tau", "value", "mcse", "n", "source"]], sup], ignore_index=True)
    truth.to_csv(OUT / "E4_ellipticity_truth.csv", index=False)

    # ---- 4. compact summary of headline numbers -------------------------------------------
    b = anyt[(anyt.u == "base") & (anyt.r_mean == 0.64) & (anyt.N_beats == 3)]
    b.to_csv(OUT / "E4_any_trigger_base.csv", index=False)
    print("wrote", sorted(p.name for p in OUT.glob("E4_*.csv")))


if __name__ == "__main__":
    main()
