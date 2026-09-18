#!/usr/bin/env python
"""E1 analytic check (protocol 6a) and offset decomposition of the oracle estimator (exploratory).

Part 1. Noise-only setting (view_errors = False): the order-statistic inflation of the
maximum of view means, D = A3 - A1 (paired, per patient, AP axis, estimand T1), is compared
with E[max of K iid N(0,1)] * SE, where
  P_brief : SE = SD of the A1 error vs T1 in the same cell (as specified in the analysis brief);
  P_exact : SE_i per patient from the model, SE_i^2 = (g_i S_i)^2 Var(mean of N lognormal
            beat factors) + sigma_cal^2 / N, averaged as c_K * mean_i(SE_i) (valid without the
            acceptance window, i.e. variants V2 and V3 only).
Variants
  V1 literal : base case with view_errors False (window 0.15, reader bias, instrument factor, case mix)
  V2 no_shared_no_window : V1 with sigma_rb 0, g_log_sd 0, window None
  V3 homoscedastic : V2 with S_log_sd 0 (every AP span 10 mm)
Reader bias and the instrument factor are shared by all views of a patient, so they enter the
A1 error SD but not D; P_brief is therefore expected to overstate D in V1.

Part 2. Offset decomposition of A7 bias vs T1 at the base cell (view errors on): base; beat
noise mean-centred; plus g_log_sd 0; plus window None.

Seeds: SeedSequence(20260918, spawn_key=(1, 900000 + j)), one child per 20,000 patients.
The spawn keys do not overlap the E1 grid (cells 0 to 2303). Library code is used unchanged.
Output: results/2026-09-18_full/analysis/E1E3_e1_analytic_check.csv, E1E3_e1_a7_offsets.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from duomaxsim import config as C  # noqa: E402
from duomaxsim.estimators import a7_oracle  # noqa: E402
from duomaxsim.experiments import read_once  # noqa: E402
from duomaxsim.metrics import expected_max_std_normals  # noqa: E402
from duomaxsim.model import draw_latent, log_sd_from_cv  # noqa: E402

OUT = ROOT / "results" / "2026-09-18_full" / "analysis"
N_PAT = 100_000
CHUNK = 20_000
Z = 1.959964
VARIANTS = {
    "V1_literal": dict(view_errors=False),
    "V2_no_shared_no_window": dict(view_errors=False, sigma_rb_mm=0.0, g_log_sd=0.0, window=None),
    "V3_homoscedastic": dict(view_errors=False, sigma_rb_mm=0.0, g_log_sd=0.0, window=None, S_log_sd=0.0),
}
KS = [2, 3, 4]
CVS = [0.05, 0.15, 0.30]


def rngs(j):
    ss = np.random.SeedSequence(entropy=20260918, spawn_key=(1, 900000 + j))
    for i, ch in enumerate(ss.spawn(-(-N_PAT // CHUNK))):
        yield min(CHUNK, N_PAT - i * CHUNK), np.random.Generator(np.random.PCG64(ch))


def run_check(cfg):
    rows, j = [], 0
    for vname, ov in VARIANTS.items():
        for K in KS:
            for cv in CVS:
                p = C.base_params(cfg)
                p.update(ov)
                p.update(K=K, beat_cv=cv)
                D, E1, SEi = [], [], []
                for n, rng in rngs(j):
                    lat = draw_latent(p, n, rng)
                    rb = p["sigma_rb_mm"] * rng.standard_normal(n)
                    _, br = read_once(lat, p, rng, rb)
                    vm = br.value[:, 0, :]                       # AP axis, (n, K)
                    D.append(vm.max(axis=1) - vm[:, 0])
                    E1.append(vm[:, 0] - lat.S[:, 0])
                    s2 = log_sd_from_cv(cv) ** 2
                    var_f = (np.exp(s2) - 1.0) * np.exp(s2) / p["N_beats"]
                    SEi.append(np.sqrt((lat.g * lat.S[:, 0]) ** 2 * var_f + p["sigma_cal_mm"] ** 2 / p["N_beats"]))
                D, E1, SEi = map(np.concatenate, (D, E1, SEi))
                cK = expected_max_std_normals(K)
                sd1 = E1.std(ddof=1)
                sd1_se = sd1 / np.sqrt(2 * (E1.size - 1))
                mc, mc_se = D.mean(), D.std(ddof=1) / np.sqrt(D.size)
                pb, pe = cK * sd1, cK * SEi.mean()
                rows.append(dict(variant=vname, K=K, beat_cv=cv, N_beats=p["N_beats"], n=D.size, c_K=cK,
                                 mc_A3_minus_A1=mc, mc_mcse=mc_se,
                                 mc_lo95=mc - Z * mc_se, mc_hi95=mc + Z * mc_se,
                                 sd_err_A1=sd1, sd_err_A1_mcse=sd1_se,
                                 pred_brief=pb, pred_brief_mcse=cK * sd1_se,
                                 ratio_mc_to_brief=mc / pb,
                                 z_brief=(mc - pb) / np.hypot(mc_se, cK * sd1_se),
                                 pred_exact=pe if ov.get("window", "x") is None else np.nan,
                                 ratio_mc_to_exact=mc / pe if ov.get("window", "x") is None else np.nan,
                                 z_exact=(mc - pe) / mc_se if ov.get("window", "x") is None else np.nan))
                j += 1
    return pd.DataFrame(rows)


def run_offsets(cfg):
    steps = [("base", {}),
             ("centring_mean", dict(beat_noise_centering="mean")),
             ("centring_mean_g0", dict(beat_noise_centering="mean", g_log_sd=0.0)),
             ("centring_mean_g0_nowindow", dict(beat_noise_centering="mean", g_log_sd=0.0, window=None))]
    rows = []
    for k, (name, ov) in enumerate(steps):
        p = C.base_params(cfg)
        p.update(ov)
        e = {"A7": [], "A1": [], "A3": []}
        for n, rng in rngs(100 + k):  # common random numbers across steps
            lat = draw_latent(p, n, rng)
            rb = p["sigma_rb_mm"] * rng.standard_normal(n)
            _, br = read_once(lat, p, rng, rb)
            vm = br.value[:, 0, :]
            e["A7"].append(a7_oracle(br.value, lat)[:, 0] - lat.S[:, 0])
            e["A1"].append(vm[:, 0] - lat.S[:, 0])
            e["A3"].append(vm.max(axis=1) - lat.S[:, 0])
        for est, v in e.items():
            v = np.concatenate(v)
            rows.append(dict(step=name, estimator=est, axis="AP", estimand="T1", bias_mm=v.mean(),
                             mcse=v.std(ddof=1) / np.sqrt(v.size), n=v.size))
    return pd.DataFrame(rows)


def main():
    t0 = time.process_time()
    cfg = C.load_config(ROOT / "code" / "configs" / "base.yaml")
    OUT.mkdir(parents=True, exist_ok=True)
    chk = run_check(cfg)
    chk.to_csv(OUT / "E1E3_e1_analytic_check.csv", index=False)
    off = run_offsets(cfg)
    off.to_csv(OUT / "E1E3_e1_a7_offsets.csv", index=False)
    pd.set_option("display.width", 250)
    print(chk[["variant", "K", "beat_cv", "mc_A3_minus_A1", "mc_mcse", "pred_brief", "ratio_mc_to_brief",
               "pred_exact", "ratio_mc_to_exact", "z_exact"]].round(4).to_string())
    print(off.round(4).to_string())
    print(f"cpu {time.process_time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
