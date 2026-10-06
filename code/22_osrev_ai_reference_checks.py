#!/usr/bin/env python
"""Protocol amendment 3, package A3-5: two diagnostic checks (login node, about 2 CPU-min).

1. Shape of the inter-reader difference d = read 1 - read 2 within a fixed reader pair. Regenerates
   the first N_BLOCKS blocks of the nested simulation (same seeds as the batch job:
   SeedSequence(20261005, spawn_key=(75, 3, 0, block))) and reports the within-pair skewness g1
   and excess kurtosis g2 of d. For independent, identically distributed differences the large-n
   variance of the upper limit L = mean + 1.96 SD is
       n Var(L) / sigma^2 = 1 + 1.96^2 (g2 + 2) / 4 + 1.96 g1,
   against 1 + 1.96^2 / 2 (rounded to 3 by Bland and Altman) for normal differences. The check
   compares the implied ratio of the conditional SD of L to the Bland-Altman SE with the ratio
   observed in loa_conditional_vs_marginal.csv.
2. Independence of the second AI-noise array z2 from the reference error in cell cv30, where the
   analytic check of the apparent MSE gave z scores down to -3.7 for the models using z2
   (regenerates cell 2 of the e5 part with its seed and tests mean(z2 x e_ref) against 0 per
   block of studies).

Writes check_interreader_difference_shape.csv and check_noise_independence_cv30.csv to
results/2026-10-05_osrev/ai_reference/.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from duomaxsim import config as C  # noqa: E402
from duomaxsim import osrev_ai_reference as OA  # noqa: E402

spec = importlib.util.spec_from_file_location("osrev_run", ROOT / "code" / "20_osrev_ai_reference_run.py")
RUN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RUN)

OUT = ROOT / "results" / "2026-10-05_osrev" / "ai_reference"
N_BLOCKS = 20          # 100 reader pairs x 50 samples x 200 cases = 1,000,000 double-read cases


def shape_check(oc):
    nc = oc["loa_nested"]
    p = RUN._e6_base_params(oc, nc["scenarios"][0]["sigma_rb_mm"])
    m2 = m3 = m4 = 0.0
    n = 0
    big = 0
    for block in range(N_BLOCKS):
        reads, _ = OA.reader_panel_reads(p, RUN.gen(oc, oc["streams"]["loa_nested"], 0, block),
                                         int(nc["pairs_per_block"]), int(nc["n_rep"]), int(nc["n_cases"]), 2)
        d = (reads[..., 0] - reads[..., 1]).reshape(int(nc["pairs_per_block"]), -1)
        c = d - d.mean(axis=1, keepdims=True)                 # centred within reader pair
        m2 += (c ** 2).sum()
        m3 += (c ** 3).sum()
        m4 += (c ** 4).sum()
        n += c.size
        big += int((np.abs(c) > 3 * c.std(axis=1, keepdims=True)).sum())
    v = m2 / n
    g1 = (m3 / n) / v ** 1.5
    g2 = (m4 / n) / v ** 2 - 3.0
    k_normal = 1 + 1.96 ** 2 / 2
    k = 1 + 1.96 ** 2 * (g2 + 2) / 4 + 1.96 * g1
    tab = pd.read_csv(OUT / "loa_conditional_vs_marginal.csv")
    obs = tab[(tab.scenario == "rb075")].set_index("n_cases").ratio_conditional_sd_to_ba_se
    row = dict(scenario="rb075", n_reader_pairs=N_BLOCKS * int(nc["pairs_per_block"]), n_differences=n,
               seed=f"SeedSequence(20261005, spawn_key=(75, 3, 0, 0..{N_BLOCKS - 1}))",
               var_within_pair=v, sd_within_pair=np.sqrt(v), skewness=g1, excess_kurtosis=g2,
               n_beyond_3sd=big, p_beyond_3sd=big / n, p_beyond_3sd_normal=0.0026998,
               n_var_limit_over_sigma2_normal=k_normal, n_var_limit_over_sigma2_implied=k,
               ratio_sd_to_ba_se_implied_large_n=float(np.sqrt(k / 3.0)),
               ratio_observed_n25=float(obs[25]), ratio_observed_n50=float(obs[50]),
               ratio_observed_n100=float(obs[100]), ratio_observed_n200=float(obs[200]))
    df = pd.DataFrame([row])
    df.to_csv(OUT / "check_interreader_difference_shape.csv", index=False)
    return df


def noise_check(oc):
    ci = 2
    spec_ = oc["e5"]["cells"][ci]
    cfg = C.load_config(ROOT / oc["meta"]["base_config"])
    p = C.cell_params(cfg, "E5", RUN.find_cell(cfg, "E5", spec_["overrides"]))
    d = OA.e5_draws(p, RUN.seedseq(oc, oc["streams"]["e5_validation"], ci), int(oc["e5"]["n_studies"]),
                    int(oc["meta"]["chunk_size"]), extra_noise=True)
    rows = []
    e_img = d["R_img"] - d["T1"]
    for nm, x in (("e_ref_single", d["r1"] - d["T1"]), ("e_ref_mean2", 0.5 * (d["r1"] + d["r2"]) - d["T1"]),
                  ("e_img", e_img), ("T1", d["T1"])):
        for zn in ("z", "z2"):
            prod = (d[zn] * (x - x.mean())).mean(axis=1)          # per-study mean of the cross product
            m, se = prod.mean(), prod.std(ddof=1) / np.sqrt(prod.size)
            halves = [prod[:1000].mean() / (prod[:1000].std(ddof=1) / np.sqrt(1000)),
                      prod[1000:].mean() / (prod[1000:].std(ddof=1) / np.sqrt(1000))]
            rows.append(dict(cell_id=spec_["id"], noise=zn, variable=nm, mean_cross_product=m, mcse=se, z=m / se,
                             z_first_1000_studies=halves[0], z_last_1000_studies=halves[1],
                             correlation=float(np.corrcoef(d[zn].ravel(), x.ravel())[0, 1])))
    for zn in ("z", "z2"):
        zz = d[zn]
        rows.append(dict(cell_id=spec_["id"], noise=zn, variable="mean", mean_cross_product=float(zz.mean()),
                         mcse=float(1 / np.sqrt(zz.size)), z=float(zz.mean() * np.sqrt(zz.size))))
        rows.append(dict(cell_id=spec_["id"], noise=zn, variable="variance_minus_1", mean_cross_product=float(zz.var() - 1),
                         mcse=float(np.sqrt(2 / zz.size)), z=float((zz.var() - 1) / np.sqrt(2 / zz.size))))
    rows.append(dict(cell_id=spec_["id"], noise="z,z2", variable="correlation_z_z2",
                     mean_cross_product=float(np.corrcoef(d["z"].ravel(), d["z2"].ravel())[0, 1]),
                     mcse=float(1 / np.sqrt(d["z"].size)),
                     z=float(np.corrcoef(d["z"].ravel(), d["z2"].ravel())[0, 1] * np.sqrt(d["z"].size))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "check_noise_independence_cv30.csv", index=False)
    return df


if __name__ == "__main__":
    oc = RUN.load()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(shape_check(oc).T.to_string())
    print(noise_check(oc).to_string())
