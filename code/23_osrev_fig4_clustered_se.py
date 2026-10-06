#!/usr/bin/env python
"""Patient-clustered standard errors for Figure 4, panels c and d (third amendment, display only).

The stored E4 analysis tables hold pair-level counts but no per-patient data, so the published
Figure 4 drew a binomial standard error multiplied by sqrt(3). This script re-simulates the six E4
cells behind panels c and d (base view accuracy, view errors on, mean SL/AP ratio 0.64, three beats,
beat-to-beat variation 5% to 30%: cells 181, 187, 193, 199, 205, 211 of code/configs/base.yaml) with
their original seeds, SeedSequence(20260918, spawn_key=(4, cell)), keeps the per-patient pattern of
true-positive, false-positive and false-negative view pairs, and writes pair-level rates with
patient-level linearization (delta method) standard errors.

Every count is asserted equal to the stored value in results/2026-09-18_full/analysis/E4_operating.csv,
so the point estimates of Figure 4 are unchanged.

Run: sbatch --wait code/slurm/osrev_fig4_clustered.sh
Output: results/2026-10-05_osrev/fig4_clustered/fig4_clustered_se.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from duomaxsim import config as C  # noqa: E402
from duomaxsim import osrev_triggers as T  # noqa: E402

SPEC = yaml.safe_load(open(ROOT / "code" / "configs" / "osrev_triggers.yaml"))
META = SPEC["meta"]
CHUNK = int(META["chunk_size"])
THR = float(META["error_threshold_mm"])
TAUS = [float(t) for t in META["taus_mm"]]
CFG = C.load_config(ROOT / META["base_config"])
AN = ROOT / "results" / "2026-09-18_full" / "analysis"
AXIS, D = "AP", 0
Z = 1.959964


def cells():
    oc = pd.read_csv(AN / "E4_operating.csv")
    s = oc[(oc.u == "base") & oc.view_errors & (oc.r_mean == 0.64) & (oc.N_beats == 3) & (oc.axis == AXIS)]
    return s


def work(task):
    cell, i, n_total = task
    p = C.cell_params(CFG, "E4", int(cell))
    ss = C.cell_seed(CFG, "E4", int(cell))
    _, rc = next(T.cohort_chunks(p, ss, n_total, CHUNK, only=i))
    lat, vm = rc["lat"], rc["vm"]
    out = {}
    for tau in TAUS:
        fired = T.trigger_fired(vm, tau)
        ev = T.pair_event_definitions(lat, THR)["D0_published"]
        out[tau] = T.pattern_hist(fired[:, D], ev[:, D])
    return cell, i, out, (int(ss.entropy), tuple(int(k) for k in ss.spawn_key)), float(p["beat_cv"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results" / "2026-10-05_osrev" / "fig4_clustered"))
    a = ap.parse_args()
    stored = cells()
    cl = sorted(stored.cell.unique())
    n_total = int(stored.n_pairs.iloc[0]) // 3
    nch = T.n_chunks(n_total, CHUNK)
    tasks = [(c, i, n_total) for c in cl for i in range(nch)]
    H, seed, cv = {}, {}, {}
    with mp.Pool(min(a.workers, len(tasks))) as pool:
        for cell, i, out, sd, bcv in pool.imap_unordered(work, tasks, chunksize=1):
            seed[cell], cv[cell] = sd, bcv
            for tau, h in out.items():
                H[(cell, tau)] = H.get((cell, tau), 0) + h
    rows = []
    for (cell, tau), h in sorted(H.items()):
        r = T.rates_from_patterns(h)
        st = stored[(stored.cell == cell) & (stored.tau == tau)].iloc[0]
        c = r["counts"]
        # point estimates must be those of the stored analysis (bit-identical counts)
        assert c["tp"] == int(st.n_fired_event), (cell, tau, c, st.n_fired_event)
        assert c["tp"] + c["fp"] == int(st.n_fired_pairs), (cell, tau)
        assert c["tp"] + c["fn"] == int(st.n_event_pairs), (cell, tau)
        assert c["fp"] + c["tn"] == int(st.n_nonevent_pairs), (cell, tau)
        assert abs(r["ppv"]["value"] - st.ppv) < 1e-12 and abs(r["sensitivity"]["value"] - st.hit) < 1e-12
        assert abs(r["false_positive_rate"]["value"] - st.false_alarm) < 1e-12
        assert abs(cv[cell] - st.beat_cv) < 1e-12
        for m in ("ppv", "sensitivity", "false_positive_rate"):
            q = r[m]
            rows.append(dict(cell=int(cell), beat_cv=cv[cell], axis=AXIS, tau_mm=tau, measure=m,
                             value=q["value"], numerator=q["num"], denominator=q["den"],
                             se_binomial_pairs_independent=q["se_iid"], se_sqrt3_v05=q["se_sqrt_m"],
                             se_patient_clustered=q["se_cluster"], design_effect=q["deff"],
                             lo95_clustered=max(0.0, q["value"] - Z * q["se_cluster"]),
                             hi95_clustered=min(1.0, q["value"] + Z * q["se_cluster"]),
                             n_patients=c["n_patients"], pairs_per_patient=c["pairs_per_patient"],
                             seed=f"SeedSequence({seed[cell][0]}, spawn_key={seed[cell][1]})",
                             counts_identical_to_E4_operating=True))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "fig4_clustered_se.csv", index=False)
    b = df[(df.cell == 193) & (df.tau_mm == 5.0) & (df.measure == "ppv")].iloc[0]
    print(f"cells {cl}; rows {len(df)}; all counts identical to E4_operating.csv")
    print(f"base cell 193, 5 mm PPV {b.value:.4f} ({b.numerator}/{b.denominator}), clustered 95% MCI "
          f"{b.lo95_clustered:.4f} to {b.hi95_clustered:.4f}, design effect {b.design_effect:.3f}")
    print("design effect range, PPV:", df[df.measure == "ppv"].design_effect.min().round(3),
          df[df.measure == "ppv"].design_effect.max().round(3))


if __name__ == "__main__":
    main()
