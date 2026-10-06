# results/2026-10-05_osrev/thresholds

Protocol amendment 3, package A3-3 (thresholds), written 2026-10-05. Post hoc reporting detail requested by the senior author (comment C201): classification at the illustrative cut-offs 7, 10 and 13 mm with full 2 by 2 tables, the true-span distribution, and near-threshold behaviour. The generative model, the rules and the estimand of experiment E3 are unchanged. The cut-offs are illustrative values on the simulated span scale and are not clinical eligibility criteria.

Findings and suggested manuscript text: `notes/scratch/2026-10-05-findings-thresholds.md` (local, untracked).

## How to regenerate

Run from the project root with `/project/home/p201509/envs/duomax-sim/bin/python` (3.11.16).

1. `python code/20_osrev_thresholds_run.py published` (login node, about 20 CPU-s). Rebuilds the patients of six stored E3 cells with their original seeds, asserts that the rebuilt summary rows equal the stored parquet rows exactly, and writes `raw/published_*`.
2. `sbatch code/slurm/osrev_thresholds.sh` (job 5301601, 36 s on one node, 1.28 core-hours). Writes `raw/replicates_*` and `provenance_replicates.txt`.
3. `python code/21_osrev_thresholds_analyse.py` (login node, about 10 CPU-s). Writes the csv tables below.
4. `python code/figures/figS6_true_span_distribution.py`. Writes `figures/figS6_true_span_distribution.*` and its source csv.
5. `python -m pytest tests/test_osrev_thresholds.py tests/test_regression.py` (20 tests: 19 for this package and the regression fixture).

Configuration: `code/configs/osrev_thresholds.yaml`. Library: `code/lib/duomaxsim/osrev_thresholds.py`. Checksums, versions and seeds: `provenance.txt`.

## Samples

- `published_seed`: the 100,000 patients of a stored E3 cell, regenerated with that cell's original seed (base.yaml cells 29 and 77, master seed 20260918; amend2 cells 0, 2, 9 and 11, master seed 20260920; spawn key (3, cell)). These are the patients behind the published numbers.
- `amend3_pooled`: 100 new replicates of 100,000 patients per scenario, pooled (10,000,000 patients). SeedSequence(entropy 20261005, spawn_key (73, 1, replicate)); the same key for every scenario, so scenarios share case mix, latent error drivers, beats, reader bias and caliper errors (common random numbers).

Scenario ids: `base` (anchor 4.8%, long-axis 20% mean underestimation; pre-specified base case), `low8` (2.4%, 8%; the 8% long-axis setting, which at three views has the same parameters as the most accurate view-accuracy scenario), `acc_2p4_8` (same parameters as `low8`, amendment-2 seed; published-seed sample only), `acc_20_40`, `acc_2p4_40`, `acc_20_8` (view-accuracy corners), `base_median13` (base case with median true AP span 13 mm; pooled sample only). All: three views, three beats, ±15% window, beat variation 15%, overestimation on.

Rules: A1 anchor-view mean, A2 mean across views, A3 largest view mean, A4 largest view mean after review. Event: true maximal span T1 at or above the cut-off on the analysed axis. Positive: reported value at or above the cut-off.

## Files

| File | Rows | Produced by | Content |
|---|---|---|---|
| `thresholds_true_span_distribution.csv` | population x axis | step 3 | Family and parameters, mean, SD, percentiles and proportions at or above 7, 10 and 13 mm (closed form for AP, 400-node Gauss-Legendre quadrature over the SL/AP ratio for SL); simulated counts at or above each cut-off (`sim_n_ge_*`, of `sim_n`); empirical quantiles of the published base-case sample |
| `thresholds_true_span_density.csv` | kind x population x axis x x_mm | step 3 | `kind = density`: density and CDF on a 0.1 mm grid; `kind = histogram`: pooled simulated counts in 0.25 mm bins |
| `thresholds_2x2.csv` | sample x scenario x axis x cut-off x rule | step 3 | Unrounded tp, fp, fn, tn; the same per 100,000 and per 1,000; sensitivity, specificity, ppv, npv, prevalence, accuracy, each with `_mcse`, `_lo95`, `_hi95`, `_n`; reclassification counts against A1 (`ev_up`, `ev_dn`, `ne_up`, `ne_dn`); `nri_events_vs_A1` (difference in sensitivity), `nri_nonevents_vs_A1` (difference in specificity), `nri_vs_A1` (difference in Youden index), `net_correct_per_1000_vs_A1` (difference in accuracy per 1,000 patients), with paired MCSE |
| `thresholds_near_band.csv` | sample x scenario x axis x cut-off x half-width x rule | step 3 | Patients whose true span is within 1 or 2 mm of the cut-off: counts, false positives and false negatives per 1,000 such patients, conditional rates on each side, and the share of all false positives or false negatives that lies in the band |
| `thresholds_by_distance.csv` | sample x scenario x axis x cut-off x rule x 0.5 mm bin | step 3 | Misclassification probability by true span minus cut-off, bins [lo, lo + 0.5), -5 to +5 mm; false positives below the cut-off, false negatives at or above it |
| `thresholds_reproduction.csv` | scenario x axis x cut-off x rule x metric | step 3 | Stored E3 value, rebuilt value (identical), pooled amendment-3 value and z score |
| `thresholds_replicate_check.csv` | scenario x axis x cut-off x rule | step 3 | SD across the 100 replicates against the binomial MCSE for n = 100,000 |
| `raw/published_reproduction_check.csv` | scenario | step 1 | Exact-agreement check against the stored parquet rows (294 rows per cell) |
| `raw/published_base_true_span_empirical.csv` | quantile | step 1 | Empirical quantiles, mean and SD of the true spans in the published base-case sample |
| `raw/*_overall.parquet`, `*_band.parquet`, `*_bins.parquet`, `*_hist.parquet` | per sample, scenario and replicate | steps 1 and 2 | Count tables before pooling |
| `raw/*_manifest.json` | | steps 1 and 2 | Command, configuration, versions, host, date |

95% Monte Carlo interval = estimate ± 1.959964 MCSE. MCSE of a proportion = sqrt(p(1 - p)/n) with n the denominator of that proportion; reclassification MCSEs are paired (same patients under both rules). Intervals describe simulation precision only.
