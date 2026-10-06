# Protocol amendment 3, package A3-1: beat rules

Project DUO-Max, `the project root`. Run 2026-10-05, Slurm job 5301653 (MeluXina, node mel0557, partition cpu, qos short, elapsed 00:01:08, 128 cores, about 2.4 core-hours, 0.019 node-hours).

Post hoc sensitivity analyses requested by the senior author on 2026-10-05 (comments C185, C196 and the beat part of C324), defined in `notes/protocol-amendment-3-2026-10-05.md` before coding. They replace no pre-specified analysis.

## How to regenerate

```
sbatch --wait code/slurm/osrev_beats.sh        # runs the two scripts below on one cpu node
```

1. `code/20_osrev_beats_run.py --workers 48` simulates every condition of `code/configs/osrev_beats.yaml` (copied here as `config_used.yaml`) with the model variants of `code/lib/duomaxsim/osrev_beats.py` and writes the `beats_*.csv` files that do not start with `beats_table_`.
2. `code/21_osrev_beats_tables.py` derives the `beats_table_*.csv` files from them (no simulation).

Tests: `tests/test_osrev_beats.py` (30 cases) and `tests/test_regression.py` passed on 2026-10-05 with `/project/home/p201509/envs/duomax-sim/bin/python -m pytest`.

Versions, commit and sha256 of every code and configuration file used are in `provenance.txt`. The new files are untracked additions to commit 57371d3; no pre-existing file was changed.

## Design in brief

- 100,000 simulated patients per condition, in five chunks of 20,000. Anchor view only, K = 1 as in the pre-specified experiment E2 (condition `base_K3` repeats the base case with three views).
- Seeds: numpy `SeedSequence(entropy=20261005, spawn_key=(71, grid, condition, chunk, stream))`; stream 0 = model and measurement, 1 = permutation for the matched arm, 2 = bootstrap (chunk index 0). Grid numbers: 1 budget, 2 sens, 3 retro, 4 check_K3, 5 check_exhaustive. The keys are written in the column `seed_spawn_key`.
- Within a condition all arms are computed on the same patients, beats, reader bias and caliper errors (common random numbers), so contrasts between arms are paired. Conditions are independently seeded.
- Reproduction (grid 0): 16 published E2 cells re-run with the unchanged library, `code/configs/base.yaml` and their original seeds (entropy 20260918).
- Estimands: `T1` = true maximal span; `Tv` = g x mu_anchor, the anchor-view expected span under the study's instrument setting. Errors are estimate minus estimand, in mm; `relbias_pct` is 100 x mean(error / estimand).
- MCSE = Monte Carlo standard error. Paired MCSE: SD of the per-patient difference / sqrt(n) for bias, MAE and relative bias; delta method on the paired squared errors for RMSE (`paired_summary` in the module). Columns ending `_mcse_indep` give the value the same data would yield if the two arms were treated as independent.

## Arms (column `arm`)

| Arm | Definition |
|---|---|
| `plain_k` | mean of the first k measured beats, k = 1 to 30 |
| `winN_bB` | ±15% window rule for N accepted beats (N = 3 or 5) with at most B beats acquired in total (B = 30 is the published prospective rule); fallback = mean of all B beats |
| `winNfn_bB` | as above with the alternative fallback, mean of the first N beats |
| `sameacqN` | mean of all beats that the unlimited window rule (`winN_b30`) acquired for that view (same acquisition, no selection) |
| `permacqN` | mean of the first k beats, where k is the number of beats acquired by `winN_b30` in another, randomly permuted patient of the same chunk (same distribution of beat counts, unrelated to the view's values) |

## Files

Written by `code/20_osrev_beats_run.py`:

| File | Rows | Content |
|---|---|---|
| `beats_arm_errors.csv` | 7,192 | bias, RMSE, MAE, relative bias with MCSE for every arm, condition, axis and estimand |
| `beats_paired_contrasts.csv` | 8,352 | paired differences between arms (`arm_a` minus `arm_b`) with paired and independent-formula MCSE |
| `beats_window_rule_counts.csv` | 812 | unrounded counts per window arm: views, window met by the first N beats, accepted within budget, limited; beats acquired (sum, mean, MCSE, median, 90th percentile, maximum); mean caliper SD |
| `beats_rmse_difference_intervals.csv` | 16 | base case, sinus rhythm and AF: RMSE difference window minus plain, delta-method paired MCSE, independent-formula MCSE, patient-level bootstrap (2,000 resamples) |
| `beats_retro_limited_denominators.csv` | 8 | retrospective mode: counts of views with fewer than N available beats, limited without and with a window, and limited despite at least N beats; mean available beats; correlation of available beats with T1 |
| `beats_retro_errors.csv` | 112 | retrospective mode: errors of the plain and windowed rules, all views and by limited flag; paired window effect |
| `beats_retro_available_distribution.csv` | 30 | counts of views by number of available beats |
| `beats_check_exhaustive_subsets.csv` | 4 | library search (adjacent sorted values) against a search over every N-subset, budget 10 |
| `beats_reproduction_E2_cells.csv` | 2,096 | every row of 16 published E2 cells: stored value, re-run value, `identical` flag |

Written by `code/21_osrev_beats_tables.py`:

| File | Rows | Content |
|---|---|---|
| `beats_table_published_numbers_check.csv` | 19 | manuscript beat-rule numbers: stored value, bit-identical re-run, amendment-3 estimate from new seeds, z of the difference |
| `beats_table_limited_denominators.csv` | 32 | limited-sampling counts and denominators, published cells and amendment 3 |
| `beats_table_equal_budget.csv` | 336 | window rule against the plain mean of all B beats and of the first N beats, by condition, axis, estimand, N and budget B |
| `beats_table_equal_expected_beats.csv` | 232 | window rule at the 30-beat cap against plain means with the same number of beats; decomposition into selection and acquisition |
| `beats_table_rmse_difference_published_vs_paired.csv` | 16 | published independent-arm interval next to the paired intervals |
| `beats_table_sensitivity.csv` | 44 | serial correlation, respiratory component, span-dependent caliper noise: pass rates, RMSE by number of beats, window effects, contrasts with the reference |
| `beats_table_rmse_by_n_beats.csv` | 3,360 | long table of the plain mean of the first n beats (n = 1 to 30) for the budget and sensitivity grids |

Proportions are not stored; every percentage can be formed from the count columns (`n_*`) and `n_views`.
