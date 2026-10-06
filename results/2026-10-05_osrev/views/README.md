# results/2026-10-05_osrev/views

Protocol amendment 3, package A3-2 (view rules). Post hoc sensitivity analyses requested by the senior author on 2026-10-05 (comments C200, C218, C259, C396 and the minimax part of C188). Nothing here replaces a pre-specified analysis.

## How to regenerate

```
sbatch --wait code/slurm/osrev_views.sh                                        # simulation, one cpu node, about 2.5 min
/project/home/p201509/envs/duomax-sim/bin/python code/21_osrev_views_analyse.py   # tables, login node, about 10 s
/project/home/p201509/envs/duomax-sim/bin/python -m pytest tests/test_osrev_views.py tests/test_regression.py
```

Run as Slurm job 5301599 on 2026-10-05 (node mel0429, elapsed 00:02:24, 128 physical cores, 5.12 core-hours, 0.04 node-hours; Slurm bills 256 hardware threads, CPUTimeRAW 36,864 s). Versions, git commit and sha256 of every code file are in `provenance.txt`.

## Code

| File | Role |
| --- | --- |
| `code/lib/duomaxsim/osrev_views.py` | Per-patient values of the four view rules, counterfactual arms, error summaries with Monte Carlo standard errors |
| `code/configs/osrev_views.yaml` | Families, scenarios and seeds |
| `code/20_osrev_views_run.py` | Runner (142 tasks, 986 cells of 100,000 patients) and merge |
| `code/slurm/osrev_views.sh` | Batch script |
| `code/21_osrev_views_analyse.py` | Tables below |
| `tests/test_osrev_views.py` | 14 tests: bit-identity with `experiments.run_E1`, known answers, edge cases |

## Seeds

| Family | Cells | Seed |
| --- | ---: | --- |
| `repro_e1b` | 864 | original: entropy 20260920, spawn key (1, cell), cells 0 to 863 of `code/configs/amend2_2026-09-18.yaml` experiment E1 |
| `repro_main` | 2 | original: entropy 20260918, spawn key (1, 703) and (1, 1855) of `code/configs/base.yaml` experiment E1 |
| `zero_under` | 36 | new: entropy 20261005, spawn key (72, 1, cell) |
| `decomp` | 72 | new: entropy 20261005, spawn key (72, 4, 3 x scenario + K index); the overestimation-on and overestimation-off cells of a scenario and K share the seed |
| `cond` | 12 | new: entropy 20261005, spawn key (72, 5, cell) |

Chunk size 20,000 patients, one child SeedSequence per chunk, as in the published runs. Each cell row carries `seed_entropy` and `seed_spawn_key`.

## Simulation outputs (written by the batch job)

| File | Content |
| --- | --- |
| `sim_metrics.parquet` | Long format: one row per cell, arm, rule, axis and metric (value, mcse, mcse_method, count, n). Metrics prefixed `acc_` are the library accumulators (`ErrAcc`, `MeanAcc`, `PropAcc`) used for the exact comparison with stored results |
| `sim_jack.parquet` | Leave-one-group-out replicates (50 groups of 2,000 patients) of bias, SD, RMSE, MAE and the 95th percentile of absolute error; used for paired Monte Carlo standard errors of differences between rules |
| `sim_decomp.parquet` | Counterfactual decomposition terms per cell, axis and contrast |
| `sim_cond.parquet` | Conditional error summaries by true-span bin |
| `provenance.txt` | Date, host, job id, versions, git commit, sha256 of code |

## Tables (written by `code/21_osrev_views_analyse.py`)

All csv files are tidy, unrounded, and give counts with every proportion. Error is measured minus true span T1 on the stated axis; AP is the true maximal span. `window` 0 means no window. `scenario` is written anchor / long-axis mean underestimation in per cent. Columns ending `_mcse` are Monte Carlo standard errors.

| File | Request | Content |
| --- | --- | --- |
| `views_repro_check.csv` | requirement 2 | Stored versus re-simulated bias, RMSE, SD, relative bias, limited-sampling proportion and selection inflation: number of values compared and number bit-identical |
| `views_a_12scenarios.csv` | (a) | 12 view-accuracy scenarios (K 3, beat CV 15%, overestimation on, no window, median span 10 mm) x 4 rules x 2 axes: bias, SD, RMSE, MAE, signed-error percentiles (2.5, 5, 25, 50, 75, 95, 97.5), absolute-error percentiles (50, 90, 95), proportion and count with absolute error above 1, 2 and 3 mm, ratio of the 95th percentile of absolute error to RMSE |
| `views_a_all_cells_wide.csv` | (a) | The same columns for all 902 cells of `repro_e1b`, `repro_main` and `zero_under` |
| `views_a_added_scenarios.csv` | (a) | Rows of the base family and of each one-factor variant (overestimation off, beat CV 30% and 5%, two and four views, window, median span 13 mm) plus the zero-underestimation cells, labelled in `variant` |
| `views_a_variant_ranges.csv` | (a) | Minimum and maximum over the 12 scenarios of each metric, per variant, rule and axis, with the scenario attaining each |
| `views_b_worstcase.csv` | (b) | Per condition (72), axis and rule: worst-case absolute bias, RMSE, MAE, 95th percentile of absolute error and RMSE regret over the 12 scenarios, and scenario means |
| `views_b_decision.csv` | (b) | Per condition, axis and objective: preferred rule, runner-up, gap, gap MCSE and method, `clear` flag, ranking of the four rules |
| `views_b_decision_13scenarios.csv` | (b) | The same after adding the zero-underestimation scenario as a 13th scenario (36 conditions at median span 10 mm) |
| `views_b_tally.csv` | (b) | Number of conditions won by each rule per objective: all 72, by beat CV, the base condition, and the 13-scenario set |
| `views_c_table1_s2.csv` | (c) | Bias, RMSE and MAE with 95% Monte Carlo interval limits per rule in each setting separately (pre-specified base case; pre-specified low scenario; amendment-2 cells at 4.8/20, 2.4/8 and 4.8/8 with and without window); flags mark the rule and setting giving 1.70 and 2.34 mm |
| `views_d_decomposition.csv` | (d) | Per scenario (12), K (2, 3, 4), overestimation (on, off), axis and contrast (A3-A1, A4-A1): excess in the four arms, interaction, first-in and last-in contributions and shares, counts of patients whose largest view was not the anchor, was the view with the largest expected span, or carried overestimation |
| `views_d_check_published.csv` | (d) | Full-arm excess (new seed) against the stored selection inflation (independent seed): difference and z |
| `views_e_conditional_quantiles.csv` | (e) | Per cell of family `cond` (view accuracy 4.8/20, 2.4/8, 20/40 x window on, off x sinus, AF), rule, axis and true-span bin: n, bias, SD, RMSE, MAE, 2.5th, 50th and 97.5th percentiles of error, 95th percentile of absolute error, its ratio to RMSE, half-width of the central 95% error interval |
| `views_e_ratio_all_cells.csv` | (e) | RMSE, 95th percentile of absolute error and their ratio for all 902 cells |
| `views_e_ratio_summary.csv` | (e) | Minimum, median and maximum of the ratio over the 864 grid cells per rule and axis |

## Monte Carlo standard errors

Bias: SD / sqrt(n). RMSE: delta method, SD(e^2) / sqrt(n) / (2 RMSE), as in `metrics.ErrAcc`. MAE: SD(|e|) / sqrt(n). Proportions: binomial. SD of error, every percentile and the ratio of the 95th percentile of absolute error to RMSE: delete-a-group jackknife with 50 consecutive groups of 2,000 patients. Paired contrasts (decomposition terms, gaps between rules within one cell) use per-patient differences or the jackknife of the difference, because all rules and all counterfactual arms within a cell share random numbers. Gaps between maxima attained in different cells combine the two standard errors as independent.
