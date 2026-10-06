# Amendment 3, package A3-4: triggers and review (run 2026-10-05)

Project: DUO-Max simulation methods paper, `the project root`.
Post hoc sensitivity analyses requested by the senior author on 2026-10-05 (comments C49, C186, C188 and the
AP/SL part of C324; `notes/protocol-amendment-3-2026-10-05.md`, item A3-4). Nothing here replaces a pre-specified
analysis. Findings and suggested manuscript text: `notes/scratch/2026-10-05-findings-triggers.md`.

## How to regenerate

```bash
sbatch --wait code/slurm/osrev_triggers.sh          # one cpu node, 85 s; job 5301795 on 2026-10-05
# or, tables only, on the login node (about 30 s):
/project/home/p201509/envs/duomax-sim/bin/python code/21_osrev_triggers_tables.py
```

The Slurm script writes `provenance.txt` (date, host, job id, Python and package versions, git commit, sha256 of
every code and configuration file used, pytest summary), runs `code/20_osrev_triggers_run.py` (simulation, writes
`raw/`), then `code/21_osrev_triggers_tables.py` (derived tables, this directory), then
`tests/test_osrev_triggers.py` and `tests/test_regression.py`.

Code: `code/lib/duomaxsim/osrev_triggers.py` (new functions only; the pre-specified library is unchanged),
`code/configs/osrev_triggers.yaml` (conditions, grids and seeds).

## Cohorts and seeds

| Block | Content | Patients | Seed (numpy `SeedSequence`) |
|---|---|---|---|
| A, cohort `published` | E4 base cell 193 of `base.yaml` (four views), exact rerun of the published cohort | 100,000 | entropy 20260918, spawn key (4, 193) |
| A, cohorts `new_rep0` to `new_rep9` | the same condition, ten independent cohorts; `new_pooled` is their sum | 10 x 100,000 | entropy 20261005, spawn key (74, 1, rep) |
| A, cluster bootstrap | 10,000 resamples of patients per table | none new | entropy 20261005, spawn key (74, 9) |
| D, `run = new` | reviewer grid, 4 view-accuracy scenarios x 3 overestimation levels (three views) | 12 x 100,000 | entropy 20261005, spawn key (74, 4), the same for every condition (common random numbers) |
| D, `run = reproduce` | published cells E1 703 and 1855, E3 29 (`base.yaml`), E1 36 and 828 (`amend2_2026-09-18.yaml`) | 5 x 100,000 | library cell seeds (20260918 or 20260920, (experiment, cell)) |
| E | 23 joint AP/SL scenarios (four views), all on the same patients and measurement errors | 23 x 100,000 | latent state: entropy 20261005, spawn key (74, 5, 0); measurement: (74, 5, 1) |

Each cohort is drawn in five chunks of 20,000 patients, one child seed per chunk (`SeedSequence.spawn`), as in the
library.

## Files in `raw/` (written by `code/20_osrev_triggers_run.py`; additive counts, no rounding)

| File | Content |
|---|---|
| `A_pair_patterns.csv` | per cohort, axis, trigger threshold (2 to 6 mm) and error definition: number of patients with each combination of true-positive, false-positive, false-negative and true-negative pairs among their three anchor-other pairs. Every pair-level table and its patient-clustered standard error follows from this file |
| `A_patient_2x2.csv` | patient-level 2 by 2 counts (any trigger in the axis, or in either axis, by truth); `tp_matched` = patients in whom the trigger fired in a pair that itself carries the error |
| `A_pair_partition.csv` | pairs by mutually exclusive class of the two views' signed systematic errors (`error_kind` total or net), with trigger counts |
| `A_exam_offsets.csv` | patients in whom every view of an axis is wrong in the same direction by more than 2 mm |
| `A_axis_diff.csv` | patients with a true or measured absolute AP minus SL difference of at least each threshold |
| `A_instrument.csv` | patients whose instrument factor alone shifts the span by more than 2 mm |
| `D_review_metrics.csv` | bias, RMSE, MAE, sensitivity and specificity at 7, 10 and 13 mm for the anchor-view mean (A1), mean across views (A2), largest view mean (A3) and the reviewed rule (A4) at each pair of reviewer probabilities |
| `D_review_paired.csv` | paired differences (same patients) of each reviewed-rule variant against A3 and against the published variant (0.8, 0.1) |
| `D_review_process.csv` | views reviewed, excluded, and excluded although valid |
| `E_axis_counts.csv`, `E_axis_summary.csv` | per AP/SL scenario: counts of true and measured differences, SL-axis and AP-axis trigger counts, ratio moments, correlations, and AP-axis checksums (identical in every scenario) |

## Derived tables (written by `code/21_osrev_triggers_tables.py`)

| File | Content | Source |
|---|---|---|
| `triggers_pair_2x2_counts.csv` | the pair-level 2 by 2 table (trigger fired or not, by error present or not) for cohorts `published`, `new_rep0`, `new_pooled`; every axis, threshold and error definition | `A_pair_patterns.csv` |
| `triggers_pair_rates.csv` | prevalence, sensitivity, false-positive rate, specificity, PPV and NPV with numerator and denominator; SE treating pairs as independent, the manuscript's earlier 'conservative' SE (x sqrt 3), the patient-clustered SE (linearization), the cluster-bootstrap SE and percentile interval, and the SD across the ten new cohorts | `A_pair_patterns.csv` |
| `triggers_pair_consistency.csv` | AP axis, published cohort: PPV from counts, from the unrounded rates, and from the rounded rates printed in manuscript v05, with the range compatible with rounding | `triggers_pair_2x2_counts.csv` |
| `triggers_interval_methods.csv` | four interval methods side by side for the published error definition at 3 and 5 mm | `triggers_pair_rates.csv` |
| `triggers_replicates.csv` | published error definition, 3 and 5 mm: each of the eleven cohorts separately | `A_pair_patterns.csv` |
| `triggers_patient_2x2.csv` | patient-level counts, sensitivity, false-positive rate, PPV, NPV (binomial MCSE; the unit is the patient) | `A_patient_2x2.csv` |
| `triggers_error_definitions.csv` | one row per error definition, threshold, axis and cohort: pair-level and patient-level prevalence, sensitivity, false-positive rate and PPV | the two tables above |
| `triggers_pair_partition.csv` | proportion of pairs and of patients in each error class, trigger rate within the class; `*_replicate_se` is the SD of the ten cohort estimates divided by sqrt 10 | `A_pair_partition.csv` |
| `triggers_exam_offsets.csv`, `triggers_instrument_shift.csv` | whole-examination offsets; instrument factor | `A_exam_offsets.csv`, `A_instrument.csv` |
| `triggers_reproduction_E4_cell193.csv` | stored values of `results/2026-09-18_full/E4_*.parquet`, cell 193, against the rerun (all identical) | parquet, `triggers_pair_rates.csv`, `triggers_patient_2x2.csv`, `A_axis_diff.csv` |
| `review_metrics_long.csv`, `review_paired_long.csv` | copies of the raw long tables | `D_review_*.csv` |
| `review_grid_wide.csv` | one row per run, scenario, overestimation level, axis and rule: every metric with its MCSE, the counts behind sensitivity and specificity, and paired differences against the largest view mean and against the published reviewer | `D_review_metrics.csv`, `D_review_paired.csv` |
| `review_dependence_summary.csv` | per scenario, overestimation level and axis: published advantage of review, its range over the grid, and the number of grid points at which review lowers or raises RMSE (95% Monte Carlo interval of the paired difference excluding zero) | `review_grid_wide.csv` |
| `review_process.csv` | review workload and exclusions with proportions | `D_review_process.csv` |
| `review_reproduction.csv` | stored E1 and E3 values against the reruns (largest absolute difference 4.4e-16) | parquet, `review_grid_wide.csv` |
| `axes_long.csv`, `axes_scenarios.csv` | per AP/SL scenario: true and measured between-axis differences of at least 3 and 5 mm (counts, proportion, MCSE, quadrature value for the truth), SL-axis and AP-axis trigger results | `E_axis_*.csv` |

## Definitions used in the column names

- Trigger fired: other view mean minus anchor-view mean of the same axis at least tau mm (one-sided).
- `D0_published`: g S o_other > 2 mm or g S u_anchor > 2 mm (manuscript definition). `D0a`, `D0b`: its two components.
- `D1_plus_other_under`: D0 or g S u_other > 2 mm.
- `D2_net_either_view`: |g (mu - S)| > 2 mm in the anchor or the other view. `D3_total_either_view`: |g mu - S| > 2 mm in
  either view (includes the instrument factor g).
- `D4_expected_excess`: g (mu_other - mu_anchor) > 2 mm. `X_expected_excess_ge_tau`: the same quantity at least tau.
- `D5_shared_total` (`D5n_shared_net`): both views wrong by more than 2 mm in the same direction.
  `H_shared_total_agree` (`Hn_...`): D5 (D5n) and |g (mu_other - mu_anchor)| < tau.
- Reviewer probabilities: `s_det` = P(exclude | reviewed view truly overestimated), `f_rej` = P(exclude | reviewed
  view not overestimated).

## Checks

- 28 tests passed (`tests/test_osrev_triggers.py`, 27; `tests/test_regression.py`, 1); summary appended to `provenance.txt`.
- Rerun of the published cohorts is bit-identical to the stored results for the trigger metrics (17 of 17 values) and
  within 4.4e-16 for the E1 and E3 metrics (92 values).
- The reviewed rule with probabilities (0, 0) equals the largest view mean exactly in every chunk (asserted at run time).
- AP-axis checksums are identical across the 23 AP/SL scenarios (asserted at run time).
- Monte Carlo proportions of the true between-axis difference agree with quadrature: over 110 comparisons with a
  non-zero Monte Carlo standard error (22 scenarios, thresholds 2 to 6 mm, all on the same patients) the largest
  standardized difference is 2.05 and 2 exceed 1.96.
