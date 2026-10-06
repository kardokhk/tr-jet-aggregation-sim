# results/2026-10-05_osrev/ai_reference

Protocol amendment 3, package A3-5 (AI validation and reference sets). Post hoc sensitivity analyses requested by the senior author on 2026-10-05 (comments C189, C209, C321, C399). Secondary to the pre-specified experiments E5 and E6 and to amendment 2 (E5b, E6b). All results are conditional on the generative model.

## How to regenerate

```
sbatch --wait code/slurm/osrev_ai_reference.sh      # runs scripts 20 and 21 on one cpu node (about 3 min)
/project/home/p201509/envs/duomax-sim/bin/python code/22_osrev_ai_reference_checks.py   # login node, about 1 CPU-min
/project/home/p201509/envs/duomax-sim/bin/python -m pytest tests/test_osrev_ai_reference.py tests/test_regression.py
```

- Library extension: `code/lib/duomaxsim/osrev_ai_reference.py` (no existing library file was changed).
- Configuration: `code/configs/osrev_ai_reference.yaml`. Model parameters come unchanged from `code/configs/amend2_2026-09-18.yaml`.
- Seeds: `numpy.random.SeedSequence(20261005, spawn_key=(75, stream, ...))`; streams 1 to 7 are listed in the configuration. The reproduction part uses the stored seeds of the published runs.
- Batch job 5301941 (2026-10-05, node mel0038, 2 min 34 s). `provenance.txt` holds date, host, Python and package versions, git commit and sha256 of the code.
- Findings and methods as run: `notes/scratch/2026-10-05-findings-ai_reference.md`.

## Files

| File | Produced by | Content |
|---|---|---|
| `raw/*.parquet`, `raw/run_cost.csv` | `code/20_osrev_ai_reference_run.py` | Per-study, per-pair and per-cell raw output; CPU time per part |
| `reproduction_check.csv` | script 20 (part repro), copied by script 21 | Stored published values against values recomputed with the extension module and the stored seeds (1,375 values, maximum absolute difference 0) |
| `published_vs_new_seed.csv` | script 21 | Published values against the same quantities on the new seeds, with z scores |
| `ai_reference_error_components.csv` | scripts 20, 21 | Moments of the reference error and its components by reference design and cell; block-jackknife MCSE (50 blocks of 40 studies) |
| `ai_inherited_calibration.csv` | scripts 20, 21 | Intercept and slope of the inherited-error model per cell |
| `ai_analytic_check.csv` | scripts 20, 21 | Second-moment prediction against simulation for every model and reference: MSE, MAE under normality, limits-of-agreement width |
| `ai_model_summary.csv` | script 21 | Mean MAE, bias and limits-of-agreement width per model and reference, with apparent-minus-true differences (paired) |
| `ai_matching.csv` | scripts 20, 21 | Own-error SD that equates true MAE with the comparator, and the true MAE achieved |
| `ai_ranking_matched.csv` | script 21 | Ranking of each candidate model against the independent-error comparator (own-error SD 2 mm) by reference design: counts and proportions of studies preferring the candidate, paired differences |
| `loa_conditional_vs_marginal.csv` | script 21 | Upper limit of agreement: conditional (within reader pair), between-pair and marginal SD by number of cases; Bland-Altman interval width and coverage for each estimand |
| `loa_nested_targets.csv` | script 21 | Variance components of the inter-reader difference and the marginal target limit |
| `loa_multireader_designs.csv` | script 21 | Designs with 2, 3, 4 or 6 readers at about 400 reads: estimator of the marginal limit and of the between-reader SD |
| `sentinel_size_power.csv` | scripts 20, 21 | Rejection counts and rates for five tests, 54 cells x 4 sentinel sizes x 5 drift levels |
| `sentinel_size_summary.csv`, `sentinel_power_summary.csv`, `sentinel_base_cell.csv` | script 21 | Size and power summaries; the base cell |
| `sentinel_difference_distribution.csv` | scripts 20, 21 | Skewness and excess kurtosis of the manual-minus-draft difference per cell; seeds |
| `check_interreader_difference_shape.csv` | `code/22_osrev_ai_reference_checks.py` | Skewness and excess kurtosis of the inter-reader difference; implied inflation of the SD of a limit |
| `check_noise_independence_cv30.csv` | script 22 | Diagnostic for one departure in the analytic check (cell cv30) |

Column conventions: `*_mcse` is the Monte Carlo standard error of the column without the suffix; `n_*` columns are unrounded counts; lengths in mm, variances in mm². Reference designs: `true` (true maximal span), `single`, `mean2`, `adj_tol1`, `adj_tol2`, `adj_tol3`. Model names: `indep_sd<s>`; `share_lam<lambda>_<eqsd|eqmae>_<ind|crn>`; `inherit_<eqsd|eqmae>_<ind|crn>`, where `eqsd` is equal own-error SD (2 mm), `eqmae` is equal true MAE, `ind` is own error independent of the comparator's and `crn` is the same own-error realization as the comparator (the design of the published runs).
