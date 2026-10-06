# duomaxsim: simulation code for the tricuspid jet beat and view aggregation study

Implements the study protocol of 18 September 2026 (not part of the repository; summarized in the manuscript supplement): the generative model, estimators A1 to A7, beat rules and cross-view triggers, and the six planned experiments E1 to E6. The same library ran the first amendment (second level of long-axis underestimation), the post hoc exploratory sensitivity analyses and the second amendment. The third amendment of 5 October 2026 (post hoc sensitivity analyses) is implemented in separate modules, described under 'Third-amendment packages' below.

## Layout

| Path | Content |
|---|---|
| `lib/duomaxsim/config.py` | YAML loading, grid-cell enumeration, per-cell `SeedSequence` |
| `lib/duomaxsim/model.py` | case mix, view errors (u, o, rho), instrument factor g, beats with RR dependence (sinus/AF), available beats, reader measurement |
| `lib/duomaxsim/rules.py` | N-beat rule with +-W acceptance window and "limited sampling" flag; cross-view warn/adjudication triggers; rule A4 with reader detection (s_det, f_rej) |
| `lib/duomaxsim/estimators.py` | A1 anchor mean, A2 mean of view means, A3 max, A5 median, A6 index beat, A7 oracle-corrected mean |
| `lib/duomaxsim/metrics.py` | streaming bias/RMSE/SD/relative-bias accumulators with Monte Carlo SE; proportions; ICC(A,1) with McGraw-Wong CI; Bland-Altman; Clark (1961) max-of-2 moments; E[max of K normals]; numerical MVN E[max] |
| `lib/duomaxsim/experiments.py` | `run_E1` ... `run_E6`, `run_cells` (long-format DataFrame) |
| `configs/base.yaml` | every parameter with `value`, `grid` (where the protocol has one), `unit`, `source` (all `provisional`) and `note`, plus the per-experiment `vary` lists |
| `01_run_experiment.py` | command-line entry point |

## Running

```bash
PY=/project/home/p201509/envs/duomax-sim/bin/python
$PY code/01_run_experiment.py --experiment E1 --config code/configs/base.yaml --list-cells
$PY code/01_run_experiment.py --experiment E1 --config code/configs/base.yaml --cells 0:96 --out results/2026-09-18_full
$PY -m pytest tests -q -p no:cacheprovider
```

- `--cells i:j` is half-open. Each invocation writes one parquet file (`E1_cells_00000-00096.parquet`) plus a `.manifest.json` holding the command, the full config and the package versions. Cells are never written one per file, because of the inode quota.
- `--n-rep` overrides the number of replicates (patients for E1 to E4, simulated studies for E5 and E6). Use it for pilots only; the reported runs use the config values (10^5 and 2,000).
- A cell's results do not depend on how the cells are split across invocations (tested). They do depend on `meta.chunk_size`, so keep that value fixed.

## Output schema (long format)

Each row holds one metric for one cell × estimator/variant × axis × estimand. The columns are:

- `experiment`, `cell`, `p_*` (the cell's grid values);
- variant columns where relevant: `s_det`, `f_rej`, `t_warn`, `t_adj`, `cutoff_mm`, `subset`, `ai`, `sigma_ai_mm`, `ref`, `design`, `n_double`, `overlap_frac`, `sentinel_n`, `anchor_alpha`;
- `estimator`, `axis`, `estimand`, `metric`, `value`, `mcse`, `n`;
- `seed_entropy`, `seed_spawn_key`, `chunk_size`, `cpu_s`, `wall_s`.

Estimands are `T1` (true maximal span S), `T2` (anchor-view expected span mu_anchor) and, in E2 only, `Tv` (g × mu_anchor, the view expectation under the study's instrument settings).

## Tests (`tests/`)

- `test_known_answers.py` covers the bias of max of K view means (K = 2, 3, 4) against E[max of K N(0,1)]·sigma/sqrt(N); Clark (1961) moments; numerical MVN integration for k = 3 and 4; unbiasedness of the mean estimators; the lognormal median-centring bias; and the N = 3 window probability against 2D numerical integration for lognormal and additive-Gaussian beats.
- `test_edge_cases.py` covers K = 1, N = 1, zero noise giving the truth exactly, A7 under view error only, retrospective short clips, reductions of A4 to limiting cases, the A4 detection rates, seed invariance to cell chunking, ICC CI coverage, and degenerate agreement.
- `test_regression.py` covers the fixed-seed fixture `tests/fixtures/regression_v01.csv`, built by `tests/make_fixture.py`. Regenerate it only for an intended model change, and log the reason in `notes/lab-notebook.md`.
- `test_amend2.py` covers the second-amendment extensions (shared-error AI models, Brown-Forsythe test, grid definitions).
- `test_osrev_<package>.py` (six files, 153 tests) cover the third-amendment packages: exact reproduction of stored results with the new options off, known-answer cases and edge cases.
- Full suite: 211 tests. `sbatch --wait code/slurm/pytest_full.sh` writes `tests/logs/pytest-<date>.log` with the commit, the working-tree status, checksums and package versions.
- `tests/independent/` holds second implementations of the core model, the beat window and the second-amendment analyses, written by a separate AI coding agent; `code/osrev_verification.py` reruns them against the library. This is software verification, not independent scientific review.

## Third-amendment packages (post hoc sensitivity analyses, 5 October 2026)

The six packages extend the library without modifying it. Each first reproduces stored results of the earlier runs from their original seeds and then adds its analyses under master seed 20261005. Results and a file guide per package are in `results/2026-10-05_osrev/` (index in its `README.md`).

| Package | Library module | Runner | Analysis | Config and batch script |
|---|---|---|---|---|
| Beat rules | `lib/duomaxsim/osrev_beats.py` | `20_osrev_beats_run.py` | `21_osrev_beats_tables.py` | `configs/osrev_beats.yaml`, `slurm/osrev_beats.sh` |
| View rules | `lib/duomaxsim/osrev_views.py` | `20_osrev_views_run.py` | `21_osrev_views_analyse.py` | `configs/osrev_views.yaml`, `slurm/osrev_views.sh` |
| Cut-offs | `lib/duomaxsim/osrev_thresholds.py` | `20_osrev_thresholds_run.py` (`published` on the login node, `replicates` in the batch job) | `21_osrev_thresholds_analyse.py` | `configs/osrev_thresholds.yaml`, `slurm/osrev_thresholds.sh` |
| Triggers and review | `lib/duomaxsim/osrev_triggers.py` | `20_osrev_triggers_run.py` | `21_osrev_triggers_tables.py`; `23_osrev_fig4_clustered_se.py` for Figure 4 | `configs/osrev_triggers.yaml`, `slurm/osrev_triggers.sh`, `slurm/osrev_fig4_clustered.sh` |
| AI evaluation and reference sets | `lib/duomaxsim/osrev_ai_reference.py` | `20_osrev_ai_reference_run.py` | `21_osrev_ai_reference_analyse.py`, `22_osrev_ai_reference_checks.py` | `configs/osrev_ai_reference.yaml`, `slurm/osrev_ai_reference.sh` |
| Calibration of long-axis underestimation | `lib/duomaxsim/osrev_calibration.py` | `20_osrev_calibration_run.py` (stages A to D) | `21_osrev_calibration_analyse.py` (`--followup` for stages C and D) | `configs/osrev_calibration.yaml`, `slurm/osrev_calibration.sh` |

`23_make_tables_v06.py` builds the tables of manuscript v06 from the result files; it writes into `drafts/`, which is not part of the repository. `figures/figS6_true_span_distribution.py`, `figS7_beats_matched.py` and `figS8_error_distribution.py` draw Supplementary Figures S6 to S8.

## Tools (`tools/`)

- `number_refs.py`: numbers bracketed at-key citations in order of first appearance from verified reference files.
- `wordcount_ehjci.py`: itemized word count of a manuscript draft against the journal limits (main text, references, legends, tables; abstract).
- `slurm_job_table.py`: markdown table of the project's Slurm jobs from the accounting records (source of the job table in `environment.md`).

## Known issues

These are documented and left unchanged so that the scripts of releases 0.3.1 and 0.4.0 keep reproducing their stored outputs. None changes a reported number unless stated.

1. `06_analyse_sensitivity.py`, line 50: the comment "conservative: ignores positive CRN covariance" is wrong. The two conditions are simulated with independent random-number streams, so the root sum of squares of the two Monte Carlo standard errors is the exact standard error of the difference.
2. `04_analyse_E4.py`: the column `ppv_mcse_conservative` (binomial standard error multiplied by the square root of 3) is not a valid cluster standard error and is about 1.7 times too wide for the base case. Patient-clustered standard errors are in `results/2026-10-05_osrev/triggers/` and `results/2026-10-05_osrev/fig4_clustered/`, and Figure 4 uses them from release 0.5.0.
3. The Monte Carlo standard error of the SD of error stored by the runs of 18 September 2026 assumes normal errors and is too small (median ratio of a jackknife estimate to the stored value 1.49 in the audit of the view-rule package). Do not quote it; use the jackknife values in `results/2026-10-05_osrev/views/`.
4. `08_analyse_amend2_e1e3.py`: the `*_clear` flags of the worst-case comparison (printed as 'Separated' in the supplementary table of release 0.4.0) use the standard error for independently simulated cells. The comparison is between rules, and rules share simulated patients within a cell, so that standard error is larger than needed and the flag understates how many choices are clear; paired jackknife standard errors are in `results/2026-10-05_osrev/views/views_b_decision.csv`.
5. `results/2026-09-18_full/provenance.txt` and `results/2026-09-18_sens/provenance.txt` record sha256 values of `experiments.py` and `metrics.py` as they were before the second amendment extended both files. The earlier versions are in neither the repository nor its history. That the planned outputs are unaffected rests on the frozen regression fixture (`tests/test_regression.py`), an identity check of seven conditions recorded in `notes/lab-notebook.md`, and the exact regeneration of stored conditions by the third-amendment packages (for example `results/2026-10-05_osrev/views/views_repro_check.csv`).
6. `results/2026-09-18_pilot` has run manifests but no `provenance.txt`.
7. `configs/base.yaml` and `notes/parameter-table-2026-09-18.md` label the source of beat-to-beat variation (a proximal isovelocity surface area measurement) as 'close'; by the project's own construct definitions it is an indirect source. The manuscript supplement carries the corrected status of every parameter.
8. Docstrings and configuration comments of the third-amendment files call the earlier library 'pre-specified' and cite internal review comment numbers (C followed by three digits). Read 'pre-specified' there as 'unchanged since the runs of 18 September 2026'; only the six experiments of the initial protocol were planned before any simulation.
9. The second implementation of the core model (`tests/independent/indep_check_e1.py`) used the provisional parameter values of the first protocol draft and covers the anchor-view mean, the largest view mean and the reviewed rule only; no second implementation exists for the other estimators.

