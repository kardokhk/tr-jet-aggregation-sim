# Amendment 3, package A3-6: calibration against a matched estimand

Project DUO-Max methods paper, `the project root`. Run 2026-10-05.
Post hoc sensitivity analysis requested by the senior author (comments C176 and C175;
`notes/protocol-amendment-3-2026-10-05.md`, item A3-6). Findings and suggested manuscript text:
`notes/scratch/2026-10-05-findings-calibration.md`.

## How to regenerate

```
sbatch --wait code/slurm/osrev_calibration.sh      # stage A, stage B, analysis; about 2.5 min on one cpu node
```

The job runs `code/20_osrev_calibration_run.py` (tasks in parallel, then `--merge A`, `--merge B`)
and `code/21_osrev_calibration_analyse.py`. Library: `code/lib/duomaxsim/osrev_calibration.py`.
Config: `code/configs/osrev_calibration.yaml` (copied here as `config_used.yaml`). Tests:
`tests/test_osrev_calibration.py` (the version of 2026-10-05, checksum in `provenance.txt`, collects 30 tests although the reports of that day say 31; 42 since the follow-up below). Job 5301947 produced every simulation output in this
directory; the analysis script was rerun once on the login node afterwards (deterministic, see
`provenance.txt`). An earlier job, 5301846, was superseded (its csv of re-simulated stored cells was
written with 10 significant digits, too few for a bit-level comparison).

Seeds: numpy `SeedSequence(entropy=20261005, spawn_key=(76, family, index))`. Family 0 fitting
sample, 1 evaluation sample, 2 view rules (one seed for all scenarios: common random numbers),
3 bootstrap target draws (index = replicate 0 to 499, common to all variants), 4 library-model match.
The 12 stored E1b cells are re-simulated with their original seeds (entropy 20260920, spawn key (1, cell)).

## Files

| File | Produced by | Content |
|---|---|---|
| `singh_reproduction.csv` | 21, deterministic | Percentage derivations from Singh et al. 2026 Tables 2 and 3 (reported or derived) |
| `orifice_fit.csv` | 20 `geom:*` | Orifice distribution: targets, fitted moments (quadrature), evaluation-sample means, parameters |
| `geom_fit.csv` | 20 `geom:*` | Per variant and view: fitted plane angle and offset scale (or scale factor), fit residuals, model and published mean widths, count of patients at the offset cap |
| `geom_checks.csv` | 20 `geom:*` | Model against published quantities that were not fitted (SDs, Bland-Altman SDs, 2D sphericity index, Pearson r), jackknife MCSE |
| `geom_underestimation.csv` | 20 `geom:*` | Per variant, view, plane, reference and component: mean, SD and quantiles of the per-patient fractional underestimation, ratio-of-means version, MCSE, unrounded counts |
| `calibration_summary.csv` | 21 | Selected rows of the previous table with the target-sampling intervals from the bootstrap |
| `calibration_variant_range.csv` | 21 | Smallest and largest mean across variants whose fit reproduced the published means |
| `geom_bootstrap.csv` | 20 `boot:*` | 500 replicates x 4 variants x 3 views: refit after perturbing the eight published means by their standard errors |
| `geom_bootstrap_summary.csv` | 21 | Percentiles over replicates (all replicates, and converged fits only) |
| `identifiability_ellipse.csv` | 21, deterministic | Mean published ellipse: longest chord and projected extent along an axis, cross-plane maximum and biplane average by angle |
| `libmatch.csv` | 20 `lib` | Published estimand evaluated in the library model: grid of long-axis scales and scales solved for the published percentages |
| `matched_values.json` | 20 `--merge A` | Values passed from stage A to the view-rule scenarios |
| `viewrules_metrics.csv` | 20 `rules:*` | Bias, RMSE, MAE and SD of error of the four view rules, both axes, 56 scenarios, with MCSE and paired differences from the base scenario (anchor scale 0.06, long-axis scale 0.25) |
| `viewrules_table_ap.csv` | 21 | The same, AP axis, one row per scenario |
| `viewrules_ranges.csv` | 21 | Range across the four anchor levels per scenario, rule, axis and metric, next to the range of the 12 published scenarios at the new seed |
| `repro_e1b_resimulated.csv` | 20 `repro:*` | The 12 stored E1b cells (three views, beat CV 15%, overestimation on, no window, median span 10 mm) re-simulated with their original seeds |
| `repro_published_check.csv` | 21 | Comparison with `results/2026-09-18_amend2`: 384 of 384 values bit-identical |
| `provenance.txt`, `config_used.yaml`, `log_stage_*.txt` | job | Date, host, versions, git commit, sha256 of code, per-task CPU seconds |

Rule labels: A1 anchor-view mean, A2 mean across views, A3 largest view mean, A4 largest view mean
after review. Fractions are not percentages. `n` and `count_*` columns are unrounded counts.

## Follow-up of 2026-10-06 (stages C and D)

Reason: the independent audit (findings note, items 5.2 to 5.4) found that the scenario "constant
rotation plus chord-distributed offset" had about half the between-patient spread of the calibrated
geometry, that the lowest calibrated value (about 6%) had not been run, and that the calibration
depends on how the misprinted confidence interval of the 3D maximal diameter is read. Post hoc, third
amendment. Nothing above was regenerated; sha256 of all stage A and B outputs is unchanged.

```
sbatch --wait --export=ALL,STAGES="C D" code/slurm/osrev_calibration.sh   # about 1 min on one cpu node
```

Job 5306863 (mel0398, 00:00:55) produced every `followup_*` file; job 5306860 is superseded (see
`provenance.txt`). `code/21_osrev_calibration_analyse.py --followup` was rerun once on the login node
afterwards (deterministic). Tests: `tests/test_osrev_calibration.py`, 42 tests (30 before, 12 added).

New option (default off): key `u_geom` in the parameter dict of `draw_latent_axis` and
`simulate_rules` (`geometric_underestimation` in the library). Per patient: an ellipticity ratio from
the calibrated orifice distribution, shared by the views; per long-axis view: a plane angle and a
relative offset. Reading 1 (rotation counted as error) and reading 2 (off-centre placement only).
Without the key every array and every stored number is bit-identical to the version of 2026-10-05.

Seeds: replicate 0 uses the stage B stream (76, 2, 0) for everything the library draws, so the new
scenarios are paired with the 56 stored ones, and (76, 5, 0) for the additional geometric normals.
Replicate 1 repeats every follow-up scenario on new streams (76, 5, 1) and (76, 5, 2). Stage C uses
the stage A fitting and evaluation samples (76, 0, 0) and (76, 1, 0).

Scenario variants (`followup_matched_values.json`): ind15, mid15, ind00 (SD of the 3D maximal
diameter 5.73 mm, the stage A reading) and ind15_sd707, mid15_sd707, ind00_sd707 (7.07 mm). Shapes
per variant: r1_halfnormal, r1_chord, r2_halfnormal, r2_chord (stage B definitions), r1_patient,
r2_patient (patient-level draw, primary design), r1_patient_indoff, r2_patient_indoff (offsets
independent between views), r1_patient_spanq (ellipticity linked to the true AP span). Four anchor
levels each; 100,000 patients.

| File | Produced by | Content |
|---|---|---|
| `calibration_table_s25.csv` | 21 `--followup` | Tidy source of Supplementary Table S25: A published comparison; B corresponding quantity in the simulation model; C elliptical-orifice calibration; C2 sensitivity to the confidence-interval reading; D view-rule ranges under each error shape. `display` is the formatted cell, `core` marks the 53 rows suggested for print, `source` names file and column |
| `followup_sd_orifice.csv`, `followup_sd_fit.csv`, `followup_sd_checks.csv`, `followup_sd_underestimation.csv` | 20 `sdsens:*` | Stage A tables for ind15, mid15, ind00 with the SD of the 3D maximal diameter from the lower half-width (5.73 mm), half the full width (7.07 mm) and the upper half-width (8.41 mm) |
| `followup_sd_sensitivity.csv` | 21 `--followup` | Selected rows of the previous tables with the implied correlation of maximal and minimal diameter and the implied mean sphericity index |
| `followup_matched_values.json` | 20 `--merge C` | Values passed from stage C to stage D |
| `followup_viewrules_metrics.csv` | 20 `frules:*` | Bias, RMSE, MAE and SD of error, four rules, both axes, 216 scenarios x 2 replicates, with MCSE, paired difference from the base scenario, and realized mean, SD and between-view correlation of the long-axis underestimation fraction |
| `followup_viewrules_table_ap.csv` | 21 `--followup` | The same, AP axis, one row per scenario and replicate |
| `followup_viewrules_ranges.csv` | 21 `--followup` | Range across scenarios per shape group, rule, axis and metric, next to the stored 12-scenario range; `bracketed_by_stored12`, and by how much a range end lies outside |
| `followup_dispersion.csv` | 21 `--followup` | Realized mean, SD and between-view correlation of the long-axis fraction per shape against the calibrated mean and SD |
| `followup_checks.csv` | 21 `--followup` | Internal checks (all passed): 1,024 of 1,024 stage B values re-simulated identically; reading 2 patient-level draw identical to the chord-distributed offset (1,536 of 1,536); stage C at 5.73 mm identical to stage A (108 of 108 rows); anchor-view mean identical across scenarios; realized mean and SD within 0.003 of the calibration |
| `config_used_followup.yaml`, `log_stage_C.txt`, `log_stage_D.txt` | job | Config copy; per-task CPU seconds (both jobs of 2026-10-06) |
