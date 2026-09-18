# Lab notebook: DUO-Max simulation

## 2026-09-18: simulation package built (duomaxsim 0.1.0)

**Built.** The protocol is in `notes/protocol-2026-09-18-v01.md`. The package is `code/lib/duomaxsim/`, with modules config, model, rules, estimators, metrics and experiments. The CLI is `code/01_run_experiment.py` and the config is `code/configs/base.yaml`. Every parameter in the config carries `source: provisional`; the protocol grids are included, and in-cell variants are evaluated on common random numbers. Tests are in `tests/`. The environment is `/project/home/p201509/envs/duomax-sim` (see `environment.md`).

**Grid sizes.** E1 has 1,152 cells (K × beat CV × view_over × rho × N). E2 has 1,440 (rhythm/RR-CV × beat CV × W × N × data mode, anchor view only). E3 has 48 (K × beat CV × view_over). E4 has 288 (view_errors × r_mean × beat CV × N, K = 4). E5 has 12 (beat CV × view_over). E6 has 54 (beat CV × sigma_rb × AI-draft SD).

**Tests.** 41 of 41 passed (`pytest tests`, 58 s), after two fixes:
- The first run had 3 failures. Two were errors in the tests themselves (a wrong expected beat count, and a duplicate kwarg).
- The third was a real defect. The window check `lo >= (1-W)·mean` rejects a single beat whose measured value is at or below 0. Negative readings occur because the caliper error is Gaussian: at base, 0.05% of AP and 0.50% of SL beats are at or below 0.
- Fix: the window check is now sign-robust (`|x - mean| <= W·|mean|`), and I added an optional parameter `measure_floor_mm` (default null, protocol-literal).
- The regression fixture was regenerated after the fix.

**Pilot timings.** These ran on the login node (login02) and are CPU seconds per cell, taken from `cpu_s` in `results/2026-09-18_pilot/`. Every invocation took under 11 s of CPU, plus 12 to 34 s of wall time to import from Lustre.

| Exp | Pilot n | Mean CPU/cell | Max CPU/cell | Scale to protocol n | Cells | Est. full CPU (mean to max) |
|---|---|---|---|---|---|---|
| E1 | 10^4 patients | 0.28 s | 0.85 s (K = 4, N = 13) | ×10 | 1,152 | 0.9 to 2.7 h |
| E2 | 10^4 | 0.04 s | 0.12 s | ×10 | 1,440 | 0.2 to 0.5 h |
| E3 | 10^4 | 0.15 s | 0.25 s | ×10 | 48 | < 0.05 h |
| E4 | 10^4 | 0.37 s | 0.84 s | ×10 | 288 | 0.3 to 0.7 h |
| E5 | 100 studies | 0.93 s | 0.94 s | ×20 | 12 | 0.06 h |
| E6 | 100 studies | 1.21 s | 1.22 s | ×20 | 54 | 0.36 h |
| Total | | | | | 2,994 | about 1.9 to 4.3 CPU-hours (single core) |

The pilot sampled the cheapest and most expensive corners of each grid, so the mean overstates the grid average and the max is an upper bound. Memory per process is below 1 GB at chunk size 20,000.

**Decisions where the protocol is ambiguous (assumptions).**
1. **T2.** T2 = mu_i,anchor, taken literally from the protocol symbol: reference settings, excluding g, reader bias and the lognormal mean inflation. E2 also reports Tv = g·mu_anchor.
2. **Beat-noise centring.** Beat noise is eta ~ N(0, s^2) on the log scale, with s = sqrt(ln(1+CV^2)). This is median-centred and protocol-literal, so beat means exceed mu by exp(s^2/2) (+4.3% at CV 30%). The alternative `beat_noise_centering: mean` is available.
3. **Axes.** AP and SL have independent view errors, beat noise and caliper error. Each view's RR sequence and available-beat count are shared across axes (same clip).
4. **View errors.** u is half-normal, driven by a correlated latent Gaussian (shared component, rho) and capped at 0.9. Occurrence of o uses a Gaussian copula with the same rho. The magnitude of o is lognormal (median 0.15, log-SD 0.5) and independent across views.
5. **Instrument factor.** g is lognormal with median 1 (the reference settings), is study-level and multiplies the beat spans.
6. **Reader terms.** Caliper error and reader bias are additive after g. In E1 to E4 each patient gets its own reader bias; drift is modelled only in E5 and E6 (default 0). There is no floor on readings (see Tests).
7. **Acceptance window.** The reader measures N beats. If the window fails, beats are added one at a time and the reader keeps the N most mutually consistent beats (a contiguous block of the sorted values with the smallest relative range, (max - min)/|mean|). If no block qualifies once all available beats are used, the view is flagged "limited sampling" and the mean of all acquired beats is reported. If the view has fewer than N available beats, it is limited from the start.
8. **Beat availability.** "Unlimited on request" (prospective) is capped at 30 beats; at AF, CV 30%, W 25% and N 13 the cap binds in 1.4% of pilot views. Retrospective availability is zero-truncated Poisson(lambda 4, mean 4.07), capped at 15.
9. **AF and the index beat.** In AF, log RR is iid (no autocorrelation) and span depends on the preceding RR (beta_RR 0.2), with an extra 5% residual CV. Sinus RR CV is 3%. The protocol gives no mechanism by which RR1/RR2 close to 1 identifies a representative beat, so `beta_drr` = 0 and the index beat (A6) has no built-in advantage. A6 is a single anchor-view beat chosen among the acquired beats.
10. **A4 scrutiny.** The reader judges only verification views that raised a warning or an adjudication trigger (`a4_scrutiny: triggered`). The alternatives `adj_only` and `all_higher` are implemented.
11. **A4 detection.** "True overestimation" means o > 0, whatever its size. The anchor is never rejected. Warnings and adjudications are judged with the same s_det and f_rej.
12. **A7.** A7 divides each view mean by its (1 − u + o). It does not correct g or reader bias.
13. **E1 fixed settings.** W is held at its base value (15%), with prospective acquisition and sinus rhythm; N is on the grid. A4 is evaluated for all 12 combinations of s_det × f_rej at t_warn 3 / t_adj 5.
14. **E3.** Net reclassification is computed against A1, per axis, versus that axis's T1.
15. **E4.** A true error is present for pair (anchor, v) when g·S·o_v > 2 mm or g·S·u_anchor > 2 mm. The noise-only scenario sets view_errors = false. The r_mean grid {0.55, 0.614, 0.75, 0.9} is my addition, because the protocol names no ellipticity grid. Cross-axis differences within the anchor view are reported separately from cross-view triggers, and only valid pairs with t_warn < t_adj are used (8 pairs).
16. **E5.**
    - Reads of the same patient share the images (beats, u, o, g). They differ in caliper error, reader bias and the A4 decision draws.
    - Each study has a pool of 4 readers, and each patient gets 3 distinct readers.
    - Adjudication happens when |r1 − r2| > tol, and the adjudicator's read becomes the reference.
    - The inherited-bias AI is a + b·T1 + e, where (a, b) comes from regressing one protocol read on T1 in 20,000 separately seeded patients.
    - Validation studies have n_val = 200 and use the AP axis only.
17. **E6.**
    - The gold set is read by two fixed readers per study.
    - The overlap design is treated as a gold set of size round(500·f).
    - Precision is reported as the empirical SD across studies plus the mean nominal CI width. The ICC(A,1) CI uses the McGraw-Wong formula; the pytest coverage check passed within [0.92, 0.985].
    - Sentinel: one production reader. Two tests are applied: Welch t on the mean of (read − AI) and a one-sided F test on its variance, comparing sentinel (manual) cases with assisted cases. The AI draft has a 1 mm bias. alpha = 0 is included as a size check.
18. **Seeds.** Each cell's seed is `SeedSequence(20260918, spawn_key=(exp, cell))`, with one child per 20,000-patient chunk. A cell's results do not change with how cells are split across runs (tested).

**Not done.**
- The independent second-coder reimplementation required by protocol 6(b) is not done; it is outside this build.
- The parameter values are unreconciled with the literature table.
- No full-grid run has been made, and no Slurm job has been submitted.
- The project is not under git, so outputs cannot be traced to a commit.

## 2026-09-18 (afternoon): parameter reconciliation, full run, audits, sensitivity

- Config base v02 reconciled against verified evidence (notes/parameter-table-2026-09-18.md). Long-axis underestimation scale raised from 0.10 to 0.25 on evidence (Singh 2026; bRIGHT); the old value retained as the "low" scenario (u_level axis added to E1, E3, E4 before the full run).
- Regression fixture now uses frozen tests/fixtures/config_v01.yaml (detects code changes, not parameter changes). 41/41 tests pass.
- Full run: Slurm job 5223287, 00:04:24, 18.8 core-hours, 216 chunks, 0 failures; results/2026-09-18_full/ (provenance.txt holds sha256 of code/config/tests).
- Corrections to assumptions after audit:
  - Assumption 5: g is drawn per simulated patient, i.e. per echocardiographic examination (all views of one examination share it). "Study-level" in the earlier text meant examination, not validation study.
  - Assumption 15: the E4 r_mean grid run was {0.53, 0.64, 0.8, 0.9} (576 cells with the u_level axis), not the earlier grid.
  - Assumption 19 (new, implicit in E5 code): the AI model's error is independent of the image-level errors (view errors, beat noise, g) of the reads; the inherited-bias AI shares only the fitted systematic component.
- Pre-specified sentinel F test fails its nominal size (median 0.076, max 0.151 across 216 combinations); reported as a result, not repaired.
- EXPLORATORY post hoc sensitivity analyses (decided after reading audits; config code/configs/sens_2026-09-18.yaml, master seed 20260919, job 5223386, 00:01:16, 5.4 core-hours; analysis code/06_analyse_sensitivity.py): mean-centred beat noise (E1, E2); A4 scrutiny all_higher and adj_only (E1); closest-pair-mean adjudication (E5). No qualitative conclusion changed (see results/2026-09-18_sens/analysis/).

## 2026-09-18 (evening): protocol amendment 2 implemented (code, config, tests, pilot; not run)

Label: all output of this config is "amendment 2", secondary to E1 to E6.

**Code changes** (duomaxsim version string unchanged at 0.1.0; every new behaviour is gated so existing configs give bit-identical output).
- `code/lib/duomaxsim/metrics.py:brown_forsythe` (new): vectorised Brown-Forsythe test. Two-sided p equals `scipy.stats.levene(center='median')` (checked to rel 1e-9). One-sided p: pooled-variance t test on absolute deviations from group medians (the signed square root of the two-group BF ANOVA F), H1 sentinel (group 1) more dispersed than assisted (group 2), matching the direction of the existing one-sided F test.
- `experiments.py:protocol_read`: new `return_u=False` argument returns the A4 decision uniforms U; random-number consumption unchanged. Shared logic moved into new helper `_final_value`.
- `experiments.py:image_composite` (new): R_img, the reference rule (A4 in the amendment config) applied to the true beat spans (beats, u, o, g, availability) with caliper error 0 and reader bias 0; `measure_floor_mm` still applied; beat rule as for a read.
- `experiments.py:ai_shared` (new): AI = T1 + lambda (R_img - T1) + sigma_AI z.
- `experiments.py:run_E5`: if parameter `ai_shared_lambda` is present, adds AI "shared" (same z as the independent AI), reference row `R_img` (MAE and bias vs T1), and per (AI in {shared, inherited}, sigma_AI, reference incl. T1) the paired per-study MAE difference versus the independent AI: `p_lower_mae_than_independent` (binomial MCSE), `paired_mae_diff_mean` (MCSE), `paired_mae_diff_sd` (normal-theory MCSE); plus a `shared_ai` row recording lambda. Reads now loop explicitly to keep reader 1's U (same draw order).
- `experiments.py:run_E6`: if `e6_brown_forsythe` is true, adds `power_reject_bf` (two-sided) and `power_reject_bf_onesided`; F and Welch rows unchanged. Size = rows with anchor_alpha 0.
- `experiments.py:run_E1`: if `report_selection_inflation` is true, adds `selection_inflation_mm` (estimator "A3-A1", E[A3 - A1] per axis, MCSE from the paired differences).
- New config `code/configs/amend2_2026-09-18.yaml` (config_version amend2-2026-09-18-v01, master seed 20260920; parameter block copied from base v02 plus `ai_shared_lambda` {0, 0.5, 1}, `e6_brown_forsythe`, `report_selection_inflation`). Cells: E1 (E1b) 864, E3 (E3b) 12, E5 (E5b) 18, E6 (E6b) 54. u_combo levels carry `u_anchor` and `u_long` as descriptive keys, so outputs have `p_u_anchor` and `p_u_long` columns alongside `p_u_scale` = [a, L, L, 1.2 L].
- New `code/slurm/run_amend2.sh` (not submitted): OUT results/2026-09-18_amend2, 80 tasks (E1 16 cells, E3 6, E5 3, E6 3 per task), 242 files, 01:00:00.
- New `tests/test_amend2.py` (17 tests).

**Decisions and assumptions.**
1. A4 decisions for R_img reuse reader 1's decision uniforms U: a view that reader 1 would reject if scrutinised is rejected in R_img if scrutinised there; which views are scrutinised depends on the error-free view means. No extra random numbers are drawn, which keeps lambda = 0 identical to the previous E5. Consequence: R_img shares detection decisions with the "single" reference (reader 1), a modest extra coupling with that reference only.
2. R_img uses `reference_estimator` (A4 in the amendment config, as the protocol states). It includes g, as the protocol's "from the images" implies.
3. lambda is a grid-cell factor (as specified), not an in-cell variant, so comparisons across lambda are not on common random numbers; the shared-versus-independent comparison within a cell is (same images, reads and z).
4. "Inherited-bias AI" in the paired comparison is reported for both the new shared AI and the existing regression-inherited AI; comparisons are also made against T1 (true MAE) for context.
5. E1b: rho 0.3, N 3, sinus, prospective fixed; A4 still evaluated at all 12 s_det x f_rej combinations in cell. Selection inflation E[A3 - A1] reported as its own metric; A1 bias remains in the usual rows.
6. E3b: K 3, beat CV 0.15, overestimation on, window 0.15 (base).

**Tests.** `pytest tests`: 58 passed (41 existing + 17 new), 52 s. Regression fixture (frozen config_v01) unchanged and passing. Additional check: outputs of base.yaml for E1 cells 0 and 1000, E3 cell 40, E5 cells 0 and 11, E6 cells 0 and 53 (small n) were identical, as full-precision strings, before and after the change. New tests: lambda = 0 reproduces all fixture E5 rows and the shared AI equals the independent AI exactly; lambda = 1, sigma_AI -> 0 gives AI = R_img, and end to end with no caliper error or reader bias the shared AI has apparent MAE 0 against the single read; BF equals scipy levene; BF size under a normal null with 200 datasets within 4 binomial SE of 0.05 (two- and one-sided, n 10/490 and 50/450); one-sided direction; cell counts and u_combo levels.

**Pilot timings** (login node, CPU s per cell, scratchpad output only; not in results/).

| Exp | Pilot n | Cells piloted | CPU/cell | Full n scale | Cells | Est. full CPU |
|---|---|---|---|---|---|---|
| E1b | 10^4 patients | 0, 2, 861, 863 | 0.03 to 0.26 s | ×10 | 864 | 0.07 to 0.62 h |
| E3b | 10^4 | 0, 11 | 0.17 to 0.18 s | ×10 | 12 | < 0.01 h |
| E5b | 100 studies | 0, 17 | 0.90 to 1.13 s | ×20 | 18 | 0.09 to 0.11 h |
| E6b | 100 studies | 0, 53 | 1.26 to 1.44 s | ×20 | 54 | 0.38 to 0.43 h |
| Total | | | | | 948 | about 0.6 to 1.2 CPU-hours |

Pilot plausibility: at lambda 0 the shared AI equals the independent AI (paired difference 0); at lambda 1 (cell 17) the shared AI had lower apparent MAE than the independent AI against the single read in 100/100 pilot studies while its true MAE was higher by 1.02 mm. BF rejection at anchor_alpha 0 was 0.05 to 0.12 in 100-study pilots (not a size estimate). These pilot numbers are for code checking only.

**Not done.** Face-validity comparison (amendment 2, last bullet) is an analysis step, not implemented here. No analysis scripts updated for the new metrics. Job not submitted.
