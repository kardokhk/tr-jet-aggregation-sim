# Environment (captured 2026-09-18; rechecked and extended 2026-10-06)

- Host: MeluXina login node (login02.meluxina.lxp.lu), Red Hat Enterprise Linux 8.10, kernel 4.18.0-553.159.1.el8_10.x86_64.
- Environment manager: micromamba 2.9.0 (`~/.local/bin/micromamba`), `MAMBA_ROOT_PREFIX=/project/home/p201509/micromamba`.
- Environment: `/project/home/p201509/envs/duomax-sim`, channel conda-forge only.
- Create: `micromamba create -y -p /project/home/p201509/envs/duomax-sim -c conda-forge python=3.11 numpy scipy pandas matplotlib pytest pyyaml pyarrow`.
- Exact rebuild: `micromamba create -p <prefix> --file environment-explicit.txt` (explicit conda-forge URLs).
- Pinned pip-style list: `requirements.txt`.
- Recheck of 2026-10-06 (login02, kernel 4.18.0-553.164.1.el8_10.x86_64): the live environment equals `environment-explicit.txt` (208 package URLs, `micromamba list --explicit`) and `requirements.txt` (32 pins, `pip list --format=freeze`). sha256: `environment-explicit.txt` c32dc057a01affeb1616dc74ce04598cc48bcca67b9382366024231609aace8d, `requirements.txt` e619996d732558e1731d82cd665c704dc13ee7e2f1f32599de049ada3da01b54.

| Package | Version | Build |
|---|---|---|
| python | 3.11.16 | h5f976f7_2_cpython |
| numpy | 2.4.6 | py311h2e04523_0 |
| scipy | 1.17.1 | py311hbe70eeb_1 |
| pandas | 3.0.5 | py311h8032f78_1 |
| pyarrow | 25.0.0 | py311h38be061_0 |
| pyyaml | 6.0.3 | py311h3778330_1 |
| matplotlib | 3.11.2 | py311h968eb38_0 |
| pytest | 9.1.1 | pyhc364b38_2 |
| libopenblas | 0.3.34 | pthreads_hf13c14d_2 |

## In-house code

- `duomaxsim` 0.1.0 (`code/lib/duomaxsim/`). The project is under git: commit e650519 (2026-09-18, tag v0.3.1) and commit 57371d33737904059ed8272d53aa3933bf9ad985 (2026-09-19, tag v0.4.0). The three batch runs of 18 September 2026 predate the first commit; their `provenance.txt` files record sha256 values of the code instead of a commit (see `code/README.md`, known issue 5).
- The third-amendment runs of 5 and 6 October 2026 ran at commit 57371d3 with the third-amendment files as untracked additions; each package's `provenance.txt` records the sha256 of every file that ran. These files enter the history with release 0.5.0 (version 0.5.0, DOI to be assigned at release).
- Library files of the earlier runs (`config.py`, `model.py`, `rules.py`, `estimators.py`, `metrics.py`, `experiments.py`) are unchanged since commit e650519.

## Randomness

- Master seed 20260918 (`code/configs/base.yaml`, `meta.master_seed`). Cell seed = `SeedSequence(entropy=20260918, spawn_key=(experiment_index, cell_index))`, with E1..E6 mapped to 1..6. Each cell spawns one child per chunk of 20,000 patients, and each child drives a PCG64 generator. Every output row records `seed_entropy`, `seed_spawn_key` and `chunk_size`.
- The test fixtures use their own fixed seeds (numpy default_rng seeds 1, 3, 5, 7 and 11) plus the master seed.
- Second amendment: master seed 20260920 (`code/configs/amend2_2026-09-18.yaml`), same spawn-key scheme `(experiment_index, cell_index)`.
- Third amendment: master seed 20261005, `SeedSequence(entropy=20261005, spawn_key=(prefix, ...))`. Prefixes: 71 beat rules, 72 view rules, 73 cut-offs, 74 triggers and review, 75 AI evaluation and reference sets, 76 calibration (packages); 131 stand-alone check of the simulated AP minus SL difference; 191 paired replicates of the window effect (`code/osrev_verification.py`); 971 to 976 the independent audits of the six packages, in the same order. The key layout below each prefix is given in the package configuration (`code/configs/osrev_*.yaml`) and in each `provenance.txt`. Stored conditions were regenerated with their original seeds (entropy 20260918 or 20260920), never with new ones.
- Conditions are simulated with independent streams; rules, review settings, reference designs and AI models compared within a condition share the same simulated patients or studies.

## E1/E3 analysis (2026-09-18, login02)

- No packages installed. Scripts: `code/02_analyse_e1e3.py`, `code/02_check_e1_analytic.py`, `code/figures/fig2_estimator_bias.py`, `code/figures/fig3_misclassification.py`, `code/figures/figS1_e1_sensitivity.py` (shared `code/figures/estimator_style.py`); figures rendered in Nimbus Sans via `figstyle.use_print_style()`.
- `02_check_e1_analytic.py` runs new simulations with the unchanged library: seeds `SeedSequence(20260918, spawn_key=(1, 900000 + j))` for j = 0..26 (analytic check) and j = 100..103 (offset decomposition), one child per 20,000 patients, 100,000 patients each; these spawn keys do not overlap the E1 grid (cells 0 to 2303).

## Amendment 2, E5b/E6b analysis (2026-09-18, login node)

- No packages installed; env `/project/home/p201509/envs/duomax-sim` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, pandas 3.0.5, matplotlib 3.11.2).
- `code/10_amend2_E5b_decomposition.py` regenerates E5b draws with the unchanged library and the stored seeds `SeedSequence(20260920, spawn_key=(5, cell))` for cells 9 (2,000 studies), 0, 3, 6, 12, 15 (first 500 studies); reproduces stored cell-9 values exactly. Block jackknife (20 blocks of studies) for MCSE. 33 CPU s.
- `code/11_amend2_E5bE6b_tables.py` (no random numbers, 6 CPU s) and `code/figures/fig6_reference.py` (replaced; Nimbus Sans; v01 kept in `notes/scratch/2026-09-18-fig6_reference_v1_script.py`, `notes/scratch/2026-09-18-fig6-v1/`).

## Third amendment (2026-10-05 and 2026-10-06)

- No packages installed; same environment. Document tools on the login node: pandoc 3.10.1 (`~/.local/bin/pandoc`), system Python 3.6.8 for the reference and word-count helpers in `code/tools/`.
- Batch jobs ran on one `cpu` node each (128 physical cores, 256 hardware threads; partition cpu, qos short). Login-node steps: `code/20_osrev_thresholds_run.py published`, `code/21_*` and `code/22_*` analysis scripts, `code/23_make_tables_v06.py`, figure scripts (each under about one CPU-minute).
- Full test suite, 2026-10-06: see the last log in `tests/logs/`.

## Slurm jobs (cluster accounting, `python3 code/tools/slurm_job_table.py`)

Times are local (CEST). The node is billed whole: node-hours are elapsed time on one node. sacct counts hardware threads, so its CPU-hours are twice the core-hours on the 128 physical cores.

<!-- JOBTABLE:START -->
Accounting read 2026-10-06 10:24.

| Job | Name | Start | Node | State | Elapsed | Node-hours | CPU-hours (sacct) | Core-hours |
|---|---|---|---|---|---|---|---|---|
| 5223287 | duomax-grid | 2026-09-18 14:40:18 | mel0393 | COMPLETED | 00:04:24 | 0.073 | 18.77 | 9.39 |
| 5223386 | duomax-sens | 2026-09-18 15:13:35 | mel0327 | COMPLETED | 00:01:16 | 0.021 | 5.40 | 2.70 |
| 5223669 | duomax-amend2 | 2026-09-18 15:59:05 | mel0230 | COMPLETED | 00:02:21 | 0.039 | 10.03 | 5.01 |
| 5301518 | osrev_verification | 2026-10-05 18:03:40 | mel0503 | COMPLETED | 00:00:47 | 0.013 | 3.34 | 1.67 |
| 5301599 | osrev-views | 2026-10-05 18:08:05 | mel0429 | COMPLETED | 00:02:24 | 0.040 | 10.24 | 5.12 |
| 5301601 | osrev_thresholds | 2026-10-05 18:08:39 | mel0146 | COMPLETED | 00:00:36 | 0.010 | 2.56 | 1.28 |
| 5301653 | duomax-osrev-beats | 2026-10-05 18:10:55 | mel0557 | COMPLETED | 00:01:08 | 0.019 | 4.84 | 2.42 |
| 5301795 | osrev-triggers | 2026-10-05 18:18:09 | mel0487 | COMPLETED | 00:01:25 | 0.024 | 6.04 | 3.02 |
| 5301846 | osrev-calibration | 2026-10-05 18:19:16 | mel0038 | COMPLETED | 00:02:28 | 0.041 | 10.52 | 5.26 |
| 5301894 | osrev_ai_reference | 2026-10-05 18:21:04 | mel0030 | FAILED | 00:00:55 | 0.015 | 3.91 | 1.96 |
| 5301941 | osrev_ai_reference | 2026-10-05 18:22:33 | mel0038 | COMPLETED | 00:02:34 | 0.043 | 10.95 | 5.48 |
| 5301947 | osrev-calibration | 2026-10-05 18:24:23 | mel0315 | COMPLETED | 00:02:14 | 0.037 | 9.53 | 4.76 |
| 5306852 | duomax-pytest | 2026-10-06 09:57:14 | mel0032 | COMPLETED | 00:01:34 | 0.026 | 6.68 | 3.34 |
| 5306853 | osrev-fig4-clustered | 2026-10-06 09:57:20 | mel0495 | COMPLETED | 00:00:28 | 0.008 | 1.99 | 1.00 |
| 5306860 | osrev-calibration | 2026-10-06 10:05:04 | mel0398 | FAILED | 00:00:58 | 0.016 | 4.12 | 2.06 |
| 5306863 | osrev-calibration | 2026-10-06 10:08:24 | mel0398 | COMPLETED | 00:00:55 | 0.015 | 3.91 | 1.96 |
| 5306870 | duomax-pytest | 2026-10-06 10:14:58 | mel0398 | COMPLETED | 00:01:26 | 0.024 | 6.12 | 3.06 |
| total | | | | | | 0.465 | 118.97 | 59.48 |
<!-- JOBTABLE:END -->

Job roles: 5223287 planned experiments (`results/2026-09-18_full`); 5223386 post hoc exploratory sensitivity analyses (`_sens`); 5223669 second amendment (`_amend2`); 5301518 software-verification reruns; 5301599 view rules; 5301601 cut-offs; 5301653 beat rules; 5301795 triggers and review; 5301846 calibration (superseded by 5301947: same simulation results, csv precision too low for the bit-level check); 5301894 AI evaluation (failed after 55 s on an unmatched model pair in a sensitivity cell; fixed and resubmitted as 5301941); 5306852 and later `duomax-pytest` jobs full test suite; 5306853 patient-clustered standard errors for Figure 4; `osrev-calibration` jobs of 2026-10-06 the calibration follow-up (stages C and D).

