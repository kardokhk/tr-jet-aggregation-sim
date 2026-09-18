# Environment (captured 2026-09-18)

- Host: MeluXina login node (login02.meluxina.lxp.lu), Red Hat Enterprise Linux 8.10, kernel 4.18.0-553.159.1.el8_10.x86_64.
- Environment manager: micromamba 2.9.0 (`~/.local/bin/micromamba`), `MAMBA_ROOT_PREFIX=/project/home/p201509/micromamba`.
- Environment: `/project/home/p201509/envs/duomax-sim`, channel conda-forge only.
- Create: `micromamba create -y -p /project/home/p201509/envs/duomax-sim -c conda-forge python=3.11 numpy scipy pandas matplotlib pytest pyyaml pyarrow`.
- Exact rebuild: `micromamba create -p <prefix> --file environment-explicit.txt` (explicit conda-forge URLs).
- Pinned pip-style list: `requirements.txt`.

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

- `duomaxsim` 0.1.0 (`code/lib/duomaxsim/`). The project is not under git yet, so there is no commit hash; each run manifest stores the full config snapshot and package versions instead. Put the project under git before the confirmatory full-grid run.

## Randomness

- Master seed 20260918 (`code/configs/base.yaml`, `meta.master_seed`). Cell seed = `SeedSequence(entropy=20260918, spawn_key=(experiment_index, cell_index))`, with E1..E6 mapped to 1..6. Each cell spawns one child per chunk of 20,000 patients, and each child drives a PCG64 generator. Every output row records `seed_entropy`, `seed_spawn_key` and `chunk_size`.
- The test fixtures use their own fixed seeds (numpy default_rng seeds 1, 3, 5, 7 and 11) plus the master seed.

## E1/E3 analysis (2026-09-18, login02)

- No packages installed. Scripts: `code/02_analyse_e1e3.py`, `code/02_check_e1_analytic.py`, `code/figures/fig2_estimator_bias.py`, `code/figures/fig3_misclassification.py`, `code/figures/figS1_e1_sensitivity.py` (shared `code/figures/estimator_style.py`); figures rendered in Nimbus Sans via `figstyle.use_print_style()`.
- `02_check_e1_analytic.py` runs new simulations with the unchanged library: seeds `SeedSequence(20260918, spawn_key=(1, 900000 + j))` for j = 0..26 (analytic check) and j = 100..103 (offset decomposition), one child per 20,000 patients, 100,000 patients each; these spawn keys do not overlap the E1 grid (cells 0 to 2303).

## Amendment 2, E5b/E6b analysis (2026-09-18, login node)

- No packages installed; env `/project/home/p201509/envs/duomax-sim` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, pandas 3.0.5, matplotlib 3.11.2).
- `code/10_amend2_E5b_decomposition.py` regenerates E5b draws with the unchanged library and the stored seeds `SeedSequence(20260920, spawn_key=(5, cell))` for cells 9 (2,000 studies), 0, 3, 6, 12, 15 (first 500 studies); reproduces stored cell-9 values exactly. Block jackknife (20 blocks of studies) for MCSE. 33 CPU s.
- `code/11_amend2_E5bE6b_tables.py` (no random numbers, 6 CPU s) and `code/figures/fig6_reference.py` (replaced; Nimbus Sans; v01 kept in `notes/scratch/2026-09-18-fig6_reference_v1_script.py`, `notes/scratch/2026-09-18-fig6-v1/`).
