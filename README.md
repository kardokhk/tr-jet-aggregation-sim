# Beat and view aggregation rules for tricuspid regurgitant jet dimensions: simulation code and results

Code, configurations, results and figures for a pre-specified Monte Carlo simulation study of
composite multi-beat, multi-view echocardiographic measurement rules for tricuspid regurgitant jet
dimensions, and of how reference standards built from them change the apparent performance of
artificial intelligence (AI) models.

Author: Kardokh Kakabra (RCSI / CVRI Dublin). Archived releases (all versions): https://doi.org/10.5281/zenodo.22833343 (see `CITATION.cff` for the current version). Funding: Disruptive Technologies Innovation Fund (DTIF),
Enterprise Ireland, grant DT20240543A (project code 26151); the funder had no role in the study.

## Contents

- The study protocol is not included; its content and both amendments are summarized in the manuscript and its supplement
- `notes/parameter-table-2026-09-18.md`: parameter values and literature sources
- `notes/lab-notebook.md`: implementation log, assumptions and deviations
- `code/lib/duomaxsim/`: simulation package (generative model, rules, estimators, metrics, experiments)
- `code/01_run_experiment.py`: command-line runner; `code/configs/*.yaml`: configurations and seeds
- `code/slurm/*.sh`: batch scripts used on the MeluXina supercomputer
- `code/02_*.py` to `code/15_*.py`: analysis, face-validity and table scripts (`15_make_tables_v04.py` builds the tables of manuscript v04); `code/figures/`: figure scripts
- `tests/`: unit tests (`pytest tests`) and independent reimplementations (`tests/independent/`)
- `results/2026-09-18_full` (pre-specified E1 to E6), `_sens` (exploratory sensitivity analyses),
  `_amend2` (amendment 2), `_pilot` (timing pilot); each run directory holds `provenance.txt` and
  `config_used.yaml`, and `analysis/` holds the tables behind every reported number
- `figures/`: final figures (PDF, PNG, TIFF) with source data; see `figures/README.md`

## Reproduce

```bash
micromamba create -p ./env --file environment-explicit.txt   # or see environment.md
bash run_all.sh simulate   # submits the three Slurm jobs (adapt account/partition)
bash run_all.sh analyse    # tables and figures from the stored results
```

The simulation uses no patient data. Results are conditional on the generative model described in the manuscript and supplement.

## Licence

Code: MIT (`LICENSE`). Results, figures and documentation: CC BY 4.0 (`LICENSE-DATA`).
