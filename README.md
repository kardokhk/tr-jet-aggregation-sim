# Beat and view aggregation rules for tricuspid regurgitant jet dimensions: simulation code and results

Code, configurations, results and figures for a Monte Carlo simulation study of composite multi-beat,
multi-view echocardiographic measurement rules for tricuspid regurgitant jet dimensions, and of how
reference standards built from them change the apparent performance of artificial intelligence (AI)
models.

Author: Kardokh Kakabra (RCSI / CVRI Dublin). Archived releases (all versions): https://doi.org/10.5281/zenodo.22833343 (version 0.5.0: https://doi.org/10.5281/zenodo.23189369). Funding: Disruptive Technologies Innovation Fund (DTIF),
Enterprise Ireland, grant DT20240543A; the funder had no role in the study.

## Protocol stages

The study is not pre-specified as a whole. Each run directory belongs to one stage:

| Run directory | Stage | Status of its results |
|---|---|---|
| `results/2026-09-18_pilot` | Timing pilot (login node), run after the initial protocol was written and before the full run | not reported as results |
| `results/2026-09-18_full` | Initial protocol of 18 September 2026, written before any simulation: six planned experiments (E1 to E6). The run also holds the second level of long-axis underestimation of the first amendment, which was implemented before the run and documented after it | planned analyses |
| `results/2026-09-18_sens` | Sensitivity analyses run after the results of the full run were known | post hoc exploratory |
| `results/2026-09-18_amend2` | Second amendment, written after first results were known and before its own analyses were run (12 view-accuracy scenarios, cut-offs across view accuracy, AI models that share image error, Brown-Forsythe test, comparison with published values) | secondary |
| `results/2026-10-05_osrev` | Third amendment of 5 October 2026: sensitivity analyses requested by the senior author at internal review and specified before they were run (beat rules, view rules, cut-offs, triggers and review, AI evaluation and reference sets, calibration of long-axis underestimation), with software-verification reruns | post hoc |

The protocol was not registered and the protocol file is not included; its content and the three
amendments are summarized in the manuscript and its supplement. Job times are held in the cluster
accounting records (`environment.md`), whereas the timing of the protocol text rests on the authors' files.

## Contents

- `notes/parameter-table-2026-09-18.md`: parameter values and literature sources (assumed, or informed by indirect sources; none is a calibration of the jet span)
- `notes/lab-notebook.md`: implementation log, assumptions and deviations
- `code/lib/duomaxsim/`: simulation package (generative model, rules, estimators, metrics, experiments) and the third-amendment modules `osrev_*.py`, which extend the library without modifying it
- `code/01_run_experiment.py`: command-line runner; `code/configs/*.yaml`: configurations and seeds
- `code/slurm/*.sh`: batch scripts used on the MeluXina supercomputer
- `code/02_*.py` to `code/15_*.py`: analysis, face-validity and table scripts of the runs of 18 September 2026
- `code/20_osrev_*_run.py`, `21_osrev_*`, `22_osrev_*`, `23_*`: runners, analysis scripts and table script of the third amendment; `code/osrev_verification.py`: software-verification reruns
- `code/figures/`: figure scripts; `code/tools/`: reference numbering, word count and job-accounting helpers
- `tests/`: automated tests (`pytest tests`; log of the last full run in `tests/logs/`) and second implementations written by a separate AI coding agent (`tests/independent/`); this is software verification, not independent scientific review
- `results/`: the three batch runs of 18 September 2026 each hold `provenance.txt` and `config_used.yaml`, with `analysis/` holding the tables behind their reported numbers (the timing pilot has run manifests only); every third-amendment package holds `provenance.txt`, and `results/2026-10-05_osrev/README.md` indexes the packages
- `figures/`: final figures (PDF, PNG, TIFF) with source data; see `figures/README.md`

Figure 1 of the manuscript is the schematic `figures/fig1_design.*`.

## Reproduce

```bash
micromamba create -p ./env --file environment-explicit.txt   # or see environment.md
bash run_all.sh simulate   # submits the Slurm jobs of all stages (adapt account/partition)
bash run_all.sh analyse    # tables and figures from the stored results
bash run_all.sh test       # full test suite as a batch job
```

The simulation uses no patient data. Results are conditional on the generative model described in the
manuscript and supplement. Known issues of the code and of the stored provenance records are listed in
`code/README.md`.

## Licence

Code: MIT (`LICENSE`). Results, figures and documentation: CC BY 4.0 (`LICENSE-DATA`).
