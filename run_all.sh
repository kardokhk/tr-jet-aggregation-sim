#!/bin/bash -l
# Regenerate every result, table and figure of the simulation study from code.
#
#   bash run_all.sh simulate   submit the simulation jobs (Slurm, MeluXina; adapt account and partition)
#   bash run_all.sh analyse    tables and figures from the stored results (login node; default)
#   bash run_all.sh test       full test suite as a batch job, log in tests/logs/
#
# Protocol stage of each run (see README.md):
#   results/2026-09-18_full     planned analyses of the initial protocol (six experiments), with the second
#                               level of long-axis underestimation of the first amendment
#   results/2026-09-18_sens     post hoc exploratory sensitivity analyses
#   results/2026-09-18_amend2   second amendment (secondary)
#   results/2026-10-05_osrev    third amendment (post hoc sensitivity analyses, six packages) and software
#                               verification reruns
# The third-amendment jobs reproduce stored results of the earlier runs before they add anything, so they are
# submitted with a dependency on those runs.  Each analysis script takes less than about one CPU-minute.
set -euo pipefail
cd "$(dirname "$0")"
PY=/project/home/p201509/envs/duomax-sim/bin/python
OSREV=results/2026-10-05_osrev
mkdir -p logs
case "${1:-analyse}" in
  test)
    sbatch --wait code/slurm/pytest_full.sh
    tail -2 "tests/logs/pytest-$(date +%F).log" ;;
  simulate)
    $PY -m pytest -q -p no:cacheprovider tests/test_known_answers.py tests/test_edge_cases.py tests/test_regression.py tests/test_amend2.py
    j1=$(sbatch --parsable code/slurm/run_grid.sh)      # planned experiments  -> results/2026-09-18_full
    j2=$(sbatch --parsable code/slurm/run_sens.sh)      # post hoc exploratory -> results/2026-09-18_sens
    j3=$(sbatch --parsable code/slurm/run_amend2.sh)    # second amendment     -> results/2026-09-18_amend2
    echo "submitted $j1 $j2 $j3"
    # Third amendment: every code/slurm/osrev_*.sh except the Figure 4 standard errors, which need an
    # analysis table and are submitted by the analyse stage.
    for s in code/slurm/osrev_*.sh; do
      [ "$s" = code/slurm/osrev_fig4_clustered.sh ] && continue
      echo "submitted $(sbatch --parsable --dependency=afterok:$j1:$j2:$j3 "$s") ($s)"
    done
    echo "Rerun 'bash run_all.sh analyse' when 'squeue --me' is empty." ;;
  analyse)
    # Planned, first-amendment, second-amendment and post hoc exploratory analyses of 18 September 2026.
    for s in 02_analyse_e1e3 02_check_e1_analytic 03_analyse_e2 04_analyse_E4 05_analyse_E5E6 \
             06_analyse_sensitivity 08_analyse_amend2_e1e3 10_amend2_E5b_decomposition \
             11_amend2_E5bE6b_tables 12_face_validity; do
      echo "== $s"; $PY code/$s.py
    done
    # Third amendment (post hoc): the batch jobs already ran the 21_* table scripts of the beats, triggers,
    # calibration and AI-reference packages; they are repeated here so that the stage is self-contained.
    echo "== 20_osrev_thresholds_run published"; $PY code/20_osrev_thresholds_run.py published
    for s in code/21_osrev_*.py code/22_osrev_*.py; do echo "== $s"; $PY "$s"; done
    echo "== 21_osrev_calibration_analyse --followup"; $PY code/21_osrev_calibration_analyse.py --followup
    if [ ! -f $OSREV/fig4_clustered/fig4_clustered_se.csv ]; then   # patient-clustered standard errors, Figure 4
      sbatch --wait code/slurm/osrev_fig4_clustered.sh
    fi
    # Figures (main figures 2 to 5, Supplementary Figures S1 to S8), schematics and tables.
    for f in code/figures/fig*.py; do echo "== $f"; $PY "$f"; done
    $PY code/13_schematic_figures.py all        # schematic Figure 1 and graphical abstract
    $PY code/14_make_tables_v02.py
    $PY code/15_make_tables_v04.py      # tables of manuscript v04 (release 0.4.0)
    if [ -d drafts ]; then              # manuscript drafts are kept outside the public repository
      $PY code/23_make_tables_v06.py    # tables of manuscript v06: Table 1, Supplementary Tables S1 to S6, S8 to S27
      echo "Manuscript and supplement are assembled by 'bash submission/source/build_all.sh' (not public)."
    fi ;;
  *) echo "usage: bash run_all.sh [simulate|analyse|test]" >&2; exit 2 ;;
esac
