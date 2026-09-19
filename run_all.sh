#!/bin/bash -l
# Regenerate every result, table and figure of the DUO-Max simulation paper from code.
# Stage 1 (simulation) runs as Slurm batch jobs on MeluXina; stages 2-4 run on the login node
# after the jobs finish (each script < 1 CPU-minute).  Usage: bash run_all.sh [simulate|analyse]
set -euo pipefail
cd "$(dirname "$0")"
PY=/project/home/p201509/envs/duomax-sim/bin/python
case "${1:-analyse}" in
  simulate)
    $PY -m pytest -q tests
    sbatch code/slurm/run_grid.sh      # pre-specified E1-E6  -> results/2026-09-18_full
    sbatch code/slurm/run_sens.sh      # exploratory sensitivity -> results/2026-09-18_sens
    sbatch code/slurm/run_amend2.sh    # amendment 2 -> results/2026-09-18_amend2
    echo "Submitted; rerun 'bash run_all.sh analyse' when squeue --me is empty." ;;
  analyse)
    for s in 02_analyse_e1e3 02_check_e1_analytic 03_analyse_e2 04_analyse_E4 05_analyse_E5E6 \
             06_analyse_sensitivity 08_analyse_amend2_e1e3 10_amend2_E5b_decomposition \
             11_amend2_E5bE6b_tables 12_face_validity; do
      echo "== $s"; $PY code/$s.py
    done
    for f in code/figures/fig*.py; do echo "== $f"; $PY "$f"; done
    $PY code/13_schematic_figures.py all
    $PY code/14_make_tables_v02.py
    $PY code/15_make_tables_v04.py      # v04 tables (main Table 1, Supplementary Tables S1 to S6)
    if [ -d drafts ]; then   # manuscript drafts are kept outside the public repository
      python3 code/tools/number_refs.py drafts/manuscript-2026-09-19-v04.md drafts/manuscript-2026-09-19-v04-numbered.md \
        notes/scratch/2026-09-18-refs-verified.json notes/refs-extra-verified-2026-09-18.json
    fi ;;
esac
