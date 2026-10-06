#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=osrev_thresholds
#SBATCH --output=logs/%x-%j.out
# POST HOC threshold classification detail (comment C201; amendment 3, A3-3). Run from the project root.
# 6 scenarios x 100 replicates x 100,000 patients, one replicate per task, 120 worker processes.
# Pilot (login node, 2026-10-05): about 2 CPU-s per task, so about 20 CPU-min in total.
set -euo pipefail
PY=/project/home/p201509/envs/duomax-sim/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/2026-10-05_osrev/thresholds
mkdir -p "$OUT/raw" logs
{ date -Is; hostname; $PY --version; git rev-parse HEAD; git status --short code tests;
  sha256sum code/lib/duomaxsim/*.py code/20_osrev_thresholds_run.py code/configs/osrev_thresholds.yaml \
            code/configs/base.yaml code/configs/amend2_2026-09-18.yaml tests/test_osrev_thresholds.py; } > "$OUT/provenance_replicates.txt"
$PY code/20_osrev_thresholds_run.py replicates --workers 120
echo done; date -Is
