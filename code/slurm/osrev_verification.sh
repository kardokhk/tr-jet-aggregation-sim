#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=osrev_verification
#SBATCH --output=logs/%x-%j.out
# POST HOC software-verification reruns (comment C191; amendment 3, A3-1c). Run from the project root.
set -euo pipefail
PY=/project/home/p201509/envs/duomax-sim/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/2026-10-05_osrev/verification
mkdir -p "$OUT"
{ date -Is; hostname; $PY --version; git rev-parse HEAD; git status --short code tests;
  sha256sum code/lib/duomaxsim/*.py code/osrev_verification.py tests/independent/*.py tests/fixtures/config_v01.yaml \
            code/configs/base.yaml code/configs/amend2_2026-09-18.yaml environment-explicit.txt requirements.txt; } > "$OUT/provenance.txt"
$PY code/osrev_verification.py --workers 120
echo done; date -Is
