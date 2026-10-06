#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=duomax-osrev-beats
#SBATCH --output=logs/%x-%j.out
# Protocol amendment 3, package A3-1 (beat rules): POST HOC sensitivity analyses requested by the
# senior author on 2026-10-05. Run from the project root: sbatch --wait code/slurm/osrev_beats.sh
set -euo pipefail
OUT=${OUT:-results/2026-10-05_osrev/beats}
CFG=code/configs/osrev_beats.yaml
PY=/project/home/p201509/envs/duomax-sim/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
  echo "python: $($PY --version 2>&1)"
  $PY -c 'import numpy, scipy, pandas; print("numpy:", numpy.__version__, "scipy:", scipy.__version__, "pandas:", pandas.__version__)'
  echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown) (working tree; the files below are untracked additions or unchanged library files)"
  echo "sha256:"
  sha256sum code/lib/duomaxsim/{__init__,config,model,rules,estimators,metrics,experiments,osrev_beats}.py code/20_osrev_beats_run.py code/21_osrev_beats_tables.py $CFG \
            code/configs/base.yaml code/slurm/osrev_beats.sh tests/test_osrev_beats.py
} > "$OUT/provenance.txt"
cp $CFG "$OUT/config_used.yaml"
# one worker per condition (48 conditions and published cells); the node is billed whole
$PY code/20_osrev_beats_run.py --workers 48 --out "$OUT"
$PY code/21_osrev_beats_tables.py --dir "$OUT"
echo done; date -Is
