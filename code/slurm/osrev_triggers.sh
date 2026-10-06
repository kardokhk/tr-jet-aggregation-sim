#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=osrev-triggers
#SBATCH --output=logs/%x-%j.out
# Protocol amendment 3, package A3-4 (triggers and review). Run from the project root:
#   sbatch --wait code/slurm/osrev_triggers.sh
# 51 conditions of 100,000 patients = 255 chunk tasks of 20,000 patients on 120 workers,
# then the derived tables and the tests of this package.
set -euo pipefail
OUT=${OUT:-results/2026-10-05_osrev/triggers}
PY=/project/home/p201509/envs/duomax-sim/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT/raw"
{ echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "slurm_job_id: ${SLURM_JOB_ID:-none}";
  echo "python: $($PY --version 2>&1)";
  $PY -c 'import numpy, scipy, pandas, pyarrow, yaml; print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "pyarrow", pyarrow.__version__, "pyyaml", yaml.__version__)';
  echo "git_commit: $(git rev-parse HEAD) (amendment-3 files are untracked at run time)";
  echo "sha256:";
  sha256sum code/lib/duomaxsim/__init__.py code/lib/duomaxsim/config.py code/lib/duomaxsim/model.py \
            code/lib/duomaxsim/rules.py code/lib/duomaxsim/estimators.py code/lib/duomaxsim/metrics.py \
            code/lib/duomaxsim/experiments.py code/lib/duomaxsim/osrev_triggers.py \
            code/20_osrev_triggers_run.py code/21_osrev_triggers_tables.py code/configs/osrev_triggers.yaml \
            code/configs/base.yaml code/configs/amend2_2026-09-18.yaml code/slurm/osrev_triggers.sh \
            tests/test_osrev_triggers.py; } > "$OUT/provenance.txt"
date -Is
$PY code/20_osrev_triggers_run.py --out "$OUT/raw" --workers 120
date -Is
$PY code/21_osrev_triggers_tables.py --dir "$OUT"
date -Is
{ echo "pytest (tests/test_osrev_triggers.py tests/test_regression.py), $(date -Is):";
  $PY -m pytest tests/test_osrev_triggers.py tests/test_regression.py -q -p no:cacheprovider 2>&1 | tail -3; } | tee -a "$OUT/provenance.txt"
echo done; date -Is
