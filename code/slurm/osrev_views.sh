#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=osrev-views
#SBATCH --output=logs/%x-%j.out
# Protocol amendment 3, package A3-2 (view rules). Run from the project root:
#   sbatch --wait code/slurm/osrev_views.sh
# 142 tasks (986 cells of 100,000 patients) on one node, then merge into four parquet files.
set -euo pipefail
OUT=${OUT:-results/2026-10-05_osrev/views}
PY=/project/home/p201509/envs/duomax-sim/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
{ echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "slurm_job_id: ${SLURM_JOB_ID:-none}";
  echo "python: $($PY --version 2>&1)";
  $PY -c 'import numpy, scipy, pandas, pyarrow; print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "pyarrow", pyarrow.__version__)';
  echo "git_commit: $(git rev-parse HEAD) (new amendment-3 files are untracked at run time)";
  echo "sha256:";
  sha256sum code/lib/duomaxsim/*.py code/20_osrev_views_run.py code/configs/osrev_views.yaml \
            code/configs/amend2_2026-09-18.yaml code/configs/base.yaml code/slurm/osrev_views.sh \
            tests/test_osrev_views.py; } > "$OUT/provenance.txt"
rm -rf "$OUT/parts"
n=$($PY code/20_osrev_views_run.py --list 2>/dev/null | wc -l)
echo "tasks: $n"; date -Is
# longest tasks (decomposition, four arms) first
{ seq 109 $((n-1)); seq 0 108; } | xargs -P 126 -I{} bash -c \
  "$PY code/20_osrev_views_run.py --task {} --out $OUT || echo FAIL {}"
date -Is
$PY code/20_osrev_views_run.py --merge --out "$OUT"
rm -rf "$OUT/parts"
echo done; date -Is
