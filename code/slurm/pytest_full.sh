#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=duomax-pytest
#SBATCH --output=logs/%x-%j.out
# Full test suite (tests of the 2026-09-18 library and of the six third-amendment packages), with a
# provenance header, written to a tracked log.  Run from the project root:
#   sbatch --wait code/slurm/pytest_full.sh            (log: tests/logs/pytest-<date>.log)
set -uo pipefail
PY=/project/home/p201509/envs/duomax-sim/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG="tests/logs/pytest-$(date +%F).log"
mkdir -p tests/logs
{
  echo "# Full test suite: pytest -q tests"
  echo "date: $(date -Is)"
  echo "host: $(hostname)   slurm job: ${SLURM_JOB_ID:-none}"
  echo "git commit (HEAD): $(git rev-parse HEAD)"
  echo "git describe: $(git describe --tags --always --dirty 2>/dev/null)"
  echo "python: $($PY --version 2>&1)"
  echo "numpy: $($PY -c 'import numpy; print(numpy.__version__)')"
  echo "scipy: $($PY -c 'import scipy; print(scipy.__version__)')   pandas: $($PY -c 'import pandas; print(pandas.__version__)')   pytest: $($PY -m pytest --version 2>&1)"
  echo "git status --short, summary (working tree relative to HEAD; untracked directories collapsed):"
  echo "  modified tracked files: $(git status --short | grep -c '^ M\|^M')"
  echo "  untracked entries: $(git status --short | grep -c '^??')"
  echo "git status --short code tests (the files the tests exercise):"
  git status --short code tests | sed 's/^/  /'
  echo "sha256 of the library and test files:"
  sha256sum code/lib/duomaxsim/*.py tests/*.py tests/independent/*.py | sed 's/^/  /'
  echo "tests collected per file:"
  $PY -m pytest --collect-only -q -p no:cacheprovider tests 2>/dev/null | grep '::' | cut -d: -f1 | sort | uniq -c | sed 's/^/  /'
  echo "# ---- pytest output ----"
} > "$LOG"
$PY -m pytest -q -p no:cacheprovider tests >> "$LOG" 2>&1
rc=$?
echo "# pytest exit status: $rc   finished: $(date -Is)" >> "$LOG"
tail -3 "$LOG"
exit $rc
