#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=osrev_ai_reference
#SBATCH --output=logs/%x-%j.out
# POST HOC sensitivity analyses, protocol amendment 3, A3-5 (comments C189, C209, C321, C399).
# Run from the project root: sbatch --wait code/slurm/osrev_ai_reference.sh
set -euo pipefail
PY=/project/home/p201509/envs/duomax-sim/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/2026-10-05_osrev/ai_reference
mkdir -p "$OUT/raw"
{ echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "slurm_job_id: ${SLURM_JOB_ID:-none}";
  echo "python: $($PY --version 2>&1)";
  $PY -c "import numpy, scipy, pandas, pyarrow; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__, 'pyarrow', pyarrow.__version__)";
  echo "git_commit: $(git rev-parse HEAD)"; echo "git_status_code_tests:"; git status --short code tests;
  echo "sha256:";
  sha256sum code/lib/duomaxsim/*.py code/20_osrev_ai_reference_run.py code/21_osrev_ai_reference_analyse.py \
            code/configs/osrev_ai_reference.yaml code/configs/amend2_2026-09-18.yaml code/configs/base.yaml \
            code/slurm/osrev_ai_reference.sh tests/test_osrev_ai_reference.py; } > "$OUT/provenance.txt"
$PY code/20_osrev_ai_reference_run.py --workers 120 --out "$OUT"
$PY code/21_osrev_ai_reference_analyse.py --dir "$OUT"
echo done; date -Is
