#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=osrev-fig4-clustered
#SBATCH --output=logs/%x-%j.out
# Patient-clustered standard errors for Figure 4 (panels c, d): six stored E4 cells re-simulated with
# their original seeds, 30 chunk tasks of 20,000 patients. Run from the project root:
#   sbatch --wait code/slurm/osrev_fig4_clustered.sh
set -euo pipefail
OUT=results/2026-10-05_osrev/fig4_clustered
PY=/project/home/p201509/envs/duomax-sim/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
{ echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "slurm_job_id: ${SLURM_JOB_ID:-none}";
  echo "python: $($PY --version 2>&1)";
  $PY -c 'import numpy, scipy, pandas, yaml; print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "pyyaml", yaml.__version__)';
  echo "git_commit: $(git rev-parse HEAD) (third-amendment files are untracked at run time)";
  echo "sha256:";
  sha256sum code/lib/duomaxsim/config.py code/lib/duomaxsim/model.py code/lib/duomaxsim/rules.py \
            code/lib/duomaxsim/experiments.py code/lib/duomaxsim/osrev_triggers.py \
            code/23_osrev_fig4_clustered_se.py code/configs/osrev_triggers.yaml code/configs/base.yaml \
            code/slurm/osrev_fig4_clustered.sh results/2026-09-18_full/analysis/E4_operating.csv; } > "$OUT/provenance.txt"
date -Is
$PY code/23_osrev_fig4_clustered_se.py --workers 30 --out "$OUT" | tee -a "$OUT/provenance.txt"
date -Is
