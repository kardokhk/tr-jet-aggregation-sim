#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=osrev-calibration
#SBATCH --output=logs/%x-%j.out
# Protocol amendment 3, package A3-6 (calibration against a matched estimand; post hoc).
# Run from the project root: sbatch --wait code/slurm/osrev_calibration.sh
# STAGES (default "A B C D"): A and B are the run of 2026-10-05; C and D the follow-up of 2026-10-06
# (confidence-interval sensitivity; patient-level geometric draw). The follow-up alone:
#   sbatch --wait --export=ALL,STAGES="C D" code/slurm/osrev_calibration.sh
# It needs matched_values.json and the stage A and B tables in $OUT and leaves them untouched.
set -euo pipefail
OUT=${OUT:-results/2026-10-05_osrev/calibration}
STAGES=${STAGES:-A B C D}
PY=/project/home/p201509/envs/duomax-sim/bin/python
RUN=code/20_osrev_calibration_run.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT" logs
prov() {
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "slurm_job_id: ${SLURM_JOB_ID:-none}"; echo "stages: $STAGES"; $PY --version
  $PY -c "import numpy, scipy, pandas; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__)"
  echo "git_commit: $(git rev-parse HEAD) (new files of this package are untracked; checksums below)"
  echo "seeds: SeedSequence entropy 20261005, spawn keys (76, family, index); see code/configs/osrev_calibration.yaml"
  sha256sum code/lib/duomaxsim/*.py $RUN code/21_osrev_calibration_analyse.py code/configs/osrev_calibration.yaml \
            code/configs/amend2_2026-09-18.yaml code/slurm/osrev_calibration.sh tests/test_osrev_calibration.py
}
case " $STAGES " in
  *" A "*) prov > "$OUT/provenance.txt"; cp code/configs/osrev_calibration.yaml "$OUT/config_used.yaml" ;;
  *) { echo; echo "---- follow-up run (stages $STAGES); stage A and B outputs above are not regenerated ----"; prov; } >> "$OUT/provenance.txt"
     cp code/configs/osrev_calibration.yaml "$OUT/config_used_followup.yaml" ;;
esac
run_stage() {
  local stage=$1
  $PY $RUN --list "$stage" > "$OUT/tasks_$stage.txt"
  echo "stage $stage tasks: $(wc -l < "$OUT/tasks_$stage.txt")"
  xargs -P 126 -I{} bash -c '$0 '"$RUN"' --task "$1" --out '"$OUT"' >> '"$OUT"'/log_stage_'"$stage"'.txt 2>&1 || echo "FAIL $1"' "$PY" {} < "$OUT/tasks_$stage.txt" | tee "$OUT/failures_$stage.txt"
  if [ -s "$OUT/failures_$stage.txt" ]; then echo "stage $stage had failures"; exit 1; fi
  $PY $RUN --merge "$stage" --out "$OUT"
  rm -f "$OUT/tasks_$stage.txt" "$OUT/failures_$stage.txt"
}
for st in $STAGES; do run_stage "$st"; done
rmdir "$OUT/parts" 2>/dev/null || true
case " $STAGES " in *" B "*) $PY code/21_osrev_calibration_analyse.py --out "$OUT" ;; esac
case " $STAGES " in *" D "*) $PY code/21_osrev_calibration_analyse.py --out "$OUT" --followup ;; esac
echo done; date -Is
