#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=duomax-sens
#SBATCH --output=logs/%x-%j.out
# EXPLORATORY post hoc sensitivity grid (config sens_2026-09-18). Run from the project root.
set -euo pipefail
OUT=${OUT:-results/2026-09-18_sens}
CFG=code/configs/sens_2026-09-18.yaml
PY=/project/home/p201509/envs/duomax-sim/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
# provenance: no git repository yet, so record checksums of code, config and tests
{ date -Is; hostname; $PY --version; sha256sum code/lib/duomaxsim/*.py code/01_run_experiment.py $CFG tests/*.py; } > "$OUT/provenance.txt"
cp $CFG "$OUT/config_used.yaml"
tasks=$(mktemp)
chunk() { local e=$1 n=$2 size=$3; for ((i=0;i<n;i+=size)); do j=$((i+size>n?n:i+size)); echo "$e $i:$j"; done; }
{ chunk E1 384 4; chunk E2 360 12; chunk E5 24 1; } > "$tasks"
echo "tasks: $(wc -l < "$tasks")"
xargs -P 120 -L 1 bash -c '$0 code/01_run_experiment.py --experiment $1 --config '"$CFG"' --cells $2 --out '"$OUT"' > '"$OUT"'/log_$1_${2/:/-}.txt 2>&1 || echo FAIL $1 $2' "$PY" < "$tasks"
echo done; date -Is
