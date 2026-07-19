#!/bin/bash
# A8.3 P0 placebo-only arena-floor rehearsal (transfer arena). BLIND-SAFE:
# the charge is never on (the driver is structurally incapable of building
# a priced announcement); the nulled-notice trigger fires at the real
# transition offsets; N=20 CRN members. One pre-firing full-node run
# (A8.7(iv)); the floor is reported standalone BEFORE the BT2 firing.
#
#   sbatch -A <account> -p <partition> jobs/p0_floor.sbatch
#
#SBATCH --job-name=p0-floor
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=p0-floor-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source jobs/site_env.sh
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p runs/p0_floor_cache

python -m evaluation.run_p0_floor \
  --generator vllm \
  --cache runs/p0_floor_cache/gen_cache.jsonl \
  --tensor-parallel 4 \
  --members 20 \
  --out runs/p0_floor

echo "P0_FLOOR_DONE"
