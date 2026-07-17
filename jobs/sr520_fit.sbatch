#!/bin/bash
# SR 520 joint calibration (A4.3) + rehearsal/drift-floor (A3.3(b), A4.2).
# Full-node discipline: one vLLM engine, tensor-parallel over all 4 GPUs; the
# loop drives it with onset/tail rewrite batches while the CPU side runs the
# fit sweep. Account/partition/qos are passed on the sbatch command line, not
# hardcoded (jobs/ rules).
#
#   sbatch -A <account> -p <partition> [--qos <qos>] jobs/sr520_fit.sbatch
#
#SBATCH --job-name=sr520-fit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=sr520-fit-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source jobs/site_env.sh
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m calibration.sr520_fit \
  --cards data/cards/cards_m2_masked_r2b.jsonl \
  --m4-gates runs/m4_prep \
  --context runs/sr520_fit/context_corridor.json \
  --generator vllm \
  --cache runs/sr520_fit/gen_cache.jsonl \
  --tensor-parallel 4 \
  --out runs/sr520_fit \
  --freeze

echo "SR520_FIT_DONE"
