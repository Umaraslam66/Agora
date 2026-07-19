#!/bin/bash
# BT2 — THE SINGLE TRANSFER-ARENA BLIND FIRING (A8). HELD.
#
# WRITTEN, NEVER RUN (build order of 2026-07-19). This script deliberately
# does NOT carry AGORA_BT1_AUTHORIZED: submitted as-is it stops at the
# driver's refusal. At go time the OWNER edits the launch line to carry the
# inline flag (the BT1 protocol: authorization is typed at go, never stored
# ahead of it), fires ONCE, and the scored verdict seals permanently
# against evaluation/truth/transfer_p123 (A8; A2.5 E6). The driver refuses
# a non-empty runs/bt2 — there is no re-submission.
#
# Prerequisites verified by the driver at load (fail before any run):
# transfer population + adopted-weights sha pin, schedule manifest ==
# world constants, protocol constants match, borrowed-car ruling block,
# and the A8.3 P0 floor rehearsal's results on disk (reported standalone
# first).
#
# Full-node discipline: one vLLM engine, tensor-parallel over all 4 GPUs.
#
#   sbatch -A <account> -p <partition> jobs/bt2_firing.sbatch
#
#SBATCH --job-name=bt2-firing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=bt2-firing-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source jobs/site_env.sh
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Fresh, BT2-dedicated generation cache OUTSIDE the sealed output dir.
mkdir -p runs/bt2_cache

# OWNER: at go, prefix the next line with the inline authorization.
python -m evaluation.run_bt2 \
  --generator vllm \
  --cache runs/bt2_cache/gen_cache.jsonl \
  --tensor-parallel 4 \
  --runs 20 \
  --out runs/bt2

echo "BT2_FIRING_DONE"
