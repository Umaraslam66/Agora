#!/bin/bash
# BT1 — THE SINGLE BLIND FIRING (M4). Fires ONCE; the scored verdict seals
# permanently against evaluation/truth (pre-registration §2 wall; A1.1).
#
# Written at go time on the project owner's explicit inline authorization,
# 2026-07-18: "GO. AGORA_BT1_AUTHORIZED=1 — fire BT1." The flag below is the
# owner's, carried inline per the sealed firing protocol; this script may
# never be re-submitted — a second run is post-hoc and cannot overwrite the
# verdict (the driver refuses a non-empty runs/bt1).
#
# Full-node discipline: one vLLM engine, tensor-parallel over all 4 GPUs.
# Account/partition/qos passed on the sbatch command line (jobs/ rules):
#
#   sbatch -A <account> -p <partition> jobs/bt1_firing.sbatch
#
#SBATCH --job-name=bt1-firing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=bt1-firing-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source jobs/site_env.sh
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Fresh, BT1-dedicated generation cache OUTSIDE the sealed output dir (the
# driver requires runs/bt1 to start empty — single firing). Not seeded from
# the rehearsal cache: every BT1 generation is fresh from the frozen model.
mkdir -p runs/bt1_cache

# The owner's inline authorization. All arms, tiers, thresholds and
# ablations are scored in this ONE invocation; --runs/--e6-runs per the
# owner's driver-review ruling (c).
AGORA_BT1_AUTHORIZED=1 python -m evaluation.run_bt1 \
  --generator vllm \
  --cache runs/bt1_cache/gen_cache.jsonl \
  --tensor-parallel 4 \
  --runs 20 \
  --e6-runs 20 \
  --out runs/bt1

echo "BT1_FIRING_DONE"
