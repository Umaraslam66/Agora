#!/bin/bash
# BT2 — THE SINGLE TRANSFER-ARENA BLIND FIRING (A8).
#
# Written 2026-07-19 (held, no stored authorization). GO given 2026-07-20
# on the project owner's explicit inline authorization: "GO.
# AGORA_BT1_AUTHORIZED=1 — fire BT2." The flag below is the owner's,
# carried inline per the sealed firing protocol. Fires ONCE; the scored
# verdict seals permanently against evaluation/truth/transfer_p123 (A8;
# A2.5 E6). The driver refuses a non-empty runs/bt2.
#
# Walltime pre-declaration (owner ruling 2026-07-20, pushed pre-firing in
# runs/bt2_predeclaration.md): a walltime kill BEFORE the scoring phase
# reads any truth quantity leaves no blind quantity in existence;
# resubmission is then permitted and recorded openly. QOS boost_qos_lprod,
# 48h, one full node — timeline NOT re-pinned.
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
#SBATCH --time=48:00:00
#SBATCH --output=bt2-firing-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source jobs/site_env.sh
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Fresh, BT2-dedicated generation cache OUTSIDE the sealed output dir.
mkdir -p runs/bt2_cache

# The owner's inline authorization (go of 2026-07-20).
AGORA_BT1_AUTHORIZED=1 python -m evaluation.run_bt2 \
  --generator vllm \
  --cache runs/bt2_cache/gen_cache.jsonl \
  --tensor-parallel 4 \
  --runs 20 \
  --out runs/bt2

echo "BT2_FIRING_DONE"
