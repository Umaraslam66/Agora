#!/bin/bash
# E5(ii) fictional-price contamination probe (sealed A2.3(ii)) through the
# deployed announced-onset pipeline at the FROZEN fitted point. Full-node
# discipline: one vLLM engine, tensor-parallel over all 4 GPUs. Run AFTER
# the A4.3 fit manifest freezes (the probe reads theta/VoT from it).
#
#   sbatch -A <account> -p <partition> [--qos <qos>] jobs/e5_price_probe.sbatch
#
#SBATCH --job-name=e5-price-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=e5-price-probe-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source jobs/site_env.sh
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT=runs/e5_price_probe
mkdir -p "$OUT"
# prompt-keyed cache seeded from the fit run (probe multiplier 1.0 shares
# the config-schedule onset prompts only if they were generated; all other
# multipliers are fresh prompts)
[ -f "$OUT/gen_cache.jsonl" ] || cp runs/sr520_fit_sr520sched/gen_cache.jsonl "$OUT/gen_cache.jsonl"

python -m evaluation.run_e5_price_probe \
  --cards data/cards/cards_m2_masked_r2b.jsonl \
  --m4-gates runs/m4_prep \
  --context runs/sr520_fit/context_corridor.json \
  --generator vllm \
  --cache "$OUT/gen_cache.jsonl" \
  --tensor-parallel 4 \
  --manifest calibration/sr520_fit_manifest.json \
  --out "$OUT"

echo "E5_PRICE_PROBE_DONE"
