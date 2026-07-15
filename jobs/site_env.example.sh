#!/bin/bash
# jobs/site_env.example.sh — per-site environment TEMPLATE.
#
# Copy this file to jobs/site_env.sh on the cluster and fill in the real
# paths for your account/allocation. jobs/site_env.sh is gitignored — it is
# never committed, since real paths would embed usernames/account strings
# (jobs/README.md hard rule). Every jobs/*.sbatch script sources
# "$REPO_ROOT/jobs/site_env.sh" (REPO_ROOT from $SLURM_SUBMIT_DIR or
# --repo-root), so this file is the ONLY place cluster-specific paths live.
#
# This example mirrors the proven stack layout from the predecessor project
# (vLLM 0.24.0 offline API, CUDA-13 compat shim, offline HF cache). Replace
# every placeholder below with your real allocation's paths.

set -uo pipefail

# Module environment (cluster module system; adjust the module name/version
# to whatever the target cluster provides).
module load python/3.11.7

# Python virtualenv with vLLM installed (kept OUTSIDE the git repo, on
# scratch/work storage -- never commit a venv).
VENV_PATH="/path/to/your/allocation/tools/venv-vllm"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# Offline Hugging Face cache (model weights pre-downloaded; no cluster
# compute node has outbound internet access).
export HF_HOME="/path/to/your/allocation/data/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# CUDA compat shim, only needed if the cluster's driver is older than the
# CUDA version vLLM/torch were built against.
export LD_LIBRARY_PATH="/path/to/your/allocation/tools/cuda-compat/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"

# vLLM sampler backend pin (matches the proven-working configuration; unset
# or flip to 1 if your vLLM build's default FlashInfer sampler is verified
# working on your GPU generation).
export VLLM_USE_FLASHINFER_SAMPLER=0

# OPTIONAL: bitwise run-to-run reproducibility of temperature-0 generation.
# Measured on the M2 smoke test: with default kernels, one of 8 cards
# differed across reruns in its free-text "voice" field only (run-to-run
# numeric jitter flipping a near-tie token under greedy decoding; all
# guided/structured fields were identical). With this flag on, reruns were
# byte-identical even across different physical nodes. Cost: ~30-40% wall
# time (batch-invariant kernels + a fresh torch.compile). The production
# card pass generates once, so leave this off unless regenerate-and-diff
# auditability of the raw card file itself is required.
# export VLLM_BATCH_INVARIANT=1
