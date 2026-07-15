#!/bin/bash
# jobs/gen_persona_cards.sbatch — M2 slow-brain persona-card generation (D7).
#
# Full-node discipline: exactly 4 vLLM shard processes, one per A100
# (CUDA_VISIBLE_DEVICES=0..3), each running serving/batch_gen.py against its
# deterministic 1-of-4 slice of the prompts file. No GPU on the held node
# idles. Waits for all 4, concatenates shard outputs into one cards_raw.jsonl,
# and writes a manifest JSON into runs/.
#
# No account, partition, or qos is hardcoded (jobs/README.md hard rule) --
# those are cluster-specific and belong on the sbatch command line. Example
# (production run, normal QOS, 2h budget already set below):
#
#   sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> \
#          jobs/gen_persona_cards.sbatch \
#          --prompts /path/to/prompts.jsonl \
#          [--model Qwen/Qwen3-8B] [--max-attempts 3] [--max-tokens 1024] \
#          [--run-dir runs/my_label]
#
# Example (debug-queue smoke test, short wall clock, small prompts file):
#
#   sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> \
#          --qos=<DEBUG_QOS> --time=00:20:00 \
#          jobs/gen_persona_cards.sbatch --prompts /path/to/smoke_prompts.jsonl
#
# Site-specific paths (venv, module loads, HF cache, CUDA compat library) are
# NOT in this file. They come from sourcing jobs/site_env.sh, which is
# gitignored. Commit only jobs/site_env.example.sh (placeholder paths); copy
# it to jobs/site_env.sh on the cluster and fill in real paths there.
#
#SBATCH --job-name=gen-persona-cards
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=gen-persona-cards-%j.out

set -uo pipefail

NUM_SHARDS=4  # one per A100; full-node discipline, not configurable here

# NOTE: Slurm executes a COPY of this script from a spool directory
# (/var/spool/slurmd/...), so ${BASH_SOURCE[0]}/$0 do NOT point back at the
# repo. Do not resolve paths from them. Instead, REPO_ROOT comes from
# --repo-root (explicit) or $SLURM_SUBMIT_DIR (the directory `sbatch` was
# invoked from -- set this by `cd`-ing into the repo root before submitting).

# ---- args --------------------------------------------------------------
MODEL="Qwen/Qwen3-8B"
MAX_ATTEMPTS=3
MAX_TOKENS=1024
PROMPTS=""
RUN_DIR=""
REPO_ROOT="${SLURM_SUBMIT_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompts) PROMPTS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    *) echo "unrecognized argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PROMPTS" ]]; then
  echo "FATAL: --prompts is required" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "FATAL: this script must be run under sbatch (SLURM_JOB_ID unset)" >&2
  exit 2
fi
if [[ -z "$REPO_ROOT" ]]; then
  echo "FATAL: REPO_ROOT unresolved. \$SLURM_SUBMIT_DIR was empty and no" >&2
  echo "--repo-root was given. Submit with 'cd <repo>; sbatch jobs/gen_persona_cards.sbatch ...'" >&2
  echo "or pass --repo-root <repo> explicitly." >&2
  exit 2
fi
if [[ ! -f "$REPO_ROOT/jobs/site_env.sh" ]]; then
  echo "FATAL: $REPO_ROOT/jobs/site_env.sh not found (REPO_ROOT=$REPO_ROOT)." >&2
  echo "Copy jobs/site_env.example.sh to jobs/site_env.sh on the cluster first." >&2
  exit 2
fi

# ---- site-specific environment (gitignored; see jobs/site_env.example.sh) --
# shellcheck disable=SC1091
source "$REPO_ROOT/jobs/site_env.sh"

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$REPO_ROOT/runs/$SLURM_JOB_ID"
fi
mkdir -p "$RUN_DIR"

SCHEMA="$REPO_ROOT/grounding/card_schema.json"
SCRIPT="$REPO_ROOT/serving/batch_gen.py"

echo "REPO_ROOT=$REPO_ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "PROMPTS=$PROMPTS"
echo "MODEL=$MODEL NUM_SHARDS=$NUM_SHARDS MAX_ATTEMPTS=$MAX_ATTEMPTS MAX_TOKENS=$MAX_TOKENS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

START_TS=$(date -u +%s)
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

declare -a PIDS
for ((k = 0; k < NUM_SHARDS; k++)); do
  SHARD_OUT="$RUN_DIR/cards_raw.shard$(printf '%02d' "$k").jsonl"
  SHARD_LOG="$RUN_DIR/shard$(printf '%02d' "$k").log"
  CUDA_VISIBLE_DEVICES="$k" python3 "$SCRIPT" \
    --prompts "$PROMPTS" \
    --out "$SHARD_OUT" \
    --schema "$SCHEMA" \
    --shard-index "$k" --num-shards "$NUM_SHARDS" \
    --model "$MODEL" --max-attempts "$MAX_ATTEMPTS" --max-tokens "$MAX_TOKENS" \
    > "$SHARD_LOG" 2>&1 &
  PIDS[$k]=$!
done

FAIL=0
for ((k = 0; k < NUM_SHARDS; k++)); do
  if ! wait "${PIDS[$k]}"; then
    echo "SHARD $k FAILED (pid ${PIDS[$k]}); see $RUN_DIR/shard$(printf '%02d' "$k").log" >&2
    FAIL=1
  fi
done

END_TS=$(date -u +%s)
FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
WALL_SECONDS=$((END_TS - START_TS))

cat "$RUN_DIR"/cards_raw.shard*.jsonl > "$RUN_DIR/cards_raw.jsonl" 2>/dev/null || true

python3 - "$RUN_DIR" "$SLURM_JOB_ID" "$MODEL" "$PROMPTS" "$NUM_SHARDS" \
         "$STARTED_AT" "$FINISHED_AT" "$WALL_SECONDS" <<'PYEOF'
import glob
import hashlib
import json
import re
import socket
import sys

run_dir, job_id, model, prompts_path, num_shards, started_at, finished_at, wall_seconds = sys.argv[1:9]
num_shards = int(num_shards)
wall_seconds = int(wall_seconds)

summary_re = re.compile(r"^BATCH_GEN_SUMMARY (\{.*\})\s*$", re.MULTILINE)

per_shard = []
for k in range(num_shards):
    log_path = f"{run_dir}/shard{k:02d}.log"
    summary = None
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        matches = summary_re.findall(text)
        if matches:
            summary = json.loads(matches[-1])
    except FileNotFoundError:
        pass
    per_shard.append(summary if summary is not None else {"shard_index": k, "error": "no BATCH_GEN_SUMMARY found"})

try:
    with open(prompts_path, "rb") as fh:
        input_sha256 = hashlib.sha256(fh.read()).hexdigest()
except FileNotFoundError:
    input_sha256 = None

def total(key):
    return sum(s.get(key, 0) or 0 for s in per_shard if isinstance(s.get(key, 0), (int, float)))

manifest = {
    "job_id": job_id,
    "node": socket.gethostname(),
    "model": model,
    "prompts_path": prompts_path,
    "input_sha256": input_sha256,
    "num_shards": num_shards,
    "per_shard": per_shard,
    "totals": {
        "n_shard_total": total("n_shard_total"),
        "n_runnable": total("n_runnable"),
        "n_skipped_attempt_budget": total("n_skipped_attempt_budget"),
        "n_gen_ok": total("n_gen_ok"),
        "n_parsed_not_ok": total("n_parsed_not_ok"),
        "n_failed_parse": total("n_failed_parse"),
        "output_tokens_total": total("output_tokens_total"),
    },
    "started_at": started_at,
    "finished_at": finished_at,
    "wall_seconds": wall_seconds,
    "cards_raw_path": f"{run_dir}/cards_raw.jsonl",
}

out_path = f"{run_dir}/manifest.json"
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")

print(f"MANIFEST_WRITTEN {out_path}")
PYEOF

if [[ "$FAIL" -ne 0 ]]; then
  echo "gen_persona_cards: one or more shards failed; manifest reflects partial results" >&2
  exit 1
fi

echo "gen_persona_cards: OK, wall=${WALL_SECONDS}s, manifest=$RUN_DIR/manifest.json"
