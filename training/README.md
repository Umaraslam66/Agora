# training/

Transplanted verbatim (logic-preserving) from the predecessor project's proven LoRA harness,
with all predecessor-specific naming, paths and prompt-construction imports stripped.

- `lora_sft.py` — LoRA SFT (bf16, peft, manual PyTorch loop, completion-only loss) on
  chat-pairs JSONL; default model `Qwen/Qwen3-8B` (pre-registration target). Optional
  class-rebalanced sampling for rare modes.
- `eval_choice.py` — first-token/candidate log-likelihood choice eval: top-1 accuracy,
  per-mode recall, shares, belief/habit probes, fare channel, logit-blend sweeps, probe dumps.

RENDER-PARITY: neither script constructs persona/world prompt text. Prompts arrive
pre-rendered in the pairs JSONL (rendered by `grounding.render`); synthetic probe/fare
text must be injected via `eval_choice.py --renderer <module[:attr]>` backed by that
same render path. Seams are marked `# RENDER-PARITY` in both files.

Run (single GPU per process; full-node discipline = pack N independent runs per node,
or wrap with torchrun as launcher — the scripts carry no in-file DDP, as in the source):

    torchrun --standalone --nproc_per_node=1 training/lora_sft.py \
        --pairs /path/to/pairs.jsonl --out /path/to/adapter --max-steps 200
    python3 training/eval_choice.py --model Qwen/Qwen3-8B --adapter /path/to/adapter \
        --pairs /path/to/pairs.jsonl --split test --out /path/to/metrics.json
