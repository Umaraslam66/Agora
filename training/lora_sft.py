#!/usr/bin/env python3
"""LoRA SFT on mode-choice pairs (chat-pairs JSONL: one
{"messages": [...], "assistant": "<mode>", "meta": {"split": "train"|...}}
object per line).

Deliberately a manual PyTorch loop + peft, NOT TRL/Trainer: fewer moving
parts and no fast-moving trainer APIs to break on an offline cluster.
bf16 LoRA on Qwen3-8B fits one A100-64GB without quantization.

Loss is completion-only: prompt tokens are masked to -100; only the
assistant answer (the mode word) + EOS contribute.

  python3 lora_sft.py --pairs pairs.jsonl --model Qwen/Qwen3-8B \
      --out /path/adapter --max-steps 200

Full-node GPU discipline (00_PROJECT_BRIEF.md: "All GPU work uses full
allocated nodes"): this script is single-process/single-GPU by design,
matching the source transplant -- no torch.distributed/DDP code existed
in the original and none has been added here (adding it would be a
rewrite, not a transplant). Full-node use is a job-submission concern:
launch one process per GPU on the node (packed parallel single-GPU
runs, e.g. a hyperparameter/seed sweep) or wrap invocation with
`torchrun --nproc_per_node=N` if/when this script grows real DDP
support. See jobs/ (not yet written) for the orchestration layer.
"""
import argparse
import json
import math
import random
import sys

import torch
from torch.utils.data import DataLoader, Dataset


def load_model_bf16(name):
    """AutoModelForCausalLM first; VL-class archs (e.g. Qwen3.5's
    *ForConditionalGeneration) may only be reachable via the image-text-to-
    text auto class in transformers 5 — fall back and say so."""
    import transformers
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            try:
                model = cls.from_pretrained(name, dtype=torch.bfloat16)
            except TypeError:  # older transformers spell it torch_dtype
                model = cls.from_pretrained(name, torch_dtype=torch.bfloat16)
            print(f"[load] {name} via {cls_name}")
            return model
        except ValueError as e:  # wrong auto-class for this config
            print(f"[load] {cls_name} rejected {name}: {str(e)[:120]}")
    raise RuntimeError(f"no auto class could load {name}")


# RENDER-PARITY: prompt text must originate from grounding.render. This
# function (and encode_pair below) never constructs prompt text itself --
# it only tokenizes `pair["messages"]`, which must already be rendered
# upstream (the pairs JSONL is expected to be produced by the same single
# render path used at serve time; see 00_PROJECT_BRIEF.md's "render-parity
# is a day-one test"). This file deliberately does not import
# grounding.render; it only consumes its output via the pairs file.
def chat_prompt_ids(tok, messages):
    """Version-proof prompt encoding: render the chat template to a string
    (stable across transformers 4/5 — tokenize=True returns a dict in v5)
    and tokenize explicitly, without re-adding special tokens."""
    text = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def encode_pair(tok, pair, max_len):
    prompt_ids = chat_prompt_ids(tok, pair["messages"])
    ans_ids = tok(pair["assistant"], add_special_tokens=False)["input_ids"]
    ans_ids = ans_ids + [tok.eos_token_id]
    ids = list(prompt_ids) + ans_ids
    labels = [-100] * len(prompt_ids) + ans_ids
    if len(ids) > max_len:  # keep the tail; the answer must survive
        ids, labels = ids[-max_len:], labels[-max_len:]
    return ids, labels


class PairsDataset(Dataset):
    def __init__(self, tok, pairs, max_len):
        self.rows = [encode_pair(tok, p, max_len) for p in pairs]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collate(pad_id):
    def collate(batch):
        width = max(len(ids) for ids, _ in batch)
        input_ids, labels, attn = [], [], []
        for ids, lab in batch:
            pad = width - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return (torch.tensor(input_ids), torch.tensor(attn),
                torch.tensor(labels))
    return collate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--target-modules",
                    default="q_proj,k_proj,v_proj,o_proj,"
                            "gate_proj,up_proj,down_proj",
                    help="comma-separated Linear suffixes to adapt")
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--balance", choices=["none", "sqrt_inv", "inv"],
                    default="none",
                    help="class-rebalanced sampling over answer labels: "
                         "inv ~ 1/freq, sqrt_inv ~ 1/sqrt(freq) (softer - "
                         "tiny classes repeat fewer times, less risk of "
                         "memorizing their few phrasings)")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pairs = [json.loads(l) for l in open(args.pairs)]
    train = [p for p in pairs if p["meta"]["split"] == "train"]
    print(f"[sft] {len(train)} train pairs of {len(pairs)} total")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = load_model_bf16(args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=[m.strip() for m in args.target_modules.split(",")],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.cuda().train()

    ds = PairsDataset(tok, train, args.max_len)
    if args.balance == "none":
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=make_collate(tok.pad_token_id), drop_last=True)
    else:
        # Rebalanced sampling for rare modes (e.g. transit/bike trips can
        # be under 5% of a travel-survey sample -- plain shuffling then
        # shows the model a rare-mode example only once every N steps).
        # Weights are per-SAMPLE, indexed in ds order (= train list order).
        from collections import Counter
        counts = Counter(p["assistant"] for p in train)
        power = 0.5 if args.balance == "sqrt_inv" else 1.0
        weights = [1.0 / (counts[p["assistant"]] ** power) for p in train]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(train), replacement=True,
            generator=torch.Generator().manual_seed(args.seed))
        dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=make_collate(tok.pad_token_id), drop_last=True)
        print(f"[sft] balance={args.balance} label counts={dict(counts)}")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10)  # short warmup, then constant
    )

    step = micro = 0
    running = 0.0
    while step < args.max_steps:
        for input_ids, attn, labels in dl:
            out = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
                        labels=labels.cuda())
            (out.loss / args.grad_accum).backward()
            running += out.loss.item()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == args.max_steps:
                    print(f"[sft] step {step}/{args.max_steps} "
                          f"loss {running / (10 * args.grad_accum):.4f}",
                          flush=True)
                    running = 0.0
                if step >= args.max_steps:
                    break

    model.save_pretrained(args.out)  # adapter only (a few tens of MB)
    tok.save_pretrained(args.out)
    print(f"[sft] adapter saved to {args.out}")
    print("LORA_SFT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
