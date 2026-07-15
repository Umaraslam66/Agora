"""serving/batch_gen.py — offline batched persona-card generation driver (D7).

This is the slow brain's ONE card-writing pass, run offline (no HTTP server)
via vLLM's `LLM.chat` batch API. It reads a shard of already-rendered prompts
(produced upstream by `grounding.render.render_seed_prompt` — this module
builds no prompt text of its own, per the render-parity doctrine) and writes
one generated+parsed record per line.

This driver does ONLY generation + parsing. It does not validate cards
against taxonomy/mask-lint/replay-lint — that is a separate later stage
(D7's "validation loop"). A retry pass for cards that fail validation is just
another invocation of this same driver against a new `--prompts` file whose
lines carry an incremented "attempt" and a prompt that embeds the
machine-readable failure block (constructed upstream, not here).

Input (--prompts), one JSON object per line:
    {"persona_id": "...", "prompt": "...", "attempt": 1}

Output (--out), one JSON object per line:
    {"persona_id", "raw_json" (parsed object or null), "finish_reason",
     "gen_ok": bool, "attempt", "model", "prompt_sha256"}

Sharding is deterministic and stateless: line index i (0-based, in file
order) belongs to shard `i % num_shards`. Four shards (one per A100) driven
by jobs/gen_persona_cards.sbatch cover the full node.

Guided decoding: vLLM 0.24 offline structured outputs via
`SamplingParams(structured_outputs=StructuredOutputsParams(json=<schema>))`.
Thinking is disabled through the Qwen3 chat template's own switch, passed as
`chat_template_kwargs={"enable_thinking": False}` to `LLM.chat` (verified
against the installed Qwen3-8B tokenizer_config.json template: it emits an
empty `<think>\n\n</think>\n\n` block when `enable_thinking` is defined and
false). Temperature 0 (greedy) throughout, per D7.

vLLM is imported lazily (inside `_import_vllm`), not at module scope, so the
pure-Python sharding/parsing logic here is importable and unit-testable on a
machine without vLLM or a GPU (see tests/test_batch_gen.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def prompt_sha256(prompt: str) -> str:
    """Stable hash of a rendered prompt string, for provenance/dedup."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_prompts(path) -> list[dict]:
    """Load a prompts.jsonl file into a list of dicts, in file order."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def shard_line_indices(n_total: int, shard_index: int, num_shards: int) -> list[int]:
    """Deterministic shard split: line index i belongs to shard i % num_shards.

    Stateless in n_total: this is the SAME split whether called once with the
    full prompts file or independently per shard process, as long as
    num_shards and each record's original line index agree.
    """
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= shard_index < num_shards):
        raise ValueError("--shard-index must be in [0, num-shards)")
    return [i for i in range(n_total) if i % num_shards == shard_index]


def select_shard(records: list[dict], shard_index: int, num_shards: int) -> list[tuple[int, dict]]:
    """Return (original_line_index, record) pairs belonging to this shard."""
    idxs = shard_line_indices(len(records), shard_index, num_shards)
    return [(i, records[i]) for i in idxs]


def _load_schema(schema_path) -> dict:
    with open(schema_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _import_vllm():
    """Import vLLM lazily so shard-split/parsing logic is testable without it."""
    try:
        from vllm import LLM
        from vllm.sampling_params import SamplingParams, StructuredOutputsParams
    except ImportError as exc:  # pragma: no cover - exercised only on the cluster
        raise RuntimeError(
            "vLLM is required to run generation. Activate the venv with vLLM "
            "installed (this is only needed for the actual --prompts run, not "
            "for shard-split unit tests)."
        ) from exc
    return LLM, SamplingParams, StructuredOutputsParams


def build_output_record(
    persona_id: str,
    prompt: str,
    attempt: int,
    model: str,
    text: str,
    finish_reason: str | None,
    skipped_reason: str | None = None,
) -> dict:
    """Parse one raw generation into the driver's output record shape.

    gen_ok is True only if the text parses as JSON AND the sequence finished
    cleanly (finish_reason == "stop", i.e. not truncated by max_tokens). A
    truncated-but-parseable object (should not happen with guided decoding
    honoring max_tokens exactly, but keep the distinction honest) is surfaced
    as raw_json set / gen_ok False rather than silently dropped.
    """
    raw_json = None
    gen_ok = False
    if skipped_reason is None:
        try:
            candidate = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            candidate = None
        if candidate is not None:
            raw_json = candidate
            gen_ok = finish_reason == "stop"
    return {
        "persona_id": persona_id,
        "raw_json": raw_json,
        "finish_reason": finish_reason if skipped_reason is None else skipped_reason,
        "gen_ok": gen_ok,
        "attempt": attempt,
        "model": model,
        "prompt_sha256": prompt_sha256(prompt),
    }


def run_generation(args: argparse.Namespace) -> dict:
    records = load_prompts(args.prompts)
    shard = select_shard(records, args.shard_index, args.num_shards)

    runnable = [(i, r) for i, r in shard if r.get("attempt", 1) <= args.max_attempts]
    over_budget = [(i, r) for i, r in shard if r.get("attempt", 1) > args.max_attempts]

    output_by_index: dict[int, dict] = {}
    for i, rec in over_budget:
        output_by_index[i] = build_output_record(
            rec["persona_id"],
            rec["prompt"],
            rec.get("attempt", 1),
            args.model,
            text="",
            finish_reason=None,
            skipped_reason="attempt_budget_exceeded",
        )

    n_ok = n_parsed_not_ok = n_failed_parse = 0
    total_output_tokens = 0
    wall_start = time.monotonic()

    if runnable:
        schema = _load_schema(args.schema)
        LLM, SamplingParams, StructuredOutputsParams = _import_vllm()
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
        conversations = [[{"role": "user", "content": rec["prompt"]}] for _, rec in runnable]
        outputs = llm.chat(
            conversations,
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        for (i, rec), out in zip(runnable, outputs):
            completion = out.outputs[0]
            total_output_tokens += len(completion.token_ids) if completion.token_ids else 0
            result = build_output_record(
                rec["persona_id"],
                rec["prompt"],
                rec.get("attempt", 1),
                args.model,
                text=completion.text,
                finish_reason=completion.finish_reason,
            )
            output_by_index[i] = result
            if result["gen_ok"]:
                n_ok += 1
            elif result["raw_json"] is not None:
                n_parsed_not_ok += 1
            else:
                n_failed_parse += 1

    wall_seconds = time.monotonic() - wall_start

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i in sorted(output_by_index):
            fh.write(json.dumps(output_by_index[i], sort_keys=True))
            fh.write("\n")

    summary = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "model": args.model,
        "n_shard_total": len(shard),
        "n_runnable": len(runnable),
        "n_skipped_attempt_budget": len(over_budget),
        "n_gen_ok": n_ok,
        "n_parsed_not_ok": n_parsed_not_ok,
        "n_failed_parse": n_failed_parse,
        "wall_seconds": round(wall_seconds, 3),
        "output_tokens_total": total_output_tokens,
        "tokens_per_sec": round(total_output_tokens / wall_seconds, 2) if wall_seconds > 0 else None,
    }
    # A single machine-parseable line the sbatch wrapper greps out of each
    # shard's log to assemble the run manifest.
    print("BATCH_GEN_SUMMARY " + json.dumps(summary, sort_keys=True))
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True, help="Input prompts.jsonl (persona_id, prompt, attempt per line)")
    p.add_argument("--out", required=True, help="Output cards_raw.jsonl path for this shard")
    p.add_argument("--schema", required=True, help="Path to grounding/card_schema.json (guided-decoding schema)")
    p.add_argument("--shard-index", type=int, required=True, help="This process's shard, 0-based")
    p.add_argument("--num-shards", type=int, required=True, help="Total number of shards (e.g. 4, one per GPU)")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-attempts", type=int, default=3, help="Refuse (no model call) lines with attempt > this")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_generation(args)


if __name__ == "__main__":
    main()
