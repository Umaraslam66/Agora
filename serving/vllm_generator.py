"""In-process cached vLLM rewrite generator (the SR 520 fit / M4 shock loop's
live slow-brain seam).

The M3 scored path ships rewrite prompts to the cluster as files
(jobs/rewrite_roundtrip + serving/batch_gen). The A4.3 calibration sweep
cannot: fitting the strong-habit threshold and the VoT/elasticity dial means
re-running the shock loop many times, each with its own onset/tail rewrite
batches — a file round trip per batch per sweep point is unworkable. This
module is the live equivalent: a generator for
``agents.slow_brain.GatedSlowBrain`` that calls vLLM in-process with the
IDENTICAL generation contract as ``serving/batch_gen`` (guided-JSON decoding
against the frozen card schema, temperature 0, ``enable_thinking: False``,
same defaults), plus a persistent prompt-sha cache:

* temperature-0 decoding is a pure function of the prompt, so identical
  prompts across ensemble runs, arms (toll/placebo), and sweep points are
  generated ONCE — the cache is what makes the sweep affordable;
* the cache file (JSONL: prompt_sha256, text, finish_reason, model) makes
  every generation auditable and lets a CPU-side process replay a sweep
  without GPUs (``offline=True`` raises on any cache miss instead of loading
  the model).

RENDER-PARITY: this module never builds prompt text; it receives prompts the
slow brain rendered through grounding.render.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from serving.batch_gen import _import_vllm, _load_schema, prompt_sha256

#: Generation contract mirrored from serving/batch_gen (its CLI defaults).
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85


class CachedRewriteGenerator:
    """Generator seam for ``GatedSlowBrain`` (called as ``gen(requests,
    prompts)``): returns one raw completion text per request, from the cache
    when the byte-identical prompt was generated before, else from vLLM."""

    def __init__(
        self,
        cache_path: str,
        schema_path: str = "grounding/card_schema.json",
        model: str = DEFAULT_MODEL,
        tensor_parallel_size: int = 4,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        offline: bool = False,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.schema_path = schema_path
        self.model = model
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.max_tokens = int(max_tokens)
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.offline = bool(offline)
        self._llm = None
        self._sampling_params = None
        self._cache: Dict[str, dict] = {}
        self.n_hits = 0
        self.n_generated = 0
        if self.cache_path.exists():
            with self.cache_path.open() as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._cache[rec["prompt_sha256"]] = rec

    # -- vLLM lifecycle ----------------------------------------------------

    def _ensure_llm(self):
        if self._llm is None:
            LLM, SamplingParams, StructuredOutputsParams = _import_vllm()
            schema = _load_schema(self.schema_path)
            self._llm = LLM(
                model=self.model,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                seed=self.seed,
            )
            self._sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                structured_outputs=StructuredOutputsParams(json=schema),
            )
        return self._llm

    # -- the generator seam ------------------------------------------------

    def __call__(self, requests: Sequence, prompts: Sequence[str]) -> List[str]:
        shas = [prompt_sha256(p) for p in prompts]
        missing: List[int] = [i for i, s in enumerate(shas) if s not in self._cache]
        if missing:
            if self.offline:
                raise RuntimeError(
                    f"{len(missing)} prompt(s) not in cache {self.cache_path} "
                    "and offline=True (no GPU generation allowed here)"
                )
            llm = self._ensure_llm()
            conversations = [
                [{"role": "user", "content": prompts[i]}] for i in missing
            ]
            outputs = llm.chat(
                conversations,
                sampling_params=self._sampling_params,
                chat_template_kwargs={"enable_thinking": False},
                use_tqdm=False,
            )
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                for i, out in zip(missing, outputs):
                    completion = out.outputs[0]
                    rec = {
                        "prompt_sha256": shas[i],
                        "text": completion.text,
                        "finish_reason": completion.finish_reason,
                        "model": self.model,
                    }
                    self._cache[shas[i]] = rec
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                    self.n_generated += 1
        self.n_hits += len(prompts) - len(missing)
        return [self._cache[s]["text"] for s in shas]

    def stats(self) -> dict:
        return {
            "cache_path": str(self.cache_path),
            "cache_size": len(self._cache),
            "n_hits": self.n_hits,
            "n_generated": self.n_generated,
        }
