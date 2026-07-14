# serving/ — discrete-choice gateway

`gateway.py` is an OpenAI-compatible HTTP proxy that turns a chat-completions
call into an auditable discrete choice. It scores each candidate mode by its
**first-token logprob** (one forced-decode step, `top_logprobs`), optionally
**blends** two models' scores (`base + lambda*(adapter-base)`, scalar or
per-mode lambda, plus a per-mode bias), picks by argmax (T=0) or softmax (T>0).
Sampling is seeded by a **common-random-numbers key** (the OpenAI `user`
field, contract `<agentId>:<simDay>:<tripIndex>`), so the same agent-day
draws identically across counterfactual twin worlds. Every choice appends a
JSONL audit record of per-mode scores. Provenance: transplanted from a
predecessor project's gateway; algorithmic behavior (choice extraction,
blend, CRN) is preserved exactly.

Callers: when the fast/slow brain loop routes a decision to the LLM, it POSTs
an already-rendered prompt (produced by `grounding.render` — serving/ builds
no prompt text) with candidates in `guided_choice`/`structured_outputs`;
slow-brain reflection calls pass through untouched. Upstream is a locally
served vLLM instance (default model: Qwen3-8B); `--backend logit` serves a
calibrated MNL chooser GPU-free. Self-test: `python3 gateway.py --self-test`.
