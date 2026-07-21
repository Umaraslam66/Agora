# Agora

Agora is a domain-agnostic **method** for simulating human populations with LLM
agents — and a blind test of whether it works. Every agent is seeded from one
real anonymous travel-diary record (never a segment average), lives in a masked
world under real time/money consequences, and is scored **blind** against a
famous natural experiment it is never allowed to recognize: the world is an
anonymized "City K", zones are codes, dates are shifted, prices perturbed. The
method's pillars: **say-do calibration** (agents answer hypotheticals in survey
mode and act under consequences in life mode; the gap is measured, corrected
once, frozen, and tested for transfer), a **two-brain architecture** (a slow
LLM brain writes compact persona cards of rules; a fast plain-code brain
executes daily choices, with per-rule habit-strength counters as the memory
substrate), and a **pre-registered evaluation with a hard calibration/validation
wall** — the blind periods are scored once and sealed, whatever they say.

## The five layers

1. **Grounding** — each agent seeded from one real individual diary record:
   skeleton (demographics, zones, car ownership) + habits (observed trip
   patterns) + voice (published qualitative material).
2. **Population / variance** — human spread is inherited from record-level
   seeding, not injected; measured, not assumed.
3. **Decision (two-brain)** — slow brain (LLM) writes/rewrites the persona
   card at initialization and on surprises; fast brain (plain code) executes
   daily choices by following the card. Habit counters make old rules resist
   reversal — the memory mechanism under test.
4. **Say-do calibration** — the stated-vs-revealed gap is measured on small
   calibration events, corrected once, frozen, and tested for transfer to the
   policy shock.
5. **Evaluation** — distributional metrics only, ensemble uncertainty with
   coverage checks, contamination masking, and a strict wall between
   calibration and blind periods. Governed by the pre-registration.

## Repo map

```
01_PREREGISTRATION.md   frozen pre-registration — outranks everything
world/                  zones, congestion feedback, toll cordon, GTFS ingest
agents/                 persona cards, fast brain, slow-brain client
grounding/              diary adapters, seeding, render path, masking + mask-lint
calibration/            say-do measurement + frozen correction
evaluation/             E1–E7 harness; truth/ is quarantined (import-boundary test)
serving/                choice gateway, model serving configs
training/               LoRA SFT + choice eval (transplants)
jobs/                   cluster job files — full-node discipline, no usernames
tests/                  day-one doctrine tests: render-parity, mask-lint, truth boundary
docs/                   gate + decision records — private, published with results
data/synthetic/         schema-identical synthetic stand-in (DEV only, never cited)
```

## Gate & decision records

Gate records and decision records — the `docs/…` files the frozen
`01_PREREGISTRATION.md` cites (e.g. `docs/M2_GATE_RECORD.md`,
`docs/DECISION_M4_HAS_PASS_GATE.md`) — are **maintained privately and published
with the release.** Every sealed blind quantity is reproducible from this
public repo's `runs/` manifests (`runs/bt1/`, `runs/bt2/`, `runs/e7_ess/`;
results SHA-256 pinned at firing).

## Status & results (both blind arenas SPENT; verdicts sealed)

**BT1 — Seattle SR 99 tunnel tolling (fired once 2026-07-19, sealed): PASS.**
The placebo-corrected blind prediction ΔQ = 0.2752, 80% [0.2673, 0.2864],
covers the observed −28% and beats both the official −45% forecast and the
frozen no-LLM comparator (0.3096 [0.2980, 0.3212], non-covering). Per the
sealed mechanism amendment (A7), every summary carries this verbatim: *"the
two-brain method beat the aggregate comparator through calibration transfer —
the adaptation channel's presence at fit time moved the dial to the right
place — not through runtime persona intelligence."* ≥98.6% of the response
ran through the calibrated route dial; card rewrites carried ≈1%.

**BT2 — Stockholm congestion charge, METHOD-TRANSFER (fired once 2026-07-20,
sealed): NULL — the method does not transfer.** In the cordon arena (no route
dial by construction; the card channel isolated) the blind response is three
orders of magnitude below every bar (P1 −0.00054 vs target 0.21; E6 residual
0.000176 vs band [0.04, 0.12] at every habit threshold; memory not
load-bearing), and the null is clean — drift control green in all phases.
**Negative results are product**; the falsification stands sealed beside the
pass with equal prominence.

**E7 information value (non-blind, protocol pinned pre-scoring,
`runs/e7_ess/`):** persona cards without behavioral evidence are worth ≤10
real diary records at individual-level prediction; an observed day's value is
almost entirely replay, not generalization (held-out ESS ≲10); and a
deterministic template card built from the same diary strictly outperforms the
LLM-written card (paired +0.095 [0.092, 0.098]).

## The paper

The results are written up as an arXiv-style article: **"Blind,
Pre-Registered Validation of LLM Population Simulation: A Pass, Its
Mechanism, and a Falsification"** — v1.0, sealed by the owner 2026-07-21
(source in `docs/paper/`, PDF at `outputs/agora_paper_v1.0.pdf`). It covers
the five-layer method, the pre-registration harness (amendments A1–A8, the
wall, the placebo doctrine), both sealed verdicts, the mechanism audit, and
the individual-level information-value analysis.

## Pre-registration

**`01_PREREGISTRATION.md` is frozen and outranks everything in this repository,
including this README.** Metrics and decision rules were fixed before any agent
ran. Numeric bars marked [M0] are set once, at the end of the data audit, via a
single dated amendment. All other changes are append-only, dated amendments —
the original text is never rewritten. Blind-test results are scored once and
sealed whatever they say; re-runs after seeing a blind score are labeled
post-hoc and cannot overwrite the sealed verdict.

## Doctrine enforced in CI, from day one

- **Render parity** — exactly one function renders any persona/world state for
  both training and serving (`grounding/render.py`). A second render path is a
  test failure, not a style issue.
- **Mask-lint** — a versioned forbidden-token list; any real-world name, date,
  or price in agent-visible text fails the build.
- **Truth import boundary** — `evaluation/truth/` cannot be imported by agent
  or serving code (static scan + runtime tripwire).

## Provenance

The core machinery (choice gateway with first-token-logit blending, LoRA SFT
and choice-eval harness, bootstrap/permutation statistics, calibrated-MNL
falsification arm, adapter schema) is transplanted from a predecessor
Berlin/MATSim project whose **sealed negative results** motivated this design:
narrative memory that never reached decisions, a native price response an
order of magnitude too weak, a habit channel that failed its gate twice, and a
train/serve rendering confound that silently flipped a verdict. Each of those
failures is answered here by a pre-registered eval or a day-one test. Negative
results are product in this repo too.

**Working-name history:** the project brief uses the working name *SAGA*
("say-do aligned generative agents"); the project is named **Agora**.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright The Agora Authors.
