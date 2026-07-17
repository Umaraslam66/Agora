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
evaluation/             E1–E6 harness; truth/ is quarantined (import-boundary test)
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
with the results.** The blind test has not yet fired, so they are not in this
public repo; the pre-registration's references to them resolve on release.

## Status

Pre-M1. **M0 (data audit) is in progress**: acquiring/inspecting data sources,
building the synthetic stand-in, and freezing the numeric bars in the
pre-registration before any agent runs. No results exist yet; per project
rules, no number appears in this README unless it is reproducible from a file
in `runs/`.

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
