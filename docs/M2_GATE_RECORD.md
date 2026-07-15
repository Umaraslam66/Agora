# M2 gate record — grounding milestone (E1, E2, E5(i))

*Dated 2026-07-15. Sealed record per the pre-registration build order
("each milestone ends with a scored eval"). Every number below is
reproducible from a tracked manifest in `runs/`; the scoring protocols are
the sealed §7 Amendment A2 texts, unmodified.*

## Verdicts against the sealed bars — final card set (r2b)

| Eval | Sealed criterion (A2) | Result | Verdict |
|---|---|---|---|
| E1 pooled | ensemble-mean pooled TVD ≤ 0.10 | **0.0617** | **PASS** |
| E1 cells | every protected cell ≤ 0.20 (ten cells, guard merge) | worst `car0\|remainder` 0.1352 | **PASS** |
| E1 falsification | paired-bootstrap 95% CI of (TVD_method − TVD_MNL) entirely below +0.00655 | **[−0.0508, −0.0419]**; MNL absolute 0.1125 | **PASS — method beats MNL** |
| E2(i) | spread ratios ∈ [0.8, 1.2], three dimensions | trips/day **0.988**, car-share 1.002, ride-share 1.000 | **PASS** |
| E2(ii) | mean pairwise error correlation ≤ 0.20 (realized choices) | 0.0002 | **PASS** |
| E5(i) | flag iff unmasked improves > 25% relative (paired CRN) | **2.7%** relative improvement | **NO FLAG** |

Population: 11,940 personas, 1:1 from the pooled two-wave seeding records;
final composition 8,496 LLM-written cards (71.2%) + 3,444 deterministic
fallback cards (28.8%). Residence banding via the committed tract→zone map
(A2.6 re-pin, `docs/M2_RING_REPIN.md`). Ensemble N=20, base seed 20260717,
seed derivations in each run manifest.

## Iteration history (all preserved, nothing overwritten)

E1/E2 at M2 are calibration-side milestone evals, not blind tests; the
pipeline was revised twice on calibration data only, and both failing
rounds are part of this record:

1. **Round 1 — FAIL** (`runs/e1_m2_round1`, `runs/e2_m2_round1`): E1
   pooled 0.139; the MNL arm BEAT the method (Δ CI [+0.019, +0.066]);
   E2 trips/day spread 0.60. Measured causes: slow-brain compression
   biases — mean trips/day −24.7%, between-person variance ratio 0.415,
   mode-diversity distortion up to 16.7pp, and an unsupported quiet-day
   pattern in 97.8% of cards (vs 16.6% of persons with an observed
   quiet weekday).
2. **Fix 1**: quantitative FIT CHECK rewrite of the seed template + a
   fifth per-card fidelity gate (each card checked against its own
   person's observed trips/day, mode mix, quiet share; numeric
   corrective feedback in retry prompts). Commit 9e4dacc.
3. **Round 2** (`runs/e1_m2_round2`, `runs/e2_m2_round2`): E1 PASS
   (0.0662, beats MNL); E2 trips/day spread 0.767 — still short.
   Decomposition: accepted LLM cards nearly faithful (variance ratio
   0.900, corr 0.994); the residual shrinkage sat in the fallback
   subpopulation (variance ratio 0.585), where the fidelity gate had
   concentrated heavy-trip persons and the fallback's 8-trip truncation
   + fold-into-modal compressed them.
4. **Fix 2** (harness-side only, no regeneration; commit e1e3bd7):
   fallback keeps heavy days whole under a relaxed 16-trip cap (the
   committed schema remains the LLM contract at 8) and the
   anti-enumeration fold targets the closest-trip-count signature.
5. **Round 2b — PASS** (`runs/e1_m2_round2b`, `runs/e2_m2_round2b`):
   the headline table above.

## Generation record

Qwen3-8B (pre-registration §5), vLLM 0.24 structured JSON decoding,
temperature 0, full-node 4-shard jobs on the allocated cluster; ~31,000
guided generations across all rounds with ZERO schema violations and zero
mask-lint hits on LLM-owned content. Attempt policy: two attempts with
numeric fidelity feedback (round-2 retry recovery 43%; temperature-0
retries saturate), then deterministic fallback. Manifests:
`runs/m2_cards/` (round 1), `runs/m2_cards_r2/` (round 2),
`runs/m2_cards_r2b_manifest.json` (fallback rebuild),
`runs/m2_cards_unmasked/` (E5 unmasked arm). Card sets are record-derived
and are never committed (gitignored), per the standing privacy rule.

## E5(i) detail

Unmasked arm generated with the identical final pipeline and CRN
namespaces, real vocabulary substituted at the data level through the one
render path (quarantined `evaluation/truth/unmasked_vocabulary.py`).
Probe per sealed A2.3(i): both arms scored once against the same fixed
reference; relative improvement 2.7% (threshold 25%) → no contamination
flag. Two probe-adjacent observations recorded without interpretation:
(i) the unmasked arm was measurably WORSE at numeric fidelity at identical
gates (987 more retries, 446 more terminal fallbacks); (ii) 0.6% of
unmasked LLM cards carry real system vocabulary in their free-text voice.
20.8% of the population is byte-identical across arms (overlapping
deterministic fallbacks) and dilutes the probe conservatively.

## Findings and open items (honest, carried forward)

- **Fallback share 28.8%** is the price of the per-card fidelity gate at
  temperature 0 with two attempts: the LLM-authorship claim at M2 covers
  71.2% of the population; the fallback is a deterministic empirical
  compression of the same records, and the paired E1 falsification uses
  the population as deployed. Reported, not hidden.
- **FIT-CHECK gaming**: 6.8% of accepted LLM cards are duplicate-pattern
  cards (identical patterns at different weights) — numerically faithful,
  but they zero within-person day-to-day variance. E2(i) (between-person)
  passes regardless; within-person variance realism matters from M3 on.
- **Near-enumeration**: 8.7% of multi-day persons' cards tile their
  observed sequences (soft check, passed the hard replay gate); an
  expected side-effect of fidelity pressure. Flagged for the owner.
- **Borrowed-car distortion**: the executor's hard car→ride availability
  gate cannot reproduce the diary's borrowed-car driving in zero-vehicle
  households (7.3% of car0 trip weight); this is the binding contributor
  to the guard cell's 0.135 TVD, inside the cap. A dated decision before
  M4 may revisit the gate.
- **Structural ceiling**: persons with observed mean > ~9.4 trips/weekday
  cannot pass the fidelity gate with a schema-valid LLM card (8-trip
  pattern cap) and land in fallback by construction.
- **Child personas** carry CRN-drawn toll passes (`has_pass` is not
  age-gated); cosmetic at M2, must be age/licence-gated before toll
  exposure at M4.
- E5(i) runs continuously from here on, per §3.

## Reproduce

```bash
.venv/bin/python -m evaluation.run_e1 --cards data/cards/cards_m2_masked_r2b.jsonl \
  --out runs/e1_check --runs 20 --bootstrap 500 --seed 20260717
.venv/bin/python -m evaluation.run_e2 --cards data/cards/cards_m2_masked_r2b.jsonl \
  --out runs/e2_check --runs 20 --seed 20260717
AGORA_EVAL_CONTEXT=1 .venv/bin/python -m evaluation.run_e5 \
  --masked-cards data/cards/cards_m2_masked_r2b.jsonl \
  --unmasked-cards data/cards/cards_m2_unmasked_r2b.jsonl \
  --out runs/e5_check --runs 20 --seed 20260717
```
