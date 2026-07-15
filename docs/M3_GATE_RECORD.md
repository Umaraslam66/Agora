# M3 gate record — baseline dynamics (two-brain loop; E1/E2 rescored end-to-end; E5(i) continues)

*Dated 2026-07-15. Sealed record per the pre-registration build order.
Every number below is reproducible from a tracked manifest in `runs/`;
scoring protocols are the sealed §7 Amendment A2 texts, unmodified. M3
adds NO new sealed bars: E1/E2 verdicts are against the A2 bars; all
other quantities here are REPORTED diagnostics or machinery records.*

## What M3 is

The ordinary-day two-brain loop, live end-to-end for the first time:
the fast brain (`agents/card_executor.py`) lives every persona through
warm-up plus scoring days with habit counters advancing per lived day;
realized corridor travel times come from the world's congestion feedback
through the new card→world bridge (`world/bridge.py` — no such coupling
existed at M2, where E1/E2 scored static card executions); prediction
errors feed the ported numeric substrate; surprises above threshold
trigger the slow brain (`agents/slow_brain.py` — the rewrite half of L3,
absent until now), whose rewrites pass the SAME five validation gates as
generation (fidelity included) plus mechanical strong-habit protection.

Committed loop constants (build decisions, not sealed bars; rationale in
the module docstrings): warm-up 10 days (triggers disabled, expectations
initialize), scoring window 7 days (observed slot j lives on global day
10+j carrying its slot weight), surprise z ≥ 0.5 recorded, rewrite
trigger = same-context surprise on ≥3 of trailing 5 days OR single
|z| ≥ 2.0, post-accept cooldown 7 days, strong-habit threshold 14 net
days followed, 2 attempts then the old card stands.

## Verdicts against the sealed bars — deployed population (r2b), through the loop

| Eval | Sealed criterion (A2) | M2 (static) | M3 (end-to-end) | Verdict |
|---|---|---|---|---|
| E1 pooled | ensemble-mean pooled TVD ≤ 0.10 | 0.0617 | **0.0614** | **PASS** |
| E1 cells | every protected cell ≤ 0.20 | worst 0.1352 | worst `car0\|remainder` **0.1305** | **PASS** |
| E1 falsification | Δ CI entirely < +0.00655 | [−0.0508, −0.0419] | **[−0.0510, −0.0420]**; MNL 0.1125 | **PASS — beats MNL** |
| E2(i) | spreads ∈ [0.8, 1.2] × 3 dims | 0.988 / 1.002 / 1.000 | **0.987 / 1.002 / 0.998** | **PASS** |
| E2(ii) | error correlation ≤ 0.20 | 0.0002 | **≈ 0.0000** | **PASS** |
| E5(i) | flag iff unmasked improves > 25% | 2.7% | **4.6%** | **NO FLAG** |

Same population (11,940 personas; 8,496 LLM / 3,444 fallback), same
ensemble N=20, base seed 20260717. The world coupling moved no sealed
quantity by more than 0.005 — the loop neither manufactures nor destroys
grounding fidelity at baseline. Plumbing regression: the M2 reproduce
commands re-run after the producer-seam changes are byte-identical to
the committed round-2b results (`runs/m3_regression_e1`, `_e2`).

**Rewrite activity in the scored baseline: ZERO.** Across 20 runs × 17
days, 22,800 raw surprise observations were recorded (route-draw noise,
~0.03% of person-day observations) and the trigger policy fired never —
ordinary days are ordinary. The rewrite machinery is validated instead
by the injected-jam rehearsal below. E5(i)'s loop-mode probe HALTS by
construction if either arm ever collects an organic rewrite request
(unmasked rewrite prompts would need quarantined vocabulary); neither
arm did.

## Reported diagnostic 1 — within-person day-to-day variance (owner directive)

Simulated within-person variance of trips/day across each multi-slot
persona's scoring days, against the same person's observed day-to-day
variance (n = 2,765 qualifying of 2,975 multi-slot personas):

| Subpopulation | sim/obs variance ratio | n |
|---|---|---|
| **Population** | **0.457** | 2,765 |
| clean LLM cards | 0.334 | 1,134 |
| clean fallback | 0.506 | 1,398 |
| duplicate-pattern (LLM) | 0.000 | 2 |
| tiling (LLM) | 0.706 | 131 |
| tiling (fallback) | 0.676 | 100 |

Findings, stated plainly: day-to-day variability is roughly HALVED in
simulation. The flagged subpopulations are NOT the driver — tiling cards
(which replay observed distinct days) preserve within-person variance
BEST; duplicate-pattern cards zero it exactly as predicted but are tiny;
clean LLM cards are the WORST (0.334). The compression is structural to
weight-drawn pattern cards (few patterns, skewed weights → repeated
identical days). Between-person variance (E2, sealed) is unaffected.

**Carried-forward risk (E6/M4), flagged now:** overly-regular simulated
days inflate habit-strength counters faster than real living would —
after only ~14 lived days the rehearsal population already carried
strong (immutable) rules on most active cards. At M4/M6 horizons this
biases toward rigidity: hysteresis (E6) could be overstated and
post-shock rewrites over-constrained. No fix is applied at M3; a dated
decision belongs with the pre-M4 batch if M4 design work confirms the
distortion.

## Reported diagnostic 2 — flagged subpopulations (owner directive)

- **Duplicate-pattern cards: 577/8,496 LLM cards = 6.8%** (persona ids
  now listed in `runs/m3_baseline/diagnostics_m3.json`; M2 recorded only
  counts). Fallback cards: structurally zero.
- **Tiling (near-enumeration): 270/2,975 multi-day persons = 9.1% on
  the deployed r2b set** (165 LLM + 105 fallback). Provenance of the
  difference vs the M2 record's 8.7%: that figure (260/2,975, fallback
  95) was computed on the round-2 card file; the round-2b fallback
  rebuild changed fallback card content. The ported audit logic
  reproduces the historical 260/2,975 on the round-2 file exactly; the
  M2 record is correct for what it measured and is not rewritten.
- **Sensitivity (do they distort M3 dynamics?):** excluding either or
  both flagged sets moves E1 pooled by ≤ 0.002 (0.0601 baseline run →
  0.0616/0.0604/0.0618) and WORSENS the guard cell (0.1299 → up to
  0.1447) — the duplicate-pattern cards are numerically faithful where
  the guard cell needs them. **No distortion of sealed M3 quantities →
  no regeneration proposed** (per the owner's no-silent-regeneration
  directive, a dated fix would only be proposed on evidence of
  distortion; the evidence is against).

## Reported diagnostic 3 — E1/E2 by card provenance (owner directive; diagnostic only)

Truth side restricted person-level to each split (adapter gained an
additive `person_ids` filter; default path byte-identical to the sealed
truth, regression-tested; 55.2% of fallback personas share a household
with an LLM persona — household-level restriction would have
contaminated exactly this comparison):

| Split | E1 pooled | E1 worst cell | E2 trips/day spread |
|---|---|---|---|
| LLM-only (8,496) | 0.074 | `mid\|car0\|catchment` 0.119 | 0.936 |
| fallback-only (3,444) | 0.043 | `car0\|remainder` **0.287** | 1.028 |

The deterministic fallback is pooled-closer to truth but concentrates
its error in the guard cell (0.287 — above the 0.20 cap IN ISOLATION;
the sealed verdict is population-as-deployed, where the cell sits at
0.1305). The LLM subpopulation is smoother but pooled-worse and
under-disperses within-person (see diagnostic 1). Sealed verdicts are
untouched by this table.

## Rewrite-path rehearsal (real model, injected jam) — and a contract revision

No organic rewrites exist at baseline, so the path was validated by
rehearsal: corridor capacity cut to 12% from day 12 (two-segment run
through the loop's checkpoint seam) over the 514 corridor car-driver
personas → day-12 surprises spike 40 → 277 → **95 personas trigger**
(first-request dedup; 194 raw). Prompts rendered by the production path
(`grounding/render.render_rewrite_prompt`), mask-lint clean, generated
by Qwen3-8B on the allocated cluster (vLLM structured decoding,
temperature 0, full-node 4-shard; 168 generations across all rounds,
zero schema violations, 3 truncations — one persona truncated in every
round it entered).

**Failure history, preserved:** under the original byte-immutability
contract (a rewrite that drops or alters a strong rule is rejected),
first-pass acceptance was 30/95; the model deterministically simplifies
strong rules' `when` clauses while keeping rule ids, and named-rule
retry feedback recovered 1/65 — byte-copying JSON through an LLM is the
wrong enforcement mechanism (`runs/m3_rehearsal/gate_stats_round*_byteimmutable.json`).
**Dated revision (2026-07-15, before any scored rewrite exists):**
strong rules are now mechanically RESTORED — verbatim content, original
relative order, placed ahead of every proposed rule so nothing a rewrite
adds can preempt an established habit under first-match-wins — and the
five validation gates run on the repaired object as the only rejectors
(`agents.slow_brain.restore_strong_rules`; doctrine unchanged:
resistance is mechanical, not rhetorical). Re-gated on the same
generations: **87/95 accepted (91.6%)**, 57 with restorations, residual
rejections = 7 fidelity-gate failures + 1 truncated generation; the
fidelity retry round recovered 1 of 8 (temperature-0 retries saturate,
matching the M2 generation record), leaving 7 terminal rejections whose
old cards stand. **Two-round rehearsal acceptance: 88/95 = 92.6%**
(`runs/m3_rehearsal/gate_stats_round*.json`, byte-immutability history
preserved alongside).

## Carried forward to the pre-M4 decision batch (owner seals)

1. `has_pass` age/licence gating — 1,215 minors and 1,717 non-drivers
   carry CRN-drawn passes; charge incidence at M4 depends on it
   (proposal drafted, owner decides).
2. Borrowed-car availability gate — binding on the guard cell (7.3% of
   car0 trip weight is genuine car driving; both the deployed-population
   0.1305 and the fallback-split 0.287 diagnostics bound what the gate
   can fix; proposal drafted).
3. Fidelity gate vs post-shock adaptive rewrites — at M3 the same-gate
   rule is correct and is enforced; at M4 a genuinely adapting rewrite
   (fewer car trips under a toll) can fail a gate anchored to pre-toll
   diaries, clamping the very adaptation the milestone measures. Needs a
   dated decision before M4.
4. Within-person variance compression (0.457) → habit-strength inflation
   risk for E6 horizons (diagnostic 1).
5. VoT income ladder in the bridge is a documented placeholder pending
   M4 toll-response calibration inputs.

## Reproduce

```bash
.venv/bin/python -m evaluation.run_m3 --cards data/cards/cards_m2_masked_r2b.jsonl \
  --out runs/m3_check --runs 20 --seed 20260717 --warmup 10 --scoring-days 7 \
  --slow-brain batch --label m3_check
AGORA_EVAL_CONTEXT=1 .venv/bin/python -m evaluation.run_e5 \
  --masked-cards data/cards/cards_m2_masked_r2b.jsonl \
  --unmasked-cards data/cards/cards_m2_unmasked_r2b.jsonl \
  --out runs/e5_m3_check --runs 20 --seed 20260717 --loop --warmup 10 --scoring-days 7
# diagnostics: see runs/m3_baseline/diagnostics_m3.json (driver in the run manifest)
# rehearsal: python -m jobs.rewrite_roundtrip collect|gate (runs/m3_rehearsal/)
```
