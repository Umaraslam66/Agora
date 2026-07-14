# PRE-REGISTRATION — SAGA
*Status: FROZEN at first commit. Metrics and decision rules below may not be edited. Numeric bars marked [M0] are set once, at the end of the M0 data audit, before any agent runs — via a single dated amendment appended to §7. Any other change requires a dated amendment in §7; the original text is never rewritten.*

## 1. The claim under test
A population of LLM agents, seeded one-to-one from real individual travel-diary records and acting under consequences in a masked world, can (a) reproduce ordinary behavior with human-like variance, (b) exhibit a say-do gap matching the real one in direction and rough size, and (c) predict — blind — the population response to a policy shock it was never calibrated on, including the partial-rebound (hysteresis) signature when the policy is removed.

## 2. The wall
Timeline of the anchoring natural experiment (masked from agents, known to the harness):
- **P0** pre-toll period → **CALIBRATION**
- **P1** toll ON (trial, ~6 months) → **CALIBRATION**
- **P2** toll OFF (~12 months) → **BLIND TEST 1**
- **P3** toll ON permanently → **BLIND TEST 2**

Rules: nothing measured in P2/P3 may influence any tuning, prompt, correction, threshold, or model choice. The truth series lives in `evaluation/truth/`, which agent/serving code cannot import (enforced by test). Each blind test is scored **once**; the result is sealed in docs/ whatever it says. Re-runs after seeing a blind score are reported as post-hoc, clearly labeled, and cannot overwrite the sealed verdict.

## 3. The evals

**E1 — Grounding fidelity.** Split diary records 80/20, household-atomic. Build agents from the 80%. Compare simulated ordinary-day behavior of each segment against the held-out 20%: distributions of trips/day, mode shares, departure-time bands. Metric: total variation distance (TVD) per distribution, reported per segment and pooled. Bar: pooled TVD ≤ [M0]; no protected segment (income × car-ownership cells) exceeds 2× pooled. Falsification arm: the calibrated MNL must be scored identically; the method must beat or match it before any architecture claim is made.

**E2 — Variance preservation.** (i) Spread ratio = simulated between-agent variance / real between-individual variance, per behavior dimension; pass band [M0] (target ≈ 0.8–1.2, finalized at M0). (ii) Independence: mean pairwise correlation of agent-level prediction errors ≤ [M0] — a hidden single-mind fails here even if means are right. Both scored at M2 (agents alone) and M3 (full loop).

**E3 — Say-do direction and transfer.** Measured on calibration events only, corrected once, correction frozen, then applied unchanged to the shock. Pass requires, in survey mode vs life mode: (i) gap **direction** matches the documented human pattern (stated resistance exceeding behavioral resistance); (ii) gap magnitude within a factor of [M0] (target ≤ 2×) of the real stated-vs-revealed discrepancy; (iii) the frozen correction **improves** blind E4 scores versus the uncorrected ablation — if it degrades them, the transfer claim FAILS and is reported as such.

**E4 — Blind shock prediction (the headline).** Calibrate on P0+P1 only. Predict P2 (removal) and P3 (return) as **distributions** from an ensemble of N ≥ 20 runs (varied seeds and persona-sampling). Scored on: (i) sign and magnitude of the aggregate cordon-crossing change — reality must fall inside the ensemble's 80% interval (coverage check, per period); (ii) TVD between predicted and observed mode-share shift; (iii) interval honesty across all reported quantities — an 80% interval that contains reality far more than 80% of the time is overwide and scores as a miss on sharpness, reported alongside coverage. Point predictions are never reported without intervals.

**E5 — Contamination probes (continuous from M2).** (i) Masked vs unmasked A/B on a calibration-period task: if unmasked materially outperforms masked (threshold [M0]), memorization is suspected and all results carry a contamination flag until resolved. (ii) Fictional-price probe: sweep toll prices never used in history (including implausible ones); response must be monotone in price and must NOT reproduce the famous historical aggregate at non-historical prices. (iii) Mask-lint in CI: forbidden-token list (city, policy, district names, real dates/prices) on every prompt template; any hit fails the build.

**E6 — Memory as hysteresis.** Real anchor: after removal, traffic did not fully rebound (documented residual reduction; exact band fixed at [M0] from published sources, cited). Protocol: run BLIND TEST 1 with (a) habit-strength counters active, (b) memory ablated (rules revert instantly). Pass: (a) predicts a partial rebound within the [M0] band while (b) predicts near-full rebound, with non-overlapping 80% intervals. If (a) and (b) are indistinguishable, the finding is "memory not load-bearing" — sealed and reported, as in enact.

## 4. What counts as the method winning
Not accuracy alone. The method claim stands if: E1 matches or beats the MNL falsification arm, E2 passes (variance is real), E4 achieves coverage on both blind tests, and at least one of E3-transfer / E6-separation passes. Any weaker pattern is reported as a partial or negative result with the same prominence.

## 5. What is fixed in advance
Model: Qwen3-8B (+LoRA where trained) served locally; gateway blend machinery inherited from enact. Ensemble N ≥ 20. Segment definitions, masking scheme, and the forbidden-token list are committed at M0 and versioned. Compute: EuroHPC Leonardo A100 nodes, full-node discipline.

## 6. Known threats, stated now
- The natural experiment is famous → masking may be insufficient; E5 exists for this, and a contamination flag suppresses headline claims.
- Diary microdata may arrive late or aggregated → M0's synthetic stand-in keeps the pipeline honest but only real-record seeding satisfies L1; results on stand-in data are labeled DEV and never cited.
- Aggregate-only truth data limits distributional scoring → E4(ii) reduces to the finest published breakdown; stated here so it is not a post-hoc excuse.
- Small calibration-event signal may be too weak to fit the E3 correction → then E3 is reported unpowered, not silently dropped.

## 7. Amendments
*(append-only, dated, signed)*
