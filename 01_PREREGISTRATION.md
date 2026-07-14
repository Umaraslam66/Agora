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

---

### Amendment A1 — 2026-07-14, project owner

**A1.0 Scope and rationale.** The primary validation arena moves from the
Stockholm congestion charge to the Seattle SR 99 tunnel tolling event.
Reason: the Swedish RVU microdata required for record-level seeding (L1) is
gated behind an application with uncertain timeline, while the Puget Sound
Regional Council Household Travel Survey provides equivalent microdata as a
public download. This amendment re-anchors the wall (§2) and re-maps the
arenas of the evals (§3). **No metric definition, decision rule, or scoring
procedure in §3–§4 is altered.** This amendment is adopted before any agent
has been run in either arena and before any [M0] bar has been sealed; the
re-anchoring is a data-access decision, not a results-driven one.

**A1.1 Primary arena wall (Seattle).**
- **C0 — CALIBRATION**: all data through 2019-11-08 inclusive. This spans
  the PSRC 2017 wave, the PSRC 2019 wave, the permanent closure of the
  Alaskan Way Viaduct (2019-01-11, WSDOT one-year report), and the
  toll-free tunnel period (opened 2019-02-04, tolled from 2019-11-09,
  WSTC/WSDOT). The viaduct closure and free-tunnel
  opening are network shocks inside the calibration window and serve as
  say-do calibration events where stated-intention data exist.
- **BT1 — BLIND TEST 1**: the tolling response, 2019-11-09 through
  2020-02-29, scored once and sealed. The window ends 2020-02-29
  unconditionally: from 2020-03 the pandemic contaminates all traffic data.
  No quantity measured after 2019-11-08 may influence any tuning, prompt,
  correction, threshold, or model choice.
- **Aggregate falsification benchmark for BT1.** Scored quantity: change in
  average weekday whole-day tunnel volume over BT1 relative to the pre-toll
  baseline of ~77,000 veh/weekday (WSDOT formal baseline period
  2019-09-23 to 2019-10-31). Observed truth: **−28%** (three-month period
  ending 2020-02, WSDOT one-year report), with the first-two-weeks
  snapshot (−26%, 2019-11-12 to 2019-11-22) reported as a secondary
  trajectory point. Benchmark: the official pre-tolling forecast, taken in
  the reading most favorable to the forecast — **−45%** (the WSDOT
  forecast of 44,000 fewer daily trips, proportional off its own 97,000
  forecast base; WSTC/WSDOT 2018, as reported with the caveat that it was
  deliberately conservative for revenue purposes). The method's headline
  claim requires (i) the observed −28% inside the ensemble's 80% interval
  per the frozen E4 rules, and (ii) the ensemble's central prediction
  strictly closer to the observed value than the −45% benchmark is.
  The forecast's base-definition ambiguity (trips-vs-percent, forecast
  base 97k vs actual base 77k, peak-weighted vs whole-day) is recorded
  here precisely so it cannot be re-litigated after scoring.
- There is no Seattle BLIND TEST 2: every post-BT1 period is pandemic-
  contaminated. The second blind battery is the transfer arena (A1.2).

**A1.2 Transfer arena (Stockholm, frozen method).** After BT1 is scored and
sealed, the method — architecture, prompts, hyperparameters, say-do
correction, and all thresholds — is frozen. It is then applied to the
Stockholm arena and scored ONCE against the aggregate truth series already
extracted and quarantined in evaluation/truth/ (extraction of record:
2026-07-14, from Eliasson 2009; Börjesson et al. 2012; Eliasson 2014):
P0→P1 (introduction), P2 (removal, incl. E6 hysteresis), P3 (return).
Only the following Stockholm-side inputs are permitted before scoring:
world structure (zones, network, toll schedule, all masked), P0 aggregate
baselines for level-matching, and persona seeding per A1.3. Nothing may be
refit to P1–P3 outcomes. E6 runs exactly as frozen in §3, in this arena.
Deviation from the original §2: for the transfer arena, P1 moves from
CALIBRATION to scored territory (only P0 remains usable), strengthening
the original design rather than weakening it.

**A1.3 Seeding class and evidentiary labels (extends §6).** L1 seeding in
the primary arena uses PSRC records (public microdata). The transfer
arena's seeding class is: PSRC-seeded personas reweighted to published
Stockholm-County marginals, labeled **METHOD-TRANSFER**, with the label
carried on every reported number derived from them. This is a deliberate
design choice, not a data-access fallback: seeding the transfer arena from
its own population's microdata would reduce the second battery to a second
within-population test, whereas cross-population seeding under
published-marginal reweighting makes the Stockholm score a test of whether
the method — not source-population idiosyncrasy — carries the signal. That
is the stronger claim, and it is adopted as such before any agent has run
in either arena. Accordingly, the Swedish RVU/RES microdata application is
**WITHDRAWN** as of this amendment's date (withdrawn, not deferred; no
seeding class conditional on it exists). A transfer failure under
METHOD-TRANSFER seeding is a failure of the headline transfer claim and
may not be re-attributed post hoc to seeding class. The synthetic stand-in
remains DEV-only, never cited, per §6. The seeding class and the published
marginals used for reweighting are recorded in the sealed verdict.

**A1.4 Masking (extends §5).** The masked world remains "City K", reskinned
to the Seattle geometry: zone codes, a tolled tunnel link that replaced a
closed elevated highway, untolled parallel alternatives, time-of-day toll
schedule perturbed ±10% preserving relative structure, shifted dates.
Forbidden-token list v0.3 (committed 2026-07-14; v0.2 added the
Seattle/Puget Sound vocabulary, v0.3 contamination-assessment variants);
both arenas' vocabularies are enforced simultaneously.
E5 contamination probes run in both arenas; the fictional-price probe's
"famous aggregate" is the observed BT1 reduction in the primary arena and
the trial reduction in the transfer arena.

**A1.5 Known-outcome disclosure (extends §6).** The observed outcomes of
both natural experiments are public knowledge, known to the experimenters,
and recorded in this repository's working documents before any run.
Blindness here is procedural, exactly as in the original §2: the wall
forbids these outcomes from influencing any model input, not from being
known. This was equally true of the original single-arena design; it is
restated because the BT1 benchmark figures appear in the amendment itself.

**A1.6 Additional threats, stated now.** (i) The tunnel toll is a
link-level instrument; the Stockholm charge is a cordon instrument. The
transfer test is therefore cross-instrument as well as cross-country —
harder, and stated as such in advance. (ii) The PSRC 2019 wave was fielded
2019-04-22 to 2019-06-10 (trip-date verification + technical report) —
entirely inside the free-tunnel period. It is therefore NOT a neutral
baseline: it is the "corridor open, toll = 0" calibration point, with the
2017 wave as the pre-change anchor. This two-point structure is used
deliberately; no steady-state assumption is attached to 2019 data.
(iii) BT1's window is only ~16 weeks —
short-run response only; no long-run equilibrium claim may be made from
it. (iv) **E3 is declared unpowered in the primary arena now**: no
stated-preference survey with a matched revealed-preference counterpart
exists for the SR 99 toll (the sole stated-intention datum, a Sept-2018
media poll, lacks methodology and an RP match; usable only as a flagged
soft direction check). E3's transfer clause is scored in the Stockholm
arena only, per the §6 unpowered rule. (v) Only period-average truth is
firmly sourced for BT1 (baseline, two-week snapshot, 3-month average); if
a month-by-month tunnel series is not obtained before sealing, E4's
trajectory scoring reduces to those period averages — stated now so it is
not a post-hoc excuse. (vi) PSRC public weights are per-wave expansion
weights with all Friday–Sunday days carrying zero weight: every weighted
quantity in E1/E2/E4 is weekday-only by construction, waves are never
pooled without per-wave rescaling, and survey distances are recomputed in
the world layer (public files mix beeline and GPS-route distances — the
exact confound class that flipped a verdict in the predecessor project).
