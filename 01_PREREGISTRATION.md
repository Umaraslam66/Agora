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

---

### Amendment A2 — 2026-07-14, Umar Aslam

**A2.0 Scope.** This amendment sets every [M0] slot in §3, once, before any
agent has been run, from measurements on the primary arena's real seeding
microdata (measurement record retained in the project's private working
directory; summary tables in this amendment's drafting record). It also
freezes the scoring-protocol details those slots require (bins, bands,
folds, arm-pairing), the mode-taxonomy and segment pins of §5, and the
forbidden-token-list version. Where a measured noise floor made an
aspirational bar dishonest, the floor governs — recorded here so it cannot
be re-litigated after scoring.

**A2.1 E1 — grounding fidelity.** Protocol adopted (a pre-agent,
power-motivated §3 protocol amendment, measured before adoption):
**five deterministic household-atomic folds
(sha256(household_id) mod 5); every fitted component (choice model,
corrections, the MNL arm) trained strictly out-of-fold; the pooled
out-of-fold simulated population is scored against the full-sample
distributions.** Rationale sealed with the choice: with record-level 1:1
seeding, truth and simulation legitimately share the seeding sample, so
the residual TVD isolates pipeline distortion instead of holdout-draw
luck; the original single-holdout reading is noise-dominated at this
data's effective size (self-resampling floor 0.0909 → its honest bar
would be 0.182; the five-fold per-fold-averaged variant measures 0.160
for the same reason: averaging fold scores shrinks spread, not the
nonnegative per-fold bias). Under the adopted protocol the measured
regeneration-noise floor is 0.00205 (ensemble-of-20 scoring, p97.5), so
the bar is set by the pre-measurement aspiration, now honest:
**pooled TVD ≤ 0.10** — ≈24× the measured ensemble floor, leaving the
full margin for genuine pipeline error (the measured null contains no
fitted-component variation, which is stated here so the margin cannot be
re-read as slack later). Fallback readings and bars, recorded so the
choice is auditable: P0 single-holdout 0.182; P1 five-fold-averaged
0.160. Per-cell cap: 2× pooled = 0.20 over the ten cells; every cell is
scoreable under the adopted protocol (worst measured cell floor 0.0298;
merged guard cell 102 households).
Scoring details frozen for every reading: trips/day bins {0,1,…,7,8+}
(day-weighted, zero-trip weekdays included); mode shares over the frozen
five modes (trip-weighted); departure bands night <07:00, am_peak
07:00–08:59, midday 09:00–15:59, pm_peak 16:00–17:59, evening ≥18:00
(trip-weighted); ensemble of N ≥ 20 runs, TVD on the ensemble-mean
distributions; fold assignment (where used) =
sha256(household_id) mod 5, versioned with the adapter.
**Protected segments: ten cells** — the twelve cells of the committed
segment definition with the three car-free/remainder cells merged into one
guard cell (their individual holdout memberships, 5–12 households, cannot
be scored; merging preserves protection for car-free households outside
the catchment rather than silently dropping them). Per-cell cap: no cell
exceeds 2 × the pooled bar, per §3.
**Falsification arm (paired):** the calibrated MNL is trained and scored
under the identical fold structure, seeds, and comparison sample. "Beat or
match" (§3) is decided on the DIFFERENCE, not the absolute bars: the
method passes iff the household-atomic paired-bootstrap 95% CI of
(TVD_method − TVD_MNL), computed with both arms scored against the same
resampled truth in every replicate, lies entirely below +ε, with
**ε = 0.00655** (the measured p97.5 of |Δ| between two equal-skill arms
under the paired null; the unpaired equivalent is 0.0249 — pairing
tightens the difference CI ≈3.7×, which is where E1's discriminating
power lives). Absolute TVDs are reported alongside.

**A2.2 E2 — variance preservation.**
(i) Spread-ratio pass band **[0.8, 1.2]** per dimension; dimensions:
person-mean weekday trips/day, person car-share, person ride-share.
Protocol: **one-arm, preservation reading** — the denominator is the
between-individual variance of the SEEDING records in force (the train
side under the E1 protocol adopted above), the numerator the simulated
between-agent variance. E2 tests whether the pipeline preserves the
variance it was fed; generalization is E1's job. (The two-arm null at
holdout scale reaches [0.74, 1.38] — sealed here as the reason the one-arm
protocol is specified, so the band cannot be quietly re-read later.)
(ii) Mean pairwise correlation of agent-level prediction errors ≤ **0.20**.
Measured survey-world common-shock floor: 0.0105 (calendar-date ICC on
multi-day persons). Congestion feedback legitimately adds common shocks
the survey world lacks; 0.20 sits ~19× above the measured floor and well
below single-mind territory (≥0.5). Scoring E2(ii) requires realized
per-agent choices (the discrete-draw layer), not expected-value loads.

**A2.3 E5 — contamination probes.**
(i) Threshold: flag if the unmasked arm improves on the masked arm by more
than **25% relative** on ensemble-mean pooled TVD, under the **paired
protocol**: identical agent populations, common-random-number-paired
seeds, both arms scored once against the same fixed reference sample (no
resampling in scoring). Evidence sealed with the number: the UNPAIRED
two-arm null has p97.5 = 48.9% relative and would false-flag 23.8% of
honest runs; pairing removes that noise class. If the paired protocol is
infeasible in some future setting, the unpaired threshold is 98%; using
25% unpaired is not permitted.
(ii) Fictional-price probe flag windows: primary arena — aggregate change
within **±2pp of −28%** (the sealed BT1 truth) at any probe price ≥25%
away from the perturbed historical schedule; transfer arena — within
**±2pp of −22%** (the published stabilized trial effect), same price-
distance condition.

**A2.4 E3 — say-do transfer (scored in the transfer arena only, per
A1.6(iv)).** Factor **2.0**. R_real derivation, fixed now: surveys of
self-reported adaptation (fall 2004, fall 2005, spring 2006) imply an
equivalent aggregate reduction of only 5–10%, against an observed ~30%
reduction in private car trips across the cordon (Eliasson 2014, CTS WP
2014:7, §5.2 pp. 26–27; the same ~¾-unnoticed figure at §3.6 p. 15).
Midpoint: R_real = 30 / 7.5 = **4.0** (enacted exceeds stated), so E3(ii)
passes iff R_sim ∈ [2.0, 8.0] with the direction of E3(i) unchanged.

**A2.5 E6 — hysteresis band.** Residual reduction in **[4%, 12%]** of the
pre-trial baseline for arm (a); arm (b), memory-ablated, expected <4%
(near-full rebound); non-overlapping 80% intervals per §3. The published
anchor is the 5–10% off-year band (Börjesson et al. 2012 §3.1), widened
±2pp for the documented confound, quoted here so it cannot become a
post-hoc excuse in either direction: autumn 2006 traffic was "a few
percent lower" than autumn 2005 but concentrated at two roadwork-affected
bridges — "uncertain what conclusions can be drawn" (Eliasson 2009, §3,
p. 243); Börjesson et al. 2012 likewise note June–July 2007 figures are
roadwork-affected.

**A2.6 Pins and standing build decisions.** Mode taxonomy m0-1.0
(grounding/taxonomy.py, committed 2026-07-14) with the five-mode order as
the deterministic tie-break; segment definition as committed there, with
the A2.1 guard-cell merge; forbidden-token list v0.3; seeding waves
pooled two-wave with equal wave mass (weights × 0.5) and wave kept as a
covariate; weighted scoring is weekday-only by construction (per-wave
zero-weighted weekends); income-refusal households excluded from
segmented statistics (4.4% of household weight) and reported; zero-trip
weekdays included in trips/day. The residence-ring assignment currently
uses the core-jurisdiction proxy and is PROVISIONAL: it must be re-pinned
to the committed tract→zone map at M2 by a dated note, without moving any
bar sealed here.

### Amendment A3 — 2026-07-15, project owner

**A3.0 Scope and disclosures.** This amendment re-powers E3 in the
primary arena by decomposing the say-do correction into two channels
with distinct evidentiary anchors, adds one in-window aggregate
calibration anchor (the SR 520 tolling event), and revises A1.6(iv)
accordingly. Metric definitions and decision rules in §3 are not
altered; A2.4's transfer-arena E3 protocol and numbers stand unchanged.
Disclosures: (i) this amendment is adopted AFTER the M2 grounding runs
(persona cards exist; E1/E2 have been scored on calibration data) but
BEFORE any E3 quantity, any say-do correction, or any blind test has
been scored — no quantity measured after 2019-11-08 has influenced it;
(ii) every anchor below lies inside the C0 calibration wall;
(iii) survey waves after 2019 (2021/2023/2025) are post-wall for the
primary arena and are EXCLUDED from all fitting; they may be cited as
context only.

**A3.1 E3-primary, recall channel (FITTED).** The survey-mode/life-mode
gap is fitted person-level on stated-vs-revealed pairs present in the
seeding microdata (2017+2019 waves and their archived Public Release 2
attitudinal module, person-joinable 1:1), items pinned now:
(a) stated typical commute mode vs revealed commute-trip mode
(measured agreement 0.83–0.87, joint N ≈ 2,700–2,900 per wave);
(b) stated commute days/week vs revealed weekday commute rate
(stated exceeds revealed by ≈0.4–0.65 days/week);
(c) stated past-30-day mode-frequency bands (transit, walk, bike,
hired-ride, shared-car) vs revealed weekday usage (strongest family:
transit, Spearman 0.63–0.67; stated any-use prevalence 0.52 vs
observed 0.24 on the diary window);
(d) stated telework days/week vs revealed diary-day telework minutes;
(e) stated residence-factor importance vs revealed use (archived
module; monotone gradient, Spearman 0.32, N = 11,940).
Fitting discipline: the recall-channel correction is fitted ONCE on
these pairs under the A2.1 fold structure (fitted components strictly
out-of-fold), frozen, and applied unchanged thereafter (§3 E3's
fit-once-freeze rule, unchanged). Diary-window/recall-window mismatch
is part of what the correction absorbs and is stated here so it cannot
be re-read later as a confound discovered post hoc; the fit leans on
the multi-day (rMove) subsample where window noise roughly halves.
Pass bars, primary arena, recall channel: E3-primary(i) direction —
the simulated survey-mode claim must exceed life-mode behavior in the
same direction as every pinned pair family measures (stated typical/
frequency claims exceed revealed rates); E3-primary(ii) magnitude —
the simulated stated-vs-revealed discrepancy per pinned family within
a factor of **2.0** of the measured discrepancy (mirroring A2.4's
factor; measured R_real values are computed by the frozen procedure
above from the seeding waves at fit time and recorded in the fit
manifest, not re-derivable after scoring).

**A3.2 E3-primary, price channel (PRIOR-ANCHORED, declared).** No
instrument in the primary arena pairs stated price response with
revealed price response at person level; the price-channel correction
is therefore anchored on the published literature and is a PRIOR, not
a fit — recorded as such: central anchor, revealed toll response =
**2–3×** stated response (Brownstone & Small 2005, Transportation
Research A 39(4): paired RP/SP on two real tolled corridors; median SP
value-of-time about half median RP across two independent studies,
mixed-logit robustness check 3×); sensitivity band **[1.35, 3.0]**
(lower bound: Murphy et al. 2005, Environmental & Resource Economics
30(3), median same-mechanism hypothetical bias 1.35 — opposite-signed
construct, quoted as the conservative floor; tie-cite Loomis 2011,
J. Econ. Surveys 25(2)). The single-corridor counter-example showing
the gap can be small (Devarasetty, Burris & Shaw 2012, Katy Freeway)
is cited as the reason this channel carries a sensitivity band rather
than a point value. The price-channel correction's ONLY pass/fail test
is the frozen E3(iii) transfer clause, unchanged: the frozen correction
must improve blind E4 versus the uncorrected ablation, else the
transfer claim FAILS and is reported as such. No E3-primary(ii)
magnitude bar is set for the price channel — setting one against a
prior would be circular, and this is stated now so it cannot be
attempted later.

**A3.3 SR 520 calibration anchor (aggregate; NOT say-do, NOT E6).**
The SR 520 floating-bridge tolling event (tolling began 2011-12-29,
all-electronic, both directions; opening weekday pass rate $3.50 peak
with time-of-day ladder and pay-by-mail surcharge; six July-1 rate
steps 2013–2018; overnight tolling added FY2018) is adopted as an
in-window aggregate anchor, roles pinned now:
(a) habit/anchoring dynamics — the corridor shows a ~36–40% AADT drop
at tolling with drop-and-plateau persistence (no recovery to baseline
through the wall) on the published monthly transaction series
(WSDOT/Stantec 2019 T&R study, Table 3.2, FY2012–FY2019) and the
FY2013 monthly actuals; used to calibrate habit-persistence parameters
under a permanent level shift;
(b) calibration rehearsal — before BT1 is scored, the frozen pipeline
predicts the SR 520 drop as a labeled CALIBRATION exercise against the
forecast-vs-actual pair pinned now (official pre-tolling forecast: 48%
AADT drop; realized: 35–40%; WSDOT Toll Division, IBTTA 2012), scored
with E4 machinery but never as a blind result;
(c) revealed heterogeneity — the FHWA/Volpe SR-520 household panel
(~2,000 households, Nov 2010 wave 1, Apr–May 2012 wave 2; 43% recorded
trip reduction and 52% VMT reduction on the tolled facility, ~¼ of
former users diverting to the free parallel crossing; diversion
strongest among males, lower-income, and low-schedule-flexibility
respondents) is adopted as a validation target for the persona-level
response DISTRIBUTION (E2-adjacent), not as a say-do pair — the panel
is a revealed before/after design with no matched stated-intention
wave.
Confounds pinned now so they cannot be post-hoc excuses: the new
6-lane bridge opened 2016-04 (capacity break — dynamics calibration
restricted to 2011-12→2016-03); the FY2018 overnight-tolling counting
break; the concurrent transit-service boost (+90–130 daily bus trips
with tolling start — the observed response is joint price+service);
six rate steps; post-recession growth trend; parallel-crossing
construction during the window. The event is a LEVEL SHIFT with no
removal arm: it cannot and does not power E6, and the long-run
investment-grade forecast comparison remains UNVERIFIED from primary
sources and is excluded from every pinned fact above pending retrieval
of the 2010 investment-grade study.

**A3.4 A1.6(iv) revised.** A1.6(iv) declared E3 unpowered in the
primary arena outright. As of this amendment: the RECALL channel is
powered per A3.1 (fitted, person-level, in-window); the PRICE channel
remains unfitted locally and is prior-anchored per A3.2, with E3(iii)
as its only test. The Sept-2018 media poll remains excluded (no
methodology, no RP match). E3's transfer-arena battery per A1.2/A2.4
is unchanged in every respect. The fit-freeze-transfer structure,
explicit: FIT the recall channel (A3.1) and SET the price prior (A3.2)
on calibration-window data only, at M5, before any blind scoring;
FREEZE both in a dated fit manifest; TRANSFER the frozen corrections
unchanged into BT1 (E4) and the Stockholm arena; the corrections may
never be refit after any blind or transfer quantity is seen.

**A3.5 Frozen-text collision register.** (i) A1.6(iv): superseded as
stated in A3.4 — append-only supersession before any affected quantity
is scored. (ii) A2.4: untouched; its factor 2.0 and R_real = 4.0
govern the transfer arena exactly as sealed. (iii) §3 E3's [M0]
magnitude-factor slot: A2 set it for the transfer arena only (the
primary arena being then-unpowered); A3.1's primary-arena recall
factor 2.0 is a NEW application of that slot to a battery that did not
exist at A2 sealing — flagged as the sharpest edge of this amendment;
the owner seals it knowingly. (iv) §2 wall: every anchor and every
fitting input above predates 2019-11-08; post-2019 survey waves are
excluded from fitting per A3.0(iii). (v) §5 model/compute pins:
unaffected. (vi) E6: explicitly NOT powered by SR 520 (A3.3); its
anchor remains A2.5, transfer arena.

### Amendment A4 — 2026-07-16, umaraslam66

**A4.0 Scope and disclosures.** This amendment prepares the primary-arena
shock milestone (M4 / BLIND TEST 1) by (i) adding **E7**, an information-value
ablation reported in equivalent-sample-size units; (ii) specifying M4's
**price channel** (an announced, known-price onset), replacing the M3
same-gate rewrite rule UNDER THE SHOCK with a structural-only rewrite gate,
and adding a **placebo negative-control arm** that revises how E4's (and
E6's) reported response is computed; and (iii) **calibrating the
habit-persistence parameter** on the SR 520 aggregate anchor (A3.3(a)
authorizes it) and adding an **E6 sensitivity band**. Disclosures: (i) adopted
AFTER the M3 baseline (E1/E2 rescored end-to-end, PASS; zero organic rewrites
at baseline) but BEFORE any E4/E6/BT1 quantity, any toll-exposed run, or any
SR 520 calibration fit has been scored — no quantity measured after
2019-11-08 has influenced it; (ii) every calibration input below (SR 520) lies
inside the C0 wall (A1.1); (iii) the §3 metric definitions for E1–E6 are
unchanged EXCEPT the two items revised explicitly here (E4/E6 reported
response computed as a toll−placebo difference, A4.2; an E6 sensitivity band,
A4.3). E7 is a new §3 eval that sets NO new pass bar. WALL NOTE, stated here
because A4.2(ii) reasons about the instrument: the M4 price channel is fixed
from an INSTITUTIONAL fact about the toll (rates adopted 2018-10-16, ~13
months before the 2019-11-09 onset — public rate-setting, not a response
measurement); the blind BT1 trajectory (the two-week vs three-month response
equality) is a SCORED quantity and is used ONLY as what E4's trajectory
scoring checks, never as an input to any mechanism, threshold, or prompt.

**A4.1 E7 — information-value ablation (rising information tiers).** The FULL
deployed population (11,940 personas) is rebuilt at five NESTED information
tiers, each tier the previous plus one increment of evidence handed to the
SAME generation pipeline (render → generate → gate → assemble, with the
deterministic fallback backstop), CRN-PAIRED across tiers (identical persona
set, sampling seeds, temperature-0 decoding, fallback determinism; only the
evidence bundle grows, so tier-to-tier differences are information, not
sampling noise):
**T1 demographics** — skeleton only (age, sex, income class, household
type/size, licence, vehicles, home/work zone codes). No behavior, no stated
claims, no world context beyond the skeleton.
**T2 +context** — T1 plus the masked world the persona inhabits (zone
geometry and access, ring, corridor membership, mode availability):
placement without personal behavior.
**T3 +stated claims** — T2 plus the person's stated/attitudinal module (the
A3.1 recall-channel items: stated typical commute mode, stated commute
days/week, stated past-30-day mode-frequency bands, stated telework,
residence-factor importance) — what the person SAYS, not what they DID.
**T4 +one observed day** — T3 plus a single observed weekday diary day; the
CRN-fixed day-selection rule is pinned in the E7 fit manifest.
**T5 +full trace** — T4 plus the full multi-day diary (the M2/M3 evidence
bundle). This is the FULL-INFORMATION arm.

Fidelity-gate applicability: the generation-time fidelity gate (card
self-consistency to the person's OWN shown diary) applies only where a diary
is in evidence — T4 (to the one shown day) and T5 (full). T1–T3 carry no shown
diary and are fidelity-exempt at generation (as fallback cards already are),
but every tier is scored against held-out truth identically.
**Identification pins (owner directive):**
(a) **Marginal value of stated claims given behavior — the T4-noclaims arm.**
The nested ladder yields only the value of a behavior day GIVEN stated claims
(the T3→T4 step: day-given-claims). To also measure the value of stated
claims GIVEN behavior — once you know what a person DID, do their stated
claims add anything? — a side arm **T4-noclaims** (T2 + one observed day, the
stated-claims module WITHHELD) is built and scored on ordinary days AND BT1;
the contrast **T4 − T4-noclaims** is that marginal value (claims-given-day).
Both arms carry the observed day and are fidelity-gated, so the contrast is a
stated-claims contrast, not a gate contrast (the earlier gated/ungated label
is not used — it would wrongly imply the fidelity gate differs between them).
T4 (= T3 + one observed day) keeps its name as the nested tier; T4-noclaims
is the most informative rung and is kept.
(b) **Separating one day's information from the verification it enables — the
T4-nofidelity arm.** T1–T3 are fidelity-exempt (no diary in evidence); T4 is
fidelity-gated. The T3→T4 step therefore bundles the observed day's raw
INFORMATION with the VERIFICATION the generation fidelity gate performs once a
diary is present — and that gate is not small: at M2 it moved E1 from 0.139
(round 1, which lost to the MNL arm) to 0.062 (round 2b), ≈0.07 TVD, larger
than most rungs here. Left bundled, E7 would report verification as
information. So a diagnostic arm **T4-nofidelity** (T3 + one observed day, the
generation fidelity gate OFF) is built and scored on ORDINARY DAYS ONLY (no
BT1 exposure — zero cost to the single firing); the decomposition is
**T4 − T4-nofidelity = the fidelity gate's contribution** and
**T4-nofidelity − T3 = the day's raw information**.
(c) **Top rung identified on multi-day persons only.** For a single-observed-
day persona, T4 and T5 are identical by construction (the full trace IS the
one day). The T4→T5 rung — the value of the full trace over one day — is
therefore computed on the MULTI-DAY subpopulation only (the multi-slot
personas, ≈2,975; the qualifying count is reported), single-day personas
excluded from the top-rung increment so structural zeros do not dilute it.
(d) Scoring and units, below, apply per tier and per arm.

Scoring: each tier is scored (a) on ORDINARY DAYS under the sealed A2.1 E1
protocol (five household-atomic folds, pooled out-of-fold, ensemble N ≥ 20,
TVD) and (b) on BT1 under the E4 blind-shock machinery (A4.2), per tier — all
tier populations frozen before BT1 and scored TOGETHER in the single firing
(the operational reason M4 waits on this seal). Units: each tier's ordinary-day
and blind-shock performance is reported in EQUIVALENT-SAMPLE-SIZE units (Gao,
Han & Liang 2026, arXiv 2601.12343): tier k's cards achieve the predictive
accuracy of a flexible baseline trained on ESS(k) real diary records, and the
increment ESS(T_k) − ESS(T_{k−1}) is the marginal value, in equivalent real
records, of that tier's added evidence; inference by the paper's asymptotic
theory for cross-validated prediction error, with the flexible baseline and CV
protocol pinned in the E7 manifest before scoring. Headline discipline: the
full-information arm (T5) is the SOLE headline — E1/E4 PASS/FAIL verdicts and
every §4 method-winning claim are declared on T5 ONLY; T1–T4 are the reported
information-value curve (diagnostic) and may NEVER be promoted to an
alternative headline, stated now so a lower tier that happens to score better
cannot be substituted post hoc. E7 sets no new pass bar; T5's verdicts ARE the
sealed E1/E4 verdicts, so no post-M0 bar is created.

**A4.2 Post-shock adaptive rewrites — structural gate, M4 price channel, and
placebo control.**
**(i) Structural-only rewrite gate (shock mode).** At M3 the same-gate rule
(a rewrite passes the identical five `validate_card` gates as generation,
fidelity included, plus mechanical strong-rule restoration) is correct for
ordinary days and is unchanged there. UNDER THE SHOCK a genuinely adapting
rewrite (fewer car trips, mode/route switch, fewer cordon crossings)
deviates from the persona's pre-toll observed diary and fails the FIDELITY
gate (which scores mean trips/day, mode-share TVD, and quiet-day discipline
against the pre-toll observed stats), mechanically clamping the adaptation
the milestone exists to measure. A gate cannot distinguish adaptation from
drift; only a control can. So a shock-mode rewrite is accepted iff it passes
the STRUCTURAL channel only — schema, mask-lint, replay-smell, and
feasibility — PLUS mechanical strong-rule restoration. The FIDELITY gate
alone is dropped for shock-mode rewrites (it is the only one of the five
scored against the pre-toll diary; replay-smell and feasibility constrain
contamination and physical possibility, not adaptation direction, and stay).
This extends the existing fidelity exemption (fallback cards are already
fidelity-exempt) to the shock-rewrite class. Ordinary-day rewrites keep the
full five-gate rule.
**(ii) M4 price channel — an announced, known-price onset (NOT a discovered
surprise).** The toll is a KNOWN price: its rates were adopted and published
in advance of onset (institutional fact; A4.0 wall note). Agents therefore
perceive the toll as a known price at a known onset day and re-optimize
against it immediately — there is no discovery lag. Mechanically, M4 adds a
new trigger TYPE alongside the ordinary time-surprise trigger: an
**announced-onset trigger** fires the slow brain ONCE for every corridor
agent on the onset day, handing it the new masked per-period price (and pass
semantics), and the agent re-optimizes its card against the new generalized
cost through the structural-only gate. The ordinary time-surprise trigger
continues to govern the post-onset TAIL (equilibrium-shift surprises from
real diversion). This reproduces an instant step with no learning curve;
modeling the toll as a discovered time-surprise instead would make agents lag
reality by construction and is rejected. The mechanism is fixed from the
adoption fact, NOT from the blind trajectory (A4.0 wall note); whether the
instant step matches the observed two-week/three-month equality is part of
what E4 scores.
**(iii) Placebo negative-control arm, coherent with (ii).** Drift control
moves from the removed fidelity gate to a placebo arm run alongside the toll
arm on the two-arm CRN-paired loop-mode harness built for E5(i) (with its
quarantined-vocabulary halt rule). The placebo **yokes the TRIGGER and nulls
the REASON**: it fires the announced-onset trigger for the SAME corridor
agents on the SAME onset day (yoked 1:1; the onset set is deterministic, so
the match is exact), but the placebo world holds the baseline UNTOLLED
generalized cost and the onset cue is a content-free reconsideration notice
with the price/toll fields nulled (mask-lint clean by construction). It
carries ZERO actionable content — no price, no cost change, no time change.
Because the placebo agent's generalized cost is unchanged, a rational
re-optimization is a NO-OP; any change to its free channel (mode, route, trip
counts) is machinery-induced DRIFT. This is NOT a sham surprise ("told
surprised, shown nothing"): it is a content-free reconsideration under an
unchanged world, which is coherent — the agent is cued to review and, having
no reason to change, a faithful machine does not. The post-onset tail is
arm-native (the untolled placebo world produces the M3 baseline's ≈zero
surprises; tail surprises are NOT injected into the placebo — injecting a
sham time-surprise would carry an actionable signal and is forbidden).

Estimand: for every E4/E6 scored quantity Q, the REPORTED response is the
difference ΔQ = Q_toll − Q_placebo, PAIRED across CRN-matched ensemble members
(paired differencing removes shared route-noise and machinery variance and
tightens the 80% interval — the pairing-for-power doctrine of E1's ε (A2.1),
E5's paired protocol (A2.3), and E2(ii)). Because the placebo carries no
actionable signal, it removes ONLY drift, so **ΔQ is the TOTAL drift-corrected
toll response — the quantity the observed −28% measures — not a marginal price
effect**; a placebo that carried any actionable signal would make ΔQ marginal,
which the cost-invariant design of (iii) forbids. E4's frozen bars (A1.1:
observed −28% inside the ensemble's 80% interval; central prediction strictly
closer than the −45% benchmark) apply to ΔQ; E6's arm-(a)/arm-(b) hysteresis
contrast (A2.5) is likewise a toll−placebo difference, so the residual-rebound
band [4%, 12%] is scored on the drift-corrected quantity. **Known bias pinned
now:** the observed −28% (three-month window ending 2020-02 against the
Sep–Oct baseline) carries an unquantified SEASONAL component that the placebo
cannot correct — the placebo controls simulation drift, not real-world
seasonality — so the ΔQ-vs-−28% comparison (in E4 and in E7's BT1 scoring)
is stated with this uncontrolled real-world component; a seasonal adjustment
from the pre-toll year's autumn→winter weekday pattern may be added if sourced
pre-BT1, but is pinned here as a caveat, not a later excuse.
**Identifying assumption (additivity), stated plainly.** Writing drift_toll and
drift_placebo for the machinery drift each arm produces under the yoked
trigger, ΔQ = Q_toll − Q_placebo = true_response + (drift_toll − drift_placebo);
ΔQ equals the true response ONLY if the two drifts cancel
(drift_toll = drift_placebo). That cancellation is the placebo's identifying
assumption and is UNVERIFIABLE by design — for the same reason a gate cannot
separate adaptation from drift, the control cannot observe true_response alone.
It has a known failure direction: the model's trip-deflation bias (the tendency
that made round-1 cards under-count trips, the reason the fidelity gate was
built) may AMPLIFY in the toll arm, where the price gives the model a reason to
cut trips, against the content-free placebo where it has none — so
drift_toll ≥ drift_placebo, biasing ΔQ to OVERSTATE the response. The
assumption grows more load-bearing as the measured drift floor grows: at a
small floor the residual (drift_toll − drift_placebo) is small whatever its
sign; at a large floor it can dominate ΔQ. Pinned now so it cannot become a
post-hoc excuse in either direction, and it is why the drift rule below carries
an absolute leg, not only an anomaly leg.
**Tail residual — uncontrolled by design, bounded by a tail-off ablation.**
The placebo controls the announced-onset drift (the dominant exposure: onset
forces every corridor agent to reconsider). The post-onset TAIL rewrites (the
ordinary time-surprise trigger firing on real diversion-induced congestion)
have NO placebo counterpart — the untolled placebo world produces the M3
baseline's ≈zero tail, and a sham tail-surprise is forbidden (it would carry
an actionable signal, (iii)) — so tail-rewrite drift enters ΔQ uncorrected.
This residual is uncontrolled by design and is BOUNDED, without any blind
quantity, by a pre-declared TAIL-OFF ablation on the T5 headline arm, scored
in the single BT1 firing: ΔQ at T5 is computed with the post-onset
time-surprise trigger ON (the headline) and OFF (only the announced-onset
trigger fires — the bound), against the same placebo. The gap between the two
bounds the tail channel's total contribution to ΔQ, hence caps the
uncontrolled tail-drift, and is reported alongside the headline.

**Drift threshold — MEASURED, two legs.** The placebo is run on the SR 520
calibration rehearsal (A3.3(b): an announced level shift, in-window, with real
trigger load); the placebo/toll response-magnitude RATIO under that load is the
measured DRIFT FLOOR, itself REPORTED as a standalone quantity in a dated note
BEFORE BT1 fires. For each headline quantity at BT1 the same placebo/toll
magnitude ratio is computed, and the run is reported DRIFT-DOMINATED at that
quantity if EITHER leg trips: (anomaly leg) the ratio EXCEEDS 2× the floor —
BT1 drifting more than calibration did; OR (absolute leg) the ratio is ≥ 0.5 —
ΔQ has become a residual of two comparable numbers (at 0.5, ΔQ equals the drift
it subtracts, signal-to-drift 1:1, and the difference's relative error
amplifies as the arms converge). The two legs guard different failures: the
anomaly leg ALONE would bless a consistently high floor (a uniformly drifty
machinery never trips a multiple of its own high floor), and E1's 2×-floor
doctrine does NOT transfer to sanction that — an E1 noise floor is IRREDUCIBLE,
so a multiple of it is honest, whereas a drift floor is a DEFECT, so a high
floor is itself the harm; the absolute leg catches it. The placebo arm's 80%
interval zero-containment is REPORTED for every headline quantity but is NOT a
failure trigger: under paired CRN a trivial systematic drift excludes zero, so
detectability is not materiality — magnitude gates, significance informs. The
placebo halts on quarantined vocabulary, as E5(i).

This revises §3 E4's scored quantity (now the toll−placebo difference) and the
M3 same-gate doctrine (shock-mode fidelity exemption); both are made BEFORE any
toll-exposed or blind quantity is scored.

**A4.3 Habit-persistence calibration (SR 520) and E6 sensitivity band.** The
strong-habit threshold — the net days-followed at which a rule becomes
immutable — is 14, a chosen build constant, not a calibrated one. With
within-person day-to-day variance compressed to 0.457 of observed (M3
diagnostic 1: simulated days are too regular), a rule is followed near-daily,
its strength climbs ≈+1/day, it reaches immutability after ≈14 lived days, and
`days_to_weaken = max(0, strength − threshold + 1)` then makes it strongly
revert-resistant. Habits therefore harden too early, which (i) OVERSTATES E6
hysteresis (over-resistance to reversion → too little predicted rebound) and
(ii) CLAMPS E4 (prematurely-immutable rules over-constrain post-shock
adaptation). Calibration (A3.3(a)): the habit-persistence parameter is
calibrated on the SR 520 aggregate anchor — the ~36–40% AADT drop-and-plateau
with drop persistence (no recovery to baseline through the wall) on the
published monthly transaction series (WSDOT/Stantec 2019 T&R study Table 3.2,
FY2012–FY2019, and the FY2013 monthly actuals), restricted to
2011-12 → 2016-03 (before the 2016-04 six-lane capacity break, per the A3.3
confound pins). Procedure: a permanent, masked corridor level-shift scenario
(announced known price, per A4.2(ii), on at day T, never removed — via the
`network_override`/config seam used for the M3 injected-jam rehearsal) is run
through the two-brain loop, and the strong-habit threshold (in sim-days) is fit
so the simulated aggregate corridor-volume trajectory reproduces SR 520's
observed transition speed to plateau AND its drop-and-plateau persistence (no
simulated drift-back). Because the plateau LEVEL (−36–40%) is a price-elasticity
property (value of time / toll / logit), not a habit property, and the VoT
ladder is still a DEV placeholder pending M4 toll-response calibration, the
habit threshold is fit to the transition SHAPE and persistence jointly with the
elasticity to the level, on the same series, and BOTH are frozen together in a
dated fit manifest (`calibration/`), calibration-window only, never refit after
any blind quantity (§2 wall; A3.4 fit-freeze discipline). Honest limit, pinned
now: SR 520 is a permanent level shift with NO removal arm (A3.3: "cannot and
does not power E6"); it disciplines habit FORMATION and persistence, not the
removal-side hysteresis E6 measures. E6's rebound-on-removal remains a
PREDICTION from the frozen-formation habit machinery, tested against the A2.5
Stockholm band. **E6 sensitivity band (new; owner-accepted source):** E6 is
scored across a BAND of the strong-habit threshold bracketing the fitted point
value, the band being the range of thresholds whose simulated trajectory stays
inside the CONFOUND-WIDENED envelope of the observed SR 520 drop (the envelope
widened for the A3.3 confounds — concurrent transit boost, six rate steps,
post-recession growth, parallel-crossing construction, FY2018 counting break),
pinned in the fit manifest. E6's verdict (arm-(a) partial rebound in [4%, 12%],
arm-(b) near-full rebound, non-overlapping 80% intervals per §3 / A2.5) is
reported at the fitted value AND across the band; a verdict that flips inside
the band is reported as habit-parameter-sensitive, not sealed as a clean
pass/fail — stated now so a knife-edge threshold cannot be read as a robust
result. The A2.5 band [4%, 12%] and the non-overlap rule are UNCHANGED; this
adds a reporting-robustness band and pins the threshold as calibrated-and-
frozen rather than chosen.

**A4.4 Pre-M4 freeze inventory and frozen-text collision register.** BT1 fires
once (A1.1); everything below is frozen before it fires, which is why M4 waits
on this seal. Freeze inventory: (1) the two pre-M4 gate decisions sealed
2026-07-15 (`docs/DECISION_M4_HAS_PASS_GATE.md`, household transponder
inheritance; `docs/DECISION_M4_BORROWED_CAR_GATE.md`, calibrated borrowed-car
availability), implemented into the executor and population builder, with the
M3 baseline loop re-run to pick them up so ordinary and shock dynamics share
one executor; (2) the habit-persistence threshold and E6 band, fit on SR 520
and frozen (A4.3); (3) the E3 recall-channel fit and price prior
(A3.1/A3.2/A3.4), frozen at M5-before-blind — cross-referenced, unaltered
here; (4) M4's price channel (A4.2(ii)), the E7 five-tier populations and the
two T4 decomposition arms (T4-noclaims, scored on BT1; T4-nofidelity,
ordinary-day only) (A4.1), the E4 blind-shock scorer, the placebo two-arm
harness with its measured two-leg drift rule and reported floor, and the T5
tail-off ablation arm (A4.2), all built and exercised on the SR 520
calibration rehearsal (A3.3(b))
before BT1; (5) the VoT/elasticity calibration, co-frozen
with the habit fit (A4.3). Collision register: (i) §3 E4 scored quantity —
revised to the toll−placebo difference (A4.2), a §3 change made pre-BT1; the
owner seals it knowingly. (ii) §3 E6 — a sensitivity band is ADDED (A4.3); the
A2.5 [4%, 12%] bar and non-overlap rule are untouched. (iii) M3 same-gate
rewrite doctrine — shock-mode fidelity exemption (A4.2(i)) extends the existing
fallback exemption; the ordinary-day rule is unchanged. (iv) §3 new eval E7
(A4.1) — sets no new pass bar (descriptive ablation; T5's verdicts are the
sealed E1/E4 bars), so no post-M0 bar is created. (v) §2 wall — every
calibration input predates 2019-11-08; M4's price-channel mechanism is fixed
from the 2018-10-16 rate-adoption fact, not from any BT1-measured quantity
(A4.0 wall note); BT1-frozen-before-firing is the reason M4 waits on this seal.
(vi) §5 model/compute pins — unaffected; E7 multiplies runs 5× × ensemble ×
arms and adds the placebo, T5 tail-off, and T4 decomposition arms (T4-noclaims
on BT1; T4-nofidelity ordinary-day only), a compute note under the full-node
discipline, not a pin change.

### Amendment A5 — 2026-07-17, project owner

**A5.0 Scope and rationale.** BT1's sealed aggregate benchmark (A1.1) is the
official pre-tolling forecast, −45% — deliberately conservative for revenue
purposes, and therefore a weak opponent. This amendment adds ONE sealed
comparator arm to the BT1 firing set: a plain calibrated statistical model
with no LLM anywhere, so the blind result can also be read against the
sharpest cheap alternative. No §3 metric, bar, or pass condition changes:
the −45% benchmark remains E4's sealed closeness benchmark, and the method's
PASS/FAIL verdicts are untouched. Adopted BEFORE any blind quantity is
scored; the comparator's prediction is produced and frozen before any blind
number is read.

**A5.1 The comparator, pinned.** Implementation
`evaluation/comparator_arm.py`, frozen at seal time:
- **Seeding:** the identical 1:1 PSRC seeding records; each persona's
  OBSERVED weekday diary is replayed verbatim (no cards, no generation, no
  habit machinery, no rewrites).
- **Behavioral response channel:** corridor route choice ONLY, through the
  frozen world equilibrium (`world.network.solve_corridor_equilibrium`,
  frozen logit theta, VoT lognormal × income ladder), under the same masked
  M4 toll schedule and the same sealed pre-M4 charge gates (household pass
  inheritance, car-trips-only discount).
- **Calibration:** the comparator's own VoT scale is fitted once, by
  bisection, to the SAME SR 520 aggregate anchor and level criterion the
  method's A4.3 fit uses (plateau drop = the pinned band midpoint), on the
  same rehearsal scenario, and frozen in `runs/comparator_arm/manifest.json`.
  Whatever rehearsal-schedule decision the owner takes for A4.3 (see the
  pre-M4 record's flagged design edge) applies to this arm identically.
- **Prediction:** the weekday tunnel-volume change over the BT1 window as a
  CRN ensemble (N ≥ 20 route-draw namespaces), central = mean, 80% interval
  by the E4 percentile convention. Deterministic behavior + CRN draws means
  no machinery drift: the raw ensemble is the prediction (no placebo
  differencing; recorded as a structural property, not an omission).

**A5.2 Scoring and discipline, pinned now.**
- Scored ONCE, inside the single BT1 firing, with the same E4 kernels
  (interval coverage of the observed value; central-prediction distance),
  and reported alongside the method's T5 headline with equal prominence
  whatever it says.
- The comparator is REPORTED, not a pass bar: E4's sealed pass conditions
  (coverage + closer-than-−45%) are unchanged. The comparative reading —
  whether the method's central prediction is strictly closer to the observed
  value than the comparator's, and both arms' coverage — is declared HERE as
  a headline-adjacent reported quantity, so it can neither be promoted to a
  pass bar nor quietly dropped after the result is seen.
- Interpretation pinned in advance: if the comparator matches or beats the
  method on BT1, the honest reading is that individual agent detail carried
  no signal beyond the calibrated aggregate dial in this arena — the E7
  information-value curve is the diagnostic that localizes where detail
  stopped paying. This is stated now so neither side of the comparison can
  be reframed post hoc.
- The comparator code path, its frozen VoT scale, and its prediction file
  are sealed before firing; any later change is post-hoc and cannot touch
  the sealed reading.

**A5.3 Collision register.** (i) §3 E4: unchanged (the −45% benchmark and
both pass conditions stand; the comparator adds a reported reading only).
(ii) A2.1's MNL falsification arm: distinct and unchanged — that arm scores
ordinary-day grounding fidelity out-of-fold and carries no price channel;
this arm exists precisely because the MNL cannot respond to a toll. (iii)
A3.2: not touched — the comparator carries no stated-response channel, so
the price-prior discipline does not apply to it; its price response is
anchored on the same revealed SR 520 aggregate as the method's (A3.3(a)
authorizes the anchor). (iv) §5 compute pins: unaffected (CPU-only arm).
