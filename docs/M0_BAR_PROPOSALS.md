# PROPOSED numeric bars for the [M0] slots — NOT SEALED
*Status: architect's proposal, 2026-07-14. These numbers are NOT in force. Per
01_PREREGISTRATION.md, the [M0] bars enter the contract only via a single dated
amendment appended to §7, signed by the project owner, before any agent runs.
This document is input to that decision, nothing more.*

*Anchor numbers cited below are from docs/M0_DATA_INVENTORY.md (primary-source
extractions: Eliasson 2009, Börjesson et al. 2012, Eliasson 2014).*

## E1 — Grounding fidelity: pooled TVD ≤ **0.10**
Rationale: the falsification MNL typically lands in the 0.05–0.10 TVD range on
mode-share-type distributions when calibrated on the same records, so 0.10 makes
the bar meaningful without being unreachable; the frozen 2×-pooled cap for
protected segments then allows 0.20 in the worst cell. Condition to verify
before sealing: compute the 20%-holdout self-resampling noise floor on the real
microdata (bootstrap the holdout against itself); if 2× that floor exceeds 0.10
for the pooled statistic, seal at 2× the measured floor instead and record the
measurement in the amendment. Do not seal from synthetic (DEV) data.

## E2 — Variance preservation
- Spread-ratio pass band: **[0.8, 1.2]** per behavior dimension (as targeted in
  the frozen text). Tight enough that LLM mode-collapse (ratios typically <0.5)
  and noise-inflation (>1.5) both fail; wide enough for sampling error at the
  planned population size.
- Mean pairwise correlation of agent-level prediction errors: **≤ 0.20**.
  Note for sealing: congestion feedback legitimately correlates errors (common
  shocks), so zero is not attainable by honest models; 0.20 is roughly double a
  plausible common-shock floor. If the M0 real-data measurement of day-to-day
  common variation suggests a floor above 0.10, revisit before sealing — the
  bar must sit clearly above the honest floor and clearly below single-mind
  territory (≥0.5).

## E3 — Say-do transfer: gap magnitude within factor **2.0**
The real anchor is unusually strong: stated adaptation equivalent to only a
5–10% aggregate reduction versus ~30% observed private-trip reduction — a 3–6×
understatement, direction "behavior change exceeds stated intention" (Eliasson
2014 §5.2). Proposal: pin R_real to the midpoint derivation from that source
(document the exact arithmetic in the amendment), and require
R_sim ∈ [R_real/2, 2·R_real]. Direction match (i) and the frozen-correction
transfer test (iii) are already fully specified in the frozen text.

## E5 — Contamination threshold
Flag contamination if the unmasked arm improves on the masked arm by more than
**25% relative** on the calibration-period task's pooled TVD (primary), or if
the fictional-price probe produces an aggregate change within **±2 percentage
points of −22%** at any probe price differing from the (perturbed) historical
price by ≥25% (secondary; −22% is the famous stabilized trial effect the model
could only know by memorization at a price never seen in history).

## E6 — Hysteresis band: residual reduction in **[4%, 12%]** of pre-trial baseline
Published: off-year (Aug 2006–Jul 2007) traffic stayed 5–10% below the 2005
baseline (Börjesson et al. 2012 §3.1). Proposal widens the published band by
±2pp because the P2 truth series is the noisiest period and part of the
autumn-2006 residual is confounded by bridge roadwork (Eliasson 2009 caveat —
this caveat must be quoted in the amendment so it cannot become a post-hoc
excuse in either direction). Arm (a) must land its 80% interval inside a
partial-rebound verdict within this band; arm (b) ablated is expected near full
rebound (<4% residual); non-overlapping 80% intervals are already frozen text.

## Also to freeze at M0 (prereg §5 requires these committed now)
- **Mode taxonomy**: five modes, frozen order `walk, transit, ride, car, bike`
  (the gateway's temperature-0 tie-break follows this order; training's 4-mode
  default must be overridden by the renderer to match).
- **Segment definitions (protected cells for E1)**: household income tertile ×
  car ownership (0 / 1+) × residence ring (inside cordon / county remainder).
- **Forbidden-token list**: grounding/masking/forbidden_tokens.txt v0.1 is
  committed; sealing at M0 should bump it to v1.0 after one adversarial review
  pass (suggested additions: bridge/roadwork names, "Klarastrandsleden",
  referendum vocabulary).

## Sealing checklist for the §7 amendment
1. One dated entry, all six numbers together, signed.
2. E1 noise-floor measurement attached (real data only).
3. R_real derivation arithmetic written out for E3.
4. Roadwork caveat quoted verbatim for E6.
5. Token list version pinned (v1.0) and mode taxonomy stated.
