# Sealed decision — `has_pass` household transponder inheritance (pre-M4)

*SEALED 2026-07-15, project owner. Public sealed record per the proposal's
discipline clause; originating draft: `docs/internal/DECISION_PROPOSAL_M4_HAS_PASS_GATE.md`.
Not a §7 amendment — no sealed bar or §3 text is touched; this is an
executor/world-semantics decision pinned by a dated note before BT1, exactly
like the A2.6 ring re-pin. Takes effect before any toll-exposed run (M4).*

## Decision

**Option B — household transponder inheritance — is adopted, as proposed.**
Options A (person-level hard gate) and C (status quo) are rejected: C leaves
measurably wrong charge incidence on 10.2% of personas at the milestone whose
headline quantity is charge response; A under-discounts genuine household pass
coverage for children riding the household vehicle.

## What is sealed

1. **Household-level draw.** The pass is re-modelled as a household attribute:
   drawn ONCE per household, gated on the household having **≥1 licensed adult
   AND ≥1 vehicle**. A household failing either condition holds no pass.
2. **Inheritance rule.** Every member inherits the household pass for **CAR
   trips in the household vehicle only**. Ride and hired trips NEVER receive
   the persona's pass discount — the fee belongs to the vehicle operator, whose
   pass status is unknowable. This kills the minor-pass anomaly (1,215 minors)
   and the non-driver-pass anomaly (1,717 `can_drive=false` personas) in one
   move and makes M4 charge incidence household-coherent.
3. **CRN site key.** The household pass draw uses the new site key
   **`"{namespace}:{household_id}:hh_pass"`** — a fresh key, so every existing
   per-persona stream (`:vot`, `:{day}:route`, `:{day}:caraccess`) stays
   bit-identical; determinism and twin-world CRN pairing are preserved by
   construction.
4. **Segment pass-rate preservation check (required).** The per-household draw
   is re-weighted so that person-level pass coverage for car trips, aggregated
   over the population, reproduces the sealed segment pass-rate pins within
   tolerance. The realized preservation result (per-segment observed-vs-target
   pass rate) is recorded in the implementation manifest referenced below; the
   check is a gate on the implementation, not a new sealed bar.

## Discipline

- No E1/E2 sealed verdict is re-opened. If the re-model shifts any E1 cell, the
  shift is reported in the M4 gate record as provenance, never as rescored
  history.
- The re-model is calibration-window-safe (household composition, licence, and
  vehicle counts are seeding-record attributes, all pre-wall) and is frozen in
  a dated manifest before the pre-M4 M3 baseline re-run, so ordinary-day and
  shock dynamics share one executor.
- Implementation manifest (populated at build, before M4):
  `runs/m4_prep/has_pass_household/manifest.json` — records the gating counts,
  the `hh_pass` draw seed/namespace, and the segment preservation table.
