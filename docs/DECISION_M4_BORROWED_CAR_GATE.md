# Sealed decision — borrowed-car availability gate (pre-M4)

*SEALED 2026-07-15, project owner. Public sealed record per the proposal's
discipline clause; originating draft:
`docs/internal/DECISION_PROPOSAL_M4_BORROWED_CAR_GATE.md`. Not a §7 amendment —
the guard-cell cap (0.20) and every sealed bar are untouched. Binding on the
`car0|remainder` guard cell under toll dynamics. Takes effect before any
toll-exposed run (M4).*

## Decision

**Option B — calibrated availability draw — is adopted, as proposed.** Option A
(status quo hard gate) is rejected: it enters M4 with a known 7.3%-of-weight
behavioral hole and a wrong-incidence charge on coerced trips in the guard
cell, making every M4 gap in that cell un-attributable (gate artifact vs true
response). Option C (card-level authority over availability) is rejected:
availability is a physical constraint and must stay world-side, or the guard
cell becomes un-auditable and the "cards cannot break hard constraints"
doctrine breaks.

## What is sealed

1. **Availability draw.** Licensed adults in zero-vehicle households receive a
   **per-day borrowed-car availability draw** through the CRN layer. When the
   day's draw grants access, the card's car trips execute as car; otherwise
   they coerce to ride exactly as today. The habit substrate sees a
   borrowed-car day as an ordinary lived day — no special casing.
2. **CRN site key.** **`"{namespace}:{persona_id}:{day_index}:caraccess"`** —
   a fresh per-day key; existing streams stay bit-identical, and twin-world CRN
   pairing is preserved by construction.
3. **Fit-once-freeze.** The availability rate is fitted ONCE on the calibration
   window to reproduce the observed **7.3% of car0 trip weight** that is genuine
   car-driver travel, **restricted to persons whose OWN seeding record shows
   car-driver trips in a zero-vehicle household** (no invented availability for
   persons who never showed it — mirrors the per-card fidelity discipline). One
   scalar per such person class (with/without observed car-driving is permitted
   as at most two scalars), then FROZEN in a dated manifest.
4. **Expected side-effect (diagnostic only).** Guard-cell E1 TVD is expected to
   improve from 0.135; the improvement is reported in the M3/M4 record as a
   diagnostic, NEVER as a rescored M2 verdict.

## Discipline

- The fit uses calibration-window data only (wall-safe) and is never refit
  after any blind quantity is seen (§2 wall; A3.4 fit-freeze discipline).
- The M3 baseline loop is re-run to pick this up BEFORE M4, so ordinary-day and
  shock dynamics share one executor.
- Implementation manifest (populated at build, before M4):
  `runs/m4_prep/borrowed_car/manifest.json` — records the fitted rate(s), the
  fit window, the `caraccess` CRN key, and the realized car0 car-trip weight
  after the draw.
