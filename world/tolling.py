"""Time-of-day toll schedule for the masked corridor world (M1).

WHY THIS FILE EXISTS (01_PREREGISTRATION.md §5 masking; E5 fictional-price
probe): the policy instrument the blind test measures is a time-of-day fee on
one facility. Everything the schedule computes is in generic "credits" — never
a real currency amount, never a real rate (the numbers below are pre-perturbed
per the world plan: relative structure preserved, levels shifted). The
machinery is deliberately open at two seams the pre-registration needs:

  * an arbitrary GLOBAL PRICE MULTIPLIER — the E5 fictional-price sweep hook
    (sweep never-observed toll levels, including 0 = "free facility"); and
  * arbitrary RATE OVERRIDES — swap the per-period rates without touching any
    other machinery.

A schedule is an immutable value; with_multiplier / with_overrides return new
schedules, so a toll sweep never mutates shared state (CRN determinism).

Agents see only the raw schedule ("this facility charges these credits at
these hours"), never the era name or any policy history.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Tuple

import numpy as np

# Frozen period vocabulary (order = the code order used to vectorize over a
# population's per-agent departure period). Generic time-of-day bands.
PERIODS: Tuple[str, ...] = ("overnight", "am_peak", "pm_peak", "offpeak")
PERIOD_INDEX: Dict[str, int] = {p: i for i, p in enumerate(PERIODS)}

# Pre-perturbed per-trip pass-holder rates, in "credits" (world plan; relative
# structure preserved, absolute levels shifted so no real rate appears).
DEFAULT_RATES: Dict[str, float] = {
    "overnight": 1.1,
    "am_peak": 1.7,
    "pm_peak": 2.4,
    "offpeak": 1.4,
}

# Flat per-trip surcharge an agent WITHOUT a pass pays on top of the rate
# (the pass/casual-payer split; pre-perturbed "credits").
DEFAULT_NONPASS_SURCHARGE = 1.9

# Transfer-arena (cityk_cordon) per-crossing charge, in "credits" — the
# masked A8.5(i) schedule (TRANSFER_MASKING_NOTE derivation: real ladder ->
# hour-weighted period collapse -> per-period credits factors -> ±10% CRN
# perturbation; recorded in runs/transfer_schedule/manifest.json). The SAME
# schedule governs P1 and P3 (the real ladders are identical, so per-period
# P1:P3 credit ratios equal the real ratios exactly). No pass instrument
# exists in this arena (A8.5(ii)) — the surcharge is structurally zero.
CORDON_RATES: Dict[str, float] = {
    "overnight": 0.0,
    "am_peak": 2.2003,
    "pm_peak": 2.2423,
    "offpeak": 0.9928,
}

# Masked daily-cap constant (credits). RECORDED, not enforced by the
# per-trip charging machinery: the cap binds only at >= 4 peak-rate
# cordon crossings in one day, and adding a cap mechanism would be new
# method (A1.2). The understatement is stated in the schedule manifest.
CORDON_DAILY_CAP: float = 8.0723


def period_for_hour(hour: int) -> str:
    """Map an hour-of-day (0..23) to its toll period.

    overnight 21:00-04:59, am_peak 06:00-08:59, pm_peak 15:00-17:59, offpeak
    otherwise (05:00-05:59, 09:00-14:59, 18:00-20:59)."""
    h = int(hour) % 24
    if h >= 21 or h < 5:
        return "overnight"
    if 6 <= h < 9:
        return "am_peak"
    if 15 <= h < 18:
        return "pm_peak"
    return "offpeak"


@dataclass(frozen=True)
class TollSchedule:
    """Immutable time-of-day toll schedule (all amounts in credits).

    per_trip_toll(period, has_pass) = price_multiplier *
        (rate[period] + (0 if has_pass else nonpass_surcharge)).

    The multiplier scales the WHOLE per-trip charge, so multiplier 0 makes the
    facility free (the natural zero point of the E5 price sweep) and the charge
    is strictly increasing in the multiplier for every agent (rate + surcharge
    > 0) — which is what makes the monotone-diversion acceptance test clean.
    """

    rates: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_RATES))
    nonpass_surcharge: float = DEFAULT_NONPASS_SURCHARGE
    price_multiplier: float = 1.0

    def per_trip_toll(self, period: str, has_pass: bool) -> float:
        base = self.rates[period] + (0.0 if has_pass else self.nonpass_surcharge)
        return self.price_multiplier * base

    def with_multiplier(self, multiplier: float) -> "TollSchedule":
        """E5 hook: a new schedule at an arbitrary global price level."""
        return replace(self, price_multiplier=float(multiplier))

    def with_overrides(self, **rate_overrides: float) -> "TollSchedule":
        """A new schedule with some per-period rates replaced (others kept)."""
        merged = dict(self.rates)
        for period, rate in rate_overrides.items():
            if period not in PERIOD_INDEX:
                raise KeyError(f"unknown toll period {period!r}")
            merged[period] = float(rate)
        return replace(self, rates=merged)

    def rate_vector(self) -> np.ndarray:
        """Per-period pass-holder rates in PERIODS order (for vectorization)."""
        return np.array([self.rates[p] for p in PERIODS], dtype=float)

    def toll_array(self, period_codes: np.ndarray, has_pass: np.ndarray) -> np.ndarray:
        """Vectorized per-trip toll over a population: `period_codes` are
        PERIOD_INDEX integer codes, `has_pass` is a boolean array."""
        base = self.rate_vector()[period_codes]
        base = base + np.where(has_pass, 0.0, self.nonpass_surcharge)
        return self.price_multiplier * base


def default_schedule() -> TollSchedule:
    """The frozen M1 corridor schedule (pre-perturbed credits)."""
    return TollSchedule(rates=dict(DEFAULT_RATES),
                        nonpass_surcharge=DEFAULT_NONPASS_SURCHARGE)


def cordon_schedule() -> TollSchedule:
    """The frozen transfer-arena cordon schedule (masked credits, A8.5(i)).

    Surcharge zero: the arena has no pass instrument (A8.5(ii)); every
    crossing pays the period rate. The charge-off phases (P0/P2) use
    ``cordon_schedule().with_multiplier(0.0)`` so on/off is a price level,
    never a machinery change."""
    return TollSchedule(rates=dict(CORDON_RATES), nonpass_surcharge=0.0)


# ---------------------------------------------------------------------------
# M4 announced-onset notices (A4.2(ii)/(iii))
# ---------------------------------------------------------------------------

#: §7 A3.2 say-do price-prior CENTRAL, frozen (owner decision 2026-07-17,
#: sealed in calibration/e3_fit_manifest.json -> price_prior). The revealed
#: toll response runs ~2-3x the stated response; the slow brain's card
#: adaptation is a stated-like channel, so the ANNOUNCED charge it reasons
#: over is scaled by this factor while the WORLD charges the un-scaled
#: schedule (stimulus-side application; the route-choice/VoT channel never
#: sees it). Applied INSIDE the pipeline BEFORE the A4.3 SR 520 elasticity
#: fit, so the fitted VoT absorbs only the residual; BT1 runs the identical
#: corrected pipeline, and the E3(iii) uncorrected-ablation arm passes 1.0
#: here with everything else (elasticity included) unchanged.
SAY_DO_PRICE_CORRECTION = 2.5


def announcement_of(
    schedule: TollSchedule, say_do_price_correction: float = 1.0
) -> Dict[str, object]:
    """The masked announced-onset notice content (A4.2(ii)): the new per-period
    per-trip charge in credits, with the pass semantics. Everything here is
    already masked by construction (generic period names, pre-perturbed
    credits) — the renderer (grounding.render) turns it into prompt lines and
    the mask-lint gate re-checks the result.

    ``say_do_price_correction`` is the A3.2 seam: drivers on the corrected
    pipeline pass :data:`SAY_DO_PRICE_CORRECTION` so the slow brain sees the
    prior-corrected charge; the default 1.0 is the uncorrected stimulus (unit
    tests, the E3(iii) ablation arm). It scales ONLY this notice — never the
    schedule the world charges."""
    factor = float(say_do_price_correction)
    return {
        "kind": "toll_onset",
        "per_trip_credits": {
            p: factor * schedule.per_trip_toll(p, True) for p in PERIODS
        },
        "nonpass_surcharge_credits": (
            factor * schedule.price_multiplier * schedule.nonpass_surcharge
        ),
        "pass_semantics": (
            "a household pass covers car trips in the household vehicle only; "
            "trips as a passenger in someone else's vehicle pay the full charge"
        ),
    }


def placebo_announcement() -> Dict[str, object]:
    """The A4.2(iii) placebo notice: a content-free reconsideration cue with
    every price/toll field NULLED. It yokes the TRIGGER and nulls the REASON —
    it carries ZERO actionable content (no price, no cost change, no time
    change), so a rational re-optimization under it is a no-op and any change
    it induces is machinery drift, which is what the placebo arm measures."""
    return {
        "kind": "reconsideration",
        "per_trip_credits": None,
        "nonpass_surcharge_credits": None,
        "pass_semantics": None,
    }
