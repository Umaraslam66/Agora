"""Transfer-arena (BT2) phase timeline and scoring windows — A8.1 constants.

ONE source of truth for the protocol plumbing shared by the P0 placebo-only
floor rehearsal (A8.3, blind-safe) and the BT2 assembly driver (single
firing). Owner-reviewed pre-firing per A8.1's window-constants discipline.

Derivation (from the truth series' PERIOD DEFINITIONS only — calendar
spans, which are pre-trial/institutional facts; no outcome quantity appears
here): the arena's real phases are P0 baseline (pre-charge months), P1
introduction (a seven-calendar-month charged trial beginning in the first
days of January), P2 removal (the twelve-month uncharged year that
followed), P3 return (the permanent reintroduction the following August).
The sim compresses uniformly at 10 sim days per real calendar month:

  P0  days   1- 40   4 baseline months; days 1-10 are warmup, scored 11-40
  P1  days  41-110   7 months, charge on; scored 61-100 = months 3-6 (the
                     truth series' stabilized reading excludes the first
                     two months' overreaction — its period definition,
                     carried; the excluded months are REPORTED as a
                     trajectory diagnostic, never scored)
  P2  days 111-230   12 months, charge off (E6's phase); scored over the
                     full off-year 111-230 (the truth residual is a
                     whole-period reading, rebound path included)
  P3  days 231-290   6 months, charge on permanently; scored 231-290 (the
                     truth series' first post-return calendar-year reading;
                     the first 10 days = first-month analog additionally
                     REPORTED as a diagnostic)

Transitions are ANNOUNCED onsets (A4.2 trigger, carried; A8.1): the slow
brain fires once per cordon-crossing agent at the start of each transition
day. The 2.5x say-do stimulus correction multiplies the announced charge;
at the P2 removal it therefore announces zero — mechanically harmless
(noted in the transfer inventory §2).

Charge on/off is a price LEVEL, never a machinery change:
``phase_multiplier`` maps each phase to the global price multiplier applied
to the frozen masked cordon schedule (world.tolling.cordon_schedule).
"""
from __future__ import annotations

SIM_DAYS_PER_REAL_MONTH: int = 10

#: Day indices are the loop's 0-indexed global days (BT1's convention:
#: day 0 is the first simulated day). All bounds inclusive.
PHASE_BOUNDS = {
    "P0": (0, 39),
    "P1": (40, 109),
    "P2": (110, 229),
    "P3": (230, 289),
}

#: Scored weekday windows per phase (inclusive day bounds).
SCORED_WINDOWS = {
    "P0": (10, 39),
    "P1": (60, 99),
    "P2": (110, 229),
    "P3": (230, 289),
}

#: Announced-transition days: each fires the A4.2 announced-onset trigger
#: once per cordon-crossing agent at the start of that day. The P0
#: placebo-only rehearsal fires its NULLED notice at these SAME offsets
#: (A8.3).
TRANSITION_DAYS = {"P1": 40, "P2": 110, "P3": 230}

#: Warmup days excluded from every quantity (BT1 discipline; days 0-9).
WARMUP_DAYS: int = 10

#: Total simulated days (0 .. TOTAL_DAYS-1).
TOTAL_DAYS: int = 290

#: Phase -> global price multiplier on the frozen cordon schedule.
PHASE_MULTIPLIER = {"P0": 0.0, "P1": 1.0, "P2": 0.0, "P3": 1.0}

#: Reported-only diagnostic windows (never scored, never a bar).
DIAGNOSTIC_WINDOWS = {
    "P1_overreaction": (40, 59),   # months 1-2 of the trial
    "P3_first_month": (230, 239),  # first-month return analog
}


def phase_of_day(day: int) -> str:
    for phase, (lo, hi) in PHASE_BOUNDS.items():
        if lo <= day <= hi:
            return phase
    raise ValueError(f"day {day} outside the BT2 timeline (0..{TOTAL_DAYS - 1})")
