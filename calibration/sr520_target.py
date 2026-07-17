"""SR 520 aggregate calibration target (pre-registration §7 A3.3(a)/(b), A4.3).

HARNESS-SIDE ONLY. This module pins the observed SR 520 tolling response that
A3.3 authorizes as an in-window aggregate anchor, for (i) the habit-persistence
+ elasticity calibration (A4.3) and (ii) the A3.3(b) calibration rehearsal that
exercises the E4 machinery on a labeled, non-blind event. It must NEVER be
imported by ``agents/``, ``serving/``, or ``grounding/`` (agent-facing code):
it carries real (un-masked) figures about the calibration event, exactly as the
import-quarantined blind-truth series is kept out of agent-facing code — the
wall keeps such figures out of anything the agents can see.

Every value below is pinned from the SEALED §7 A3.3 text (which itself carries
the primary citations: WSDOT/Stantec 2019 T&R study Table 3.2, FY2012–FY2019;
the FY2013 monthly actuals; WSDOT Toll Division / IBTTA 2012 for the
forecast/actual pair). Nothing here is a new fact; anything A3.3 marks
UNVERIFIED is excluded. No monthly time series is fabricated — A3.3 pins the
aggregate drop and its drop-and-plateau shape, not a month-by-month series, so
only those are exposed; a real monthly series would be added here (with its
citation) if and when it is obtained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from world.tolling import DEFAULT_NONPASS_SURCHARGE, DEFAULT_RATES, TollSchedule, period_for_hour

#: Aggregate AADT drop band at tolling, drop-and-plateau with no recovery to
#: baseline through the calibration window (A3.3(a): "~36–40% AADT drop ... with
#: drop-and-plateau persistence"). Stored as positive fractions of baseline.
DROP_BAND: Tuple[float, float] = (0.36, 0.40)

#: Qualitative shape the calibration must reproduce (A3.3(a)): a monotone drop to
#: a plateau that persists (no drift back), under a PERMANENT level shift.
SHAPE = "drop_and_plateau_no_recovery"

#: Dynamics-calibration window: tolling start (2011-12) through the month before
#: the 2016-04 six-lane bridge opening (a capacity break), per the A3.3 confound
#: pins. (year, month) inclusive bounds.
WINDOW_START: Tuple[int, int] = (2011, 12)
WINDOW_END: Tuple[int, int] = (2016, 3)

#: A3.3(b) calibration-rehearsal forecast/actual pair (WSDOT Toll Division,
#: IBTTA 2012): the official pre-tolling forecast drop, and the realized drop
#: band. Positive fractions of baseline. Scored with E4 machinery as a LABELED
#: calibration exercise, never as a blind result.
REHEARSAL_FORECAST_DROP: float = 0.48
REHEARSAL_REALIZED_DROP_BAND: Tuple[float, float] = (0.35, 0.40)

#: Confounds pinned in A3.3 so they cannot become post-hoc excuses; carried as
#: metadata so the A4.3 E6 "confound-widened envelope" can be derived downstream.
CONFOUNDS: Tuple[str, ...] = (
    "concurrent transit-service boost (+90–130 daily bus trips at tolling start; "
    "observed response is joint price+service)",
    "six July-1 rate steps 2013–2018",
    "post-recession growth trend",
    "parallel-crossing construction during the window",
    "FY2018 overnight-tolling counting break",
    "new 6-lane bridge opened 2016-04 (capacity break; window ends 2016-03)",
)


@dataclass(frozen=True)
class SR520Target:
    """The pinned SR 520 calibration target (A3.3). All drops are positive
    fractions of the pre-toll baseline."""

    drop_band: Tuple[float, float] = DROP_BAND
    shape: str = SHAPE
    window_start: Tuple[int, int] = WINDOW_START
    window_end: Tuple[int, int] = WINDOW_END
    rehearsal_forecast_drop: float = REHEARSAL_FORECAST_DROP
    rehearsal_realized_drop_band: Tuple[float, float] = REHEARSAL_REALIZED_DROP_BAND
    confounds: Tuple[str, ...] = field(default=CONFOUNDS)

    @property
    def drop_midpoint(self) -> float:
        """Midpoint of the observed drop band, the point calibration target."""
        return sum(self.drop_band) / 2.0

    def within_window(self, year: int, month: int) -> bool:
        """Is (year, month) inside the dynamics-calibration window (inclusive)?"""
        return self.window_start <= (year, month) <= self.window_end


def sr520_target() -> SR520Target:
    """The frozen SR 520 calibration target pinned from §7 A3.3."""
    return SR520Target()


# ---------------------------------------------------------------------------
# A4.3 masked REHEARSAL schedule (owner ruling 2026-07-17; pre-M4 record §7.1)
# ---------------------------------------------------------------------------
# The A4.3 fit originally ran under the SAME masked M4 schedule BT1 uses, so
# the calibration mechanically centered the BT1 toll arm at the SR 520 level —
# E4 coverage would fail BY CONSTRUCTION (the real events differ: SR 520's
# opening rates were materially higher than the tunnel's). The owner adopted
# an SR 520-DERIVED masked rehearsal schedule: calibrate at an SR 520-like
# rate LEVEL and let the fitted VoT transfer through the utility model to the
# (lower) M4 schedule, turning the BT1 level into a genuine prediction.
#
# Real figures below are institutional facts inside the C0 wall and live in
# THIS quarantined module only; agents ever see only the derived credits.

#: SR 520 opening weekday toll ladder (Good To Go! pass, two-axle), dollars,
#: as (hour_start, hour_end, rate) half-open bands covering 0..24. Pinned from
#: WAC 468-270-071 (WSR 11-04-007, adopted 2011-01-05, effective 2011-12-03;
#: collection began 2011-12-29 05:00), figures corroborated by WSR 12-08-059's
#: strikethrough baseline and contemporaneous SDOT/press reporting.
SR520_OPENING_WEEKDAY_USD: Tuple[Tuple[int, int, float], ...] = (
    (0, 5, 0.00), (5, 6, 1.60), (6, 7, 2.80), (7, 9, 3.50), (9, 10, 2.80),
    (10, 14, 2.25), (14, 15, 2.80), (15, 18, 3.50), (18, 19, 2.80),
    (19, 21, 2.25), (21, 23, 1.60), (23, 24, 0.00),
)

#: SR 520 opening no-pass (Pay By Mail) surcharge, dollars: a constant +$1.50
#: at every non-zero band (same WAC table; SDOT: "$1.50 more than the posted
#: Good To Go! pass rate").
SR520_OPENING_NONPASS_SURCHARGE_USD: float = 1.50

#: The real SR 99 tunnel rates the masked M4 config schedule was derived from
#: (WSTC adoption release 2018-10-16; see the world plan): pass holders
#: $1.50 AM peak / $2.25 PM peak / $1.25 off-peak / $1.00 overnight, no-pass
#: (Pay By Mail) +$2.00. Pinned here ONLY to recover the per-period
#: credits-per-dollar factors of the existing masking, so both schedules
#: speak the same masked unit.
SR99_ADOPTED_RATES_USD: Dict[str, float] = {
    "overnight": 1.00,
    "am_peak": 1.50,
    "pm_peak": 2.25,
    "offpeak": 1.25,
}
SR99_NONPASS_SURCHARGE_USD: float = 2.00


def sr520_rehearsal_schedule() -> TollSchedule:
    """The masked A4.3 rehearsal schedule, in credits.

    Derivation (deterministic, no traffic data, no tuning dial):
    1. Collapse the SR 520 opening weekday ladder onto the four masked
       periods by HOUR-WEIGHTED mean over each period's hours (the masked
       world charges one rate per period; hour-weighting is the canonical
       collapse that needs no volume data — it slightly understates the
       traffic-weighted effective rate, stated in the adoption note).
    2. Map dollars -> credits with the SAME per-period credits-per-dollar
       factors the M4 config schedule's masking implies (DEFAULT_RATES /
       SR99 adopted dollars), so a credit means the same thing in both
       schedules and the rehearsal-to-M4 credit ratio per period equals the
       real-world dollar ratio — which is what lets the fitted VoT transfer.
    """
    hours_by_period: Dict[str, int] = {}
    usd_sum_by_period: Dict[str, float] = {}
    for hour in range(24):
        period = period_for_hour(hour)
        rate = next(r for lo, hi, r in SR520_OPENING_WEEKDAY_USD if lo <= hour < hi)
        hours_by_period[period] = hours_by_period.get(period, 0) + 1
        usd_sum_by_period[period] = usd_sum_by_period.get(period, 0.0) + rate

    rates = {}
    for period, usd_sum in usd_sum_by_period.items():
        usd_mean = usd_sum / hours_by_period[period]
        credits_per_usd = DEFAULT_RATES[period] / SR99_ADOPTED_RATES_USD[period]
        rates[period] = usd_mean * credits_per_usd

    surcharge = SR520_OPENING_NONPASS_SURCHARGE_USD * (
        DEFAULT_NONPASS_SURCHARGE / SR99_NONPASS_SURCHARGE_USD
    )
    return TollSchedule(rates=rates, nonpass_surcharge=surcharge)
