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
from typing import Tuple

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
