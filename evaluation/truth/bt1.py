"""BT1 sealed answer key (pre-registration §7 A1.1) — QUARANTINED.

Transcribes, verbatim, the aggregate falsification quantities the SEALED
A1.1 text pins for BLIND TEST 1. Nothing here is a new fact: every figure
below appears in the public pre-registration (A1.5 discloses that the
outcomes are public knowledge; blindness is procedural — the wall forbids
these numbers from influencing any model input, not from being known).

This module lives in ``evaluation/truth/`` so the import-quarantine
(static scan + the package tripwire requiring AGORA_EVAL_CONTEXT=1)
mechanically keeps it out of every agent-facing package. It is imported by
exactly one caller: the BT1 assembly driver, at scoring time, under the
owner's explicit authorization.
"""
from __future__ import annotations

#: A1.1 scored quantity: change in average weekday whole-day tunnel volume
#: over BT1 (2019-11-09 .. 2020-02-29) relative to the pre-toll baseline.
#: Observed truth: -28% (three-month period ending 2020-02, WSDOT one-year
#: report). Stored as a positive drop fraction.
OBSERVED_DROP_3MO: float = 0.28

#: A1.1 secondary trajectory point: the first-two-weeks snapshot, -26%
#: (2019-11-12 .. 2019-11-22). REPORTED only — never a bar.
OBSERVED_DROP_2WK: float = 0.26

#: A1.1 benchmark: the official pre-tolling forecast in the reading most
#: favorable to the forecast, -45% (44,000 fewer daily trips off its own
#: 97,000 forecast base; deliberately conservative for revenue purposes).
FORECAST_BENCHMARK_DROP: float = 0.45

#: A1.1 pre-toll baseline: ~77,000 veh/weekday (WSDOT formal baseline
#: period 2019-09-23 .. 2019-10-31). Context only; the scored quantity is
#: the relative drop.
BASELINE_WEEKDAY_VOLUME: float = 77_000.0

#: The two sealed E4 pass conditions on the drift-corrected dQ (A1.1 +
#: A4.2): (i) OBSERVED_DROP_3MO inside the ensemble's 80% interval;
#: (ii) the central prediction strictly closer to OBSERVED_DROP_3MO than
#: FORECAST_BENCHMARK_DROP is. Encoded here as data so the driver cannot
#: restate them.
E4_INTERVAL_LEVEL: float = 0.80
