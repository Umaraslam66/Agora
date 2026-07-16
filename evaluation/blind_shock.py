"""E4 / ΔQ blind-shock scoring core (pre-registration §3 E4, A1.1; §7 A4.2).

The scoring kernels for the toll shock (BT1) and the SR 520 calibration
rehearsal (A3.3(b)). This module is deliberately GENERIC and target-agnostic:
the observed value and the benchmark are ALWAYS passed in by the caller, and
this module NEVER imports ``evaluation.truth`` (the import-quarantined BT1 truth
series) — exactly the discipline ``evaluation.e1`` states for itself. It holds
no sealed numbers; it holds the sealed *rules*.

Sealed protocol implemented here:

* **ΔQ estimand (A4.2).** The reported response is ΔQ = Q_toll − Q_placebo,
  PAIRED across CRN-matched ensemble members. Under the placebo's identifying
  (additivity) assumption ΔQ is the total drift-corrected response the observed
  −28% measures (:func:`paired_delta`). E4's frozen bars (A1.1: observed inside
  the ensemble's 80% interval; central prediction strictly closer than the −45%
  benchmark) apply to ΔQ (:func:`interval_coverage`, :func:`closer_than_benchmark`).
* **Tail-off ablation (A4.2).** ΔQ at T5 is computed with the post-onset
  time-surprise trigger ON (headline) and OFF (bound), against the same placebo;
  the gap bounds the uncontrolled tail-drift channel (:func:`tail_off_gap`).
* **Two-leg drift rule (A4.2).** The placebo/toll response-magnitude ratio on
  the SR 520 rehearsal is the DRIFT FLOOR, reported standalone
  (:func:`measured_floor`). A BT1 quantity is DRIFT-DOMINATED iff EITHER leg
  trips — (anomaly) ratio > 2× floor, or (absolute) ratio ≥ 0.5 — where the
  absolute leg catches a consistently high floor the anomaly leg would bless
  (E1's 2×-floor doctrine does NOT transfer: a noise floor is irreducible, a
  drift floor is a defect). The placebo 80%-interval zero-containment is
  REPORTED but never enters the verdict — detectability is not materiality
  (:func:`drift_verdict`).

``central prediction`` = the ensemble MEAN, matching ``evaluation.e1``'s point
convention (bootstrap median returned alongside as a robustness check). 80%
intervals use ``numpy.percentile`` (linear), matching the paired-bootstrap CI in
``evaluation.e1.paired_bootstrap``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Frozen protocol constants (A4.2 / A1.1)
# ---------------------------------------------------------------------------

#: The ensemble interval level for E4 coverage (§3 E4 / A1.1: the 80% interval).
INTERVAL_LEVEL = 0.80

#: Anomaly leg (A4.2): drift-dominated when the BT1 placebo/toll magnitude ratio
#: strictly EXCEEDS this multiple of the SR 520-rehearsal drift floor.
DRIFT_ANOMALY_MULTIPLE = 2.0

#: Absolute leg (A4.2): drift-dominated when the ratio is >= this cap, where ΔQ
#: has become a residual of two comparable numbers (at 0.5, ΔQ equals the drift
#: it subtracts — signal-to-drift 1:1). Inclusive at the boundary.
DRIFT_ABSOLUTE_CAP = 0.5


# ---------------------------------------------------------------------------
# Small numeric kernels
# ---------------------------------------------------------------------------

def interval(ensemble, level: float = INTERVAL_LEVEL) -> Tuple[float, float]:
    """The central ``level`` interval of an ensemble via linear percentiles
    (matching ``e1.paired_bootstrap``). For 0.80 this is the [10, 90] band."""
    a = np.asarray(ensemble, dtype=float)
    lo_p = (1.0 - level) / 2.0 * 100.0
    hi_p = (1.0 + level) / 2.0 * 100.0
    return float(np.percentile(a, lo_p)), float(np.percentile(a, hi_p))


def _central(ensemble) -> float:
    """The ensemble central prediction = mean (e1's point convention)."""
    return float(np.mean(np.asarray(ensemble, dtype=float)))


def _magnitude_ratio(toll_ensemble, placebo_ensemble) -> float:
    """|central(placebo)| / |central(toll)| — the placebo/toll response-magnitude
    ratio. ``inf`` when the toll central magnitude is exactly zero (degenerate
    guard; a real toll response is never exactly zero)."""
    tc = abs(_central(toll_ensemble))
    pc = abs(_central(placebo_ensemble))
    return float("inf") if tc == 0.0 else pc / tc


# ---------------------------------------------------------------------------
# E4 coverage + closeness (A1.1), applied to ΔQ (A4.2)
# ---------------------------------------------------------------------------

@dataclass
class Coverage:
    lo: float
    hi: float
    observed: float
    covered: bool
    width: float
    level: float


def interval_coverage(ensemble, observed: float, level: float = INTERVAL_LEVEL) -> Coverage:
    """A1.1 coverage: does ``observed`` fall inside the ensemble's ``level``
    interval? ``width`` is the sharpness input for the E4(iii) interval-honesty
    check (an interval that covers far more than ``level`` of the time across
    quantities is overwide — that across-quantities judgment lives with the
    caller; this reports the per-quantity width)."""
    lo, hi = interval(ensemble, level)
    return Coverage(lo, hi, float(observed), bool(lo <= observed <= hi), hi - lo, level)


@dataclass
class Closeness:
    central: float
    observed: float
    dist_central: float
    dist_benchmark: float
    closer: bool


def closer_than_benchmark(ensemble, observed: float, benchmark: float) -> Closeness:
    """A1.1: is the ensemble central prediction STRICTLY closer to ``observed``
    than the ``benchmark`` (the −45% pre-tolling forecast) is?"""
    c = _central(ensemble)
    dc = abs(c - observed)
    db = abs(benchmark - observed)
    return Closeness(c, float(observed), dc, db, bool(dc < db))


# ---------------------------------------------------------------------------
# ΔQ paired differencing + tail-off bound (A4.2)
# ---------------------------------------------------------------------------

@dataclass
class PairedDelta:
    delta: np.ndarray
    central: float
    median: float
    lo: float
    hi: float
    n: int


def paired_delta(toll_ensemble, placebo_ensemble, level: float = INTERVAL_LEVEL) -> PairedDelta:
    """ΔQ = Q_toll − Q_placebo, paired per CRN-matched ensemble member (A4.2).
    The two ensembles MUST be 1:1 CRN-paired (equal shape); paired differencing
    is what removes the shared route-noise/machinery variance."""
    t = np.asarray(toll_ensemble, dtype=float)
    p = np.asarray(placebo_ensemble, dtype=float)
    if t.shape != p.shape:
        raise ValueError("toll and placebo ensembles must be CRN-paired 1:1 (equal shape)")
    d = t - p
    lo, hi = interval(d, level)
    return PairedDelta(d, float(d.mean()), float(np.median(d)), lo, hi, int(d.size))


@dataclass
class TailOffGap:
    gap: np.ndarray
    central: float
    lo: float
    hi: float


def tail_off_gap(delta_tail_on, delta_tail_off, level: float = INTERVAL_LEVEL) -> TailOffGap:
    """The T5 tail-off bound (A4.2): the paired gap between ΔQ with the post-onset
    time-surprise trigger ON (headline) and OFF (bound), both against the same
    placebo. Bounds the tail channel's total contribution to ΔQ, hence caps the
    uncontrolled tail-drift. Inputs are CRN-paired ΔQ ensembles (equal shape)."""
    on = np.asarray(delta_tail_on, dtype=float)
    off = np.asarray(delta_tail_off, dtype=float)
    if on.shape != off.shape:
        raise ValueError("tail-on and tail-off ΔQ must be CRN-paired 1:1 (equal shape)")
    g = on - off
    lo, hi = interval(g, level)
    return TailOffGap(g, float(g.mean()), lo, hi)


# ---------------------------------------------------------------------------
# Two-leg drift rule (A4.2)
# ---------------------------------------------------------------------------

def measured_floor(toll_ensemble, placebo_ensemble) -> float:
    """The SR 520-rehearsal DRIFT FLOOR = placebo/toll response-magnitude ratio,
    reported as a standalone quantity before BT1 fires (A4.2)."""
    return _magnitude_ratio(toll_ensemble, placebo_ensemble)


@dataclass
class DriftVerdict:
    ratio: float
    floor: float
    anomaly_leg: bool
    absolute_leg: bool
    drift_dominated: bool
    placebo_lo: float
    placebo_hi: float
    placebo_zero_contained: bool


def drift_verdict(
    toll_ensemble,
    placebo_ensemble,
    floor: float,
    level: float = INTERVAL_LEVEL,
) -> DriftVerdict:
    """The two-leg drift rule (A4.2). ``floor`` is the SR 520-rehearsal floor
    from :func:`measured_floor`. DRIFT-DOMINATED iff the BT1 placebo/toll
    magnitude ratio EXCEEDS 2× the floor (anomaly leg, strict) OR is ≥ 0.5
    (absolute leg, inclusive). The placebo's 80%-interval zero-containment is
    REPORTED but does NOT enter the verdict — under paired CRN a trivial
    systematic drift excludes zero, so detectability is not materiality."""
    ratio = _magnitude_ratio(toll_ensemble, placebo_ensemble)
    anomaly = bool(ratio > DRIFT_ANOMALY_MULTIPLE * floor)
    absolute = bool(ratio >= DRIFT_ABSOLUTE_CAP)
    plo, phi = interval(placebo_ensemble, level)
    zero_contained = bool(plo <= 0.0 <= phi)
    return DriftVerdict(
        ratio=float(ratio),
        floor=float(floor),
        anomaly_leg=anomaly,
        absolute_leg=absolute,
        drift_dominated=bool(anomaly or absolute),
        placebo_lo=plo,
        placebo_hi=phi,
        placebo_zero_contained=zero_contained,
    )
