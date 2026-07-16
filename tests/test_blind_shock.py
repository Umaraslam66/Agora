"""Tests for evaluation/blind_shock.py — the E4/ΔQ scoring core (§7 A4.2).

Deterministic, synthetic data. Exercises the sealed rules: A1.1 coverage +
closeness applied to ΔQ, paired differencing, the tail-off bound, and the
two-leg drift rule with its exact boundary semantics.
"""
import numpy as np
import pytest

from evaluation import blind_shock as bs


def test_interval_and_coverage():
    ens = np.arange(0, 101, dtype=float)  # 0..100
    assert bs.interval(ens, 0.80) == pytest.approx((10.0, 90.0))
    assert bs.interval_coverage(ens, 50.0).covered is True
    assert bs.interval_coverage(ens, 5.0).covered is False
    assert bs.interval_coverage(ens, 95.0).covered is False
    assert bs.interval_coverage(ens, 10.0).covered is True   # boundary inclusive
    assert bs.interval_coverage(ens, 50.0).width == pytest.approx(80.0)


def test_closer_than_benchmark():
    # central = observed -> trivially closer than any offset benchmark
    c = bs.closer_than_benchmark(np.full(50, -28.0), -28.0, -45.0)
    assert c.closer is True and c.dist_central == 0.0 and c.dist_benchmark == 17.0
    # central -40 vs observed -28, benchmark -45: 12 < 17 -> closer
    assert bs.closer_than_benchmark(np.full(10, -40.0), -28.0, -45.0).closer is True
    # central -46: 18 > 17 -> NOT closer than the benchmark
    assert bs.closer_than_benchmark(np.full(10, -46.0), -28.0, -45.0).closer is False


def test_paired_delta_and_pairing_guard():
    d = bs.paired_delta(np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0]))
    assert list(d.delta) == [9.0, 18.0, 27.0]
    assert d.central == 18.0 and d.median == 18.0 and d.n == 3
    with pytest.raises(ValueError):
        bs.paired_delta(np.zeros(3), np.zeros(4))  # not CRN-paired 1:1


def test_tail_off_gap():
    g = bs.tail_off_gap(np.array([9.0, 18.0, 27.0]), np.array([8.0, 16.0, 24.0]))
    assert list(g.gap) == [1.0, 2.0, 3.0] and g.central == 2.0
    with pytest.raises(ValueError):
        bs.tail_off_gap(np.zeros(2), np.zeros(3))


def test_measured_floor():
    assert bs.measured_floor(np.full(4, 20.0), np.full(4, 2.0)) == pytest.approx(0.1)
    assert bs.measured_floor(np.full(4, 0.0), np.full(4, 2.0)) == float("inf")


def test_drift_anomaly_leg_is_strict_at_2x_floor():
    floor, toll = 0.1, np.full(50, 100.0)          # 2x floor = 0.2
    at = bs.drift_verdict(toll, np.full(50, 20.0), floor)   # ratio == 0.2 exactly
    assert at.ratio == pytest.approx(0.2)
    assert at.anomaly_leg is False and at.absolute_leg is False
    assert at.drift_dominated is False              # EXCEEDS is strict; 0.2 is not > 0.2
    above = bs.drift_verdict(toll, np.full(50, 21.0), floor)  # ratio 0.21
    assert above.anomaly_leg is True and above.drift_dominated is True


def test_drift_absolute_leg_catches_a_consistently_high_floor():
    # A high floor (0.3 -> 2x = 0.6) that the anomaly leg would bless; the
    # absolute leg (>= 0.5) catches the harm. This is the case E1's floor
    # doctrine would wrongly wave through.
    floor, toll = 0.3, np.full(50, 100.0)
    v = bs.drift_verdict(toll, np.full(50, 50.0), floor)     # ratio 0.5
    assert v.ratio == pytest.approx(0.5)
    assert v.anomaly_leg is False                   # 0.5 not > 0.6
    assert v.absolute_leg is True                   # 0.5 >= 0.5 (inclusive)
    assert v.drift_dominated is True
    below = bs.drift_verdict(toll, np.full(50, 49.0), floor)  # ratio 0.49
    assert below.anomaly_leg is False and below.absolute_leg is False
    assert below.drift_dominated is False           # 0.49 escapes both legs


def test_zero_containment_reported_but_never_triggers():
    # Placebo tightly positive -> 80% interval excludes zero (drift is
    # DETECTABLE) but the ratio is tiny -> NOT drift-dominated. Detectability
    # is not materiality (paired CRN excludes zero on trivial systematic drift).
    v = bs.drift_verdict(np.full(50, 1000.0), np.full(50, 10.0), floor=0.1)
    assert v.placebo_zero_contained is False        # interval [10,10] excludes 0
    assert v.absolute_leg is False and v.anomaly_leg is False
    assert v.drift_dominated is False               # ratio 0.01 -> not flagged
