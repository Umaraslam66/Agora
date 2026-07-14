"""Toll-period edge coverage: the schedule's hour->period mapping is the kind
of boundary logic that regresses silently, so every hour of the day is pinned
here, including the overnight wrap-around."""
import pytest

from world.tolling import DEFAULT_NONPASS_SURCHARGE, TollSchedule, period_for_hour

_EXPECTED = (
    # hour range (inclusive start, exclusive end), expected period
    ((0, 5), "overnight"),
    ((5, 6), "offpeak"),
    ((6, 9), "am_peak"),
    ((9, 15), "offpeak"),
    ((15, 18), "pm_peak"),
    ((18, 21), "offpeak"),
    ((21, 24), "overnight"),
)


def test_every_hour_maps_to_its_period():
    for (start, end), period in _EXPECTED:
        for hour in range(start, end):
            assert period_for_hour(hour) == period, hour


def test_wraparound_and_normalization():
    assert period_for_hour(23) == "overnight"
    assert period_for_hour(24) == period_for_hour(0) == "overnight"
    assert period_for_hour(4) == "overnight"
    assert period_for_hour(5) == "offpeak"


def test_nonpass_surcharge_and_multiplier_compose():
    sched = TollSchedule()
    base = sched.per_trip_toll("am_peak", has_pass=True)
    assert sched.per_trip_toll("am_peak", has_pass=False) == pytest.approx(
        base + DEFAULT_NONPASS_SURCHARGE
    )
    zeroed = sched.with_multiplier(0.0)
    assert zeroed.per_trip_toll("pm_peak", has_pass=False) == 0.0
