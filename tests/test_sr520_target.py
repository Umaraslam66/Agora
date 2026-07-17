"""Tests for calibration/sr520_target.py — the pinned SR 520 anchor (§7 A3.3).

Asserts the module reproduces exactly what the sealed A3.3 text pins, and that
it stays harness-side (never imports agent-facing code).
"""
import pytest

from calibration import sr520_target as sr


def test_pinned_values_match_a33():
    t = sr.sr520_target()
    assert t.drop_band == (0.36, 0.40)                 # A3.3(a) ~36–40% AADT drop
    assert t.drop_midpoint == pytest.approx(0.38)
    assert t.shape == "drop_and_plateau_no_recovery"
    assert t.window_start == (2011, 12)                # tolling start
    assert t.window_end == (2016, 3)                   # before 2016-04 capacity break
    assert t.rehearsal_forecast_drop == 0.48           # A3.3(b) forecast
    assert t.rehearsal_realized_drop_band == (0.35, 0.40)


def test_window_membership_inclusive():
    t = sr.sr520_target()
    assert t.within_window(2013, 6) is True
    assert t.within_window(2011, 12) is True           # inclusive start
    assert t.within_window(2016, 3) is True            # inclusive end
    assert t.within_window(2016, 4) is False           # capacity break excluded
    assert t.within_window(2011, 11) is False          # before tolling


def test_confounds_pinned():
    t = sr.sr520_target()
    assert len(t.confounds) >= 5
    joined = " ".join(t.confounds).lower()
    assert "transit" in joined
    assert "rate step" in joined
    assert "2016-04" in joined


def test_harness_side_only():
    # Must carry no import of agent-facing code (wall discipline).
    src = open(sr.__file__, encoding="utf-8").read()
    for forbidden in (
        "import agents", "from agents",
        "import serving", "from serving",
        "import grounding", "from grounding",
    ):
        assert forbidden not in src


# ---------------------------------------------------------------------------
# A4.3 rehearsal schedule (owner ruling 2026-07-17; docs/REHEARSAL_SCHEDULE_NOTE.md)
# ---------------------------------------------------------------------------

def test_rehearsal_schedule_derivation_is_hour_weighted_and_unit_consistent():
    """Per-period credits must equal the hour-weighted SR 520 dollar mean times
    the SAME credits-per-dollar factor the M4 masking implies — the unit
    consistency that lets the fitted VoT transfer between schedules."""
    from world.tolling import DEFAULT_RATES, PERIODS, period_for_hour

    sched = sr.sr520_rehearsal_schedule()
    assert set(sched.rates) == set(PERIODS)
    for period in PERIODS:
        hours = [h for h in range(24) if period_for_hour(h) == period]
        usd = [
            next(r for lo, hi, r in sr.SR520_OPENING_WEEKDAY_USD if lo <= h < hi)
            for h in hours
        ]
        usd_mean = sum(usd) / len(usd)
        factor = DEFAULT_RATES[period] / sr.SR99_ADOPTED_RATES_USD[period]
        assert sched.rates[period] == pytest.approx(usd_mean * factor)


def test_rehearsal_schedule_level_materially_above_m4_schedule():
    """The whole point of the ruling: an SR 520-LIKE level, materially higher
    than the M4 schedule (the contemporaneous characterization was 'about
    double the tunnel'), so the BT1 level becomes a genuine prediction."""
    from world.config import cityk_corridor

    sched = sr.sr520_rehearsal_schedule()
    m4 = cityk_corridor().toll_schedule
    weights = {"overnight": 0.10, "am_peak": 0.28, "pm_peak": 0.24, "offpeak": 0.38}
    reh = sum(weights[p] * sched.rates[p] for p in weights)
    cfg = sum(weights[p] * m4.rates[p] for p in weights)
    assert 1.5 <= reh / cfg <= 2.2
    # peak rates individually well above the M4 peaks; overnight below
    # (SR 520 was free overnight at opening)
    assert sched.rates["am_peak"] > m4.rates["am_peak"] * 1.5
    assert sched.rates["pm_peak"] > m4.rates["pm_peak"] * 1.3
    assert sched.rates["overnight"] < m4.rates["overnight"]


def test_rehearsal_schedule_surcharge_mapped_like_m4_surcharge():
    from world.tolling import DEFAULT_NONPASS_SURCHARGE

    sched = sr.sr520_rehearsal_schedule()
    factor = DEFAULT_NONPASS_SURCHARGE / sr.SR99_NONPASS_SURCHARGE_USD
    assert sched.nonpass_surcharge == pytest.approx(
        sr.SR520_OPENING_NONPASS_SURCHARGE_USD * factor
    )


def test_sr520_ladder_covers_24h_and_matches_pinned_wac_facts():
    covered = sorted(
        h for lo, hi, _ in sr.SR520_OPENING_WEEKDAY_USD for h in range(lo, hi)
    )
    assert covered == list(range(24))
    rates = {(lo, hi): r for lo, hi, r in sr.SR520_OPENING_WEEKDAY_USD}
    assert rates[(7, 9)] == 3.50 and rates[(15, 18)] == 3.50  # WAC peaks
    assert rates[(0, 5)] == 0.0 and rates[(23, 24)] == 0.0    # free overnight
    assert sr.SR520_OPENING_NONPASS_SURCHARGE_USD == 1.50
