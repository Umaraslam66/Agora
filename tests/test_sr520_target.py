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
