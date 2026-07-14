"""Tests for agents/habit_memory.py — the habit-strength memory substrate.

Every numeric expectation below is HAND-COMPUTED from the ported
formulas (see the module docstring of agents/habit_memory.py):

    ema'  = (1 - alpha) * ema + alpha * x          alpha = 0.3, ema0 = 0.0
    z     = |realized - expected| / sigma          sigma = 10.0 (0 if sigma <= 0)
    alpha_eff (weighting ON)
          = min(alpha_base * (1 + gain * min(z, z_cap) / z_cap), alpha_max)
    outcome: delta <= -5 -> -1, delta >= +5 -> +1, else 0 (inclusive)
    shock: replace iff empty OR z strictly greater OR age > window(30);
           ages only on untouched-day boundaries; expires when age > 30
    trailing counters: 30-day capped deque; k-day count sums the newest
           min(k, days-so-far); None only with no history at all
"""

import json

import pytest

from agents.habit_memory import (
    MAX_TRAILING_DAYS,
    SHORT_TRAILING_DAYS,
    SHOCK_WINDOW_DAYS,
    SURPRISE_Z_THRESHOLD,
    DEFAULT_CONFIG,
    EMPTY_SHOCK,
    HabitCounter,
    HabitMemory,
    LastShock,
    SubstrateCell,
    SubstrateConfig,
    encode_outcome,
    is_surprise,
    prediction_error_z,
)

TOL = 1e-9


# =====================================================================
# EMA update
# =====================================================================

class TestEmaUpdate:
    def test_empty_cell_initialization(self):
        cell = SubstrateCell.empty()
        assert cell.ema_realized_minutes == 0.0
        assert cell.ema_delta_vs_expectation == 0.0
        assert cell.n == 0
        assert cell.last_outcome == 0
        assert cell.days_since_tried == 0
        assert cell.last_shock.is_empty()
        assert cell.last_shock.age_in_days == -1

    def test_first_observation_starts_from_zero_seed(self):
        # EMAs are seeded at 0.0, so the first observation yields
        # alpha * x, NOT the raw observation:
        #   ema_r = 0.3 * 30 = 9.0 ; ema_d = 0.3 * (30 - 20) = 3.0
        cell = SubstrateCell.empty().with_observation(30.0, 20.0)
        assert cell.ema_realized_minutes == pytest.approx(9.0, abs=TOL)
        assert cell.ema_delta_vs_expectation == pytest.approx(3.0, abs=TOL)
        assert cell.n == 1
        assert cell.days_since_tried == 0

    def test_ema_sequence_hand_computed(self):
        # obs1 (30, 20): d=+10  ema_r = 0.3*30                  = 9.0
        #                       ema_d = 0.3*10                  = 3.0
        # obs2 (24, 20): d=+4   ema_r = 0.7*9    + 0.3*24       = 13.5
        #                       ema_d = 0.7*3    + 0.3*4        = 3.3
        # obs3 (10, 20): d=-10  ema_r = 0.7*13.5 + 0.3*10       = 12.45
        #                       ema_d = 0.7*3.3  + 0.3*(-10)    = -0.69
        # obs4 (20, 20): d=0    ema_r = 0.7*12.45 + 0.3*20      = 14.715
        #                       ema_d = 0.7*(-0.69)             = -0.483
        # obs5 (16, 20): d=-4   ema_r = 0.7*14.715 + 0.3*16     = 15.1005
        #                       ema_d = 0.7*(-0.483) + 0.3*(-4) = -1.5381
        # Closed-form cross-check for ema_r after obs3:
        #   0.3*(0.7^2*30 + 0.7*24 + 10) = 0.3*41.5 = 12.45  ✓
        cell = SubstrateCell.empty()
        cell = cell.with_observation(30.0, 20.0)
        cell = cell.with_observation(24.0, 20.0)
        assert cell.ema_realized_minutes == pytest.approx(13.5, abs=TOL)
        assert cell.ema_delta_vs_expectation == pytest.approx(3.3, abs=TOL)
        cell = cell.with_observation(10.0, 20.0)
        assert cell.ema_realized_minutes == pytest.approx(12.45, abs=TOL)
        assert cell.ema_delta_vs_expectation == pytest.approx(-0.69, abs=TOL)
        cell = cell.with_observation(20.0, 20.0)
        cell = cell.with_observation(16.0, 20.0)
        assert cell.ema_realized_minutes == pytest.approx(15.1005, abs=TOL)
        assert cell.ema_delta_vs_expectation == pytest.approx(-1.5381, abs=TOL)
        assert cell.n == 5

    def test_surprise_weighting_off_is_plain_ema_kill_switch(self):
        # The ported kill-switch guarantee: weighting OFF (the default)
        # must be byte-identical to the plain EMA.
        explicit_off = SubstrateConfig(surprise_weighting=False)
        a = SubstrateCell.empty().with_observation(60.0, 20.0)
        b = SubstrateCell.empty().with_observation(60.0, 20.0, explicit_off)
        assert a == b
        assert a.ema_realized_minutes == pytest.approx(0.3 * 60.0, abs=TOL)

    def test_surprise_weighted_alpha_hand_computed(self):
        on = SubstrateConfig(surprise_weighting=True)
        # (60, 20): d=40, z=4.0, capped to 3.0
        #   alpha = min(0.3*(1 + 1.0*3/3), 0.9) = min(0.6, 0.9) = 0.6
        #   ema_r = 0.6*60 = 36.0
        cell = SubstrateCell.empty().with_observation(60.0, 20.0, on)
        assert cell.ema_realized_minutes == pytest.approx(36.0, abs=TOL)
        # (35, 20): d=15, z=1.5 (below cap)
        #   alpha = 0.3*(1 + 1.5/3) = 0.45 ; ema_r = 0.45*35 = 15.75
        cell = SubstrateCell.empty().with_observation(35.0, 20.0, on)
        assert cell.ema_realized_minutes == pytest.approx(15.75, abs=TOL)

    def test_surprise_weighted_alpha_max_clamp(self):
        # gain=5: alpha_eff = 0.3*(1 + 5*3/3) = 1.8 -> clamped to 0.9
        #   ema_r = 0.9*60 = 54.0
        cfg = SubstrateConfig(surprise_weighting=True, surprise_gain=5.0)
        cell = SubstrateCell.empty().with_observation(60.0, 20.0, cfg)
        assert cell.ema_realized_minutes == pytest.approx(54.0, abs=TOL)


# =====================================================================
# Outcome encoding — inclusive +-5.0 boundaries
# =====================================================================

class TestOutcomeEncoding:
    def test_boundaries(self):
        assert encode_outcome(-5.0) == -1          # inclusive
        assert encode_outcome(-4.999999999) == 0
        assert encode_outcome(0.0) == 0
        assert encode_outcome(4.999999999) == 0
        assert encode_outcome(5.0) == 1            # inclusive
        assert encode_outcome(12.0) == 1

    def test_cell_records_last_outcome(self):
        cell = SubstrateCell.empty().with_observation(25.0, 20.0)  # d=+5.0
        assert cell.last_outcome == 1
        cell = cell.with_observation(24.9, 20.0)                    # d=+4.9
        assert cell.last_outcome == 0
        cell = cell.with_observation(15.0, 20.0)                    # d=-5.0
        assert cell.last_outcome == -1


# =====================================================================
# Shock record
# =====================================================================

class TestShock:
    def test_first_observation_always_records_shock_even_at_zero_z(self):
        cell = SubstrateCell.empty().with_observation(20.0, 20.0)  # d=0, z=0
        assert not cell.last_shock.is_empty()
        assert cell.last_shock.peak_abs_z == pytest.approx(0.0, abs=TOL)
        assert cell.last_shock.age_in_days == 0
        assert cell.last_shock.delta_minutes == pytest.approx(0.0, abs=TOL)

    def test_replacement_is_strictly_greater(self):
        cell = SubstrateCell.empty().with_observation(30.0, 20.0)  # d=+10, z=1.0
        assert cell.last_shock.peak_abs_z == pytest.approx(1.0, abs=TOL)
        assert cell.last_shock.delta_minutes == pytest.approx(10.0, abs=TOL)
        # Equal |z| (d=-10, z=1.0): NOT replaced — strict >, old +10 kept.
        cell = cell.with_observation(10.0, 20.0)
        assert cell.last_shock.delta_minutes == pytest.approx(10.0, abs=TOL)
        # Strictly greater (d=+12, z=1.2): replaced.
        cell = cell.with_observation(32.0, 20.0)
        assert cell.last_shock.peak_abs_z == pytest.approx(1.2, abs=TOL)
        assert cell.last_shock.delta_minutes == pytest.approx(12.0, abs=TOL)
        # Greater again, negative delta (d=-20, z=2.0): replaced.
        cell = cell.with_observation(0.0, 20.0)
        assert cell.last_shock.peak_abs_z == pytest.approx(2.0, abs=TOL)
        assert cell.last_shock.delta_minutes == pytest.approx(-20.0, abs=TOL)

    def test_aging_and_expiry_boundary(self):
        # A shock survives ages 0..30 (window = 30) and expires on the
        # tick that would take it to 31 (new_age > window -> EMPTY).
        cell = SubstrateCell.empty().with_observation(30.0, 20.0)
        for _ in range(SHOCK_WINDOW_DAYS):
            cell = cell.aged()
        assert cell.last_shock.age_in_days == 30
        assert not cell.last_shock.is_empty()
        assert cell.days_since_tried == 30
        cell = cell.aged()  # 31st tick
        assert cell.last_shock.is_empty()
        assert cell.days_since_tried == 31

    def test_empty_shock_stays_empty_when_aged(self):
        assert EMPTY_SHOCK.aged() is EMPTY_SHOCK
        assert LastShock().aged().is_empty()

    def test_stale_shock_replaced_even_by_smaller_z(self):
        # Defensive ported branch: age > window in with_observation
        # forces replacement regardless of peak comparison.
        stale = SubstrateCell(
            ema_realized_minutes=10.0,
            ema_delta_vs_expectation=0.0,
            n=5,
            last_outcome=0,
            days_since_tried=31,
            last_shock=LastShock(peak_abs_z=5.0, age_in_days=31, delta_minutes=50.0),
        )
        cell = stale.with_observation(21.0, 20.0)  # d=+1, z=0.1 << 5.0
        assert cell.last_shock.peak_abs_z == pytest.approx(0.1, abs=TOL)
        assert cell.last_shock.age_in_days == 0
        assert cell.last_shock.delta_minutes == pytest.approx(1.0, abs=TOL)

    def test_sigma_zero_guard(self):
        # sigma <= 0 -> z = 0.0 (ported ternary guard), no division error.
        cfg = SubstrateConfig(sigma_prior_minutes=0.0)
        cell = SubstrateCell.empty().with_observation(60.0, 20.0, cfg)
        assert cell.last_shock.peak_abs_z == pytest.approx(0.0, abs=TOL)
        assert prediction_error_z(60.0, 20.0, 0.0) == 0.0


# =====================================================================
# Surprise hook (slow-brain trigger)
# =====================================================================

class TestSurpriseHook:
    def test_threshold_boundary_inclusive(self):
        # Default threshold: z >= 0.5 (= 5 min at sigma 10), inclusive —
        # derived from the outcome encoder's inclusive +-5.0 boundary.
        assert SURPRISE_Z_THRESHOLD == pytest.approx(0.5, abs=TOL)
        assert is_surprise(25.0, 20.0) is True        # z = 0.5 exactly
        assert is_surprise(15.0, 20.0) is True        # z = 0.5, early side
        assert is_surprise(24.999999, 20.0) is False  # just below
        assert is_surprise(20.0, 20.0) is False

    def test_prediction_error_z(self):
        assert prediction_error_z(30.0, 20.0) == pytest.approx(1.0, abs=TOL)
        assert prediction_error_z(10.0, 20.0) == pytest.approx(1.0, abs=TOL)

    def test_observe_returns_surprise_and_updates_cell(self):
        mem = HabitMemory()
        mem.begin_day()
        assert mem.observe("car|b2|am", 30.0, 20.0) is True   # z=1.0
        assert mem.observe("car|b2|am", 21.0, 20.0) is False  # z=0.1
        cell = mem.cell("car|b2|am")
        assert cell.n == 2
        # ema_r: 0.3*30 = 9.0 ; then 0.7*9 + 0.3*21 = 12.6
        assert cell.ema_realized_minutes == pytest.approx(12.6, abs=TOL)


# =====================================================================
# Day cycle: touched cells do not age
# =====================================================================

class TestDayCycle:
    def test_touched_cells_keep_state_untouched_cells_age(self):
        mem = HabitMemory()
        # Day 1: both cells observed.
        mem.begin_day()
        mem.observe("A", 30.0, 20.0)
        mem.observe("B", 30.0, 20.0)
        mem.end_day()
        assert mem.cell("A").days_since_tried == 0
        assert mem.cell("B").days_since_tried == 0
        # Day 2: only A observed. B ages; A keeps observation state.
        mem.begin_day()
        mem.observe("A", 30.0, 20.0)
        mem.end_day()
        assert mem.cell("A").days_since_tried == 0
        assert mem.cell("A").last_shock.age_in_days == 0
        assert mem.cell("B").days_since_tried == 1
        assert mem.cell("B").last_shock.age_in_days == 1


# =====================================================================
# Trailing counters
# =====================================================================

class TestTrailingCounters:
    def test_none_before_any_history(self):
        mem = HabitMemory()
        assert mem.has_trailing_history is False
        assert mem.trailing_count("transit") is None
        assert mem.trailing_count("walk", SHORT_TRAILING_DAYS) is None
        assert mem.usual_work_mode() is None

    def test_scripted_sequence(self):
        mem = HabitMemory()
        # Day 1: 2 pt trips (1 to work), 1 walk trip.
        mem.record_daily_tally({"transit": 2, "walk": 1}, {"transit": 1})
        assert mem.trailing_count("transit") == 2
        assert mem.trailing_count("walk", SHORT_TRAILING_DAYS) == 1
        assert mem.trailing_count("car") == 0  # history exists -> 0, not None
        # Days 2-8: 1 car work trip per day (7 days).
        for _ in range(7):
            mem.record_daily_tally({"car": 1}, {"car": 1})
        assert mem.trailing_count("transit") == 2  # still inside 30-day window
        # The walk day has aged out of the 7-day window (days 2-8 fill it).
        assert mem.trailing_count("walk", SHORT_TRAILING_DAYS) == 0
        assert mem.trailing_count("car", SHORT_TRAILING_DAYS) == 7
        # Modal work mode: car 7 vs pt 1.
        assert mem.usual_work_mode() == "car"
        # Day 9: a quiet day is a real day in the window, not a gap.
        mem.record_daily_tally({}, {})
        assert mem.has_trailing_history is True
        # 7-day window is now days 3-9: six car days + the quiet day.
        assert mem.trailing_count("car", SHORT_TRAILING_DAYS) == 6
        assert mem.trailing_count("transit") == 2

    def test_window_hard_cap_at_30_days(self):
        mem = HabitMemory()
        for _ in range(35):
            mem.record_daily_tally({"bike": 1}, {})
        assert mem.trailing_count("bike") == MAX_TRAILING_DAYS  # 30, not 35
        assert mem.trailing_count("bike", SHORT_TRAILING_DAYS) == 7
        # The deque itself is capped: oldest 5 days were dropped.
        assert len(mem.to_dict()["tally"]["days"]) == MAX_TRAILING_DAYS

    def test_usual_work_mode_tie_break_and_none_cases(self):
        # Tie pt=2 vs car=2 -> pt wins (fixed priority order, strict >).
        mem = HabitMemory()
        mem.record_daily_tally({"car": 2}, {"car": 2})
        mem.record_daily_tally({"transit": 2}, {"transit": 2})
        assert mem.usual_work_mode() == "transit"
        # History exists but no work trips at all -> None.
        mem2 = HabitMemory()
        mem2.record_daily_tally({"walk": 3}, {})
        assert mem2.usual_work_mode() is None


# =====================================================================
# Serialization
# =====================================================================

class TestSerialization:
    def test_memory_roundtrip_through_json(self):
        mem = HabitMemory()
        mem.begin_day()
        mem.observe("car|b2|am", 30.0, 20.0)
        mem.observe("pt|b2|am", 55.0, 35.0)
        mem.record_daily_tally({"car": 1, "transit": 1}, {"car": 1})
        mem.end_day()

        payload = json.dumps(mem.to_dict())        # must be JSON-clean
        restored = HabitMemory.from_dict(json.loads(payload))

        assert restored.config == mem.config
        for key in mem.keys():
            assert restored.cell(key) == mem.cell(key)
        assert restored.trailing_count("car") == mem.trailing_count("car")
        assert restored.usual_work_mode() == mem.usual_work_mode()
        # Continued evolution is identical on both instances.
        mem.begin_day()
        restored.begin_day()
        assert (
            mem.observe("car|b2|am", 26.0, 20.0)
            == restored.observe("car|b2|am", 26.0, 20.0)
        )
        assert restored.cell("car|b2|am") == mem.cell("car|b2|am")

    def test_habit_counter_roundtrip_through_json(self):
        counter = HabitCounter()
        for _ in range(5):
            counter.record_day(True)
        for _ in range(2):
            counter.record_day(False)
        assert counter.strength == 3          # 5 - 2
        assert counter.total_days_followed == 5
        assert counter.days_observed == 7
        assert counter.days_followed_in_window == 5

        restored = HabitCounter.from_dict(json.loads(json.dumps(counter.to_dict())))
        assert restored.to_dict() == counter.to_dict()
        counter.record_day(True)
        restored.record_day(True)
        assert restored.to_dict() == counter.to_dict()
        assert restored.strength == 4


# =====================================================================
# HabitCounter domain API
# =====================================================================

class TestHabitCounter:
    def test_basic_increment_decrement_floor(self):
        counter = HabitCounter()
        assert counter.strength == 0
        for _ in range(3):
            counter.record_day(False)   # floored at 0
        assert counter.strength == 0
        for _ in range(5):
            counter.record_day(True)
        assert counter.strength == 5
        counter.record_day(False)
        assert counter.strength == 4

    def test_window_uses_ported_trailing_semantics(self):
        counter = HabitCounter()
        for _ in range(60):
            counter.record_day(True)
        assert counter.strength == 60                    # uncapped net counter
        assert counter.days_followed_in_window == 30     # window hard cap
        assert counter.total_days_followed == 60

    def test_reset_on_rule_rewrite(self):
        counter = HabitCounter()
        for _ in range(10):
            counter.record_day(True)
        counter.reset()  # slow brain rewrote the rule: fresh habit
        assert counter.strength == 0
        assert counter.total_days_followed == 0
        assert counter.days_observed == 0
        assert counter.days_followed_in_window == 0
        assert counter.is_strong(1) is False

    def test_days_to_weaken_validation_and_weak_case(self):
        counter = HabitCounter()
        with pytest.raises(ValueError):
            counter.days_to_weaken(0)
        assert counter.days_to_weaken(1) == 0  # already weak


# =====================================================================
# E6 hysteresis scenario (the make-or-break memory eval)
# =====================================================================

class TestE6Hysteresis:
    def test_strong_habit_resists_revert_longer_than_weak(self):
        """Intended E6 behavior: a rule followed 60 days resists a world
        revert far longer than a rule followed 3 days.

        Scenario: the fast brain followed "avoid the toll bridge" for
        60 days (strong) vs 3 days (weak).  The toll is then removed
        (the world reverts) and every subsequent day pressures the
        agent NOT to follow the rule.  The habit stays behaviorally
        binding while is_strong(threshold) holds; hysteresis = the
        number of not-followed days before it flips.

        With strength = net days-followed (+1/-1, floor 0) and
        threshold t, resistance = strength - t + 1 days:
          strong: 60 - 2 + 1 = 59 days;  weak: 3 - 2 + 1 = 2 days.
        The strong habit resists ~30x longer — this asymmetry is the
        measurable behavioral signature of memory that E6 scores.
        """
        threshold = 2
        strong = HabitCounter()
        weak = HabitCounter()
        for _ in range(60):
            strong.record_day(True)
        for _ in range(3):
            weak.record_day(True)

        assert strong.is_strong(threshold)
        assert weak.is_strong(threshold)
        assert strong.days_to_weaken(threshold) == 59
        assert weak.days_to_weaken(threshold) == 2

        # Simulate the revert day by day; the closed form must match.
        strong_days = 0
        while strong.is_strong(threshold):
            strong.record_day(False)
            strong_days += 1
        weak_days = 0
        while weak.is_strong(threshold):
            weak.record_day(False)
            weak_days += 1

        assert strong_days == 59
        assert weak_days == 2
        assert strong_days > weak_days
        # Window view agrees directionally: 30/30 vs 3/30 days followed
        # before the revert (checked pre-revert in
        # test_window_uses_ported_trailing_semantics).
