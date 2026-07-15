"""agents/card_executor.py — the fast brain: determinism, CRN pairing, rules,
availability coercion, weight fidelity, habit counters.

Synthetic masked cards only. The weight-fidelity test is a chi-square-ish
sanity check over 2000 simulated days for one card.
"""
from __future__ import annotations

import copy

from agents.card_executor import RealizedDay, RealizedTrip, execute_day, execute_days
from world.crn import pick_weighted

SKELETON = {
    "home_zone": "Z04", "work_zone": "Z11", "age": 39, "employed": True,
    "student": False, "can_drive": True, "household_size": 3,
    "household_cars": 1, "income_class": 4, "has_pass": True,
}


def make_card(persona_id="P00001", skeleton=None, rules=None) -> dict:
    """Three patterns with distinct purposes so the drawn pattern is
    identifiable from the realized trips."""
    return {
        "card_version": "m2-1.0",
        "persona_id": persona_id,
        "skeleton": dict(SKELETON if skeleton is None else skeleton),
        "patterns": [
            {"id": "workday", "weight": 7, "trips": [
                {"purpose": "work", "mode": "car", "depart_band": "am_peak"},
                {"purpose": "home", "mode": "car", "depart_band": "pm_peak"},
            ]},
            {"id": "errand_day", "weight": 2, "trips": [
                {"purpose": "shop_daily", "mode": "walk", "depart_band": "midday"},
            ]},
            {"id": "quiet_day", "weight": 1, "trips": []},
        ],
        "rules": list(rules or []),
        "voice": "I mostly commute and run the odd errand.",
        "surprise_log": [],
        "habit_counters": {},
        "provenance": {"card_source": "llm"},
    }


def _drawn_pattern_id(trips) -> str:
    if not trips:
        return "quiet_day"
    return "workday" if trips[0].purpose == "work" else "errand_day"


# ---------------------------------------------------------------------------
# determinism + CRN key contract
# ---------------------------------------------------------------------------

def test_execute_day_is_deterministic():
    card = make_card()
    a = execute_day(card, 17, "run0")
    b = execute_day(card, 17, "run0")
    assert a == b  # dataclass equality, field for field


def test_pattern_draw_uses_documented_crn_key():
    card = make_card()
    weights = [p["weight"] for p in card["patterns"]]
    for day in range(40):
        expected = pick_weighted(
            f"run0:{card['persona_id']}:{day}:pattern", card["patterns"], weights
        )["id"]
        assert _drawn_pattern_id(execute_day(card, day, "run0")) == expected


def test_crn_pairing_same_keys_two_configs_same_pattern_draw():
    # two arm variants sharing persona_id + patterns/weights but differing in
    # everything else draw the SAME pattern every day — the paired-arms
    # property behind E1/E5
    card_a = make_card(rules=[])
    card_b = make_card(
        skeleton=dict(SKELETON, income_class=1, has_pass=False),
        rules=[{"id": "never", "when": {"purpose": "education"}, "then": {"mode": "walk"}}],
    )
    card_b["voice"] = "A different second-arm voice."
    for day in range(60):
        pa = _drawn_pattern_id(execute_day(card_a, day, "run3"))
        pb = _drawn_pattern_id(execute_day(card_b, day, "run3"))
        assert pa == pb


def test_namespaces_give_independent_streams():
    card = make_card()
    seq0 = [_drawn_pattern_id(execute_day(card, d, "run0")) for d in range(60)]
    seq1 = [_drawn_pattern_id(execute_day(card, d, "run1")) for d in range(60)]
    assert seq0 != seq1  # different ensemble members decouple


def test_execute_days_bitwise_deterministic_including_counters():
    slots = {"P00001": [(d, 1.0) for d in range(15)]}
    card1, card2 = make_card(), make_card()
    out1 = execute_days([card1], slots, "run0")
    out2 = execute_days([card2], slots, "run0")
    assert out1 == out2
    assert card1["habit_counters"] == card2["habit_counters"]


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

def test_rule_first_match_wins_and_overrides():
    rules = [
        {"id": "r_first", "when": {"purpose": "work"},
         "then": {"mode": "transit", "depart_band": "midday"}},
        {"id": "r_second", "when": {"purpose": "work"}, "then": {"mode": "bike"}},
    ]
    card = make_card(rules=rules)
    # find a day where the workday pattern is drawn
    for day in range(50):
        trips = execute_day(card, day, "run0")
        if trips and any(t.purpose == "work" for t in trips):
            work = [t for t in trips if t.purpose == "work"][0]
            assert work.rule_applied == "r_first"  # ordered, first match
            assert work.mode == "transit"
            assert work.depart_band == "midday"
            home = [t for t in trips if t.purpose == "home"][0]
            assert home.rule_applied is None  # untouched pattern trip
            assert home.mode == "car"
            break
    else:
        raise AssertionError("workday pattern never drawn in 50 days")


def test_rule_when_can_condition_on_band():
    rules = [
        {"id": "r_band", "when": {"purpose": "work", "depart_band": "pm_peak"},
         "then": {"mode": "walk"}},
    ]
    card = make_card(rules=rules)
    for day in range(50):
        trips = execute_day(card, day, "run0")
        if trips and trips[0].purpose == "work":
            # work trip departs am_peak: the pm_peak condition must NOT match
            assert trips[0].rule_applied is None
            assert trips[0].mode == "car"
            break


# ---------------------------------------------------------------------------
# availability coercion
# ---------------------------------------------------------------------------

def test_car_coerced_to_ride_when_no_household_car():
    card = make_card(skeleton=dict(SKELETON, household_cars=0))
    log = []
    for day in range(50):
        trips = execute_day(card, day, "run0", coercion_log=log)
        assert all(t.mode != "car" for t in trips)
        if trips and trips[0].purpose == "work":
            assert [t.mode for t in trips] == ["ride", "ride"]
    assert log  # coercions were counted
    entry = log[0]
    assert set(entry) == {"persona_id", "day_index", "trip_index"}
    assert entry["persona_id"] == "P00001"


def test_car_coerced_when_cannot_drive():
    card = make_card(skeleton=dict(SKELETON, can_drive=False))
    log = []
    trips_all = [execute_day(card, d, "run0", coercion_log=log) for d in range(50)]
    assert all(t.mode != "car" for trips in trips_all for t in trips)
    assert log


def test_no_coercion_when_car_available():
    card = make_card()
    log = []
    for day in range(50):
        execute_day(card, day, "run0", coercion_log=log)
    assert log == []


def test_rule_forced_car_also_coerced():
    rules = [{"id": "r_car", "when": {"purpose": "shop_daily"}, "then": {"mode": "car"}}]
    card = make_card(skeleton=dict(SKELETON, household_cars=0), rules=rules)
    for day in range(80):
        trips = execute_day(card, day, "run0")
        for t in trips:
            if t.purpose == "shop_daily":
                assert t.rule_applied == "r_car"
                assert t.mode == "ride"  # the belt catches rule output too


# ---------------------------------------------------------------------------
# weights honored (chi-square-ish sanity, 2000 simulated days)
# ---------------------------------------------------------------------------

def test_pattern_frequencies_track_weights():
    card = make_card()  # weights 7 / 2 / 1
    n = 2000
    counts = {"workday": 0, "errand_day": 0, "quiet_day": 0}
    for day in range(n):
        counts[_drawn_pattern_id(execute_day(card, day, "run0"))] += 1
    for pid, expected in (("workday", 0.7), ("errand_day", 0.2), ("quiet_day", 0.1)):
        observed = counts[pid] / n
        # ~4 standard deviations at n=2000 for the loosest cell
        assert abs(observed - expected) < 0.05, (pid, observed)


# ---------------------------------------------------------------------------
# batch API + habit counters
# ---------------------------------------------------------------------------

def test_execute_days_shapes_and_inherits_slot_weights():
    card = make_card()
    slots = {"P00001": [(0, 1.5), (1, 0.5)]}
    out = execute_days([card], slots, "run0")
    assert set(out) == {"P00001"}
    days = out["P00001"]
    assert [d.day_index for d in days] == [0, 1]
    assert [d.day_weight for d in days] == [1.5, 0.5]
    assert all(isinstance(d, RealizedDay) for d in days)
    assert all(isinstance(t, RealizedTrip) for d in days for t in d.trips)
    # persona missing from day_slots -> zero simulated days, not an error
    other = make_card(persona_id="P00002")
    out2 = execute_days([other], slots, "run0")
    assert out2["P00002"] == []


def test_habit_counters_accumulate_per_lived_day():
    rules = [{"id": "r_work", "when": {"purpose": "work"}, "then": {"mode": "transit"}}]
    card = make_card(rules=rules)
    n_days = 30
    slots = {"P00001": [(d, 1.0) for d in range(n_days)]}
    out = execute_days([card], slots, "run0")

    counters = card["habit_counters"]
    assert set(counters) == {"workday", "errand_day", "quiet_day", "r_work"}
    # every id observes every lived day
    for cd in counters.values():
        assert cd["days_observed"] == n_days
    # exactly one pattern followed per day
    followed = sum(counters[p]["total_days_followed"]
                   for p in ("workday", "errand_day", "quiet_day"))
    assert followed == n_days
    # the rule follows exactly on workday-pattern days (it fires on work trips)
    n_workdays = sum(
        1 for d in out["P00001"] if d.trips and d.trips[0].purpose == "work"
    )
    assert counters["r_work"]["total_days_followed"] == n_workdays
    # serialized as plain dicts (JSON-clean, embeddable in the card)
    assert all(isinstance(cd, dict) for cd in counters.values())


def test_habit_counters_accumulate_across_calls():
    card = make_card()
    slots = {"P00001": [(d, 1.0) for d in range(10)]}
    execute_days([card], slots, "run0")
    execute_days([card], {"P00001": [(d, 1.0) for d in range(10, 20)]}, "run0")
    assert card["habit_counters"]["workday"]["days_observed"] == 20


def test_update_habits_false_leaves_card_untouched():
    card = make_card()
    before = copy.deepcopy(card)
    execute_days([card], {"P00001": [(d, 1.0) for d in range(5)]}, "run0",
                 update_habits=False)
    assert card == before
