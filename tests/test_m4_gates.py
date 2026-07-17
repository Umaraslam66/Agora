"""The two sealed pre-M4 gates (both sealed 2026-07-15):

* household transponder inheritance (`docs/DECISION_M4_HAS_PASS_GATE.md`,
  Option B): household-level CRN pass draw under the fresh ``hh_pass`` site,
  member inheritance, and the car-trips-only charge semantics in the
  corridor traveler table;
* borrowed-car availability (`docs/DECISION_M4_BORROWED_CAR_GATE.md`,
  Option B): the executor's per-day ``caraccess`` draw for the qualifying
  class, its mechanical belt, CRN-stream isolation, and the feasibility-gate
  relaxation for personas whose own diary shows car driving.

All synthetic masked cards.
"""
from __future__ import annotations

import numpy as np

from agents.card_executor import (
    BorrowedCarAccess,
    execute_day,
    execute_days,
    expected_mode_counts,
)
from grounding.card_validation import feasibility
from world import bridge, crn
from world.config import cityk_corridor
from world.household_pass import (
    draw_household_pass,
    hh_pass_key,
    persona_pass_from_households,
)

CFG = cityk_corridor()


def _card(pid, home="Z16", work="Z04", has_pass=True, cars=1, can_drive=True,
          age=35, mode="car"):
    return {
        "persona_id": pid,
        "card_version": "m4-test",
        "skeleton": {
            "home_zone": home, "work_zone": work, "has_pass": has_pass,
            "income_class": 3, "household_cars": cars, "can_drive": can_drive,
            "age": age,
        },
        "patterns": [{"id": "w", "weight": 1, "trips": [
            {"purpose": "work", "mode": mode, "depart_band": "am_peak"},
            {"purpose": "home", "mode": mode, "depart_band": "pm_peak"},
        ]}],
        "rules": [], "voice": "v", "surprise_log": [], "habit_counters": {},
        "provenance": {"card_source": "llm"},
    }


# ---------------------------------------------------------------------------
# household pass: draw + inheritance
# ---------------------------------------------------------------------------

def test_household_draw_is_deterministic_and_rate_bounded():
    rates = {"H1": 1.0, "H2": 0.0, "H3": 0.5}
    a = draw_household_pass(rates)
    b = draw_household_pass(rates)
    assert a == b  # CRN determinism
    assert a["H1"] is True and a["H2"] is False
    # H3 matches the raw CRN uniform against 0.5
    assert a["H3"] == (crn.draw(hh_pass_key("seed", "H3")) < 0.5)


def test_household_draw_site_key_is_the_sealed_fresh_site():
    assert hh_pass_key("seed", "H9") == "seed:H9:hh_pass"


def test_persona_inheritance_and_unknown_household():
    hh_pass = {"H1": True, "H2": False}
    persona_household = {"P1": "H1", "P2": "H1", "P3": "H2", "P4": "H_absent"}
    m = persona_pass_from_households(persona_household, hh_pass)
    assert m == {"P1": True, "P2": True, "P3": False, "P4": False}


def test_population_persona_pass_overrides_skeleton():
    cards = [_card("P1", has_pass=True), _card("P2", has_pass=False)]
    # default: skeleton respected (pre-decision behavior, byte-identical)
    pop = bridge.population_from_cards(cards, CFG, "m3_run0")
    assert pop.has_pass.tolist() == [True, False]
    # override: the household-inheritance map wins; absent persona -> False
    pop2 = bridge.population_from_cards(
        cards, CFG, "m3_run0", persona_pass={"P2": True}
    )
    assert pop2.has_pass.tolist() == [False, True]


def test_traveler_table_pass_is_car_trips_only():
    # both personas hold the pass; the ride traveler must NOT carry it into
    # the charge path (sealed item 2: the fee belongs to the vehicle operator)
    cards = [
        _card("P_car", mode="car", has_pass=True),
        _card("P_ride", mode="ride", has_pass=True),
    ]
    days = {
        "P_car": [_rd(0)], "P_ride": [_rd(0, mode="ride")],
    }
    table = bridge.corridor_travelers_of_day(days, cards, CFG, namespace="m4t")
    by_pid = dict(zip(table.persona_ids, table.has_pass.tolist()))
    assert by_pid == {"P_car": True, "P_ride": False}


def _rd(day, mode="car"):
    from agents.card_executor import RealizedDay, RealizedTrip
    return RealizedDay(day, 1.0, [RealizedTrip("work", mode, "am_peak")])


# ---------------------------------------------------------------------------
# borrowed-car availability: the executor draw
# ---------------------------------------------------------------------------

def _car0_card(pid="P_bc", **kw):
    return _card(pid, cars=0, can_drive=True, age=40, mode="car", **kw)


def test_no_access_policy_coerces_exactly_as_before():
    card = _car0_card()
    trips = execute_day(card, 0, "m4t")
    assert [t.mode for t in trips] == ["ride", "ride"]


def test_rate_one_grants_and_rate_zero_denies():
    card = _car0_card()
    grant_all = BorrowedCarAccess(rate=1.0, qualifying=frozenset({"P_bc"}))
    deny_all = BorrowedCarAccess(rate=0.0, qualifying=frozenset({"P_bc"}))
    assert [t.mode for t in execute_day(card, 0, "m4t", car_access=grant_all)] == ["car", "car"]
    assert [t.mode for t in execute_day(card, 0, "m4t", car_access=deny_all)] == ["ride", "ride"]


def test_non_qualifying_persona_never_granted():
    card = _car0_card()
    access = BorrowedCarAccess(rate=1.0, qualifying=frozenset({"someone_else"}))
    assert [t.mode for t in execute_day(card, 0, "m4t", car_access=access)] == ["ride", "ride"]


def test_mechanical_belt_on_the_sealed_class():
    # qualifying id but NOT a licensed adult in a zero-vehicle household ->
    # the belt denies regardless of the fitted set
    access = BorrowedCarAccess(rate=1.0, qualifying=frozenset({"P_bc"}))
    minor = _card("P_bc", cars=0, can_drive=True, age=16, mode="car")
    unlicensed = _card("P_bc", cars=0, can_drive=False, age=40, mode="car")
    assert [t.mode for t in execute_day(minor, 0, "m4t", car_access=access)] == ["ride", "ride"]
    assert [t.mode for t in execute_day(unlicensed, 0, "m4t", car_access=access)] == ["ride", "ride"]


def test_draw_is_per_day_deterministic_and_uses_sealed_site():
    card = _car0_card()
    rate = 0.5
    access = BorrowedCarAccess(rate=rate, qualifying=frozenset({"P_bc"}))
    for day in range(6):
        granted = crn.draw(f"m4t:P_bc:{day}:caraccess") < rate
        trips = execute_day(card, day, "m4t", car_access=access)
        want = "car" if granted else "ride"
        assert [t.mode for t in trips] == [want, want], f"day {day}"


def test_caraccess_site_does_not_disturb_existing_streams():
    # a policy that never grants leaves execution bit-identical to no policy
    card = _card("P_multi", cars=0, age=40)
    card["patterns"].append({"id": "e", "weight": 2, "trips": [
        {"purpose": "leisure", "mode": "walk", "depart_band": "midday"}]})
    never = BorrowedCarAccess(rate=0.0, qualifying=frozenset({"P_multi"}))
    slots = {"P_multi": [(d, 1.0) for d in range(10)]}
    base = execute_days([dict(card)], slots, "m4t", update_habits=False)
    gated = execute_days([dict(card)], slots, "m4t", update_habits=False, car_access=never)
    assert base == gated


# ---------------------------------------------------------------------------
# feasibility-gate relaxation (generation-time half of the decision)
# ---------------------------------------------------------------------------

_OBS_WITH_CAR = {"mode_counts": {"car": 2, "walk": 5}}
_OBS_NO_CAR = {"mode_counts": {"walk": 5}}


def test_feasibility_relaxed_for_qualifying_class():
    card = _car0_card()
    obj = {"patterns": card["patterns"], "rules": []}
    skeleton = card["skeleton"]
    assert feasibility(obj, skeleton, _OBS_WITH_CAR) == []


def test_feasibility_still_rejects_without_observed_car_driving():
    card = _car0_card()
    obj = {"patterns": card["patterns"], "rules": []}
    skeleton = card["skeleton"]
    assert feasibility(obj, skeleton, _OBS_NO_CAR) != []
    assert feasibility(obj, skeleton, None) != []       # no diary in evidence
    assert feasibility(obj, skeleton) != []             # legacy call shape


def test_feasibility_relaxation_needs_licensed_adult():
    obj = {"patterns": _car0_card()["patterns"], "rules": []}
    minor = dict(_car0_card()["skeleton"], age=16)
    unlicensed = dict(_car0_card()["skeleton"], can_drive=False)
    assert feasibility(obj, minor, _OBS_WITH_CAR) != []
    assert feasibility(obj, unlicensed, _OBS_WITH_CAR) != []


# ---------------------------------------------------------------------------
# expected_mode_counts (the fit's expectation kernel)
# ---------------------------------------------------------------------------

def test_expected_mode_counts_weights_patterns_and_applies_rules():
    card = {
        "persona_id": "P_e",
        "skeleton": {"household_cars": 0, "can_drive": True, "age": 40},
        "patterns": [
            {"id": "a", "weight": 3, "trips": [
                {"purpose": "work", "mode": "car", "depart_band": "am_peak"}]},
            {"id": "b", "weight": 1, "trips": [
                {"purpose": "leisure", "mode": "walk", "depart_band": "midday"}]},
        ],
        "rules": [{"id": "r1", "when": {"purpose": "leisure"}, "then": {"mode": "transit"}}],
    }
    on = expected_mode_counts(card, car_ok=True)
    off = expected_mode_counts(card, car_ok=False)
    assert np.isclose(on["car"], 0.75)
    assert np.isclose(on["transit"], 0.25)      # rule fired on the leisure trip
    assert "walk" not in on
    assert "car" not in off and np.isclose(off["ride"], 0.75)  # coercion
    # total trips are coercion-invariant
    assert np.isclose(sum(on.values()), sum(off.values()))
