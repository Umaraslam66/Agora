"""grounding/card_validation.py — schema validator, lints, fallback, assembly.

Synthetic masked fixtures only. Covers: a known-good card accepted; each
rejection class (bad enum, extra key, >6 patterns, bad weight/id/voice,
missing required, day reference, HH:MM, infeasible car); replay-smell
semantics; fallback determinism + full self-validity.
"""
from __future__ import annotations

import json

import pandas as pd

from grounding import card_validation as cv
from grounding.masking.mask_lint import default_token_path, load_forbidden_tokens

FORBIDDEN = load_forbidden_tokens(default_token_path())
PLANTED_TOKEN = "gamla stan"
assert PLANTED_TOKEN in FORBIDDEN

SKELETON = {
    "home_zone": "Z04", "work_zone": "Z11", "age": 39, "employed": True,
    "student": False, "can_drive": True, "household_size": 3,
    "household_cars": 1, "income_class": 4, "has_pass": True,
}


def good_card() -> dict:
    return {
        "patterns": [
            {
                "id": "workday_commute",
                "weight": 7,
                "trips": [
                    {"purpose": "work", "mode": "car", "depart_band": "am_peak"},
                    {"purpose": "home", "mode": "car", "depart_band": "pm_peak"},
                ],
            },
            {"id": "quiet_day", "weight": 2, "trips": []},
        ],
        "rules": [
            {
                "id": "r1",
                "when": {"purpose": "shop_daily"},
                "then": {"mode": "walk", "depart_band": "midday"},
            }
        ],
        "voice": "I mostly drive to work and keep errands close to home.",
    }


# ---------------------------------------------------------------------------
# validate_card_json
# ---------------------------------------------------------------------------

def test_known_good_card_accepted():
    assert cv.validate_card_json(good_card()) == []


def test_assembled_card_accepted_harness_fields_stripped():
    card = cv.assemble_card("P00001", SKELETON, good_card(), {"card_source": "llm"})
    assert cv.validate_card_json(card) == []


def test_non_object_rejected():
    assert cv.validate_card_json([1, 2]) != []


def test_bad_mode_enum_rejected():
    card = good_card()
    card["patterns"][0]["trips"][0]["mode"] = "jetpack"
    errs = cv.validate_card_json(card)
    assert errs and any("mode" in e and "jetpack" in e for e in errs)


def test_bad_purpose_enum_rejected():
    card = good_card()
    card["patterns"][0]["trips"][0]["purpose"] = "commuting"
    assert cv.validate_card_json(card)


def test_bad_band_enum_rejected():
    card = good_card()
    card["rules"][0]["then"]["depart_band"] = "noon"
    assert cv.validate_card_json(card)


def test_extra_key_rejected_at_every_level():
    card = good_card()
    card["notes"] = "extra"
    errs = cv.validate_card_json(card)
    assert any("notes" in e and "not allowed" in e for e in errs)

    card2 = good_card()
    card2["patterns"][0]["trips"][0]["distance_km"] = 4.2
    errs2 = cv.validate_card_json(card2)
    assert any("distance_km" in e for e in errs2)

    card3 = good_card()
    card3["rules"][0]["when"]["mode"] = "car"  # `when` cannot condition on mode
    assert cv.validate_card_json(card3)


def test_more_than_six_patterns_rejected_by_schema_and_replay():
    card = good_card()
    card["patterns"] = [
        {"id": f"p{i}", "weight": 1, "trips": []} for i in range(7)
    ]
    errs = cv.validate_card_json(card)
    assert any("maxItems" in e for e in errs)
    smells = cv.replay_smell(card, [])
    assert any("exceed the cap" in s for s in smells)  # double guard


def test_weight_bounds_and_type_rejected():
    for bad in (0, 11, 7.5, True):
        card = good_card()
        card["patterns"][0]["weight"] = bad
        assert cv.validate_card_json(card), f"weight {bad!r} must be rejected"


def test_bad_id_pattern_rejected():
    for bad in ("Bad-ID", "9starts_with_digit", "x" * 30, ""):
        card = good_card()
        card["patterns"][0]["id"] = bad
        assert cv.validate_card_json(card), f"id {bad!r} must be rejected"


def test_voice_too_long_rejected():
    card = good_card()
    card["voice"] = "x" * 201
    errs = cv.validate_card_json(card)
    assert any("maxLength" in e for e in errs)


def test_missing_required_rejected():
    card = good_card()
    del card["voice"]
    assert any("voice" in e for e in cv.validate_card_json(card))

    card2 = good_card()
    del card2["patterns"][0]["weight"]
    assert any("weight" in e for e in cv.validate_card_json(card2))

    card3 = good_card()
    del card3["rules"][0]["then"]
    assert any("then" in e for e in cv.validate_card_json(card3))


def test_empty_rule_when_rejected():
    card = good_card()
    card["rules"][0]["when"] = {}
    errs = cv.validate_card_json(card)
    assert any("minProperties" in e for e in errs)


def test_too_many_trips_rejected():
    card = good_card()
    card["patterns"][0]["trips"] = [
        {"purpose": "other", "mode": "walk", "depart_band": "midday"}
    ] * 9
    assert any("maxItems" in e for e in cv.validate_card_json(card))


# ---------------------------------------------------------------------------
# lint_card_text
# ---------------------------------------------------------------------------

def test_lint_clean_card_passes():
    assert cv.lint_card_text(good_card()) == []


def test_lint_catches_forbidden_token_in_voice():
    card = good_card()
    card["voice"] = f"I like walking around {PLANTED_TOKEN} in the evening."
    hits = cv.lint_card_text(card)
    assert hits and PLANTED_TOKEN in hits[0]


def test_lint_scans_every_string_including_nested_ids():
    card = cv.assemble_card("P00001", SKELETON, good_card(), {"card_source": "llm"})
    card["provenance"]["note"] = PLANTED_TOKEN
    assert cv.lint_card_text(card)


# ---------------------------------------------------------------------------
# replay_smell
# ---------------------------------------------------------------------------

def test_replay_clean_card_passes():
    observed = [
        (("work", "car", "am_peak"), ("home", "car", "pm_peak")),
        (("work", "car", "am_peak"), ("home", "car", "pm_peak")),
        (),
    ]
    assert cv.replay_smell(good_card(), observed) == []


def test_replay_flags_day_reference_in_voice():
    card = good_card()
    card["voice"] = "On day 3 I drove to work."
    assert any("day-index" in s for s in cv.replay_smell(card, []))


def test_replay_flags_day_reference_in_id():
    card = good_card()
    card["patterns"][0]["id"] = "day3_commute"
    assert any("day-index" in s for s in cv.replay_smell(card, []))


def test_replay_flags_clock_time_in_voice_and_id():
    card = good_card()
    card["voice"] = "I leave home at 08:15 sharp."
    assert any("clock-time" in s for s in cv.replay_smell(card, []))

    card2 = good_card()
    card2["patterns"][0]["id"] = "leave_08:15"  # schema also rejects ':' in ids
    assert any("clock-time" in s for s in cv.replay_smell(card2, []))


def test_replay_flags_date_like_token():
    card = good_card()
    card["voice"] = "Since 2031-04-02 I bike more."
    assert any("date-like" in s for s in cv.replay_smell(card, []))


def test_replay_flags_multi_day_exact_enumeration():
    seq_a = (("work", "car", "am_peak"),)
    seq_b = (("shop_daily", "walk", "midday"),)
    seq_c = (("leisure", "transit", "evening"),)
    card = {
        "patterns": [
            {"id": "a", "weight": 1,
             "trips": [{"purpose": p, "mode": m, "depart_band": b} for p, m, b in seq]}
            for seq in (seq_a, seq_b, seq_c)
        ],
        "rules": [],
        "voice": "I go out once most days.",
    }
    for i, p in enumerate(card["patterns"]):
        p["id"] = f"p{i}"
    smells = cv.replay_smell(card, [seq_a, seq_b, seq_c])
    assert any("enumeration" in s for s in smells)


def test_replay_compressed_card_not_flagged():
    seq_a = (("work", "car", "am_peak"),)
    seq_b = (("shop_daily", "walk", "midday"),)
    card = {
        "patterns": [
            {"id": "p0", "weight": 2,
             "trips": [{"purpose": "work", "mode": "car", "depart_band": "am_peak"}]},
        ],
        "rules": [],
        "voice": "I keep my days simple.",
    }
    # two active patterns would enumerate; one compressed pattern does not
    assert cv.replay_smell(card, [seq_a, seq_b]) == []


def test_replay_ignores_quiet_days_in_enumeration():
    # a quiet day + one active day: the two-pattern card (mandated no-trip
    # pattern + the active pattern) is the unique faithful compression and
    # must NOT be flagged
    active = (("work", "car", "am_peak"), ("home", "car", "pm_peak"))
    card = {
        "patterns": [
            {"id": "quiet_day", "weight": 5, "trips": []},
            {"id": "workday", "weight": 5,
             "trips": [{"purpose": p, "mode": m, "depart_band": b} for p, m, b in active]},
        ],
        "rules": [],
        "voice": "Some days I stay in, some days I commute.",
    }
    assert cv.replay_smell(card, [(), active]) == []


def test_replay_single_day_never_enumeration():
    active = (("work", "car", "am_peak"),)
    card = {
        "patterns": [{"id": "workday", "weight": 1,
                      "trips": [{"purpose": "work", "mode": "car", "depart_band": "am_peak"}]}],
        "rules": [],
        "voice": "I commute.",
    }
    assert cv.replay_smell(card, [active]) == []


# ---------------------------------------------------------------------------
# feasibility
# ---------------------------------------------------------------------------

def test_feasibility_passes_with_car_available():
    assert cv.feasibility(good_card(), SKELETON) == []


def test_feasibility_rejects_car_trip_in_carless_household():
    skeleton = dict(SKELETON, household_cars=0)
    errs = cv.feasibility(good_card(), skeleton)
    assert len(errs) == 2  # both car trips in the commute pattern
    assert all("car trip" in e for e in errs)


def test_feasibility_rejects_car_trip_when_cannot_drive():
    skeleton = dict(SKELETON, can_drive=False)
    assert cv.feasibility(good_card(), skeleton)


def test_feasibility_rejects_rule_forcing_car():
    card = good_card()
    card["rules"][0]["then"] = {"mode": "car"}
    skeleton = dict(SKELETON, household_cars=0)
    errs = cv.feasibility(card, skeleton)
    assert any("rule sets mode car" in e for e in errs)


# ---------------------------------------------------------------------------
# assemble_card
# ---------------------------------------------------------------------------

def test_assemble_card_shape_and_seeded_counters():
    provenance = {"card_source": "llm", "model": "Qwen3-8B", "attempt": 1}
    card = cv.assemble_card("P00001", SKELETON, good_card(), provenance)
    assert card["card_version"] == cv.CARD_VERSION
    assert card["persona_id"] == "P00001"
    assert card["skeleton"] == SKELETON
    assert card["surprise_log"] == []
    assert card["provenance"] == provenance
    # one empty HabitCounter dict per pattern id AND per rule id
    assert set(card["habit_counters"]) == {"workday_commute", "quiet_day", "r1"}
    for counter in card["habit_counters"].values():
        assert counter["strength"] == 0
        assert counter["days_observed"] == 0
        assert counter["window"] == []


# ---------------------------------------------------------------------------
# fallback_card
# ---------------------------------------------------------------------------

def _frames(days, trip_rows):
    pdays = pd.DataFrame([{"daynum": d, "n_collapsed": 0} for d in days])
    trips = pd.DataFrame(
        trip_rows, columns=["daynum", "tripnum", "purpose", "mode", "band"]
    )
    return pdays, trips


def _all_gates(card, observed, skeleton):
    return (
        cv.validate_card_json(card)
        + cv.lint_card_text(card)
        + cv.replay_smell(card, observed)
        + cv.feasibility(card, skeleton)
    )


def test_fallback_is_deterministic():
    pdays, trips = _frames(
        [1, 2], [(1, 1, "work", "car", "am_peak"), (1, 2, "home", "car", "pm_peak")]
    )
    a = cv.fallback_card("P00001", SKELETON, pdays, trips)
    b = cv.fallback_card("P00001", SKELETON, pdays, trips)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["provenance"]["card_source"] == "fallback"
    assert a["rules"] == []


def test_fallback_passes_every_gate_across_case_zoo():
    cases = {
        "repeated signature": (
            [1, 2, 3],
            [(1, 1, "work", "car", "am_peak"), (2, 1, "work", "car", "am_peak"),
             (3, 1, "shop_daily", "walk", "midday")],
        ),
        "all-distinct multi-day": (
            [1, 2, 3],
            [(1, 1, "work", "car", "am_peak"), (2, 1, "shop_daily", "walk", "midday"),
             (3, 1, "leisure", "transit", "evening")],
        ),
        "quiet plus one active": (
            [1, 2],
            [(1, 1, "work", "car", "am_peak"), (1, 2, "home", "car", "pm_peak")],
        ),
        "all quiet": ([1, 2], []),
        "no observed days": ([], []),
        "two quiet two distinct active": (
            [1, 2, 3, 4],
            [(3, 1, "work", "car", "am_peak"), (4, 1, "shop_daily", "walk", "midday")],
        ),
    }
    for name, (days, trip_rows) in cases.items():
        pdays, trips = _frames(days, trip_rows)
        card = cv.fallback_card("P00001", SKELETON, pdays, trips)
        observed = cv.day_signatures(pdays, trips)
        assert _all_gates(card, observed, SKELETON) == [], f"case {name!r} failed a gate"


def test_fallback_caps_six_patterns_for_many_distinct_days():
    purposes = ["work", "shop_daily", "leisure", "personal_business",
                "education", "shop_other", "pickup_dropoff", "other"]
    days = list(range(1, 9))
    trip_rows = [(d, 1, p, "walk", "midday") for d, p in zip(days, purposes)]
    pdays, trips = _frames(days, trip_rows)
    card = cv.fallback_card("P00001", SKELETON, pdays, trips)
    assert 1 <= len(card["patterns"]) <= 6
    observed = cv.day_signatures(pdays, trips)
    assert _all_gates(card, observed, SKELETON) == []


def test_fallback_keeps_quiet_pattern_and_proportional_weights():
    # 5 identical commute days + 1 quiet day
    commute = [(d, 1, "work", "transit", "am_peak") for d in range(1, 6)]
    pdays, trips = _frames([1, 2, 3, 4, 5, 6], commute)
    card = cv.fallback_card("P00001", SKELETON, pdays, trips)
    by_id = {p["id"]: p for p in card["patterns"]}
    assert "quiet_day" in by_id and by_id["quiet_day"]["trips"] == []
    active = [p for p in card["patterns"] if p["trips"]]
    assert len(active) == 1
    assert active[0]["weight"] == 10  # most-frequent signature pinned to 10
    assert by_id["quiet_day"]["weight"] == 2  # round(1/5 * 10)
    for p in card["patterns"]:
        assert 1 <= p["weight"] <= 10


def test_fallback_coerces_infeasible_car_trips():
    skeleton = dict(SKELETON, household_cars=0)
    pdays, trips = _frames(
        [1, 2], [(1, 1, "work", "car", "am_peak"), (2, 1, "work", "car", "am_peak")]
    )
    card = cv.fallback_card("P00001", skeleton, pdays, trips)
    modes = [t["mode"] for p in card["patterns"] for t in p["trips"]]
    assert modes and all(m == "ride" for m in modes)
    assert cv.feasibility(card, skeleton) == []
