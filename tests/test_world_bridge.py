"""world/bridge.py — the card->world bridge (M3 D1/D2).

Field mapping, CRN-keyed value-of-time determinism and the income ladder,
corridor membership over the config OD geometry, the card-band -> world-period
alignment, the peak-direction commute-traveler pin, and the bias-corrected EMA
expectation (n=0 and n=3 hand checks). All synthetic masked cards.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from agents.card_executor import RealizedDay, RealizedTrip
from agents.habit_memory import HabitMemory, SubstrateConfig
from world import bridge, crn
from world.config import cityk_corridor
from world.tolling import PERIOD_INDEX, PERIODS


def _card(pid, home="Z16", work="Z04", has_pass=True, income=3, cars=1, mode="car",
          band="am_peak"):
    return {
        "persona_id": pid,
        "card_version": "m3-test",
        "skeleton": {
            "home_zone": home, "work_zone": work, "has_pass": has_pass,
            "income_class": income, "household_cars": cars, "can_drive": True,
        },
        "patterns": [{"id": "w", "weight": 1, "trips": [
            {"purpose": "work", "mode": mode, "depart_band": band},
            {"purpose": "home", "mode": mode, "depart_band": "pm_peak"},
        ]}],
        "rules": [], "voice": "v", "surprise_log": [], "habit_counters": {},
        "provenance": {"card_source": "llm"},
    }


CFG = cityk_corridor()


# ---------------------------------------------------------------------------
# field mapping
# ---------------------------------------------------------------------------

def test_skeleton_fields_map_directly():
    cards = [
        _card("P1", home="Z16", work="Z04", has_pass=True),
        _card("P2", home="Z28", work="Z10", has_pass=False),
    ]
    pop = bridge.population_from_cards(cards, CFG, "m3_run0")
    # has_pass straight from skeleton
    assert pop.has_pass.tolist() == [True, False]
    # home/work zones resolve to the world zone codes (Z16 -> index 15, ...)
    from world.geometry import ZONE_INDEX
    assert pop.home_zone[0] == ZONE_INDEX["Z16"]
    assert pop.work_zone[0] == ZONE_INDEX["Z04"]
    assert len(pop) == 2
    assert pop.vot.shape == (2,)
    assert (pop.vot > 0).all()


def test_placeholder_zone_is_off_corridor_and_water():
    # Z00 is the non-commuter work-zone placeholder; it has no world ring, so the
    # persona is forced OFF both the corridor and water masks (no defined OD).
    cards = [_card("P1", home="Z28", work="Z00")]
    pop = bridge.population_from_cards(cards, CFG, "m3_run0")
    assert not bool(pop.is_corridor[0])
    assert not bool(pop.is_water[0])


# ---------------------------------------------------------------------------
# corridor membership (config OD geometry)
# ---------------------------------------------------------------------------

def test_corridor_membership_matches_od_geometry():
    cards = [
        _card("north_core", home="Z16", work="Z04"),   # outer_north -> core: corridor
        _card("north_south", home="Z16", work="Z22"),  # outer_north -> outer_south: corridor
        _card("core_core", home="Z01", work="Z04"),    # core -> core: local, not corridor
        _card("east_core", home="Z28", work="Z04"),    # east_water -> core: water, not corridor
    ]
    pop = bridge.population_from_cards(cards, CFG, "m3_run0")
    assert pop.is_corridor.tolist() == [True, True, False, False]
    assert bool(pop.is_water[3])  # the east trip is a water crossing


# ---------------------------------------------------------------------------
# value of time: CRN determinism + income ladder
# ---------------------------------------------------------------------------

def test_vot_is_crn_deterministic_and_namespace_paired():
    cards = [_card("P1"), _card("P2")]
    a = bridge.population_from_cards(cards, CFG, "m3_run0").vot
    b = bridge.population_from_cards(cards, CFG, "m3_run0").vot
    np.testing.assert_array_equal(a, b)  # bit-identical rerun
    c = bridge.population_from_cards(cards, CFG, "m3_run1").vot
    assert not np.allclose(a, c)  # different ensemble namespace -> independent draw


def test_vot_matches_lognormal_times_income_ladder():
    # same persona_id -> same CRN uniform, so vot differs ONLY by the ladder.
    cards = [
        _card("PX", income=3),     # anchor 1.00
        _card("PX", income=5),     # 1.35
        _card("PX", income=1),     # 0.80
        _card("PX", income=None),  # None -> 1.00 (mid anchor)
    ]
    vot = bridge.population_from_cards(cards, CFG, "m3_run0").vot
    # reconstruct the base draw from the CRN doctrine and the config lognormal
    from scipy.special import ndtri
    u = crn.draw("m3_run0:PX:vot")
    z = ndtri(u)
    base = math.exp(math.log(CFG.vot_median) + CFG.vot_sigma * z)
    assert vot[0] == pytest.approx(base * 1.00)
    assert vot[1] == pytest.approx(base * 1.35)
    assert vot[2] == pytest.approx(base * 0.80)
    assert vot[3] == pytest.approx(base * 1.00)  # None anchors at mid
    # monotone in income (higher income -> higher value of time)
    assert vot[2] < vot[0] < vot[1]


def test_income_ladder_constants_documented():
    assert bridge.INCOME_VOT_MULTIPLIER == {1: 0.80, 2: 0.90, 3: 1.00, 4: 1.15, 5: 1.35}
    assert bridge._income_multiplier(None) == 1.0
    assert bridge._income_multiplier(99) == 1.0  # unknown class -> anchor


# ---------------------------------------------------------------------------
# card band -> world period alignment
# ---------------------------------------------------------------------------

def test_period_code_for_band_alignment():
    assert bridge.period_code_for_band("night") == PERIOD_INDEX["overnight"]
    assert bridge.period_code_for_band("am_peak") == PERIOD_INDEX["am_peak"]
    assert bridge.period_code_for_band("midday") == PERIOD_INDEX["offpeak"]
    assert bridge.period_code_for_band("pm_peak") == PERIOD_INDEX["pm_peak"]
    assert bridge.period_code_for_band("evening") == PERIOD_INDEX["offpeak"]
    # a token already in the world vocabulary passes through
    assert bridge.period_code_for_band("offpeak") == PERIOD_INDEX["offpeak"]
    # an unknown token falls back to offpeak
    assert bridge.period_code_for_band("???") == PERIOD_INDEX["offpeak"]


# ---------------------------------------------------------------------------
# corridor_travelers_of_day: the first-car/ride commute pin
# ---------------------------------------------------------------------------

def test_corridor_travelers_pin_first_car_or_ride():
    cards = [
        _card("P1", home="Z16", work="Z04"),  # corridor
        _card("P2", home="Z01", work="Z04"),  # not corridor (core->core)
    ]
    pop = bridge.population_from_cards(cards, CFG, "m3_run0")
    row_index = bridge.persona_row_index(cards)
    realized = {
        # first trip is a walk, second a car; the pin must select the car and
        # take ITS band (evening) for the period.
        "P1": [RealizedDay(3, 1.0, [
            RealizedTrip("shop_daily", "walk", "midday"),
            RealizedTrip("work", "car", "evening"),
        ])],
        # P2 is not a corridor OD -> excluded even though it drives.
        "P2": [RealizedDay(3, 1.0, [RealizedTrip("work", "car", "am_peak")])],
    }
    table = bridge.corridor_travelers_of_day(
        realized, cards, CFG, population=pop, row_index=row_index)
    assert table.persona_ids == ["P1"]
    assert table.mode == ["car"]
    assert table.day_index == [3]
    assert table.period_codes.tolist() == [PERIOD_INDEX["offpeak"]]  # evening -> offpeak
    assert table.vot[0] == pytest.approx(pop.vot[0])
    assert table.access[0] == pytest.approx(pop.access[0])


def test_corridor_travelers_skips_days_with_no_car_or_ride():
    cards = [_card("P1", home="Z16", work="Z04")]
    pop = bridge.population_from_cards(cards, CFG, "m3_run0")
    row_index = bridge.persona_row_index(cards)
    realized = {"P1": [RealizedDay(1, 1.0, [RealizedTrip("work", "transit", "am_peak")])]}
    table = bridge.corridor_travelers_of_day(
        realized, cards, CFG, population=pop, row_index=row_index)
    assert len(table) == 0


# ---------------------------------------------------------------------------
# expected_minutes: bias-corrected EMA (n=0 and n=3 hand checks)
# ---------------------------------------------------------------------------

def test_expected_minutes_freeflow_when_cell_empty():
    mem = HabitMemory()
    # no cell yet -> free-flow config value
    assert bridge.expected_minutes(mem, "car|corridor|am_peak", 12.5) == 12.5


def test_expected_minutes_bias_corrected_ema_hand_check():
    mem = HabitMemory()  # default alpha_base = 0.3
    key = "car|corridor|am_peak"
    x = 20.0
    for _ in range(3):
        mem.observe(key, x, 0.0)  # the expected arg does not affect ema_realized
    cell = mem.cell(key)
    assert cell.n == 3
    # raw zero-seeded EMA after three observations of x=20 at alpha=0.3
    ema1 = 0.3 * x
    ema2 = 0.7 * ema1 + 0.3 * x
    ema3 = 0.7 * ema2 + 0.3 * x
    assert cell.ema_realized_minutes == pytest.approx(ema3)
    # bias correction divides out (1 - (1-alpha)**n) = 1 - 0.7**3 and recovers x
    expected = bridge.expected_minutes(mem, key, freeflow=999.0)
    assert expected == pytest.approx(ema3 / (1.0 - 0.7 ** 3))
    assert expected == pytest.approx(x)  # a constant series de-biases to x exactly


def test_expected_minutes_reads_alpha_from_memory_config():
    # alpha is read from the memory config, not hardcoded 0.7 = 1 - 0.3.
    mem = HabitMemory(config=SubstrateConfig(alpha_base=0.5))
    key = "car|corridor|am_peak"
    x = 10.0
    for _ in range(2):
        mem.observe(key, x, 0.0)
    cell = mem.cell(key)
    ema = 0.5 * x
    ema = 0.5 * ema + 0.5 * x
    got = bridge.expected_minutes(mem, key, freeflow=0.0)
    assert got == pytest.approx(ema / (1.0 - 0.5 ** 2))
    assert got == pytest.approx(x)


def test_period_vocabulary_alignment_is_total():
    # every card band and every world period must resolve to a valid world code
    for band in ("night", "am_peak", "midday", "pm_peak", "evening"):
        assert 0 <= bridge.period_code_for_band(band) < len(PERIODS)
    for period in PERIODS:
        assert bridge.period_code_for_band(period) == PERIOD_INDEX[period]
