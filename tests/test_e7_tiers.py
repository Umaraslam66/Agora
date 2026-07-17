"""A4.1 E7 tier evidence builder: nesting, CRN day selection, fidelity
applicability, and tier-blind fallbacks. Synthetic masked frames."""
from __future__ import annotations

import pandas as pd

from grounding import e7_tiers
from world import crn

SKELETON = {
    "home_zone": "Z16", "work_zone": "Z04", "age": 40, "employed": True,
    "student": False, "can_drive": True, "household_size": 2,
    "household_cars": 1, "income_class": 3, "has_pass": True,
}

PERSON_ROW = {
    "stated_typical_mode": "transit",
    "commute_freq": "3 days a week",
    "transit_freq": "5 days a week",
    "telecommute_freq": "2 days a week",
    "res_factors_transit": "Very important",
}

DAYS = pd.DataFrame([
    {"daynum": 1, "n_collapsed": 2, "day_weight": 1.0},
    {"daynum": 2, "n_collapsed": 1, "day_weight": 3.0},
])
TRIPS = pd.DataFrame([
    {"daynum": 1, "tripnum": 1, "purpose": "work", "mode": "car", "band": "am_peak"},
    {"daynum": 1, "tripnum": 2, "purpose": "home", "mode": "car", "band": "pm_peak"},
    {"daynum": 2, "tripnum": 1, "purpose": "leisure", "mode": "walk", "band": "midday"},
])


def _lines(tier, selected=None):
    lines, vd, vt, n = e7_tiers.tier_evidence(
        tier, SKELETON, DAYS, TRIPS, PERSON_ROW, selected
    )
    return lines, vd, vt, n


def test_tiers_are_nested():
    t1, *_ = _lines("T1")
    t2, *_ = _lines("T2")
    t3, *_ = _lines("T3")
    t4, *_ = _lines("T4", selected=1)
    t5, *_ = _lines("T5")
    assert t1 == []
    assert set(t1) <= set(t2) < set(t3) < set(t4)
    # T5 shows the full trace: contains every T3 (world+stated) line
    assert set(t3) < set(t5)


def test_t2_is_placement_without_behavior_and_t3_adds_claims():
    t2, _vd, _vt, n2 = _lines("T2")
    assert n2 == 0
    assert any("commute corridor" in ln for ln in t2)
    assert not any("Self-reports" in ln for ln in t2)
    t3, *_ = _lines("T3")
    assert any("usually commuting by shared transit" in ln for ln in t3)
    assert any("working from home about two days a week" in ln for ln in t3)
    assert any("rated access to shared transit as very important" in ln for ln in t3)


def test_t4_shows_only_the_selected_day_and_t4_noclaims_withholds_claims():
    t4, vd, vt, n = _lines("T4", selected=1)
    assert n == 1 and len(vd) == 1 and len(vt) == 2
    assert any("Contributed 1 recorded weekday day." in ln for ln in t4)
    nc, *_ = _lines("T4_noclaims", selected=1)
    assert not any("Self-reports" in ln or "choosing where to live" in ln for ln in nc)
    assert any("Contributed 1 recorded weekday day." in ln for ln in nc)
    # T4-nofidelity shows the SAME evidence as T4 (the gate, not the bundle,
    # differs)
    nf, *_ = _lines("T4_nofidelity", selected=1)
    assert nf == t4


def test_day_selection_is_crn_deterministic_and_weighted():
    d = e7_tiers.t4_day_of("P00042", [1, 2], [1.0, 3.0])
    assert d == e7_tiers.t4_day_of("P00042", [1, 2], [1.0, 3.0])
    key = e7_tiers.T4_DAY_SITE.format(persona_id="P00042")
    assert d == crn.pick_weighted(key, [1, 2], [1.0, 3.0])


def test_fidelity_reference_follows_tier_visibility():
    _t4, vd, vt, _n = _lines("T4", selected=1)
    obs = e7_tiers.tier_fidelity_observed("T4", vd, vt)
    assert obs["n_observed_weekdays"] == 1 and obs["mode_counts"] == {"car": 2}
    # fidelity-exempt tiers get an empty reference (gate no-ops)
    assert e7_tiers.tier_fidelity_observed("T3", vd, vt) == {}
    assert e7_tiers.tier_fidelity_observed("T4_nofidelity", vd, vt) == {}


def test_tier_fallbacks_are_tier_blind():
    # T1: skeleton prior, availability-respecting
    c1 = e7_tiers.tier_fallback_card("T1", "P1", SKELETON, DAYS.iloc[0:0],
                                     TRIPS.iloc[0:0], {})
    assert c1["provenance"]["card_source"] == "fallback"
    assert c1["patterns"][0]["trips"][0]["mode"] == "car"  # has car + licence
    # T3: stated typical mode + stated days -> commute weight 3, quiet 2
    c3 = e7_tiers.tier_fallback_card("T3", "P1", SKELETON, DAYS.iloc[0:0],
                                     TRIPS.iloc[0:0], PERSON_ROW)
    w = {p["id"]: p["weight"] for p in c3["patterns"]}
    assert w == {"typical_day": 3, "home_day": 2}
    assert c3["patterns"][0]["trips"][0]["mode"] == "transit"
    # T4 arm: empirical fallback on the ONE visible day only
    vis_d = DAYS[DAYS.daynum == 1]
    vis_t = TRIPS[TRIPS.daynum == 1]
    c4 = e7_tiers.tier_fallback_card("T4", "P1", SKELETON, vis_d, vis_t, PERSON_ROW)
    modes = {t["mode"] for p in c4["patterns"] for t in p["trips"]}
    assert modes == {"car"}  # day-2 walk trip is invisible to T4
