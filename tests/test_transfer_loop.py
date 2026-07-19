"""A8.1 transfer-arena loop machinery: multi-onset transition scheduling,
the cordon trigger population, and the cordon recording seam (A8.4).

All synthetic masked cards; no slow brain (client=None collects requests —
the offline-cluster mode), so nothing here rewrites, charges, or scores.
"""
from __future__ import annotations

from agents.baseline_loop import AnnouncedOnset, run_baseline_loop
from agents.slow_brain import StandardSurprisePolicy
from world import bridge
from world.config import cityk_cordon, cityk_corridor
from world.tolling import placebo_announcement

CFG = cityk_cordon()


def _card(pid, home, work, mode="car"):
    return {
        "persona_id": pid, "card_version": "bt2-test",
        "skeleton": {"home_zone": home, "work_zone": work, "has_pass": False,
                     "income_class": 3, "household_cars": 1, "can_drive": True,
                     "age": 40},
        "patterns": [{"id": "commute", "weight": 1, "trips": [
            {"purpose": "work", "mode": mode, "depart_band": "am_peak"},
            {"purpose": "home", "mode": mode, "depart_band": "pm_peak"}]}],
        "rules": [], "voice": "v", "surprise_log": [], "habit_counters": {},
        "provenance": {"card_source": "llm"},
    }


# Z04 = core, Z08 = inner (both cordon rings); Z16 = outer_north; Z28 = east.
CROSSER = _card("P1", "Z16", "Z04")            # outside -> core: crosses
INSIDER = _card("P2", "Z04", "Z08")            # core -> inner: never crosses
WALKER = _card("P3", "Z28", "Z00", mode="walk")  # no valid work OD
OUTSIDE = _card("P4", "Z16", "Z21")            # outer -> outer: never crosses


def test_cordon_crossing_rows_semantics():
    rows = bridge.cordon_crossing_rows([CROSSER, INSIDER, WALKER, OUTSIDE], CFG)
    assert rows.tolist() == [True, False, False, False]


def test_cordon_crossing_rows_empty_for_corridor_world():
    rows = bridge.cordon_crossing_rows([CROSSER], cityk_corridor())
    assert rows.tolist() == [False]


def _run(config, cards, extra_days=(3, 6, 9), n_days=12):
    onsets = [AnnouncedOnset(day=d, announcement=placebo_announcement(),
                             tail_surprises=True) for d in extra_days]
    return run_baseline_loop(
        cards, config, {}, namespace="t", n_days=n_days, warmup_days=1,
        policy=StandardSurprisePolicy(), client=None, keep_full_window=False,
        onset=onsets[0], extra_onsets=onsets[1:],
    )


def test_multi_onset_fires_per_transition_for_crossing_personas_only():
    res = _run(CFG, [CROSSER, INSIDER, WALKER, OUTSIDE])
    onset_reqs = [r for r in res.pending_rewrites
                  if r.get("reason") or r.get("announcement")]
    days = sorted({int(r["day_index"]) for r in onset_reqs})
    assert days == [3, 6, 9]
    assert {r["persona_id"] for r in onset_reqs} == {"P1"}
    assert len(onset_reqs) == 3  # once per crossing persona per transition


def test_cordon_tally_recorded_and_reconstructs_exactly():
    res = _run(CFG, [CROSSER, INSIDER, WALKER, OUTSIDE])
    assert sorted(res.cordon_daily) == list(range(12))
    for d, rec in res.cordon_daily.items():
        # three car commuters travel every day; one of them crosses
        assert rec["n_car"] == 3
        assert rec["n_crossing"] == 1
    # per-agent reconstruction (the A8.4 self-check contract)
    n_car = {d: 0 for d in res.cordon_daily}
    n_cross = {d: 0 for d in res.cordon_daily}
    for pid, days in res.cordon_car_days.items():
        for d in days:
            n_car[d] += 1
    for pid, days in res.cordon_crossing_days.items():
        for d in days:
            n_cross[d] += 1
    for d, rec in res.cordon_daily.items():
        assert n_car[d] == rec["n_car"]
        assert n_cross[d] == rec["n_crossing"]
    assert set(res.cordon_crossing_days) == {"P1"}


def test_corridor_world_records_no_cordon_tally():
    res = _run(cityk_corridor(), [CROSSER, INSIDER])
    assert res.cordon_daily == {}
    assert res.cordon_car_days == {}


def test_state_roundtrip_carries_cordon_records():
    res = _run(CFG, [CROSSER, INSIDER])
    st = res.state.__class__.from_dict(res.state.to_dict())
    assert st.cordon_daily == res.state.cordon_daily
    assert st.cordon_crossing_days == res.state.cordon_crossing_days
