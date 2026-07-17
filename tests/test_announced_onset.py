"""A4.2 M4 shock machinery: the announced-onset price channel (ii), the
structural-only shock rewrite gate (i), the placebo negative-control arm
(iii), and the T5 tail-off ablation.

All synthetic masked cards; the slow brain is the deterministic
OnsetStubGenerator (adapts under a priced notice, rational no-op under the
placebo's nulled notice).
"""
from __future__ import annotations

import copy
import json


from agents.baseline_loop import AnnouncedOnset, run_baseline_loop
from agents.habit_memory import SubstrateConfig
from agents.slow_brain import (
    GatedSlowBrain,
    OnsetStubGenerator,
    StandardSurprisePolicy,
)
from agents.two_brain import REASON_ANNOUNCED_ONSET
from grounding.card_validation import validate_card, validate_card_structural
from grounding.masking.mask_lint import default_token_path, lint_text, load_forbidden_tokens
from grounding.render import render_onset_rewrite_prompt
from world.config import cityk_corridor
from world.network import NetworkState
from world.tolling import announcement_of, default_schedule, placebo_announcement

CFG = cityk_corridor()

#: warm-up/toll states for the override seam: the corridor spine plus the
#: tolled fast crossing after onset.
_FREE = NetworkState(("T", "S"), None, None)
_TOLLED = NetworkState(("T", "S"), "T", default_schedule())


def _commuter(pid, cars=1):
    return {
        "persona_id": pid, "card_version": "m4-test",
        "skeleton": {"home_zone": "Z16", "work_zone": "Z04", "has_pass": True,
                     "income_class": 3, "household_cars": cars, "can_drive": True,
                     "age": 40},
        "patterns": [{"id": "commute", "weight": 1, "trips": [
            {"purpose": "work", "mode": "car", "depart_band": "am_peak"},
            {"purpose": "home", "mode": "car", "depart_band": "pm_peak"}]}],
        "rules": [], "voice": "v", "surprise_log": [], "habit_counters": {},
        "provenance": {"card_source": "llm"},
    }


def _walker(pid):
    """Off-corridor persona: never receives the onset trigger."""
    c = _commuter(pid)
    c["skeleton"]["home_zone"] = "Z28"
    c["skeleton"]["work_zone"] = "Z00"
    c["patterns"] = [{"id": "stroll", "weight": 1, "trips": [
        {"purpose": "leisure", "mode": "walk", "depart_band": "midday"}]}]
    return c


#: a car-only observed diary, so the stub's car->transit adaptation genuinely
#: FAILS the fidelity gate (mode-mix TVD 0.5 > 0.2) — the A4.2(i) case.
_OBSERVED_CAR = {
    "n_observed_weekdays": 5, "n_observed_trips": 10,
    "mean_trips_per_weekday": 2.0,
    "mode_counts": {"car": 10}, "mode_shares": {"car": 1.0},
    "has_quiet_weekday": False, "quiet_share": 0.0,
}


def _ctx(cards):
    return {
        c["persona_id"]: {
            "skeleton": c["skeleton"],
            "observed": dict(_OBSERVED_CAR),
            "observed_day_sequences": [],
        }
        for c in cards
    }


def _slots(cards, n_days, warmup):
    return {c["persona_id"]: [(100 + j, 1.0) for j in range(n_days - warmup)]
            for c in cards}


def _loop(cards, *, onset, client, n_days=6, warmup=2, namespace="onset_t",
          override=None):
    return run_baseline_loop(
        cards, CFG, _slots(cards, n_days, warmup), namespace=namespace,
        n_days=n_days, warmup_days=warmup, policy=StandardSurprisePolicy(),
        client=client, onset=onset,
        network_override=override or (lambda d: _TOLLED if onset and d >= onset.day else _FREE),
    )


def _client(cards):
    return GatedSlowBrain(OnsetStubGenerator(), _ctx(cards))


# ---------------------------------------------------------------------------
# the announced-onset trigger (A4.2(ii))
# ---------------------------------------------------------------------------

def test_onset_fires_once_per_corridor_agent_and_adapts_from_onset_day():
    cards = [_commuter("P00001"), _commuter("P00002"), _walker("P00099")]
    onset = AnnouncedOnset(day=3, announcement=announcement_of(default_schedule()))
    r = _loop(cards, onset=onset, client=_client(cards))

    # exactly one accepted onset rewrite per CORRIDOR persona, on the onset day
    audit = [a for a in r.rewrite_audit if a.day_index == 3]
    assert sorted(a.persona_id for a in audit) == ["P00001", "P00002"]
    assert all(a.accepted for a in audit)
    assert len(r.rewrite_audit) == len(audit)  # walker got nothing

    # adaptation is in force ON the onset day (announced = no discovery lag):
    # the realized work trip is transit from day 3 onward, car before
    for pid in ("P00001", "P00002"):
        by_day = {rd.day_index: rd for rd in r.realized_days_full[pid]}
        assert by_day[2].trips[0].mode == "car"
        assert by_day[3].trips[0].mode == "transit"
        assert by_day[5].trips[0].mode == "transit"
    # the walker's behavior is untouched
    assert all(rd.trips[0].mode == "walk" for rd in r.realized_days_full["P00099"])


def test_onset_rewrite_passes_structural_gate_but_would_fail_five_gates():
    card = _commuter("P00001")
    adapted = copy.deepcopy({"patterns": card["patterns"], "rules": [], "voice": "v"})
    adapted["patterns"][0]["trips"][0]["mode"] = "transit"
    skeleton, observed = card["skeleton"], dict(_OBSERVED_CAR)
    # the ordinary five-gate compose rejects the adaptation (fidelity mode-mix)
    assert any("mode mix" in e for e in validate_card(adapted, skeleton, observed, []))
    # the A4.2(i) structural channel accepts it
    assert validate_card_structural(adapted, skeleton, observed, []) == []


def test_onset_requests_collected_in_batch_mode_with_announcement():
    cards = [_commuter("P00001")]
    onset = AnnouncedOnset(day=3, announcement=announcement_of(default_schedule()))
    r = _loop(cards, onset=onset, client=None)
    onset_reqs = [p for p in r.pending_rewrites if p["reason"] == REASON_ANNOUNCED_ONSET]
    assert len(onset_reqs) == 1
    req = onset_reqs[0]
    assert req["shock_mode"] is True and req["day_index"] == 3
    assert req["announcement"]["per_trip_credits"]
    assert req["announcement"]["household_has_pass"] is True


# ---------------------------------------------------------------------------
# the placebo arm (A4.2(iii))
# ---------------------------------------------------------------------------

def test_placebo_yokes_trigger_and_stub_noop_leaves_behavior_identical():
    cards_a = [_commuter("P00001"), _commuter("P00002"), _walker("P00099")]
    cards_b = copy.deepcopy(cards_a)
    onset = AnnouncedOnset(day=3, announcement=placebo_announcement())
    # placebo world stays UNTOLLED (cost-invariant by design)
    r_pl = _loop(cards_a, onset=onset, client=_client(cards_a),
                 override=lambda d: _FREE)
    r_base = _loop(cards_b, onset=None, client=_client(cards_b),
                   override=lambda d: _FREE)

    # trigger yoked 1:1: same corridor personas, same day, all accepted
    audit = [a for a in r_pl.rewrite_audit if a.day_index == 3]
    assert sorted(a.persona_id for a in audit) == ["P00001", "P00002"]
    assert all(a.accepted for a in audit)
    # nulled reason -> the faithful stub's no-op -> realized behavior is
    # bit-identical to the no-onset baseline (zero machinery drift by stub)
    for pid in ("P00001", "P00002", "P00099"):
        assert r_pl.realized_days_full[pid] == r_base.realized_days_full[pid]


def test_placebo_announcement_carries_zero_actionable_content():
    ann = placebo_announcement()
    assert ann["per_trip_credits"] is None
    assert ann["nonpass_surcharge_credits"] is None
    prompt = render_onset_rewrite_prompt(
        skeleton={"home_zone": "Z16"}, evidence_lines=["(evidence)"],
        current_obj={"patterns": [], "rules": [], "voice": "v"},
        announcement=ann, immutable_rule_ids=[],
    )
    assert "credits" not in prompt.split("=== NOTICE")[1].split("===")[0]
    tokens = load_forbidden_tokens(default_token_path())
    assert lint_text(prompt, tokens) == []


def test_toll_onset_prompt_is_mask_lint_clean():
    ann = dict(announcement_of(default_schedule()), household_has_pass=False)
    prompt = render_onset_rewrite_prompt(
        skeleton={"home_zone": "Z16", "work_zone": "Z04"},
        evidence_lines=["(evidence)"],
        current_obj={"patterns": [], "rules": [], "voice": "v"},
        announcement=ann, immutable_rule_ids=["r1"],
    )
    tokens = load_forbidden_tokens(default_token_path())
    assert lint_text(prompt, tokens) == []
    assert "am_peak" in prompt and "does not hold a pass" in prompt


# ---------------------------------------------------------------------------
# T5 tail-off ablation + shock-mode tail
# ---------------------------------------------------------------------------

def _tail_setup():
    """Post-onset facility closure so the tail time-surprise trigger fires."""
    cards = [_commuter(f"P{i:05d}") for i in range(10)]
    jam = NetworkState(("S",), None, None)     # slow arterial only
    fast = NetworkState(("V",), None, None)    # fast spine (warm-up state)
    override = lambda d: (jam if d >= 8 else fast)  # noqa: E731
    mem_cfg = SubstrateConfig(alpha_base=0.05)
    return cards, override, mem_cfg


def _tail_loop(cards, override, mem_cfg, *, tail_surprises):
    onset = AnnouncedOnset(day=7, announcement=announcement_of(default_schedule()),
                           tail_surprises=tail_surprises)
    return run_baseline_loop(
        cards, CFG, _slots(cards, 11, 6), namespace="tail_t", n_days=11,
        warmup_days=6, policy=StandardSurprisePolicy(), client=None,
        network_override=override, memory_config=mem_cfg, onset=onset,
    )


def test_tail_off_arm_suppresses_time_surprise_rewrites_but_not_observation():
    cards, override, mem_cfg = _tail_setup()
    r_on = _tail_loop(copy.deepcopy(cards), override, mem_cfg, tail_surprises=True)
    r_off = _tail_loop(copy.deepcopy(cards), override, mem_cfg, tail_surprises=False)

    # both arms fire the onset batch (day 7) in collect mode
    for r in (r_on, r_off):
        assert sum(1 for p in r.pending_rewrites
                   if p["reason"] == REASON_ANNOUNCED_ONSET) == 10

    # the jam produces surprises in BOTH arms (observation is never suppressed)
    assert sum(r_on.surprise_counts.values()) > 0
    assert sum(r_off.surprise_counts.values()) > 0
    # tail-on collects time-surprise requests, tail-off collects none
    tail_on = [p for p in r_on.pending_rewrites if p["reason"] == "surprise"]
    tail_off = [p for p in r_off.pending_rewrites if p["reason"] == "surprise"]
    assert len(tail_on) > 0
    assert tail_off == []
    # tail requests under the shock are marked shock-mode (structural gate)
    assert all(p["shock_mode"] for p in tail_on)


# ---------------------------------------------------------------------------
# facility-load recording (the A4.2 scored-volume seam)
# ---------------------------------------------------------------------------

def test_facility_loads_recorded_per_day_with_tolled_code():
    cards = [_commuter("P00001"), _commuter("P00002")]
    onset = AnnouncedOnset(day=3, announcement=announcement_of(default_schedule()))
    r = _loop(cards, onset=onset, client=_client(cards))
    assert set(r.facility_loads) == set(range(6))
    pre, post = r.facility_loads[2], r.facility_loads[3]
    assert pre["tolled"] is None and post["tolled"] == "T"
    assert sum(pre["loads"]) == pre["n_travelers"] == 2
    # after adaptation the work commute left the car pool: the corridor car
    # equilibrium sees fewer car travelers post-onset... but the home trip is
    # still car, so travelers remain > 0
    assert post["n_travelers"] >= 1
    # loads survive the LoopState JSON round-trip
    from agents.baseline_loop import LoopState
    st = LoopState.from_dict(json.loads(json.dumps(r.state.to_dict())))
    assert st.facility_loads == r.facility_loads
