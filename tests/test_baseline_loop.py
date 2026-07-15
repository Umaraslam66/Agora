"""agents/baseline_loop.py + the e1/e2 producer-injection seam (M3 D5/D6).

Determinism (bit-identical reruns), checkpoint resume, daily habit-counter
advance vs a hand-walked example, the slot->scoring-day mapping and weights, an
injected-jam integration test that exercises the surprise -> gated-rewrite ->
cooldown machinery with inline two_brain protocol stubs (never importing
agents.slow_brain), and the producer-injection default-equivalence for both
sealed scorers. All synthetic masked cards.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from agents.baseline_loop import (
    LoopState,
    run_baseline_loop,
    _days_map_to_dict,
    STRONG_HABIT_THRESHOLD,
)
from agents.card_executor import execute_days
from agents.habit_memory import SURPRISE_LOG_CAP, SubstrateConfig
from agents.two_brain import RewriteOutcome
from evaluation import e1, e2
from grounding.seeding import _persona_id_map
from world.config import cityk_corridor
from world.crn import pick_weighted
from world.network import NetworkState

CFG = cityk_corridor()


# ---------------------------------------------------------------------------
# card + stub helpers
# ---------------------------------------------------------------------------

def make_card(pid="P00001", home="Z16", work="Z04", cars=1, rules=None):
    """Corridor commuter (outer_north -> core) with three distinct-purpose
    patterns so the drawn pattern is identifiable, like test_card_executor."""
    return {
        "persona_id": pid,
        "card_version": "m3-test",
        "skeleton": {
            "home_zone": home, "work_zone": work, "age": 39, "employed": True,
            "student": False, "can_drive": True, "household_size": 3,
            "household_cars": cars, "income_class": 3, "has_pass": True,
        },
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
        "voice": "v", "surprise_log": [], "habit_counters": {},
        "provenance": {"card_source": "llm"},
    }


class NoTriggerPolicy:
    """Records surprises to the card log, never triggers (baseline)."""

    def __init__(self):
        self.logged = 0

    def log_surprise(self, card, event):
        log = card.setdefault("surprise_log", [])
        log.append({"day_index": event.day_index, "context_key": event.context_key})
        while len(log) > SURPRISE_LOG_CAP:
            log.pop(0)
        self.logged += 1

    def should_rewrite(self, card, day_index):
        return False


class ShockCooldownPolicy:
    """A test SurprisePolicy: fires when a surprise was logged for THIS day,
    respecting warm-up and a per-persona cooldown read from the card's rewrite
    provenance (the two_brain seam; no agents.slow_brain import)."""

    def __init__(self, warmup, cooldown):
        self.warmup = warmup
        self.cooldown = cooldown
        self.should_calls = []

    def log_surprise(self, card, event):
        log = card.setdefault("surprise_log", [])
        log.append({"day_index": event.day_index, "context_key": event.context_key,
                    "z": event.z, "status": "open"})
        while len(log) > SURPRISE_LOG_CAP:
            log.pop(0)

    def should_rewrite(self, card, day_index):
        self.should_calls.append((card["persona_id"], day_index))
        if day_index < self.warmup:
            return False
        fired_today = any(e["day_index"] == day_index for e in card.get("surprise_log", []))
        rewrites = card.get("provenance", {}).get("rewrites", [])
        accepted_days = [r["day_index"] for r in rewrites if r.get("accepted")]
        if accepted_days and (day_index - max(accepted_days)) <= self.cooldown:
            return False
        return fired_today


class AcceptingClient:
    """A test SlowBrainClient: accepts every request, marking the card and
    appending to its rewrite provenance (so cooldown works). Records each batch
    (day, persona ids) so the test can assert what fired."""

    def __init__(self):
        self.batches = []

    def rewrite_batch(self, requests):
        self.batches.append((requests[0].day_index if requests else None,
                             [r.persona_id for r in requests]))
        outcomes = []
        for req in requests:
            new_card = copy.deepcopy(dict(req.card))
            prov = new_card.setdefault("provenance", {})
            prov.setdefault("rewrites", []).append(
                {"day_index": req.day_index, "attempt": req.attempt, "accepted": True})
            new_card["voice"] = new_card.get("voice", "") + " [rw]"
            outcomes.append(RewriteOutcome(
                persona_id=req.persona_id, day_index=req.day_index, accepted=True,
                card=new_card, attempts_used=1, gate_failures=()))
        return outcomes


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def _slots(pids, n, w=1.0, start=100):
    return {pid: [(start + j, w + 0.1 * j) for j in range(n)] for pid in pids}


def test_loop_is_bit_identical_across_reruns():
    cards = [make_card(f"P{i:05d}") for i in range(12)]
    slots = _slots([c["persona_id"] for c in cards], 3)
    kw = dict(namespace="m3_run0", n_days=13, warmup_days=10, policy=NoTriggerPolicy())
    r1 = run_baseline_loop(cards, CFG, slots, client=None, **kw)
    r2 = run_baseline_loop(cards, CFG, slots, client=None, **kw)
    assert _days_map_to_dict(r1.scoring_days) == _days_map_to_dict(r2.scoring_days)
    # habit counters advanced identically; input cards untouched (copy_cards)
    assert [c["habit_counters"] for c in r1.cards] == [c["habit_counters"] for c in r2.cards]
    assert cards[0]["habit_counters"] == {}  # caller's cards never mutated
    assert dict(r1.surprise_counts) == dict(r2.surprise_counts)


# ---------------------------------------------------------------------------
# checkpoint resume
# ---------------------------------------------------------------------------

def test_checkpoint_resume_reproduces_uninterrupted_run():
    cards = [make_card(f"P{i:05d}") for i in range(10)]
    slots = _slots([c["persona_id"] for c in cards], 3)
    kw = dict(config=CFG, day_slots=slots, namespace="m3_run0", n_days=13,
              warmup_days=10, policy=NoTriggerPolicy(), client=None)

    full = run_baseline_loop(cards, **kw)

    partial = run_baseline_loop(cards, run_through_day=6, **kw)  # days 0..6
    assert partial.state.day_index == 7
    # round-trip the checkpoint through JSON to prove it is fully serializable
    blob = json.dumps(partial.state.to_dict())
    state2 = LoopState.from_dict(json.loads(blob))
    resumed = run_baseline_loop(cards, resume_state=state2, config=CFG, day_slots=slots,
                                namespace="m3_run0", n_days=13, warmup_days=10,
                                policy=NoTriggerPolicy(), client=None)

    assert _days_map_to_dict(full.scoring_days) == _days_map_to_dict(resumed.scoring_days)
    assert [c["habit_counters"] for c in full.cards] == [c["habit_counters"] for c in resumed.cards]
    assert dict(full.surprise_counts) == dict(resumed.surprise_counts)


# ---------------------------------------------------------------------------
# habit counters advance daily (hand-walked 3-day example)
# ---------------------------------------------------------------------------

def _drawn_pattern_id(card, day, namespace):
    weights = [p["weight"] for p in card["patterns"]]
    return pick_weighted(f"{namespace}:{card['persona_id']}:{day}:pattern",
                         card["patterns"], weights)["id"]


def test_habit_counters_advance_daily_hand_walk():
    card = make_card("P00001")
    slots = {"P00001": [(0, 1.0), (1, 1.0), (2, 1.0)]}
    # loop with warmup 0 so all three global days 0,1,2 are lived and scored
    r = run_baseline_loop([card], CFG, slots, namespace="run0", n_days=3,
                          warmup_days=0, policy=NoTriggerPolicy(), client=None)
    counters = r.cards[0]["habit_counters"]
    assert set(counters) == {"workday", "errand_day", "quiet_day"}
    for cd in counters.values():
        assert cd["days_observed"] == 3

    # hand-walk: which pattern is drawn on each of global days 0,1,2
    drawn = [_drawn_pattern_id(card, d, "run0") for d in (0, 1, 2)]
    expected_follows = {pid: drawn.count(pid) for pid in ("workday", "errand_day", "quiet_day")}
    for pid, n in expected_follows.items():
        assert counters[pid]["total_days_followed"] == n
    assert sum(expected_follows.values()) == 3  # exactly one pattern followed per day
    # the loop's counters equal a direct execute_days over the same global days
    ref_card = make_card("P00001")
    execute_days([ref_card], slots, "run0", update_habits=True)
    assert counters == ref_card["habit_counters"]


# ---------------------------------------------------------------------------
# slot -> scoring-day mapping and weights (D5)
# ---------------------------------------------------------------------------

def test_slot_to_scoring_day_mapping_and_weights():
    card = make_card("P00001")
    # three observed slots (their daynums are irrelevant to the mapping)
    slots = {"P00001": [(41, 1.5), (58, 0.5), (12, 2.0)]}
    warmup, n_days = 5, 8  # scoring window = global days 5,6,7
    r = run_baseline_loop([card], CFG, slots, namespace="run0", n_days=n_days,
                          warmup_days=warmup, policy=NoTriggerPolicy(), client=None)
    days = r.scoring_days["P00001"]
    # emitted day_index is the OBSERVED slot daynum (execute_days's exact
    # shape — downstream day-index joins depend on it); slot j is LIVED on
    # global day warmup+j, which drives the CRN draws (asserted below).
    assert [d.day_index for d in days] == [41, 58, 12]
    assert [d.day_weight for d in days] == [1.5, 0.5, 2.0]    # slot weights carried
    # trips equal the realized trips on those GLOBAL days (not the diary daynums)
    for j, gd in enumerate((5, 6, 7)):
        ref = execute_days([make_card("P00001")], {"P00001": [(gd, 1.0)]}, "run0",
                           update_habits=False)["P00001"][0]
        assert [(t.mode, t.depart_band) for t in days[j].trips] == \
               [(t.mode, t.depart_band) for t in ref.trips]


def test_scoring_days_truncated_and_padded_to_slot_count():
    cards = [make_card("P00001"), make_card("P00002")]
    # P1 has more slots than the scoring window (4 vs 2); P2 has fewer (1)
    slots = {"P00001": [(0, 1.0)] * 4, "P00002": [(0, 1.0)]}
    r = run_baseline_loop(cards, CFG, slots, namespace="run0", n_days=6, warmup_days=4,
                          policy=NoTriggerPolicy(), client=None)
    assert len(r.scoring_days["P00001"]) == 2  # capped at scoring_days = n_days-warmup
    assert len(r.scoring_days["P00002"]) == 1  # only as many as observed slots
    # a card with no slots is present as a key with an empty list (execute_days shape)
    r2 = run_baseline_loop([make_card("P99999")], CFG, {}, namespace="run0", n_days=6,
                           warmup_days=4, policy=NoTriggerPolicy(), client=None)
    assert r2.scoring_days["P99999"] == []


# ---------------------------------------------------------------------------
# injected-jam integration: surprise -> gated rewrite -> cooldown
# ---------------------------------------------------------------------------

_FAST = NetworkState(("V",), None, None)   # fast elevated spine only (warm-up)
_JAM = NetworkState(("S",), None, None)    # slow arterial only (facility closure)


def _jam_card(pid):
    """A single-pattern corridor commuter who drives EVERY lived day, so every
    persona is a corridor car traveler every day (isolates the jam signal)."""
    return {
        "persona_id": pid, "card_version": "m3-test",
        "skeleton": {"home_zone": "Z16", "work_zone": "Z04", "has_pass": True,
                     "income_class": 3, "household_cars": 1, "can_drive": True},
        "patterns": [{"id": "commute", "weight": 1, "trips": [
            {"purpose": "work", "mode": "car", "depart_band": "am_peak"}]}],
        "rules": [], "voice": "v", "surprise_log": [], "habit_counters": {},
        "provenance": {"card_source": "llm"},
    }


def _jam_setup(n_cards=40, jam_day=7, warmup=6, n_days=10):
    cards = [_jam_card(f"P{i:05d}") for i in range(n_cards)]
    slots = {c["persona_id"]: [(200 + j, 1.0) for j in range(n_days - warmup)] for c in cards}
    override = lambda d: (_JAM if d >= jam_day else _FAST)  # noqa: E731
    # a slow-adapting memory so the injected jam produces SUSTAINED surprises
    # (default alpha absorbs a moderate jam in one day), isolating the cooldown
    mem_cfg = SubstrateConfig(alpha_base=0.05)
    return cards, slots, override, mem_cfg


def test_injected_jam_fires_surprises_and_gated_rewrite_with_cooldown():
    cards, slots, override, mem_cfg = _jam_setup(warmup=6, jam_day=7, n_days=10)
    policy = ShockCooldownPolicy(warmup=6, cooldown=3)
    client = AcceptingClient()
    r = run_baseline_loop(cards, CFG, slots, namespace="jam", n_days=10, warmup_days=6,
                          policy=policy, client=client, network_override=override,
                          memory_config=mem_cfg)

    # warm-up quiet (expectations seeded on the fast state); jam fires post-warm-up
    assert all(r.surprise_counts[d] == 0 for d in range(7))
    assert r.surprise_counts[7] == 40  # every corridor car persona surprises

    # the stub policy and client were both driven
    assert policy.should_calls
    assert len(client.batches) == 1                 # exactly ONE batch fired...
    day, pids = client.batches[0]
    assert day == 7 and len(pids) == 40             # ...on the jam day, for all 40

    # accepted outcomes replaced the cards in place
    assert all(r.cards[i]["voice"].endswith(" [rw]") for i in range(40))
    assert all(r.cards[i]["provenance"]["rewrites"][-1]["accepted"] for i in range(40))
    audit = r.rewrite_audit
    assert len(audit) == 40 and all(a.accepted and a.day_index == 7 for a in audit)

    # cooldown suppresses the immediate re-fire on days 8 and 9 (within cooldown 3)
    # even though surprises keep firing there
    assert r.surprise_counts[8] > 0 and r.surprise_counts[9] > 0
    assert all(day != 8 and day != 9 for day, _ in client.batches)


def test_injected_jam_client_none_collects_pending_without_rewriting():
    cards, slots, override, mem_cfg = _jam_setup(warmup=6, jam_day=7, n_days=10)
    policy = ShockCooldownPolicy(warmup=6, cooldown=3)
    r = run_baseline_loop(cards, CFG, slots, namespace="jam", n_days=10, warmup_days=6,
                          policy=policy, client=None, network_override=override,
                          memory_config=mem_cfg)
    # requests are collected, not applied
    assert len(r.pending_rewrites) >= 40
    assert {req["day_index"] for req in r.pending_rewrites} >= {7}
    assert r.rewrite_audit == []
    # cards are untouched by any rewrite (no marker, no rewrite provenance)
    assert all(not c["voice"].endswith(" [rw]") for c in r.cards)
    assert all("rewrites" not in c.get("provenance", {}) for c in r.cards)


def test_strong_rule_ids_read_from_counters():
    from agents.baseline_loop import _strong_rule_ids
    card = make_card("P1", rules=[{"id": "r_strong", "when": {"purpose": "work"},
                                   "then": {"mode": "transit"}},
                                  {"id": "r_weak", "when": {"purpose": "home"},
                                   "then": {"mode": "walk"}}])
    card["habit_counters"] = {
        "r_strong": {"strength": STRONG_HABIT_THRESHOLD, "window_days": 30, "window": [],
                     "total_days_followed": 20, "days_observed": 20},
        "r_weak": {"strength": STRONG_HABIT_THRESHOLD - 1, "window_days": 30, "window": [],
                   "total_days_followed": 5, "days_observed": 20},
    }
    assert _strong_rule_ids(card) == ("r_strong",)


# ---------------------------------------------------------------------------
# producer-injection equivalence (E1 simulate_arm, E2 score_e2) — D6
# ---------------------------------------------------------------------------

def test_e1_simulate_arm_producer_default_equivalence_and_namespaces():
    cards = [make_card(f"P{i:05d}") for i in range(8)]
    day_slots = _slots([c["persona_id"] for c in cards], 3)
    persona_cell = {c["persona_id"]: None for c in cards}

    seen = []

    def replica(namespace):
        seen.append(namespace)
        return execute_days(cards, day_slots, namespace, update_habits=False)

    given = e1.simulate_arm(cards, None, persona_cell, day_slots, n_runs=3,
                            namespace_prefix="method_", producer=replica)
    default = e1.simulate_arm(cards, None, persona_cell, day_slots, n_runs=3,
                              namespace_prefix="method_")  # producer=None -> internal path
    # run count + namespaces respected
    assert seen == ["method_run0", "method_run1", "method_run2"]
    # the injected replica reproduces the default path family-for-family
    for fam in e1.FAMILIES:
        np.testing.assert_array_equal(given.pooled[fam], default.pooled[fam])
    assert given.n_runs == default.n_runs == 3


def _e2_dataset():
    persons = pd.DataFrame([
        {"person_id": "1101", "household_id": "H1"},
        {"person_id": "1102", "household_id": "H1"},
        {"person_id": "1201", "household_id": "H2"},
    ])
    person_days = pd.DataFrame([
        {"person_id": "1101", "daynum": 1, "w_day": 1.0, "n_collapsed": 2, "w_person": 1.0},
        {"person_id": "1101", "daynum": 2, "w_day": 2.0, "n_collapsed": 0, "w_person": 1.0},
        {"person_id": "1102", "daynum": 1, "w_day": 2.0, "n_collapsed": 2, "w_person": 1.0},
        {"person_id": "1201", "daynum": 1, "w_day": 1.0, "n_collapsed": 1, "w_person": 1.0},
    ])
    weekday_trips = pd.DataFrame([
        {"person_id": "1101", "daynum": 1, "tripnum": 1, "mode": "car", "band": "am_peak", "w_trip": 1.0},
        {"person_id": "1102", "daynum": 1, "tripnum": 1, "mode": "ride", "band": "midday", "w_trip": 1.0},
        {"person_id": "1201", "daynum": 1, "tripnum": 1, "mode": "car", "band": "am_peak", "w_trip": 1.0},
    ])
    return SimpleNamespace(persons=persons, person_days=person_days, weekday_trips=weekday_trips)


def _e2_cards():
    ids = _persona_id_map(pd.Series(["1101", "1102", "1201"]))
    def card(pid):
        return {"persona_id": pid, "skeleton": {"household_cars": 1, "can_drive": True},
                "patterns": [
                    {"id": "commute", "weight": 6, "trips": [
                        {"purpose": "work", "mode": "car", "depart_band": "am_peak"}]},
                    {"id": "transit_day", "weight": 4, "trips": [
                        {"purpose": "work", "mode": "transit", "depart_band": "am_peak"}]},
                    {"id": "quiet", "weight": 2, "trips": []}],
                "rules": []}
    return [card(ids["1101"]), card(ids["1102"]), card(ids["1201"])]


def test_e2_score_producer_default_equivalence_and_namespaces():
    ds = _e2_dataset()
    cards = _e2_cards()
    id_map = _persona_id_map(ds.persons["person_id"].astype(str))
    slots = e2.day_slots_of(ds.person_days, id_map)

    seen = []

    def replica(namespace):
        seen.append(namespace)
        return execute_days(cards, slots, namespace, update_habits=False)

    given = e2.score_e2(cards, ds, n_runs=3, seed=0, producer=replica)
    default = e2.score_e2(_e2_cards(), ds, n_runs=3, seed=0)  # producer=None
    assert seen == ["run0", "run1", "run2"]
    # byte-identical verdict dicts (the injected replica == the internal path)
    assert json.dumps(default, sort_keys=True, default=str) == \
           json.dumps(given, sort_keys=True, default=str)
