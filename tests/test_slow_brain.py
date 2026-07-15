"""agents/slow_brain.py — the surprise-triggered slow-brain rewrite module (M3).

Synthetic masked fixtures only. Covers: surprise-log cap + resolved-before-open
eviction order; the trigger policy (sustained run, single shock, warm-up
suppression, post-accept cooldown); mechanical strong-rule immutability
rejection; fidelity-gate rejection with a retry that still fails leaving the old
card; habit-counter continuity on accept (kept / reset / dropped); provenance
append with a stable card_source; surprise resolution; masked-lint cleanliness
of the rendered rewrite prompt; and StubGenerator determinism.
"""
from __future__ import annotations

import copy
import json

import pandas as pd

from agents import slow_brain as sb
from agents.habit_memory import HabitCounter
from agents.two_brain import RewriteRequest, SurpriseEvent
from grounding import card_validation as cv
from grounding.masking.mask_lint import (
    default_token_path,
    lint_text,
    load_forbidden_tokens,
)
from grounding.seeding import observed_stats_of

FORBIDDEN = load_forbidden_tokens(default_token_path())
PLANTED_TOKEN = "gamla stan"
assert PLANTED_TOKEN in FORBIDDEN

SKELETON = {
    "home_zone": "Z04", "work_zone": "Z11", "age": 39, "employed": True,
    "student": False, "can_drive": True, "household_size": 3,
    "household_cars": 1, "income_class": 4, "has_pass": True,
}


# ---------------------------------------------------------------------------
# fixtures: a faithful two-trip commuter card + its observed reference
# ---------------------------------------------------------------------------

def _llm_obj(rules=None):
    return {
        "patterns": [
            {
                "id": "commute",
                "weight": 5,
                "trips": [
                    {"purpose": "work", "mode": "car", "depart_band": "am_peak"},
                    {"purpose": "home", "mode": "car", "depart_band": "pm_peak"},
                ],
            }
        ],
        "rules": [
            {"id": "r1", "when": {"purpose": "shop_daily"},
             "then": {"mode": "walk", "depart_band": "midday"}}
        ] if rules is None else rules,
        "voice": "I keep to a plain, steady commute.",
    }


def _frames():
    pdays = pd.DataFrame([{"daynum": 1, "n_collapsed": 2}])
    trips = pd.DataFrame(
        [[1, 1, "work", "car", "am_peak"], [1, 2, "home", "car", "pm_peak"]],
        columns=["daynum", "tripnum", "purpose", "mode", "band"],
    )
    return pdays, trips


def _observed():
    return observed_stats_of(*_frames())


def _seqs():
    return cv.day_signatures(*_frames())


def _card(rules=None, source="llm"):
    return cv.assemble_card("P00001", SKELETON, _llm_obj(rules), {"card_source": source})


def _vctx():
    return {"P00001": {"skeleton": SKELETON, "observed": _observed(),
                       "observed_day_sequences": _seqs()}}


def _event(day, key="car|corridor|am_peak", exp=30.0, real=35.0, z=0.5):
    return SurpriseEvent("P00001", day, key, exp, real, z)


def _request(card, day=12, surprises=(), strong=()):
    return RewriteRequest("P00001", day, card, tuple(surprises), tuple(strong))


# ---------------------------------------------------------------------------
# surprise-log cap + eviction order
# ---------------------------------------------------------------------------

def test_log_appends_open_entries_with_masked_fields():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    policy.log_surprise(card, _event(11, exp=25.0, real=31.0, z=0.6))
    assert len(card["surprise_log"]) == 1
    e = card["surprise_log"][0]
    assert e["status"] == "open"
    assert e["context_key"] == "car|corridor|am_peak"
    assert e["expected_minutes"] == 25.0 and e["realized_minutes"] == 31.0
    assert e["z"] == 0.6 and e["day_index"] == 11


def test_log_cap_drops_oldest_when_no_resolved():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    for d in range(sb.SURPRISE_LOG_CAP + 2):  # 7 appends, cap 5
        policy.log_surprise(card, _event(d, key=f"car|corridor|k{d}"))
    log = card["surprise_log"]
    assert len(log) == sb.SURPRISE_LOG_CAP
    # the two oldest (days 0, 1) were evicted; newest five remain in order
    assert [e["day_index"] for e in log] == [2, 3, 4, 5, 6]


def test_log_cap_evicts_resolved_before_open():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    # fill to cap, then resolve the MIDDLE (not oldest) entry
    for d in range(sb.SURPRISE_LOG_CAP):  # days 0..4
        policy.log_surprise(card, _event(d, key=f"car|corridor|k{d}"))
    card["surprise_log"][2]["status"] = "resolved"  # day 2 resolved
    policy.log_surprise(card, _event(99, key="car|corridor|new"))
    days = [e["day_index"] for e in card["surprise_log"]]
    # the resolved day-2 entry is dropped even though days 0,1 are older
    assert 2 not in days
    assert days == [0, 1, 3, 4, 99]


# ---------------------------------------------------------------------------
# trigger policy: warm-up, shock, sustained, cooldown
# ---------------------------------------------------------------------------

def test_no_trigger_during_warmup_even_on_shock():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    policy.log_surprise(card, _event(9, z=sb.SHOCK_TRIGGER_Z + 1.0))
    assert policy.should_rewrite(card, day_index=9) is False  # 9 < WARMUP_DAYS=10


def test_shock_triggers_after_warmup():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    policy.log_surprise(card, _event(10, z=sb.SHOCK_TRIGGER_Z))
    assert policy.should_rewrite(card, day_index=10) is True


def test_small_single_surprise_does_not_trigger():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    policy.log_surprise(card, _event(12, z=0.6))  # one sub-shock, not sustained
    assert policy.should_rewrite(card, day_index=12) is False


def test_sustained_run_triggers_same_context_key():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    # same key, sub-shock z, on 3 of the trailing 5 days (10,11,12) -> trigger
    for d in (10, 11, 12):
        policy.log_surprise(card, _event(d, key="car|corridor|am_peak", z=0.7))
    assert policy.should_rewrite(card, day_index=12) is True


def test_sustained_needs_same_key_not_scattered():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    for d in (10, 11, 12):  # three different keys -> no sustained run
        policy.log_surprise(card, _event(d, key=f"car|corridor|k{d}", z=0.7))
    assert policy.should_rewrite(card, day_index=12) is False


def test_sustained_window_excludes_stale_days():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    # days 6,7 are outside the trailing-5 window at day 12; only day 12 inside
    for d in (6, 7, 12):
        policy.log_surprise(card, _event(d, key="car|corridor|am_peak", z=0.7))
    assert policy.should_rewrite(card, day_index=12) is False


def test_cooldown_suppresses_after_recent_accept():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    card["provenance"]["rewrites"] = [
        {"day_index": 12, "attempt": 1, "model": "m", "prompt_sha": "x", "accepted": True}
    ]
    policy.log_surprise(card, _event(15, z=sb.SHOCK_TRIGGER_Z + 1.0))
    # day 15 is 3 days after an accepted rewrite (< cooldown 7) -> suppressed
    assert policy.should_rewrite(card, day_index=15) is False
    # ... but a shock 7 days out clears cooldown
    policy.log_surprise(card, _event(19, z=sb.SHOCK_TRIGGER_Z + 1.0))
    assert policy.should_rewrite(card, day_index=19) is True


def test_rejected_rewrite_does_not_start_cooldown():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    card["provenance"]["rewrites"] = [
        {"day_index": 12, "attempt": 2, "model": "m", "prompt_sha": "x", "accepted": False}
    ]
    policy.log_surprise(card, _event(13, z=sb.SHOCK_TRIGGER_Z + 1.0))
    assert policy.should_rewrite(card, day_index=13) is True  # rejection is not a cooldown


# ---------------------------------------------------------------------------
# strong_rule_ids_for + mechanical immutability rejection
# ---------------------------------------------------------------------------

def test_strong_rule_ids_reads_habit_counters():
    card = _card()
    strong = HabitCounter()
    for _ in range(sb.STRONG_HABIT_THRESHOLD):
        strong.record_day(True)
    card["habit_counters"]["r1"] = strong.to_dict()
    assert sb.strong_rule_ids_for(card) == ("r1",)
    # one below threshold is not strong
    weak = HabitCounter()
    for _ in range(sb.STRONG_HABIT_THRESHOLD - 1):
        weak.record_day(True)
    card["habit_counters"]["r1"] = weak.to_dict()
    assert sb.strong_rule_ids_for(card) == ()


def test_dropped_strong_rule_is_mechanically_restored():
    # D4 revision (M3 rehearsal evidence): a rewrite that drops a strong rule
    # is REPAIRED — the rule re-enters verbatim — never rejected for it.
    def drop_r1(requests):
        outs = []
        for req in requests:
            obj = {"patterns": copy.deepcopy(list(req.card["patterns"])),
                   "rules": [], "voice": req.card["voice"]}
            outs.append(json.dumps(obj))
        return outs

    card = _card()
    gsb = sb.GatedSlowBrain(drop_r1, _vctx())
    req = _request(card, surprises=(_event(12, z=2.5),), strong=("r1",))
    out = gsb.rewrite_batch([req])[0]
    assert out.accepted is True
    restored = [r for r in out.card["rules"] if r["id"] == "r1"]
    assert restored == [card["rules"][0]]  # original content, verbatim


def test_altered_strong_rule_is_restored_and_shadow_guarded():
    def alter_r1_and_add(requests):
        outs = []
        for req in requests:
            rules = copy.deepcopy(list(req.card["rules"]))
            rules[0]["then"]["mode"] = "bike"  # content change to strong rule
            # plus a new rule FIRST that would shadow r1 under first-match-wins
            rules.insert(0, {"id": "shadow", "when": dict(rules[0]["when"]),
                             "then": {"mode": "walk"}})
            obj = {"patterns": copy.deepcopy(list(req.card["patterns"])),
                   "rules": rules, "voice": req.card["voice"]}
            outs.append(json.dumps(obj))
        return outs

    card = _card()
    gsb = sb.GatedSlowBrain(alter_r1_and_add, _vctx())
    req = _request(card, surprises=(_event(12, z=2.5),), strong=("r1",))
    out = gsb.rewrite_batch([req])[0]
    assert out.accepted is True
    rules = out.card["rules"]
    # restored strong rule sits AHEAD of every proposed rule (shadow guard)
    assert rules[0] == card["rules"][0]
    assert [r["id"] for r in rules].index("r1") < [r["id"] for r in rules].index("shadow")


def test_restore_strong_rules_passthrough_when_verbatim():
    card = _card()
    obj = {"patterns": copy.deepcopy(card["patterns"]),
           "rules": copy.deepcopy(card["rules"]), "voice": card["voice"]}
    repaired, audit = sb.restore_strong_rules(obj, card, ("r1",))
    assert audit == []
    assert repaired["rules"] == obj["rules"]  # order-preserved, byte-unchanged


def test_stub_preserves_strong_rule_and_is_accepted():
    gsb = sb.GatedSlowBrain(sb.StubGenerator(), _vctx())
    req = _request(_card(), surprises=(_event(12, z=2.5),), strong=("r1",))
    out = gsb.rewrite_batch([req])[0]
    assert out.accepted is True
    rule_ids = [r["id"] for r in out.card["rules"]]
    assert "r1" in rule_ids  # strong rule kept verbatim
    assert any(rid.startswith("shift_") for rid in rule_ids)  # new override added


# ---------------------------------------------------------------------------
# fidelity-gate rejection + retry-then-keep-old
# ---------------------------------------------------------------------------

def test_fidelity_failure_retries_then_keeps_old_card():
    # a generator that always returns an UNFAITHFUL card (mean 1 vs observed 2)
    calls = {"n": 0}

    def unfaithful(requests):
        calls["n"] += 1
        outs = []
        for _req in requests:
            obj = {"patterns": [{"id": "half", "weight": 5,
                                 "trips": [{"purpose": "work", "mode": "car",
                                            "depart_band": "am_peak"}]}],
                   "rules": [], "voice": "I only do half."}
            outs.append(json.dumps(obj))
        return outs

    card = _card()
    before = copy.deepcopy(card)
    gsb = sb.GatedSlowBrain(unfaithful, _vctx(), model="mX")
    out = gsb.rewrite_batch([_request(card, surprises=(_event(12, z=2.5),))])[0]

    assert out.accepted is False
    assert out.attempts_used == sb.MAX_REWRITE_ATTEMPTS
    assert calls["n"] == sb.MAX_REWRITE_ATTEMPTS  # generator called twice (retry)
    assert any("trips per weekday" in f for f in out.gate_failures)
    # old card stands: patterns/rules/voice/counters/log untouched
    assert out.card["patterns"] == before["patterns"]
    assert out.card["voice"] == before["voice"]
    assert out.card["habit_counters"] == before["habit_counters"]
    # only an audit record with accepted:false was appended
    rewrites = out.card["provenance"]["rewrites"]
    assert len(rewrites) == 1 and rewrites[0]["accepted"] is False


def test_retry_prompt_appends_failure_block():
    # attempt-2 prompt must carry the machine-readable rejection reasons
    reqs_seen = {"prompts": []}

    def capture_then_fail(requests):
        outs = []
        for req in requests:
            # render what the client renders to inspect the retry block
            outs.append(json.dumps({"patterns": [{"id": "half", "weight": 5,
                        "trips": [{"purpose": "work", "mode": "car",
                                   "depart_band": "am_peak"}]}],
                        "rules": [], "voice": "half"}))
        return outs

    gsb = sb.GatedSlowBrain(capture_then_fail, _vctx())
    req = _request(_card(), surprises=(_event(12, z=2.5),))
    # attempt 1 prompt has no failure block; attempt 2 does
    p1 = gsb._attempt_text(req, ())
    p2 = gsb._attempt_text(req, ("card implies 1.00 trips per weekday; the evidence shows 2.00",))
    assert "PREVIOUS ATTEMPT FAILED VALIDATION" not in p1
    assert "PREVIOUS ATTEMPT FAILED VALIDATION" in p2
    assert "1.00 trips per weekday" in p2


def test_invalid_json_is_a_rejection_not_an_error():
    def broken(requests):
        return ["this is not json" for _ in requests]

    gsb = sb.GatedSlowBrain(broken, _vctx())
    out = gsb.rewrite_batch([_request(_card(), surprises=(_event(12, z=2.5),))])[0]
    assert out.accepted is False
    assert any("valid JSON" in f for f in out.gate_failures)


# ---------------------------------------------------------------------------
# habit-counter continuity on accept: kept / reset / dropped
# ---------------------------------------------------------------------------

def test_counter_continuity_kept_reset_dropped():
    # start from a card whose counters carry real history
    card = _card(rules=[
        {"id": "r_keep", "when": {"purpose": "shop_daily"}, "then": {"mode": "walk"}},
        {"id": "r_drop", "when": {"purpose": "leisure"}, "then": {"mode": "bike"}},
    ])
    for cid in ("commute", "r_keep", "r_drop"):
        counter = HabitCounter()
        for _ in range(20):
            counter.record_day(True)
        card["habit_counters"][cid] = counter.to_dict()

    # generator: keep 'commute' + 'r_keep' unchanged, DROP 'r_drop', CHANGE
    # 'commute'... no — change a NEW pattern id, keep commute identical.
    def rewrite(requests):
        outs = []
        for req in requests:
            obj = {
                "patterns": [
                    # 'commute' unchanged by id AND content -> counter kept
                    copy.deepcopy(req.card["patterns"][0]),
                ],
                "rules": [
                    # 'r_keep' unchanged -> kept; 'r_drop' absent -> dropped
                    copy.deepcopy(req.card["rules"][0]),
                    # a brand-new rule -> fresh counter
                    {"id": "r_new", "when": {"purpose": "work"},
                     "then": {"depart_band": "midday"}},
                ],
                "voice": req.card["voice"],
            }
            outs.append(json.dumps(obj))
        return outs

    gsb = sb.GatedSlowBrain(rewrite, _vctx())
    out = gsb.rewrite_batch([_request(card, surprises=(_event(12, z=2.5),))])[0]
    assert out.accepted is True
    counters = out.card["habit_counters"]
    assert set(counters) == {"commute", "r_keep", "r_new"}   # r_drop dropped
    assert counters["commute"]["strength"] == 20             # kept intact
    assert counters["r_keep"]["strength"] == 20              # kept intact
    assert counters["r_new"]["strength"] == 0                # fresh (reset)
    assert counters["r_new"]["days_observed"] == 0


def test_changed_content_same_id_resets_counter():
    card = _card()
    counter = HabitCounter()
    for _ in range(20):
        counter.record_day(True)
    card["habit_counters"]["commute"] = counter.to_dict()

    def bump_weight(requests):
        outs = []
        for req in requests:
            pat = copy.deepcopy(req.card["patterns"][0])
            pat["weight"] = 6  # same id, different content -> reset
            obj = {"patterns": [pat], "rules": copy.deepcopy(list(req.card["rules"])),
                   "voice": req.card["voice"]}
            outs.append(json.dumps(obj))
        return outs

    gsb = sb.GatedSlowBrain(bump_weight, _vctx())
    out = gsb.rewrite_batch([_request(card, surprises=(_event(12, z=2.5),))])[0]
    assert out.accepted is True
    assert out.card["habit_counters"]["commute"]["strength"] == 0  # rewritten -> reset


# ---------------------------------------------------------------------------
# provenance append + card_source stability + surprise resolution
# ---------------------------------------------------------------------------

def test_accept_appends_provenance_and_keeps_card_source():
    card = _card(source="fallback")  # fallback cards are rewrite-eligible too
    gsb = sb.GatedSlowBrain(sb.StubGenerator(), _vctx(), model="qwen-x")
    out = gsb.rewrite_batch([_request(card, day=12, surprises=(_event(12, z=2.5),))])[0]
    assert out.accepted is True
    prov = out.card["provenance"]
    assert prov["card_source"] == "fallback"  # NEVER changes
    rec = prov["rewrites"][-1]
    assert rec == {"day_index": 12, "attempt": 1, "model": "qwen-x",
                   "prompt_sha": rec["prompt_sha"], "accepted": True}
    assert isinstance(rec["prompt_sha"], str) and len(rec["prompt_sha"]) == 64


def test_accept_resolves_open_surprises():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    policy.log_surprise(card, _event(11, z=0.7))
    policy.log_surprise(card, _event(12, key="car|corridor|am_peak", z=2.5))
    assert all(e["status"] == "open" for e in card["surprise_log"])
    surprises = tuple(SurpriseEvent("P00001", e["day_index"], e["context_key"],
                                    e["expected_minutes"], e["realized_minutes"], e["z"])
                      for e in card["surprise_log"])
    gsb = sb.GatedSlowBrain(sb.StubGenerator(), _vctx())
    out = gsb.rewrite_batch([_request(card, day=12, surprises=surprises)])[0]
    assert out.accepted is True
    assert all(e["status"] == "resolved" for e in out.card["surprise_log"])
    # a subsequent append can now evict a resolved entry before an open one
    policy.log_surprise(out.card, _event(13, key="car|corridor|fresh"))
    assert out.card["surprise_log"][-1]["status"] == "open"


def test_reject_leaves_counters_and_log_untouched():
    policy = sb.StandardSurprisePolicy()
    card = _card()
    policy.log_surprise(card, _event(12, z=2.5))
    before_log = copy.deepcopy(card["surprise_log"])

    def broken(requests):
        return ["nope" for _ in requests]

    gsb = sb.GatedSlowBrain(broken, _vctx())
    out = gsb.rewrite_batch([_request(card, surprises=(_event(12, z=2.5),))])[0]
    assert out.accepted is False
    assert out.card["surprise_log"] == before_log  # still open, untouched
    assert out.card["provenance"]["rewrites"][-1]["accepted"] is False


# ---------------------------------------------------------------------------
# rendered rewrite prompt: masked-lint clean + batch record shape
# ---------------------------------------------------------------------------

def test_rewrite_prompt_is_mask_lint_clean():
    req = _request(_card(), surprises=(_event(12, z=2.5),), strong=("r1",))
    records = sb.build_rewrite_prompt_records([req])
    assert len(records) == 1
    rec = records[0]
    assert set(rec) == {"persona_id", "prompt", "attempt"}
    assert rec["persona_id"] == "P00001" and rec["attempt"] == 1
    assert lint_text(rec["prompt"], FORBIDDEN) == []
    # the prompt strips harness fields but shows the model-owned card
    assert "habit_counters" not in rec["prompt"]
    assert "surprise_log" not in rec["prompt"]
    assert "commute" in rec["prompt"]  # a pattern id is shown


def test_rewrite_prompt_catches_planted_contamination():
    dirty = _card()
    dirty["voice"] = f"I stroll through {PLANTED_TOKEN} on my way home."
    req = _request(dirty, surprises=(_event(12, z=2.5),))
    records = sb.build_rewrite_prompt_records([req])
    hits = lint_text(records[0]["prompt"], FORBIDDEN)
    assert hits and any(PLANTED_TOKEN in v.token for v in hits)


def test_rewrite_prompt_carries_surprise_and_immutable_block():
    req = _request(_card(), surprises=(_event(12, key="car|corridor|am_peak",
                                              exp=30.0, real=55.0, z=2.5),),
                   strong=("r1",))
    prompt = sb.build_rewrite_prompt_records([req])[0]["prompt"]
    assert "car|corridor|am_peak" in prompt
    assert "remove rules: r1" in prompt  # the immutable-rules list is rendered
    assert "30.0 min" in prompt and "55.0 min" in prompt


# ---------------------------------------------------------------------------
# StubGenerator determinism
# ---------------------------------------------------------------------------

def test_stub_generator_is_deterministic():
    req = _request(_card(), surprises=(_event(12, key="car|corridor|am_peak", z=2.5),))
    gen = sb.StubGenerator()
    a = gen([req])
    b = gen([req])
    assert a == b
    obj = json.loads(a[0])
    # patterns unchanged, one new band-shift rule keyed off the surprise context
    assert obj["patterns"] == _card()["patterns"]
    assert obj["rules"][-1] == {"id": "shift_am_peak",
                                "when": {"depart_band": "am_peak"},
                                "then": {"depart_band": "midday"}}


def test_stub_generator_output_passes_full_gate():
    req = _request(_card(), surprises=(_event(12, z=2.5),), strong=("r1",))
    obj = json.loads(sb.StubGenerator()([req])[0])
    errs = cv.validate_card(obj, SKELETON, _observed(), _seqs())
    assert errs == []  # schema, lint, replay, feasibility, fidelity all clean


def test_batch_preserves_request_order():
    cards = [_card(), _card(), _card()]
    reqs = [RewriteRequest(f"P0000{i}", 12, c, (_event(12, z=2.5),), ())
            for i, c in enumerate(cards)]
    # give each persona its own validation context
    vc = {f"P0000{i}": {"skeleton": SKELETON, "observed": _observed(),
                        "observed_day_sequences": _seqs()} for i in range(3)}
    gsb = sb.GatedSlowBrain(sb.StubGenerator(), vc)
    outs = gsb.rewrite_batch(reqs)
    assert [o.persona_id for o in outs] == ["P00000", "P00001", "P00002"]
    assert all(o.accepted for o in outs)
