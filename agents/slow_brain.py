"""The slow brain: surprise-triggered persona-card rewrites (M3 D3/D4).

The fast brain (agents/card_executor.py) lives a persona's ordinary weekdays
deterministically; when realized travel times drift far enough from what the
persona had come to expect, the numeric substrate (agents/habit_memory.py)
flags a surprise. This module is the OTHER half of the two-brain loop: it books
those surprises against the card, decides when the drift is sustained or sharp
enough to warrant a rewrite, and — through the SAME five-gate compose and
mask-lint that governed generation — asks an LLM to revise the card while the
persona's lived habits stay intact.

Why a separate file, and why so much mechanism around one LLM call:

* **The gate is the whole point.** A rewrite is accepted only if it passes the
  identical ``validate_card`` five gates as generation (schema, mask-lint,
  replay-smell, feasibility, fidelity) PLUS a mechanical strong-habit
  immutability check. Doctrine (M3 D4): strong habits resist, and resistance
  must be mechanical, not rhetorical. On terminal failure the OLD card stands —
  it is always valid — and the surprise entries stay open.
* **Determinism and replay.** Every decision here is a pure function of the
  card's own state (its surprise log and provenance) plus the observations fed
  in; there is no hidden clock and no randomness. The generator is a seam
  (``Callable[[Sequence[RewriteRequest]], Sequence[str]]``) so tests and cluster
  rehearsals drive a deterministic stub and the scored path drives the offline
  batch driver.
* **Masking discipline.** Nothing agent-facing here may leak the real arena:
  the rewrite prompt is built ONLY through grounding.render (render-parity), the
  surprise block carries masked zone/mode codes and minutes, and the card view
  handed back to the model strips every harness field (the M2 lesson where log
  and provenance fields tripped the lints).

The frozen interface (SurpriseEvent, RewriteRequest, RewriteOutcome,
SlowBrainClient, SurprisePolicy) lives in agents/two_brain.py and is never
modified here.
"""
from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from agents.habit_memory import SURPRISE_LOG_CAP, HabitCounter
from agents.two_brain import (
    SURPRISE_LOG_KEY,
    RewriteOutcome,
    RewriteRequest,
    SurpriseEvent,
)
from grounding.card_validation import CARD_VERSION, validate_card
from grounding.render import (
    build_rewrite_prompt_records,
    render_rewrite_prompt,
    rewrite_render_inputs,
)
from serving.batch_gen import prompt_sha256

# ``build_rewrite_prompt_records`` is DEFINED in grounding.render (prompt
# construction is grounding's alone — the render-parity AST guard forbids any
# ``build*prompt`` def under agents/) and re-exported here so the loop imports
# it from the slow brain as the brief's "slow_brain helper".
__all__ = [
    "WARMUP_DAYS",
    "REWRITE_COOLDOWN_DAYS",
    "SUSTAINED_SURPRISE_K",
    "SUSTAINED_SURPRISE_WINDOW",
    "SHOCK_TRIGGER_Z",
    "STRONG_HABIT_THRESHOLD",
    "MAX_REWRITE_ATTEMPTS",
    "StandardSurprisePolicy",
    "GatedSlowBrain",
    "StubGenerator",
    "apply_rewrite",
    "restore_strong_rules",
    "strong_rule_ids_for",
    "build_rewrite_prompt_records",
    "render_rewrite_prompt",
    "rewrite_render_inputs",
]

# ---------------------------------------------------------------------------
# Committed trigger-policy constants (M3 D3). These are build decisions with
# rationale — NOT sealed bars; the E1/E2 verdicts stay governed by §7 A2.
# ---------------------------------------------------------------------------

#: Lived days before triggers arm. Observations are still recorded during
#: warm-up; the persona has lived their city before day 0, so warm-up
#: initializes expectations to the ordinary state and also kills the
#: cold-start surprise storm a freshly-seeded EMA would otherwise manufacture.
WARMUP_DAYS = 10

#: Lived days a persona is exempt from a fresh trigger after an ACCEPTED
#: rewrite. A just-rewritten card needs time to re-converge before its own
#: settling is misread as new surprise; tracked via provenance rewrite records.
REWRITE_COOLDOWN_DAYS = 7

#: Sustained-surprise trigger: the SAME context key must carry an open surprise
#: on at least this many of the trailing ``SUSTAINED_SURPRISE_WINDOW`` lived
#: days. A persistent bias in one context (not one bad draw) warrants a rewrite.
SUSTAINED_SURPRISE_K = 3

#: Trailing lived-day window the sustained-surprise count reads.
SUSTAINED_SURPRISE_WINDOW = 5

#: Shock trigger: a single open surprise at |z| >= this fires immediately
#: (2.0 z = 20 minutes at the ported sigma=10) — a sharp one-off shock does not
#: need to sustain to justify reconsidering the card.
SHOCK_TRIGGER_Z = 2.0

#: A rule whose habit-strength counter has reached this many net days-followed
#: is a strong habit: it is stated immutable in the prompt AND enforced
#: mechanically (a rewrite that drops or alters it is rejected).
STRONG_HABIT_THRESHOLD = 14

#: Attempt budget per rewrite, mirroring generation: one first attempt plus one
#: retry carrying the numeric gate feedback. On terminal failure the old card
#: stands (there is NO fallback builder for rewrites — M3 D4).
MAX_REWRITE_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------

def _canonical(item: Mapping) -> str:
    """Canonical JSON of a pattern/rule dict, for content-identity compares."""
    return json.dumps(item, sort_keys=True)


def strong_rule_ids_for(card: Mapping, threshold: int = STRONG_HABIT_THRESHOLD) -> Tuple[str, ...]:
    """The ids of rules whose habit-strength counter is >= ``threshold`` — the
    strong (immutable) rules the loop stamps onto each RewriteRequest. A helper
    for the loop builder; the slow brain itself only enforces the ids it is
    handed on the request."""
    counters = card.get("habit_counters", {}) or {}
    out: List[str] = []
    for rule in card.get("rules", []):
        rid = rule.get("id")
        cd = counters.get(rid)
        if cd is not None and HabitCounter.from_dict(cd).is_strong(threshold):
            out.append(rid)
    return tuple(out)


def _strong_rule_violations(
    new_obj: Mapping, old_card: Mapping, strong_rule_ids: Sequence[str]
) -> List[str]:
    """Mechanical strong-habit immutability check: which immutable rule ids
    are missing from, or content-changed in, the new object (canonical-JSON
    comparison of the rule dict). Returns masked-clean strings — used for
    AUDIT (restoration counts), not rejection; see :func:`restore_strong_rules`
    and the dated D4 revision note in the M3 design record."""
    old_rules = {r.get("id"): _canonical(r) for r in old_card.get("rules", [])}
    new_rules = {r.get("id"): _canonical(r) for r in new_obj.get("rules", [])}
    errors: List[str] = []
    for rid in strong_rule_ids:
        if rid not in new_rules:
            errors.append(
                f"strong rule {rid} was dropped; an established habit may not be removed"
            )
        elif new_rules[rid] != old_rules.get(rid):
            errors.append(
                f"strong rule {rid} was altered; an established habit may not be changed"
            )
    return errors


def restore_strong_rules(
    new_obj: Mapping, old_card: Mapping, strong_rule_ids: Sequence[str]
) -> Tuple[dict, List[str]]:
    """Mechanically restore immutable strong rules into a proposed rewrite.

    Dated D4 revision (2026-07-15, M3 rehearsal evidence): the original
    contract REJECTED a rewrite that dropped or altered a strong rule. The
    real-model rehearsal showed the model deterministically simplifies rule
    conditions while keeping ids (65/95 first-pass rejections, unrecovered by
    retry feedback) — byte-copying JSON through an LLM is the wrong mechanism
    for immutability. Doctrine already said it: resistance must be MECHANICAL,
    not rhetorical. So the machinery restores instead of rejecting: the model
    proposes, the gate disposes.

    Restoration contract: every strong rule re-enters with its ORIGINAL
    content, in its ORIGINAL relative order, placed AHEAD of every proposed
    non-strong rule — first-match-wins means nothing the rewrite adds may
    preempt an established habit (the shadow guard). The model's proposed
    content for a strong rule id is discarded. Returns the repaired object
    and the audit list of what was restored (empty when the model preserved
    everything verbatim — then the object passes through order-preserved,
    byte-unchanged).
    """
    audit = _strong_rule_violations(new_obj, old_card, strong_rule_ids)
    if not audit:
        return dict(new_obj), []
    strong_set = set(strong_rule_ids)
    originals = [r for r in old_card.get("rules", []) if r.get("id") in strong_set]
    proposed = [
        r for r in new_obj.get("rules", []) if r.get("id") not in strong_set
    ]
    repaired = dict(new_obj)
    repaired["rules"] = [copy.deepcopy(r) for r in originals] + [dict(r) for r in proposed]
    return repaired, audit


# ---------------------------------------------------------------------------
# card mutation on acceptance / rejection (M3 D4)
# ---------------------------------------------------------------------------

def apply_rewrite(
    old_card: Mapping,
    new_obj: Mapping,
    day_index: int,
    attempt: int,
    model: str,
    prompt_sha: str,
) -> Dict:
    """Assemble the NEW card from an accepted rewrite, preserving lived habits.

    Habit-counter continuity (M3 D4): a pattern/rule that is unchanged by id AND
    content keeps its counter; a changed or new id gets a fresh counter (a
    rewritten rule is a new habit — HabitCounter.reset() semantics = an
    absent/zeroed entry); a deleted id drops. Surprise-log entries backing the
    trigger flip to ``resolved`` (they drop before open entries at the next
    append). Provenance appends a ``rewrites`` record with ``accepted: true``;
    ``card_source`` NEVER changes (population-as-deployed classification is
    stable — fallback cards are rewrite-eligible too).
    """
    new_patterns = [dict(p) for p in new_obj.get("patterns", [])]
    new_rules = [dict(r) for r in new_obj.get("rules", [])]
    old_counters = old_card.get("habit_counters", {}) or {}

    old_content: Dict[str, str] = {}
    for item in list(old_card.get("patterns", [])) + list(old_card.get("rules", [])):
        old_content[item.get("id")] = _canonical(item)

    new_counters: Dict[str, dict] = {}
    for item in new_patterns + new_rules:
        iid = item.get("id")
        if (
            iid in old_content
            and old_content[iid] == _canonical(item)
            and iid in old_counters
        ):
            new_counters[iid] = copy.deepcopy(old_counters[iid])  # unchanged -> keep
        else:
            new_counters[iid] = HabitCounter().to_dict()          # new/changed -> reset

    new_log = []
    for entry in old_card.get(SURPRISE_LOG_KEY, []) or []:
        resolved = dict(entry)
        if resolved.get("status") == "open":
            resolved["status"] = "resolved"
        new_log.append(resolved)

    provenance = copy.deepcopy(old_card.get("provenance", {}) or {})
    rewrites = list(provenance.get("rewrites", []))
    rewrites.append(_rewrite_record(day_index, attempt, model, prompt_sha, True))
    provenance["rewrites"] = rewrites

    return {
        "card_version": old_card.get("card_version", CARD_VERSION),
        "persona_id": old_card.get("persona_id"),
        "skeleton": copy.deepcopy(old_card.get("skeleton", {})),
        "patterns": new_patterns,
        "rules": new_rules,
        "voice": new_obj.get("voice", ""),
        SURPRISE_LOG_KEY: new_log,
        "habit_counters": new_counters,
        "provenance": provenance,
    }


def _append_rejection(
    old_card: Mapping, day_index: int, attempt: int, model: str, prompt_sha: Optional[str]
) -> Dict:
    """A terminally-failed rewrite: the old card is returned UNCHANGED except
    for an audit-trail ``rewrites`` record with ``accepted: false`` (counters,
    surprise log, patterns/rules/voice all left intact)."""
    new_card = copy.deepcopy(dict(old_card))
    provenance = new_card.setdefault("provenance", {})
    rewrites = list(provenance.get("rewrites", []))
    rewrites.append(_rewrite_record(day_index, attempt, model, prompt_sha, False))
    provenance["rewrites"] = rewrites
    return new_card


def _rewrite_record(
    day_index: int, attempt: int, model: str, prompt_sha: Optional[str], accepted: bool
) -> dict:
    return {
        "day_index": int(day_index),
        "attempt": int(attempt),
        "model": model,
        "prompt_sha": prompt_sha,
        "accepted": accepted,
    }


# ---------------------------------------------------------------------------
# surprise bookkeeping + trigger decision (SurprisePolicy protocol, M3 D3)
# ---------------------------------------------------------------------------

class StandardSurprisePolicy:
    """Card-side surprise log + rewrite-trigger decision.

    The loop calls only :meth:`log_surprise` (per surprising observation) and
    :meth:`should_rewrite` (once at day end); it never inspects the log format.
    All thresholds default to the committed module constants.
    """

    def __init__(
        self,
        warmup_days: int = WARMUP_DAYS,
        cooldown_days: int = REWRITE_COOLDOWN_DAYS,
        log_cap: int = SURPRISE_LOG_CAP,
        sustained_k: int = SUSTAINED_SURPRISE_K,
        sustained_window: int = SUSTAINED_SURPRISE_WINDOW,
        shock_z: float = SHOCK_TRIGGER_Z,
    ) -> None:
        self.warmup_days = int(warmup_days)
        self.cooldown_days = int(cooldown_days)
        self.log_cap = int(log_cap)
        self.sustained_k = int(sustained_k)
        self.sustained_window = int(sustained_window)
        self.shock_z = float(shock_z)

    def log_surprise(self, card: Dict, event: SurpriseEvent) -> None:
        """Append one surprise (status ``open``) to the card's log, enforcing
        the cap. When the cap forces an eviction, the oldest RESOLVED entry is
        dropped before any open one (a resolved surprise has served its
        purpose); with no resolved entry, the oldest entry drops."""
        log = card.setdefault(SURPRISE_LOG_KEY, [])
        log.append(
            {
                "day_index": int(event.day_index),
                "context_key": event.context_key,
                "expected_minutes": float(event.expected_minutes),
                "realized_minutes": float(event.realized_minutes),
                "z": float(event.z),
                "status": "open",
            }
        )
        while len(log) > self.log_cap:
            evict = next(
                (i for i, e in enumerate(log) if e.get("status") == "resolved"), 0
            )
            log.pop(evict)

    def should_rewrite(self, card: Mapping, day_index: int) -> bool:
        """True iff the trigger fires at the end of ``day_index``: past warm-up,
        not in post-accept cooldown, and either a sustained-surprise run or a
        single shock among the OPEN log entries."""
        if day_index < self.warmup_days:
            return False
        last_accepted = self._last_accepted_day(card)
        if last_accepted is not None and day_index - last_accepted < self.cooldown_days:
            return False

        open_entries = [
            e for e in card.get(SURPRISE_LOG_KEY, []) or [] if e.get("status") == "open"
        ]
        if not open_entries:
            return False

        if any(abs(float(e.get("z", 0.0))) >= self.shock_z for e in open_entries):
            return True

        window_lo = day_index - self.sustained_window + 1
        days_by_key: Dict[str, set] = {}
        for e in open_entries:
            di = int(e.get("day_index", -1))
            if window_lo <= di <= day_index:
                days_by_key.setdefault(e.get("context_key"), set()).add(di)
        return any(len(days) >= self.sustained_k for days in days_by_key.values())

    @staticmethod
    def _last_accepted_day(card: Mapping) -> Optional[int]:
        provenance = card.get("provenance", {}) or {}
        days = [
            int(r.get("day_index"))
            for r in provenance.get("rewrites", [])
            if r.get("accepted") and r.get("day_index") is not None
        ]
        return max(days) if days else None


# ---------------------------------------------------------------------------
# the gated LLM client (SlowBrainClient protocol, M3 D4)
# ---------------------------------------------------------------------------

class GatedSlowBrain:
    """Batch rewrite client: render -> generate -> gate -> (retry) -> apply.

    ``generator`` is the model seam ``Callable[[Sequence[RewriteRequest]],
    Sequence[str]]`` returning one raw JSON text per request (a deterministic
    stub in tests/rehearsals; a wrapper over the offline cluster output in the
    scored path). ``validation_context`` maps persona_id -> {"skeleton",
    "observed", "observed_day_sequences"} — exactly what ``validate_card``
    needs. ``render_context`` (optional) supplies each persona's seeding
    evidence lines + observed FIT CHECK figures for the rewrite prompt.

    Per request: render the prompt through grounding.render.render_rewrite_prompt
    ONLY, call the generator, parse JSON, run the five ``validate_card`` gates
    PLUS strong-rule immutability. On failure, retry once (attempt 2) with the
    gate failure strings appended; on terminal failure the old card stands
    (accepted=False) with a rejected provenance record. Never raises on a gate
    failure — that is an outcome, not an error.
    """

    def __init__(
        self,
        generator,
        validation_context: Mapping[str, Mapping],
        render_context: Optional[Mapping[str, Mapping]] = None,
        model: str = "stub",
        max_attempts: int = MAX_REWRITE_ATTEMPTS,
        mode: str = "serve",
    ) -> None:
        self.generator = generator
        self.validation_context = validation_context
        self.render_context = render_context
        self.model = model
        self.max_attempts = int(max_attempts)
        self.mode = mode

    def rewrite_batch(self, requests: Sequence[RewriteRequest]) -> List[RewriteOutcome]:
        outcomes: Dict[int, RewriteOutcome] = {}
        pending: List[Tuple[int, RewriteRequest]] = list(enumerate(requests))
        failures: Dict[int, Tuple[str, ...]] = {i: () for i, _ in pending}
        last_sha: Dict[int, Optional[str]] = {i: None for i, _ in pending}

        attempt = 1
        while pending and attempt <= self.max_attempts:
            attempt_reqs = [replace(req, attempt=attempt) for _, req in pending]
            prompts = [
                self._attempt_text(req, failures[idx])
                for (idx, _), req in zip(pending, attempt_reqs)
            ]
            raws = list(self._call_generator(attempt_reqs, prompts))

            next_pending: List[Tuple[int, RewriteRequest]] = []
            for (idx, orig_req), prompt, raw in zip(pending, prompts, raws):
                sha = prompt_sha256(prompt)
                last_sha[idx] = sha
                obj, errs, _restored = self._gate(orig_req, raw)
                if not errs:
                    new_card = apply_rewrite(
                        orig_req.card, obj, orig_req.day_index, attempt, self.model, sha
                    )
                    outcomes[idx] = RewriteOutcome(
                        persona_id=orig_req.persona_id,
                        day_index=orig_req.day_index,
                        accepted=True,
                        card=new_card,
                        attempts_used=attempt,
                        gate_failures=(),
                    )
                else:
                    failures[idx] = tuple(errs)
                    next_pending.append((idx, orig_req))
            pending = next_pending
            attempt += 1

        for idx, orig_req in pending:
            new_card = _append_rejection(
                orig_req.card, orig_req.day_index, self.max_attempts, self.model, last_sha[idx]
            )
            outcomes[idx] = RewriteOutcome(
                persona_id=orig_req.persona_id,
                day_index=orig_req.day_index,
                accepted=False,
                card=new_card,
                attempts_used=self.max_attempts,
                gate_failures=failures[idx],
            )

        return [outcomes[i] for i in range(len(requests))]

    # -- internals ------------------------------------------------------------

    def _call_generator(self, requests: Sequence[RewriteRequest], prompts: Sequence[str]):
        """Invoke the generator seam. A generator that can take a second
        positional argument receives the EXACT rendered prompt per request
        (retry feedback included) — required by any real model client, whose
        output must be attributable to the byte-exact prompt it saw. A
        one-argument generator (the stub, simple test lambdas) gets requests
        only."""
        try:
            params = list(inspect.signature(self.generator).parameters.values())
        except (TypeError, ValueError):
            return self.generator(requests)
        positional = [
            p
            for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        takes_var = any(p.kind == p.VAR_POSITIONAL for p in params)
        if takes_var or len(positional) >= 2:
            return self.generator(requests, prompts)
        return self.generator(requests)

    def _attempt_text(self, request: RewriteRequest, failure_reasons: Sequence[str]) -> str:
        """Render one attempt's rewrite prompt (grounding.render only)."""
        ctx = self.validation_context.get(request.persona_id, {})
        inputs = rewrite_render_inputs(
            request, self.render_context, observed=ctx.get("observed")
        )
        return render_rewrite_prompt(mode=self.mode, failure_reasons=failure_reasons, **inputs)

    def _gate(self, request: RewriteRequest, raw) -> Tuple[Optional[dict], List[str], List[str]]:
        """Parse -> strong-rule restoration -> five gates. Returns
        (repaired_obj, errors, restorations); obj is None only when the raw
        text was not a JSON object. Strong-rule drift is REPAIRED (restored
        verbatim, shadow-guarded), never a rejection; the five validate_card
        gates run on the repaired object and remain the only rejectors."""
        try:
            obj = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError, ValueError):
            return None, ["rewrite output was not valid JSON"], []
        if not isinstance(obj, dict):
            return None, ["rewrite output was not a JSON object"], []

        ctx = self.validation_context.get(request.persona_id)
        if ctx is None:
            return obj, ["no validation context for this persona; cannot gate the rewrite"], []

        obj, restorations = restore_strong_rules(obj, request.card, request.strong_rule_ids)
        errors = validate_card(
            obj, ctx["skeleton"], ctx["observed"], ctx["observed_day_sequences"]
        )
        return obj, errors, restorations


# ---------------------------------------------------------------------------
# deterministic stub generator (tests + the loop's injected-jam integration)
# ---------------------------------------------------------------------------

_BANDS = ("night", "am_peak", "midday", "pm_peak", "evening")

#: A fixed "shift to an alternative band" map: each departure band maps to one
#: deterministic neighbour, so the stub's rewrite is reproducible.
_ALT_BAND = {
    "night": "am_peak",
    "am_peak": "midday",
    "midday": "pm_peak",
    "pm_peak": "evening",
    "evening": "night",
}


class StubGenerator:
    """Deterministic, schema-valid rewrite generator built purely from request
    contents — no randomness, no model.

    For each request it returns the card's patterns/rules/voice UNCHANGED (so
    the rewrite still passes the fidelity gate the current card already passed,
    and every strong rule is preserved verbatim) plus ONE new override rule that
    shifts the first surprising context's departure band to an alternative. The
    unchanged patterns/rules keep their habit counters; the new rule gets a
    fresh one. Importable by other test files and the loop's injected-jam test.
    """

    def __call__(
        self, requests: Sequence[RewriteRequest], prompts: Optional[Sequence[str]] = None
    ) -> List[str]:
        """``prompts`` (the rendered per-request prompt texts) is accepted for
        seam parity with real generators and deliberately unused — the stub is
        a pure function of the requests."""
        return [json.dumps(self._rewrite_obj(req), sort_keys=True) for req in requests]

    def _rewrite_obj(self, request: RewriteRequest) -> dict:
        card = request.card
        patterns = copy.deepcopy(list(card.get("patterns", [])))
        rules = copy.deepcopy(list(card.get("rules", [])))
        voice = card.get("voice", "")

        if request.surprises and len(rules) < 6:
            band = self._band_of(request.surprises[0].context_key)
            existing = {r.get("id") for r in rules} | {p.get("id") for p in patterns}
            rid = self._fresh_rule_id(band, existing)
            rules.append(
                {
                    "id": rid,
                    "when": {"depart_band": band},
                    "then": {"depart_band": _ALT_BAND[band]},
                }
            )
        return {"patterns": patterns, "rules": rules, "voice": voice}

    @staticmethod
    def _band_of(context_key: str) -> str:
        """The departure band embedded in a ``mode|od_class|period`` key, or a
        safe default when the period is not a recognized band."""
        parts = str(context_key).split("|")
        candidate = parts[-1] if parts else ""
        return candidate if candidate in _BANDS else "am_peak"

    @staticmethod
    def _fresh_rule_id(band: str, existing: set) -> str:
        base = f"shift_{band}"
        if base not in existing:
            return base
        suffix = 2
        while f"{base}{suffix}" in existing:
            suffix += 1
        return f"{base}{suffix}"
