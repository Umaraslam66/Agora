"""Harness-side GOLDEN-format (prompt, chosen-mode) builder for the MNL arm.

The E1 falsification arm (``evaluation.mnl_arm``, M2 spec D6) fits and applies
``agents.logit_chooser`` — a multinomial logit whose feature channel is the
byte-exact GOLDEN prompt grammar ``logit_chooser.parse_prompt`` reads. This
module builds those GOLDEN strings from the SAME person-level evidence the
slow-brain LLM saw: the demographic skeleton, the person's usual (modal) mode,
and the person's diary mode-use counts. One builder produces BOTH the training
pairs (fit) and the serve-time choice prompt (apply), so the MNL arm has no
train/serve parity gap.

THIS IS NOT A RENDER PATH. It is a harness-only feature channel for the
falsification arm; it is never shown to an agent and is never routed through
``grounding.render`` (the single agent-facing render path). It deliberately
lives in ``evaluation/`` for exactly this reason. Because it is harness-only
and never agent-facing, it carries no mask-lint obligation; its only contract
is that ``logit_chooser.parse_prompt`` round-trips what it emits.

Feature note (documented so it is auditable): the persona cards carry NO
distances and NO clock times (masking), so every GOLDEN trip here is emitted
with a missing distance (the chooser's ``dmiss`` feature fires) and a masked
``??:??`` departure. The MNL's discriminating signal is therefore the
person-level evidence — usual mode + diary mode-use counts + demographics —
which is precisely the "deliberately dumb" baseline D6 asks for.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from agents import logit_chooser
from grounding.taxonomy import MODES

#: Frozen five-mode availability/tie-break order (== taxonomy MODES).
MODES_ALL = list(logit_chooser.MODES_ALL)

#: Reverse of the chooser's usual-mode phrase table ("You usually {phrase} to
#: work"). Every frozen mode has a phrase.
_MODE_TO_PHRASE: Dict[str, str] = {v: k for k, v in logit_chooser.PHRASE_TO_MODE.items()}

#: income_class {1..5} -> a monotone monthly-income proxy for the chooser's
#: ``inc`` feature (log(income/1800)); the value is harness-only and only its
#: ordering matters. 1800 is the chooser's own missing-income default.
_INCOME_PROXY: Dict[int, float] = {1: 1000.0, 2: 1800.0, 3: 2600.0, 4: 3600.0, 5: 6000.0}


def car_ok(skeleton: Mapping) -> bool:
    """Whether the person can physically make a car (driver) trip — the same
    availability gate the executor enforces (household owns a car AND licensed).
    """
    cars = skeleton.get("household_cars")
    can_drive = skeleton.get("can_drive", True)
    return (cars is None or cars >= 1) and bool(can_drive)


def feasible_modes(skeleton: Mapping) -> List[str]:
    """The person's available mode set in frozen order (car dropped when the
    person cannot drive one) — availability by exclusion, as the chooser expects.
    """
    ok = car_ok(skeleton)
    return [m for m in MODES_ALL if m != "car" or ok]


def _car_avail_label(skeleton: Mapping) -> str:
    """carAvail persona field: never / sometimes / always."""
    cars = skeleton.get("household_cars")
    can_drive = skeleton.get("can_drive", True)
    if not can_drive or cars == 0:
        return "never"
    if cars is None or cars == 1:
        return "sometimes"
    return "always"


def _modal_mode(modes: Sequence[str]) -> Optional[str]:
    """Modal mode with the frozen five-mode order as the deterministic tie-break;
    None for an empty sequence."""
    if not modes:
        return None
    counts = Counter(modes)
    best, best_n = None, -1
    for m in MODES:  # frozen order == tie-break
        n = counts.get(m, 0)
        if n > best_n:
            best_n = n
            best = m
    return best


@dataclass
class PersonEvidence:
    """The person-level evidence channel for one persona (constant across that
    persona's trips — the MNL feature vector is person-level, not trip-level)."""

    persona_id: str
    persona_fields: Dict[str, object]
    usual_mode: Optional[str]
    transit30: int
    ride30: int
    walk7: int
    bike7: int
    feasible: List[str]


def person_evidence(
    persona_id: str, skeleton: Mapping, trip_modes: Sequence[str], trip_purposes: Sequence[str]
) -> PersonEvidence:
    """Build one persona's GOLDEN evidence from its skeleton and diary trips.

    ``usual_mode`` is the modal mode over WORK trips (falling back to the
    overall modal mode); the mode-use counts are the raw diary counts of each
    mode (the GOLDEN grammar's "last month / last week" wording carries the
    signal; the values are diary counts, not literal 30-/7-day tallies).
    """
    work_modes = [m for m, p in zip(trip_modes, trip_purposes) if p == "work"]
    usual = _modal_mode(work_modes) or _modal_mode(list(trip_modes))
    counts = Counter(trip_modes)
    age = skeleton.get("age")
    fields = {
        "age": int(age) if age is not None else 40,
        "income": _INCOME_PROXY.get(skeleton.get("income_class"), 1800.0),
        "employed": "true" if skeleton.get("employed") else "false",
        "carAvail": _car_avail_label(skeleton),
    }
    return PersonEvidence(
        persona_id=str(persona_id),
        persona_fields=fields,
        usual_mode=usual,
        transit30=int(counts.get("transit", 0)),
        ride30=int(counts.get("ride", 0)),
        walk7=int(counts.get("walk", 0)),
        bike7=int(counts.get("bike", 0)),
        feasible=feasible_modes(skeleton),
    )


def build_prompt(ev: PersonEvidence, purpose: str = "work", prior_mode: Optional[str] = None) -> str:
    """Assemble one GOLDEN choice prompt. ``purpose`` and ``prior_mode`` are the
    only trip-level inputs; the chooser's feature map uses ``prior_mode`` (a
    habit term) but ignores the purpose token and the masked ``??:??`` depart,
    so the prompt round-trips a stable person-level feature vector."""
    persona_str = ", ".join(f"{k}={v}" for k, v in ev.persona_fields.items())
    trip = f"Next trip: from home to {purpose}, departing ??:??"
    if prior_mode:
        trip += f", currently via {prior_mode}"
    trip += ". "
    avail = "Available modes: " + ", ".join(ev.feasible) + "."
    exp: List[str] = []
    if ev.usual_mode is not None:
        exp.append(f"You usually {_MODE_TO_PHRASE[ev.usual_mode]} to work")
    exp.append(
        f"Last month you used public transport {ev.transit30} times "
        f"and ride services {ev.ride30} times"
    )
    exp.append(
        f"Last week you made {ev.walk7} walking and {ev.bike7} cycling trips"
    )
    tail = " Your experience: " + "; ".join(exp) + "."
    return f"Persona: {persona_str}. Today so far: usual routine. {trip}{avail}{tail}"


def training_record(ev: PersonEvidence, purpose: str, chosen_mode: str, split: str) -> Optional[dict]:
    """One JSONL (messages, assistant, meta) fit record for an observed trip,
    or ``None`` if the observed mode is not in the person's feasible set (rare;
    e.g. a coerced car trip)."""
    if chosen_mode not in ev.feasible:
        return None
    prompt = build_prompt(ev, purpose=purpose)
    return {
        "messages": [
            {"role": "system", "content": "choose a mode"},
            {"role": "user", "content": prompt},
        ],
        "assistant": chosen_mode,
        "meta": {"available_modes": list(ev.feasible), "split": split},
    }
