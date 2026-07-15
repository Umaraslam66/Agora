"""Two-brain interface contract (M3): the seam between the fast-brain
ordinary-day loop and the slow-brain rewrite machinery.

This module holds ONLY frozen dataclasses and protocols — no logic, no
imports from the modules that implement them. The baseline loop (world/
agents) and the slow brain (agents/slow_brain.py) both depend on this
file and never on each other's internals; tests substitute stub
implementations of the protocols.

Masking discipline: everything carried here is masked-clean by
construction — zone codes, mode/purpose vocabulary, minutes, z-scores.
No real place name, agency, date, or bare wave-year may enter any field
(the card-level mask-lint gate re-checks every string on acceptance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Protocol, Sequence, Tuple

#: Card-level surprise-log key. The log is capped at
#: agents.habit_memory.SURPRISE_LOG_CAP entries (= 5, per the brief);
#: the cap is enforced by the slow brain's policy implementation.
SURPRISE_LOG_KEY = "surprise_log"


@dataclass(frozen=True)
class SurpriseEvent:
    """One prediction-error observation that crossed the surprise
    threshold, in masked units (minutes)."""

    persona_id: str
    day_index: int
    context_key: str  # "{mode}|{od_class}|{period}", e.g. "car|corridor|am_peak"
    expected_minutes: float
    realized_minutes: float
    z: float


@dataclass(frozen=True)
class RewriteRequest:
    """A persona whose surprise policy fired at the end of ``day_index``.
    The rewrite is applied before day ``day_index + 1`` begins."""

    persona_id: str
    day_index: int
    card: Mapping  # current assembled card (counters, log, provenance intact)
    surprises: Tuple[SurpriseEvent, ...]  # entries backing the trigger, <= cap
    strong_rule_ids: Tuple[str, ...]  # immutable under this rewrite
    attempt: int = 1  # 1-based; retries increment


@dataclass(frozen=True)
class RewriteOutcome:
    """Result of one gated rewrite. ``accepted`` False means every attempt
    failed a gate and ``card`` is the UNCHANGED old card (always valid)."""

    persona_id: str
    day_index: int
    accepted: bool
    card: Dict  # new assembled card if accepted, else the old card
    attempts_used: int = 1
    gate_failures: Tuple[str, ...] = ()  # masked-clean failure strings


class SlowBrainClient(Protocol):
    """Batch rewrite seam. Implementations: the gated LLM client
    (agents/slow_brain.py, offline batch via serving/batch_gen) and the
    deterministic stub used by tests/rehearsals. Must return one outcome
    per request, order-preserving, and must never raise on a gate
    failure (that is an ``accepted=False`` outcome, not an error)."""

    def rewrite_batch(self, requests: Sequence[RewriteRequest]) -> List[RewriteOutcome]: ...


class SurprisePolicy(Protocol):
    """Card-side surprise bookkeeping + trigger decision. Implemented in
    agents/slow_brain.py (StandardSurprisePolicy); the loop only calls
    these two methods and never inspects the log format directly."""

    def log_surprise(self, card: Dict, event: SurpriseEvent) -> None:
        """Append to the card's surprise log (cap enforced, oldest
        dropped; resolved entries fold/drop per the brief)."""
        ...

    def should_rewrite(self, card: Mapping, day_index: int) -> bool:
        """True iff the trigger policy fires at the end of ``day_index``
        (sustained-surprise or shock rule, warm-up and cooldown
        respected)."""
        ...
