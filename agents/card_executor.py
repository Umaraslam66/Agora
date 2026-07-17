"""The fast brain: execute a persona card into realized weekday choices (D2).

Plain deterministic code, no LLM in the loop. Per (persona, simulated day):

1. Draw one day pattern by its integer weights through the CRN layer under the
   site key ``"{namespace}:{persona_id}:{day_index}:pattern"``.
2. For each trip in the drawn pattern, apply the FIRST matching rule (rules are
   ordered) to override mode/depart_band; otherwise the pattern trip stands.
3. Availability gating (a hard physical constraint cards cannot break): a car
   trip becomes a ride when the household owns no vehicle or the person is not
   licensed; each coercion is logged.
4. Habit counters record follow/not-follow per pattern and per rule per lived
   day, so E6's hysteresis machinery has real data from M2 on.

Determinism: same (cards, day slots, namespace) -> bit-identical realized trips.
Two paired arms/twin-worlds that build the same CRN key reuse the same uniform,
so the difference between arms is a paired comparison (the E1/E5 pairing power).

Masking discipline: no real place name, agency, date, or bare wave-year appears
in any literal or comment here (mask-lint gate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from agents.habit_memory import HabitCounter
from world.crn import draw as crn_draw
from world.crn import pick_weighted

_COERCED_TO = "ride"  # the fallback mode when a car trip is physically infeasible


@dataclass(frozen=True)
class BorrowedCarAccess:
    """The sealed pre-M4 borrowed-car availability draw
    (`docs/DECISION_M4_BORROWED_CAR_GATE.md`, Option B, sealed 2026-07-15).

    Licensed adults in zero-vehicle households receive a per-day availability
    draw through the CRN layer under the sealed fresh site key
    ``"{namespace}:{persona_id}:{day_index}:caraccess"``. When the day's draw
    grants access, the card's car trips execute as car; otherwise they coerce
    to ride exactly as before. The habit substrate sees a borrowed-car day as
    an ordinary lived day — no special casing.

    ``rate`` is the availability rate fitted ONCE on the calibration window
    and FROZEN in `runs/m4_prep/borrowed_car/manifest.json`
    (`calibration.m4_gates_fit`); ``qualifying`` is the frozen persona-id set
    whose OWN seeding record shows car-driver trips in a zero-vehicle
    household — no invented availability for persons who never showed it.
    Because the site key is fresh, every pre-existing CRN stream stays
    bit-identical, and twin-world pairing is preserved by construction.
    """

    rate: float
    qualifying: frozenset

    def grants(self, persona_id: str, day_index: int, namespace: str) -> bool:
        """Does this persona's day draw grant borrowed-car access?"""
        if persona_id not in self.qualifying:
            return False
        key = f"{namespace}:{persona_id}:{day_index}:caraccess"
        return crn_draw(key) < self.rate


def _borrowed_car_eligible(skeleton: Mapping) -> bool:
    """Mechanical belt on the sealed class: the draw applies ONLY to licensed
    adults in zero-vehicle households (the qualifying set is the fitted
    restriction on top of this)."""
    return (
        skeleton.get("household_cars") == 0
        and bool(skeleton.get("can_drive", False))
        and (skeleton.get("age") or 0) >= 18
    )


@dataclass
class RealizedTrip:
    """One realized weekday trip. ``rule_applied`` is the id of the rule that
    overrode this trip, or None when the pattern trip stood unchanged."""

    purpose: str
    mode: str
    depart_band: str
    rule_applied: Optional[str] = None


@dataclass
class RealizedDay:
    """One simulated day for one persona: the day index, the E1 slot weight the
    day inherits, and its realized trips."""

    day_index: int
    day_weight: float
    trips: List[RealizedTrip] = field(default_factory=list)


def _car_allowed(skeleton: Mapping) -> bool:
    cars = skeleton.get("household_cars")
    can_drive = skeleton.get("can_drive", True)
    return (cars is None or cars >= 1) and bool(can_drive)


def _match_rule(rule: Mapping, purpose: str, band: str) -> bool:
    when = rule.get("when", {})
    if "purpose" in when and when["purpose"] != purpose:
        return False
    if "depart_band" in when and when["depart_band"] != band:
        return False
    return bool(when)  # an empty `when` never matches (schema requires >=1 anyway)


def _execute_day_detail(
    card: Mapping,
    day_index: int,
    namespace: str,
    coercion_log: Optional[List[dict]] = None,
    car_access: Optional[BorrowedCarAccess] = None,
) -> Tuple[Optional[str], set, List[RealizedTrip]]:
    """Return (drawn_pattern_id, applied_rule_ids, realized_trips) for one day."""
    persona_id = card["persona_id"]
    patterns = card.get("patterns", [])
    skeleton = card.get("skeleton", {})
    rules = card.get("rules", [])

    if not patterns:
        return None, set(), []

    weights = [p["weight"] for p in patterns]
    key = f"{namespace}:{persona_id}:{day_index}:pattern"
    pattern = pick_weighted(key, patterns, weights)
    pattern_id = pattern.get("id")

    car_ok = _car_allowed(skeleton)
    if (
        not car_ok
        and car_access is not None
        and _borrowed_car_eligible(skeleton)
        and car_access.grants(persona_id, day_index, namespace)
    ):
        car_ok = True
    applied_rule_ids, realized = _realize_pattern_trips(
        persona_id, pattern, rules, car_ok, day_index, coercion_log
    )
    return pattern_id, applied_rule_ids, realized


def _realize_pattern_trips(
    persona_id: str,
    pattern: Mapping,
    rules: Sequence[Mapping],
    car_ok: bool,
    day_index: Optional[int] = None,
    coercion_log: Optional[List[dict]] = None,
) -> Tuple[set, List[RealizedTrip]]:
    """Realize one pattern's trips: first-matching rule override, then the
    availability coercion. The single source of the per-trip execution
    semantics — used by the day executor and by the calibration-side
    expectation (:func:`expected_mode_counts`)."""
    applied_rule_ids: set = set()
    realized: List[RealizedTrip] = []

    for ti, trip in enumerate(pattern.get("trips", [])):
        purpose = trip["purpose"]
        mode = trip["mode"]
        band = trip["depart_band"]
        rule_applied: Optional[str] = None

        for rule in rules:  # ordered; first match wins
            if _match_rule(rule, purpose, band):
                then = rule.get("then", {})
                if "mode" in then:
                    mode = then["mode"]
                if "depart_band" in then:
                    band = then["depart_band"]
                rule_applied = rule.get("id")
                applied_rule_ids.add(rule_applied)
                break

        if mode == "car" and not car_ok:
            mode = _COERCED_TO
            if coercion_log is not None:
                coercion_log.append(
                    {"persona_id": persona_id, "day_index": day_index, "trip_index": ti}
                )

        realized.append(RealizedTrip(purpose, mode, band, rule_applied))

    return applied_rule_ids, realized


def expected_mode_counts(card: Mapping, car_ok: bool) -> Dict[str, float]:
    """Pattern-weight expected per-day realized mode counts for one card under
    a FIXED availability state (no CRN): sum over patterns of
    (weight / total_weight) x that pattern's realized trip modes, rules applied
    exactly as the executor applies them. Harness-side helper for the sealed
    borrowed-car availability fit (`calibration.m4_gates_fit`), where the
    per-day ``caraccess`` draw makes the realized day a mixture of the
    ``car_ok=True`` and ``car_ok=False`` expectations."""
    patterns = card.get("patterns", [])
    rules = card.get("rules", [])
    total = float(sum(p["weight"] for p in patterns)) if patterns else 0.0
    counts: Dict[str, float] = {}
    if total <= 0.0:
        return counts
    for pattern in patterns:
        share = float(pattern["weight"]) / total
        _ids, realized = _realize_pattern_trips(
            str(card.get("persona_id")), pattern, rules, car_ok
        )
        for t in realized:
            counts[t.mode] = counts.get(t.mode, 0.0) + share
    return counts


def execute_day(
    card: Mapping,
    day_index: int,
    namespace: str,
    coercion_log: Optional[List[dict]] = None,
    car_access: Optional[BorrowedCarAccess] = None,
) -> List[RealizedTrip]:
    """Execute one simulated weekday for one persona card. Pure and
    deterministic (no card mutation). Pass ``coercion_log`` to collect
    availability coercions."""
    _pattern_id, _rules, trips = _execute_day_detail(
        card, day_index, namespace, coercion_log, car_access
    )
    return trips


def _record_habits(card: dict, pattern_id: Optional[str], applied_rule_ids: set) -> None:
    """Fold one lived day into the card's HabitCounters: the drawn pattern is
    followed, every other pattern is not; each rule is followed iff it fired."""
    counters = card.setdefault("habit_counters", {})
    pattern_ids = [p.get("id") for p in card.get("patterns", [])]
    rule_ids = [r.get("id") for r in card.get("rules", [])]
    for pid in pattern_ids:
        counter = _counter_for(counters, pid)
        counter.record_day(pid == pattern_id)
        counters[pid] = counter.to_dict()
    for rid in rule_ids:
        counter = _counter_for(counters, rid)
        counter.record_day(rid in applied_rule_ids)
        counters[rid] = counter.to_dict()


def _counter_for(counters: Mapping, key: str) -> HabitCounter:
    if key in counters:
        return HabitCounter.from_dict(counters[key])
    return HabitCounter()


def execute_days(
    cards: Sequence[dict],
    day_slots: Mapping[str, Sequence[Tuple[int, float]]],
    namespace: str,
    update_habits: bool = True,
    coercion_log: Optional[List[dict]] = None,
    car_access: Optional[BorrowedCarAccess] = None,
) -> Dict[str, List[RealizedDay]]:
    """Batch execution shaped for the E1 harness.

    ``day_slots`` maps persona_id -> list of ``(day_index, day_weight)`` — one
    simulated day per observed weighted weekday person-day slot (D5); the
    simulated day inherits the slot's weight. Returns persona_id -> list of
    RealizedDay. When ``update_habits`` is set, each card's HabitCounters are
    advanced per lived day and serialized back into the card dict in place.
    """
    out: Dict[str, List[RealizedDay]] = {}
    for card in cards:
        persona_id = card["persona_id"]
        slots = day_slots.get(persona_id, [])
        days: List[RealizedDay] = []
        for day_index, day_weight in slots:
            pattern_id, applied_rule_ids, trips = _execute_day_detail(
                card, day_index, namespace, coercion_log, car_access
            )
            if update_habits:
                _record_habits(card, pattern_id, applied_rule_ids)
            days.append(RealizedDay(int(day_index), float(day_weight), trips))
        out[persona_id] = days
    return out
