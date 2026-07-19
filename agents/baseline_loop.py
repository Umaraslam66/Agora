"""The baseline ordinary-day loop (M3 D5): the fast brain lived day-by-day,
coupled to the corridor world, with the surprise-triggered slow-brain seam.

WHY THIS FILE EXISTS — M2 executed each persona card once per observed slot and
scored the result; nothing lived through time or felt the world. M3's baseline
is the "ordinary day" dynamic loop that L3 left half-built: every persona lives
every global day, the fast brain realizes trips (habit counters LIVE), the
day's corridor commuters are pushed through ``solve_corridor_equilibrium`` +
the realized-facility CRN draw, the realized door-to-door time updates each
persona's :class:`~agents.habit_memory.HabitMemory`, and a bias-corrected
prediction error is the surprise signal. Post warm-up, personas whose injected
:class:`~agents.two_brain.SurprisePolicy` fires are handed to a
:class:`~agents.two_brain.SlowBrainClient` for a gated rewrite (or, with no
client, their requests are collected for the offline cluster).

The loop depends ONLY on the frozen ``agents.two_brain`` seam for the slow
brain — never on ``agents.slow_brain`` internals (that module is built in
parallel; the runner imports it lazily behind a CLI flag, so tests never need
it). Determinism is total: same (cards, day_slots, namespace) -> bit-identical
LoopResult, all randomness flowing through CRN keys, and a mid-run checkpoint
resumes bit-identically.

No real place name, agency, date, or price appears in any literal or comment
here (mask-lint gate); no prompt/persona is built here (render-parity gate).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.card_executor import BorrowedCarAccess, RealizedDay, RealizedTrip, execute_days
from agents.habit_memory import DEFAULT_CONFIG, SURPRISE_LOG_CAP, HabitMemory, SubstrateConfig
from agents.two_brain import (
    REASON_ANNOUNCED_ONSET,
    RewriteOutcome,
    RewriteRequest,
    SlowBrainClient,
    SurpriseEvent,
    SurprisePolicy,
)
from world import bridge, crn
from world.config import WorldConfig
from world.network import (
    NetworkState,
    facility_times_from_loads,
    realized_facilities,
    solve_corridor_equilibrium,
)
from world.tolling import PERIODS

#: A rule is immutable under a rewrite once its habit strength reaches this bar
#: (M3 design D4). Computed loop-side from the card's serialized counters and
#: carried on each :class:`RewriteRequest`; the canonical constant is mirrored
#: in agents/slow_brain.py, which owns the trigger policy. CALIBRATED value
#: (A4.3): the SR 520 joint fit under the owner-adopted rehearsal schedule
#: froze 22 (sole passing threshold; the provisional 14 and the earlier
#: config-schedule 18 both fail under it — calibration/sr520_fit_manifest
#: .json). E6 is scored across a sensitivity band — which is why
#: :func:`run_baseline_loop` takes it as a parameter rather than baking the
#: fitted value in.
STRONG_HABIT_THRESHOLD = 22


@dataclass(frozen=True)
class AnnouncedOnset:
    """The A4.2(ii) M4 price channel: an announced, known-price onset (NOT a
    discovered surprise). On global day ``day`` the slow brain fires ONCE for
    every corridor agent at the START of the day (the price was announced in
    advance, so adapted behavior begins on the onset day — no discovery lag),
    handing it ``announcement`` (`world.tolling.announcement_of` for the toll
    arm; `world.tolling.placebo_announcement` for the A4.2(iii) placebo arm,
    which yokes this trigger and nulls the reason). Onset rewrites pass the
    A4.2(i) structural-only gate.

    ``tail_surprises=False`` is the T5 tail-off ablation arm (A4.2): the
    ordinary time-surprise trigger is suppressed from ``day`` onward (only the
    announced-onset trigger fires; surprises are still observed and counted),
    bounding the uncontrolled tail-drift channel.
    """

    day: int
    announcement: Mapping
    tail_surprises: bool = True


# ---------------------------------------------------------------------------
# Result / audit / checkpoint records
# ---------------------------------------------------------------------------

@dataclass
class RewriteAuditRecord:
    """One gated-rewrite outcome, recorded for the manifest audit summary."""

    persona_id: str
    day_index: int
    accepted: bool
    attempts_used: int
    gate_failures: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "persona_id": self.persona_id,
            "day_index": self.day_index,
            "accepted": self.accepted,
            "attempts_used": self.attempts_used,
            "gate_failures": list(self.gate_failures),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "RewriteAuditRecord":
        return cls(
            persona_id=str(d["persona_id"]),
            day_index=int(d["day_index"]),
            accepted=bool(d["accepted"]),
            attempts_used=int(d["attempts_used"]),
            gate_failures=tuple(d.get("gate_failures", ())),
        )


def _realized_trip_to_dict(t: RealizedTrip) -> dict:
    return {"purpose": t.purpose, "mode": t.mode, "depart_band": t.depart_band,
            "rule_applied": t.rule_applied}


def _realized_trip_from_dict(d: Mapping) -> RealizedTrip:
    return RealizedTrip(d["purpose"], d["mode"], d["depart_band"], d.get("rule_applied"))


def _realized_day_to_dict(rd: RealizedDay) -> dict:
    return {"day_index": rd.day_index, "day_weight": rd.day_weight,
            "trips": [_realized_trip_to_dict(t) for t in rd.trips]}


def _realized_day_from_dict(d: Mapping) -> RealizedDay:
    return RealizedDay(int(d["day_index"]), float(d["day_weight"]),
                       [_realized_trip_from_dict(t) for t in d["trips"]])


def _days_map_to_dict(m: Mapping[str, Sequence[RealizedDay]]) -> dict:
    return {pid: [_realized_day_to_dict(rd) for rd in days] for pid, days in m.items()}


def _days_map_from_dict(m: Mapping[str, Sequence[Mapping]]) -> Dict[str, List[RealizedDay]]:
    return {pid: [_realized_day_from_dict(d) for d in days] for pid, days in m.items()}


@dataclass
class LoopState:
    """Serializable loop snapshot at a day boundary (D5 checkpointing).

    Carries everything needed to resume the run bit-identically: the cards
    (with live counters, surprise logs, provenance), the per-persona HabitMemory
    snapshots, the next day index, and the accumulated results (scoring/full
    realized days, rewrite audit, surprise counts, coercion log, pending
    rewrites, and the ever-surprised set that gates the trigger scan). The
    population, corridor membership, and scoring-weight map are NOT stored: they
    are re-derived deterministically from the cards + namespace + day_slots on
    resume (skeletons never change under a rewrite, so the vot draw is stable).
    """

    day_index: int  # next day to execute
    cards: List[dict]
    memories: Dict[str, HabitMemory]
    scoring_days: Dict[str, List[RealizedDay]]
    realized_days_full: Dict[str, List[RealizedDay]]
    rewrite_audit: List[RewriteAuditRecord]
    surprise_counts: Dict[int, int]
    coercion_log: List[dict]
    pending_rewrites: List[dict]
    ever_surprised: set
    n_days: int
    warmup_days: int
    namespace: str
    keep_full_window: bool = True
    #: day -> {"codes": [...], "loads": [...], "tolled": code|None} — realized
    #: corridor facility loads (A4.2: the scored volume quantities read the
    #: tolled facility's realized load; recorded every day the corridor ran).
    facility_loads: Dict[int, dict] = field(default_factory=dict)
    #: Cordon-instrument recording seam (cityk_cordon only; empty for the
    #: corridor world — recording, never behavior). Per day: unweighted
    #: counts {"n_car": car/ride travelers, "n_crossing": of which OD-crossing}
    #: (A8.4 aggregate); per persona: the day lists that let the scoring
    #: driver rebuild any weighted aggregate exactly (per-agent
    #: reconstruction self-check).
    cordon_daily: Dict[int, dict] = field(default_factory=dict)
    cordon_car_days: Dict[str, List[int]] = field(default_factory=dict)
    cordon_crossing_days: Dict[str, List[int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "day_index": self.day_index,
            "cards": self.cards,
            "memories": {pid: mem.to_dict() for pid, mem in self.memories.items()},
            "scoring_days": _days_map_to_dict(self.scoring_days),
            "realized_days_full": _days_map_to_dict(self.realized_days_full),
            "rewrite_audit": [r.to_dict() for r in self.rewrite_audit],
            "surprise_counts": {str(k): v for k, v in self.surprise_counts.items()},
            "coercion_log": self.coercion_log,
            "pending_rewrites": self.pending_rewrites,
            "ever_surprised": sorted(self.ever_surprised),
            "n_days": self.n_days,
            "warmup_days": self.warmup_days,
            "namespace": self.namespace,
            "keep_full_window": self.keep_full_window,
            "facility_loads": {str(k): v for k, v in self.facility_loads.items()},
            "cordon_daily": {str(k): v for k, v in self.cordon_daily.items()},
            "cordon_car_days": self.cordon_car_days,
            "cordon_crossing_days": self.cordon_crossing_days,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "LoopState":
        return cls(
            day_index=int(d["day_index"]),
            cards=list(d["cards"]),
            memories={pid: HabitMemory.from_dict(m) for pid, m in d["memories"].items()},
            scoring_days=_days_map_from_dict(d["scoring_days"]),
            realized_days_full=_days_map_from_dict(d["realized_days_full"]),
            rewrite_audit=[RewriteAuditRecord.from_dict(r) for r in d["rewrite_audit"]],
            surprise_counts={int(k): v for k, v in d["surprise_counts"].items()},
            coercion_log=list(d["coercion_log"]),
            pending_rewrites=list(d["pending_rewrites"]),
            ever_surprised=set(d["ever_surprised"]),
            n_days=int(d["n_days"]),
            warmup_days=int(d["warmup_days"]),
            namespace=str(d["namespace"]),
            keep_full_window=bool(d.get("keep_full_window", True)),
            facility_loads={int(k): v for k, v in d.get("facility_loads", {}).items()},
            cordon_daily={int(k): v for k, v in d.get("cordon_daily", {}).items()},
            cordon_car_days={k: list(v) for k, v in d.get("cordon_car_days", {}).items()},
            cordon_crossing_days={k: list(v) for k, v in d.get("cordon_crossing_days", {}).items()},
        )


@dataclass
class LoopResult:
    """The finished (or partially-run) loop. ``scoring_days`` is shaped exactly
    like ``card_executor.execute_days`` output for the sealed scorers: slot j of
    a persona's ``day_slots`` maps to global day ``warmup_days + j`` carrying the
    slot weight (D5). ``state`` is the resumable snapshot after the last executed
    day."""

    cards: List[dict]
    scoring_days: Dict[str, List[RealizedDay]]
    realized_days_full: Dict[str, List[RealizedDay]]
    rewrite_audit: List[RewriteAuditRecord]
    surprise_counts: Dict[int, int]
    coercion_log: List[dict]
    pending_rewrites: List[dict]
    n_days: int
    warmup_days: int
    namespace: str
    state: LoopState
    facility_loads: Dict[int, dict] = field(default_factory=dict)
    cordon_daily: Dict[int, dict] = field(default_factory=dict)
    cordon_car_days: Dict[str, List[int]] = field(default_factory=dict)
    cordon_crossing_days: Dict[str, List[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _strong_rule_ids(
    card: Mapping, threshold: int = STRONG_HABIT_THRESHOLD
) -> Tuple[str, ...]:
    """Rule ids whose habit strength has reached the immutability bar (D4),
    read straight from the card's serialized counters (no reconstruction).
    ``threshold`` is the A4.3-calibratable strong-habit bar."""
    counters = card.get("habit_counters", {})
    out: List[str] = []
    for rule in card.get("rules", []):
        rid = rule.get("id")
        c = counters.get(rid)
        if c is not None and int(c.get("strength", 0)) >= int(threshold):
            out.append(rid)
    return tuple(out)


def _request_to_dict(req: RewriteRequest) -> dict:
    """A collected (client=None) rewrite request, JSON-clean for the offline
    cluster driver. The card is included by reference (already JSON-clean)."""
    return {
        "persona_id": req.persona_id,
        "day_index": req.day_index,
        "card": req.card,
        "strong_rule_ids": list(req.strong_rule_ids),
        "attempt": req.attempt,
        "reason": req.reason,
        "announcement": dict(req.announcement) if req.announcement is not None else None,
        "shock_mode": req.shock_mode,
        "surprises": [
            {
                "persona_id": s.persona_id,
                "day_index": s.day_index,
                "context_key": s.context_key,
                "expected_minutes": s.expected_minutes,
                "realized_minutes": s.realized_minutes,
                "z": s.z,
            }
            for s in req.surprises
        ],
    }


def _mode_tally(trips: Sequence[RealizedTrip]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """(all-mode counts, work-purpose focal counts) for one lived day."""
    mode_counts: Dict[str, int] = {}
    work_counts: Dict[str, int] = {}
    for t in trips:
        mode_counts[t.mode] = mode_counts.get(t.mode, 0) + 1
        if t.purpose == "work":
            work_counts[t.mode] = work_counts.get(t.mode, 0) + 1
    return mode_counts, work_counts


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def run_baseline_loop(
    cards: Sequence[dict],
    config: WorldConfig,
    day_slots: Mapping[str, Sequence[Tuple[int, float]]],
    *,
    namespace: str,
    n_days: int,
    warmup_days: int,
    policy: SurprisePolicy,
    client: Optional[SlowBrainClient] = None,
    network_override: Optional[Callable[[int], Optional[NetworkState]]] = None,
    memory_config: Optional[SubstrateConfig] = None,
    resume_state: Optional[LoopState] = None,
    run_through_day: Optional[int] = None,
    keep_full_window: bool = True,
    copy_cards: bool = True,
    car_access: Optional[BorrowedCarAccess] = None,
    persona_pass: Optional[Mapping[str, bool]] = None,
    onset: Optional[AnnouncedOnset] = None,
    extra_onsets: Sequence[AnnouncedOnset] = (),
    strong_habit_threshold: int = STRONG_HABIT_THRESHOLD,
) -> LoopResult:
    """Run the ordinary-day loop over global days ``0 .. n_days-1`` (D5).

    Every persona lives every day: the fast brain executes with
    ``update_habits=True`` (counters LIVE, an owner directive), the day's
    corridor commuters are assigned + realized, their door-to-door times update
    each persona's HabitMemory (bias-corrected expectation, D2), surprises are
    logged to the injected ``policy``, and post warm-up a batch of triggered
    rewrites is sent to ``client`` (or collected into ``pending_rewrites`` when
    ``client is None`` — the offline-cluster mode).

    HabitMemory (the corridor-time substrate) is kept for corridor-eligible
    personas only — the sole personas that ever observe a realized corridor
    time or can surprise. Non-corridor personas still LIVE every day (their card
    habit_counters advance via the fast brain); their HabitMemory tally, which
    is not load-bearing at M3 (it feeds only the E6 hysteresis machinery), is
    not maintained — a documented performance decision, revisit if a later eval
    needs the full population's day tally.

    Resumption: pass ``resume_state`` (from a prior partial run's
    ``result.state``, optionally round-tripped through ``to_dict``/``from_dict``)
    to continue; ``run_through_day`` (inclusive) stops early to take a
    checkpoint. A resumed run reproduces the uninterrupted run bit-identically.

    Sealed pre-M4 gates (both default None = pre-decision behavior, byte-
    identical): ``car_access`` is the frozen borrowed-car availability draw
    (`agents.card_executor.BorrowedCarAccess`); ``persona_pass`` is the
    household-inherited pass map (`world.household_pass`). Both are
    deterministic configuration, not state — a resumed run must be given the
    SAME values it started with (like ``policy``/``client``).

    M4 shock machinery (A4.2; ``onset=None`` = the M3 ordinary-day loop,
    byte-identical): ``onset`` fires the announced-onset trigger at the START
    of its day for every corridor agent (rewrites through the structural-only
    gate; the placebo arm passes the nulled announcement); from the onset day
    onward, ordinary time-surprise rewrites also run shock-mode (they adapt to
    diversion-induced congestion and would otherwise be clamped by the
    pre-toll fidelity gate), or are suppressed entirely on the tail-off arm.
    ``strong_habit_threshold`` is the A4.3-calibratable immutability bar.

    Transfer-arena phase machinery (A8.1; ``extra_onsets=()`` = BT1 behavior,
    byte-identical): ``extra_onsets`` schedules FURTHER firings of the SAME
    announced-onset trigger (same request shape, same structural-only gate) at
    later transition days — the P1/P2/P3 timeline. Shock-mode starts at the
    EARLIEST onset day and persists. In a cordon world
    (``config.policy_instrument == "cordon"``) the trigger population is the
    OD-crossing personas (the cordon analog of corridor membership) and the
    loop additionally RECORDS the per-day cordon travel tally (counts +
    per-persona day lists) — recording only, never behavior.
    """
    memory_config = memory_config or DEFAULT_CONFIG
    last_day = n_days - 1 if run_through_day is None else run_through_day

    # -- init or restore state -------------------------------------------
    if resume_state is not None:
        st = resume_state
        cards_list = st.cards
        namespace = st.namespace
        n_days = st.n_days
        warmup_days = st.warmup_days
        keep_full_window = st.keep_full_window
        memories = st.memories
        start_day = st.day_index
    else:
        cards_list = copy.deepcopy(list(cards)) if copy_cards else list(cards)
        st = None
        memories = {}
        start_day = 0

    persona_ids = [str(c["persona_id"]) for c in cards_list]
    idx_of = {pid: i for i, pid in enumerate(persona_ids)}
    card_of = {pid: cards_list[i] for i, pid in enumerate(persona_ids)}

    # Population + corridor membership are deterministic in (skeletons, namespace)
    # and invariant under rewrites (skeletons never change), so they are rebuilt
    # once here and reused every day (including after a resume).
    population = bridge.population_from_cards(
        cards_list, config, namespace, persona_pass=persona_pass
    )
    row_index = bridge.persona_row_index(cards_list)
    corridor_pids = [pid for i, pid in enumerate(persona_ids) if bool(population.is_corridor[i])]

    # Cordon world (A8.1): the announced-onset trigger population is the
    # OD-crossing personas; the loop also records the daily cordon tally.
    is_cordon_world = (config.policy_instrument == "cordon"
                       and bool(config.cordon_rings))
    crossing_mask = (bridge.cordon_crossing_rows(cards_list, config)
                     if is_cordon_world else None)
    onset_pids = ([pid for i, pid in enumerate(persona_ids) if bool(crossing_mask[i])]
                  if is_cordon_world else corridor_pids)
    all_onsets: List[AnnouncedOnset] = ([onset] if onset is not None else []) \
        + list(extra_onsets)
    first_onset_day = min((o.day for o in all_onsets), default=None)

    # Scoring-weight map: persona slot j -> global day warmup+j (D5).
    scoring_span = n_days - warmup_days
    # global day -> (observed slot daynum, slot weight). The emitted scoring
    # RealizedDay carries the OBSERVED daynum, not the loop's global index —
    # execute_days's exact shape, which downstream day-index joins (e.g. the
    # within-person variance diagnostic) rely on; E1/E2 read only weights.
    scoring_weight: Dict[str, Dict[int, Tuple[int, float]]] = {}
    for pid in persona_ids:
        slots = day_slots.get(pid, [])
        wmap: Dict[int, Tuple[int, float]] = {}
        for j, (dn, w) in enumerate(slots):
            if j >= scoring_span:
                break
            wmap[warmup_days + j] = (int(dn), float(w))
        scoring_weight[pid] = wmap

    if st is None:
        for pid in corridor_pids:
            memories[pid] = HabitMemory(config=memory_config)
        scoring_days: Dict[str, List[RealizedDay]] = {pid: [] for pid in persona_ids}
        realized_full: Dict[str, List[RealizedDay]] = {pid: [] for pid in persona_ids}
        rewrite_audit: List[RewriteAuditRecord] = []
        surprise_counts: Dict[int, int] = {}
        coercion_log: List[dict] = []
        pending_rewrites: List[dict] = []
        ever_surprised: set = set()
    else:
        scoring_days = st.scoring_days
        realized_full = st.realized_days_full
        rewrite_audit = st.rewrite_audit
        surprise_counts = st.surprise_counts
        coercion_log = st.coercion_log
        pending_rewrites = st.pending_rewrites
        ever_surprised = st.ever_surprised
    facility_loads: Dict[int, dict] = st.facility_loads if st is not None else {}
    cordon_daily: Dict[int, dict] = st.cordon_daily if st is not None else {}
    cordon_car_days: Dict[str, List[int]] = st.cordon_car_days if st is not None else {}
    cordon_crossing_days: Dict[str, List[int]] = (
        st.cordon_crossing_days if st is not None else {})

    def _apply_outcomes(outcomes: List[RewriteOutcome], d: int) -> None:
        for outcome in outcomes:
            rewrite_audit.append(RewriteAuditRecord(
                persona_id=outcome.persona_id, day_index=d,
                accepted=outcome.accepted,
                attempts_used=outcome.attempts_used,
                gate_failures=tuple(outcome.gate_failures),
            ))
            if outcome.accepted:
                i = idx_of[outcome.persona_id]
                cards_list[i] = outcome.card
                card_of[outcome.persona_id] = outcome.card

    # -- day loop ---------------------------------------------------------
    for d in range(start_day, last_day + 1):
        # A4.2(ii): the announced-onset trigger fires at the START of the
        # onset day — the price was announced in advance, so the agent
        # re-optimizes before living the day (no discovery lag). Once per
        # corridor agent; yoked identically in the placebo arm.
        for o in all_onsets:
            if d != o.day:
                continue
            onset_requests: List[RewriteRequest] = []
            for pid in onset_pids:
                card = card_of[pid]
                row = idx_of[pid]
                ann = dict(o.announcement)
                ann["household_has_pass"] = bool(population.has_pass[row])
                onset_requests.append(RewriteRequest(
                    persona_id=pid, day_index=d, card=card,
                    surprises=(),
                    strong_rule_ids=_strong_rule_ids(card, strong_habit_threshold),
                    reason=REASON_ANNOUNCED_ONSET,
                    announcement=ann,
                    shock_mode=True,
                ))
            if onset_requests:
                if client is None:
                    pending_rewrites.extend(_request_to_dict(r) for r in onset_requests)
                else:
                    _apply_outcomes(client.rewrite_batch(onset_requests), d)

        per_day_slots = {pid: [(d, 1.0)] for pid in persona_ids}
        day_out = execute_days(
            cards_list, per_day_slots, namespace,
            update_habits=True, coercion_log=coercion_log,
            car_access=car_access,
        )

        # Cordon tally (recording seam, A8.4): who traveled car/ride today,
        # and of those, whose OD crosses the cordon. Counts + per-persona day
        # lists so the scoring driver can rebuild any weighted aggregate and
        # verify the reconstruction exactly.
        if is_cordon_world:
            n_car = 0
            n_crossing = 0
            for i, pid in enumerate(persona_ids):
                recs = day_out.get(pid)
                trips = recs[0].trips if recs else []
                if bridge._first_car_or_ride(trips) is None:
                    continue
                n_car += 1
                cordon_car_days.setdefault(pid, []).append(d)
                if bool(crossing_mask[i]):
                    n_crossing += 1
                    cordon_crossing_days.setdefault(pid, []).append(d)
            cordon_daily[d] = {"n_car": n_car, "n_crossing": n_crossing}

        for pid in corridor_pids:
            memories[pid].begin_day()

        state_net = None
        if network_override is not None:
            state_net = network_override(d)
        if state_net is None:
            state_net = config.network_state_for_day(d)

        table = bridge.corridor_travelers_of_day(
            day_out, cards_list, config,
            population=population, row_index=row_index,
        )
        surprises_today = 0
        day_events: Dict[str, List[SurpriseEvent]] = {}
        if len(table) > 0:
            facilities = [config.facility(c) for c in state_net.facility_codes]
            min_t0 = min(f.t0 for f in facilities)
            eq = solve_corridor_equilibrium(
                facilities,
                access=table.access,
                vot=table.vot,
                period_codes=table.period_codes,
                has_pass=table.has_pass,
                state=state_net,
                theta=config.logit_theta,
            )
            keys = ["%s:%s:%d:route" % (namespace, pid, d) for pid in table.persona_ids]
            uniforms = crn.draws(keys)
            choice = realized_facilities(uniforms, eq.choice_probs)
            loads = np.bincount(choice, minlength=len(facilities)).astype(float)
            facility_loads[d] = {
                "codes": list(state_net.facility_codes),
                "loads": loads.tolist(),
                "tolled": state_net.tolled_facility,
                "n_travelers": int(len(table)),
            }
            times = facility_times_from_loads(facilities, loads)
            realized_dtd = times[choice] + table.access
            freeflow = table.access + min_t0
            for i in range(len(table)):
                pid = table.persona_ids[i]
                key = "%s|corridor|%s" % (table.mode[i], PERIODS[int(table.period_codes[i])])
                mem = memories[pid]
                exp = bridge.expected_minutes(mem, key, float(freeflow[i]))
                realized = float(realized_dtd[i])
                if mem.observe(key, realized, exp):
                    z = mem.prediction_error_z(realized, exp)
                    event = SurpriseEvent(pid, d, key, exp, realized, z)
                    policy.log_surprise(card_of[pid], event)
                    day_events.setdefault(pid, []).append(event)
                    ever_surprised.add(pid)
                    surprises_today += 1

        for pid in corridor_pids:
            trips = day_out[pid][0].trips if day_out.get(pid) else []
            mode_counts, work_counts = _mode_tally(trips)
            mem = memories[pid]
            mem.record_daily_tally(mode_counts, work_counts)
            mem.end_day()

        surprise_counts[d] = surprises_today

        # scoring-window capture (D5): slot j -> global day warmup+j, slot
        # weight; the emitted day_index is the slot's OBSERVED daynum.
        if warmup_days <= d < n_days:
            for pid in persona_ids:
                slot = scoring_weight[pid].get(d)
                if slot is not None:
                    dn, w = slot
                    trips = day_out[pid][0].trips if day_out.get(pid) else []
                    scoring_days[pid].append(RealizedDay(dn, w, list(trips)))
        if keep_full_window:
            for pid in persona_ids:
                trips = day_out[pid][0].trips if day_out.get(pid) else []
                realized_full[pid].append(RealizedDay(d, 1.0, list(trips)))

        # rewrite trigger (post warm-up); only ever-surprised personas can fire.
        # Under the shock (d >= onset day): requests run shock-mode (the tail
        # adapts to diversion-induced congestion; the pre-toll fidelity gate
        # would clamp it, A4.2(i)) — or, on the T5 tail-off ablation arm, the
        # tail trigger is suppressed entirely (surprises still observed above).
        in_shock = first_onset_day is not None and d >= first_onset_day
        tail_suppressed = in_shock and any(
            not o.tail_surprises for o in all_onsets if d >= o.day
        )
        if d >= warmup_days and ever_surprised and not tail_suppressed:
            requests: List[RewriteRequest] = []
            for pid in sorted(ever_surprised):
                card = card_of[pid]
                if policy.should_rewrite(card, d):
                    evts = tuple(day_events.get(pid, ())[-SURPRISE_LOG_CAP:])
                    requests.append(RewriteRequest(
                        persona_id=pid, day_index=d, card=card,
                        surprises=evts,
                        strong_rule_ids=_strong_rule_ids(card, strong_habit_threshold),
                        shock_mode=in_shock,
                    ))
            if requests:
                if client is None:
                    pending_rewrites.extend(_request_to_dict(r) for r in requests)
                else:
                    _apply_outcomes(client.rewrite_batch(requests), d)

    final_state = LoopState(
        day_index=last_day + 1,
        cards=cards_list,
        memories=memories,
        scoring_days=scoring_days,
        realized_days_full=realized_full,
        rewrite_audit=rewrite_audit,
        surprise_counts=surprise_counts,
        coercion_log=coercion_log,
        pending_rewrites=pending_rewrites,
        ever_surprised=ever_surprised,
        n_days=n_days,
        warmup_days=warmup_days,
        namespace=namespace,
        keep_full_window=keep_full_window,
        facility_loads=facility_loads,
        cordon_daily=cordon_daily,
        cordon_car_days=cordon_car_days,
        cordon_crossing_days=cordon_crossing_days,
    )
    return LoopResult(
        cards=cards_list,
        scoring_days=scoring_days,
        realized_days_full=realized_full,
        rewrite_audit=rewrite_audit,
        surprise_counts=surprise_counts,
        coercion_log=coercion_log,
        pending_rewrites=pending_rewrites,
        n_days=n_days,
        warmup_days=warmup_days,
        namespace=namespace,
        state=final_state,
        facility_loads=facility_loads,
        cordon_daily=cordon_daily,
        cordon_car_days=cordon_car_days,
        cordon_crossing_days=cordon_crossing_days,
    )
