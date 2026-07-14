"""Facilities, congestion (VDF), the network timeline, and the daily corridor
equilibrium for the masked corridor world (M1, scripted agents; no LLM).

WHY THIS FILE EXISTS — the M1 novelty is ROUTE CHOICE with consequences. The
north-south corridor is served by a few parallel facilities; car/ride
travellers pick one by generalized cost, and everyone diverting onto a free
alternative makes that alternative worse. That feedback is a per-facility
volume-delay function (BPR):

    t(v) = t0 * (1 + alpha * (v / cap) ** beta)

steep on the freeway bypass (fast when empty, collapses under load) and on the
core street grid (capacity-poor), milder on the tunnel and surface arterial,
steep on the water crossing. The day's loads are a damped fixed point:
assign -> load -> retime -> reassign until loads stop moving. It MUST converge
(acceptance test 4) and be bit-identical for a given population (test 5): the
assignment is the smooth multinomial-logit split (expected volumes), a
deterministic function of the population arrays, so there is no stochastic
draw to make loads drift between runs — the closest network analogue of the
project's CRN determinism doctrine.

THE NETWORK TIMELINE is scripted by day index (four eras). Crucially, the
consumer of the network — the assignment, the population, the demo table —
receives only a raw NetworkState (which facilities exist right now, which one
charges, and its schedule). It never receives an era NAME or any history
semantics: an agent must not be able to read "the squeeze era" off the world
(01_PREREGISTRATION.md §5). The era labels live only in the timeline mapping
and are surfaced solely by the human-facing demo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from world.tolling import TollSchedule

# ---------------------------------------------------------------------------
# Facilities and the volume-delay function
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Facility:
    """One corridor facility (or the water crossing). BPR volume-delay params.

    code     one-letter facility code (e.g. "T" tunnel, "F" freeway bypass).
    t0       free-flow travel time (minutes).
    capacity practical capacity (trips/day) — the BPR reference volume.
    alpha    delay coefficient (steeper = collapses harder under load).
    beta     delay exponent (classic BPR uses 4; kept configurable).
    """

    code: str
    t0: float
    capacity: float
    alpha: float
    beta: float

    def travel_time(self, load: float) -> float:
        return bpr_time(load, self.t0, self.capacity, self.alpha, self.beta)


def bpr_time(
    load: np.ndarray | float,
    t0: np.ndarray | float,
    capacity: np.ndarray | float,
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
) -> np.ndarray | float:
    """BPR volume-delay: t0 * (1 + alpha * (load/capacity)**beta). Vectorized."""
    ratio = np.asarray(load, dtype=float) / np.asarray(capacity, dtype=float)
    return t0 * (1.0 + alpha * np.power(np.maximum(ratio, 0.0), beta))


# ---------------------------------------------------------------------------
# The network timeline (four scripted eras) and the raw NetworkState
# ---------------------------------------------------------------------------
# Era labels are documentation only; they are NEVER passed to the assignment
# or the population (see module docstring). The demo is the sole reader.
ERA_LABELS: Tuple[str, ...] = ("elevated", "squeeze", "free_tunnel", "toll_on")


@dataclass(frozen=True)
class NetworkState:
    """The raw network the world exposes on a given day: which corridor
    facilities exist, which one (if any) charges, and its schedule. No era
    name, no history — exactly what an agent is allowed to perceive."""

    facility_codes: Tuple[str, ...]
    tolled_facility: Optional[str]
    toll_schedule: Optional[TollSchedule]


def era_index_for_day(day_index: int, boundaries: Sequence[int]) -> int:
    """Which era (0..3) a day index falls in, given three ascending day-index
    boundaries. day < b0 -> era 0, < b1 -> era 1, < b2 -> era 2, else era 3."""
    b0, b1, b2 = boundaries
    if day_index < b0:
        return 0
    if day_index < b1:
        return 1
    if day_index < b2:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Daily corridor equilibrium (damped multinomial-logit fixed point)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquilibriumResult:
    """Outcome of one day's corridor assignment."""

    facility_codes: Tuple[str, ...]
    loads: np.ndarray           # per-facility volume, same order as codes
    times: np.ndarray           # per-facility congested minutes
    mean_door_to_door: float    # load-weighted mean corridor time (minutes)
    n_travelers: int            # car/ride corridor travellers assigned
    iterations: int
    residual: float             # final max per-facility load change / n
    converged: bool

    def load_of(self, code: str) -> float:
        return float(self.loads[self.facility_codes.index(code)])

    def time_of(self, code: str) -> float:
        return float(self.times[self.facility_codes.index(code)])


def _assign(theta: float, gc: np.ndarray) -> np.ndarray:
    """Multinomial-logit split: P(facility) ∝ exp(-theta * generalized_cost),
    returned as an (n_agents, n_facilities) probability matrix. Softmax is
    computed in a max-shifted, overflow-safe form."""
    u = -theta * gc
    u -= u.max(axis=1, keepdims=True)
    e = np.exp(u)
    return e / e.sum(axis=1, keepdims=True)


def solve_corridor_equilibrium(
    facilities: Sequence[Facility],
    access: np.ndarray,
    vot: np.ndarray,
    period_codes: np.ndarray,
    has_pass: np.ndarray,
    state: NetworkState,
    *,
    theta: float,
    damping: float = 0.5,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> EquilibriumResult:
    """Damped logit fixed point over the corridor facilities.

    Each of the ``n`` car/ride corridor travellers splits across the available
    facilities by generalized cost GC(f) = time(f) + toll(f)/VoT, where only
    the tolled facility carries a toll (converted to minutes-equivalent by the
    agent's value of time). Expected volumes are the column sums of the logit
    matrix, so total volume is conserved exactly (each agent's probabilities
    sum to 1) and the whole map is a deterministic function of the population
    arrays — same population, bit-identical loads.
    """
    codes = tuple(f.code for f in facilities)
    n_f = len(facilities)
    n = int(access.shape[0])

    t0 = np.array([f.t0 for f in facilities], dtype=float)
    cap = np.array([f.capacity for f in facilities], dtype=float)
    alpha = np.array([f.alpha for f in facilities], dtype=float)
    beta = np.array([f.beta for f in facilities], dtype=float)

    # Per-agent toll term (minutes-equivalent) for the tolled facility column,
    # constant across the fixed-point iterations (it depends only on schedule,
    # period, and pass — not on congestion).
    toll_col = -1
    toll_term = np.zeros(n, dtype=float)
    if state.tolled_facility in codes and state.toll_schedule is not None:
        toll_col = codes.index(state.tolled_facility)
        toll_credits = state.toll_schedule.toll_array(period_codes, has_pass)
        toll_term = toll_credits / vot

    def times_of(load: np.ndarray) -> np.ndarray:
        return bpr_time(load, t0, cap, alpha, beta)

    def loads_from_times(times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gc = np.broadcast_to(times, (n, n_f)).copy()
        if toll_col >= 0:
            gc[:, toll_col] += toll_term
        probs = _assign(theta, gc)
        return probs.sum(axis=0), probs

    if n == 0:
        loads = np.zeros(n_f)
        return EquilibriumResult(codes, loads, t0.copy(), 0.0, 0, 0, 0.0, True)

    # Seed from a free-flow assignment, then damp toward the fixed point.
    load, _ = loads_from_times(t0)
    residual = float("inf")
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        target, _ = loads_from_times(times_of(load))
        new_load = (1.0 - damping) * load + damping * target
        residual = float(np.max(np.abs(new_load - load)) / n)
        load = new_load
        if residual < tol:
            converged = True
            break

    times = times_of(load)
    _, probs = loads_from_times(times)
    # Expected per-agent door-to-door time: access + logit-weighted facility
    # time; mean over the corridor travellers.
    door = access + (probs * times[np.newaxis, :]).sum(axis=1)
    mean_door = float(door.mean())

    return EquilibriumResult(
        facility_codes=codes,
        loads=load,
        times=times,
        mean_door_to_door=mean_door,
        n_travelers=n,
        iterations=iterations,
        residual=residual,
        converged=converged,
    )


def water_crossing_time(load: float, facility: Facility) -> float:
    """Congested time on the (choice-free) water crossing. It has no route
    alternative and mode is fixed per agent-day, so its load is a fixed count
    and its time is a static VDF evaluation — included for the car/ride
    zone-to-zone times and for narrative completeness."""
    return facility.travel_time(load)
