"""One simulated day for the masked corridor world (M1) — the entry point the
acceptance tests and the demo drive.

WHY THIS FILE EXISTS — it is the single seam that turns (config, population,
raw network state) into a day's outcome: the corridor equilibrium (route
choice + congestion), the water-crossing load, and, for the cordon world, the
cordon charges. It carries NO policy history and NO era name; it consumes only
a NetworkState (network.py), so the same function serves every era and both
world shapes. Determinism is inherited from the population and the equilibrium
solver — same population -> bit-identical loads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from world import crn
from world.config import WorldConfig
from world.network import (
    EquilibriumResult,
    NetworkState,
    facility_times_from_loads,
    realized_facilities,
    solve_corridor_equilibrium,
    water_crossing_time,
)
from world.population import AgentPopulation, cordon_crossing_mask


@dataclass(frozen=True)
class DayResult:
    """Everything one simulated day produces."""

    facility_codes: Tuple[str, ...]
    facility_loads: Dict[str, float]
    facility_times: Dict[str, float]
    mean_door_to_door: float
    n_corridor_travelers: int
    iterations: int
    residual: float
    converged: bool

    water_load: float
    water_time: float

    # Cordon instrument (cityk_cordon only; zero/empty for the corridor world).
    cordon_charged_trips: int
    cordon_revenue: float

    # --- Realized-choice layer (D4). Present only when the caller requests the
    # discrete-draw layer; None otherwise, so the default path is bit-identical
    # to the pre-realized-layer world. ---
    #
    # The expected fields above are the smooth logit fixed point (test-only,
    # retained). The realized fields below are what every SCORED quantity
    # downstream consumes (01_PREREGISTRATION.md §7 A2.2(ii) needs realized
    # per-agent choices; E1/E2/E5 score realized). Each corridor traveller
    # draws ONE realized facility through the CRN layer, so realized_loads are
    # multinomial tallies that sit around the expected loads with O(sqrt(n))
    # sampling noise (~±1/sqrt(n) relative per facility); realized_times are
    # those realized loads pushed through the SAME loads->times path
    # (facility_times_from_loads) the solver uses — one code path, never a
    # second physics.
    realized_loads: Optional[Dict[str, float]] = None
    realized_times: Optional[Dict[str, float]] = None
    realized_choice: Optional[np.ndarray] = None  # facility index per corridor agent

    def total_corridor_volume(self) -> float:
        return float(sum(self.facility_loads.values()))

    def total_realized_volume(self) -> float:
        """Realized corridor travellers tallied (conserves n exactly)."""
        if self.realized_loads is None:
            return 0.0
        return float(sum(self.realized_loads.values()))


def simulate_day(
    config: WorldConfig,
    population: AgentPopulation,
    state: NetworkState,
    *,
    max_iter: int = 60,
    tol: float = 1e-4,
    realized: bool = False,
    agent_ids: Optional[np.ndarray] = None,
    day_index: int = 0,
    namespace: str = "run0",
) -> DayResult:
    """Simulate one day: assign the corridor equilibrium under ``state``, load
    the water crossing, and (for the cordon world) tally cordon charges.

    The expected-value equilibrium is always computed. When ``realized`` is set
    (or an ``agent_ids`` array is supplied) the discrete-draw layer runs on top
    of it (D4): each corridor traveller draws its REALIZED facility through the
    CRN layer under the key ``"{namespace}:{agent_id}:{day_index}:route"``,
    realized loads are the tallies of those draws, and realized times come from
    the SAME loads->times path as the solver. With the layer OFF (the default)
    the realized_* fields are None and the result is bit-identical to the
    pre-realized-layer world — existing tests see zero change.

    Pairing: twin worlds (e.g. a toll-on vs toll-off counterfactual) that pass
    the SAME ``agent_ids``/``day_index``/``namespace`` share keys, so each agent
    reuses one uniform and only re-routes when a probability shift crosses its
    threshold. Ensembles use distinct ``namespace`` (``run0``, ``run1``, …) for
    independent streams. ``agent_ids`` (length == len(population)) are the
    caller-supplied persona ids; when omitted, positional indices are used."""
    facilities = [config.facility(code) for code in state.facility_codes]

    corridor = population.corridor_travelers
    eq: EquilibriumResult = solve_corridor_equilibrium(
        facilities,
        access=population.access[corridor],
        vot=population.vot[corridor],
        period_codes=population.period[corridor],
        has_pass=population.has_pass[corridor],
        state=state,
        theta=config.logit_theta,
        max_iter=max_iter,
        tol=tol,
    )

    # Realized-choice layer (D4): drawn only when requested; otherwise the
    # realized_* fields stay None and the day is bit-identical to before.
    realized_loads: Optional[Dict[str, float]] = None
    realized_times: Optional[Dict[str, float]] = None
    realized_choice: Optional[np.ndarray] = None
    if realized or agent_ids is not None:
        if agent_ids is None:
            ids = np.arange(len(population))
        else:
            ids = np.asarray(agent_ids)
            if ids.shape[0] != len(population):
                raise ValueError(
                    "agent_ids length %d != population length %d"
                    % (ids.shape[0], len(population))
                )
        corridor_ids = ids[corridor]
        # Key contract: "{namespace}:{persona_id}:{day_index}:{site}", site=route.
        keys = ["%s:%s:%d:route" % (namespace, aid, day_index) for aid in corridor_ids]
        uniforms = crn.draws(keys)
        idx = realized_facilities(uniforms, eq.choice_probs)
        n_f = len(eq.facility_codes)
        load_arr = np.bincount(idx, minlength=n_f).astype(float)
        time_arr = facility_times_from_loads(facilities, load_arr)
        realized_loads = {c: float(load_arr[i]) for i, c in enumerate(eq.facility_codes)}
        realized_times = {c: float(time_arr[i]) for i, c in enumerate(eq.facility_codes)}
        realized_choice = idx

    # Water crossing: fixed count of car/ride crossers, static VDF (no route
    # alternative), so its time is one evaluation.
    water = population.water_travelers
    water_load = float(water.sum())
    water_time = water_crossing_time(water_load, config.water_facility)

    # Cordon instrument: charge car/ride trips that cross into the cordon rings.
    cordon_trips = 0
    cordon_revenue = 0.0
    if config.policy_instrument == "cordon" and config.cordon_rings and state.toll_schedule is None:
        # The cordon fee reuses the corridor toll schedule machinery. It is
        # active whenever the cordon world is simulated (no era gating here at
        # M1 — the transfer arena keeps the instrument minimal).
        schedule = config.toll_schedule
        crossing = cordon_crossing_mask(population, config.cordon_rings)
        charges = schedule.toll_array(
            population.period[crossing], population.has_pass[crossing]
        )
        cordon_trips = int(crossing.sum())
        cordon_revenue = float(charges.sum())

    return DayResult(
        facility_codes=eq.facility_codes,
        facility_loads={c: float(eq.loads[i]) for i, c in enumerate(eq.facility_codes)},
        facility_times={c: float(eq.times[i]) for i, c in enumerate(eq.facility_codes)},
        mean_door_to_door=eq.mean_door_to_door,
        n_corridor_travelers=eq.n_travelers,
        iterations=eq.iterations,
        residual=eq.residual,
        converged=eq.converged,
        water_load=water_load,
        water_time=water_time,
        cordon_charged_trips=cordon_trips,
        cordon_revenue=cordon_revenue,
        realized_loads=realized_loads,
        realized_times=realized_times,
        realized_choice=realized_choice,
    )


def simulate_era(
    config: WorldConfig,
    population: AgentPopulation,
    era_index: int,
    *,
    schedule=None,
    **kwargs,
) -> DayResult:
    """Convenience: simulate a representative day of a given era (0..3) by
    picking a day index inside that era's window."""
    day = _representative_day(config, era_index)
    state = config.network_state_for_day(day, schedule=schedule)
    return simulate_day(config, population, state, **kwargs)


def _representative_day(config: WorldConfig, era_index: int) -> int:
    """A day index that lands inside the requested era's window."""
    b0, b1, b2 = config.era_boundaries
    windows = {0: 0, 1: b0, 2: b1, 3: b2}
    return windows[era_index]


def toll_multiplier_sweep(
    config: WorldConfig,
    population: AgentPopulation,
    multipliers,
    *,
    era_index: int = 3,
    **kwargs,
) -> List[Tuple[float, DayResult]]:
    """Run the tolled era at a series of E5 price multipliers. Returns
    (multiplier, DayResult) pairs — the diversion table the demo prints and
    the monotone-diversion acceptance test consumes."""
    results: List[Tuple[float, DayResult]] = []
    for m in multipliers:
        schedule = config.toll_schedule.with_multiplier(m)
        results.append((float(m), simulate_era(config, population, era_index,
                                              schedule=schedule, **kwargs)))
    return results
