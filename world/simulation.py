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
from typing import Dict, List, Tuple

from world.config import WorldConfig
from world.network import (
    EquilibriumResult,
    NetworkState,
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

    def total_corridor_volume(self) -> float:
        return float(sum(self.facility_loads.values()))


def simulate_day(
    config: WorldConfig,
    population: AgentPopulation,
    state: NetworkState,
    *,
    damping: float = 0.5,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> DayResult:
    """Simulate one day: assign the corridor equilibrium under ``state``, load
    the water crossing, and (for the cordon world) tally cordon charges."""
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
        damping=damping,
        max_iter=max_iter,
        tol=tol,
    )

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
