"""Acceptance tests for the M1 corridor world's determinism and performance.

Determinism is the world-side analogue of the project's CRN doctrine: a
population is one deterministic function of (config, seed), so the same seed
must reproduce bit-identical facility loads. Performance: one full simulated
day of 10k agents, INCLUDING the equilibrium loop, must run in seconds.
"""
from __future__ import annotations

import time

import numpy as np

from world.config import get_config
from world.population import build_population
from world.simulation import simulate_day, simulate_era

_N_AGENTS = 10_000


# --- Test 5: same seed -> identical facility loads --------------------------

def test_same_seed_gives_identical_loads():
    """Two populations built from the same seed produce byte-identical facility
    loads across every era (exact array equality, not approximate)."""
    config = get_config("cityk_corridor")
    pop_a = build_population(config, seed=99, n_agents=_N_AGENTS)
    pop_b = build_population(config, seed=99, n_agents=_N_AGENTS)

    # The population arrays themselves must match exactly.
    assert np.array_equal(pop_a.home_zone, pop_b.home_zone)
    assert np.array_equal(pop_a.work_zone, pop_b.work_zone)
    assert np.array_equal(pop_a.mode, pop_b.mode)
    assert np.array_equal(pop_a.vot, pop_b.vot)
    assert np.array_equal(pop_a.has_pass, pop_b.has_pass)
    assert np.array_equal(pop_a.period, pop_b.period)

    for era in range(4):
        ra = simulate_era(config, pop_a, era)
        rb = simulate_era(config, pop_b, era)
        # Exact equality of the per-facility load floats, not approximate.
        assert ra.facility_loads == rb.facility_loads, (
            f"era {era} loads differ between identical seeds: "
            f"{ra.facility_loads} vs {rb.facility_loads}"
        )
        assert ra.facility_times == rb.facility_times


def test_different_seed_changes_loads():
    """Sanity check that the loads actually depend on the seed (so the
    determinism test above is not passing vacuously)."""
    config = get_config("cityk_corridor")
    pop_a = build_population(config, seed=1, n_agents=_N_AGENTS)
    pop_b = build_population(config, seed=2, n_agents=_N_AGENTS)
    ra = simulate_era(config, pop_a, 3)
    rb = simulate_era(config, pop_b, 3)
    assert ra.facility_loads != rb.facility_loads


# --- Test 6: performance -----------------------------------------------------

def test_full_day_runs_in_seconds():
    """One full simulated day (10k agents, incl. the equilibrium loop) runs in
    well under 10 seconds — the target is 'seconds'. Population build is timed
    too and reported; both together stay comfortably inside the budget."""
    config = get_config("cityk_corridor")

    build_start = time.perf_counter()
    population = build_population(config, seed=7, n_agents=_N_AGENTS)
    build_seconds = time.perf_counter() - build_start

    state = config.network_state_for_day(config.era_boundaries[2])  # tolled era
    day_start = time.perf_counter()
    result = simulate_day(config, population, state)
    day_seconds = time.perf_counter() - day_start

    assert result.converged
    assert day_seconds < 10.0, f"one day took {day_seconds:.3f}s (budget 10s)"
    # Generous headroom check on the combined build+simulate path.
    assert build_seconds + day_seconds < 10.0
