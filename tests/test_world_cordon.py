"""Acceptance test for the transfer-arena cordon world (cityk_cordon).

The cordon world is the v1 shape kept for the transfer arena: the policy
instrument is a fee for crossing into the central rings ("core"+"inner"), not
a tolled link. It ships on the SAME machinery as the corridor world (only the
config data differs), so the one thing to prove here is that it runs and
actually charges cordon crossings.
"""
from __future__ import annotations

from world.config import get_config
from world.population import build_population, cordon_crossing_mask
from world.simulation import simulate_day

_N_AGENTS = 10_000


def test_cordon_world_runs_and_charges_crossings():
    config = get_config("cityk_cordon")
    assert config.policy_instrument == "cordon"
    assert config.cordon_rings == ("core", "inner")

    population = build_population(config, seed=314, n_agents=_N_AGENTS)
    state = config.network_state_for_day(400)  # a day well past the last era boundary
    result = simulate_day(config, population, state)

    # It runs (the shared corridor equilibrium still converges) ...
    assert result.converged

    # ... and it charges cordon crossings: a positive number of trips at
    # positive revenue.
    assert result.cordon_charged_trips > 0
    assert result.cordon_revenue > 0.0

    # The charged-trip count matches the independently-computed cordon mask.
    expected = int(cordon_crossing_mask(population, config.cordon_rings).sum())
    assert result.cordon_charged_trips == expected

    # The corridor tunnel is NEVER tolled in the cordon world (the instrument
    # is the cordon, not the link) — no era charges facility T.
    assert all(tolled is None for tolled in config.era_tolled.values())


def test_corridor_world_does_not_charge_a_cordon():
    """The corridor world's instrument is the tolled link, not a cordon: it
    reports no cordon charges."""
    config = get_config("cityk_corridor")
    population = build_population(config, seed=314, n_agents=_N_AGENTS)
    state = config.network_state_for_day(config.era_boundaries[2])
    result = simulate_day(config, population, state)
    assert result.cordon_charged_trips == 0
    assert result.cordon_revenue == 0.0
