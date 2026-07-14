"""Acceptance tests for the M1 masked corridor world — the route-choice
mechanism itself: monotone diversion, its congestion consequences, the
network-timeline era ordering, and equilibrium convergence.

These are written against the PUBLIC world surface only (build a config, build
a population, simulate). They assert the qualitative properties the scripted
world must exhibit before any LLM agent is allowed in (world plan / M1 build
order), not fitted numbers.
"""
from __future__ import annotations

import pytest

from world.config import get_config
from world.population import build_population
from world.simulation import simulate_era, toll_multiplier_sweep

_SEED = 4242
_N_AGENTS = 10_000
_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0, 4.0)
_MAX_ITER = 10  # the spec's "~10 iterations" budget
_TOL = 1e-4


@pytest.fixture(scope="module")
def config():
    return get_config("cityk_corridor")


@pytest.fixture(scope="module")
def population(config):
    return build_population(config, _SEED, _N_AGENTS)


@pytest.fixture(scope="module")
def sweep(config, population):
    return toll_multiplier_sweep(
        config, population, _MULTIPLIERS,
        max_iter=_MAX_ITER, tol=_TOL,
    )


def _strictly_decreasing(values):
    return all(a > b for a, b in zip(values, values[1:]))


def _strictly_increasing(values):
    return all(a < b for a, b in zip(values, values[1:]))


# --- Test 1: monotone diversion + volume conservation -----------------------

def test_monotone_diversion_and_conservation(sweep):
    """As the toll multiplier rises {0, 0.5, 1, 2, 4}: the tunnel T's corridor
    volume strictly decreases; the free substitutes S+F+D strictly increase;
    total corridor volume is conserved (route choice moves travellers between
    facilities, it never creates or destroys them)."""
    t_loads = [r.facility_loads["T"] for _, r in sweep]
    sfd_loads = [
        r.facility_loads["S"] + r.facility_loads["F"] + r.facility_loads["D"]
        for _, r in sweep
    ]
    totals = [r.total_corridor_volume() for _, r in sweep]

    assert _strictly_decreasing(t_loads), f"T not strictly decreasing: {t_loads}"
    assert _strictly_increasing(sfd_loads), f"S+F+D not strictly increasing: {sfd_loads}"

    # Total corridor volume identical across every multiplier (= number of
    # car/ride corridor travellers), to numerical tolerance.
    for total in totals:
        assert total == pytest.approx(totals[0], abs=1e-6)


# --- Test 2: diversion has consequences -------------------------------------

def test_diversion_worsens_the_alternatives(sweep):
    """The free substitutes get worse as more travellers divert onto them:
    the equilibrium travel times on the freeway bypass F and the core street
    grid D both rise strictly with the toll multiplier."""
    f_times = [r.facility_times["F"] for _, r in sweep]
    d_times = [r.facility_times["D"] for _, r in sweep]
    assert _strictly_increasing(f_times), f"F time not strictly increasing: {f_times}"
    assert _strictly_increasing(d_times), f"D time not strictly increasing: {d_times}"


# --- Test 3: network-timeline era sanity ------------------------------------

def test_era_ordering_and_free_tunnel_plurality(config, population):
    """Mean corridor door-to-door time is worst in the squeeze era (era1: the
    fast spine V is gone and the tunnel T is not yet open) — worse than both
    the elevated era (era0) and the free-tunnel era (era2). And once T opens
    free (era2), it carries the plurality of corridor car/ride volume."""
    results = {era: simulate_era(config, population, era,
                                max_iter=_MAX_ITER, tol=_TOL)
               for era in range(4)}

    dd = {era: r.mean_door_to_door for era, r in results.items()}
    assert dd[1] > dd[0], f"squeeze not worse than elevated: {dd}"
    assert dd[1] > dd[2], f"squeeze not worse than free_tunnel: {dd}"

    era2 = results[2]
    t_load = era2.facility_loads["T"]
    assert all(
        t_load > era2.facility_loads[c] for c in ("S", "F", "D")
    ), f"T is not the plurality facility in era2: {era2.facility_loads}"


# --- Test 4: equilibrium converges within tolerance and budget --------------

def test_equilibrium_converges_all_eras(config, population):
    """Every era's daily equilibrium converges within the load-change
    tolerance and the iteration budget (damped fixed point, damping ~0.5)."""
    for era in range(4):
        r = simulate_era(config, population, era, max_iter=_MAX_ITER, tol=_TOL)
        assert r.converged, (
            f"era {era} did not converge: residual={r.residual:.2e} "
            f"after {r.iterations} iterations"
        )
        assert r.iterations <= _MAX_ITER
        assert r.residual < _TOL


def test_equilibrium_converges_across_toll_sweep(sweep):
    """Convergence is not a fluke of one price: every swept toll multiplier
    also converges within budget."""
    for multiplier, r in sweep:
        assert r.converged, (
            f"multiplier {multiplier} did not converge: "
            f"residual={r.residual:.2e} after {r.iterations} iterations"
        )
        assert r.iterations <= _MAX_ITER
