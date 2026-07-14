"""Acceptance tests for the world's REALIZED-choice layer (M2_ARCH_SPEC D4).

The solver gives the smooth expected-value equilibrium; the realized layer
draws each corridor traveller's discrete facility through the CRN layer and
retimes the realized loads through the ONE loads->times path. These tests pin:
conservation, hash-based determinism (incl. across processes), CRN twin-world
pairing (inverse-CDF over a fixed facility order), the single loads->times
path, ensemble independence, that the layer is bit-identical to the pre-layer
world when off, and the performance budget.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from world import crn
from world.config import get_config
from world.network import (
    facility_times_from_loads,
    realized_facilities,
    solve_corridor_equilibrium,
)
from world.simulation import simulate_day, simulate_era

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEED = 4242
_N_AGENTS = 10_000

# Golden expected-value output for (cityk_corridor, seed 4242, n 10k, era 3),
# captured from the solver BEFORE the realized layer existed. The layer-off
# path must reproduce these bit-for-bit (guard against silent drift).
_GOLD_LOADS = {"T": 418.26870830756025, "S": 559.8661289221931,
               "F": 750.8868250209961, "D": 523.9783377492505}
_GOLD_TIMES = {"T": 8.036728462343204, "S": 15.452348987973643,
               "F": 12.512740157025984, "D": 16.110141067324058}
_GOLD_MEAN_DOOR = 23.189011031278685
_GOLD_RESIDUAL = 9.980571514614185e-05
_GOLD_ITERS = 35
_GOLD_N_CORRIDOR = 2253


@pytest.fixture(scope="module")
def config():
    return get_config("cityk_corridor")


@pytest.fixture(scope="module")
def population(config):
    from world.population import build_population
    return build_population(config, _SEED, _N_AGENTS)


@pytest.fixture(scope="module")
def agent_ids(population):
    return np.array(["P%05d" % i for i in range(len(population))])


def _realized(config, population, agent_ids, *, mult=None, day_index=5,
              namespace="run0"):
    """Simulate the tolled era with the realized layer on, optionally at a toll
    multiplier. Returns the DayResult."""
    schedule = None
    if mult is not None:
        schedule = config.toll_schedule.with_multiplier(mult)
    return simulate_era(
        config, population, 3, schedule=schedule,
        realized=True, agent_ids=agent_ids, day_index=day_index,
        namespace=namespace, max_iter=60, tol=1e-4,
    )


# --- Conservation -----------------------------------------------------------

def test_realized_loads_conserve_n_travelers(config, population, agent_ids):
    """Realized facility loads sum to the corridor-traveller count EXACTLY: a
    realized draw moves each traveller onto one facility, never creating or
    destroying travellers (the discrete analogue of the solver's exact
    volume conservation)."""
    r = _realized(config, population, agent_ids)
    total = sum(r.realized_loads.values())
    assert total == float(r.n_corridor_travelers)
    assert int(total) == r.n_corridor_travelers
    # realized_choice is one facility index per corridor traveller.
    assert r.realized_choice.shape[0] == r.n_corridor_travelers
    assert r.realized_choice.min() >= 0
    assert r.realized_choice.max() < len(r.facility_codes)


# --- Determinism ------------------------------------------------------------

def test_realized_determinism_two_calls(config, population, agent_ids):
    """Same (population, state, namespace, day) -> bit-identical realized
    arrays across two calls (no global RNG state)."""
    a = _realized(config, population, agent_ids)
    b = _realized(config, population, agent_ids)
    assert np.array_equal(a.realized_choice, b.realized_choice)
    assert a.realized_loads == b.realized_loads
    assert a.realized_times == b.realized_times


def test_realized_determinism_across_processes(config, population, agent_ids):
    """A fresh interpreter reproduces the realized choices bit-for-bit — the
    draw is a pure function of the CRN keys (hash-based, stateless)."""
    r = _realized(config, population, agent_ids)
    here_hash = hashlib.sha256(
        np.ascontiguousarray(r.realized_choice).tobytes()
    ).hexdigest()
    snippet = (
        "import hashlib, numpy as np;"
        "from world.config import get_config;"
        "from world.population import build_population;"
        "from world.simulation import simulate_era;"
        "cfg=get_config('cityk_corridor');"
        "pop=build_population(cfg,%d,%d);"
        "ids=np.array(['P%%05d'%%i for i in range(len(pop))]);"
        "r=simulate_era(cfg,pop,3,realized=True,agent_ids=ids,day_index=5,"
        "namespace='run0',max_iter=60,tol=1e-4);"
        "print(hashlib.sha256(np.ascontiguousarray(r.realized_choice)"
        ".tobytes()).hexdigest())" % (_SEED, _N_AGENTS)
    )
    out = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == here_hash


# --- CRN pairing (twin-world coupling) --------------------------------------

def test_crn_pairing_unchanged_probs_keep_facility(config, population, agent_ids):
    """Two WORLD states (toll multiplier 0 vs 2) sharing the SAME keys: no
    agent changes facility unless its choice probabilities changed. Because the
    realized draw is inverse-CDF over a FIXED facility order reusing the agent's
    per-key uniform, an unchanged probability row must keep the same facility;
    a probability shift only re-routes an agent whose uniform now sits on the
    other side of a cumulative boundary."""
    corridor = population.corridor_travelers
    facilities0, r0, probs0 = _solve_probs_realized(config, population, agent_ids, 0.0)
    facilities2, r2, probs2 = _solve_probs_realized(config, population, agent_ids, 2.0)

    probs_changed = ~np.all(probs0 == probs2, axis=1)
    facility_changed = r0 != r2

    # Sanity: the toll genuinely moved probabilities and rerouted some agents,
    # so the coupling assertion below is not vacuous.
    assert probs_changed.any()
    assert facility_changed.any()

    # The coupling guarantee: facility_changed is a SUBSET of probs_changed.
    assert not np.any(facility_changed & ~probs_changed)

    # And the realized draw computed inside simulate_day matches the documented
    # kernel exactly (integration uses realized_facilities on crn.draws).
    day = _realized(config, population, agent_ids, mult=0.0)
    assert np.array_equal(day.realized_choice, r0)


def test_inverse_cdf_only_threshold_crossers_move():
    """Direct unit test of the fixed-order inverse-CDF kernel: shifting the
    facility-0 boundary from 0.5 to 0.8 re-routes exactly the agents whose
    uniform lies in [0.5, 0.8); everyone else is untouched."""
    uniforms = np.array([0.10, 0.55, 0.70, 0.90])
    base = np.tile([0.5, 0.5], (4, 1))
    shifted = np.tile([0.8, 0.2], (4, 1))
    r_base = realized_facilities(uniforms, base)
    r_shift = realized_facilities(uniforms, shifted)
    # base: <0.5 -> 0 else 1  =>  [0,1,1,1]
    assert list(r_base) == [0, 1, 1, 1]
    # shift: <0.8 -> 0 else 1  =>  [0,0,0,1]
    assert list(r_shift) == [0, 0, 0, 1]
    # Only the agents with uniform in [0.5,0.8) moved (indices 1 and 2).
    moved = r_base != r_shift
    assert list(moved) == [False, True, True, False]


# --- One code path ----------------------------------------------------------

def test_one_loads_to_times_code_path(config, population, agent_ids):
    """realized_times equals the SHARED loads->times function applied to
    realized_loads — exact equality, proving there is no second physics for the
    realized branch (D4 'one code path')."""
    r = _realized(config, population, agent_ids)
    facilities = [config.facility(c) for c in r.facility_codes]
    load_arr = np.array([r.realized_loads[c] for c in r.facility_codes])
    times = facility_times_from_loads(facilities, load_arr)
    for i, c in enumerate(r.facility_codes):
        assert r.realized_times[c] == times[i]


# --- Expected-value fields unchanged when the layer is off ------------------

def test_layer_off_is_bit_identical_to_golden(config, population):
    """With the realized layer OFF (the default), the DayResult's expected
    fields reproduce the pre-realized-layer golden output bit-for-bit and the
    realized_* fields are None — zero behaviour change for existing callers."""
    r = simulate_era(config, population, 3, max_iter=60, tol=1e-4)
    assert r.realized_loads is None
    assert r.realized_times is None
    assert r.realized_choice is None
    assert r.facility_loads == _GOLD_LOADS
    assert r.facility_times == _GOLD_TIMES
    assert r.mean_door_to_door == _GOLD_MEAN_DOOR
    assert r.residual == _GOLD_RESIDUAL
    assert r.iterations == _GOLD_ITERS
    assert r.n_corridor_travelers == _GOLD_N_CORRIDOR


# --- Ensemble independence --------------------------------------------------

def test_ensemble_namespaces_are_independent(config, population, agent_ids):
    """run0 and run1 namespaces give DIFFERENT realized draws for the same
    population/day (independent ensemble streams), while each is internally
    deterministic."""
    r0 = _realized(config, population, agent_ids, namespace="run0")
    r1 = _realized(config, population, agent_ids, namespace="run1")
    assert not np.array_equal(r0.realized_choice, r1.realized_choice)
    # Both still sum to n (each is a valid realized assignment).
    assert sum(r0.realized_loads.values()) == float(r0.n_corridor_travelers)
    assert sum(r1.realized_loads.values()) == float(r1.n_corridor_travelers)


# --- Performance ------------------------------------------------------------

def test_realized_full_day_performance(config, population, agent_ids):
    """One full simulated day of 10k agents INCLUDING the realized layer stays
    well under 50 ms (observed ~7 ms). Matches the existing world perf-guard
    style: warm once, time a run, assert a budget."""
    _realized(config, population, agent_ids)  # warm
    state = config.network_state_for_day(config.era_boundaries[2])
    t0 = time.perf_counter()
    r = simulate_day(config, population, state, realized=True,
                     agent_ids=agent_ids, day_index=5)
    elapsed = time.perf_counter() - t0
    assert r.realized_choice is not None
    assert elapsed < 0.05, f"realized day took {elapsed * 1e3:.1f} ms (budget 50 ms)"


# --- helper -----------------------------------------------------------------

def _solve_probs_realized(config, population, agent_ids, mult):
    """Solve the tolled-era equilibrium at a toll multiplier and draw the
    realized facilities with the shared keys — mirrors simulate_day's realized
    path exactly, but also returns the per-agent choice_probs (which DayResult
    does not carry) so pairing can be checked."""
    schedule = config.toll_schedule.with_multiplier(mult)
    state = config.network_state_for_day(config.era_boundaries[2], schedule=schedule)
    facilities = [config.facility(c) for c in state.facility_codes]
    corridor = population.corridor_travelers
    eq = solve_corridor_equilibrium(
        facilities,
        access=population.access[corridor],
        vot=population.vot[corridor],
        period_codes=population.period[corridor],
        has_pass=population.has_pass[corridor],
        state=state,
        theta=config.logit_theta,
        max_iter=60,
        tol=1e-4,
    )
    corridor_ids = agent_ids[corridor]
    keys = ["run0:%s:5:route" % aid for aid in corridor_ids]
    realized = realized_facilities(crn.draws(keys), eq.choice_probs)
    return facilities, realized, eq.choice_probs
