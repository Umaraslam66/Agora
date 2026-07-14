"""Scripted agent population for the masked corridor world (M1).

WHY THIS FILE EXISTS — M1 has NO LLM anywhere. The population here is a
plain, seeded, numpy-vectorized draw: it exists so the world's route-choice
and congestion machinery can be exercised and its acceptance tests (sane
diversion under a toy toll) can pass before any agent brain is introduced.
Every agent is one representative person-day trip (home -> work) with a fixed
travel mode, a value of time, a toll-pass boolean, and a departure period.

DETERMINISM (CRN doctrine, 01_PREREGISTRATION.md): the whole population is one
deterministic function of (config, seed). numpy's default_rng is drawn in a
fixed field order, so the same seed yields bit-identical arrays and therefore
bit-identical facility loads (acceptance test 5). No import from agents/,
serving/, or grounding/ — world/ stays self-contained at M1 — and no torch,
no transformers, no LLM.

HETEROGENEITY that matters for route choice: value of time is per-agent
lognormal, so the toll bites differently across agents (a high-VoT driver
keeps paying for the fast tolled facility; a low-VoT driver diverts). That
spread is what makes the diversion response continuous and monotone rather
than a step function.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from world.geometry import (
    RING_INDEX,
    RING_ORDER,
    RING_ZONES,
    ZONE_INDEX,
    access_minutes,
    crosses_cordon,
    is_corridor_od,
    is_water_crossing,
)
from world.tolling import PERIODS

if TYPE_CHECKING:  # avoid a runtime import cycle; config imports population
    from world.config import WorldConfig

# Five-mode vocabulary, frozen order (matches grounding.taxonomy.MODES; kept
# local so world/ stays self-contained).
MODES = ("walk", "transit", "ride", "car", "bike")
MODE_INDEX = {m: i for i, m in enumerate(MODES)}
_CAR = MODE_INDEX["car"]
_RIDE = MODE_INDEX["ride"]


@dataclass
class AgentPopulation:
    """Column-oriented (numpy) population. One row per agent-day.

    Zone / ring / mode / period are integer codes (see geometry / tolling);
    vot is credits per minute; has_pass is boolean. The derived masks
    (is_corridor / is_water / is_car_or_ride) are precomputed once."""

    home_zone: np.ndarray
    work_zone: np.ndarray
    home_ring: np.ndarray
    work_ring: np.ndarray
    mode: np.ndarray
    vot: np.ndarray
    has_pass: np.ndarray
    period: np.ndarray
    access: np.ndarray            # corridor off-facility access minutes
    is_corridor: np.ndarray       # bool: OD traverses the corridor
    is_water: np.ndarray          # bool: OD crosses the water barrier
    is_car_or_ride: np.ndarray    # bool: mode is car or ride

    def __len__(self) -> int:
        return int(self.home_zone.shape[0])

    @property
    def corridor_travelers(self) -> np.ndarray:
        """Mask of agents who make a route choice on the corridor: a corridor
        OD travelled by car or ride. Everyone else uses static times."""
        return self.is_corridor & self.is_car_or_ride

    @property
    def water_travelers(self) -> np.ndarray:
        """Mask of car/ride agents whose trip uses the water crossing."""
        return self.is_water & self.is_car_or_ride


def _draw_rings(rng: np.random.Generator, weights, n: int) -> np.ndarray:
    """Draw ring codes for n agents from a {ring: weight} mapping."""
    probs = np.array([weights[r] for r in RING_ORDER], dtype=float)
    probs /= probs.sum()
    return rng.choice(len(RING_ORDER), size=n, p=probs).astype(np.int16)


def _rings_to_zones(rng: np.random.Generator, ring_codes: np.ndarray) -> np.ndarray:
    """Pick a uniform-random zone within each agent's ring."""
    zone_of = np.empty(ring_codes.shape[0], dtype=np.int16)
    for ring_code, ring in enumerate(RING_ORDER):
        mask = ring_codes == ring_code
        count = int(mask.sum())
        if count == 0:
            continue
        zone_idxs = np.array([ZONE_INDEX[z] for z in RING_ZONES[ring]])
        zone_of[mask] = rng.choice(zone_idxs, size=count).astype(np.int16)
    return zone_of


def build_population(config: "WorldConfig", seed: int, n_agents: int = 10_000) -> AgentPopulation:
    """Deterministically build ``n_agents`` scripted agent-days for a config.

    Draw order is fixed (home ring, home zone, work ring, work zone, mode,
    value of time, pass, period) so a given (config, seed, n_agents) is
    bit-identical across runs and machines.
    """
    rng = np.random.default_rng(seed)

    home_ring = _draw_rings(rng, config.ring_population_weights, n_agents)
    home_zone = _rings_to_zones(rng, home_ring)
    work_ring = _draw_rings(rng, config.ring_workplace_weights, n_agents)
    work_zone = _rings_to_zones(rng, work_ring)

    mode_probs = np.array([config.mode_distribution[m] for m in MODES], dtype=float)
    mode_probs /= mode_probs.sum()
    mode = rng.choice(len(MODES), size=n_agents, p=mode_probs).astype(np.int16)

    # Per-agent value of time (credits/minute), lognormal heterogeneity.
    vot = rng.lognormal(mean=np.log(config.vot_median), sigma=config.vot_sigma,
                        size=n_agents)

    has_pass = rng.random(n_agents) < config.pass_prior

    period_probs = np.array([config.period_distribution[p] for p in PERIODS], dtype=float)
    period_probs /= period_probs.sum()
    period = rng.choice(len(PERIODS), size=n_agents, p=period_probs).astype(np.int16)

    is_corridor = is_corridor_od(home_ring, work_ring)
    is_water = is_water_crossing(home_ring, work_ring)
    is_car_or_ride = (mode == _CAR) | (mode == _RIDE)
    access = access_minutes(home_zone, work_zone)

    return AgentPopulation(
        home_zone=home_zone,
        work_zone=work_zone,
        home_ring=home_ring,
        work_ring=work_ring,
        mode=mode,
        vot=vot,
        has_pass=has_pass,
        period=period,
        access=access,
        is_corridor=is_corridor,
        is_water=is_water,
        is_car_or_ride=is_car_or_ride,
    )


def cordon_crossing_mask(population: AgentPopulation, cordon_rings) -> np.ndarray:
    """Car/ride agents whose trip crosses into the cordon rings (cityk_cordon
    instrument). Cordon rings are given as ring-name strings."""
    codes = tuple(RING_INDEX[r] for r in cordon_rings)
    crossing = crosses_cordon(population.home_ring, population.work_ring, codes)
    return crossing & population.is_car_or_ride
