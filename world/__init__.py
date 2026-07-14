"""World layer — masked, scripted simulation worlds for Agora (M1).

M1 is the "City K v2" masked corridor world with SCRIPTED agents only: no LLM
anywhere (that arrives in later milestones). The novelty over the v1 cordon
world is corridor ROUTE CHOICE — a tolled tunnel with free substitutes, where
diversion is the binding margin and everyone diverting makes the alternatives
worse (congestion feedback).

Doctrine this package upholds:
  * MASKING (01_PREREGISTRATION.md §5): every place is a code (Z01..Z30),
    money is "credits", time is day indices from an arbitrary epoch; no real
    place, agency, date, or currency appears anywhere in world/ (mask-lint).
  * SELF-CONTAINED: world/ imports nothing from agents/, serving/, grounding/,
    never torch or transformers, and never the quarantined truth series (the
    blind answer key must not be reachable from any world/ code path).
  * ONE CODE PATH: the two shipped configs (cityk_corridor, cityk_cordon)
    share ALL machinery and differ only in config data — the same discipline
    as the single render path (00_PROJECT_BRIEF.md, "one render path, ever").
  * CRN DETERMINISM: a population is one deterministic function of (config,
    seed); the same seed yields bit-identical facility loads.

Public surface: build a config, build a population, simulate a day.
"""
from __future__ import annotations

from world.config import WorldConfig, cityk_cordon, cityk_corridor, get_config
from world.network import (
    ERA_LABELS,
    EquilibriumResult,
    Facility,
    NetworkState,
    bpr_time,
    solve_corridor_equilibrium,
)
from world.population import AgentPopulation, build_population
from world.simulation import (
    DayResult,
    simulate_day,
    simulate_era,
    toll_multiplier_sweep,
)
from world.tolling import TollSchedule, default_schedule, period_for_hour

__all__ = [
    "WorldConfig",
    "get_config",
    "cityk_corridor",
    "cityk_cordon",
    "Facility",
    "NetworkState",
    "EquilibriumResult",
    "bpr_time",
    "solve_corridor_equilibrium",
    "ERA_LABELS",
    "AgentPopulation",
    "build_population",
    "DayResult",
    "simulate_day",
    "simulate_era",
    "toll_multiplier_sweep",
    "TollSchedule",
    "default_schedule",
    "period_for_hour",
]
