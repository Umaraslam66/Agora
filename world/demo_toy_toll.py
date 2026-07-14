"""Toy-toll demo for the masked corridor world (M1) — the orchestrator's
acceptance driver.

Run: ``python -m world.demo_toy_toll``

Prints, for the cityk_corridor world with 10k scripted agents:
  1. per-era corridor facility volumes and congested times (the network
     timeline: elevated -> squeeze -> free_tunnel -> toll_on);
  2. a toll-multiplier diversion table at the tolled era (the E5 price-sweep
     hook), showing the tunnel bleeding volume onto the free substitutes as
     the price rises, with total corridor volume conserved; and
  3. the wall-clock time for one full simulated day INCLUDING the equilibrium
     loop.

Everything printed is generic (facility codes, "credits", day indices) — no
real place, agency, date, or currency (01_PREREGISTRATION.md §5). This module
only prints numbers the simulation produced; there is no second world here.
"""
from __future__ import annotations

import time

from world.config import get_config
from world.population import build_population
from world.simulation import (
    DayResult,
    simulate_day,
    simulate_era,
    toll_multiplier_sweep,
)

_SEED = 20260714
_N_AGENTS = 10_000
_SWEEP = (0.0, 0.5, 1.0, 2.0, 4.0)
_CORRIDOR_CODES = ("V", "T", "S", "F", "D")


def _fmt_cell(result: DayResult, code: str, key: str) -> str:
    table = result.facility_loads if key == "load" else result.facility_times
    if code not in table:
        return f"{'-':>8}"
    return f"{table[code]:8.0f}" if key == "load" else f"{table[code]:8.1f}"


def _print_era_table(config, population) -> None:
    print("Per-era corridor facilities (volumes then congested minutes)")
    header = "  era  label        " + "".join(f"{c:>8}" for c in _CORRIDOR_CODES)
    print(header + "   mean_dd  iters  conv")
    for era in range(4):
        r = simulate_era(config, population, era)
        loads = "".join(_fmt_cell(r, c, "load") for c in _CORRIDOR_CODES)
        label = config.era_label_for_day(_era_day(config, era))
        print(f"  {era:>3}  {label:<11} {loads}   {r.mean_door_to_door:7.2f}"
              f"  {r.iterations:>5}  {str(r.converged):>5}")
    print("  (times)")
    for era in range(4):
        r = simulate_era(config, population, era)
        times = "".join(_fmt_cell(r, c, "time") for c in _CORRIDOR_CODES)
        label = config.era_label_for_day(_era_day(config, era))
        print(f"  {era:>3}  {label:<11} {times}")


def _era_day(config, era: int) -> int:
    b0, b1, b2 = config.era_boundaries
    return {0: 0, 1: b0, 2: b1, 3: b2}[era]


def _print_sweep_table(config, population) -> None:
    print("\nToll-multiplier diversion table (tolled era)")
    print("  mult      T       S       F       D    S+F+D    total   T_time  F_time  D_time  conv")
    sweep = toll_multiplier_sweep(config, population, _SWEEP)
    for m, r in sweep:
        L = r.facility_loads
        others = L["S"] + L["F"] + L["D"]
        print(f"  {m:>4}  {L['T']:6.0f}  {L['S']:6.0f}  {L['F']:6.0f}  {L['D']:6.0f}"
              f"  {others:7.0f}  {r.total_corridor_volume():7.0f}"
              f"  {r.facility_times['T']:6.1f}  {r.facility_times['F']:6.1f}"
              f"  {r.facility_times['D']:6.1f}  {str(r.converged):>5}")


def _time_full_day(config, population) -> float:
    state = config.network_state_for_day(_era_day(config, 3))
    start = time.perf_counter()
    simulate_day(config, population, state)
    return time.perf_counter() - start


def main() -> int:
    config = get_config("cityk_corridor")

    build_start = time.perf_counter()
    population = build_population(config, _SEED, _N_AGENTS)
    build_seconds = time.perf_counter() - build_start

    n_corridor = int(population.corridor_travelers.sum())
    print(f"World: {config.name}  |  agents: {len(population):,}  |  "
          f"car/ride corridor travellers: {n_corridor:,}")
    print(f"Population build: {build_seconds:.3f} s\n")

    _print_era_table(config, population)
    _print_sweep_table(config, population)

    day_seconds = _time_full_day(config, population)
    print(f"\nOne full simulated day (10k agents, incl. equilibrium loop): "
          f"{day_seconds:.4f} s")
    print(f"Population build + one day: {build_seconds + day_seconds:.4f} s")

    # A quick look at the transfer-arena cordon world on the same machinery.
    cordon = get_config("cityk_cordon")
    cordon_pop = build_population(cordon, _SEED, _N_AGENTS)
    cordon_day = simulate_day(cordon, cordon_pop, cordon.network_state_for_day(400))
    print(f"\nTransfer arena ({cordon.name}): charged cordon crossings "
          f"{cordon_day.cordon_charged_trips:,}, revenue "
          f"{cordon_day.cordon_revenue:,.0f} credits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
