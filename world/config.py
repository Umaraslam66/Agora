"""World configs for the masked corridor world (M1).

Two configs ship in world/ and share ALL machinery (geometry, VDF, toll
schedule, population synthesis, the equilibrium loop). They differ only in
CONFIG DATA — the policy instrument and its parameters:

  * ``cityk_corridor`` — the M1 novelty: a north-south corridor of parallel
    facilities where one (the tunnel) is tolled from a shock day, and route
    diversion is the binding margin.
  * ``cityk_cordon``  — the v1 world shape kept for the transfer arena: the
    instrument is a cordon fee for crossing into the central rings
    ("core" + "inner"), not a tolled link. Minimal but runnable on the same
    machinery.

Keeping both on one code path is the same discipline as the render-parity
doctrine (00_PROJECT_BRIEF.md, "one render path, ever"): a second copy of the
world machinery is exactly how two configs silently drift apart. Every numeric
value here is a documented DEV placeholder (masked/pre-perturbed), not a
calibration target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from world.network import ERA_LABELS, Facility, NetworkState, era_index_for_day
from world.tolling import TollSchedule, cordon_schedule, default_schedule

# ---------------------------------------------------------------------------
# Shared facility set (BPR params). Capacity and (alpha, beta) encode the
# spec's ordering: freeway bypass F (fast when empty, low t0) and core-grid D
# (slow, lowest capacity of the corridor facilities) collapse under load;
# tunnel T and surface arterial S are milder; the elevated spine V and the
# water crossing W are separate. All facilities use the classic BPR beta = 4.
# An earlier revision of this file ran F and D at beta = 3, purely because the
# solver's fixed damping factor (0.5) made the assign/retime loop oscillate at
# beta = 4 on these two steep curves. That was a numerical-convenience
# compromise, not a modeling choice, and it has been retired: world/network.py
# now solves the corridor fixed point by MSA (method of successive averages,
# shrinking step 1/n) instead of fixed damping, which converges at beta = 4 on
# every facility within the iteration budget (see solve_corridor_equilibrium's
# docstring). alpha still makes F and D collapse hard under load (at v/c =
# 1.5, F time inflates ~4.4x, D ~5.4x).
#
# beta itself remains a WORLD PARAMETER, not a fixed physical constant: 4 is
# the standard BPR default from the transportation-planning literature, used
# here as a DEV placeholder pending calibration against corridor observations
# at M2. The calibration-window network shocks (population.py / the shock-day
# machinery) give volume/time pairs per facility that should be used to fit
# alpha and beta per facility rather than assume the textbook value.
# ---------------------------------------------------------------------------
_FACILITIES: Dict[str, Facility] = {
    # code  t0   cap   alpha  beta
    "T": Facility("T", 8.0, 1000.0, 0.15, 4.0),   # tunnel: fast, mild curve
    "S": Facility("S", 15.0, 950.0, 0.25, 4.0),   # surface arterial: slow, mild
    "F": Facility("F", 9.0, 950.0, 1.00, 4.0),    # freeway bypass: fast-empty, steep
    "D": Facility("D", 13.0, 800.0, 1.30, 4.0),   # core street grid: slow, capacity-poor
    "V": Facility("V", 8.0, 1700.0, 0.20, 4.0),   # elevated spine: fast, high capacity
}
_WATER_FACILITY = Facility("W", 10.0, 700.0, 1.20, 4.0)  # water crossing: steep


@dataclass(frozen=True)
class WorldConfig:
    """All data that distinguishes one world from another. The machinery in
    network.py / population.py / simulation.py reads only this object."""

    name: str
    policy_instrument: str  # "corridor_toll" | "cordon"

    # Network geometry / facilities (shared object set by default).
    facilities: Mapping[str, Facility]
    water_facility: Facility

    # Scripted network timeline (day-index eras). era_facility_codes maps an
    # era index (0..3) to the corridor facilities present; era_tolled maps it
    # to the tolled facility code (or None). Boundaries are three ascending
    # day indices (arbitrary epoch — never real calendar dates).
    era_facility_codes: Mapping[int, Tuple[str, ...]]
    era_tolled: Mapping[int, Optional[str]]
    era_boundaries: Tuple[int, int, int]

    # Policy pricing.
    toll_schedule: TollSchedule
    cordon_rings: Tuple[str, ...]  # () for corridor; ("core","inner") for cordon

    # Population synthesis.
    ring_population_weights: Mapping[str, float]
    ring_workplace_weights: Mapping[str, float]
    mode_distribution: Mapping[str, float]
    period_distribution: Mapping[str, float]
    pass_prior: float
    vot_median: float
    vot_sigma: float

    # Route-choice logit sensitivity (per minute of generalized cost).
    logit_theta: float

    def network_state_for_day(
        self, day_index: int, schedule: Optional[TollSchedule] = None
    ) -> NetworkState:
        """The raw NetworkState an agent perceives on a day. Consumers get
        facilities + which one charges + its schedule — never the era name.
        An override schedule (e.g. an E5 price-swept one) may be supplied."""
        era = era_index_for_day(day_index, self.era_boundaries)
        tolled = self.era_tolled.get(era)
        toll_schedule = None
        if tolled is not None:
            toll_schedule = schedule if schedule is not None else self.toll_schedule
        return NetworkState(
            facility_codes=self.era_facility_codes[era],
            tolled_facility=tolled,
            toll_schedule=toll_schedule,
        )

    def era_label_for_day(self, day_index: int) -> str:
        """Human-facing era label (DEMO ONLY — never handed to an agent)."""
        return ERA_LABELS[era_index_for_day(day_index, self.era_boundaries)]

    def facility(self, code: str) -> Facility:
        if code == self.water_facility.code:
            return self.water_facility
        return self.facilities[code]


# ---------------------------------------------------------------------------
# Shared population defaults (DEV placeholders). Home weights spread residents
# to the outer rings; workplace weights concentrate jobs centrally, so many
# outer-north<->outer-south and core-crossing commutes ride the corridor.
# ---------------------------------------------------------------------------
_RING_POPULATION_WEIGHTS = {
    "core": 0.12,
    "inner": 0.30,
    "outer_north": 0.22,
    "outer_south": 0.22,
    "east_water": 0.14,
}
_RING_WORKPLACE_WEIGHTS = {
    "core": 0.34,
    "inner": 0.30,
    "outer_north": 0.14,
    "outer_south": 0.14,
    "east_water": 0.08,
}
_MODE_DISTRIBUTION = {
    "walk": 0.15,
    "transit": 0.28,
    "ride": 0.12,
    "car": 0.37,
    "bike": 0.08,
}
_PERIOD_DISTRIBUTION = {
    "overnight": 0.10,
    "am_peak": 0.28,
    "pm_peak": 0.24,
    "offpeak": 0.38,
}

# Era boundaries: arbitrary day indices from an arbitrary epoch (NOT dates).
_ERA_BOUNDARIES = (90, 180, 300)

# Value of time: median 0.2 credits/minute, lognormal spread. With toll rates
# ~1-2.4 credits this puts the toll at roughly a 5-15 minute penalty for a
# median driver — responsive but not overwhelming vs the 8-16 minute facility
# times, which is what makes diversion continuous and monotone.
_VOT_MEDIAN = 0.20
_VOT_SIGMA = 0.5

# Logit sensitivity (per minute of generalized cost). Lower theta spreads the
# route choice a little more, which keeps the MSA fixed point well-behaved
# while still giving a clearly monotone diversion response.
_LOGIT_THETA = 0.10


def _corridor_timeline() -> Tuple[Dict[int, Tuple[str, ...]], Dict[int, Optional[str]]]:
    """The four-era corridor script (see world plan / network.py ERA_LABELS):
    era0 elevated (V free, T absent); era1 squeeze (V gone, T not yet open);
    era2 free_tunnel (T open, toll off); era3 toll_on (T tolled)."""
    era_facility_codes = {
        0: ("V", "S", "F", "D"),
        1: ("S", "F", "D"),
        2: ("T", "S", "F", "D"),
        3: ("T", "S", "F", "D"),
    }
    era_tolled = {0: None, 1: None, 2: None, 3: "T"}
    return era_facility_codes, era_tolled


def _base_kwargs() -> dict:
    """Shared config fields both worlds start from."""
    era_facility_codes, era_tolled = _corridor_timeline()
    return dict(
        facilities=_FACILITIES,
        water_facility=_WATER_FACILITY,
        era_facility_codes=era_facility_codes,
        era_tolled=era_tolled,
        era_boundaries=_ERA_BOUNDARIES,
        toll_schedule=default_schedule(),
        ring_population_weights=_RING_POPULATION_WEIGHTS,
        ring_workplace_weights=_RING_WORKPLACE_WEIGHTS,
        mode_distribution=_MODE_DISTRIBUTION,
        period_distribution=_PERIOD_DISTRIBUTION,
        pass_prior=0.75,
        vot_median=_VOT_MEDIAN,
        vot_sigma=_VOT_SIGMA,
        logit_theta=_LOGIT_THETA,
    )


def cityk_corridor() -> WorldConfig:
    """The M1 corridor world: tolled tunnel with free substitutes."""
    return WorldConfig(
        name="cityk_corridor",
        policy_instrument="corridor_toll",
        cordon_rings=(),
        **_base_kwargs(),
    )


def cityk_cordon() -> WorldConfig:
    """The v1 world shape (transfer arena): a cordon fee to cross into the
    central rings, on the same machinery. The corridor timeline is still
    present (shared machinery) but is never tolled; the cordon is the
    instrument."""
    kwargs = _base_kwargs()
    kwargs["era_tolled"] = {0: None, 1: None, 2: None, 3: None}
    # The cordon charges the masked A8.5(i) schedule (surcharge-free); the
    # BT2 driver turns phases on/off via with_multiplier, never by editing
    # rates. pass_prior is structurally zero — no pass instrument exists in
    # this arena (A8.5(ii)).
    kwargs["toll_schedule"] = cordon_schedule()
    kwargs["pass_prior"] = 0.0
    return WorldConfig(
        name="cityk_cordon",
        policy_instrument="cordon",
        cordon_rings=("core", "inner"),
        **kwargs,
    )


_CONFIG_BUILDERS = {
    "cityk_corridor": cityk_corridor,
    "cityk_cordon": cityk_cordon,
}


def get_config(name: str) -> WorldConfig:
    """Look up a named world config."""
    if name not in _CONFIG_BUILDERS:
        raise KeyError(f"unknown world config {name!r}; known: {sorted(_CONFIG_BUILDERS)}")
    return _CONFIG_BUILDERS[name]()
