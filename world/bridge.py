"""The card -> world bridge (M3 D1): route a persona card's realized trips
through the masked corridor world.

WHY THIS FILE EXISTS — through M2 the scorers consumed ``card_executor``
output directly; nothing ever pushed a card's trips through the corridor's
route choice and congestion (world/network.py). M3 needs that coupling so the
fast brain's realized commute times come from the SAME equilibrium physics the
world uses, feeding the surprise signal that triggers a slow-brain rewrite
(and, from M4, toll exposure). This module is that seam and nothing else: it
maps a card population onto an :class:`world.population.AgentPopulation`,
selects each persona-day's corridor commute traveler, and supplies the
bias-corrected expectation the surprise hook compares against.

It stays world-self-contained (population.py's discipline): it imports only
world/ + numpy/scipy + stdlib, and operates on cards as plain dicts and on the
habit memory by duck-typed attribute access (``.cell`` / ``.config``) — never
importing agents/, serving/, or grounding/. No real place name, agency, date,
or price appears in any literal or comment here (mask-lint gate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtri  # inverse standard-normal CDF (uniform -> z)

from world import crn
from world.config import WorldConfig
from world.geometry import (
    RING_INDEX,
    ZONE_INDEX,
    ZONE_RING,
    access_minutes,
    is_corridor_od,
    is_water_crossing,
)
from world.population import AgentPopulation
from world.tolling import PERIODS, PERIOD_INDEX

# ---------------------------------------------------------------------------
# Vocabulary alignment: card departure band -> world toll period
# ---------------------------------------------------------------------------
# The card vocabulary (grounding.adapters.psrc.TIME_BANDS, five bands) and the
# world period vocabulary (world.tolling.PERIODS, four bands) both partition the
# day by hour, but along different cut points:
#
#   card band (psrc.time_band):   night<07:00  am_peak 07:00-08:59
#                                 midday 09:00-15:59  pm_peak 16:00-17:59
#                                 evening >=18:00
#   world period (period_for_hour): overnight 21:00-04:59  am_peak 06:00-08:59
#                                 pm_peak 15:00-17:59  offpeak otherwise
#
# Each card band is mapped to the world period holding the BULK of its hours:
#   night   -> overnight (pre-dawn 00:00-04:59 dominates the band)
#   am_peak -> am_peak   (07:00-08:59 sits entirely inside world am_peak)
#   midday  -> offpeak   (09:00-14:59 is world offpeak; only 15:xx spills to pm)
#   pm_peak -> pm_peak   (16:00-17:59 sits inside world pm_peak)
#   evening -> offpeak   (18:00-20:59 is world offpeak; 21:xx+ spills overnight)
#
# The period only re-enters the physics through the toll term (a per-period
# rate), so at an untolled ordinary-day corridor it affects only the surprise
# context key's grouping; it becomes load-bearing under M4 tolling.
_BAND_TO_PERIOD: Dict[str, str] = {
    "night": "overnight",
    "am_peak": "am_peak",
    "midday": "offpeak",
    "pm_peak": "pm_peak",
    "evening": "offpeak",
}

#: Fallback world period for an unrecognized band vocabulary (defensive; the
#: fast brain only ever emits the five card bands above, but a rule's overridden
#: ``depart_band`` could in principle carry a world-period token directly).
_DEFAULT_PERIOD = "offpeak"


def period_code_for_band(band: Optional[str]) -> int:
    """World toll-period integer code (``PERIOD_INDEX``) for a card departure
    band. Card bands map per :data:`_BAND_TO_PERIOD`; a token that is already a
    world period passes through; anything else falls back to ``offpeak``."""
    if band in _BAND_TO_PERIOD:
        return PERIOD_INDEX[_BAND_TO_PERIOD[band]]
    if band in PERIOD_INDEX:  # already a world-period token
        return PERIOD_INDEX[band]
    return PERIOD_INDEX[_DEFAULT_PERIOD]


# ---------------------------------------------------------------------------
# Value-of-time: per-persona lognormal draw x deterministic income ladder
# ---------------------------------------------------------------------------
# The base draw reuses the config's lognormal (median ``vot_median``, spread
# ``vot_sigma``) but is CRN-keyed per persona so it is paired across arms and
# independent across ensemble runs (namespace), rather than the population's
# fixed-seed numpy draw. The income ladder is a monotone-in-income multiplier:
# higher income -> higher value of time -> the toll bites less and the fast
# facility is kept, exactly the heterogeneity that makes corridor diversion
# continuous (world/population.py). Class 3 (the low end of the "mid" income
# band, grounding.taxonomy) is the 1.0 anchor; income_class None ("prefer not
# to answer") also anchors at 1.0. Every value is a documented DEV placeholder,
# not a calibration target (world/config.py doctrine).
INCOME_VOT_MULTIPLIER: Dict[int, float] = {
    1: 0.80,
    2: 0.90,
    3: 1.00,
    4: 1.15,
    5: 1.35,
}
_DEFAULT_VOT_MULTIPLIER = 1.0  # income_class None -> mid anchor

# Clamp CRN uniforms off the open-interval endpoints before ndtri (a uniform of
# exactly 0.0 or 1.0 would map to +-inf); the sha256-derived draws never hit an
# endpoint in practice, so the clamp only guards the degenerate corner.
_U_EPS = 1e-12


def _income_multiplier(income_class) -> float:
    if income_class is None:
        return _DEFAULT_VOT_MULTIPLIER
    try:
        ic = int(income_class)
    except (TypeError, ValueError):
        return _DEFAULT_VOT_MULTIPLIER
    return INCOME_VOT_MULTIPLIER.get(ic, _DEFAULT_VOT_MULTIPLIER)


def _draw_vot_array(
    persona_ids: Sequence[str],
    income_classes: Sequence,
    config: WorldConfig,
    namespace: str,
) -> np.ndarray:
    """Vectorized per-persona value of time (credits/minute).

    ``vot_i = lognormal(median, sigma; u_i) * ladder(income_class_i)`` where
    ``u_i`` is the persona's CRN uniform under key ``"{namespace}:{pid}:vot"``.
    Deterministic and paired-by-namespace with every other CRN draw."""
    keys = ["%s:%s:vot" % (namespace, pid) for pid in persona_ids]
    u = crn.draws(keys)
    u = np.clip(u, _U_EPS, 1.0 - _U_EPS)
    z = ndtri(u)
    base = np.exp(math.log(config.vot_median) + config.vot_sigma * z)
    ladder = np.array([_income_multiplier(ic) for ic in income_classes], dtype=float)
    return base * ladder


# ---------------------------------------------------------------------------
# Representative per-persona mode / period (population fields; the loop overrides
# with the realized per-day values, so these are documented representatives)
# ---------------------------------------------------------------------------

def _representative_mode(card: Mapping) -> str:
    """A persona's representative commute mode for the AgentPopulation row:
    ``car`` if any pattern trip drives, else ``ride`` if any is a ride, else the
    first pattern trip's mode, else ``car``. The loop never reads this (it uses
    the realized per-day mode); it exists so ``AgentPopulation`` is complete and
    ``corridor_travelers`` is meaningful for M4 static-population callers."""
    has_ride = False
    first_mode: Optional[str] = None
    for pattern in card.get("patterns", []):
        for trip in pattern.get("trips", []):
            m = trip.get("mode")
            if first_mode is None:
                first_mode = m
            if m == "car":
                return "car"
            if m == "ride":
                has_ride = True
    if has_ride:
        return "ride"
    return first_mode or "car"


def _representative_period_code(card: Mapping) -> int:
    """Representative period code from the persona's first work-purpose trip's
    departure band (else the first trip's band, else ``am_peak``)."""
    first_band: Optional[str] = None
    for pattern in card.get("patterns", []):
        for trip in pattern.get("trips", []):
            if first_band is None:
                first_band = trip.get("depart_band")
            if trip.get("purpose") == "work":
                return period_code_for_band(trip.get("depart_band"))
    if first_band is not None:
        return period_code_for_band(first_band)
    return PERIOD_INDEX["am_peak"]


# ---------------------------------------------------------------------------
# population_from_cards (D1)
# ---------------------------------------------------------------------------

# Frozen mode order (matches world.population.MODES) for the representative
# mode index; kept local so the bridge stays world-self-contained.
_MODES: Tuple[str, ...] = ("walk", "transit", "ride", "car", "bike")
_MODE_INDEX = {m: i for i, m in enumerate(_MODES)}


def population_from_cards(
    cards: Sequence[Mapping], config: WorldConfig, namespace: str
) -> AgentPopulation:
    """Map a card population onto an :class:`AgentPopulation` (D1).

    One row per card, in card order. ``home_zone`` / ``work_zone`` / ``has_pass``
    come straight from the skeleton; ``vot`` is the CRN-keyed lognormal-times-
    income-ladder draw (:func:`_draw_vot_array`); ``period`` and ``mode`` are
    documented per-persona representatives (the loop overrides them with the
    realized per-day trip); corridor / water membership is the config's OD
    geometry over (home_ring, work_ring).

    Zones outside the world's Z01..Z30 codes (e.g. the ``Z00`` work-zone
    placeholder some non-commuter skeletons carry) map to an invalid ring and
    are forced OFF the corridor and water masks — a persona with no defined
    commute OD is never a route-choice traveler. Their access is a harmless
    clamped value that the equilibrium never reads (they are masked out)."""
    n = len(cards)
    persona_ids: List[str] = []
    income_classes: List = []
    home_zone_idx = np.full(n, -1, dtype=np.int64)
    work_zone_idx = np.full(n, -1, dtype=np.int64)
    home_ring = np.full(n, -1, dtype=np.int64)
    work_ring = np.full(n, -1, dtype=np.int64)
    has_pass = np.zeros(n, dtype=bool)
    mode = np.zeros(n, dtype=np.int16)
    period = np.zeros(n, dtype=np.int16)

    for i, card in enumerate(cards):
        skeleton = card.get("skeleton", {})
        persona_ids.append(str(card["persona_id"]))
        income_classes.append(skeleton.get("income_class"))
        hz = skeleton.get("home_zone")
        wz = skeleton.get("work_zone")
        home_zone_idx[i] = ZONE_INDEX.get(hz, -1)
        work_zone_idx[i] = ZONE_INDEX.get(wz, -1)
        home_ring[i] = RING_INDEX.get(ZONE_RING.get(hz), -1)
        work_ring[i] = RING_INDEX.get(ZONE_RING.get(wz), -1)
        has_pass[i] = bool(skeleton.get("has_pass") or False)
        mode[i] = _MODE_INDEX.get(_representative_mode(card), _MODE_INDEX["car"])
        period[i] = _representative_period_code(card)

    valid_od = (home_ring >= 0) & (work_ring >= 0)
    home_ring_safe = np.where(home_ring >= 0, home_ring, 0)
    work_ring_safe = np.where(work_ring >= 0, work_ring, 0)
    is_corridor = is_corridor_od(home_ring_safe, work_ring_safe) & valid_od
    is_water = is_water_crossing(home_ring_safe, work_ring_safe) & valid_od

    home_zone_safe = np.where(home_zone_idx >= 0, home_zone_idx, 0)
    work_zone_safe = np.where(work_zone_idx >= 0, work_zone_idx, 0)
    access = access_minutes(home_zone_safe, work_zone_safe)

    vot = _draw_vot_array(persona_ids, income_classes, config, namespace)
    is_car_or_ride = (mode == _MODE_INDEX["car"]) | (mode == _MODE_INDEX["ride"])

    return AgentPopulation(
        home_zone=home_zone_safe.astype(np.int16),
        work_zone=work_zone_safe.astype(np.int16),
        home_ring=home_ring_safe.astype(np.int16),
        work_ring=work_ring_safe.astype(np.int16),
        mode=mode,
        vot=vot,
        has_pass=has_pass,
        period=period,
        access=access,
        is_corridor=is_corridor,
        is_water=is_water,
        is_car_or_ride=is_car_or_ride,
    )


def persona_row_index(cards: Sequence[Mapping]) -> Dict[str, int]:
    """persona_id -> row index into a :func:`population_from_cards` result (card
    order). The population arrays carry no ids; this is the join back to them."""
    return {str(card["persona_id"]): i for i, card in enumerate(cards)}


# ---------------------------------------------------------------------------
# corridor_travelers_of_day (D1)
# ---------------------------------------------------------------------------

@dataclass
class TravelerTable:
    """The corridor car/ride travelers selected from a set of realized days.

    One row per selected persona-day, in a deterministic (sorted persona id,
    then given day order) order so the CRN route keys and the realized-facility
    tally are reproducible. All per-traveler arrays are parallel to
    ``persona_ids``."""

    persona_ids: List[str]
    day_index: List[int]
    mode: List[str]
    period_codes: np.ndarray  # PERIOD_INDEX codes
    vot: np.ndarray
    has_pass: np.ndarray  # bool
    access: np.ndarray
    row_index: np.ndarray  # population row per traveler

    def __len__(self) -> int:
        return len(self.persona_ids)


def _first_car_or_ride(trips) -> Optional[object]:
    """The FIRST car/ride trip of a realized day, or None. This is the
    peak-direction commute pin (D1): a persona-day contributes at most one
    corridor traveler, taken as its first driven/ridden trip in realized
    order — a stable proxy for the home<->work commute leg."""
    for trip in trips:
        if getattr(trip, "mode", None) in ("car", "ride"):
            return trip
    return None


def corridor_travelers_of_day(
    realized_days_by_persona: Mapping[str, Sequence],
    cards: Sequence[Mapping],
    config: WorldConfig,
    *,
    population: Optional[AgentPopulation] = None,
    row_index: Optional[Mapping[str, int]] = None,
    namespace: str = "m3_bridge",
) -> TravelerTable:
    """Select each persona-day's corridor commute traveler (D1).

    A persona-day is a traveler iff (i) the persona's skeleton OD is a corridor
    OD (``population.is_corridor``) and (ii) the realized day has at least one
    car/ride trip; the FIRST such trip pins the traveler's departure period
    (this is the documented peak-direction commute pin). Returns a
    :class:`TravelerTable` carrying (persona_id, day_index, mode, period, vot,
    has_pass, access) per traveler.

    ``population`` / ``row_index`` may be supplied by the loop (built once per
    run, cheap to reuse); when omitted they are built from ``cards`` under
    ``namespace`` — note ``vot`` then depends on ``namespace``, so a caller that
    needs run-paired value of time must pass the namespace it built the
    population under."""
    if population is None:
        population = population_from_cards(cards, config, namespace)
    if row_index is None:
        row_index = persona_row_index(cards)

    persona_ids: List[str] = []
    day_index: List[int] = []
    modes: List[str] = []
    rows: List[int] = []
    period_list: List[int] = []

    for pid in sorted(realized_days_by_persona):
        row = row_index.get(pid)
        if row is None or not bool(population.is_corridor[row]):
            continue
        for day in realized_days_by_persona[pid]:
            trip = _first_car_or_ride(day.trips)
            if trip is None:
                continue
            persona_ids.append(pid)
            day_index.append(int(day.day_index))
            modes.append(trip.mode)
            rows.append(row)
            period_list.append(period_code_for_band(getattr(trip, "depart_band", None)))

    rows_arr = np.array(rows, dtype=np.int64)
    if len(rows_arr) == 0:
        return TravelerTable(
            persona_ids=[], day_index=[], mode=[],
            period_codes=np.zeros(0, dtype=np.int64),
            vot=np.zeros(0), has_pass=np.zeros(0, dtype=bool), access=np.zeros(0),
            row_index=rows_arr,
        )
    return TravelerTable(
        persona_ids=persona_ids,
        day_index=day_index,
        mode=modes,
        period_codes=np.array(period_list, dtype=np.int64),
        vot=population.vot[rows_arr],
        has_pass=population.has_pass[rows_arr],
        access=population.access[rows_arr],
        row_index=rows_arr,
    )


# ---------------------------------------------------------------------------
# expected_minutes (D2): bias-corrected EMA expectation for the surprise hook
# ---------------------------------------------------------------------------

def expected_minutes(memory, key: str, freeflow: float) -> float:
    """Expected minutes for a context ``key`` (D2).

    Free-flow (the config baseline) when the cell is absent or has ``n == 0``;
    otherwise the BIAS-CORRECTED EMA ``ema_realized / (1 - (1-alpha)**n)``. The
    ported EMA seeds at 0.0 (agents.habit_memory exactness rule forbids touching
    it), so after ``n`` observations of a constant ``x`` the raw EMA is only
    ``x*(1-(1-alpha)**n)``; dividing it out recovers ``x`` and prevents ~10 days
    of phantom low expectations that would manufacture surprises. ``alpha`` is
    read from the memory's own config (0.7 = 1 - ALPHA_BASE is NOT hardcoded).
    """
    cell = memory.cell(key)
    if cell is None or cell.n == 0:
        return freeflow
    alpha = memory.config.alpha_base
    denom = 1.0 - (1.0 - alpha) ** cell.n
    if denom <= 0.0:  # alpha == 0 degenerate guard (EMA never moves off its seed)
        return freeflow
    return cell.ema_realized_minutes / denom
