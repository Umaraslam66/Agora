"""Masked corridor-world geometry — zones, rings, and a deterministic
synthetic geography (M1, scripted-agents only; no LLM anywhere).

WHY THIS FILE EXISTS (01_PREREGISTRATION.md §5 masking; 00_PROJECT_BRIEF.md
"The world"): the world layer needs a network whose TOPOLOGY reproduces what
matters for the anchoring natural experiment — a north-south spine through a
central core with a small set of parallel facilities, plus a water barrier
crossed only by capacity-limited links — WITHOUT being recognizable. So every
place is a bare code (Z01..Z30), every ring is a generic direction word
("core", "outer_north", ...), and the map itself is a pure function of the
zone code (like grounding/adapters/rvu_schema.py's zone_coordinates), never a
real map. No real place name, agency, calendar date, or currency amount
appears anywhere in world/ (mask-lint gate).

The ring vocabulary is chosen to interoperate with
grounding.taxonomy.CATCHMENT_RINGS = {"inner", "core"}: those two rings are
the priced facility's catchment, so the E1 protected-segment axis survives
the zone-taxonomy upgrade from grounding's Z01..Z25 codes to this world's
Z01..Z30 codes. world/ imports nothing from grounding (it must stay
self-contained per the M1 build order), so the ring strings are re-declared
here verbatim; a test may cross-check them against grounding if desired.

GEOMETRY (a documented DEV placeholder, not a calibrated map): the five rings
are laid out as concentric/directional clusters so that
  - "core" and "inner" sit at the centre,
  - "outer_north" / "outer_south" sit north/south of the core (the corridor
    runs between them, through the core), and
  - "east_water" sits across a water barrier to the east; any trip with
    exactly one end in east_water is a WATER CROSSING (a capacity-limited
    link, network.py), never a corridor trip.
Zone-to-zone base times are straight-line distance / mode speed. At M1 only
the car/ride corridor route choice is actually TIMED (congested, network.py);
the base-time helpers here are scaffolding for M2+, when non-corridor modes
get static travel-time matrices (transit's is a placeholder until a
GTFS-derived matrix lands — documented future work, not consumed yet).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Zones and rings (codes only — no real place names; see module docstring)
# ---------------------------------------------------------------------------

ZONE_COUNT = 30
ZONES: List[str] = [f"Z{i:02d}" for i in range(1, ZONE_COUNT + 1)]

# Frozen ring order. The strings are load-bearing: "core"/"inner" must equal
# grounding.taxonomy.CATCHMENT_RINGS members so the priced-facility catchment
# is the same set of segments across the two zone taxonomies.
RING_ORDER: Tuple[str, ...] = (
    "core",
    "inner",
    "outer_north",
    "outer_south",
    "east_water",
)

# Inclusive 1-based zone-number spans per ring (frozen): 6 + 8 + 6 + 6 + 4 = 30.
_RING_SPANS: Dict[str, Tuple[int, int]] = {
    "core": (1, 6),
    "inner": (7, 14),
    "outer_north": (15, 20),
    "outer_south": (21, 26),
    "east_water": (27, 30),
}

RING_ZONES: Dict[str, List[str]] = {
    ring: [f"Z{i:02d}" for i in range(lo, hi + 1)]
    for ring, (lo, hi) in _RING_SPANS.items()
}

ZONE_RING: Dict[str, str] = {
    zone: ring for ring, zones in RING_ZONES.items() for zone in zones
}

ZONE_INDEX: Dict[str, int] = {zone: i for i, zone in enumerate(ZONES)}
RING_INDEX: Dict[str, int] = {ring: i for i, ring in enumerate(RING_ORDER)}

# ---------------------------------------------------------------------------
# North-south bands (the corridor-classification primitive)
# ---------------------------------------------------------------------------
# Every ring collapses to one of four bands. The corridor runs north<->south
# THROUGH the central bands, so a trip traverses it exactly when its two ends
# lie in different non-east bands (see is_corridor_od). east is its own band:
# east trips cross the water, never the corridor.
_BAND_CENTRAL, _BAND_NORTH, _BAND_SOUTH, _BAND_EAST = 0, 1, 2, 3

# BAND[ring_code] -> band code. Indexable by the RING_ORDER integer codes so
# the classification vectorizes over a whole population's ring arrays.
BAND: np.ndarray = np.array(
    [
        _BAND_CENTRAL,  # core
        _BAND_CENTRAL,  # inner
        _BAND_NORTH,    # outer_north
        _BAND_SOUTH,    # outer_south
        _BAND_EAST,     # east_water
    ],
    dtype=np.int8,
)

# ---------------------------------------------------------------------------
# Synthetic geometry — deterministic (x, y) km per zone (DEV placeholder)
# ---------------------------------------------------------------------------
# (centre_x, centre_y, radius_km): zones in a ring sit evenly around its
# circle. Pure function of the zone code, so distances are reproducible.
_RING_GEOMETRY: Dict[str, Tuple[float, float, float]] = {
    "core": (0.0, 0.0, 3.0),
    "inner": (0.0, 0.0, 7.5),
    "outer_north": (0.0, 22.0, 6.0),
    "outer_south": (0.0, -22.0, 6.0),
    "east_water": (16.0, 0.0, 6.0),
}

# The water barrier is a vertical line: east_water (x in ~[10, 22]) sits east
# of it, every other ring (x in ~[-7.5, 7.5]) west of it. Used only to keep
# the synthetic distances internally consistent; the crossing test itself is
# by ring membership (is_water_crossing), not by coordinate.
BARRIER_X = 9.0

# Mode free-flow speeds (km/h). Same five-mode vocabulary as the frozen
# taxonomy (walk, transit, ride, car, bike). Used only to turn synthetic
# distances into base minutes.
SPEED_KMH: Dict[str, float] = {
    "walk": 4.8,
    "transit": 22.0,
    "ride": 27.0,
    "car": 30.0,
    "bike": 14.0,
}


def _zone_xy(zone: str) -> Tuple[float, float]:
    ring = ZONE_RING[zone]
    cx, cy, radius = _RING_GEOMETRY[ring]
    zones_in_ring = RING_ZONES[ring]
    idx = zones_in_ring.index(zone)
    angle = 2.0 * np.pi * idx / len(zones_in_ring)
    return (cx + radius * np.cos(angle), cy + radius * np.sin(angle))


# Precomputed (30, 2) coordinate table and the per-ring radius each zone sits
# at (its "local" access distance from the ring centre).
ZONE_XY: np.ndarray = np.array([_zone_xy(z) for z in ZONES], dtype=float)
_LOCAL_KM: np.ndarray = np.array(
    [_RING_GEOMETRY[ZONE_RING[z]][2] for z in ZONES], dtype=float
)

# Access-time model for corridor door-to-door times: the off-facility portion
# (getting from the origin to the corridor and from the corridor to the
# destination). A small deterministic offset from ring geometry, dominated by
# the congested facility time; documented DEV placeholder.
_ACCESS_BASE_MIN = 3.0
_ACCESS_PER_KM = 0.6


def zone_coordinates(zone: str) -> Tuple[float, float]:
    """Deterministic synthetic (x, y) km for a zone code (no RNG)."""
    x, y = ZONE_XY[ZONE_INDEX[zone]]
    return (float(x), float(y))


def zone_distance_km(zone_a: str, zone_b: str) -> float:
    """Synthetic straight-line distance between two zones (small floor for
    intra-zone trips, which are still real trips)."""
    if zone_a == zone_b:
        return 0.6
    xa, ya = ZONE_XY[ZONE_INDEX[zone_a]]
    xb, yb = ZONE_XY[ZONE_INDEX[zone_b]]
    return float(np.hypot(xb - xa, yb - ya))


def base_minutes(zone_a: str, zone_b: str, mode: str) -> float:
    """Static zone-to-zone travel time (minutes) for a mode, from synthetic
    distance and free-flow speed. This is the M1 static matrix used for every
    mode except the congested car/ride corridor route choice."""
    dist = zone_distance_km(zone_a, zone_b)
    return dist / SPEED_KMH[mode] * 60.0


def _band_codes(ring_codes: np.ndarray) -> np.ndarray:
    return BAND[ring_codes]


def is_corridor_od(origin_ring_code: np.ndarray, dest_ring_code: np.ndarray) -> np.ndarray:
    """Vectorized: True where a trip traverses the north-south corridor.

    A trip is on the corridor iff its two ends lie in different NON-EAST
    bands: north<->south (the full spine), north<->central and south<->central
    (the core-crossing pairs). Intra-band trips (both central, both north,
    both south) stay local; any east end is a water crossing, never the
    corridor. This is the geometric corridor OD set the spec asks for.
    """
    bo = _band_codes(origin_ring_code)
    bd = _band_codes(dest_ring_code)
    east = (bo == _BAND_EAST) | (bd == _BAND_EAST)
    return (~east) & (bo != bd)


def is_water_crossing(origin_ring_code: np.ndarray, dest_ring_code: np.ndarray) -> np.ndarray:
    """Vectorized: True where exactly one trip end is in east_water — the
    capacity-limited east-west crossing (its own VDF, network.py)."""
    o_east = origin_ring_code == RING_INDEX["east_water"]
    d_east = dest_ring_code == RING_INDEX["east_water"]
    return o_east ^ d_east


def crosses_cordon(
    origin_ring_code: np.ndarray,
    dest_ring_code: np.ndarray,
    cordon_ring_codes: Tuple[int, ...],
) -> np.ndarray:
    """Vectorized: True where exactly one trip end is inside the cordon rings
    (the v1 instrument reused by cityk_cordon — a fee to cross into the
    central rings). Structurally the cordon analogue of is_water_crossing."""
    cordon = np.asarray(cordon_ring_codes)
    o_in = np.isin(origin_ring_code, cordon)
    d_in = np.isin(dest_ring_code, cordon)
    return o_in ^ d_in


def access_minutes(origin_zone_idx: np.ndarray, dest_zone_idx: np.ndarray) -> np.ndarray:
    """Vectorized off-facility access time (minutes) for corridor trips: the
    deterministic local portion at each end, from ring geometry. The
    congested facility time (network.py) dominates door-to-door time; this is
    a documented geometric offset, common to all four facilities for a given
    OD (so it never affects the route choice, only the reported time)."""
    local = _LOCAL_KM[origin_zone_idx] + _LOCAL_KM[dest_zone_idx]
    return _ACCESS_BASE_MIN + _ACCESS_PER_KM * local
