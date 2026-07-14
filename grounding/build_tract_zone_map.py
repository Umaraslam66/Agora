#!/usr/bin/env python3
"""Reproducible builder for the committed tract -> City K v2 zone map (M2, D10).

WHAT THIS DOES
    Reads a public Census 2020 tract gazetteer (per-tract interior-point
    centroids INTPTLAT / INTPTLONG) plus the survey households table (for
    household-count weighting), and emits ``grounding/tract_zone_map.json``:
    a many-to-one lookup from 11-digit 2020 tract FIPS to a City K v2 zone
    code (Z01..Z30), covering every tract in the four-county region.

    The output feeds ``grounding.zone_map`` (residence-ring lookup) and the
    A2.6 residence-ring re-pin. The world's zone->ring vocabulary
    (world.geometry.RING_ORDER / RING_ZONES) is the single authority for
    ring names and per-ring zone codes; this builder imports it so the two
    can never drift.

DESIGN (deterministic, seed-fixed, reproducible)
    1. Restrict the gazetteer to the four region county FIPS prefixes.
    2. Assign each tract to one of five rings by explicit geometric rules on
       the centroid (county prefix + latitude / longitude thresholds around
       the north-south corridor axis and the across-water barrier). The rule
       constants are bare numbers; the geography they encode is narrated only
       in docs/M2_RING_REPIN.md (masking discipline: no real place names in
       code). County FIPS numerals are permitted here.
    3. Within each ring, partition the tracts into that ring's zone codes by
       contiguous geographic clusters (weighted k-means on projected
       centroids, household-weighted so zones carry roughly comparable
       household counts where feasible). Fully deterministic: seed-fixed
       k-means++ by weighted farthest-point, argmin ties to lowest index,
       clusters ordered north->south / west->east onto the ring's zone codes.

    No agent ever reads this file or its coordinates; agents see only the
    masked Z-codes through the world layer. Real centroids live here and in
    the gitignored gazetteer, never in an agent-facing surface.

USAGE
    .venv/bin/python grounding/build_tract_zone_map.py            # writes JSON
    .venv/bin/python grounding/build_tract_zone_map.py --check    # rebuild-only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world.geometry import RING_ORDER, RING_ZONES  # noqa: E402
DEFAULT_GAZETTEER = REPO_ROOT / "data" / "geo" / "2020_gaz_tracts_53.txt"
DEFAULT_HOUSEHOLDS = REPO_ROOT / "data" / "psrc" / "hts_households_2017_2025_v2026.1.csv"
OUT_JSON = Path(__file__).resolve().parent / "tract_zone_map.json"

MAP_VERSION = "m2-1.0"
BUILT_DATE = "2026-07-14"
SOURCE = (
    "US Census Bureau 2020 Gazetteer, census tracts, state FIPS 53 "
    "(INTPTLAT/INTPTLONG interior-point centroids); household weighting from "
    "the survey households table, two pooled waves, hh_weight>0, x0.5 wave mass"
)

# Four region county FIPS prefixes (numerals permitted under masking rules).
_CTY_NORTH = "53061"
_CTY_SOUTH = "53053"
_CTY_ACROSS = "53035"
_CTY_CORRIDOR = "53033"
FOUR_COUNTY_FIPS: Tuple[str, ...] = (_CTY_CORRIDOR, _CTY_ACROSS, _CTY_SOUTH, _CTY_NORTH)

# Waves and wave-mass rescale mirror the M0 build (docs/internal/m0_bars).
WAVES = (2017, 2019)
WAVE_RESCALE = 0.5

# ---------------------------------------------------------------------------
# Geometric ring rule constants (bare numbers; geography narrated in the doc)
# ---------------------------------------------------------------------------
# Corridor-city downtown anchor (the corridor's central segment).
_DOWN_LAT, _DOWN_LON = 47.606, -122.335
# Core radius (km) around the downtown anchor: the city-center ring.
_CORE_RADIUS_KM = 4.2
# North / south split latitudes along the corridor axis, in the corridor county.
_LAT_SOUTH = 47.505
_LAT_NORTH = 47.735
# Small across-water islands off the west shore (ferry-served, non-corridor).
_ISLAND_LON = -122.43
_ISLAND_LAT = 47.52
# Across-water barrier: a latitude-tilted longitude boundary. East of it (in
# the corridor county's central latitude band) is the across-water ring.
_BARRIER_LON0 = -122.24
_BARRIER_LAT0 = 47.53
_BARRIER_TILT = 0.18
# Equirectangular projection reference (region interior).
_PROJ_LAT0, _PROJ_LON0 = 47.5, -122.2


def _barrier_lon(lat: float) -> float:
    """Longitude of the across-water barrier at a given latitude (tilted)."""
    return _BARRIER_LON0 - _BARRIER_TILT * (lat - _BARRIER_LAT0)


def _core_distance_km(lat: float, lon: float) -> float:
    dy = (lat - _DOWN_LAT) * 111.0
    dx = (lon - _DOWN_LON) * 111.0 * math.cos(math.radians(_DOWN_LAT))
    return math.hypot(dx, dy)


def ring_of_centroid(county_fips: str, lat: float, lon: float) -> str:
    """Assign a tract centroid to one of the five rings by geometric rules.

    Order matters: island and latitude gates fire before the across-water
    barrier so the corridor-county classification stays unambiguous.
    """
    if county_fips == _CTY_ACROSS:
        return "east_water"          # the across-water peninsula county (structural)
    if county_fips == _CTY_NORTH:
        return "outer_north"
    if county_fips == _CTY_SOUTH:
        return "outer_south"
    # corridor county (_CTY_CORRIDOR)
    if lon <= _ISLAND_LON and lat <= _ISLAND_LAT:
        return "east_water"          # ferry-served west island (across-water)
    if lat < _LAT_SOUTH:
        return "outer_south"
    if lat > _LAT_NORTH:
        return "outer_north"
    if lon > _barrier_lon(lat):
        return "east_water"          # east of the across-water barrier
    # corridor city: split into the two central rings
    if _core_distance_km(lat, lon) <= _CORE_RADIUS_KM:
        return "core"
    return "inner"


# ---------------------------------------------------------------------------
# Deterministic household-weighted recursive median split (k-d partition).
# Each split cuts the current group along its longer projected axis at the
# household-weight boundary that gives the two sides shares proportional to
# their target zone counts. Leaves are contiguous and carry roughly equal
# household weight. No randomness: stable sort + integer cuts => the split is
# exact and reproducible from the same inputs (no seed needed).
# ---------------------------------------------------------------------------

def _project(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    x = (lon - _PROJ_LON0) * 111.320 * math.cos(math.radians(_PROJ_LAT0))
    y = (lat - _PROJ_LAT0) * 110.574
    return np.column_stack([x, y])


def _balanced_partition(idx: np.ndarray, X: np.ndarray, w: np.ndarray,
                        k: int) -> List[np.ndarray]:
    """Split the point-index array ``idx`` into ``k`` contiguous, roughly
    weight-balanced groups. Returns a list of k index arrays."""
    if k <= 1 or len(idx) <= 1:
        return [idx]
    kL = k // 2
    kR = k - kL
    pts = X[idx]
    spread = pts.max(axis=0) - pts.min(axis=0)
    axis = int(np.argmax(spread))
    order = idx[np.argsort(pts[:, axis], kind="stable")]
    ww = w[order]
    cum = np.cumsum(ww)
    target = cum[-1] * (kL / k)
    cut = int(np.searchsorted(cum, target, side="left")) + 1
    cut = max(kL, min(len(order) - kR, cut))
    left, right = order[:cut], order[cut:]
    return _balanced_partition(left, X, w, kL) + _balanced_partition(right, X, w, kR)


def _order_groups(X: np.ndarray, groups: List[np.ndarray]) -> List[int]:
    """Deterministic group ordering: north->south (y desc), then west->east
    (x asc) by group centroid. Returns group positions in that order."""
    keys = []
    for g, members in enumerate(groups):
        cx = float(X[members, 0].mean()) if len(members) else 0.0
        cy = float(X[members, 1].mean()) if len(members) else 0.0
        keys.append((g, -cy, cx))
    keys.sort(key=lambda t: (t[1], t[2], t[0]))
    return [g for g, _, _ in keys]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _load_gazetteer(path: Path) -> pd.DataFrame:
    g = pd.read_csv(path, sep="\t", dtype={"GEOID": str})
    g.columns = [c.strip() for c in g.columns]
    g["GEOID"] = g["GEOID"].str.strip()
    g["lat"] = g["INTPTLAT"].astype(float)
    g["lon"] = g["INTPTLONG"].astype(float)
    g["cty5"] = g["GEOID"].str[:5]
    g = g[g["cty5"].isin(FOUR_COUNTY_FIPS)].copy()
    return g.sort_values("GEOID").reset_index(drop=True)


def _household_weight_by_tract(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    hh = pd.read_csv(
        path,
        usecols=["survey_year", "home_tract_2020", "hh_weight"],
        dtype={"home_tract_2020": str},
        low_memory=False,
    )
    hh = hh[hh["survey_year"].isin(WAVES)].copy()
    hh = hh[hh["hh_weight"] > 0]
    hh["tract"] = hh["home_tract_2020"].astype(str).str.replace(".0", "", regex=False)
    hh = hh[hh["tract"].str.len() == 11]
    g = hh.groupby("tract")["hh_weight"].sum() * WAVE_RESCALE
    return {str(t): float(v) for t, v in g.items()}


def build_map(gazetteer_path: Path = DEFAULT_GAZETTEER,
              households_path: Path = DEFAULT_HOUSEHOLDS) -> dict:
    """Build the tract->zone map dict deterministically. Requires the
    gazetteer; household weights are optional (fall back to uniform)."""
    g = _load_gazetteer(gazetteer_path)
    w_by_tract = _household_weight_by_tract(households_path)

    g["ring"] = [ring_of_centroid(c, la, lo)
                 for c, la, lo in zip(g["cty5"], g["lat"], g["lon"])]
    # weight floor keeps unsampled tracts represented without moving centroids
    g["w"] = g["GEOID"].map(w_by_tract).fillna(0.0).astype(float) + 1.0

    tracts: Dict[str, str] = {}
    for ring in RING_ORDER:
        codes = RING_ZONES[ring]
        k = len(codes)
        sub = g[g["ring"] == ring].sort_values("GEOID").reset_index(drop=True)
        if len(sub) == 0:
            continue
        X = _project(sub["lat"].to_numpy(), sub["lon"].to_numpy())
        w = sub["w"].to_numpy()
        groups = _balanced_partition(np.arange(len(sub)), X, w, k)
        order = _order_groups(X, groups)
        geoids = sub["GEOID"].to_numpy()
        for i, gpos in enumerate(order):
            code = codes[i]
            for member in groups[gpos]:
                tracts[str(geoids[int(member)])] = code

    return {
        "map_version": MAP_VERSION,
        "built": BUILT_DATE,
        "source": SOURCE,
        "county_fips": list(FOUR_COUNTY_FIPS),
        "tracts": dict(sorted(tracts.items())),
    }


def write_map(out: Path = OUT_JSON, **kwargs) -> dict:
    m = build_map(**kwargs)
    out.write_text(json.dumps(m, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the tract->zone map.")
    ap.add_argument("--gazetteer", type=Path, default=DEFAULT_GAZETTEER)
    ap.add_argument("--households", type=Path, default=DEFAULT_HOUSEHOLDS)
    ap.add_argument("--check", action="store_true",
                    help="rebuild and compare to the committed JSON; do not write")
    args = ap.parse_args(argv)

    built = build_map(args.gazetteer, args.households)
    if args.check:
        if not OUT_JSON.exists():
            print("no committed JSON to check against")
            return 1
        committed = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        same = committed.get("tracts") == built.get("tracts")
        print("MATCH" if same else "MISMATCH", f"({len(built['tracts'])} tracts)")
        return 0 if same else 1

    write_map(gazetteer_path=args.gazetteer, households_path=args.households)
    n = len(built["tracts"])
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)} : {n} tracts across "
          f"{len(FOUR_COUNTY_FIPS)} counties")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
