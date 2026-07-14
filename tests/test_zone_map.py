"""Tests for the committed tract -> City K v2 zone map (M2, D10; A2.6 re-pin).

Guards the four properties the re-pin depends on:
  (1) full four-county coverage -- every survey home tract resolves;
  (2) determinism -- rebuilding from the gazetteer reproduces the JSON byte
      for byte (skipped when the gitignored gazetteer is absent);
  (3) ring vocabulary is exactly world.geometry.RING_ORDER, and zone codes
      are exactly world's zone set;
  (4) the two catchment rings resolve to residence band "catchment" through
      grounding.taxonomy.residence_band (the E1 protected-segment axis).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from grounding import zone_map
from grounding.taxonomy import CATCHMENT_RINGS, residence_band
from world.geometry import RING_ORDER, RING_ZONES, ZONE_RING, ZONES

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_JSON = REPO_ROOT / "grounding" / "tract_zone_map.json"
GAZETTEER = REPO_ROOT / "data" / "geo" / "2020_gaz_tracts_53.txt"
HOUSEHOLDS = REPO_ROOT / "data" / "psrc" / "hts_households_2017_2025_v2026.1.csv"

FOUR_COUNTY_FIPS = {"53033", "53035", "53053", "53061"}


@pytest.fixture(scope="module")
def committed():
    return json.loads(MAP_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# (3) vocabulary: rings == world RING_ORDER, zone codes == world ZONES
# ---------------------------------------------------------------------------

def test_zone_codes_exist_in_world_config(committed):
    zones_used = set(committed["tracts"].values())
    assert zones_used <= set(ZONES), (
        f"map uses zone codes not in world config: {sorted(zones_used - set(ZONES))}"
    )
    # every world zone is populated (the map is a total partition of Z01..Z30)
    assert zones_used == set(ZONES), (
        f"world zones with no tracts: {sorted(set(ZONES) - zones_used)}"
    )


def test_ring_vocabulary_matches_world(committed):
    rings_used = {ZONE_RING[z] for z in committed["tracts"].values()}
    assert rings_used == set(RING_ORDER), (
        f"ring set {sorted(rings_used)} != world RING_ORDER {sorted(RING_ORDER)}"
    )


def test_every_ring_zonecount_matches_world(committed):
    from collections import Counter
    per_zone = Counter(committed["tracts"].values())
    for ring in RING_ORDER:
        for code in RING_ZONES[ring]:
            assert per_zone.get(code, 0) > 0, f"empty zone {code} in ring {ring}"


# ---------------------------------------------------------------------------
# (4) catchment rings resolve per taxonomy.residence_band
# ---------------------------------------------------------------------------

def test_catchment_rings_resolve_per_taxonomy(committed):
    # world's central rings are exactly the taxonomy catchment rings
    assert CATCHMENT_RINGS == {"core", "inner"}
    for zone in set(committed["tracts"].values()):
        ring = ZONE_RING[zone]
        band = residence_band(ring)
        if ring in ("core", "inner"):
            assert band == "catchment", f"{zone}/{ring} should be catchment"
        else:
            assert band == "remainder", f"{zone}/{ring} should be remainder"


def test_ring_of_household_catchment_and_remainder(committed):
    # pick one tract from a catchment zone and one from a remainder zone
    inv = {}
    for tract, zone in committed["tracts"].items():
        inv.setdefault(ZONE_RING[zone], tract)
    core_tract = inv.get("core")
    east_tract = inv.get("east_water")
    assert residence_band(zone_map.ring_of_household(core_tract)) == "catchment"
    assert residence_band(zone_map.ring_of_household(east_tract)) == "remainder"


# ---------------------------------------------------------------------------
# unknown / malformed handling
# ---------------------------------------------------------------------------

def test_unknown_tract_returns_none():
    assert zone_map.zone_of_tract("99999999999") is None
    assert zone_map.ring_of_tract("99999999999") is None
    assert zone_map.ring_of_household(None) is None
    assert zone_map.ring_of_household("") is None
    assert zone_map.ring_of_household(float("nan")) is None


def test_float_rendered_tract_normalizes(committed):
    tract = next(iter(committed["tracts"]))
    assert zone_map.ring_of_household(tract) is not None
    # the ".0" float-render must resolve identically
    assert zone_map.ring_of_household(float(tract)) == zone_map.ring_of_household(tract)


def test_only_four_counties_present(committed):
    counties = {t[:5] for t in committed["tracts"]}
    assert counties == FOUR_COUNTY_FIPS, f"unexpected counties: {counties}"


# ---------------------------------------------------------------------------
# (1) full four-county coverage of survey home tracts
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HOUSEHOLDS.exists(), reason="gitignored households CSV absent")
def test_full_four_county_home_tract_coverage():
    pd = pytest.importorskip("pandas")
    hh = pd.read_csv(
        HOUSEHOLDS,
        usecols=["survey_year", "home_tract_2020", "hh_weight"],
        dtype={"home_tract_2020": str},
        low_memory=False,
    )
    hh = hh[hh["survey_year"].isin([2017, 2019])]
    hh = hh[hh["hh_weight"] > 0]
    tracts = hh["home_tract_2020"].astype(str).str.replace(".0", "", regex=False)
    tracts = tracts[tracts.str.len() == 11]
    unmapped = sorted({t for t in tracts.unique() if zone_map.zone_of_tract(t) is None})
    assert not unmapped, f"{len(unmapped)} home tracts do not map: {unmapped[:10]}"


# ---------------------------------------------------------------------------
# (2) determinism: rebuild reproduces the committed JSON exactly
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (GAZETTEER.exists() and HOUSEHOLDS.exists()),
    reason="gitignored gazetteer/households absent -- cannot rebuild",
)
def test_determinism_rebuild_matches_committed(committed):
    pytest.importorskip("pandas")
    from grounding.build_tract_zone_map import build_map

    rebuilt = build_map(GAZETTEER, HOUSEHOLDS)
    assert rebuilt["tracts"] == committed["tracts"], "rebuild != committed JSON"
    assert rebuilt["map_version"] == committed["map_version"]
