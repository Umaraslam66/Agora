"""Committed tract -> City K v2 zone / ring lookup (M2, D10; A2.6 re-pin).

Loads ``grounding/tract_zone_map.json`` once and resolves an 11-digit 2020
tract FIPS to a masked zone code (Z01..Z30) and, through the world layer's
authoritative zone->ring map, to one of the five rings. This is the harness
input that re-pins a household's residence ring from its home tract (replacing
the M0 provisional core-jurisdiction proxy).

The zone->ring vocabulary is owned by ``world.geometry`` (RING_ORDER); this
module never re-declares ring names. Downstream segment banding
(catchment vs remainder) is applied by ``grounding.taxonomy.residence_band``.

UNKNOWN / MISSING HANDLING (documented contract): a tract FIPS that is not in
the committed map -- or a missing / malformed home-tract value -- resolves to
``None`` (never a silent default ring). The caller decides what to do; the
intended policy at the E1/E2 layer is drop-with-a-logged-count, exactly as the
M0 build drops records with no usable field. ``None`` is returned rather than
raised so a single unmapped record cannot abort a whole population pass.

No real place names appear here (masking discipline); all geographic narrative
lives in docs/M2_RING_REPIN.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from world.geometry import ZONE_RING

_MAP_PATH = Path(__file__).resolve().parent / "tract_zone_map.json"

_TRACTS: Optional[Dict[str, str]] = None
_META: Optional[dict] = None


def _load() -> Dict[str, str]:
    """Load and cache the committed map (once per process)."""
    global _TRACTS, _META
    if _TRACTS is None:
        with open(_MAP_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        _META = {k: v for k, v in data.items() if k != "tracts"}
        _TRACTS = {str(k): str(v) for k, v in data["tracts"].items()}
    return _TRACTS


def map_metadata() -> dict:
    """Return the committed map's metadata (version, build date, source)."""
    _load()
    return dict(_META or {})


def _normalize_tract(value) -> Optional[str]:
    """Coerce a home/origin/dest tract value to an 11-digit FIPS string, or
    ``None`` if it is missing or not a well-formed tract id. Tolerates the
    ``"53033000101.0"`` float-rendered form the survey CSVs sometimes carry."""
    if value is None:
        return None
    try:
        # NaN is the only value not equal to itself
        if value != value:  # noqa: PLR0124
            return None
    except TypeError:
        pass
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 11 and s.isdigit():
        return s
    return None


def zone_of_tract(geoid) -> Optional[str]:
    """Masked zone code (Z01..Z30) for a 2020 tract FIPS, or ``None`` if the
    tract is unknown / malformed (caller drops-with-logged-count)."""
    key = _normalize_tract(geoid)
    if key is None:
        return None
    return _load().get(key)


def ring_of_tract(geoid) -> Optional[str]:
    """Ring (world.geometry.RING_ORDER member) for a 2020 tract FIPS, via the
    zone code, or ``None`` if the tract is unknown / malformed."""
    zone = zone_of_tract(geoid)
    if zone is None:
        return None
    return ZONE_RING.get(zone)


def ring_of_household(home_tract) -> Optional[str]:
    """Residence ring for a household from its home tract (the A2.6 re-pin
    entry point), or ``None`` if the home tract is missing / unmapped."""
    return ring_of_tract(home_tract)
