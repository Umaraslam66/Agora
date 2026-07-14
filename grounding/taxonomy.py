"""Frozen M0 vocabulary — mode taxonomy and E1 protected-segment definitions.

Committed at M0 per pre-registration §5 ("Segment definitions, masking
scheme, and the forbidden-token list are committed at M0 and versioned").
Frozen by the project owner's M0 decisions; changing anything here after
M0 requires a dated pre-registration §7 amendment, never an edit.

MODE TAXONOMY — five modes, frozen order:

    walk, transit, ride, car, bike

The tuple ORDER is itself frozen: it is the deterministic tie-break order
of the serving gateway's temperature-0 argmax and of the fast-brain
fallback chooser. serving.gateway.DEFAULT_MODES_ORDER and
agents.logit_chooser.MODES_ALL must equal MODES; tests enforce it.

Semantics (owner decision at M0):
  - car      private vehicle, this person driving.
  - ride     vehicle passenger: riding in a private vehicle someone else
             drives, or a hired ride (ride-hail / taxi).
  - transit  scheduled shared service; timetabled water crossings and
             school runs collapse here too (both are scheduled shared
             services, and naming them separately would leak geography).
  - bike     cycle, plus small shared/rented micro-vehicles.
  - walk     on foot.

E1 PROTECTED SEGMENTS — income band x car ownership x residence band,
12 cells. Income bands group the ordinal income_class 1..5 a priori:
low = {1, 2}, mid = {3, 4}, high = {5} (survey income bands are
nominal-dollar; the grouping is fixed in advance, never fitted to data).
Car ownership is binary: none vs at least one. Residence band is
"catchment" for homes in the rings that feed the priced facility
(CATCHMENT_RINGS) and "remainder" otherwise. The zone->ring map belongs
to whichever zone taxonomy is in force (the grounding zone codes today,
the City K world config from M1 on); this module freezes only which RINGS
count as catchment, so the segment axis survives a zone-taxonomy upgrade.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

TAXONOMY_VERSION = "m0-1.0"

# ---------------------------------------------------------------------------
# Modes (frozen tuple; order = deterministic tie-break order)
# ---------------------------------------------------------------------------

MODES: Tuple[str, ...] = ("walk", "transit", "ride", "car", "bike")

# ---------------------------------------------------------------------------
# Source-survey collapse (harness-side only: consumed by real-data adapters
# and the M0 bar derivation; never by agent-facing code or templates)
# ---------------------------------------------------------------------------

# Coarse survey mode classes that map to one frozen mode unconditionally,
# or to None = drop the trip with a logged count (paratransit, airplane,
# private shuttle and missing responses have no home in the frozen set).
_DIRECT_MODE_CLASS: Dict[str, Optional[str]] = {
    "Transit": "transit",       # includes timetabled water crossings
    "Walk": "walk",
    "Bike": "bike",
    "Micromobility": "bike",    # the survey's own 5-way collapse does this
    "Ride Hail": "ride",
    "School Bus": "transit",    # scheduled shared service (children)
    "Other": None,
    "Missing Response": None,
}

# Drive classes need the driver/passenger flag: the driver made a "car"
# trip, everyone else in the vehicle made a "ride" trip.
_DRIVE_MODE_CLASSES: Tuple[str, ...] = ("Drive SOV", "Drive HOV2", "Drive HOV3+")

#: Every survey mode_class label this taxonomy knows. Adapters must fail
#: loud (not drop silently) on any label outside this set — label sets
#: drift between survey waves.
KNOWN_MODE_CLASSES = frozenset(_DIRECT_MODE_CLASS) | frozenset(_DRIVE_MODE_CLASSES)


def collapse_mode(
    mode_class: str,
    driver: Optional[str] = None,
    can_drive: bool = True,
) -> Optional[str]:
    """Collapse a survey coarse mode class to the frozen five-mode set.

    Returns None when the trip has no home in the frozen set (caller must
    drop it WITH a logged count). ``driver`` is the survey's driver/
    passenger flag for drive-class trips ("Driver" / "Passenger" /
    "Both..." — switched drivers mid-trip counts as driving). When the
    flag is absent, ``can_drive`` breaks the tie: licensed adults default
    to driver ("car"), everyone else to passenger ("ride").
    """
    if mode_class == "Drive SOV":
        return "car"  # sole occupant is the driver by definition
    if mode_class in _DRIVE_MODE_CLASSES:
        if driver is not None:
            d = driver.strip().lower()
            if d.startswith("passenger"):
                return "ride"
            if d.startswith(("driver", "both")):
                return "car"
        return "car" if can_drive else "ride"
    return _DIRECT_MODE_CLASS.get(mode_class)


# ---------------------------------------------------------------------------
# E1 protected segments (pre-registration §3 E1: "protected segment cells")
# ---------------------------------------------------------------------------

INCOME_BANDS: Tuple[str, ...] = ("low", "mid", "high")
_INCOME_BAND_OF_CLASS: Dict[int, str] = {1: "low", 2: "low", 3: "mid", 4: "mid", 5: "high"}

CAR_BANDS: Tuple[str, ...] = ("car0", "car1p")

RESIDENCE_BANDS: Tuple[str, ...] = ("catchment", "remainder")

#: Rings whose residents count as living in the priced facility's
#: catchment. Covers both zone taxonomies: "inner" (grounding's current
#: three-ring codes) and "core" (the City K corridor world, M1+).
CATCHMENT_RINGS = frozenset({"inner", "core"})


def income_band(income_class: int) -> str:
    """Frozen a-priori banding of the ordinal income_class 1..5."""
    return _INCOME_BAND_OF_CLASS[income_class]


def car_band(household_cars: int) -> str:
    return "car1p" if household_cars >= 1 else "car0"


def residence_band(ring: str) -> str:
    return "catchment" if ring in CATCHMENT_RINGS else "remainder"


def segment_cell(income_class: int, household_cars: int, ring: str) -> str:
    """Protected-segment cell id, e.g. ``"low|car0|catchment"``."""
    return f"{income_band(income_class)}|{car_band(household_cars)}|{residence_band(ring)}"


SEGMENT_CELLS: Tuple[str, ...] = tuple(
    f"{i}|{c}|{r}" for i in INCOME_BANDS for c in CAR_BANDS for r in RESIDENCE_BANDS
)
