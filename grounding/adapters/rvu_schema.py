#!/usr/bin/env python3
"""Canonical diary schema for SAGA's grounding layer — dataclasses + JSONL
(de)serialization, with a documented field-level mapping onto Sweden's
national travel survey (Trafikanalys, "Trafa RVU"; the 2005-06 sweep is
published as "RES 2005-06").

WHY THIS FILE EXISTS (see 00_PROJECT_BRIEF.md layer 1, 01_PREREGISTRATION.md
§6): every SAGA agent is meant to be seeded from one real anonymous
individual diary record. That microdata arrives on application and may be
late or delivered only as aggregates. This module defines the CANONICAL
record shape agents are seeded from, independent of source — a real RVU/RES
adapter and `grounding/synthetic_diary.py`'s M0 synthetic stand-in both
produce exactly this shape, so nothing downstream (persona construction,
E1/E2 scoring) has to know or care which one it is reading. Every record
produced by the synthetic generator carries `synthetic: True`; real-data
adapters (future work) must set it False. Per the pre-registration, results
computed on `synthetic: True` records are DEV-only and never cited.

RVU/RES STRUCTURE THIS MIRRORS (harness-side background, not agent-facing):
Trafa's RVU is a one-day trip-diary survey (ages 6-84) with three
conceptual levels: individual (individ), household (hushåll), and trip
(resa). A trip is reported as a "huvudresa" (main journey, one purpose/
errand, one dominant mode) which may itself be composed of several
"delresor" (sub-journeys/legs, e.g. walk-bus-walk); published aggregates and
most downstream analysis work at the huvudresa level, which is also what
this schema captures (`Trip.is_main_trip` is always True here — see field
mapping table below). The published RVU documentation used to ground this
schema is Trafikanalys's "RVU Sverige" statistical reports (e.g. Statistik
2015:10) and quality declarations; the RES 2005-06 microdata codebook
(exact SPSS/CSV column names) is not yet in hand — Trafa RVU tables have
been stable in CONCEPT across sweeps (1994-2015+), but literal column codes
must be verified once RES 2005-06 microdata is licensed (see TODOs at the
bottom of this docstring).

FIELD MAPPING — our field -> RVU/RES concept it stands in for
(harness-side only; these Swedish terms never reach an agent prompt):

  Household (RVU: hushåll)
    household_id     hushålls-ID (household key; join id, no RVU concept)
    household_size   hushållets storlek / antal personer i hushållet
    household_type   hushållstyp (ensamboende / sammanboende utan barn /
                      sammanboende med barn / ensamstående med barn / övrigt
                      -> single / couple_no_children / couple_with_children /
                      single_parent / other)
    household_cars   antal bilar i hushållet
    home_zone        hemkommun / bostadsort, coarsened here to a zone code
                      (see "Zone taxonomy" below) instead of a real
                      municipality
    income_class     hushållets inkomstklass (ordinal income bracket, 1
                      lowest .. 5 highest; mirrors RVU/SrV-style banded
                      household income, not exact SEK)

  Individual (RVU: individ)
    person_id           individ-ID (join id, no RVU concept)
    household_id        FK -> Household.household_id (RVU links individ to
                         hushåll by the same household key)
    age                 ålder
    sex                 kön (RVU's published sweeps record a binary m/f
                         field; noted here as an inherited survey
                         limitation, not a design choice of ours)
    driving_licence      körkortsinnehav
    employed             sysselsättning / förvärvsarbete (boolean here;
                         RVU records a fuller status typology)
    student              studerande
    home_zone            same coarsened zone as the household's home_zone
    work_zone            arbetsplatskommun / arbetsplatsort (workplace
                         zone; None if not employed)
    person_weight        individvikt / uppräkningstal (survey expansion
                         weight). DEV: the synthetic stand-in has no
                         sampling design, so this is fixed at 1.0 for every
                         record — do not use it as a real weight.

  Trip (RVU: resa / huvudresa)
    trip_index        0-based position in the person's day (no direct RVU
                       field; RVU orders trips by reported time)
    purpose            huvudresans ärende (trip purpose/errand at the
                       destination): home, work, education, shop_daily,
                       shop_other, leisure, personal_business,
                       pickup_dropoff, other
    main_mode          huvudresans huvudsakliga färdmedel, collapsed to
                       the frozen five-mode taxonomy (grounding/taxonomy.py,
                       M0 owner decision): walk, transit, ride, car, bike.
                       "ride" = vehicle passenger (personbil passagerare or
                       a hired ride); scheduled water crossings and school
                       runs collapse into transit. RVU distinguishes many
                       more mode codes (buss, tåg, spårvagn, tunnelbana,
                       cykel, till fots) — real adapters collapse via
                       taxonomy.collapse_mode, never ad hoc
    origin_zone         startpunkt (coarsened to a zone code)
    destination_zone     målpunkt (coarsened to a zone code)
    depart_time          starttid, "HH:MM"
    arrive_time           sluttid, "HH:MM"
    distance_km            reslängd (E_LAENGE-style route length in RVU/SrV
                       terms; treat as route km, not beeline — same caveat
                       carried over from the enact-era adapters)
    is_main_trip        huvudresa (True) vs delresa (False) flag; always
                       True here because this schema only models the
                       huvudresa level (see "RVU/RES STRUCTURE" above).
                       Present so a future real-data adapter can add
                       delresa-level legs without a schema change.

  DiaryRecord (one person-day; RVU: one respondent's reported diary day)
    record_id           person_id + relative day index (join id)
    household           embedded Household (not just a foreign key — see
                       "Household-atomic" note below)
    individual           embedded Individual
    trips                ordered list of Trip, one person-day's huvudresor
    day_index            RELATIVE day indicator (0-6, arbitrary), never a
                       real calendar date — RVU respondents get a real
                       assigned diary date; per 00_PROJECT_BRIEF.md's
                       contamination-masking rule, no real survey-era date
                       may appear in any agent-facing field, so this schema
                       never carries one, synthetic or real
    day_type              derived "weekday" | "weekend" from day_index
    synthetic              True for every record `synthetic_diary.py`
                       produces; a future real-microdata adapter sets it
                       False
    schema_version         see SCHEMA_VERSION below

Household-atomic IDs: `household_id` appears on Household, on Individual
(as a foreign key back to its household) and is embedded inside every
DiaryRecord. E1 (01_PREREGISTRATION.md §3) splits diary records 80/20
HOUSEHOLD-atomic — every member of a household must land in the same
train/holdout split, exactly as `nhts_adapter.py`'s deviation 1 made NHTS's
person-level split household-atomic for the same reason (correlated
household members must not leak across the split). `household_split()`
below is that deterministic hash-based split, keyed on household_id only.

Zone taxonomy (agent-facing, but contains no real place names or
municipality names — pure codes): 25 zones `Z01`..`Z25` arranged in three
concentric rings — `inner` (Z01-Z05), `inner_suburb` (Z06-Z15), `outer`
(Z16-Z25) — loosely shaped like a generic large Nordic metro without naming
one. A `cordon` boundary sits between the `inner` ring and everything else
(`is_within_cordon` / `crosses_cordon` below); this gives the world/ layer
(M1+) a structural hook for a masked toll cordon without this module having
any opinion about tolls. `zone_coordinates`/`zone_distance_km` derive a
synthetic geometry (concentric rings, evenly spaced angularly) purely from
the zone code, so trip distances are internally consistent without ever
touching a real map.

TODOs for swapping in real RVU/RES microdata (tracked here, not elsewhere):
  1. Verify the RES 2005-06 codebook's literal column names/codes against
     the mapping table above once microdata is licensed; the table above is
     written at the CONCEPT level (verified against Trafa's published RVU
     documentation) because the exact RES 2005-06 SPSS/CSV schema was not
     available at M0.
  2. `income_class` here is a household-level ordinal band; confirm whether
     RES 2005-06 reports income at household or individual level and
     which banding Trafa used for that sweep.
  3. RESOLVED at M0: the mode collapse no longer loses the driver/
     passenger distinction — the frozen five-mode taxonomy carries it as
     `ride` (vehicle passenger, incl. hired rides), per the owner's M0
     decision; see grounding/taxonomy.py `collapse_mode`.
  4. `person_weight` must be replaced with the real design/expansion weight
     (uppräkningstal) the moment real microdata lands — every weighted
     E1/E4 computation is a placebo until then.
  5. Real RVU trips include delresor (sub-legs); a real adapter should add
     delresa-level Trip records with `is_main_trip=False` sharing the same
     `trip_index` as their parent huvudresa, rather than changing this
     schema.
  6. Real home/work geography (kommun or finer) must be re-coarsened onto
     this module's Z01-Z25 zone codes (or a successor taxonomy) by the
     adapter, never passed through as a real place name (contamination
     masking, 00_PROJECT_BRIEF.md).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

from grounding.taxonomy import MODES as _TAXONOMY_MODES

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

# 0.2: mode vocabulary widened to the frozen five-mode taxonomy ("ride"
#      added; see grounding/taxonomy.py). 0.1 records remain valid (subset).
SCHEMA_VERSION = "rvu-m0-0.2"

# ---------------------------------------------------------------------------
# Zone taxonomy — codes only, no real place names (see module docstring)
# ---------------------------------------------------------------------------

ZONE_COUNT = 25
ZONES: List[str] = [f"Z{i:02d}" for i in range(1, ZONE_COUNT + 1)]

_INNER_ZONES = [f"Z{i:02d}" for i in range(1, 6)]           # Z01-Z05
_INNER_SUBURB_ZONES = [f"Z{i:02d}" for i in range(6, 16)]   # Z06-Z15
_OUTER_ZONES = [f"Z{i:02d}" for i in range(16, 26)]         # Z16-Z25

RING_ZONES: Dict[str, List[str]] = {
    "inner": _INNER_ZONES,
    "inner_suburb": _INNER_SUBURB_ZONES,
    "outer": _OUTER_ZONES,
}

ZONE_RING: Dict[str, str] = {
    z: ring for ring, zones in RING_ZONES.items() for z in zones
}

# Synthetic ring radius (km) used only to derive an internally-consistent
# zone geometry (see zone_coordinates/zone_distance_km); DEV placeholder,
# not a real map.
_RING_RADIUS_KM = {"inner": 2.5, "inner_suburb": 9.0, "outer": 18.0}


def zone_ring(zone: str) -> str:
    """Ring ("inner" | "inner_suburb" | "outer") a zone code belongs to."""
    return ZONE_RING[zone]


def is_within_cordon(zone: str) -> bool:
    """The cordon sits between the `inner` ring and everything else."""
    return zone_ring(zone) == "inner"


def crosses_cordon(origin_zone: str, destination_zone: str) -> bool:
    """True if exactly one end of the trip is inside the cordon."""
    return is_within_cordon(origin_zone) != is_within_cordon(destination_zone)


def zone_coordinates(zone: str) -> Tuple[float, float]:
    """Deterministic synthetic (x, y) km coordinates for a zone code: zones
    are placed evenly around their ring's circle. Pure function of the zone
    code (no RNG) so distances are reproducible and reusable outside the
    generator (e.g. by a future world/ layer)."""
    ring = zone_ring(zone)
    zones_in_ring = RING_ZONES[ring]
    idx = zones_in_ring.index(zone)
    radius = _RING_RADIUS_KM[ring]
    angle = 2.0 * math.pi * idx / len(zones_in_ring)
    return (radius * math.cos(angle), radius * math.sin(angle))


def zone_distance_km(zone_a: str, zone_b: str) -> float:
    """Synthetic straight-line distance between two zones. Same-zone trips
    get a small nonzero floor (intra-zone movement is still a real trip)."""
    if zone_a == zone_b:
        return 0.6
    xa, ya = zone_coordinates(zone_a)
    xb, yb = zone_coordinates(zone_b)
    return round(math.hypot(xb - xa, yb - ya), 1)


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

HOUSEHOLD_TYPES = (
    "single",
    "couple_no_children",
    "couple_with_children",
    "single_parent",
    "other",
)

PURPOSES = (
    "home",
    "work",
    "education",
    "shop_daily",
    "shop_other",
    "leisure",
    "personal_business",
    "pickup_dropoff",
    "other",
)

# Frozen at M0 (owner decision, 2026-07-14): five modes, tuple order = the
# deterministic tie-break order. Single source of truth: grounding/taxonomy.py.
MODES = _TAXONOMY_MODES

DAY_TYPES = ("weekday", "weekend")

SEXES = ("m", "f")


# ---------------------------------------------------------------------------
# Canonical dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Household:
    household_id: str
    household_size: int
    household_type: str
    household_cars: int
    home_zone: str
    income_class: int  # 1 (lowest) .. 5 (highest); ordinal RVU-style band
    synthetic: bool = True
    schema_version: str = SCHEMA_VERSION


@dataclass
class Individual:
    person_id: str
    household_id: str
    age: int
    sex: str
    driving_licence: bool
    employed: bool
    student: bool
    home_zone: str
    work_zone: Optional[str]
    person_weight: float
    synthetic: bool = True
    schema_version: str = SCHEMA_VERSION


@dataclass
class Trip:
    trip_index: int
    purpose: str
    main_mode: str
    origin_zone: str
    destination_zone: str
    depart_time: str  # "HH:MM"
    arrive_time: str  # "HH:MM"
    distance_km: float
    is_main_trip: bool = True


@dataclass
class DiaryRecord:
    record_id: str
    household: Household
    individual: Individual
    trips: List[Trip]
    day_index: int  # relative index (0-6), NEVER a real calendar date
    day_type: str  # "weekday" | "weekend"
    synthetic: bool = True
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_record(rec: DiaryRecord) -> List[str]:
    """Returns a list of human-readable violation strings; empty == valid."""
    v: List[str] = []

    if rec.household.household_id != rec.individual.household_id:
        v.append(
            f"record {rec.record_id}: household_id mismatch "
            f"({rec.household.household_id!r} != {rec.individual.household_id!r})"
        )
    if rec.household.home_zone not in ZONES:
        v.append(f"record {rec.record_id}: bad household.home_zone {rec.household.home_zone!r}")
    if rec.household.household_type not in HOUSEHOLD_TYPES:
        v.append(f"record {rec.record_id}: bad household_type {rec.household.household_type!r}")
    if not (1 <= rec.household.income_class <= 5):
        v.append(f"record {rec.record_id}: income_class out of [1,5]")

    if rec.individual.sex not in SEXES:
        v.append(f"record {rec.record_id}: bad sex {rec.individual.sex!r}")
    if rec.individual.work_zone is not None and rec.individual.work_zone not in ZONES:
        v.append(f"record {rec.record_id}: bad individual.work_zone {rec.individual.work_zone!r}")

    if rec.day_type not in DAY_TYPES:
        v.append(f"record {rec.record_id}: bad day_type {rec.day_type!r}")

    for t in rec.trips:
        if t.purpose not in PURPOSES:
            v.append(f"record {rec.record_id} trip {t.trip_index}: bad purpose {t.purpose!r}")
        if t.main_mode not in MODES:
            v.append(f"record {rec.record_id} trip {t.trip_index}: bad main_mode {t.main_mode!r}")
        if t.origin_zone not in ZONES or t.destination_zone not in ZONES:
            v.append(f"record {rec.record_id} trip {t.trip_index}: bad zone(s)")
        if t.distance_km <= 0:
            v.append(f"record {rec.record_id} trip {t.trip_index}: non-positive distance_km")
        if not rec.synthetic:
            v.append(f"record {rec.record_id}: synthetic=False from a module that only emits DEV data")

    if not rec.trips:
        v.append(f"record {rec.record_id}: no trips")

    return v


# ---------------------------------------------------------------------------
# Household-atomic split (E1, 01_PREREGISTRATION.md §3)
# ---------------------------------------------------------------------------

def household_split(household_id: str, holdout_fraction: float = 0.2) -> str:
    """Deterministic 80/20 household-atomic split: hash the HOUSEHOLD id
    (never the person id), so every member of a household always lands in
    the same split. Mirrors nhts_adapter.py's deviation 1 (household-atomic
    override of srv_pipeline's person-atomic split), for the same reason:
    household members' diary days are correlated (shared car, shared
    income, correlated trip purposes)."""
    digest = hashlib.sha256(household_id.encode("utf-8")).hexdigest()[:8]
    h = int(digest, 16) / 0xFFFFFFFF
    return "holdout" if h < holdout_fraction else "train"


def household_atomicity_violations(records: Iterable[DiaryRecord]) -> List[str]:
    """Empty return == every household_id maps to exactly one consistent
    household payload and no person_id is claimed by two households."""
    seen_household_payload: Dict[str, dict] = {}
    seen_person_household: Dict[str, str] = {}
    violations: List[str] = []

    for rec in records:
        hid = rec.household.household_id
        if rec.individual.household_id != hid:
            violations.append(
                f"record {rec.record_id}: individual.household_id "
                f"{rec.individual.household_id!r} != household.household_id {hid!r}"
            )

        payload = asdict(rec.household)
        prior = seen_household_payload.get(hid)
        if prior is not None and prior != payload:
            violations.append(f"household {hid}: conflicting household payloads across records")
        else:
            seen_household_payload[hid] = payload

        pid = rec.individual.person_id
        prior_hid = seen_person_household.get(pid)
        if prior_hid is not None and prior_hid != hid:
            violations.append(f"person {pid}: appears under two household_ids ({prior_hid!r}, {hid!r})")
        else:
            seen_person_household[pid] = hid

    return violations


# ---------------------------------------------------------------------------
# JSONL (de)serialization
# ---------------------------------------------------------------------------

DEV_NOTE = (
    "DEV STAND-IN data. Not real Trafa RVU/RES microdata. Per "
    "01_PREREGISTRATION.md §6: “results on stand-in data are "
    "labeled DEV and never cited.”"
)


def household_from_dict(d: dict) -> Household:
    return Household(
        household_id=d["household_id"],
        household_size=d["household_size"],
        household_type=d["household_type"],
        household_cars=d["household_cars"],
        home_zone=d["home_zone"],
        income_class=d["income_class"],
        synthetic=d.get("synthetic", True),
        schema_version=d.get("schema_version", SCHEMA_VERSION),
    )


def individual_from_dict(d: dict) -> Individual:
    return Individual(
        person_id=d["person_id"],
        household_id=d["household_id"],
        age=d["age"],
        sex=d["sex"],
        driving_licence=d["driving_licence"],
        employed=d["employed"],
        student=d["student"],
        home_zone=d["home_zone"],
        work_zone=d.get("work_zone"),
        person_weight=d["person_weight"],
        synthetic=d.get("synthetic", True),
        schema_version=d.get("schema_version", SCHEMA_VERSION),
    )


def trip_from_dict(d: dict) -> Trip:
    return Trip(
        trip_index=d["trip_index"],
        purpose=d["purpose"],
        main_mode=d["main_mode"],
        origin_zone=d["origin_zone"],
        destination_zone=d["destination_zone"],
        depart_time=d["depart_time"],
        arrive_time=d["arrive_time"],
        distance_km=d["distance_km"],
        is_main_trip=d.get("is_main_trip", True),
    )


def record_to_dict(rec: DiaryRecord) -> dict:
    d = asdict(rec)
    d["record_type"] = "diary_record"
    return d


def record_from_dict(d: dict) -> DiaryRecord:
    return DiaryRecord(
        record_id=d["record_id"],
        household=household_from_dict(d["household"]),
        individual=individual_from_dict(d["individual"]),
        trips=[trip_from_dict(t) for t in d["trips"]],
        day_index=d["day_index"],
        day_type=d["day_type"],
        synthetic=d.get("synthetic", True),
        schema_version=d.get("schema_version", SCHEMA_VERSION),
    )


def make_header_line(schema_version: str = SCHEMA_VERSION) -> dict:
    """First line of every JSONL file this module writes: a self-describing
    DEV-status header. Still valid JSON (so the file stays valid JSONL);
    `read_jsonl` recognizes and skips it via record_type == "meta"."""
    return {
        "record_type": "meta",
        "dev_only": True,
        "synthetic": True,
        "schema_version": schema_version,
        "note": DEV_NOTE,
    }


def write_jsonl(
    records: Iterable[DiaryRecord],
    path: Union[str, Path],
    include_header: bool = True,
) -> int:
    """Writes records as JSONL (one JSON object per line). Returns the
    number of DiaryRecord lines written (the header line, if any, is not
    counted)."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        if include_header:
            f.write(json.dumps(make_header_line(), sort_keys=True) + "\n")
        for rec in records:
            f.write(json.dumps(record_to_dict(rec), sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Union[str, Path]) -> Iterator[DiaryRecord]:
    """Yields DiaryRecord objects from a JSONL file written by
    `write_jsonl`, transparently skipping any meta/header line(s)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("record_type") == "meta":
                continue
            yield record_from_dict(d)
