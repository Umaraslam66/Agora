"""EVAL-ONLY unmasked vocabulary for the E5(i) contamination probe — QUARANTINED.

This module DELIBERATELY names the real anchoring arena: it is the mapping the
E5(i) unmasked arm substitutes into the persona-evidence data (sealed A2.3 /
A1.4; M2 architecture spec D9) so a masked-vs-unmasked A/B can measure whether
the model's memorized knowledge of the real event leaks into its behavior.

QUARANTINE: it lives inside ``evaluation/truth/`` and inherits the package
tripwire (importable only with AGORA_EVAL_CONTEXT=1); the static import
boundary (tests/test_truth_import_boundary.py) guarantees that grounding/,
agents/, serving/, world/ (and calibration/, training/) can never import it.
Nothing outside the E5 evaluation harness may read these strings — one leak
into a prompt template or an agent-facing literal and the masked arm is dead.

Zone naming: the 30 City K v2 zone codes are named after the real geography
each ring stands for, per the committed re-pin note (docs/M2_RING_REPIN.md §2):
core Z01-Z06 (center-city around the corridor's downtown segment), inner
Z07-Z14 (rest of the city ring west of the big lake), outer_north Z15-Z20,
outer_south Z21-Z26, east_water Z27-Z30 (across-water: the Eastside plus the
ferry-served peninsula/islands). Coarse district/city names — precise enough
to be recognizable, coarse enough to match a 6-zone-per-ring resolution.

Facility/era/price naming: real facility names for the world's corridor
facility codes (world/config.py), real calendar dates for the four scripted
eras (world/network.py ERA_LABELS), real currency, and the real introductory
toll schedule (WSTC-adopted two-axle pass-holder rates in effect at tolling
start) keyed by the world's toll-period vocabulary (world/tolling.py PERIODS).
Sources: docs/internal/SEATTLE_TRUTH_NOTES.md §§1-2 (harness-side, private).
"""
from __future__ import annotations

from typing import Dict, Mapping

VOCABULARY_VERSION = "e5-unmasked-1.0"

# --- zone codes -> real place names (30 zones, five rings) -------------------

ZONE_NAMES: Dict[str, str] = {
    # core (Z01-Z06): center-city Seattle around the corridor's downtown segment
    "Z01": "downtown Seattle (the central business district)",
    "Z02": "Pioneer Square and the central Seattle waterfront",
    "Z03": "Belltown and the Denny Triangle, Seattle",
    "Z04": "South Lake Union, Seattle",
    "Z05": "First Hill and Capitol Hill, Seattle",
    "Z06": "lower Queen Anne and Seattle Center",
    # inner (Z07-Z14): the rest of the Seattle urban ring, west of Lake Washington
    "Z07": "Ballard, northwest Seattle",
    "Z08": "Fremont and Wallingford, Seattle",
    "Z09": "Greenwood and Green Lake, north Seattle",
    "Z10": "the University District, Seattle",
    "Z11": "Ravenna and Lake City, northeast Seattle",
    "Z12": "Magnolia and Interbay, Seattle",
    "Z13": "West Seattle",
    "Z14": "SoDo, Georgetown and South Park along the Duwamish, south Seattle",
    # outer_north (Z15-Z20): Snohomish County + north King suburbs
    "Z15": "Shoreline and Lake Forest Park, north King County",
    "Z16": "Edmonds and Mountlake Terrace, Snohomish County",
    "Z17": "Lynnwood, Snohomish County",
    "Z18": "Bothell and Mill Creek, on the King-Snohomish line",
    "Z19": "Everett, Snohomish County",
    "Z20": "Marysville and north Snohomish County",
    # outer_south (Z21-Z26): Pierce County + south King
    "Z21": "Burien, SeaTac and Tukwila, south King County",
    "Z22": "Renton, south King County",
    "Z23": "Kent and Covington, south King County",
    "Z24": "Auburn, south King County",
    "Z25": "Federal Way and Des Moines, south King County",
    "Z26": "Tacoma and central Pierce County",
    # east_water (Z27-Z30): across-water ring — Eastside + Kitsap + islands
    "Z27": "Bellevue and Mercer Island, on the Eastside",
    "Z28": "Kirkland and Redmond, on the Eastside",
    "Z29": "Issaquah and Sammamish, east King County",
    "Z30": "Bremerton, Bainbridge Island and the Kitsap Peninsula, across Puget Sound",
}

#: The Z00 placeholder (tract missing/unknown to the committed map) still gets
#: a real-region rendering in the unmasked arm — the arm's whole point is that
#: the model knows where it is.
UNKNOWN_ZONE_NAME = "an unrecorded location in the Seattle-Puget Sound region"


def zone_name(zone_code) -> str:
    """Real place name for a City K zone code (Z00/unknown -> region name)."""
    return ZONE_NAMES.get(str(zone_code), UNKNOWN_ZONE_NAME)


def unmask_skeleton(skeleton: Mapping) -> dict:
    """A copy of a persona skeleton with zone codes replaced by real place
    names (data-level substitution per spec D9; every other field passes
    through unchanged — the render path itself is never forked)."""
    out = dict(skeleton)
    for field in ("home_zone", "work_zone"):
        if out.get(field) is not None:
            out[field] = zone_name(out[field])
    return out


# --- facility codes -> real facility names (world/config.py facility set) ----

FACILITY_NAMES: Dict[str, str] = {
    "T": "the SR 99 tunnel (the deep-bore tunnel under downtown Seattle)",
    "S": "the Alaskan Way surface street along the Seattle waterfront",
    "F": "Interstate 5 (I-5) through downtown Seattle",
    "D": "the downtown Seattle street grid",
    "V": "the Alaskan Way Viaduct (the old elevated SR 99 highway)",
    "W": "the Lake Washington floating bridges and the Puget Sound ferries",
}

# --- currency, era dates, toll prices ----------------------------------------

CURRENCY = "dollars"

#: Real calendar spans for the four scripted eras (world/network.py
#: ERA_LABELS), from the harness-side truth notes (§1 timeline).
ERA_DATES: Dict[str, str] = {
    "elevated": "before January 11, 2019 (the Alaskan Way Viaduct still carrying SR 99)",
    "squeeze": "January 11 to February 3, 2019 (the Seattle Squeeze: viaduct closed, "
               "tunnel not yet open)",
    "free_tunnel": "February 4 to November 8, 2019 (the SR 99 tunnel open toll-free)",
    "toll_on": "from November 9, 2019 (the SR 99 tunnel tolled)",
}

#: Real introductory two-axle pass-holder toll rates (in effect at tolling
#: start), keyed by the world's toll-period vocabulary (world/tolling.py
#: PERIODS). The world's masked schedule is the perturbed analogue of these.
TOLL_PRICES_DOLLARS: Dict[str, float] = {
    "overnight": 1.00,
    "am_peak": 1.50,
    "pm_peak": 2.25,
    "offpeak": 1.25,
}

#: Real per-trip surcharge for drivers without a Good To Go! account
#: (Pay By Mail) — the analogue of the world's non-pass surcharge.
NONPASS_SURCHARGE_DOLLARS = 2.00

TOLL_PRICE_STRINGS: Dict[str, str] = {
    period: f"${amount:.2f}" for period, amount in TOLL_PRICES_DOLLARS.items()
}

# --- the documented evidence-line string mapping (spec D9) --------------------
# Applied to the MASKED evidence lines before rendering: each key is a masked
# phrase the seeding pipeline deliberately writes (grounding/seeding.py
# _FREQ_MODE_PHRASE / _FREQ_LABEL_PHRASE and the recorded-day wording); the
# value restores the real-world referent. Longest-match-first application, so
# a longer masked phrase can safely contain a shorter one.

EVIDENCE_LINE_MAP: Dict[str, str] = {
    # masked self-report mode phrases -> the real services behind them
    "by shared transit": "by transit (a King County Metro bus, Sound Transit "
                         "light rail, or a Washington State Ferry)",
    "by hired ride": "by a hired ride (Uber, Lyft, or a Seattle taxi)",
    "by shared car": "by shared car (car2go or ReachNow carshare in Seattle)",
    # masked recall-window phrases -> the survey's real recall wording
    "never over a recent stretch": "never in the past 30 days",
    "a few times over a recent month": "on 1-3 days in the past month",
    "a few times a month": "a few times per month",
    # masked survey framing -> the real survey identity (this phrase appears in
    # every persona's evidence block, so every unmasked prompt names the arena
    # even when its zones fall back to Z00)
    "recorded weekday": "recorded Puget Sound Regional Council household "
                        "travel survey weekday (Seattle region, spring 2017 "
                        "or spring 2019)",
}

_MAP_LONGEST_FIRST = sorted(EVIDENCE_LINE_MAP, key=len, reverse=True)


def unmask_text(text: str) -> str:
    """Apply the documented evidence-line mapping (longest masked phrase
    first) to one masked evidence line / block of text."""
    for masked in _MAP_LONGEST_FIRST:
        text = text.replace(masked, EVIDENCE_LINE_MAP[masked])
    return text
