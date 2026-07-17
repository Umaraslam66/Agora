"""A4.1 E7 information-tier evidence builder (T1..T5 + the two T4 arms).

The FULL deployed population is rebuilt at five NESTED information tiers,
each tier the previous plus one increment of evidence handed to the SAME
generation pipeline (render -> generate -> gate -> assemble, with the
deterministic fallback backstop), CRN-paired across tiers: identical persona
set, sampling seeds, temperature-0 decoding — only the evidence bundle grows,
so tier-to-tier differences are information, not sampling noise.

Tier increments (canonical block order [world][stated][diary], recorded in
the E7 manifest):

  T1  demographics    skeleton only; no evidence lines.
  T2  +context        the masked world the persona inhabits: zone/ring
                      placement, corridor/water membership, access, mode
                      availability. Placement without personal behavior.
  T3  +stated claims  the A3.1 recall-channel module: stated typical commute
                      mode, stated commute days/week, past-30-day mode
                      frequencies, stated telework, residence-factor
                      importance. What the person SAYS, not what they DID.
  T4  +one observed day   ONE weekday diary day, selected by the CRN-fixed
                      rule pinned here: ``pick_weighted`` under key
                      ``"e7:{persona_id}:t4day"``, weighted by day weight.
  T5  +full trace     the full multi-day diary (all diary sections).

  T4-noclaims    T2 + the one observed day (stated claims WITHHELD): the
                 claims-given-day contrast arm, scored on ordinary days AND
                 BT1 (T4 - T4-noclaims = marginal value of stated claims
                 given behavior).
  T4-nofidelity  T3 + the one observed day, generation fidelity gate OFF:
                 ordinary-day-only diagnostic decomposing the day's raw
                 information from the verification the gate performs
                 (T4 - T4-nofidelity = the gate's contribution).

Fidelity-gate applicability (A4.1): the gate applies only where a diary is in
evidence and is anchored to the VISIBLE diary — T4/T4-noclaims to the one
shown day, T5 to the full diary; T1-T3 and T4-nofidelity validate with an
empty observed reference (fidelity no-ops exactly as it does for fallback
cards). The borrowed-car feasibility relaxation likewise reads the visible
diary only — a tier that cannot see car driving cannot claim it.

Diary sections are produced by the ONE existing evidence builder
(`seeding.evidence_lines_of`) on the tier-visible subset of person-days and
trips, so a T4 arm's histograms/purpose lines describe the one shown day and
T5's describe the full trace — nesting by construction, no second evidence
path.

Deterministic tier fallbacks (the terminal backstop must never see more than
its tier): T1/T2 build a minimal availability-respecting skeleton card;
T3 additionally uses the stated typical mode and stated commute days/week;
the T4 arms and T5 use the ordinary `card_validation.fallback_card` on the
visible diary subset.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from grounding import seeding
from world.crn import pick_weighted
from world.geometry import RING_INDEX, ZONE_RING, access_minutes, is_corridor_od, is_water_crossing

TIERS: Tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5", "T4_noclaims", "T4_nofidelity")

#: Tiers whose generation-time validation carries the fidelity reference
#: (anchored to the tier-visible diary).
FIDELITY_TIERS = frozenset({"T4", "T5", "T4_noclaims"})

#: Tiers that show the single CRN-selected observed day.
ONE_DAY_TIERS = frozenset({"T4", "T4_noclaims", "T4_nofidelity"})

#: Tiers that include the stated-claims module.
CLAIMS_TIERS = frozenset({"T3", "T4", "T5", "T4_nofidelity"})

#: Tiers that include the world-context block (all but bare demographics).
CONTEXT_TIERS = frozenset({"T2", "T3", "T4", "T5", "T4_noclaims", "T4_nofidelity"})

#: The pinned CRN day-selection site (E7 manifest).
T4_DAY_SITE = "e7:{persona_id}:t4day"

#: stated typical commute mode -> masked phrase (five-mode vocabulary; the
#: carpool labels resolve by licence exactly as the E3 fit does).
_WORK_MODE_PHRASE: Dict[str, str] = {
    "car": "by car",
    "ride": "by hired or shared ride",
    "transit": "by shared transit",
    "walk": "on foot",
    "bike": "by bike",
}

_TELEWORK_PHRASE: Dict[str, str] = {
    "Never": "never",
    "Less than monthly": "rarely",
    "A few times per month": "a few times a month",
    "1 day a week": "about one day a week",
    "2 days a week": "about two days a week",
    "3 days a week": "several days a week",
    "4 days a week": "most days a week",
    "5 days a week": "most weekdays",
    "6-7 days a week": "almost daily",
}

_IMPORTANCE_PHRASE: Dict[str, str] = {
    "Very unimportant": "not at all important",
    "Somewhat unimportant": "not very important",
    "Neither or N/A": "neither important nor unimportant",
    "Somewhat important": "somewhat important",
    "Very important": "very important",
}


# ---------------------------------------------------------------------------
# T2: world-context lines
# ---------------------------------------------------------------------------

def world_context_lines(skeleton: Mapping) -> List[str]:
    """Placement without personal behavior: zone/ring geometry, corridor and
    water-crossing membership, access, mode availability. Masked vocabulary
    only (zone codes, generic ring words)."""
    lines: List[str] = []
    hz, wz = skeleton.get("home_zone"), skeleton.get("work_zone")
    h_ring, w_ring = ZONE_RING.get(hz), ZONE_RING.get(wz)
    if hz:
        lines.append(
            f"Lives in zone {hz}"
            + (f", in the {h_ring} ring of City K." if h_ring else " of City K.")
        )
    if wz and wz != "Z00":
        lines.append(
            f"Works in zone {wz}" + (f", in the {w_ring} ring." if w_ring else ".")
        )
    elif skeleton.get("employed"):
        lines.append("Works at a location without a fixed zone record.")
    if h_ring and w_ring:
        hi, wi = RING_INDEX.get(h_ring, -1), RING_INDEX.get(w_ring, -1)
        if hi >= 0 and wi >= 0:
            if bool(is_corridor_od(hi, wi)):
                lines.append(
                    "The home-to-work journey runs along the city's commute "
                    "corridor, which offers one fast crossing route and "
                    "slower parallel alternatives."
                )
            if bool(is_water_crossing(hi, wi)):
                lines.append("The journey includes a water crossing.")
    if hz and wz and wz != "Z00":
        try:
            from world.geometry import ZONE_INDEX
            am = float(access_minutes(ZONE_INDEX.get(hz, 0), ZONE_INDEX.get(wz, 0)))
            lines.append(f"Door-to-network access for the commute is about "
                         f"{am:.0f} minutes in total.")
        except Exception:
            pass
    lines.append(
        "Walking, cycling, shared transit, driving, and riding along are all "
        "available ways to get around City K."
    )
    return lines


# ---------------------------------------------------------------------------
# T3: stated-claims lines (the A3.1 recall-channel module)
# ---------------------------------------------------------------------------

def stated_claims_lines(person_row: Mapping, skeleton: Mapping) -> List[str]:
    """The A3.1 items, masked. Extends the M2 bundle's self-report block with
    the typical-commute-mode, telework, and residence-importance items the
    A4.1 T3 definition names.

    ``stated_typical_mode`` must arrive as an already-masked five-mode token —
    the raw survey label carries real agency vocabulary and is mapped
    HARNESS-SIDE (the tier-build driver uses the frozen
    `calibration.e3_recall_fit.WORK_MODE_MAP`); agent-facing code never sees
    the raw label (mask discipline)."""
    lines: List[str] = []
    typical = seeding._val(person_row, "stated_typical_mode")
    if typical is not None and str(typical) in _WORK_MODE_PHRASE:
        lines.append(f"Self-reports usually commuting {_WORK_MODE_PHRASE[str(typical)]}.")
    commute_phrase = seeding._freq_phrase(seeding._val(person_row, "commute_freq"))
    if commute_phrase is not None:
        lines.append(f"Self-reports commuting to a usual workplace {commute_phrase}.")
    for col, mode_phrase in seeding._FREQ_MODE_PHRASE.items():
        phrase = seeding._freq_phrase(seeding._val(person_row, col))
        if phrase is not None:
            lines.append(f"Self-reports getting around {mode_phrase} {phrase}.")
    tw = seeding._val(person_row, "telecommute_freq")
    if tw is not None and str(tw) in _TELEWORK_PHRASE:
        lines.append(f"Self-reports working from home {_TELEWORK_PHRASE[str(tw)]}.")
    for col, what in (
        ("res_factors_transit", "access to shared transit"),
        ("res_factors_walk", "being able to walk to places"),
    ):
        v = seeding._val(person_row, col)
        if v is not None and str(v) in _IMPORTANCE_PHRASE:
            lines.append(
                f"When choosing where to live, rated {what} as "
                f"{_IMPORTANCE_PHRASE[str(v)]}."
            )
    return lines


# ---------------------------------------------------------------------------
# T4 day selection (pinned CRN rule)
# ---------------------------------------------------------------------------

def t4_day_of(persona_id: str, daynums: Sequence[int], weights: Sequence[float]) -> int:
    """The CRN-fixed T4 day: one observed weekday, drawn once per persona
    under the pinned site key, weighted by day weight. Deterministic."""
    key = T4_DAY_SITE.format(persona_id=persona_id)
    return int(pick_weighted(key, list(daynums), list(weights)))


# ---------------------------------------------------------------------------
# the tier evidence bundle
# ---------------------------------------------------------------------------

def tier_evidence(
    tier: str,
    skeleton: Mapping,
    person_days,
    trips,
    person_row: Mapping,
    selected_daynum: Optional[int],
) -> Tuple[List[str], object, object, int]:
    """(evidence_lines, visible_person_days, visible_trips, n_observed_days)
    for one persona at one tier. ``visible_*`` are the diary frames the tier
    may see — the SAME frames the fidelity reference and the tier fallback
    must be computed from (never the full frames for a one-day tier)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    pd_df = seeding._as_frame(person_days)
    tr_df = seeding._as_frame(trips)

    if tier in ONE_DAY_TIERS:
        if selected_daynum is None:
            raise ValueError(f"{tier} needs the CRN-selected daynum")
        vis_days = pd_df[pd_df["daynum"].astype(int) == int(selected_daynum)]
        vis_trips = tr_df[tr_df["daynum"].astype(int) == int(selected_daynum)]
    elif tier == "T5":
        vis_days, vis_trips = pd_df, tr_df
    else:
        vis_days, vis_trips = pd_df.iloc[0:0], tr_df.iloc[0:0]

    lines: List[str] = []
    if tier in CONTEXT_TIERS:
        lines += world_context_lines(skeleton)
    if tier in CLAIMS_TIERS:
        lines += stated_claims_lines(person_row, skeleton)
    n_observed = int(len(vis_days))
    if n_observed:
        # diary sections ONLY (1-5): the self-report block (section 6) is the
        # tier's own stated-claims increment, so the diary builder gets a
        # person_row stripped to diary-derived fields — otherwise T4-noclaims
        # would smuggle the claims back in through the diary block.
        diary_row = {"n_weekend_days": seeding._val(person_row, "n_weekend_days")}
        lines += seeding.evidence_lines_of(vis_days, vis_trips, diary_row)
    return lines, vis_days, vis_trips, n_observed


def tier_fidelity_observed(tier: str, vis_days, vis_trips) -> dict:
    """The validation-time observed reference for one tier: the tier-visible
    diary stats for fidelity-gated tiers, an EMPTY reference otherwise (the
    fidelity gate no-ops on an empty reference, and the borrowed-car
    feasibility relaxation correctly sees no observed car driving)."""
    if tier in FIDELITY_TIERS and len(seeding._as_frame(vis_days)):
        return seeding.observed_stats_of(vis_days, vis_trips)
    return {}


# ---------------------------------------------------------------------------
# deterministic tier fallbacks (the terminal backstop, tier-blind)
# ---------------------------------------------------------------------------

def _skeleton_default_mode(skeleton: Mapping, stated_typical: Optional[str]) -> str:
    if stated_typical in _WORK_MODE_PHRASE:
        return str(stated_typical)
    cars = skeleton.get("household_cars")
    if (cars is None or (isinstance(cars, int) and cars >= 1)) and skeleton.get(
        "can_drive", True
    ):
        return "car"
    return "transit"


def tier_fallback_card(
    tier: str,
    persona_id: str,
    skeleton: Mapping,
    vis_days,
    vis_trips,
    person_row: Mapping,
) -> dict:
    """Deterministic fallback for one persona at one tier, consuming ONLY the
    tier-visible evidence (a backstop must never out-inform its tier):

    * diary tiers (T4 arms, T5): the ordinary empirical fallback on the
      visible subset;
    * T3: a commute pattern in the stated typical mode with quiet weight from
      the stated commute days/week;
    * T1/T2: a minimal availability-respecting skeleton card (an invented but
      deterministic prior — recorded as such in provenance).
    """
    from grounding.card_validation import assemble_card, fallback_card

    if tier in ONE_DAY_TIERS or tier == "T5":
        card = fallback_card(persona_id, skeleton, vis_days, vis_trips)
        card.setdefault("provenance", {})["e7_tier"] = tier
        return card

    stated_typical = seeding._val(person_row, "stated_typical_mode")
    mode = _skeleton_default_mode(skeleton, stated_typical)
    employed = bool(skeleton.get("employed"))
    patterns: List[dict] = []
    if employed:
        commute = {
            "id": "typical_day",
            "weight": 5,
            "trips": [
                {"purpose": "work", "mode": mode, "depart_band": "am_peak"},
                {"purpose": "home", "mode": mode, "depart_band": "pm_peak"},
            ],
        }
        quiet_w = 0
        if tier == "T3":
            days = seeding._val(person_row, "commute_freq")
            stated_days = _STATED_DAYS.get(str(days)) if days is not None else None
            if stated_days is not None:
                w = max(1, min(5, int(round(stated_days))))
                commute = dict(commute, weight=w)
                quiet_w = 5 - w
        patterns.append(commute)
        if quiet_w > 0:
            patterns.append({"id": "home_day", "weight": quiet_w, "trips": []})
    else:
        patterns.append({
            "id": "errand_day",
            "weight": 1,
            "trips": [
                {"purpose": "personal_business", "mode": "walk" if mode != "car" else mode,
                 "depart_band": "midday"},
                {"purpose": "home", "mode": "walk" if mode != "car" else mode,
                 "depart_band": "midday"},
            ],
        })
    obj = {"patterns": patterns, "rules": [], "voice": "Keeps to a simple, settled routine."}
    card = assemble_card(persona_id, skeleton, obj, {
        "card_source": "fallback", "e7_tier": tier,
        "fallback_kind": "skeleton_prior" if tier in ("T1", "T2") else "stated_prior",
    })
    return card


#: stated commute days/week label -> numeric (mirrors the frozen E3-fit map
#: for the labels a fallback can act on; generic labels only, mask-safe).
_STATED_DAYS: Dict[str, float] = {
    "6-7 days a week": 6.5,
    "5 days a week": 5.0,
    "4 days a week": 4.0,
    "3 days a week": 3.0,
    "2 days a week": 2.0,
    "1 day a week": 1.0,
    "A few times per month": 0.7,
    "Less than monthly": 0.2,
    "Never": 0.0,
}
