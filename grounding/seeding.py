"""Deterministic per-person evidence + prompt building for the M2 seeding step.

Every person in the pooled two-wave build seeds exactly one persona. This
module turns one person's masked diary into (a) a harness-computed demographic
skeleton and (b) a deterministic, masked habit-summary evidence block, then
renders the slow brain's ONE card-writing prompt through the single render path
(grounding.render.render_seed_prompt). Nothing here is LLM-generated: the whole
module is a pure function of the committed adapter build plus the frozen CRN
draw layer, so replay is bit-deterministic.

Masking discipline (pre-registration section 5; mask-lint gate): no real place
name, agency, calendar date, or bare wave-year ever appears in a literal, a
comment, an evidence line, or a rendered prompt. The two survey waves are only
ever referred to as "the earlier wave" / "the later wave" if referred to at
all; travel timing is described in departure bands, never clock times. The
harness-side persona index carries the person_id -> persona_id map (that map is
never handed to an agent).

Column contract for the two pure builders, so both are unit-testable on small
synthetic frames without the real adapter:

* ``skeleton_of(person, household)`` reads person/household row-like mappings
  (dict or pandas Series). ``person`` must carry ``persona_id`` (for the
  has_pass CRN key).
* ``evidence_lines_of(person_days, trips, person_row)`` takes ONE person's
  weekday person-day rows and ONE person's weekday trip rows, where ``trips``
  already carries a ``purpose`` in the frozen card purpose vocabulary, a
  ``mode`` in the frozen mode vocabulary, and a ``band`` in the frozen band
  vocabulary (plus ``daynum`` / ``tripnum`` for ordering). The real-data path
  (:func:`enriched_trips`) attaches purpose from the raw survey; the anti-replay
  constraint lives in the card lint, not here (sequences ARE evidence).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from grounding import zone_map
from grounding.adapters import psrc
from grounding.masking.mask_lint import (
    default_token_path,
    lint_text,
    load_forbidden_tokens,
)
from grounding.render import render_seed_prompt, render_seed_retry_prompt
from grounding.taxonomy import MODES
from world.config import get_config
from world.crn import draw as crn_draw

# The frozen departure-band and card-purpose vocabularies (echo the adapter /
# card schema; kept here as tuples so evidence ordering is deterministic).
BANDS = psrc.TIME_BANDS
PURPOSE_ORDER = (
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

# Harness-side skeleton fields, in a fixed order (echoed into the prompt and
# reconstructed from the persona index).
SKELETON_FIELDS = (
    "home_zone",
    "work_zone",
    "age",
    "employed",
    "student",
    "can_drive",
    "household_size",
    "household_cars",
    "income_class",
    "has_pass",
)

_ZONE_PLACEHOLDER = "Z00"  # until the committed tract->zone map is injected

# --- age band -> midpoint integer ------------------------------------------
_AGE_BAND_MIDPOINT: Dict[str, int] = {
    "Under 5 years old": 2,
    "5-11 years": 8,
    "12-15 years": 13,
    "16-17 years": 16,
    "18-24 years": 21,
    "25-34 years": 29,
    "35-44 years": 39,
    "45-54 years": 49,
    "55-64 years": 59,
    "65-74 years": 69,
    "75-84 years": 79,
    "85 years or older": 88,
}

_EMPLOYED_LABELS = frozenset(
    {
        "Employed full time (35+ hours/week, paid)",
        "Employed part time (fewer than 35 hours/week, paid)",
        "Self-employed",
    }
)

_STUDENT_ADULT_LABELS = frozenset({"Full-time student", "Part-time student"})

_ENROLLED_SCHOOLTYPES = frozenset(
    {
        "K-12 public school",
        "K-12 private school",
        "K-12 home school (full-time or part-time)",
        "College, graduate, or professional school",
        "Preschool",
        "Daycare",
        "Vocational/technical school",
    }
)

# --- purpose category (dest_purpose_cat) -> frozen card purpose -------------
_PURPOSE_OF_CAT: Dict[str, str] = {
    "Home": "home",
    "Work": "work",
    "Work-related": "work",
    "School": "education",
    "Shopping": "shop_other",  # refined to shop_daily when the detail is grocery
    "Meal": "leisure",
    "Social/Recreation": "leisure",
    "Personal Business/Errand/Appointment": "personal_business",
    "Escort": "pickup_dropoff",
    "Other": "other",
    "Change mode": "other",
    "Missing Response": "other",
}

# --- self-reported past-period mode-frequency columns ----------------------
_FREQ_MODE_PHRASE: Dict[str, str] = {
    "transit_freq": "by shared transit",
    "walk_freq": "on foot",
    "bike_freq": "by bike",
    "tnc_freq": "by hired ride",
    "carshare_freq": "by shared car",
}
_FREQ_LABEL_PHRASE: Dict[str, str] = {
    "Never in the past 30 days": "never over a recent stretch",
    "1-3 days in the past month": "a few times over a recent month",
    "1 day a week": "about one day a week",
    "2 days a week": "about two days a week",
    "3 days a week": "several days a week",
    "4 days a week": "most days a week",
    "2-4 days a week": "a few days a week",
    "5 days a week": "most weekdays",
    "6-7 days a week": "almost daily",
    "A few times per month": "a few times a month",
    "Less than monthly": "rarely",
}
_MISSING_FREQ = frozenset({"Missing Response", "", "nan", "None"})

# Human-readable trip-count words for the small-count histogram.
_COUNT_WORD = {
    0: "no",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
}


# ---------------------------------------------------------------------------
# small row-access helpers (work for dict OR pandas Series)
# ---------------------------------------------------------------------------

def _val(row: Mapping, key: str, default=None):
    """Fetch ``key`` from a dict/Series row, returning ``default`` when the key
    is absent or the value is missing (NaN/None)."""
    if row is None:
        return default
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        try:
            value = row.get(key, default)  # type: ignore[union-attr]
        except AttributeError:
            return default
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# The survey's household-size column is a label ("3 people"), not a count.
_COUNT_LABEL_RE = re.compile(r"^(\d+)\s")


def _count_or_none(value) -> Optional[int]:
    """An integer count from either a plain number or a leading-count label
    like ``"3 people"`` (strict: anything else is None)."""
    n = _int_or_none(value)
    if n is not None:
        return n
    if isinstance(value, str):
        m = _COUNT_LABEL_RE.match(value.strip())
        if m:
            return int(m.group(1))
    return None


def _tract_str(value) -> Optional[str]:
    """Normalize a raw tract identifier to the committed map's string form.

    The raw CSV column is float64, so ``str()`` alone would yield a trailing
    ``.0`` that can never match a map key; integral floats are collapsed to
    their integer digits."""
    if value is None:
        return None
    n = _int_or_none(value)
    if n is not None:
        return str(n)
    return str(value)


# ---------------------------------------------------------------------------
# scalar mappers (pure)
# ---------------------------------------------------------------------------

def pass_prior() -> float:
    """The frozen regional season-pass prior, read from the world config."""
    return float(get_config("cityk_corridor").pass_prior)


def age_midpoint(age_label) -> Optional[int]:
    """Band-midpoint integer for a survey age band, or None if unrecognized."""
    if age_label is None:
        return None
    return _AGE_BAND_MIDPOINT.get(str(age_label))


def is_employed(employment_label) -> bool:
    return str(employment_label) in _EMPLOYED_LABELS


def is_student(adult_student_label, schooltype_label) -> bool:
    if str(adult_student_label) in _STUDENT_ADULT_LABELS:
        return True
    return str(schooltype_label) in _ENROLLED_SCHOOLTYPES


def map_purpose(dest_purpose_cat, dest_purpose_detail=None) -> str:
    """Collapse a survey destination-purpose category to the frozen card
    purpose vocabulary. The Shopping category is split into daily vs other
    shopping using the detail string when available (never emitted, only read).
    """
    base = _PURPOSE_OF_CAT.get(str(dest_purpose_cat), "other")
    if base == "shop_other" and dest_purpose_detail is not None:
        if "grocery" in str(dest_purpose_detail).lower():
            return "shop_daily"
    return base


# ---------------------------------------------------------------------------
# skeleton
# ---------------------------------------------------------------------------

def skeleton_of(
    person: Mapping,
    household: Mapping,
    crn_namespace: str = "seed",
    zone_of_tract: Optional[Callable[[str], str]] = None,
) -> dict:
    """Deterministic, harness-computed demographic skeleton for one person.

    Zone codes: ``home_zone`` / ``work_zone`` resolve through the COMMITTED
    tract->zone map (grounding.zone_map, the A2.6 re-pin) by default; pass
    ``zone_of_tract`` only to override in tests. A missing tract or a tract
    the map does not know falls back to the ``Z00`` placeholder (work_zone
    is None when the person is not employed and no work tract is available).

    ``has_pass`` is CRN-drawn from the frozen world ``pass_prior`` under the key
    ``"{crn_namespace}:{persona_id}:has_pass"`` (default namespace "seed"), so
    it is stable and pairs across arms like every other draw.
    """
    persona_id = _val(person, "persona_id")

    # can_drive: prefer the adapter-derived boolean, else map the raw label.
    can_drive = _val(person, "can_drive_bool")
    if can_drive is None:
        raw = _val(person, "can_drive")
        can_drive = psrc.CAN_DRIVE_LABELS.get(str(raw), True) if raw is not None else True
    can_drive = bool(can_drive)

    employed = _val(person, "employed")
    if employed is None:
        employed = is_employed(_val(person, "employment"))
    employed = bool(employed)

    student = _val(person, "student")
    if student is None:
        student = is_student(_val(person, "adult_student"), _val(person, "schooltype"))
    student = bool(student)

    age = _val(person, "age_years")
    if age is None:
        age = age_midpoint(_val(person, "age"))
    age = _int_or_none(age)

    household_size = _count_or_none(
        _val(household, "hhsize", _val(household, "household_size"))
    )

    household_cars = _val(household, "household_cars")
    if household_cars is None:
        household_cars = psrc.VEHICLE_COUNT_OF_LABEL.get(str(_val(household, "vehicle_count")))
    household_cars = _int_or_none(household_cars)

    income_class = _int_or_none(_val(household, "income_class"))

    # zones — committed map by default (A2.6 re-pin), Z00 only when the
    # tract is absent or unknown to the map
    if zone_of_tract is None:
        zone_of_tract = zone_map.zone_of_tract
    home_tract = _tract_str(_val(household, "home_tract_2020"))
    work_tract = _tract_str(_val(person, "work_tract_2020"))
    home_zone = (zone_of_tract(home_tract) if home_tract is not None else None) \
        or _ZONE_PLACEHOLDER
    if work_tract is not None:
        work_zone = zone_of_tract(work_tract) or _ZONE_PLACEHOLDER
    elif employed:
        work_zone = _ZONE_PLACEHOLDER
    else:
        work_zone = None

    # has_pass CRN draw
    if persona_id is not None:
        key = f"{crn_namespace}:{persona_id}:has_pass"
        has_pass = crn_draw(key) < pass_prior()
    else:
        has_pass = False

    return {
        "home_zone": home_zone,
        "work_zone": work_zone,
        "age": age,
        "employed": employed,
        "student": student,
        "can_drive": can_drive,
        "household_size": household_size,
        "household_cars": household_cars,
        "income_class": income_class,
        "has_pass": bool(has_pass),
    }


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

def _fmt_count_list(pairs: Iterable) -> str:
    """Render ``[(label, count), ...]`` as ``"a x2, b x1"`` (already ordered)."""
    return ", ".join(f"{label} x{count}" for label, count in pairs)


def _ordered_counts(values: Iterable, vocabulary: Iterable) -> List:
    """Counts of ``values`` restricted to ``vocabulary`` order, nonzero only."""
    counts = Counter(values)
    return [(v, counts[v]) for v in vocabulary if counts.get(v, 0) > 0]


def _trip_count_word(n: int) -> str:
    return _COUNT_WORD.get(n, f"{n}")


def _day_sequence_line(day_trips: List[dict]) -> str:
    """A neutral one-day sequence line ('One recorded day: ...'). Sequences are
    evidence; the anti-replay guard is enforced in the card lint, not here."""
    if not day_trips:
        return "One recorded day: no trips."
    parts = [
        f"{t['purpose']} {t['band']} by {t['mode']}" for t in day_trips
    ]
    return "One recorded day: " + ", then ".join(parts) + "."


def evidence_lines_of(person_days, trips, person_row: Mapping) -> list[str]:
    """Deterministic masked habit summary for ONE person.

    ``person_days`` are the person's WEEKDAY person-day rows (zero-trip days
    kept); ``trips`` are that person's weekday trip rows already carrying
    ``purpose`` / ``mode`` / ``band`` / ``daynum`` / ``tripnum``. Same person ->
    byte-identical lines (all ordering is by frozen vocabulary or by
    daynum/tripnum).
    """
    pd_df = _as_frame(person_days)
    tr_df = _as_frame(trips)

    n_days = len(pd_df)
    lines: List[str] = []

    # 1. recorded-day counts (weekday summarized; weekend mentioned only)
    day_word = "day" if n_days == 1 else "days"
    lines.append(f"Contributed {n_days} recorded weekday {day_word}.")
    n_weekend = _int_or_none(_val(person_row, "n_weekend_days"))
    if n_weekend:
        w_word = "day" if n_weekend == 1 else "days"
        lines.append(
            f"Also recorded {n_weekend} weekend {w_word}, not summarized here."
        )
    else:
        lines.append("Any weekend days are recorded separately, not summarized here.")

    # 2. trips-per-weekday histogram (in words)
    if n_days:
        counts_col = pd_df["n_collapsed"] if "n_collapsed" in pd_df else pd_df["n_trips"]
        per_day = Counter(int(c) for c in counts_col)
        hist_parts = []
        for n_trips in sorted(per_day):
            d = per_day[n_trips]
            dd = "day" if d == 1 else "days"
            tt = "trip" if n_trips == 1 else "trips"
            hist_parts.append(f"{d} {dd} with {_trip_count_word(n_trips)} {tt}")
        lines.append("Trips per recorded weekday: " + "; ".join(hist_parts) + ".")

    # 3. per-purpose lines
    if len(tr_df):
        for purpose in PURPOSE_ORDER:
            sub = tr_df[tr_df["purpose"] == purpose]
            if not len(sub):
                continue
            n = len(sub)
            modes = _ordered_counts(sub["mode"], MODES)
            usual_band = _modal_in_vocab(sub["band"], BANDS)
            tt = "trip" if n == 1 else "trips"
            line = (
                f"Purpose {purpose}: {n} {tt}; modes {_fmt_count_list(modes)}; "
                f"usually departs in the {usual_band}."
            )
            lines.append(line)

    # 4. overall weekday mode-use counts
    if len(tr_df):
        mode_counts = _ordered_counts(tr_df["mode"], MODES)
        lines.append("Overall weekday mode use: " + _fmt_count_list(mode_counts) + ".")
        band_counts = _ordered_counts(tr_df["band"], BANDS)
        lines.append("Overall departure timing: " + _fmt_count_list(band_counts) + ".")

    # 5. multi-day (rMove) per-day neutral sequences
    daynums = sorted({int(d) for d in pd_df["daynum"]}) if "daynum" in pd_df else []
    if len(daynums) > 1:
        lines.append("Recorded day sequences:")
        by_day = _trips_by_day(tr_df)
        for daynum in daynums:
            lines.append(_day_sequence_line(by_day.get(daynum, [])))

    # 6. self-reported past-period mode frequencies (masked wording)
    for col, mode_phrase in _FREQ_MODE_PHRASE.items():
        phrase = _freq_phrase(_val(person_row, col))
        if phrase is not None:
            lines.append(f"Self-reports getting around {mode_phrase} {phrase}.")
    commute_phrase = _freq_phrase(_val(person_row, "commute_freq"))
    if commute_phrase is not None:
        lines.append(f"Self-reports commuting to a usual workplace {commute_phrase}.")

    return lines


def _freq_phrase(label) -> Optional[str]:
    if label is None:
        return None
    key = str(label)
    if key in _MISSING_FREQ:
        return None
    return _FREQ_LABEL_PHRASE.get(key)


def _modal_in_vocab(values, vocabulary) -> str:
    counts = Counter(values)
    best = None
    best_n = -1
    for v in vocabulary:  # frozen order = deterministic tie-break
        n = counts.get(v, 0)
        if n > best_n:
            best_n = n
            best = v
    return best


def _trips_by_day(tr_df) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    if not len(tr_df):
        return out
    ordered = tr_df.sort_values(["daynum", "tripnum"]) if "tripnum" in tr_df else tr_df.sort_values(["daynum"])
    for daynum, grp in ordered.groupby("daynum"):
        out[int(daynum)] = [
            {"purpose": r.purpose, "mode": r.mode, "band": r.band}
            for r in grp.itertuples(index=False)
        ]
    return out


def _as_frame(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame(list(obj))


def observed_stats_of(person_days, trips) -> dict:
    """Deterministic per-person reference stats for the card fidelity gate,
    computed from the SAME two frames :func:`evidence_lines_of` consumes: the
    person's WEEKDAY person-day rows and that person's weekday trip rows.

    Weighting convention mirrors the E1 truth (M2 D5, evaluation.e1): trips/day
    and the quiet share are DAY-weighted by ``w_day`` (absent -> uniform weight
    1.0, so under the synthetic fixtures the day-weighted figures collapse to
    the raw figures the evidence lines report), and the mode reference is the
    day-weighted trip mix. That is exactly the quantity a faithful card
    reproduces: a card's weight-normalized pattern mix IS a day-weighted mix
    (pattern weight ~ day-kind frequency, pattern trips = that day's trips), so
    a card that compresses the evidence honestly matches these references
    within the fidelity tolerances. Per-day trip counts are taken from the trip
    rows — the same rows :func:`grounding.card_validation.day_signatures` and
    the evidence sequences enumerate — so the reference and the card count trips
    the same way; the raw ``mode_counts`` echo the "Overall weekday mode use"
    evidence line verbatim.

    Returns a dict with keys: ``n_observed_weekdays`` (positive-weight weekday
    person-day rows), ``n_observed_trips`` (raw weekday trip count),
    ``mean_trips_per_weekday`` (day-weighted mean), ``mode_counts``
    ({mode: raw count}), ``mode_shares`` ({mode: day-weighted share}),
    ``has_quiet_weekday`` (any observed weekday had zero trips) and
    ``quiet_share`` (day-weighted share of zero-trip weekdays).
    """
    pd_df = _as_frame(person_days)
    tr_df = _as_frame(trips)

    # per-day trip counts, keyed by daynum, from the trip rows (the same rows
    # the card patterns enumerate).
    trips_per_daynum: "Counter[int]" = Counter()
    if len(tr_df) and "daynum" in tr_df:
        for d in tr_df["daynum"]:
            trips_per_daynum[int(d)] += 1

    # observed weekday days -> day weight (positive weight only; absent w_day
    # column => uniform weight 1.0). person_days carries one row per day, so a
    # later row for the same daynum simply overwrites (idempotent).
    day_weight: Dict[int, float] = {}
    if "daynum" in pd_df and len(pd_df):
        has_w = "w_day" in pd_df.columns
        daynum_col = list(pd_df["daynum"])
        w_col = list(pd_df["w_day"]) if has_w else None
        for i, dn_raw in enumerate(daynum_col):
            dn = int(dn_raw)
            if has_w:
                wv = w_col[i]
                try:
                    if pd.isna(wv):
                        continue
                except (TypeError, ValueError):
                    pass
                w = float(wv)
            else:
                w = 1.0
            if w <= 0:
                continue
            day_weight[dn] = w

    n_observed = len(day_weight)
    total_w = sum(day_weight.values())

    empty_stats = {
        "n_observed_weekdays": n_observed,
        "n_observed_trips": int(len(tr_df)),
        "mean_trips_per_weekday": 0.0,
        "mode_counts": {},
        "mode_shares": {},
        "has_quiet_weekday": False,
        "quiet_share": 0.0,
    }
    if n_observed == 0 or total_w <= 0:
        return empty_stats

    mean_trips = sum(
        w * trips_per_daynum.get(dn, 0) for dn, w in day_weight.items()
    ) / total_w

    quiet_days = [dn for dn in day_weight if trips_per_daynum.get(dn, 0) == 0]
    has_quiet = bool(quiet_days)
    quiet_share = sum(day_weight[dn] for dn in quiet_days) / total_w

    # mode reference: raw counts (evidence line) + day-weighted shares (gate).
    mode_counts: "Counter[str]" = Counter()
    mode_wmass: Dict[str, float] = {}
    if len(tr_df) and "mode" in tr_df:
        has_dn = "daynum" in tr_df
        mode_col = list(tr_df["mode"])
        dn_col = list(tr_df["daynum"]) if has_dn else None
        for i, m in enumerate(mode_col):
            mode_counts[m] += 1
            w = day_weight.get(int(dn_col[i]), 0.0) if has_dn else 1.0
            if w > 0:
                mode_wmass[m] = mode_wmass.get(m, 0.0) + w
    total_mass = sum(mode_wmass.values())
    mode_shares = (
        {m: mode_wmass[m] / total_mass for m in mode_wmass} if total_mass > 0 else {}
    )

    return {
        "n_observed_weekdays": n_observed,
        "n_observed_trips": int(len(tr_df)),
        "mean_trips_per_weekday": float(mean_trips),
        "mode_counts": dict(mode_counts),
        "mode_shares": mode_shares,
        "has_quiet_weekday": has_quiet,
        "quiet_share": float(quiet_share),
    }


# ---------------------------------------------------------------------------
# real-data enrichment (harness-side; reads the gitignored raw survey CSVs)
# ---------------------------------------------------------------------------

_PERSON_ATTR_COLS = [
    "person_id",
    "age",
    "employment",
    "adult_student",
    "schooltype",
    "work_tract_2020",
    "transit_freq",
    "walk_freq",
    "bike_freq",
    "tnc_freq",
    "carshare_freq",
    "commute_freq",
    # A4.1 T3 stated-claims module (the A3.1 recall-channel items): the tier
    # builder renders these; the M2 evidence path ignores them.
    "work_mode",
    "telecommute_freq",
]
_HH_ATTR_COLS = ["household_id", "hhsize", "home_tract_2020"]


def enriched_person_attrs(data_dir=psrc.DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the extra per-person attribute columns the narrow adapter omits
    (age band, employment, student status, work tract, self-reported mode
    frequencies), restricted to the two pooled waves."""
    path = Path(data_dir) / psrc._PERSONS_CSV
    have = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in _PERSON_ATTR_COLS if c in have]
    df = pd.read_csv(
        path, usecols=use + ["survey_year"], dtype={"person_id": str}, low_memory=False
    )
    df = df[df.survey_year.isin(psrc.WAVES)].drop(columns=["survey_year"])
    return df


def enriched_household_attrs(data_dir=psrc.DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the extra per-household attribute columns (household size, home
    tract) the narrow adapter omits, restricted to the two pooled waves."""
    path = Path(data_dir) / psrc._HOUSEHOLDS_CSV
    have = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in _HH_ATTR_COLS if c in have]
    df = pd.read_csv(
        path, usecols=use + ["survey_year"], dtype={"household_id": str}, low_memory=False
    )
    df = df[df.survey_year.isin(psrc.WAVES)].drop(columns=["survey_year"])
    return df


def _read_raw_trip_purpose(data_dir, person_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """trip_id -> mapped card purpose, from the raw trips CSV. When
    ``person_ids`` is given the (large) file is read in filtered chunks so a
    few-person build stays fast; otherwise it is read once in full."""
    path = Path(data_dir) / psrc._TRIPS_CSV
    have = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in ("trip_id", "person_id", "survey_year", "dest_purpose_cat", "dest_purpose") if c in have]
    waves = set(psrc.WAVES)
    id_set = set(map(str, person_ids)) if person_ids is not None else None

    def _map(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame[frame.survey_year.isin(waves)]
        if id_set is not None:
            frame = frame[frame.person_id.isin(id_set)]
        if not len(frame):
            return frame.assign(purpose=[])
        detail = frame["dest_purpose"] if "dest_purpose" in frame else None
        purpose = [
            map_purpose(cat, detail.iloc[i] if detail is not None else None)
            for i, cat in enumerate(frame["dest_purpose_cat"])
        ]
        return frame.assign(purpose=purpose)[["trip_id", "purpose"]]

    if id_set is None:
        raw = pd.read_csv(path, usecols=use, dtype={"trip_id": str, "person_id": str}, low_memory=False)
        return _map(raw)
    parts = []
    for chunk in pd.read_csv(
        path, usecols=use, dtype={"trip_id": str, "person_id": str}, low_memory=False, chunksize=200_000
    ):
        mapped = _map(chunk)
        if len(mapped):
            parts.append(mapped)
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame({"trip_id": pd.Series(dtype=str), "purpose": pd.Series(dtype=str)})


def enriched_trips(dataset, data_dir=psrc.DEFAULT_DATA_DIR, person_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """The adapter's weekday trips joined with a card-vocabulary ``purpose``.

    Returns columns person_id, daynum, tripnum, purpose, mode, band, w_trip
    (restricted to ``person_ids`` when given). Purpose resolution, in order:
    a ``purpose`` column already on the trips frame (synthetic fixtures) wins;
    else the raw trips CSV is joined on trip_id; else (no raw data available)
    every trip falls back to purpose ``"other"``.
    """
    trips = dataset.weekday_trips
    if person_ids is not None:
        id_set = set(map(str, person_ids))
        trips = trips[trips.person_id.astype(str).isin(id_set)]
    keep = [
        c
        for c in ("trip_id", "person_id", "daynum", "tripnum", "mode", "band", "w_trip", "purpose")
        if c in trips.columns
    ]
    trips = trips[keep].copy()
    trips["person_id"] = trips["person_id"].astype(str)
    if "purpose" in trips.columns:
        return trips
    raw_csv = (Path(data_dir) / psrc._TRIPS_CSV) if data_dir is not None else None
    if raw_csv is None or not raw_csv.exists() or "trip_id" not in trips.columns:
        trips["purpose"] = "other"
        return trips
    purpose = _read_raw_trip_purpose(data_dir, person_ids)
    trips["trip_id"] = trips["trip_id"].astype(str)
    merged = trips.merge(purpose, on="trip_id", how="left")
    merged["purpose"] = merged["purpose"].fillna("other")
    return merged


# ---------------------------------------------------------------------------
# persona index
# ---------------------------------------------------------------------------

def _persona_id_map(person_ids: Iterable[str]) -> Dict[str, str]:
    """Masked reindex: persona_id = 'P' + zero-padded rank of person_id in
    sorted order. Stable for a fixed person-id set."""
    ids = sorted(set(map(str, person_ids)))
    pad = max(5, len(str(len(ids))))
    return {pid: f"P{rank:0{pad}d}" for rank, pid in enumerate(ids, start=1)}


def persona_index(
    dataset,
    data_dir=psrc.DEFAULT_DATA_DIR,
    zone_of_tract: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    """One row per person, with the masked persona_id, harness-side keys
    (person_id, household_id, wave, fold, segment) and the flattened skeleton
    fields. person_id is kept alongside — this frame is harness-side only."""
    persons = dataset.persons.copy()
    persons["person_id"] = persons["person_id"].astype(str)
    persons["household_id"] = persons["household_id"].astype(str)

    id_map = _persona_id_map(persons["person_id"])
    persons["persona_id"] = persons["person_id"].map(id_map)

    # enrich with the columns the narrow adapter omits, when the raw CSVs exist
    if data_dir is not None and (Path(data_dir) / psrc._PERSONS_CSV).exists():
        pa = enriched_person_attrs(data_dir)
        pa["person_id"] = pa["person_id"].astype(str)
        keep = ["person_id"] + [c for c in pa.columns if c not in persons.columns]
        persons = persons.merge(pa[keep], on="person_id", how="left")
    if data_dir is not None and (Path(data_dir) / psrc._HOUSEHOLDS_CSV).exists():
        ha = enriched_household_attrs(data_dir)
        ha["household_id"] = ha["household_id"].astype(str)
    else:
        ha = None

    households = dataset.households.copy()
    households["household_id"] = households["household_id"].astype(str)
    hh_cols = [
        c
        for c in (
            "household_id",
            "income_class",
            "household_cars",
            "vehicle_count",
            "hhsize",
            "home_tract_2020",
        )
        if c in households
    ]
    hh = households[hh_cols]
    if ha is not None:
        # never let the raw-CSV enrichment shadow columns the dataset frame
        # already carries (a suffix clash would hide both from skeleton_of)
        keep = ["household_id"] + [c for c in ha.columns if c not in hh.columns]
        hh = hh.merge(ha[keep], on="household_id", how="left")
    merged = persons.merge(hh, on="household_id", how="left", suffixes=("", "_hh"))

    rows: List[dict] = []
    for r in merged.itertuples(index=False):
        row = r._asdict()
        skeleton = skeleton_of(row, row, zone_of_tract=zone_of_tract)
        out = {
            "persona_id": row["persona_id"],
            "person_id": row["person_id"],
            "household_id": row["household_id"],
            "wave": _int_or_none(row.get("survey_year")),
            "fold": psrc.fold_id(row["household_id"]),
            "segment": row.get("segment"),
        }
        out.update(skeleton)
        rows.append(out)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# prompt building + mask-lint gate
# ---------------------------------------------------------------------------

def _forbidden_tokens():
    return load_forbidden_tokens(default_token_path())


def _gate_prompt(persona_id: str, prompt: str, tokens) -> List[str]:
    """Return offending token strings in a prompt, or [] when clean."""
    return [v.token for v in lint_text(prompt, tokens)]


_INT_SKELETON_FIELDS = frozenset({"age", "household_size", "household_cars", "income_class"})
_BOOL_SKELETON_FIELDS = frozenset({"employed", "student", "can_drive", "has_pass"})


def _skeleton_from_index_row(row: Mapping) -> dict:
    """Rebuild a skeleton dict from a persona-index frame row. Integer fields
    are re-coerced: a NaN anywhere in an int64 column re-floats the whole
    column, and a prompt must never show ``4.0`` where the value is 4."""
    out = {}
    for f in SKELETON_FIELDS:
        value = _val(row, f) if f in row else None
        if f in _INT_SKELETON_FIELDS:
            value = _int_or_none(value)
        elif f in _BOOL_SKELETON_FIELDS and value is not None:
            value = bool(value)
        out[f] = value
    return out


def build_prompts(
    dataset,
    out_path,
    data_dir=psrc.DEFAULT_DATA_DIR,
    zone_of_tract: Optional[Callable[[str], str]] = None,
    mode: str = "serve",
) -> dict:
    """Render every persona's seed prompt, write prompts JSONL, and GATE the
    whole file through the mask-lint token scan (fail loud, listing the
    offending personas). Each line is ``{"persona_id", "prompt", "attempt": 1}``.
    """
    pidx = persona_index(dataset, data_dir=data_dir, zone_of_tract=zone_of_tract)
    id_map = dict(zip(pidx["person_id"], pidx["persona_id"]))

    trips = enriched_trips(dataset, data_dir=data_dir)
    trips_by_person = {pid: grp for pid, grp in trips.groupby("person_id")}

    person_days = dataset.person_days.copy()
    person_days["person_id"] = person_days["person_id"].astype(str)
    days_by_person = {pid: grp for pid, grp in person_days.groupby("person_id")}

    person_attrs = {}
    if data_dir is not None and (Path(data_dir) / psrc._PERSONS_CSV).exists():
        pa = enriched_person_attrs(data_dir)
        pa["person_id"] = pa["person_id"].astype(str)
        person_attrs = {r.person_id: r._asdict() for r in pa.itertuples(index=False)}

    tokens = _forbidden_tokens()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    offenders: Dict[str, List[str]] = {}
    n_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for r in pidx.itertuples(index=False):
            row = r._asdict()
            persona_id = row["persona_id"]
            person_id = row["person_id"]
            skeleton = _skeleton_from_index_row(row)
            pdays = days_by_person.get(person_id, dataset.person_days.iloc[0:0])
            ptrips = trips_by_person.get(person_id, trips.iloc[0:0])
            prow = person_attrs.get(person_id, {})
            evidence = evidence_lines_of(pdays, ptrips, prow)
            # prompt surface: an absent work zone reads "none", never "None"
            render_skeleton = {k: ("none" if v is None else v) for k, v in skeleton.items()}
            prompt = render_seed_prompt(render_skeleton, evidence, len(pdays), mode)
            hits = _gate_prompt(persona_id, prompt, tokens)
            if hits:
                offenders[persona_id] = sorted(set(hits))
            fh.write(json.dumps({"persona_id": persona_id, "prompt": prompt, "attempt": 1}, sort_keys=True))
            fh.write("\n")
            n_written += 1

    if offenders:
        listed = ", ".join(f"{pid}({','.join(toks)})" for pid, toks in sorted(offenders.items()))
        raise ValueError(
            f"MASK-LINT: {len(offenders)} persona prompt(s) carry forbidden tokens: {listed}"
        )

    return {
        "n_personas": len(pidx),
        "n_prompts": n_written,
        "out_path": str(out_path),
        "n_offenders": 0,
    }


def build_retry_prompts(failures: Iterable[Mapping], out_path, mode: str = "serve") -> dict:
    """Render retry prompts for cards that failed validation. Each failure
    carries ``persona_id``, ``skeleton``, ``evidence_lines``, ``n_observed_days``,
    ``failure_reasons`` and (optionally) an ``attempt`` to bump. The output is
    lint-gated exactly like :func:`build_prompts`."""
    tokens = _forbidden_tokens()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    offenders: Dict[str, List[str]] = {}
    n_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for f in failures:
            persona_id = _val(f, "persona_id")
            skeleton = _val(f, "skeleton", {})
            evidence = list(_val(f, "evidence_lines", []) or [])
            n_days = _int_or_none(_val(f, "n_observed_days")) or 0
            reasons = list(_val(f, "failure_reasons", []) or [])
            attempt = (_int_or_none(_val(f, "attempt")) or 1) + 1
            # prompt surface: an absent work zone reads "none", never "None"
            render_skeleton = {k: ("none" if v is None else v) for k, v in dict(skeleton).items()}
            prompt = render_seed_retry_prompt(render_skeleton, evidence, n_days, reasons, mode)
            hits = _gate_prompt(persona_id, prompt, tokens)
            if hits:
                offenders[persona_id] = sorted(set(hits))
            fh.write(
                json.dumps(
                    {"persona_id": persona_id, "prompt": prompt, "attempt": attempt},
                    sort_keys=True,
                )
            )
            fh.write("\n")
            n_written += 1

    if offenders:
        listed = ", ".join(f"{pid}({','.join(toks)})" for pid, toks in sorted(offenders.items()))
        raise ValueError(
            f"MASK-LINT: {len(offenders)} retry prompt(s) carry forbidden tokens: {listed}"
        )

    return {"n_prompts": n_written, "out_path": str(out_path), "n_offenders": 0}
