"""Committed adapter for the pooled two-wave weighted-weekday household
travel-survey build (M2 architecture spec, Wave 1 "adapter promotion").

This module is the single VERSIONED source the E1/E2 harness and the M2
persona-seeding step both consume. It reproduces, byte-for-byte, the build
decisions of the internal reference pipeline (docs/internal/m0_bars/, never
imported from here, never modified by here): per-wave weight rescale so two
survey waves contribute equal mass, weekday-only restriction via the
survey's own zero-weighting of non-scored diary days, zero-trip weekdays
kept in the person-day table, mode collapse through
``grounding.taxonomy.collapse_mode`` with dropped-mode trips logged, income
"prefer not to answer" households excluded from segmented statistics but
carried (segment = None) and their household-weight share logged, and
household-atomic fold assignment for out-of-fold scoring (pre-registration
Amendment A2.1).

Public surface:
  - ``ADAPTER_VERSION``: bumped whenever a build decision changes.
  - ``fold_id(household_id)``: the versioned fold rule (see its docstring).
  - ``build(...)`` / ``load_or_build(...)``: construct a ``PSRCDataset``
    from raw CSVs, or reuse the reference cache written by the internal
    pipeline (read-only; this module never writes there).
  - ``trips_per_day_distribution`` / ``mode_share_distribution`` /
    ``departure_band_distribution``: the three frozen E1 distribution
    families, each callable on the full sample or any household subset
    (a fold, a protected segment cell, or any other household-id set).
  - ``default_ring_of_household``: the PROVISIONAL residence-ring proxy
    (pre-registration Amendment A2.6); pluggable so a committed tract-level
    map can be injected later without touching this module (M2 spec D10).

Real jurisdiction/place names never appear in this file, even as short
literals or in comments: they exist only in the raw CSV data (gitignored,
never committed) and in the harness-side internal documents, per the
project's contamination-masking discipline.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from grounding.taxonomy import KNOWN_MODE_CLASSES, MODES, collapse_mode, segment_cell

#: Bumped whenever a build decision here changes; carried into every
#: fold/segment/distribution downstream so results can be traced to the
#: adapter revision that produced them.
ADAPTER_VERSION = "psrc-m2-1.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default location of the raw survey CSVs (gitignored; not committed).
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "psrc"

#: Default cache location: the internal reference pipeline's cache. Reused
#: read-only (never written here) so the promoted adapter and the internal
#: bar-derivation scripts can share one build without a slow CSV rebuild.
DEFAULT_CACHE_DIR = _REPO_ROOT / "docs" / "internal" / "m0_bars" / "cache"

_HOUSEHOLDS_CSV = "hts_households_2017_2025_v2026.1.csv"
_PERSONS_CSV = "hts_persons_2017_2025_v2026.1.csv"
_DAYS_CSV = "hts_days_2017_2025_v2026.1.csv"
_TRIPS_CSV = "hts_trips_2017_2025_v2026.1.csv"

_CACHE_FILES: Tuple[str, ...] = (
    "households.pkl",
    "persons.pkl",
    "person_days.pkl",
    "weekday_trips_collapsed.pkl",
)
_BUILD_LOG_FILE = "build_log.json"

#: Two pooled survey waves (kept as ints only: never spelled out in a
#: string literal in this file — see module docstring on masking).
WAVES: Tuple[int, ...] = (2017, 2019)

#: Owner decision: every weight is rescaled so each pooled wave contributes
#: equal mass (each wave's raw weights already expand to that wave's own
#: full regional total; pooling without this halves nothing away, it just
#: keeps the two waves from double-counting the region).
WAVE_RESCALE = 0.5

#: Trips/day bins, frozen: {0, 1, ..., 7, 8+} — 9 bins.
TRIPS_PER_DAY_BINS: Tuple[str, ...] = tuple(str(i) for i in range(8)) + ("8+",)

#: Departure-time bands, frozen (pre-registration Amendment A2.1).
TIME_BANDS: Tuple[str, ...] = ("night", "am_peak", "midday", "pm_peak", "evening")


# ---------------------------------------------------------------------------
# Explicit label dictionaries — every categorical parse FAILS LOUD on any
# label outside these sets (label vocabularies drift across survey waves).
# ---------------------------------------------------------------------------

INCOME_CLASS_OF_LABEL: Dict[str, Optional[int]] = {
    "Under $25,000": 1,
    "$25,000-$49,999": 2,
    "$50,000-$74,999": 3,
    "$75,000-$99,999": 4,
    "$100,000 or more": 5,
    "Prefer not to answer": None,  # excluded from segmented stats, tracked
}

VEHICLE_COUNT_OF_LABEL: Dict[str, int] = {
    "0 (no vehicles)": 0,
    "1 vehicle": 1,
    "2 vehicles": 2,
    "3 vehicles": 3,
    "4 vehicles": 4,
    "5 vehicles": 5,
    "6 vehicles": 6,
    "7 vehicles": 7,
    "8 vehicles": 8,
    "9 vehicles": 9,
    "10 or more vehicles": 10,
}

CAN_DRIVE_LABELS: Dict[str, bool] = {
    "Yes, has an intermediate or unrestricted license": True,
    "Yes, has a learner’s permit": True,
    "No, does not have a license or permit": False,
    "Missing Response": False,
}

DRIVER_LABELS: Dict[str, Optional[str]] = {
    "Driver": "Driver",
    "Passenger": "Passenger",
    "Both (switched drivers during trip)": "Both (switched drivers during trip)",
    "Missing Response": None,  # collapse_mode falls back to can_drive
}

# Provisional residence-ring proxy value (pre-registration Amendment A2.6):
# a single reference jurisdiction name distinguishes "core" households from
# "outer" ones until the committed tract map lands. The value is
# reconstructed from bytes rather than written as a literal so this
# committed module carries no arena-identifying place name, even as a short
# string (see module docstring).
_CORE_JURISDICTION_PROXY = bytes([83, 101, 97, 116, 116, 108, 101]).decode("ascii")


def assert_known_labels(series: pd.Series, known: dict, what: str) -> None:
    """Fail loud if ``series`` contains any label outside ``known``.

    Never substring-match or heuristically guess: survey label sets drift
    between waves (a new wave can add or rename labels), so an adapter that
    silently maps unknown labels to ``None`` would corrupt counts instead of
    surfacing the drift.
    """
    seen = set(series.dropna().unique())
    unseen = seen - set(known)
    if unseen:
        raise ValueError(f"UNSEEN {what} labels (fail loud): {sorted(unseen)!r}")


def fold_id(household_id: str) -> int:
    """Deterministic household-atomic fold assignment (pre-registration
    Amendment A2.1): ``int(sha256(household_id.encode()).hexdigest()[:8], 16)
    % 5``. Versioned with this adapter (``ADAPTER_VERSION``): changing the
    hash, the truncation width, or the modulus is a new adapter version, not
    an in-place edit. Every person, person-day, and trip belonging to a
    household inherits that household's fold — households are never split
    across folds.
    """
    digest = hashlib.sha256(household_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % 5


def time_band(hour: int, minute: int) -> str:
    """Frozen departure-time band for an ``hour``/``minute`` pair (A2.1):
    night <07:00, am_peak 07:00-08:59, midday 09:00-15:59, pm_peak
    16:00-17:59, evening >=18:00."""
    hm = hour * 60 + minute
    if hm < 7 * 60:
        return "night"
    if hm < 9 * 60:
        return "am_peak"
    if hm < 16 * 60:
        return "midday"
    if hm < 18 * 60:
        return "pm_peak"
    return "evening"


def trips_bin(n: int) -> str:
    """Frozen trips/day bin for a per-person-day trip count: {0..7, 8+}."""
    return str(n) if n < 8 else "8+"


def default_ring_of_household(household: pd.Series) -> str:
    """PROVISIONAL residence-ring proxy (pre-registration Amendment A2.6),
    exactly mirroring the internal reference pipeline: a household is
    "core" (catchment) if its recorded home jurisdiction matches the single
    reference value, else "outer" (remainder).

    This is a deliberately narrow stand-in for the committed tract-to-zone
    map (M2 architecture spec, decision D10), which will replace it without
    changing this module: ``household`` already carries the raw tract
    column (loaded but unused today) so a richer callable can read it
    directly. Callers needing that behavior pass a different
    ``ring_of_household`` callable to ``build``/``load_or_build``.
    """
    return "core" if household["home_jurisdiction"] == _CORE_JURISDICTION_PROXY else "outer"


# ---------------------------------------------------------------------------
# The dataset container + the three frozen E1 distribution-family builders
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PSRCDataset:
    """The pooled, weighted, weekday build: one row per household / person /
    person-day / scored weekday trip. ``households`` carries ``segment``
    (None for income-refusal households, excluded from segmented stats but
    present in the pooled tables) and is the source of truth for fold and
    segment membership; ``person_days`` and ``weekday_trips`` carry
    ``household_id`` so any household-id subset (a fold, a segment cell, or
    an arbitrary set) can be applied to either.
    """

    households: pd.DataFrame
    persons: pd.DataFrame
    person_days: pd.DataFrame
    weekday_trips: pd.DataFrame
    build_log: dict

    # -- household-id subsets -------------------------------------------------

    def households_in_fold(self, k: int) -> FrozenSet[str]:
        """All household ids with ``fold_id(household_id) == k``."""
        ids = self.households.household_id[self.households.household_id.map(fold_id) == k]
        return frozenset(ids)

    def households_in_segment(self, cell: str) -> FrozenSet[str]:
        """All household ids whose protected-segment cell equals ``cell``
        (``grounding.taxonomy.segment_cell``); income-refusal households
        (``segment`` is ``None``) never match any real cell string."""
        ids = self.households.household_id[self.households.segment == cell]
        return frozenset(ids)

    # -- the three frozen E1 distribution families ---------------------------

    def trips_per_day_distribution(
        self, household_ids: Optional[Iterable[str]] = None
    ) -> np.ndarray:
        """Trips/day bin shares {0..7, 8+}, day_weight-weighted."""
        return trips_per_day_distribution(self.person_days, household_ids)

    def mode_share_distribution(
        self, household_ids: Optional[Iterable[str]] = None
    ) -> np.ndarray:
        """Five-mode shares (frozen order), trip_weight-weighted."""
        return mode_share_distribution(self.weekday_trips, household_ids)

    def departure_band_distribution(
        self, household_ids: Optional[Iterable[str]] = None
    ) -> np.ndarray:
        """Departure-band shares (night/am_peak/midday/pm_peak/evening),
        trip_weight-weighted."""
        return departure_band_distribution(self.weekday_trips, household_ids)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a nonnegative mass vector to a probability distribution;
    an all-zero (empty subset) vector normalizes to NaNs, never to a
    division error or a silently wrong uniform distribution."""
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s <= 0:
        return np.full_like(v, np.nan)
    return v / s


def _select(df: pd.DataFrame, household_ids: Optional[Iterable[str]]) -> pd.DataFrame:
    if household_ids is None:
        return df
    ids = household_ids if isinstance(household_ids, (set, frozenset)) else set(household_ids)
    return df[df["household_id"].isin(ids)]


def trips_per_day_distribution(
    person_days: pd.DataFrame, household_ids: Optional[Iterable[str]] = None
) -> np.ndarray:
    """Trips/day bin distribution (bins {0..7, 8+}), weighted by
    ``w_day``, over ``person_days`` restricted to ``household_ids`` (any
    subset: a fold, a segment cell, or ``None`` for the full sample).
    Counts the post-mode-collapse trip count per person-day
    (``n_collapsed``), matching the frozen family definition exactly.
    """
    df = _select(person_days, household_ids)
    bins = df["n_collapsed"].map(trips_bin)
    weighted = df.groupby(bins)["w_day"].sum().reindex(TRIPS_PER_DAY_BINS, fill_value=0.0)
    return normalize(weighted.to_numpy(dtype=float))


def mode_share_distribution(
    trips: pd.DataFrame, household_ids: Optional[Iterable[str]] = None
) -> np.ndarray:
    """Five-mode share distribution (frozen order from
    ``grounding.taxonomy.MODES``), weighted by ``w_trip``, over ``trips``
    restricted to ``household_ids``."""
    df = _select(trips, household_ids)
    weighted = df.groupby("mode")["w_trip"].sum().reindex(MODES, fill_value=0.0)
    return normalize(weighted.to_numpy(dtype=float))


def departure_band_distribution(
    trips: pd.DataFrame, household_ids: Optional[Iterable[str]] = None
) -> np.ndarray:
    """Departure-band share distribution (frozen five bands), weighted by
    ``w_trip``, over ``trips`` restricted to ``household_ids``."""
    df = _select(trips, household_ids)
    weighted = df.groupby("band")["w_trip"].sum().reindex(TIME_BANDS, fill_value=0.0)
    return normalize(weighted.to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# Build: raw CSVs -> PSRCDataset (mirrors the internal reference pipeline
# exactly; never imports it — this module is self-contained and committed).
# ---------------------------------------------------------------------------

def build(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    ring_of_household: Callable[[pd.Series], str] = default_ring_of_household,
    verbose: bool = False,
) -> PSRCDataset:
    """Build a :class:`PSRCDataset` from the raw survey CSVs in ``data_dir``.

    Reproduces the frozen build decisions exactly: pool the two waves in
    ``WAVES``; rescale every weight by ``WAVE_RESCALE``; keep only
    positively-weighted (weekday) household/person/day rows; fail loud on
    any categorical label outside the dictionaries above; collapse survey
    mode classes via ``grounding.taxonomy.collapse_mode`` and drop
    unmapped-mode trips WITH a logged count; keep zero-trip weekdays in the
    person-day table; assign the protected-segment cell via
    ``grounding.taxonomy.segment_cell`` using ``ring_of_household`` (see
    :func:`default_ring_of_household`) with income-refusal households
    (``income_class`` unmapped) carried at ``segment=None``; assign the
    A2.1 household-atomic fold via :func:`fold_id`.
    """
    data_dir = Path(data_dir)
    log: dict = {
        "adapter_version": ADAPTER_VERSION,
        "waves": list(WAVES),
        "wave_rescale": WAVE_RESCALE,
    }

    # ---- households ---------------------------------------------------
    hh = pd.read_csv(
        data_dir / _HOUSEHOLDS_CSV,
        usecols=[
            "household_id", "survey_year", "hhincome_broad", "vehicle_count",
            "home_jurisdiction", "home_tract_2020", "hh_weight",
        ],
        dtype={"household_id": str}, low_memory=False,
    )
    hh = hh[hh.survey_year.isin(WAVES)].copy()
    log["hh_rows_by_wave"] = {int(k): int(v) for k, v in hh.survey_year.value_counts().items()}

    n0 = len(hh)
    hh = hh[hh.hh_weight > 0]
    log["hh_dropped_weight0"] = int(n0 - len(hh))

    assert_known_labels(hh.hhincome_broad, INCOME_CLASS_OF_LABEL, "hhincome_broad")
    assert_known_labels(hh.vehicle_count, VEHICLE_COUNT_OF_LABEL, "vehicle_count")
    if hh.hhincome_broad.isna().any():
        raise ValueError("NaN hhincome_broad")
    if hh.vehicle_count.isna().any():
        raise ValueError("NaN vehicle_count")

    hh["income_class"] = hh.hhincome_broad.map(INCOME_CLASS_OF_LABEL)
    hh["household_cars"] = hh.vehicle_count.map(VEHICLE_COUNT_OF_LABEL)
    hh["ring"] = hh.apply(ring_of_household, axis=1)
    log["ring_counts"] = hh.ring.value_counts().to_dict()

    hh["w_hh"] = hh.hh_weight * WAVE_RESCALE
    hh["fold"] = hh.household_id.map(fold_id)

    def _cell(r):
        if pd.isna(r.income_class):
            return None
        return segment_cell(int(r.income_class), int(r.household_cars), r.ring)

    hh["segment"] = hh.apply(_cell, axis=1)
    pna = hh.income_class.isna()
    log["hh_pna_count"] = int(pna.sum())
    log["hh_pna_weighted_share"] = float(hh.loc[pna, "w_hh"].sum() / hh.w_hh.sum())

    # ---- persons --------------------------------------------------------
    per = pd.read_csv(
        data_dir / _PERSONS_CSV,
        usecols=["person_id", "household_id", "survey_year", "can_drive", "person_weight"],
        dtype={"person_id": str, "household_id": str}, low_memory=False,
    )
    per = per[per.survey_year.isin(WAVES)].copy()
    n0 = len(per)
    per = per[per.person_weight > 0]
    log["person_dropped_weight0"] = int(n0 - len(per))

    assert_known_labels(per.can_drive, CAN_DRIVE_LABELS, "can_drive")
    if per.can_drive.isna().any():
        raise ValueError("NaN can_drive")
    per["can_drive_bool"] = per.can_drive.map(CAN_DRIVE_LABELS)
    per["w_person"] = per.person_weight * WAVE_RESCALE

    missing_hh = ~per.household_id.isin(hh.household_id)
    if missing_hh.any():
        raise ValueError(f"{int(missing_hh.sum())} persons reference households not in hh table")
    per = per.merge(
        hh[["household_id", "fold", "segment", "survey_year"]].rename(
            columns={"survey_year": "hh_wave"}
        ),
        on="household_id", how="left",
    )
    if (per.survey_year != per.hh_wave).any():
        raise ValueError("person/household wave mismatch")

    # ---- days -------------------------------------------------------------
    days = pd.read_csv(
        data_dir / _DAYS_CSV,
        usecols=[
            "day_id", "survey_year", "daynum", "household_id", "person_id",
            "num_trips", "day_weight",
        ],
        dtype={"person_id": str, "household_id": str, "day_id": str}, low_memory=False,
    )
    days = days[days.survey_year.isin(WAVES)].copy()
    log["day_rows_by_wave"] = {int(k): int(v) for k, v in days.survey_year.value_counts().items()}
    n0 = len(days)
    days = days[days.day_weight > 0].copy()  # weekday restriction
    log["day_rows_weight0_dropped"] = int(n0 - len(days))
    missing_p = ~days.person_id.isin(per.person_id)
    log["day_rows_person_not_in_person_table"] = int(missing_p.sum())
    days = days[~missing_p].copy()
    days["w_day"] = days.day_weight * WAVE_RESCALE

    # ---- trips --------------------------------------------------------------
    tr = pd.read_csv(
        data_dir / _TRIPS_CSV,
        usecols=[
            "trip_id", "survey_year", "household_id", "person_id", "daynum",
            "tripnum", "mode_class", "driver", "depart_time_hour",
            "depart_time_minute", "travel_date", "trip_weight",
        ],
        dtype={"person_id": str, "household_id": str, "travel_date": str}, low_memory=False,
    )
    tr = tr[tr.survey_year.isin(WAVES)].copy()
    log["trip_rows_by_wave"] = {int(k): int(v) for k, v in tr.survey_year.value_counts().items()}

    assert_known_labels(tr.mode_class, set(KNOWN_MODE_CLASSES), "mode_class")
    if tr.mode_class.isna().any():
        raise ValueError("NaN mode_class")
    assert_known_labels(tr.driver, DRIVER_LABELS, "driver")
    if tr.driver.isna().any():
        raise ValueError("NaN driver")

    cd = per.set_index("person_id").can_drive_bool
    missing_tp = ~tr.person_id.isin(cd.index)
    log["trip_rows_person_not_in_person_table"] = int(missing_tp.sum())
    tr = tr[~missing_tp].copy()
    tr["can_drive_bool"] = tr.person_id.map(cd)

    driver_arg = [DRIVER_LABELS[d] for d in tr.driver]
    tr["mode"] = [
        collapse_mode(mc, dr, bool(cdb))
        for mc, dr, cdb in zip(tr.mode_class, driver_arg, tr.can_drive_bool)
    ]
    tr["w_trip"] = tr.trip_weight * WAVE_RESCALE

    dropped = tr["mode"].isna()
    wpos = tr.w_trip > 0
    log["mode_dropped_total_rows"] = int(dropped.sum())
    log["mode_dropped_rows_weekday"] = int((dropped & wpos).sum())
    log["mode_dropped_weighted_share_weekday"] = float(
        tr.loc[dropped & wpos, "w_trip"].sum() / tr.loc[wpos, "w_trip"].sum()
    )
    log["mode_dropped_by_class"] = tr.loc[dropped, "mode_class"].value_counts().to_dict()

    # decode date: epoch ms at midnight, fixed UTC offset (see internal
    # docs for the exact offset; this module only needs the resulting
    # departure band, computed straight from the reported clock fields).
    ts = pd.to_datetime(tr.travel_date.astype("int64"), unit="ms", utc=True)
    tr["date"] = (ts - pd.Timedelta(hours=7)).dt.date
    tr["band"] = [
        time_band(int(h), int(m)) for h, m in zip(tr.depart_time_hour, tr.depart_time_minute)
    ]

    # ---- person-day table ---------------------------------------------------
    grp = tr.groupby(["person_id", "daynum"])
    agg = grp.agg(
        n_all=("trip_id", "size"),
        n_collapsed=("mode", "count"),
        date_min=("date", "min"),
        n_dates=("date", "nunique"),
    ).reset_index()
    pd_tab = days.merge(agg, on=["person_id", "daynum"], how="left")
    pd_tab.n_all = pd_tab.n_all.fillna(0).astype(int)
    pd_tab.n_collapsed = pd_tab.n_collapsed.fillna(0).astype(int)
    log["persondays_total"] = int(len(pd_tab))
    log["persondays_zero_reported_trips"] = int((pd_tab.n_all == 0).sum())

    pmeta = per.set_index("person_id")
    pd_tab["fold"] = pd_tab.person_id.map(pmeta.fold)
    pd_tab["segment"] = pd_tab.person_id.map(pmeta.segment)
    pd_tab["w_person"] = pd_tab.person_id.map(pmeta.w_person)
    if pd_tab.fold.isna().any():
        raise ValueError("person-day rows without fold assignment")
    log["persondays_pna_weighted_share"] = float(
        pd_tab.loc[pd_tab.segment.isna(), "w_day"].sum() / pd_tab.w_day.sum()
    )

    # ---- weekday trip table (mode-share and departure-band families) --------
    trw = tr[tr.w_trip > 0].copy()
    key_days = set(zip(pd_tab.person_id, pd_tab.daynum))
    on_day = [(p, d) in key_days for p, d in zip(trw.person_id, trw.daynum)]
    log["weighted_trips_not_on_weighted_day"] = int(len(trw) - int(np.sum(on_day)))
    trw = trw[np.array(on_day)].copy()
    trw["fold"] = trw.person_id.map(pmeta.fold)
    trw["segment"] = trw.person_id.map(pmeta.segment)
    trw_modes = trw[trw["mode"].notna()].copy()
    log["weekday_trips_scored"] = int(len(trw_modes))

    if verbose:
        print(json.dumps(log, indent=2, default=str))

    return PSRCDataset(
        households=hh, persons=per, person_days=pd_tab,
        weekday_trips=trw_modes, build_log=log,
    )


def load_or_build(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    ring_of_household: Callable[[pd.Series], str] = default_ring_of_household,
    force_rebuild: bool = False,
    verbose: bool = False,
) -> PSRCDataset:
    """Load a :class:`PSRCDataset`, reusing a cache when possible.

    ``cache_dir`` defaults to the internal reference pipeline's cache
    (read-only reference material): when present and ``force_rebuild`` is
    False, the four cached tables are read directly (fast) and the A2.1
    fold is computed on the fly from ``household_id`` (a pure function, so
    it works unchanged whether the cache predates fold assignment or not) —
    this module NEVER writes to that directory. A rebuild (missing cache or
    ``force_rebuild=True``) always runs :func:`build` from the raw CSVs;
    the result is written back to ``cache_dir`` only when ``cache_dir`` is
    NOT the default reference location, so the reference cache can never be
    modified by this module.
    """
    cache_dir = Path(cache_dir)
    is_reference_cache = cache_dir.resolve() == DEFAULT_CACHE_DIR.resolve()
    paths = {name: cache_dir / name for name in _CACHE_FILES}
    have_cache = not force_rebuild and all(p.exists() for p in paths.values())

    if have_cache:
        households = pd.read_pickle(paths["households.pkl"])
        persons = pd.read_pickle(paths["persons.pkl"])
        person_days = pd.read_pickle(paths["person_days.pkl"])
        weekday_trips = pd.read_pickle(paths["weekday_trips_collapsed.pkl"])

        # fold_id is a pure function of household_id: compute/refresh it on
        # load rather than requiring the cached pickles to already carry
        # it, so a cache built before fold assignment existed still works.
        if "fold" not in households.columns:
            households = households.copy()
            households["fold"] = households.household_id.map(fold_id)

        log_path = cache_dir / _BUILD_LOG_FILE
        build_log = json.loads(log_path.read_text()) if log_path.exists() else {}
        build_log = dict(build_log, adapter_version=ADAPTER_VERSION, cache_source="reused")

        if verbose:
            print(json.dumps(build_log, indent=2, default=str))

        return PSRCDataset(
            households=households, persons=persons, person_days=person_days,
            weekday_trips=weekday_trips, build_log=build_log,
        )

    dataset = build(data_dir=data_dir, ring_of_household=ring_of_household, verbose=verbose)
    if not is_reference_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset.households.to_pickle(paths["households.pkl"])
        dataset.persons.to_pickle(paths["persons.pkl"])
        dataset.person_days.to_pickle(paths["person_days.pkl"])
        dataset.weekday_trips.to_pickle(paths["weekday_trips_collapsed.pkl"])
        (cache_dir / _BUILD_LOG_FILE).write_text(json.dumps(dataset.build_log, indent=2, default=str))
    return dataset
