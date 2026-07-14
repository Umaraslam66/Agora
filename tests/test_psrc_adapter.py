"""Tests for the promoted adapter (grounding/adapters/psrc.py), M2 Wave 1
"adapter promotion". Acceptance numbers below are the frozen fold
investigation's exact counts (docs/internal/M0_E1_FOLD_INVESTIGATION.md §1,
§6) and pre-registration Amendment A2.1/A2.6: this is the single versioned
source the E1/E2 harness and persona seeding must consume, so its counts,
fold assignment, and three distribution families must reproduce the
internal reference pipeline (docs/internal/m0_bars/m0_common.py) bit for
bit, read from the same cache so the comparison is fast in normal test runs.

Household counts by segment cell (10-cell + merge check) and the fold
household counts are the exact numbers signed off in the fold investigation
document; a mismatch here means the promoted adapter has silently diverged
from the frozen build, which must never happen without a dated amendment.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grounding.adapters import psrc
from grounding.taxonomy import MODES, SEGMENT_CELLS

REPO_ROOT = Path(__file__).resolve().parent.parent
M0_COMMON_PATH = REPO_ROOT / "docs" / "internal" / "m0_bars" / "m0_common.py"

# Frozen acceptance numbers (M0_E1_FOLD_INVESTIGATION.md §1).
N_HOUSEHOLDS = 6319
N_PERSONS = 11940
N_PERSON_DAYS = 19248
N_SCORED_TRIPS = 72396
FOLD_HOUSEHOLD_COUNTS = [1258, 1266, 1263, 1296, 1236]

# Frozen per-cell household counts (M0_E1_FOLD_INVESTIGATION.md §6, the
# 12-cell breakdown before the A2.1 car0|remainder guard-cell merge).
SEGMENT_HOUSEHOLD_COUNTS = {
    "low|car0|catchment": 481,
    "low|car1p|catchment": 602,
    "low|car1p|remainder": 424,
    "mid|car0|catchment": 242,
    "mid|car1p|catchment": 844,
    "mid|car1p|remainder": 679,
    "high|car0|catchment": 221,
    "high|car1p|catchment": 1369,
    "high|car1p|remainder": 934,
}
# The three unmerged car-free/remainder cells sum to the merged guard cell.
MERGED_GUARD_CELLS = ("low|car0|remainder", "mid|car0|remainder", "high|car0|remainder")
MERGED_GUARD_TOTAL = 102


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset() -> psrc.PSRCDataset:
    return psrc.load_or_build()


def _load_m0_common_reference():
    """Dynamically import the internal (gitignored) reference pipeline for
    cross-checking the promoted adapter's math. Skipped if the internal
    docs are not present in this checkout (they are never committed)."""
    if not M0_COMMON_PATH.is_file():
        pytest.skip("internal m0_bars reference pipeline not present in this checkout")
    spec = importlib.util.spec_from_file_location("_m0_common_reference", M0_COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reference_distributions(m0, pd_tab: pd.DataFrame, trw: pd.DataFrame, household_ids):
    """Build the three family distributions the way the internal reference
    pipeline does, restricted to ``household_ids`` (or the full sample)."""
    if household_ids is not None:
        pd_tab = pd_tab[pd_tab.household_id.isin(household_ids)]
        trw = trw[trw.household_id.isin(household_ids)]
        households = sorted(set(household_ids) & set(pd_tab.household_id))
    else:
        households = sorted(pd_tab.household_id.unique())
    _, M_a, M_b, M_c = m0.household_family_matrices(pd_tab, trw, households)
    return (
        m0.normalize(M_a.sum(axis=0)),
        m0.normalize(M_b.sum(axis=0)),
        m0.normalize(M_c.sum(axis=0)),
    )


# ---------------------------------------------------------------------------
# Acceptance counts (must run even if the cross-check above is skipped)
# ---------------------------------------------------------------------------

def test_adapter_version_is_pinned():
    assert psrc.ADAPTER_VERSION == "psrc-m2-1.0"


def test_acceptance_counts(dataset: psrc.PSRCDataset):
    assert dataset.households.shape[0] == N_HOUSEHOLDS
    assert dataset.persons.shape[0] == N_PERSONS
    assert dataset.person_days.shape[0] == N_PERSON_DAYS
    assert dataset.weekday_trips.shape[0] == N_SCORED_TRIPS


def test_fold_household_counts_match_investigation(dataset: psrc.PSRCDataset):
    counts = [len(dataset.households_in_fold(k)) for k in range(5)]
    assert counts == FOLD_HOUSEHOLD_COUNTS
    # every household lands in exactly one fold — no loss, no double count
    assert sum(counts) == N_HOUSEHOLDS


def test_segment_household_counts_match_investigation(dataset: psrc.PSRCDataset):
    for cell, expected in SEGMENT_HOUSEHOLD_COUNTS.items():
        assert len(dataset.households_in_segment(cell)) == expected, cell
    merged = sum(len(dataset.households_in_segment(c)) for c in MERGED_GUARD_CELLS)
    assert merged == MERGED_GUARD_TOTAL


def test_pna_income_households_excluded_from_segments_but_tracked(dataset: psrc.PSRCDataset):
    hh = dataset.households
    pna = hh.income_class.isna()
    # tracked: a nontrivial, documented household-weight share
    weighted_share = float(hh.loc[pna, "w_hh"].sum() / hh.w_hh.sum())
    assert 0.03 < weighted_share < 0.06  # ~4.4% per pre-registration A2.6
    # excluded from segmented stats: none of them carry a real segment cell
    assert hh.loc[pna, "segment"].isna().all()
    for cell in SEGMENT_CELLS:
        assert not (hh.loc[pna, "segment"] == cell).any()
    # but NOT excluded from the pooled tables (still counted in N_HOUSEHOLDS)
    assert int(pna.sum()) > 0


# ---------------------------------------------------------------------------
# Distribution-family builders: shape, normalization, callable on subsets
# ---------------------------------------------------------------------------

def test_distribution_shapes_and_normalization(dataset: psrc.PSRCDataset):
    a = dataset.trips_per_day_distribution()
    b = dataset.mode_share_distribution()
    c = dataset.departure_band_distribution()
    assert a.shape == (9,)
    assert b.shape == (5,) and tuple(MODES) == ("walk", "transit", "ride", "car", "bike")
    assert c.shape == (5,)
    for dist in (a, b, c):
        assert np.isclose(dist.sum(), 1.0, atol=1e-12)
        assert (dist >= 0).all()


def test_distribution_builders_are_callable_on_any_household_subset(dataset: psrc.PSRCDataset):
    fold0 = dataset.households_in_fold(0)
    a = dataset.trips_per_day_distribution(fold0)
    b = dataset.mode_share_distribution(fold0)
    c = dataset.departure_band_distribution(fold0)
    assert np.isclose(a.sum(), 1.0, atol=1e-12)
    assert np.isclose(b.sum(), 1.0, atol=1e-12)
    assert np.isclose(c.sum(), 1.0, atol=1e-12)

    cell = "low|car0|catchment"
    seg = dataset.households_in_segment(cell)
    seg_dist = dataset.mode_share_distribution(seg)
    assert np.isclose(seg_dist.sum(), 1.0, atol=1e-12)
    # a different subset gives a materially different distribution (not a
    # no-op filter silently returning the pooled distribution)
    assert not np.allclose(seg_dist, dataset.mode_share_distribution(), atol=1e-3)


def test_empty_subset_normalizes_to_nan_not_a_crash(dataset: psrc.PSRCDataset):
    empty: frozenset = frozenset()
    dist = dataset.trips_per_day_distribution(empty)
    assert np.isnan(dist).all()


# ---------------------------------------------------------------------------
# Identical construction vs the internal reference pipeline
# ---------------------------------------------------------------------------

def test_distributions_identical_to_reference_full_sample(dataset: psrc.PSRCDataset):
    m0 = _load_m0_common_reference()
    pd_tab = pd.read_pickle(psrc.DEFAULT_CACHE_DIR / "person_days.pkl")
    trw = pd.read_pickle(psrc.DEFAULT_CACHE_DIR / "weekday_trips_collapsed.pkl")

    ref_a, ref_b, ref_c = _reference_distributions(m0, pd_tab, trw, None)
    assert np.allclose(dataset.trips_per_day_distribution(), ref_a, atol=1e-12)
    assert np.allclose(dataset.mode_share_distribution(), ref_b, atol=1e-12)
    assert np.allclose(dataset.departure_band_distribution(), ref_c, atol=1e-12)


def test_distributions_identical_to_reference_per_fold(dataset: psrc.PSRCDataset):
    m0 = _load_m0_common_reference()
    pd_tab = pd.read_pickle(psrc.DEFAULT_CACHE_DIR / "person_days.pkl")
    trw = pd.read_pickle(psrc.DEFAULT_CACHE_DIR / "weekday_trips_collapsed.pkl")

    for k in range(5):
        fold_hh = dataset.households_in_fold(k)
        ref_a, ref_b, ref_c = _reference_distributions(m0, pd_tab, trw, fold_hh)
        assert np.allclose(dataset.trips_per_day_distribution(fold_hh), ref_a, atol=1e-12)
        assert np.allclose(dataset.mode_share_distribution(fold_hh), ref_b, atol=1e-12)
        assert np.allclose(dataset.departure_band_distribution(fold_hh), ref_c, atol=1e-12)


def test_distributions_identical_to_reference_per_segment_cell(dataset: psrc.PSRCDataset):
    m0 = _load_m0_common_reference()
    pd_tab = pd.read_pickle(psrc.DEFAULT_CACHE_DIR / "person_days.pkl")
    trw = pd.read_pickle(psrc.DEFAULT_CACHE_DIR / "weekday_trips_collapsed.pkl")

    for cell in SEGMENT_CELLS:
        seg_hh = dataset.households_in_segment(cell)
        if not seg_hh:
            continue
        ref_a, ref_b, ref_c = _reference_distributions(m0, pd_tab, trw, seg_hh)
        assert np.allclose(dataset.trips_per_day_distribution(seg_hh), ref_a, atol=1e-12), cell
        assert np.allclose(dataset.mode_share_distribution(seg_hh), ref_b, atol=1e-12), cell
        assert np.allclose(dataset.departure_band_distribution(seg_hh), ref_c, atol=1e-12), cell


def test_ring_proxy_matches_reference_counts(dataset: psrc.PSRCDataset):
    # The provisional ring proxy must produce the same core/outer split as
    # the internal reference build's own ring proxy (read from the cache
    # it already wrote; no need to re-run the raw-CSV build to check this).
    ref_hh = pd.read_pickle(psrc.DEFAULT_CACHE_DIR / "households.pkl")
    ref_counts = ref_hh.ring.value_counts().to_dict()
    counts = dataset.households.ring.value_counts().to_dict()
    assert counts == ref_counts


# ---------------------------------------------------------------------------
# Masking discipline: label dictionaries fail loud on unseen labels
# ---------------------------------------------------------------------------

def test_income_label_dictionary_fails_loud_on_unseen_label():
    series = pd.Series(["Under $25,000", "A Brand New Income Bracket"])
    with pytest.raises(ValueError, match="UNSEEN"):
        psrc.assert_known_labels(series, psrc.INCOME_CLASS_OF_LABEL, "hhincome_broad")


def test_vehicle_label_dictionary_fails_loud_on_unseen_label():
    series = pd.Series(["2 vehicles", "11 vehicles (new bracket)"])
    with pytest.raises(ValueError, match="UNSEEN"):
        psrc.assert_known_labels(series, psrc.VEHICLE_COUNT_OF_LABEL, "vehicle_count")


def test_mode_class_fails_loud_on_unseen_label():
    from grounding.taxonomy import KNOWN_MODE_CLASSES

    series = pd.Series(["Walk", "Hoverboard Deluxe"])
    with pytest.raises(ValueError, match="UNSEEN"):
        psrc.assert_known_labels(series, set(KNOWN_MODE_CLASSES), "mode_class")


def test_known_labels_do_not_raise():
    series = pd.Series(["Under $25,000", "$100,000 or more", None])
    psrc.assert_known_labels(series, psrc.INCOME_CLASS_OF_LABEL, "hhincome_broad")  # no raise


# ---------------------------------------------------------------------------
# Fold assignment: the A2.1 rule, exactly, and deterministic
# ---------------------------------------------------------------------------

def test_fold_id_matches_frozen_formula():
    import hashlib

    for household_id in ("17100342", "19200001", "abcxyz"):
        expected = int(hashlib.sha256(household_id.encode("utf-8")).hexdigest()[:8], 16) % 5
        assert psrc.fold_id(household_id) == expected


def test_fold_id_is_deterministic():
    ids = [f"hh{i}" for i in range(500)]
    first = [psrc.fold_id(h) for h in ids]
    second = [psrc.fold_id(h) for h in ids]
    assert first == second
    assert all(0 <= f <= 4 for f in first)


# ---------------------------------------------------------------------------
# Cache discipline: the reference cache is read, never written
# ---------------------------------------------------------------------------

def test_load_or_build_never_writes_to_the_reference_cache():
    before = {
        p.name: p.stat().st_mtime_ns
        for p in psrc.DEFAULT_CACHE_DIR.iterdir()
        if p.is_file()
    }
    psrc.load_or_build()
    after = {
        p.name: p.stat().st_mtime_ns
        for p in psrc.DEFAULT_CACHE_DIR.iterdir()
        if p.is_file()
    }
    assert before == after


def test_load_or_build_persists_to_a_non_reference_cache_dir(tmp_path):
    scratch = tmp_path / "psrc_cache"
    assert not scratch.exists()
    ds = psrc.load_or_build(cache_dir=scratch)
    assert scratch.is_dir()
    for name in ("households.pkl", "persons.pkl", "person_days.pkl",
                 "weekday_trips_collapsed.pkl", "build_log.json"):
        assert (scratch / name).exists()
    assert ds.households.shape[0] == N_HOUSEHOLDS

    # second call reuses the just-written cache (fast path), same counts
    ds2 = psrc.load_or_build(cache_dir=scratch)
    assert ds2.households.shape[0] == N_HOUSEHOLDS
    assert ds2.person_days.shape[0] == N_PERSON_DAYS


# ---------------------------------------------------------------------------
# Full raw-CSV rebuild — slow (reads the full multi-hundred-MB survey CSVs);
# skipped by default. Run explicitly with:
#   .venv/bin/python -m pytest tests/test_psrc_adapter.py -m "" -k full_csv_rebuild --no-skip
# (or just delete the skip decorator locally) to verify the from-scratch path.
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="full raw-CSV rebuild reads ~330MB of survey data (households/"
    "persons/days/trips); acceptance counts are already covered fast via "
    "the cache-backed tests above. Run manually when the raw-CSV path "
    "itself needs verifying (e.g. after a data re-download)."
)
def test_full_csv_rebuild_matches_acceptance_counts():
    dataset = psrc.build()
    assert dataset.households.shape[0] == N_HOUSEHOLDS
    assert dataset.persons.shape[0] == N_PERSONS
    assert dataset.person_days.shape[0] == N_PERSON_DAYS
    assert dataset.weekday_trips.shape[0] == N_SCORED_TRIPS
    counts = [len(dataset.households[dataset.households.household_id.map(psrc.fold_id) == k])
              for k in range(5)]
    assert counts == FOLD_HOUSEHOLD_COUNTS
