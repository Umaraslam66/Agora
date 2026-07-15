"""Tests for the E1 scoring harness (evaluation/e1.py, sealed A2.1 / M2 D5).

Covered: TVD machinery vs hand-computed values; ensemble-mean-BEFORE-TVD order;
guard-merge cell mapping with re-pinned ring injection; paired-bootstrap CI
reproducibility and decision rule; a small-synthetic end-to-end where a
truth-identical method arm beats an intentionally-biased MNL arm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation import e1
from grounding.adapters import psrc


# ---------------------------------------------------------------------------
# tiny synthetic dataset builder (adapter-shaped, no raw CSVs)
# ---------------------------------------------------------------------------

def _dataset(households, person_days, weekday_trips):
    return psrc.PSRCDataset(
        households=pd.DataFrame(households),
        persons=pd.DataFrame(columns=["person_id", "household_id"]),
        person_days=pd.DataFrame(person_days),
        weekday_trips=pd.DataFrame(weekday_trips),
        build_log={},
    )


def _simple_dataset(n_hh=20):
    """One person per household, one weekday, two trips each; households split
    across a couple of mode/band mixes so the family distributions are nonuniform.
    """
    hh, pdays, trips = [], [], []
    for i in range(n_hh):
        hid = f"H{i:03d}"
        pid = f"{hid}p"
        hh.append({"household_id": hid, "income_class": (i % 5) + 1, "household_cars": i % 3})
        pdays.append({"household_id": hid, "person_id": pid, "daynum": 1,
                      "n_collapsed": 2, "w_day": 1.0 + (i % 4)})
        mode = "car" if i % 2 == 0 else "walk"
        band = "am_peak" if i % 2 == 0 else "midday"
        for tn in (1, 2):
            trips.append({"household_id": hid, "person_id": pid, "daynum": 1, "tripnum": tn,
                          "mode": mode, "band": band, "w_trip": 1.0 + (i % 4)})
    return _dataset(hh, pdays, trips)


# ---------------------------------------------------------------------------
# 1. TVD machinery vs hand-computed
# ---------------------------------------------------------------------------

def test_tvd_hand_values():
    assert e1.tvd([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert e1.tvd([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert e1.tvd([0.6, 0.4], [0.4, 0.6]) == pytest.approx(0.2)
    assert e1.tvd([0.7, 0.2, 0.1], [0.1, 0.2, 0.7]) == pytest.approx(0.6)


def test_normalize_and_empty():
    np.testing.assert_allclose(e1.normalize([1, 1, 2]), [0.25, 0.25, 0.5])
    assert np.isnan(e1.normalize([0, 0, 0])).all()


def test_pooled_tvd_is_max_over_families():
    truth = {"trips_per_day": np.array([1.0, 0.0]),
             "mode_shares": np.array([0.5, 0.5]),
             "time_bands": np.array([0.5, 0.5])}
    arm = {"trips_per_day": np.array([0.9, 0.1]),   # tvd 0.1
           "mode_shares": np.array([0.2, 0.8]),      # tvd 0.3
           "time_bands": np.array([0.5, 0.5])}       # tvd 0.0
    assert e1.pooled_tvd(arm, truth) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 2. ensemble-mean BEFORE TVD (A2.1) — normalize per run, then average
# ---------------------------------------------------------------------------

class _RD:
    """A minimal RealizedDay stand-in (day_weight + trips)."""
    def __init__(self, day_weight, trips):
        self.day_weight = day_weight
        self.trips = trips


class _RT:
    def __init__(self, mode, band):
        self.mode = mode
        self.depart_band = band
        self.purpose = "x"


def test_ensemble_mean_before_tvd():
    # run0: one car trip; run1: three walk trips. Averaging NORMALIZED dists
    # gives car=0.5, walk=0.5. Normalizing the summed MASSES would give
    # walk=0.75, car=0.25 — the wrong (post-TVD) order. Distinguish them.
    def producer(namespace):
        if namespace.endswith("run0"):
            trips = [_RT("car", "am_peak")]
            return {"P": [_RD(1.0, trips)]}
        trips = [_RT("walk", "midday") for _ in range(3)]
        return {"P": [_RD(1.0, trips)]}

    arm = e1.ensemble_arm(producer, {"P": None}, n_runs=2, namespace_prefix="t_")
    modes = arm.pooled["mode_shares"]
    walk_i = e1.MODES.index("walk")
    car_i = e1.MODES.index("car")
    assert modes[walk_i] == pytest.approx(0.5)
    assert modes[car_i] == pytest.approx(0.5)


def test_trips_per_day_counts_zero_trip_days():
    # A zero-trip day lands in bin 0, day-weighted.
    def producer(namespace):
        return {"P": [_RD(2.0, []), _RD(1.0, [_RT("walk", "midday"), _RT("walk", "midday")])]}

    arm = e1.ensemble_arm(producer, {"P": None}, n_runs=1, namespace_prefix="t_")
    tpd = arm.pooled["trips_per_day"]
    # weight 2 in bin 0, weight 1 in bin 2 -> normalized [2/3, 0, 1/3, ...]
    assert tpd[0] == pytest.approx(2.0 / 3.0)
    assert tpd[2] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# 3. guard-merge cell mapping with re-pinned ring injection
# ---------------------------------------------------------------------------

def test_repinned_guard_merge_and_drops():
    hh = [
        {"household_id": "A", "income_class": 1, "household_cars": 0},  # low car0 -> ring
        {"household_id": "B", "income_class": 3, "household_cars": 2},  # mid car1p -> ring
        {"household_id": "C", "income_class": 5, "household_cars": 0},  # high car0 remainder -> guard
        {"household_id": "D", "income_class": None, "household_cars": 1},  # PNA -> None
        {"household_id": "E", "income_class": 2, "household_cars": 0},  # low car0 remainder -> guard
        {"household_id": "F", "income_class": 4, "household_cars": 1},  # unmapped ring -> drop
    ]
    ds = _dataset(hh, [], [])
    # injected ring map: A,E->core (catchment); B,C->outer (remainder); F->None
    ring_by_hh = {"A": "core", "B": "outer", "C": "outer", "E": "outer", "F": None}
    home_tracts = {k: k for k in ring_by_hh}  # identity; ring fn keys on this
    cell_of, drops = e1.repinned_cell_of_household(
        ds, ring_of_household=lambda t: ring_by_hh.get(t), home_tracts=home_tracts
    )
    assert cell_of["A"] == "low|car0|catchment"
    assert cell_of["B"] == "mid|car1p|remainder"
    # C (high car0 remainder) and E (low car0 remainder) both collapse to the guard
    assert cell_of["C"] == e1.MERGED_CELL == "car0|remainder"
    assert cell_of["E"] == "car0|remainder"
    assert cell_of["D"] is None  # PNA excluded from cells
    assert cell_of["F"] is None  # unmapped ring dropped
    assert drops == {"pna_income": 1, "unmapped_ring": 1}

    by_cell = e1.households_by_cell(cell_of)
    assert by_cell["car0|remainder"] == frozenset({"C", "E"})
    assert by_cell["low|car0|catchment"] == frozenset({"A"})


def test_repinned_membership_matches_sealed_repin_doc():
    """Integration: on the real adapter cache, the re-pinned guard-merge
    reproduces the committed re-pin table (M2_RING_REPIN.md §4)."""
    try:
        ds = psrc.load_or_build()
    except Exception:
        pytest.skip("adapter cache/raw CSVs not available")
    if not (psrc.DEFAULT_DATA_DIR / psrc._HOUSEHOLDS_CSV).exists():
        pytest.skip("raw households CSV not available")
    cell_of, drops = e1.repinned_cell_of_household(ds)
    by_cell = e1.households_by_cell(cell_of)
    # committed re-pin (map column) counts
    expected = {
        "low|car0|catchment": 481, "low|car1p|catchment": 608,
        "low|car1p|remainder": 418, "mid|car0|catchment": 244,
        "mid|car1p|catchment": 852, "mid|car1p|remainder": 671,
        "high|car0|catchment": 221, "high|car1p|catchment": 1376,
        "high|car1p|remainder": 925, "car0|remainder": 100,
    }
    got = {c: len(by_cell[c]) for c in e1.CELLS10}
    assert got == expected
    assert drops["pna_income"] == 421
    assert drops["unmapped_ring"] == 2


def test_truth_distributions_bit_match_sealed_record():
    try:
        ds = psrc.load_or_build()
    except Exception:
        pytest.skip("adapter cache not available")
    import json
    from pathlib import Path
    seal_path = Path("docs/internal/m0_bars/e1_folds_results.json")
    if not seal_path.exists():
        pytest.skip("sealed record not available")
    seal = json.loads(seal_path.read_text())["full_sample_distributions"]
    truth = e1.truth_distributions(ds)
    np.testing.assert_allclose(truth.pooled["trips_per_day"], seal["trips_per_day"], atol=1e-9)
    np.testing.assert_allclose(truth.pooled["mode_shares"], seal["mode_shares"], atol=1e-9)
    np.testing.assert_allclose(truth.pooled["time_bands"], seal["time_bands"], atol=1e-9)


# ---------------------------------------------------------------------------
# 4. paired bootstrap: reproducibility + decision rule + end-to-end winner
# ---------------------------------------------------------------------------

def test_paired_bootstrap_reproducible_under_seed():
    ds = _simple_dataset()
    matrices = e1.household_family_matrices(ds)
    truth = e1.truth_distributions(ds).pooled
    biased = {k: v.copy() for k, v in truth.items()}
    biased["mode_shares"] = np.roll(biased["mode_shares"], 1)  # shift mass
    r1 = e1.paired_bootstrap(truth, biased, matrices=matrices, B=200, base_seed=123)
    r2 = e1.paired_bootstrap(truth, biased, matrices=matrices, B=200, base_seed=123)
    assert r1["ci_lo"] == r2["ci_lo"] and r1["ci_hi"] == r2["ci_hi"]
    assert r1["seed"] == 123 + e1.SEED_OFFSET_PAIRED


def test_paired_bootstrap_truth_identical_method_beats_biased_mnl():
    ds = _simple_dataset()
    matrices = e1.household_family_matrices(ds)
    truth = e1.truth_distributions(ds).pooled
    # method arm == truth (distribution-identical); MNL arm strongly biased.
    method = {k: v.copy() for k, v in truth.items()}
    mnl = {k: v.copy() for k, v in truth.items()}
    mnl["mode_shares"] = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # all "bike"
    res = e1.paired_bootstrap(method, mnl, matrices=matrices, B=300, base_seed=7)
    assert res["ci_hi"] < e1.EPSILON  # Delta = method - MNL entirely below +eps
    assert res["pass"] is True
    # and the reverse loses: biased method vs truth MNL
    rev = e1.paired_bootstrap(mnl, method, matrices=matrices, B=300, base_seed=7)
    assert rev["pass"] is False
    assert rev["ci_lo"] > e1.EPSILON


def test_household_matrices_reconstruct_pooled_truth():
    ds = _simple_dataset()
    households, hidx, Ma, Mb, Mc = e1.household_family_matrices(ds)
    truth = e1.truth_distributions(ds).pooled
    np.testing.assert_allclose(e1.normalize(Ma.sum(0)), truth["trips_per_day"], atol=1e-12)
    np.testing.assert_allclose(e1.normalize(Mb.sum(0)), truth["mode_shares"], atol=1e-12)
    np.testing.assert_allclose(e1.normalize(Mc.sum(0)), truth["time_bands"], atol=1e-12)
