"""evaluation/e2.py + evaluation/run_e2.py — sealed A2.2 machinery.

Hand-checked spread ratios, the variance-of-sums identity vs a brute-force
O(N^2) pairwise computation, exclusion logging, end-to-end determinism, and
the CLI runner on a synthetic cache. All fixtures are MASKED synthetic data.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agents.card_executor import RealizedDay, RealizedTrip
from evaluation import e2, run_e2


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------

def make_dataset() -> SimpleNamespace:
    """Four persons, three households, weighted weekday diaries."""
    persons = pd.DataFrame(
        [
            {"person_id": "1101", "household_id": "H1"},
            {"person_id": "1102", "household_id": "H1"},
            {"person_id": "1201", "household_id": "H2"},
            {"person_id": "1301", "household_id": "H3"},
        ]
    )
    person_days = pd.DataFrame(
        [
            # person 1101: two weighted days, 2 then 0 trips
            {"person_id": "1101", "daynum": 1, "w_day": 1.0, "n_collapsed": 2, "w_person": 2.0},
            {"person_id": "1101", "daynum": 2, "w_day": 3.0, "n_collapsed": 0, "w_person": 2.0},
            # person 1102: one day, 2 trips
            {"person_id": "1102", "daynum": 1, "w_day": 2.0, "n_collapsed": 2, "w_person": 1.0},
            # person 1201: one day, 1 trip
            {"person_id": "1201", "daynum": 1, "w_day": 1.0, "n_collapsed": 1, "w_person": 1.0},
            # person 1301: one day, zero trips (share dims undefined)
            {"person_id": "1301", "daynum": 1, "w_day": 1.0, "n_collapsed": 0, "w_person": 1.0},
        ]
    )
    weekday_trips = pd.DataFrame(
        [
            {"person_id": "1101", "daynum": 1, "tripnum": 1, "mode": "car",
             "band": "am_peak", "w_trip": 2.0},
            {"person_id": "1101", "daynum": 1, "tripnum": 2, "mode": "walk",
             "band": "pm_peak", "w_trip": 2.0},
            {"person_id": "1102", "daynum": 1, "tripnum": 1, "mode": "ride",
             "band": "midday", "w_trip": 1.0},
            {"person_id": "1102", "daynum": 1, "tripnum": 2, "mode": "transit",
             "band": "evening", "w_trip": 3.0},
            {"person_id": "1201", "daynum": 1, "tripnum": 1, "mode": "car",
             "band": "am_peak", "w_trip": 1.0},
        ]
    )
    return SimpleNamespace(
        persons=persons, person_days=person_days, weekday_trips=weekday_trips
    )


# persona ids under the seeding reindex of the four synthetic person ids
# (sorted person_id -> P00001..): 1101, 1102, 1201, 1301
P1101, P1102, P1201, P1301 = "P00001", "P00002", "P00003", "P00004"


def make_card(persona_id: str, patterns, cars: int = 1, can_drive: bool = True) -> dict:
    return {
        "persona_id": persona_id,
        "skeleton": {"household_cars": cars, "can_drive": can_drive},
        "patterns": patterns,
        "rules": [],
    }


def make_cards() -> list:
    """Cards with genuinely stochastic repertoires (weights split across
    patterns) so realized choices vary across CRN namespaces."""
    commute = {"id": "commute", "weight": 6, "trips": [
        {"purpose": "work", "mode": "car", "depart_band": "am_peak"},
        {"purpose": "home", "mode": "car", "depart_band": "pm_peak"},
    ]}
    errand = {"id": "errand", "weight": 3, "trips": [
        {"purpose": "shop_daily", "mode": "walk", "depart_band": "midday"},
    ]}
    transit_day = {"id": "transit_day", "weight": 5, "trips": [
        {"purpose": "work", "mode": "transit", "depart_band": "am_peak"},
        {"purpose": "home", "mode": "transit", "depart_band": "evening"},
    ]}
    ride_day = {"id": "ride_day", "weight": 4, "trips": [
        {"purpose": "leisure", "mode": "ride", "depart_band": "evening"},
    ]}
    quiet = {"id": "quiet", "weight": 2, "trips": []}
    return [
        make_card(P1101, [dict(commute), dict(errand), dict(quiet)]),
        make_card(P1102, [dict(transit_day), dict(ride_day)]),
        make_card(P1201, [dict(commute), dict(quiet)]),
        make_card(P1301, [dict(errand), dict(quiet)]),
    ]


# ---------------------------------------------------------------------------
# weighted variance + per-person statistics (hand checks)
# ---------------------------------------------------------------------------

def test_weighted_var_hand_check():
    assert e2.weighted_var([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == pytest.approx(2.0 / 3.0)
    # zeroing the middle weight: mean 2, var ((1-2)^2 + (3-2)^2)/2 = 1
    assert e2.weighted_var([1.0, 2.0, 3.0], [1.0, 0.0, 1.0]) == pytest.approx(1.0)
    assert np.isnan(e2.weighted_var([1.0], [0.0]))


def test_person_stats_from_diary_hand_check():
    ds = make_dataset()
    stats = e2.person_stats_from_diary(ds.person_days, ds.weekday_trips)
    # 1101: (1*2 + 3*0) / (1+3) = 0.5 trips/day; car w 2 of 4 trip-weight
    assert stats.loc["1101", "mean_trips_per_day"] == pytest.approx(0.5)
    assert stats.loc["1101", "car_share"] == pytest.approx(0.5)
    assert stats.loc["1101", "ride_share"] == pytest.approx(0.0)
    assert stats.loc["1101", "w_person"] == pytest.approx(2.0)
    # 1102: ride 1 of 4 trip-weight
    assert stats.loc["1102", "ride_share"] == pytest.approx(0.25)
    assert stats.loc["1102", "car_share"] == pytest.approx(0.0)
    # 1201: single car trip
    assert stats.loc["1201", "car_share"] == pytest.approx(1.0)
    # 1301: no trips -> NaN shares but a defined trips/day of 0
    assert stats.loc["1301", "mean_trips_per_day"] == pytest.approx(0.0)
    assert np.isnan(stats.loc["1301", "car_share"])


def test_person_stats_from_realized_hand_check():
    realized = {
        "PX": [
            RealizedDay(1, 1.0, [RealizedTrip("work", "car", "am_peak"),
                                 RealizedTrip("home", "walk", "pm_peak")]),
            RealizedDay(2, 3.0, []),
        ],
        "PY": [RealizedDay(1, 2.0, [RealizedTrip("leisure", "ride", "evening")])],
        "PZ": [RealizedDay(1, 1.0, [])],  # no trips -> NaN shares
    }
    stats = e2.person_stats_from_realized(realized)
    assert stats.loc["PX", "mean_trips_per_day"] == pytest.approx(0.5)
    # car trip carries its day weight 1.0; total trip mass 2.0
    assert stats.loc["PX", "car_share"] == pytest.approx(0.5)
    assert stats.loc["PY", "ride_share"] == pytest.approx(1.0)
    assert np.isnan(stats.loc["PZ", "car_share"])
    assert stats.loc["PZ", "mean_trips_per_day"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# E2(i) spread ratios (hand checks)
# ---------------------------------------------------------------------------

def _stats_frame(values, ids=None, w=1.0):
    ids = ids or [f"P{i}" for i in range(len(values))]
    return pd.DataFrame(
        {
            "mean_trips_per_day": values,
            "car_share": values,
            "ride_share": values,
            "w_person": [w] * len(values),
        },
        index=ids,
    )


def test_spread_ratio_identity_when_sim_equals_real():
    real = _stats_frame([1.0, 2.0, 3.0])
    sim = real[list(e2.DIMENSIONS)]
    result = e2.spread_ratios(real, [sim, sim])
    for dim in e2.DIMENSIONS:
        assert result["dimensions"][dim]["ratio"] == pytest.approx(1.0)
        assert result["dimensions"][dim]["pass"] is True
    assert result["pass"] is True


def test_spread_ratio_hand_check_scaling():
    # real [1,2,3] equal weights: var 2/3; sim [0,2,4]: var 8/3 -> ratio 4.0
    real = _stats_frame([1.0, 2.0, 3.0])
    sim = _stats_frame([0.0, 2.0, 4.0])[list(e2.DIMENSIONS)]
    result = e2.spread_ratios(real, [sim])
    for dim in e2.DIMENSIONS:
        d = result["dimensions"][dim]
        assert d["real_var"] == pytest.approx(2.0 / 3.0)
        assert d["ratio"] == pytest.approx(4.0)
        assert d["pass"] is False
    assert result["pass"] is False


def test_spread_ratio_mean_over_runs():
    real = _stats_frame([1.0, 2.0, 3.0])
    sim_lo = _stats_frame([1.5, 2.0, 2.5])[list(e2.DIMENSIONS)]  # var 1/6 -> 0.25
    sim_hi = _stats_frame([0.0, 2.0, 4.0])[list(e2.DIMENSIONS)]  # var 8/3 -> 4.0
    result = e2.spread_ratios(real, [sim_lo, sim_hi])
    d = result["dimensions"]["mean_trips_per_day"]
    assert d["per_run_ratios"] == pytest.approx([0.25, 4.0])
    assert d["ratio"] == pytest.approx((0.25 + 4.0) / 2.0)


# ---------------------------------------------------------------------------
# E2(ii) variance-of-sums identity
# ---------------------------------------------------------------------------

def _brute_force_mean_pairwise(e: np.ndarray) -> float:
    corr = np.corrcoef(e)
    n = corr.shape[0]
    off = corr[~np.eye(n, dtype=bool)]
    return float(off.mean())


def test_identity_equals_brute_force_on_synthetic():
    rng = np.random.default_rng(7)
    common = rng.normal(size=12)  # a run-level common shock
    e = 0.6 * common[None, :] + rng.normal(size=(7, 12))
    result = e2.mean_pairwise_correlation(e)
    assert result["rho"] == pytest.approx(_brute_force_mean_pairwise(e), abs=1e-12)
    assert result["n_personas"] == 7
    assert result["n_zero_variance_dropped"] == 0
    assert result["n_incomplete_dropped"] == 0


def test_identity_equals_brute_force_without_common_shock():
    rng = np.random.default_rng(11)
    e = rng.normal(size=(9, 15))
    result = e2.mean_pairwise_correlation(e)
    assert result["rho"] == pytest.approx(_brute_force_mean_pairwise(e), abs=1e-12)


def test_perfect_common_shock_gives_rho_one():
    shock = np.array([0.3, -1.2, 0.8, 0.1, -0.5])
    e = np.tile(shock, (6, 1))  # every persona the identical error series
    result = e2.mean_pairwise_correlation(e)
    assert result["rho"] == pytest.approx(1.0)


def test_zero_variance_and_incomplete_rows_dropped_with_counts():
    rng = np.random.default_rng(3)
    e = rng.normal(size=(5, 8))
    e[1, :] = 4.2          # deterministic persona: zero variance across runs
    e[3, 2] = np.nan       # incomplete series
    result = e2.mean_pairwise_correlation(e)
    assert result["n_zero_variance_dropped"] == 1
    assert result["n_incomplete_dropped"] == 1
    assert result["n_personas"] == 3
    kept = np.array([e[0], e[2], e[4]])
    assert result["rho"] == pytest.approx(_brute_force_mean_pairwise(kept), abs=1e-12)


def test_error_correlation_invariant_to_diary_offset():
    """e_ik = s_ik - d_i with d_i constant in k: rho must not depend on d."""
    rng = np.random.default_rng(23)
    sim = rng.normal(size=(6, 10))
    ids = [f"P{i}" for i in range(6)]
    sim_frames = [
        pd.DataFrame({dim: sim[:, k] for dim in e2.DIMENSIONS}, index=ids)
        for k in range(10)
    ]
    diary_a = _stats_frame(list(rng.normal(size=6)), ids=ids)
    diary_b = diary_a.copy()
    for dim in e2.DIMENSIONS:
        diary_b[dim] = diary_b[dim] + 5.0
    ra = e2.error_correlations(diary_a, sim_frames)
    rb = e2.error_correlations(diary_b, sim_frames)
    for dim in e2.DIMENSIONS:
        assert ra["dimensions"][dim]["rho"] == pytest.approx(
            rb["dimensions"][dim]["rho"], abs=1e-12
        )


# ---------------------------------------------------------------------------
# end-to-end scoring
# ---------------------------------------------------------------------------

def test_score_e2_end_to_end_deterministic_and_structured():
    ds = make_dataset()
    cards = make_cards()
    a = e2.score_e2(cards, ds, n_runs=6, seed=0)
    b = e2.score_e2(make_cards(), ds, n_runs=6, seed=0)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(
        b, sort_keys=True, default=str
    )
    assert a["namespaces"] == [f"run{k}" for k in range(6)]
    assert a["n_seeding_persons_in_force"] == 4
    assert a["n_cards_without_diary_match"] == 0
    for dim in e2.DIMENSIONS:
        assert dim in a["spread_ratios"]["dimensions"]
        assert dim in a["error_correlation"]["dimensions"]
    assert isinstance(a["e2_pass"], bool)
    assert a["error_correlation"]["bar"] == 0.20
    assert a["spread_ratios"]["band"] == [0.8, 1.2]

    c = e2.score_e2(make_cards(), ds, n_runs=6, seed=100)
    assert c["namespaces"] == [f"run{100 + k}" for k in range(6)]
    assert c["namespaces"] != a["namespaces"]


def test_score_e2_max_rho_is_max_over_dimensions():
    ds = make_dataset()
    result = e2.score_e2(make_cards(), ds, n_runs=8, seed=1)
    rhos = [
        v["rho"]
        for v in result["error_correlation"]["dimensions"].values()
        if v["rho"] == v["rho"]
    ]
    assert result["error_correlation"]["max_rho"] == pytest.approx(max(rhos))


def test_day_slots_inherit_diary_day_weights():
    ds = make_dataset()
    id_map = {"1101": P1101, "1102": P1102, "1201": P1201, "1301": P1301}
    slots = e2.day_slots_of(ds.person_days, id_map)
    assert slots[P1101] == [(1, 1.0), (2, 3.0)]
    assert slots[P1301] == [(1, 1.0)]


# ---------------------------------------------------------------------------
# CLI runner (synthetic adapter cache; no raw data needed)
# ---------------------------------------------------------------------------

def _write_synthetic_cache(cache_dir):
    ds = make_dataset()
    cache_dir.mkdir(parents=True, exist_ok=True)
    households = pd.DataFrame(
        [{"household_id": h} for h in ("H1", "H2", "H3")]
    )
    households.to_pickle(cache_dir / "households.pkl")
    ds.persons.to_pickle(cache_dir / "persons.pkl")
    ds.person_days.to_pickle(cache_dir / "person_days.pkl")
    ds.weekday_trips.to_pickle(cache_dir / "weekday_trips_collapsed.pkl")


def test_run_e2_cli_end_to_end(tmp_path):
    cache_dir = tmp_path / "cache"
    _write_synthetic_cache(cache_dir)
    cards_path = tmp_path / "cards.json"
    cards_path.write_text(json.dumps(make_cards()))
    out_dir = tmp_path / "out" / "e2-test"

    rc = run_e2.main(
        [
            "--cards", str(cards_path),
            "--out", str(out_dir),
            "--runs", "5",
            "--seed", "0",
            "--cache-dir", str(cache_dir),
        ]
    )
    assert rc == 0
    results = json.loads((out_dir / "results.json").read_text())
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert results["eval"] == "e2"
    assert results["n_runs"] == 5
    assert manifest["namespaces"] == [f"run{k}" for k in range(5)]
    assert manifest["spread_band"] == [0.8, 1.2]
    assert manifest["correlation_bar"] == 0.20
    assert len(manifest["cards_sha256"]) == 64
    assert manifest["n_cards"] == 4
    assert "adapter_version" in manifest
