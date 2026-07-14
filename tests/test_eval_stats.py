"""Deterministic unit tests for evaluation/bootstrap_ci.py and
evaluation/permutation_placebo.py.

Fixed seeds throughout; numpy-only; no file I/O; designed to run in well
under 5 seconds.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.bootstrap_ci import bootstrap_ci, bootstrap_ci_per_segment
from evaluation.permutation_placebo import observed_contrast, permutation_placebo, weighted_mean


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_contains_true_mean():
    true_mean = 5.0
    data_rng = np.random.default_rng(42)
    xs = (true_mean + data_rng.normal(0.0, 1.0, size=300)).tolist()

    lo, med, hi = bootstrap_ci(xs, n_boot=2000, rng=random.Random(7))

    assert lo < true_mean < hi
    assert lo < med < hi


def test_bootstrap_ci_empty_sample_returns_zeros():
    assert bootstrap_ci([], n_boot=100, rng=random.Random(1)) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_per_segment_reproducible_and_covers_true_means():
    data_rng = np.random.default_rng(1)
    values_by_segment = {
        "alpha": (2.0 + data_rng.normal(0.0, 1.0, 200)).tolist(),
        "beta": (-1.0 + data_rng.normal(0.0, 1.0, 200)).tolist(),
    }

    r1 = bootstrap_ci_per_segment(values_by_segment, n_boot=500, seed=99)
    r2 = bootstrap_ci_per_segment(values_by_segment, n_boot=500, seed=99)

    # same seed + same inputs -> bit-identical result (shared continuing RNG
    # stream across segments, per the module's documented seed handling)
    assert r1 == r2

    assert r1["alpha"]["ci_lo"] < 2.0 < r1["alpha"]["ci_hi"]
    assert r1["beta"]["ci_lo"] < -1.0 < r1["beta"]["ci_hi"]


# ---------------------------------------------------------------------------
# permutation_placebo
# ---------------------------------------------------------------------------

def test_permutation_placebo_null_data_high_p_value_not_cleared():
    """No real segment structure: labels are effectively random already, so
    the observed contrast should look unremarkable against the permutation
    null (high p-value, placebo bar not cleared)."""
    rng = np.random.default_rng(0)
    n = 400
    values = rng.normal(0.0, 1.0, n)
    weights = rng.uniform(0.5, 2.0, n)
    mask_a = np.arange(n) < 200
    mask_b = ~mask_a

    result = permutation_placebo(values, weights, mask_a, mask_b, n_draws=1000, seed=1)

    assert result["n_draws"] == 1000
    assert result["cleared"] is False
    assert result["p_value_two_sided"] > 0.05


def test_permutation_placebo_injected_contrast_clears_bar():
    """A large injected segment contrast should be far outside the
    permutation null: low p-value, observed |contrast| exceeds perm p95,
    placebo bar cleared."""
    rng = np.random.default_rng(2)
    n = 400
    mask_a = np.arange(n) < 200
    mask_b = ~mask_a
    values = np.where(mask_a, 1.0, 0.0) + rng.normal(0.0, 0.05, n)
    weights = rng.uniform(0.5, 2.0, n)

    result = permutation_placebo(values, weights, mask_a, mask_b, n_draws=1000, seed=3)

    assert result["cleared"] is True
    assert result["p_value_two_sided"] < 0.01
    assert result["observed_abs_contrast"] > result["perm_abs_p95"]
    assert set(result["perm_abs_quantiles"].keys()) == {5, 25, 50, 75, 95}


def test_permutation_placebo_rejects_overlapping_masks():
    values = np.zeros(4)
    weights = np.ones(4)
    mask_a = np.array([True, True, False, False])
    mask_b = np.array([True, False, False, True])  # overlaps mask_a at idx 0

    try:
        permutation_placebo(values, weights, mask_a, mask_b, n_draws=10, seed=0)
        assert False, "expected ValueError for overlapping masks"
    except ValueError:
        pass


def test_weighted_mean_and_observed_contrast_match_numpy_average():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([1.0, 1.0, 2.0, 2.0])
    mask_a = np.array([True, True, False, False])
    mask_b = np.array([False, False, True, True])

    got_a = weighted_mean(values, weights, mask_a)
    want_a = np.average(values[mask_a], weights=weights[mask_a])
    assert abs(got_a - want_a) < 1e-12

    got_contrast = observed_contrast(values, weights, mask_a, mask_b)
    want_contrast = want_a - np.average(values[mask_b], weights=weights[mask_b])
    assert abs(got_contrast - want_contrast) < 1e-12
