#!/usr/bin/env python3
"""Weight-preserving segment-label permutation placebo.

Generic statistical utility for pre-registered distributional evaluation:
given per-record values (a statistic computed once per record, e.g. a
treatment/contrast effect) and per-record weights, plus two disjoint boolean
masks selecting segment A and segment B, this tests whether the observed
weighted-mean contrast between the two segments is distinguishable from
random relabeling — i.e. whether the segmentation carries real structure, or
whether an equally-sized random split of the same records would produce a
contrast just as large.

STATISTICAL SCHEME (weight-preserving segment-label permutation):
    1. Restrict to the "covered" records: those in segment A or segment B
       (mask_a | mask_b). Records in neither segment are excluded.
    2. Compute the observed contrast: weighted_mean(A) - weighted_mean(B),
       over the covered records.
    3. For each of `n_draws` permutation draws: shuffle the covered records'
       segment LABELS only. Each record keeps its own (value, weight) pair —
       only which segment it is assigned to changes. The size of the
       relabeled "A" group is held fixed at |mask_a| (segment sizes are
       preserved), so weight_sum(A) is not itself part of the null being
       tested. Compute |weighted_mean(A') - weighted_mean(B')| for the
       shuffled labels.
    4. The permutation p95 threshold is the 95th percentile of the |contrast|
       draws. The placebo bar is CLEARED iff the observed |contrast| exceeds
       this threshold. A two-sided p-value is also reported:
           p = (1 + #draws with |contrast_perm| >= |contrast_obs|) / (1 + n_draws)
       (the "+1" makes the test valid for finite `n_draws`, i.e. never
       reports p = 0).

DRAWS CONVENTION:
    At least 1000 permutation draws (`n_draws >= 1000`) is the registered
    minimum for this placebo to be treated as adequately powered; the
    default here is 1000. Pass a larger `n_draws` (e.g. 2000-10000) for a
    tighter p95/p-value estimate.

This module intentionally knows nothing about what "value", "weight",
"segment A", or "segment B" mean in any particular study — those are
supplied by the caller. It is the generic inference machinery only.
"""
from typing import Dict, Optional, Sequence

import numpy as np

DEFAULT_QUANTILES = (5, 25, 50, 75, 95)
MIN_RECOMMENDED_DRAWS = 1000


def weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    """Weighted mean of `values[mask]` with weights `weights[mask]`."""
    return float(np.average(values[mask], weights=weights[mask]))


def observed_contrast(
    values: np.ndarray, weights: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray
) -> float:
    """Weighted-mean contrast: weighted_mean(A) - weighted_mean(B)."""
    return weighted_mean(values, weights, mask_a) - weighted_mean(values, weights, mask_b)


def permutation_placebo(
    values: Sequence[float],
    weights: Sequence[float],
    mask_a: Sequence[bool],
    mask_b: Sequence[bool],
    n_draws: int = MIN_RECOMMENDED_DRAWS,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    quantiles: Sequence[int] = DEFAULT_QUANTILES,
) -> Dict:
    """Weight-preserving segment-label permutation placebo.

    Args:
        values: per-record statistic (1-D array-like).
        weights: per-record weight (1-D array-like, same length as values).
        mask_a: boolean mask selecting segment A records.
        mask_b: boolean mask selecting segment B records. Must be disjoint
            from mask_a; records covered by neither are excluded.
        n_draws: number of permutation draws (>= 1000 recommended; see
            module docstring, "DRAWS CONVENTION").
        rng: an existing `numpy.random.Generator` to draw from (share one
            across several placebo calls in a sweep for a fully
            reproducible, continuing RNG stream). If omitted, one is built
            from `seed`.
        seed: seed for a fresh `numpy.random.default_rng`, used only if
            `rng` is not supplied.
        quantiles: percentiles of the permutation |contrast| distribution to
            report alongside p95.

    Returns:
        dict with keys:
            n_draws, observed_contrast, observed_abs_contrast,
            perm_abs_p95, cleared (bool), p_value_two_sided,
            perm_abs_quantiles (dict of {quantile: value}).
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    if values.shape != weights.shape or values.shape != mask_a.shape or values.shape != mask_b.shape:
        raise ValueError("values, weights, mask_a, mask_b must all be the same shape")
    if np.any(mask_a & mask_b):
        raise ValueError("mask_a and mask_b must be disjoint")

    obs_diff = observed_contrast(values, weights, mask_a, mask_b)

    cov = mask_a | mask_b
    dc, wc = values[cov], weights[cov]
    n_a = int(mask_a.sum())
    nc = len(dc)

    perm_abs = np.empty(n_draws)
    for i in range(n_draws):
        p = rng.permutation(nc)
        la = np.zeros(nc, dtype=bool)
        la[p[:n_a]] = True
        perm_abs[i] = abs(
            np.average(dc[la], weights=wc[la]) - np.average(dc[~la], weights=wc[~la])
        )

    thr = float(np.percentile(perm_abs, 95))
    obs = abs(obs_diff)

    return {
        "n_draws": n_draws,
        "observed_contrast": float(obs_diff),
        "observed_abs_contrast": obs,
        "perm_abs_p95": thr,
        "cleared": bool(obs > thr),
        "p_value_two_sided": float((1 + int((perm_abs >= obs).sum())) / (1 + n_draws)),
        "perm_abs_quantiles": {
            int(q): float(v) for q, v in zip(quantiles, np.percentile(perm_abs, list(quantiles)))
        },
    }
