"""Equivalent-sample-size (ESS) estimator for E7 units (A4.1 pin).

Implements, exactly as published, the estimation-and-inference procedure of
Gao, Han & Liang 2026 (arXiv 2601.12343, "How Well Do LLMs Predict Human
Behavior?"), the paper the sealed A4.1 text pins for E7's units:

  * block-out cross-validation (their §3.2): for a candidate training size
    N, partition the n observations into B = floor(n/N) disjoint training
    blocks; train on each block, evaluate on its complement; the CV risk is
    the mean of the per-block test errors (their eq. 3.2);
  * the three-component asymptotic variance (their eq. 4.5):
    sigma^2 = N*V_train + V_test + 2N*C, with V_train the variance of the
    per-block risks, V_test the variance of the per-observation
    block-averaged losses, and C their cross-covariance — valid in the
    fixed-N, n -> infinity regime (their Theorem 4.2/4.3);
  * the studentized one-sided test of H_0k: e_N <= e_LLM applied to the
    LOSS DIFFERENCE delta(f, Z) = l(f(X), Y) - l(f_LLM(X), Y) (their §4.3,
    eq. 4.6 — the exact test; their Remark 4.2's sum-of-variances form is
    conservative and deliberately NOT used);
  * the sequential testing procedure (their Algorithm 3.1) delivering the
    plugin ESS and a one-sided (1-alpha) lower confidence bound
    N_hat_alpha (their eq. 3.4/3.5, with the coarse-grid rule of their
    footnotes 3-4: N_hat_alpha = max{N_k : LB_k > 0} + 1).

E7 adaptation, pinned here BEFORE any tier scoring (the A4.1 requirement
that baseline and CV protocol are recorded in advance):

  * one OBSERVATION = one persona (a real diary record — the unit the
    sealed text counts: "trained on ESS(k) real diary records");
  * blocks are HOUSEHOLD-ATOMIC (the A2.1 doctrine: household members
    never straddle a train/test boundary), built by CRN-deterministic
    household assignment — block sizes are then household-granular
    approximations of N, and the realized sizes are reported;
  * the flexible baseline and the per-persona loss are supplied by the
    caller (the E7 manifest pins the E1 MNL falsification arm's
    day-structure + mode-choice model); this module owns only the pinned
    ESTIMATOR, not the model.

HARNESS-SIDE ONLY: consumed by E7 scoring drivers; never imported by
agent-facing code. Ordinary-day tier scoring is not blind, so this runs
after firing-set freeze without touching any sealed quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from world import crn

#: CRN site used for the household-to-block assignment shuffle (fresh
#: namespace; nothing shared with generation/simulation streams).
BLOCK_SITE = "ess:block_assign"


# ---------------------------------------------------------------------------
# block construction (household-atomic, CRN-deterministic)
# ---------------------------------------------------------------------------

def household_blocks(
    household_ids: Sequence[str],
    block_size: int,
    namespace: str = "ess_default",
) -> List[np.ndarray]:
    """Partition observation indices into household-atomic training blocks.

    Households are ordered by a CRN draw (deterministic given namespace and
    household id), then packed greedily into at most B = floor(n / block_size)
    blocks until each reaches ``block_size`` persons. Per the paper's §3.2 the
    leftover observations that fit no block are DISCARDED for that candidate
    size (their footnote 2: discarding does not affect the asymptotics).
    Under household atomicity each block overshoots ``block_size`` by a
    partial household on average, so the leftover is O(B) persons rather than
    the paper's O(1) and the LAST blocks may go unfilled; only FULLY-FILLED
    blocks are returned (adaptation recorded in the E7 manifest's loss-adapter
    pin: realized block count and sizes are reported alongside the nominal N).
    Raises unless at least two blocks fill (the variance estimator needs
    B >= 2).
    """
    hh = np.asarray([str(h) for h in household_ids])
    n = len(hh)
    n_blocks = n // int(block_size)
    if n_blocks < 2:
        raise ValueError(
            f"block_size {block_size} leaves fewer than two blocks (n={n}); "
            "the variance estimator needs B >= 2"
        )
    uniq = sorted(set(hh.tolist()))
    keys = [f"{namespace}:{BLOCK_SITE}:{h}" for h in uniq]
    order = np.argsort(crn.draws(keys), kind="stable")
    shuffled = [uniq[i] for i in order]

    members: Dict[str, List[int]] = {}
    for i, h in enumerate(hh):
        members.setdefault(h, []).append(i)

    blocks: List[List[int]] = [[] for _ in range(n_blocks)]
    b = 0
    for h in shuffled:
        if b >= n_blocks:
            break  # remaining households are the discarded tail
        blocks[b].extend(members[h])
        if len(blocks[b]) >= block_size:
            b += 1
    filled = blocks[:b]  # only fully-filled blocks participate
    if len(filled) < 2:
        raise ValueError(
            f"could not fill two blocks of {block_size} from n={n} "
            "household-atomically; use a smaller block size"
        )
    return [np.asarray(sorted(ix), dtype=int) for ix in filled]


# ---------------------------------------------------------------------------
# the block-out CV risk + three-component variance (eqs. 3.2, 4.5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockOutRisk:
    """Block-out CV risk of one candidate size, with its asymptotic
    variance components (all per the paper's eqs. 3.2 and 4.5)."""

    e_cv: float          # eq. 3.2: mean over blocks of the per-block risk
    sigma2: float        # eq. 4.5: N*V_train + V_test + 2N*C
    v_train: float
    v_test: float
    c_cross: float
    n_used: int          # observations participating (after any discard)
    block_sizes: Tuple[int, ...]


def blockout_risk(loss: np.ndarray, blocks: Sequence[np.ndarray]) -> BlockOutRisk:
    """Compute the block-out CV risk and variance from a loss matrix.

    ``loss[b, i]`` is the loss of the predictor trained on block ``b``
    evaluated on observation ``i``; entries for i in the training block b
    must be NaN (they are excluded by construction). Observations in no
    block (the discarded tail) still appear as test columns — exactly as in
    the paper, where every i outside S_kb is test for block b.
    """
    loss = np.asarray(loss, dtype=float)
    n_blocks, n = loss.shape
    if n_blocks != len(blocks):
        raise ValueError("loss rows must match blocks")
    if n_blocks < 2:
        raise ValueError("need at least two blocks")

    e_b = np.array([np.nanmean(loss[b]) for b in range(n_blocks)])
    e_cv = float(e_b.mean())
    v_train = float(e_b.var(ddof=1))

    # mu_i: block-averaged test loss per observation (their §4.2.1); an
    # observation inside training block b contributes only its other-block
    # losses (the NaN row is skipped by nanmean).
    mu = np.nanmean(loss, axis=0)
    v_test = float(np.var(mu[~np.isnan(mu)], ddof=1))

    block_size = int(len(blocks[0]))
    m_b = np.array([float(np.mean(mu[np.asarray(ix, dtype=int)])) for ix in blocks])
    c_cross = float(
        np.sum((e_b - e_b.mean()) * (m_b - m_b.mean())) / (n_blocks - 1)
    )

    sigma2 = block_size * v_train + v_test + 2.0 * block_size * c_cross
    return BlockOutRisk(
        e_cv=e_cv, sigma2=sigma2, v_train=v_train, v_test=v_test,
        c_cross=c_cross, n_used=n,
        block_sizes=tuple(len(ix) for ix in blocks),
    )


# ---------------------------------------------------------------------------
# sequential ESS estimation on the loss difference (§3.4, §4.3)
# ---------------------------------------------------------------------------

@dataclass
class ESSResult:
    """Plugin ESS + one-sided lower confidence bound (their eqs. 3.3-3.5)."""

    plugin: Optional[int]          # min{N_k : delta_cv <= 0}; None = beyond grid
    lower_bound: int               # N_hat_alpha (coarse-grid rule)
    exceeds_grid: bool             # every H_0k rejected -> N* > N_K
    alpha: float
    per_size: List[dict] = field(default_factory=list)


def ess_sequential(
    sizes: Sequence[int],
    diff_loss_of: Callable[[int], Tuple[np.ndarray, Sequence[np.ndarray]]],
    alpha: float = 0.05,
) -> ESSResult:
    """Algorithm 3.1 on the loss DIFFERENCE (their exact test, eq. 4.6).

    ``sizes`` is the increasing candidate grid N_1 < ... < N_K.
    ``diff_loss_of(N)`` returns ``(diff_loss, blocks)`` where
    ``diff_loss[b, i] = l(f_N^(b)(X_i), Y_i) - l(f_LLM(X_i), Y_i)`` (NaN
    for i in training block b) — the block-out machinery applied to the
    difference loss, which is algebraically the paper's Delta_k because the
    LLM is a fixed rule (their footnote 7).

    Sequential stopping (their Algorithm 3.1 + footnotes 3-4): test
    H_0k at level alpha in increasing order; stop at the first
    non-rejection; the lower confidence bound is N_{k-1} + 1 (1 when the
    first test already fails to reject). The plugin estimator is the
    smallest size whose CV difference is <= 0. If every null is rejected,
    N* exceeds the grid and both estimates are reported as beyond N_K.
    """
    from scipy.stats import norm

    sizes = [int(s) for s in sizes]
    if sizes != sorted(sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("sizes must be strictly increasing")
    z = float(norm.ppf(1.0 - alpha))

    per_size: List[dict] = []
    plugin: Optional[int] = None
    lower: Optional[int] = None
    stopped = False
    prev_size = 0
    for k, size in enumerate(sizes):
        if stopped:
            break
        diff_loss, blocks = diff_loss_of(size)
        risk = blockout_risk(diff_loss, blocks)
        n = risk.n_used
        se = float(np.sqrt(max(risk.sigma2, 0.0) / n))
        t = risk.e_cv / se if se > 0 else float("inf") * np.sign(risk.e_cv or 1)
        reject = t > z  # H_0k: e_N <= e_LLM rejected -> ML still worse
        lb = risk.e_cv - z * se
        per_size.append({
            "size": size, "delta_cv": risk.e_cv, "sigma2": risk.sigma2,
            "se": se, "t": float(t), "reject": bool(reject),
            "lower_bound_excess": lb,
            "block_sizes": list(risk.block_sizes),
        })
        if plugin is None and risk.e_cv <= 0:
            plugin = size
        if not reject:
            lower = prev_size + 1
            stopped = True
        prev_size = size

    exceeds = not stopped
    if exceeds:
        lower = sizes[-1] + 1
    return ESSResult(
        plugin=plugin, lower_bound=int(lower), exceeds_grid=exceeds,
        alpha=alpha, per_size=per_size,
    )
