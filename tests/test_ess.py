"""Tests for evaluation/ess.py — the A4.1-pinned ESS estimator (Gao, Han &
Liang 2026): block-out CV risk, the three-component variance, household-
atomic blocking, and the sequential lower-confidence-bound procedure."""
import numpy as np
import pytest

from evaluation import ess


def _blocks_of(n, size):
    ids = [f"H{i:04d}" for i in range(n)]  # one-person households
    return ids, ess.household_blocks(ids, size, namespace="t")


# ---------------------------------------------------------------------------
# household-atomic blocking
# ---------------------------------------------------------------------------

def test_blocks_are_household_atomic_and_deterministic():
    hh = [f"H{i // 3:03d}" for i in range(300)]  # 3-person households
    b1 = ess.household_blocks(hh, 30, namespace="x")
    b2 = ess.household_blocks(hh, 30, namespace="x")
    assert all((a == b).all() for a, b in zip(b1, b2))  # CRN-deterministic
    for block in b1:
        hh_in = {hh[i] for i in block}
        for h in hh_in:  # every member of a block household is in the block
            members = {i for i, x in enumerate(hh) if x == h}
            assert members <= set(block.tolist())
    assert len(b1) == 300 // 30
    # a different namespace reshuffles
    b3 = ess.household_blocks(hh, 30, namespace="y")
    assert any((a != b).any() for a, b in zip(b1, b3))


def test_blocks_reject_degenerate_partitions():
    with pytest.raises(ValueError):
        ess.household_blocks(["H1"] * 10, 8, namespace="t")  # B < 2


def test_blocks_partial_fill_keeps_only_full_blocks():
    # 2-person households, block size 5: every block overshoots to 6, so the
    # nominal floor(100/5) = 20 blocks cannot all fill from 50 households —
    # only fully-filled blocks come back, each household-atomic and >= 5.
    hh = [f"H{i // 2:03d}" for i in range(100)]
    blocks = ess.household_blocks(hh, 5, namespace="t")
    assert 2 <= len(blocks) <= 20
    used = set()
    for block in blocks:
        assert len(block) >= 5
        hh_in = {hh[i] for i in block}
        for h in hh_in:
            members = {i for i, x in enumerate(hh) if x == h}
            assert members <= set(block.tolist())
        assert not (set(block.tolist()) & used)  # disjoint
        used |= set(block.tolist())


# ---------------------------------------------------------------------------
# block-out risk + variance (paper eqs. 3.2 / 4.5)
# ---------------------------------------------------------------------------

def _loss_matrix(blocks, n, fill):
    """fill(b, i) -> loss; training-block entries NaN."""
    L = np.full((len(blocks), n), np.nan)
    for b, ix in enumerate(blocks):
        train = set(ix.tolist())
        for i in range(n):
            if i not in train:
                L[b, i] = fill(b, i)
    return L


def test_blockout_risk_recovers_mean_and_iid_variance():
    rng = np.random.default_rng(7)
    n, size = 400, 50
    _, blocks = _blocks_of(n, size)
    # losses independent of the training block: e_cv = the plain mean,
    # V_train ~ test-sampling noise only, and sigma^2/n ~ Var(loss)/n
    base = rng.normal(2.0, 1.0, size=n)
    L = _loss_matrix(blocks, n, lambda b, i: base[i])
    r = ess.blockout_risk(L, blocks)
    assert r.e_cv == pytest.approx(float(np.mean(base)), abs=0.02)
    # with block-independent losses mu_i == base_i exactly, so V_test is
    # the sample variance of the losses
    assert r.v_test == pytest.approx(float(np.var(base, ddof=1)), rel=1e-6)
    assert r.sigma2 > 0


def test_blockout_risk_requires_matching_blocks():
    _, blocks = _blocks_of(100, 20)
    with pytest.raises(ValueError):
        ess.blockout_risk(np.zeros((3, 100)), blocks)


# ---------------------------------------------------------------------------
# sequential procedure (paper Algorithm 3.1)
# ---------------------------------------------------------------------------

def _diff_provider(n, curve, noise=0.05, seed=11):
    """Synthetic difference-loss provider: at candidate size N the true
    mean difference is curve(N) (positive = ML worse than the fixed rule),
    iid noise across observations."""
    rng = np.random.default_rng(seed)
    ids = [f"H{i:04d}" for i in range(n)]

    def provider(size):
        blocks = ess.household_blocks(ids, size, namespace=f"p{size}")
        eps = rng.normal(0.0, noise, size=n)
        L = _loss_matrix(blocks, n, lambda b, i: curve(size) + eps[i])
        return L, blocks

    return provider


def test_sequential_finds_crossing():
    # ML strictly beats the fixed rule from N=160 on (excess -0.1 there)
    curve = lambda N: 0.4 - N / 320.0  # noqa: E731
    res = ess.ess_sequential(
        [40, 80, 160, 320], _diff_provider(2000, curve), alpha=0.05
    )
    assert res.plugin == 160
    assert not res.exceeds_grid
    # the null stops being rejected at the first size whose true excess is
    # negative, so the lower bound is the previous size + 1
    assert res.lower_bound == 81
    assert [d["reject"] for d in res.per_size] == [True, True, False]


def test_sequential_all_rejected_reports_beyond_grid():
    res = ess.ess_sequential(
        [40, 80], _diff_provider(2000, lambda N: 0.5), alpha=0.05
    )
    assert res.exceeds_grid and res.plugin is None
    assert res.lower_bound == 81  # N_K + 1


def test_sequential_immediate_nonrejection_gives_one():
    res = ess.ess_sequential(
        [40, 80], _diff_provider(2000, lambda N: -0.5), alpha=0.05
    )
    assert res.plugin == 40
    assert res.lower_bound == 1
    assert len(res.per_size) == 1  # stopped at the first step


def test_sequential_rejects_bad_grid():
    with pytest.raises(ValueError):
        ess.ess_sequential([80, 40], _diff_provider(500, lambda N: 0.0))
