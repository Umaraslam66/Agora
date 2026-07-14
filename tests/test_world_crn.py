"""Tests for world.crn — the single CRN hashing/draw utility.

These pin the seeding doctrine (identical to serving.gateway.pick /
agents.logit_chooser.choose), the scalar/vectorised bit-identity, the
inverse-CDF pick, statelessness across processes, and the 10k-draw budget.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from world import crn

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _gateway_seed(key: str) -> int:
    """The literal seed formula serving.gateway.pick uses (gateway.py:106) and
    agents.logit_chooser.choose uses (logit_chooser.py:248). world.crn MUST
    match this exactly — that is the 'one hashing doctrine'."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def test_seed_matches_gateway_doctrine():
    for key in ["run0:P1:0:route", "run1:P07341:12:trip0:mode", "x", ""]:
        assert crn.seed_of(key) == _gateway_seed(key)


def test_draw_in_unit_interval():
    keys = [f"run0:P{i}:0:route" for i in range(5000)]
    u = np.array([crn.draw(k) for k in keys])
    assert (u >= 0.0).all() and (u < 1.0).all()


def test_draw_is_deterministic():
    key = "run0:P123:7:route"
    assert crn.draw(key) == crn.draw(key)
    # And equals the seed/2**48 derivation from the shared doctrine.
    assert crn.draw(key) == _gateway_seed(key) / float(1 << 48)


def test_draws_bit_identical_to_scalar():
    keys = [f"run0:P{i}:3:route" for i in range(2000)]
    vec = crn.draws(keys)
    scalar = np.array([crn.draw(k) for k in keys])
    assert np.array_equal(vec, scalar)  # exact, not approximate


def test_draws_stateless_across_processes():
    """A fresh interpreter must reproduce the same draws bit-for-bit — proof
    there is no global RNG state (sha256 is the only source)."""
    keys = [f"run0:P{i}:0:route" for i in range(256)]
    here = crn.draws(keys)
    here_hash = hashlib.sha256(here.tobytes()).hexdigest()
    snippet = (
        "import hashlib, numpy as np; from world import crn; "
        "keys=[f'run0:P{i}:0:route' for i in range(256)]; "
        "print(hashlib.sha256(crn.draws(keys).tobytes()).hexdigest())"
    )
    out = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == here_hash


def test_different_namespaces_are_independent():
    """run0 vs run1 give different draws for the same persona/day/site."""
    a = crn.draws([f"run0:P{i}:0:route" for i in range(1000)])
    b = crn.draws([f"run1:P{i}:0:route" for i in range(1000)])
    # Not equal anywhere near element-wise; assert they clearly differ.
    assert not np.array_equal(a, b)
    assert np.mean(a == b) < 0.01


def test_pick_weighted_fixed_order_and_reuse():
    items = ["T", "S", "F", "D"]
    # Empirical frequencies track the weights (loose sanity band).
    weights = [0.4, 0.1, 0.3, 0.2]
    counts = {it: 0 for it in items}
    for i in range(20000):
        counts[crn.pick_weighted(f"run0:P{i}:0:route", items, weights)] += 1
    freqs = {it: counts[it] / 20000 for it in items}
    for it, w in zip(items, weights):
        assert abs(freqs[it] - w) < 0.03

    # Twin-coupling at the pick level: the SAME key reuses the same uniform,
    # so shifting a later weight down (raising an earlier cumulative boundary)
    # only re-picks keys whose uniform sat in the shifted band.
    keys = [f"run0:P{i}:0:route" for i in range(4000)]
    base = [crn.pick_weighted(k, items, [0.4, 0.1, 0.3, 0.2]) for k in keys]
    shifted = [crn.pick_weighted(k, items, [0.5, 0.1, 0.25, 0.15]) for k in keys]
    # T's band grew (0.4 -> 0.5): every key that was T stays T; only some
    # non-T keys may become T. No key that was T leaves T.
    for b, s in zip(base, shifted):
        if b == "T":
            assert s == "T"


def test_draws_10k_performance():
    keys = [f"run0:P{i}:0:route" for i in range(10000)]
    crn.draws(keys[:10])  # warm
    t0 = time.perf_counter()
    crn.draws(keys)
    elapsed = time.perf_counter() - t0
    # Target: 10k draws well under 100 ms (observed a few ms).
    assert elapsed < 0.1, f"draws(10k) took {elapsed * 1e3:.1f} ms (budget 100 ms)"
