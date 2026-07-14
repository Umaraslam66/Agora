"""Common Random Numbers (CRN) — the single hashing/draw utility for Agora.

WHY THIS FILE EXISTS — every stochastic decision in the project (LLM mode
sampling in serving.gateway.pick, MNL sampling in agents.logit_chooser.choose,
and the world's realized route choice here) must draw from ONE hashing
doctrine, so that two *twin worlds* (e.g. a policy-on and a policy-off
counterfactual, the masked and unmasked E5 arms, or the method and MNL E1
arms) that share a decision KEY reuse the SAME uniform draw. That shared draw
is what makes the difference between two arms a *paired* comparison — the
pairing power behind the sealed E1 threshold (ε = 0.00655) and the E5 paired
protocol (25% vs the 98% unpaired threshold), see 01_PREREGISTRATION.md §7
A2.1/A2.3 and docs/internal/M2_ARCH_SPEC.md D3.

SEEDING DOCTRINE — identical to serving.gateway.pick / agents.logit_chooser:

    seed = int(sha256(key.encode()).hexdigest()[:12], 16)   # 48-bit

That 48-bit integer is the shared seed. gateway.pick then feeds it to a
Mersenne-Twister ``random.Random`` for a single softmax sample; here, because
the world layer needs a *vectorised* fast path over 10k+ agents and needs the
scalar and vectorised paths to be bit-identical, the uniform is taken directly
from the same 48-bit seed:

    draw(key) = seed / 2**48   in [0, 1)

The *seed derivation* is the shared "one hashing doctrine" (D3); mapping that
seed to a uniform by ``seed / 2**48`` (rather than through Mersenne Twister)
is a world-layer choice that makes ``draws`` a pure numpy kernel and makes
``draw(key)`` and ``draws([key])[0]`` agree to the last bit. Pure stdlib +
numpy; no global RNG state, so a draw is reproducible across processes and
machines (sha256 is fixed) — the world analogue of the CRN determinism
doctrine (01_PREREGISTRATION.md §5).

KEY CONTRACT — ``"{namespace}:{persona_id}:{day_index}:{site}"``:
  * ``namespace`` — a run-level ensemble stream label (``run0``, ``run1``, …).
    Different namespaces give INDEPENDENT draws (the N≥20 ensemble); the SAME
    namespace keeps arms/twin-worlds PAIRED within a run.
  * ``persona_id`` — the agent/persona identifier (caller-supplied).
  * ``day_index`` — the simulated day.
  * ``site`` — the decision site: ``pattern``, ``trip{i}:mode``,
    ``trip{i}:band``, or ``route`` (the world's corridor route choice).

Twin worlds and paired arms pair by SHARING keys: they build the identical key
string, so they draw the identical uniform, so a counterfactual only moves the
agents whose decision threshold the shared uniform now falls on the other side
of.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

# 48 bits = 12 hex chars = the first 6 bytes of the sha256 digest. This is the
# EXACT width serving.gateway.pick and agents.logit_chooser.choose seed with
# (`hexdigest()[:12]`); keep it in lock-step with them (one hashing doctrine).
_SEED_BITS = 48
_SEED_SPAN = float(1 << _SEED_BITS)  # 2**48


def seed_of(key: str) -> int:
    """The 48-bit CRN seed for a key — ``int(sha256(key)[:12], 16)``.

    This is the shared seed serving.gateway.pick and agents.logit_chooser use;
    exposed so callers/tests can assert the doctrine is literally identical."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:_SEED_BITS // 4], 16)


def draw(key: str) -> float:
    """A single CRN uniform in [0, 1) for ``key`` (see module doctrine).

    Deterministic and stateless: the same key always yields the same float,
    in this or any other process, with no dependence on any global RNG."""
    return seed_of(key) / _SEED_SPAN


def draws(keys: Sequence[str]) -> np.ndarray:
    """Vectorised CRN uniforms for a sequence of keys — the 10k+ agent fast
    path. Bit-identical, element-for-element, to ``draw`` on each key.

    The sha256 itself must go through hashlib (numpy has no sha256), so this
    is a tight Python loop over the digests feeding a numpy uint64 buffer; the
    [0,1) mapping is then one vectorised numpy divide. Comfortably under the
    100 ms / 10k-key budget (the hashing loop dominates at a few ms)."""
    n = len(keys)
    seeds = np.empty(n, dtype=np.uint64)
    for i, key in enumerate(keys):
        # digest()[:6] is the same 48 bits as hexdigest()[:12] (see seed_of).
        seeds[i] = int.from_bytes(hashlib.sha256(key.encode()).digest()[:6], "big")
    return seeds.astype(np.float64) / _SEED_SPAN


def pick_weighted(key: str, items: Sequence, weights: Sequence[float]):
    """Pick one of ``items`` by ``weights`` using ``key``'s CRN uniform.

    Inverse-CDF over the GIVEN item order, structurally identical to
    serving.gateway.pick's cumulative walk (``r = u * sum(weights)``; return
    the first item whose running total reaches ``r``). Because the order is
    fixed and the uniform is the shared per-key draw, a weight shift only
    re-selects when the uniform crosses a cumulative boundary — the same
    twin-coupling property the realized route layer relies on."""
    if len(items) != len(weights):
        raise ValueError("items and weights must have equal length")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    r = draw(key) * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r < acc:
            return item
    return items[-1]
