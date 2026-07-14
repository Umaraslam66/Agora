"""Habit-strength memory for persona cards.

Numeric substrate ported from the predecessor's memory substrate (Java):
a deterministic per-cell EMA/shock table plus trailing daily-use counters.
No LLM anywhere in the write path; every update is a pure function of the
previous state plus one observation, so replay is deterministic.

The substrate feeds Agora's two-brain architecture in two ways:

* **Surprise hook** — the prediction-error z-score computed here is the
  signal that triggers a slow-brain (LLM) call to rewrite the persona
  card.  The card keeps a surprise log conceptually capped at
  ``SURPRISE_LOG_CAP`` (= 5) entries; that cap is enforced by the card
  logic, not here.
* **Habit counters** — :class:`HabitCounter` wraps the trailing-counter
  machinery into per-rule habit-strength counters (days a rule has been
  followed).  When the world reverts, strong habits resist reverting —
  the hysteresis mechanism tested by eval E6.

Formulas preserved exactly from the predecessor:

EMA update (per cell, alpha = 0.3 by default)::

    ema' = (1 - alpha) * ema + alpha * x

  Both EMAs (realized minutes, and realized-minus-expected delta) are
  seeded at 0.0, so the first observation yields ``alpha * x`` — NOT the
  raw observation.  This matches the predecessor's empty-cell
  initialization and must not be "fixed".

Prediction-error z-score (fixed prior sigma, 10.0 minutes by default)::

    z = |realized - expected| / sigma_prior      (0.0 when sigma <= 0)

Surprise-weighted EMA (kill-switch, OFF by default; with it off the
update is byte-identical to the plain EMA)::

    alpha_eff = min(alpha_base * (1 + gain * min(|z|, z_cap) / z_cap),
                    alpha_max)

  Defaults: gain = 1.0, z_cap = 3.0, alpha_max = 0.9.

Outcome encoding (last observation, in raw minutes; boundaries
inclusive)::

    delta <= -5.0  ->  -1   (notably faster than expected)
    delta >= +5.0  ->  +1   (notably slower)
    otherwise      ->   0   (roughly on time)

Shock record: the largest-|z| observation in a trailing 30-day window.
A new observation replaces the record iff the record is empty, the new
|z| is STRICTLY greater, or the record is stale (age > window).  The
first observation of a cell always records a shock, even at z = 0.
Shocks age by one day at each day boundary the cell was NOT observed,
and expire once age exceeds the window.  (Cells observed on a given day
are left exactly as the observation update produced them — the
predecessor's deliberate same-cycle rule, preserved here; a touched
shock's age stalls one extra day versus a literal daily increment.)

Trailing counters: one entry per lived day (a quiet, zero-trip day is a
real entry, not a gap), deque hard-capped at 30 days, oldest dropped.
A k-day count sums the newest ``min(k, days_so_far)`` entries.  Counts
are ``None`` only when there is no history at all; a mode absent from
existing history counts 0.  The "usual" (modal) key over the window
breaks ties by a fixed priority order, deterministically.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterator, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ALPHA_BASE",
    "SIGMA_PRIOR_MINUTES",
    "SHOCK_WINDOW_DAYS",
    "Z_CAP",
    "SURPRISE_GAIN",
    "ALPHA_MAX",
    "NOTABLE_DELTA_MINUTES",
    "SURPRISE_Z_THRESHOLD",
    "MAX_TRAILING_DAYS",
    "SHORT_TRAILING_DAYS",
    "SURPRISE_LOG_CAP",
    "DEFAULT_MODE_PRIORITY",
    "SubstrateConfig",
    "DEFAULT_CONFIG",
    "LastShock",
    "EMPTY_SHOCK",
    "SubstrateCell",
    "TrailingTally",
    "HabitMemory",
    "HabitCounter",
    "encode_outcome",
    "prediction_error_z",
    "is_surprise",
]

# ---------------------------------------------------------------------
# Constants — same values as the predecessor's defaults.  (There they
# were resolvable from system properties / env vars; here they are plain
# constants, overridable per instance via SubstrateConfig.)
# ---------------------------------------------------------------------

#: Base EMA smoothing factor.
ALPHA_BASE = 0.3

#: Fixed prior stddev (minutes) for the shock z-score, in place of a
#: running per-cell stddev estimate (the predecessor's sanctioned
#: simplification).
SIGMA_PRIOR_MINUTES = 10.0

#: Trailing window (lived days) a shock record survives.
SHOCK_WINDOW_DAYS = 30

#: |z| is capped here before feeding the surprise-weighted alpha.
Z_CAP = 3.0

#: Gain on the (capped, normalized) |z| in the surprise-weighted alpha.
SURPRISE_GAIN = 1.0

#: Hard ceiling on the effective alpha when surprise weighting is on.
ALPHA_MAX = 0.9

#: |delta| (minutes) at which an outcome is "notably" early/late —
#: the boundary of the predecessor's outcome encoder (inclusive).
NOTABLE_DELTA_MINUTES = 5.0

#: Default boolean-surprise boundary in z units.  The predecessor never
#: converted z to a boolean; this default is derived from its only
#: threshold-like semantics: NOTABLE_DELTA_MINUTES / SIGMA_PRIOR_MINUTES
#: (= 0.5), inclusive, matching the outcome encoder's inclusive +-5.0.
SURPRISE_Z_THRESHOLD = NOTABLE_DELTA_MINUTES / SIGMA_PRIOR_MINUTES

#: Trailing-day window for the long habit counters (and the deque cap).
MAX_TRAILING_DAYS = 30

#: Trailing-day window for the short habit counters — reads the newest
#: 7 of the same entries.
SHORT_TRAILING_DAYS = 7

#: Persona-card surprise-log cap.  NOT enforced here — the cap lives in
#: the card logic; the constant is exported so there is one source of
#: truth for it.
SURPRISE_LOG_CAP = 5

#: Fixed priority order used ONLY as the deterministic tie-break for the
#: modal ("usual") key over the trailing window.  Same order as the
#: predecessor; callers may pass their own.
DEFAULT_MODE_PRIORITY: Tuple[str, ...] = ("transit", "car", "ride", "bike", "walk")


@dataclass(frozen=True)
class SubstrateConfig:
    """Tunable substrate parameters, defaulting to the ported values.

    ``surprise_weighting`` is the predecessor's kill-switch: OFF by
    default, and with it off the cell update is byte-identical to the
    plain-EMA formula.
    """

    alpha_base: float = ALPHA_BASE
    sigma_prior_minutes: float = SIGMA_PRIOR_MINUTES
    shock_window_days: int = SHOCK_WINDOW_DAYS
    z_cap: float = Z_CAP
    surprise_gain: float = SURPRISE_GAIN
    alpha_max: float = ALPHA_MAX
    surprise_weighting: bool = False
    surprise_z_threshold: float = SURPRISE_Z_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_base": self.alpha_base,
            "sigma_prior_minutes": self.sigma_prior_minutes,
            "shock_window_days": self.shock_window_days,
            "z_cap": self.z_cap,
            "surprise_gain": self.surprise_gain,
            "alpha_max": self.alpha_max,
            "surprise_weighting": self.surprise_weighting,
            "surprise_z_threshold": self.surprise_z_threshold,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubstrateConfig":
        return cls(
            alpha_base=float(data["alpha_base"]),
            sigma_prior_minutes=float(data["sigma_prior_minutes"]),
            shock_window_days=int(data["shock_window_days"]),
            z_cap=float(data["z_cap"]),
            surprise_gain=float(data["surprise_gain"]),
            alpha_max=float(data["alpha_max"]),
            surprise_weighting=bool(data["surprise_weighting"]),
            surprise_z_threshold=float(data["surprise_z_threshold"]),
        )


DEFAULT_CONFIG = SubstrateConfig()


# ---------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------

def encode_outcome(delta_minutes: float) -> int:
    """Encode a realized-minus-expected delta into {-1, 0, +1}.

    Ported exactly: ``delta <= -5.0`` -> -1 (notably faster),
    ``delta >= +5.0`` -> +1 (notably slower), else 0.  Boundaries are
    inclusive.
    """
    if delta_minutes <= -NOTABLE_DELTA_MINUTES:
        return -1
    if delta_minutes >= NOTABLE_DELTA_MINUTES:
        return 1
    return 0


def prediction_error_z(
    realized_minutes: float,
    expected_minutes: float,
    sigma_prior_minutes: float = SIGMA_PRIOR_MINUTES,
) -> float:
    """|z| of one observation against the fixed prior sigma.

    Ported exactly, including the guard: 0.0 whenever ``sigma <= 0``
    (the predecessor's ``sigma > 0 ? |delta/sigma| : 0`` ternary).
    """
    if sigma_prior_minutes > 0:
        return abs((realized_minutes - expected_minutes) / sigma_prior_minutes)
    return 0.0


def is_surprise(
    realized_minutes: float,
    expected_minutes: float,
    config: SubstrateConfig = DEFAULT_CONFIG,
) -> bool:
    """Boolean surprise: |z| >= threshold (inclusive).

    This is the hook that triggers a slow-brain (LLM) persona-card
    rewrite.  The predecessor exposed the raw z, never a boolean; the
    default threshold (0.5 z = 5 minutes at the default sigma) is
    derived from its outcome encoder's inclusive "notably late/early"
    boundary — see :data:`SURPRISE_Z_THRESHOLD`.
    """
    z = prediction_error_z(
        realized_minutes, expected_minutes, config.sigma_prior_minutes
    )
    return z >= config.surprise_z_threshold


# ---------------------------------------------------------------------
# Shock record
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class LastShock:
    """Largest-|z| observation in the trailing shock window, with age.

    The empty sentinel is ``age_in_days = -1`` (peak 0.0, delta 0.0),
    exactly as in the predecessor.
    """

    peak_abs_z: float = 0.0
    age_in_days: int = -1
    delta_minutes: float = 0.0

    def is_empty(self) -> bool:
        return self.age_in_days < 0

    def aged(self, shock_window_days: int = SHOCK_WINDOW_DAYS) -> "LastShock":
        """One lived-day-boundary tick: age by 1, expire past the window.

        Empty stays empty; a shock aging beyond the window (new age
        strictly greater than ``shock_window_days``) becomes empty.
        """
        if self.is_empty():
            return EMPTY_SHOCK
        new_age = self.age_in_days + 1
        if new_age > shock_window_days:
            return EMPTY_SHOCK
        return LastShock(self.peak_abs_z, new_age, self.delta_minutes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_abs_z": self.peak_abs_z,
            "age_in_days": self.age_in_days,
            "delta_minutes": self.delta_minutes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LastShock":
        return cls(
            peak_abs_z=float(data["peak_abs_z"]),
            age_in_days=int(data["age_in_days"]),
            delta_minutes=float(data["delta_minutes"]),
        )


EMPTY_SHOCK = LastShock()


# ---------------------------------------------------------------------
# Substrate cell
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SubstrateCell:
    """One cell of the numeric substrate: EMA'd realized-vs-expected
    time for one context key, plus a trailing-window shock record.

    Immutable: every update is a pure function of the previous state
    plus one new observation; :class:`HabitMemory` replaces the map
    entry with the new instance instead of mutating in place.

    Fields (all ported):

    * ``ema_realized_minutes``    — EMA of realized minutes
    * ``ema_delta_vs_expectation`` — EMA of (realized - expected), the
      surprise signal in raw units
    * ``n``                       — trial count
    * ``last_outcome``            — encoded most-recent delta bucket
      (-1 faster / 0 on-time / +1 slower)
    * ``days_since_tried``        — lived days since last observed
    * ``last_shock``              — peak-|z| trailing shock record
    """

    ema_realized_minutes: float = 0.0
    ema_delta_vs_expectation: float = 0.0
    n: int = 0
    last_outcome: int = 0
    days_since_tried: int = 0
    last_shock: LastShock = EMPTY_SHOCK

    @classmethod
    def empty(cls) -> "SubstrateCell":
        """The all-zero starting cell (EMAs seeded at 0.0, empty shock)."""
        return cls()

    def with_observation(
        self,
        realized_minutes: float,
        expected_minutes: float,
        config: SubstrateConfig = DEFAULT_CONFIG,
    ) -> "SubstrateCell":
        """Standard (or, if enabled, surprise-weighted) EMA update for
        one realized observation — pure function, ported exactly.

        Steps (see module docstring for the formulas):

        1. ``delta = realized - expected``; ``|z| = |delta| / sigma``
           (0 if sigma <= 0).
        2. ``alpha = alpha_base``; if surprise weighting is ON,
           ``alpha = min(alpha_base * (1 + gain * min(|z|, z_cap) /
           z_cap), alpha_max)``.
        3. Both EMAs advance by ``(1 - alpha) * old + alpha * new``.
        4. ``last_outcome`` re-encoded from this delta alone.
        5. The shock record is replaced by (|z|, age 0, delta) iff it is
           empty, |z| is STRICTLY greater than the recorded peak, or the
           record is stale (age > window); otherwise kept as-is.
        6. ``n + 1``; ``days_since_tried`` resets to 0.

        Divergence guard (documented, not in the predecessor): with
        surprise weighting ON and ``z_cap <= 0`` the predecessor's
        arithmetic would produce NaN; here the boost is skipped and
        ``alpha_base`` used instead.
        """
        delta = realized_minutes - expected_minutes
        if config.sigma_prior_minutes > 0:
            abs_z = abs(delta / config.sigma_prior_minutes)
        else:
            abs_z = 0.0

        alpha = config.alpha_base
        if config.surprise_weighting and config.z_cap > 0:
            z_capped = min(abs_z, config.z_cap)
            alpha_effective = config.alpha_base * (
                1 + config.surprise_gain * z_capped / config.z_cap
            )
            alpha = min(alpha_effective, config.alpha_max)

        new_ema_realized = (1 - alpha) * self.ema_realized_minutes + alpha * realized_minutes
        new_ema_delta = (1 - alpha) * self.ema_delta_vs_expectation + alpha * delta
        new_last_outcome = encode_outcome(delta)

        if (
            self.last_shock.is_empty()
            or abs_z > self.last_shock.peak_abs_z
            or self.last_shock.age_in_days > config.shock_window_days
        ):
            new_shock = LastShock(abs_z, 0, delta)
        else:
            new_shock = self.last_shock

        return SubstrateCell(
            ema_realized_minutes=new_ema_realized,
            ema_delta_vs_expectation=new_ema_delta,
            n=self.n + 1,
            last_outcome=new_last_outcome,
            days_since_tried=0,
            last_shock=new_shock,
        )

    def aged(self, config: SubstrateConfig = DEFAULT_CONFIG) -> "SubstrateCell":
        """Lived-day-boundary aging for a cell NOT observed today:
        ``days_since_tried + 1`` and the shock record ages/expires.

        Only untouched cells are aged (see :meth:`HabitMemory.end_day`)
        — the predecessor's deliberate same-cycle rule, preserved: a
        cell observed today keeps exactly the state the observation
        update produced (its shock's age stalls one extra day versus a
        literal daily increment; conservative, keeps shocks salient
        slightly longer).
        """
        return SubstrateCell(
            ema_realized_minutes=self.ema_realized_minutes,
            ema_delta_vs_expectation=self.ema_delta_vs_expectation,
            n=self.n,
            last_outcome=self.last_outcome,
            days_since_tried=self.days_since_tried + 1,
            last_shock=self.last_shock.aged(config.shock_window_days),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ema_realized_minutes": self.ema_realized_minutes,
            "ema_delta_vs_expectation": self.ema_delta_vs_expectation,
            "n": self.n,
            "last_outcome": self.last_outcome,
            "days_since_tried": self.days_since_tried,
            "last_shock": self.last_shock.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubstrateCell":
        return cls(
            ema_realized_minutes=float(data["ema_realized_minutes"]),
            ema_delta_vs_expectation=float(data["ema_delta_vs_expectation"]),
            n=int(data["n"]),
            last_outcome=int(data["last_outcome"]),
            days_since_tried=int(data["days_since_tried"]),
            last_shock=LastShock.from_dict(data["last_shock"]),
        )


# ---------------------------------------------------------------------
# Trailing daily counters
# ---------------------------------------------------------------------

class TrailingTally:
    """Trailing-day window of per-day usage counts, ported exactly.

    One entry per lived day: a mapping of key -> count, plus a parallel
    "focal" mapping (in the predecessor: trips touching a work
    activity).  Hard-capped at ``window_days`` entries, oldest dropped.
    A quiet day (all-zero counts) is still a real entry.
    """

    def __init__(self, window_days: int = MAX_TRAILING_DAYS) -> None:
        self.window_days = int(window_days)
        self._days: Deque[Tuple[Dict[str, int], Dict[str, int]]] = deque()

    def record_day(
        self,
        counts: Mapping[str, int],
        focal_counts: Optional[Mapping[str, int]] = None,
    ) -> None:
        """Fold one lived day's counts into the window (oldest dropped
        past the cap).  Call once per day, INCLUDING zero-count days —
        a quiet day is a real day in the trailing window, not a gap."""
        self._days.append((dict(counts), dict(focal_counts or {})))
        while len(self._days) > self.window_days:
            self._days.popleft()

    @property
    def has_history(self) -> bool:
        """True once at least one lived day has closed."""
        return len(self._days) > 0

    def __len__(self) -> int:
        return len(self._days)

    def _newest_first(self, window_days: int) -> Iterator[Tuple[Dict[str, int], Dict[str, int]]]:
        counted = 0
        for day in reversed(self._days):
            if counted >= window_days:
                break
            yield day
            counted += 1

    def count(self, key: str, window_days: Optional[int] = None) -> Optional[int]:
        """Trailing count of ``key`` over the newest ``min(window_days,
        days-so-far)`` days, or ``None`` if there is no history AT ALL
        yet.  A key absent from existing history counts 0 (ported
        None-vs-0 semantics)."""
        if not self.has_history:
            return None
        window = self.window_days if window_days is None else window_days
        total = 0
        for counts, _focal in self._newest_first(window):
            total += counts.get(key, 0)
        return total

    def usual_focal_key(
        self,
        priority_order: Sequence[str] = DEFAULT_MODE_PRIORITY,
        window_days: Optional[int] = None,
    ) -> Optional[str]:
        """Modal key of the focal counts over the trailing window.

        ``None`` if there is no history at all, or history exists but
        no focal count occurred in the window.  Ties broken by the
        fixed ``priority_order`` (first listed wins — the strict ``>``
        of the ported loop), never by insertion order: deterministic.
        """
        if not self.has_history:
            return None
        window = self.window_days if window_days is None else window_days
        totals: Dict[str, int] = {}
        for _counts, focal in self._newest_first(window):
            for key, count in focal.items():
                totals[key] = totals.get(key, 0) + count
        best: Optional[str] = None
        best_count = 0
        for key in priority_order:
            count = totals.get(key, 0)
            if count > best_count:
                best_count = count
                best = key
        return best

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_days": self.window_days,
            "days": [
                {"counts": dict(counts), "focal_counts": dict(focal)}
                for counts, focal in self._days
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrailingTally":
        tally = cls(window_days=int(data["window_days"]))
        for day in data["days"]:
            tally.record_day(day["counts"], day["focal_counts"])
        return tally


# ---------------------------------------------------------------------
# Agent-level memory: sparse cell table + trailing tally
# ---------------------------------------------------------------------

class HabitMemory:
    """Deterministic numeric memory of one agent: a sparse table of
    :class:`SubstrateCell` keyed by an opaque context key, plus the
    trailing daily-use counters.  Ported day-cycle contract:

    1. :meth:`begin_day` once, before the day's observations (clears
       the touched-today marker set).
    2. :meth:`observe` per realized outcome (updates the cell, marks it
       touched, returns the boolean surprise signal for the slow-brain
       trigger).
    3. :meth:`record_daily_tally` once with the day's usage counts
       (including quiet days), then :meth:`end_day` (ages every cell
       NOT touched today).

    Keys are opaque and hashable; use strings (e.g. ``"car|b2|am"``) so
    persona-card serialization stays JSON-clean.  Only keys actually
    observed are ever materialized (sparse map).
    """

    def __init__(self, config: SubstrateConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._cells: Dict[str, SubstrateCell] = {}
        self._touched_today: set = set()
        self._tally = TrailingTally(MAX_TRAILING_DAYS)

    # -- day cycle ----------------------------------------------------

    def begin_day(self) -> None:
        """Clear the touched-today marker set; call once per lived day,
        before that day's observations."""
        self._touched_today.clear()

    def observe(
        self,
        key: str,
        realized_minutes: float,
        expected_minutes: float,
    ) -> bool:
        """Deterministic cell update for one realized outcome.

        Returns the boolean surprise signal (|z| >= threshold) — the
        hook that triggers a slow-brain persona-card rewrite.  The cell
        is updated regardless of whether the observation is a surprise.
        """
        current = self._cells.get(key, SubstrateCell.empty())
        self._cells[key] = current.with_observation(
            realized_minutes, expected_minutes, self.config
        )
        self._touched_today.add(key)
        return self.is_surprise(realized_minutes, expected_minutes)

    def end_day(self) -> None:
        """Lived-day-boundary aging pass: ages every cell NOT touched
        today; touched cells keep exactly the state :meth:`observe`
        left them in (ported same-cycle rule)."""
        for key in list(self._cells.keys()):
            if key not in self._touched_today:
                self._cells[key] = self._cells[key].aged(self.config)

    # -- cell access ---------------------------------------------------

    def cell(self, key: str) -> Optional[SubstrateCell]:
        return self._cells.get(key)

    def keys(self) -> Tuple[str, ...]:
        return tuple(self._cells.keys())

    # -- surprise hook ---------------------------------------------------

    def prediction_error_z(
        self, realized_minutes: float, expected_minutes: float
    ) -> float:
        return prediction_error_z(
            realized_minutes, expected_minutes, self.config.sigma_prior_minutes
        )

    def is_surprise(
        self, realized_minutes: float, expected_minutes: float
    ) -> bool:
        return is_surprise(realized_minutes, expected_minutes, self.config)

    # -- trailing counters ----------------------------------------------

    def record_daily_tally(
        self,
        mode_counts: Mapping[str, int],
        work_mode_counts: Optional[Mapping[str, int]] = None,
    ) -> None:
        """Fold one lived day's usage counts into the trailing window.
        Call once per day, including days with zero activity."""
        self._tally.record_day(mode_counts, work_mode_counts)

    @property
    def has_trailing_history(self) -> bool:
        return self._tally.has_history

    def trailing_count(
        self, mode: str, window_days: int = MAX_TRAILING_DAYS
    ) -> Optional[int]:
        """Trailing use-count over the last ``min(window_days,
        days-so-far)`` days; ``None`` only when there is no history at
        all yet."""
        return self._tally.count(mode, window_days)

    def usual_work_mode(
        self, priority_order: Sequence[str] = DEFAULT_MODE_PRIORITY
    ) -> Optional[str]:
        """Modal focal-usage key over the trailing 30 days, tie-broken
        by ``priority_order``; ``None`` with no history or no focal
        usage in the window."""
        return self._tally.usual_focal_key(priority_order, MAX_TRAILING_DAYS)

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-clean snapshot for persona-card persistence.  (New in
        this port — the predecessor kept state in-process only.)"""
        return {
            "config": self.config.to_dict(),
            "cells": {key: cell.to_dict() for key, cell in self._cells.items()},
            "touched_today": sorted(self._touched_today),
            "tally": self._tally.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HabitMemory":
        memory = cls(config=SubstrateConfig.from_dict(data["config"]))
        for key, cell_data in data["cells"].items():
            memory._cells[key] = SubstrateCell.from_dict(cell_data)
        memory._touched_today = set(data["touched_today"])
        memory._tally = TrailingTally.from_dict(data["tally"])
        return memory


# ---------------------------------------------------------------------
# Domain API: per-rule habit-strength counter
# ---------------------------------------------------------------------

class HabitCounter:
    """Per-rule habit-strength counter for a persona card.

    A persona card holds ``{rule_id: HabitCounter}``.  The fast brain
    calls :meth:`record_day` once per lived day for each rule; the slow
    brain reads :attr:`strength` / :meth:`is_strong` when deciding
    whether a rule may be rewritten, and calls :meth:`reset` when it
    rewrites the rule (a rewritten rule is a new habit — it starts with
    no accumulated strength).

    Two published quantities:

    * :attr:`strength` — the habit-strength counter per the project
      brief: +1 for each day the rule is followed, -1 (floored at 0)
      for each day it is not.  This uncapped net "days followed" score
      is what drives E6 hysteresis: after the world reverts, a habit of
      strength S needs ``S - threshold + 1`` consecutive not-followed
      days before :meth:`is_strong` flips — so a 60-day habit resists
      roughly 20x longer than a 3-day habit.
    * :attr:`days_followed_in_window` — days followed within the
      trailing 30-day window, using EXACTLY the ported trailing-counter
      machinery (capped deque, oldest dropped, quiet days recorded):
      each day is a tally of ``{"followed": 0 or 1}``.

    Serialization via :meth:`to_dict` / :meth:`from_dict` embeds in the
    persona card's JSON.
    """

    def __init__(self, window_days: int = MAX_TRAILING_DAYS) -> None:
        self.window_days = int(window_days)
        self._window: Deque[int] = deque()
        self._strength = 0
        self._total_days_followed = 0
        self._days_observed = 0

    # -- recording -------------------------------------------------------

    def record_day(self, followed: bool) -> None:
        """Fold one lived day: followed -> strength +1; not followed ->
        strength -1, floored at 0.  The trailing window records the day
        either way (ported deque semantics: append, drop oldest past
        the cap)."""
        bit = 1 if followed else 0
        self._window.append(bit)
        while len(self._window) > self.window_days:
            self._window.popleft()
        self._days_observed += 1
        if followed:
            self._strength += 1
            self._total_days_followed += 1
        else:
            self._strength = max(0, self._strength - 1)

    # -- reading -----------------------------------------------------------

    @property
    def strength(self) -> int:
        """Net days-followed habit strength (the brief's per-rule
        counter; drives E6 resistance)."""
        return self._strength

    @property
    def days_followed_in_window(self) -> int:
        """Days followed within the trailing window (<= window_days)."""
        return sum(self._window)

    @property
    def total_days_followed(self) -> int:
        """Lifetime count of followed days (never decremented)."""
        return self._total_days_followed

    @property
    def days_observed(self) -> int:
        """Lifetime count of recorded days (followed or not)."""
        return self._days_observed

    def is_strong(self, threshold: int) -> bool:
        """True while ``strength >= threshold`` — a strong habit
        resists reverting when the world reverts (E6)."""
        return self._strength >= threshold

    def days_to_weaken(self, threshold: int) -> int:
        """Consecutive not-followed days until :meth:`is_strong` flips
        False: ``max(0, strength - threshold + 1)``.  This is the
        habit's revert-resistance horizon — the E6 hysteresis quantity.
        ``threshold`` must be >= 1 (at 0 a habit would never weaken).
        """
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if not self.is_strong(threshold):
            return 0
        return self._strength - threshold + 1

    # -- rule-rewrite reset ------------------------------------------------

    def reset(self) -> None:
        """Rule rewritten by the slow brain: the new rule starts with
        no habit at all — strength, window, and lifetime counts all
        cleared."""
        self._window.clear()
        self._strength = 0
        self._total_days_followed = 0
        self._days_observed = 0

    # -- serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_days": self.window_days,
            "window": list(self._window),
            "strength": self._strength,
            "total_days_followed": self._total_days_followed,
            "days_observed": self._days_observed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HabitCounter":
        counter = cls(window_days=int(data["window_days"]))
        counter._window = deque(int(bit) for bit in data["window"])
        counter._strength = int(data["strength"])
        counter._total_days_followed = int(data["total_days_followed"])
        counter._days_observed = int(data["days_observed"])
        return counter
