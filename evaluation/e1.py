"""E1 grounding-fidelity scoring harness (pre-registration §3 E1, A2.1; M2 D5).

This module is the ONE scoring code path both E1 arms flow through — the
method arm (persona cards executed by ``agents.card_executor``) and the MNL
falsification arm (``evaluation.mnl_arm``). Both arms emit the same
``RealizedDay`` / ``RealizedTrip`` structures; this module turns realized days
into the three frozen distribution families, scores them against the sealed
full-sample truth, and runs the paired household-atomic falsification bootstrap.

Sealed protocol implemented here (01_PREREGISTRATION.md §7 A2.1, M2_ARCH_SPEC D5):

* Truth side (FIXED): the full-sample weighted three-family distributions, all
  6,319 households / 11,940 persons, produced by the committed adapter builders
  (``grounding.adapters.psrc``) so they bit-match the sealed measurement record.
* Sim side: per run k (namespace ``f"{prefix}run{k}"``), each persona simulates
  ONE day per observed weighted weekday slot; the simulated day inherits the
  slot's ``day_weight`` and every simulated trip carries that day_weight as its
  trip weight (D5, mirroring the P2 null construction). trips/day is day-weighted
  over ALL simulated days (zero-trip weekdays included); mode and band families
  are trip-weighted (= day-weighted per trip).
* Ensemble: N>=20 runs; the 20 runs' NORMALIZED distributions are averaged
  BEFORE the TVD (A2.1 ensemble-mean-before-TVD).
* Pooled TVD headline = max over the three families of TVD(arm, truth), bar 0.10.
* Ten protected cells (guard merge per A2.1): the twelve income x car x residence
  cells with the three ``*|car0|remainder`` cells merged into one ``car0|remainder``
  guard cell. Residence banding is RE-PINNED from the committed tract->zone map
  (``grounding.zone_map.ring_of_household``, docs/M2_RING_REPIN.md), superseding
  the adapter's built-in proxy for all scoring. PNA-income households are excluded
  from cells but included in the pooled statistic (as the adapter/measurement
  record does); households whose home tract does not map to a ring are
  dropped-from-cells with a logged count. Per-cell bar 0.20 (2x pooled).
* Paired falsification bootstrap: B=500 household-atomic replicates; per replicate
  ONE multinomial resample of the 6,319 households defines the truth side (all
  families rebuilt from per-household family-mass matrices, exactly the
  investigation's vectorised machinery); BOTH arms' FIXED ensemble-mean
  distributions are scored against that same resampled truth; Delta_b =
  pooled TVD_method - pooled TVD_MNL. Pass iff the empirical 95% CI [p2.5, p97.5]
  of Delta lies entirely below +epsilon, epsilon = 0.00655.

Seeds: a single base seed with documented derived offsets (mirroring the fold
investigation's derivation, docs/internal/M0_E1_FOLD_INVESTIGATION.md §0):
the paired bootstrap uses ``base_seed + SEED_OFFSET_PAIRED`` (44). The N>=20
ensemble streams are keyed by the hash-based CRN namespace strings ``run0..runN``
(no numeric RNG seed), so they are reproducible without a seed.

This module imports only from ``grounding`` / ``agents`` / ``world`` (read-only);
it never imports from ``evaluation.truth`` (the import-quarantined truth series).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from grounding import zone_map
from grounding.adapters import psrc
from grounding.taxonomy import MODES, SEGMENT_CELLS, segment_cell

# ---------------------------------------------------------------------------
# Frozen protocol constants (A2.1)
# ---------------------------------------------------------------------------

FAMILIES: Tuple[str, ...] = ("trips_per_day", "mode_shares", "time_bands")
TRIPS_PER_DAY_BINS: Tuple[str, ...] = psrc.TRIPS_PER_DAY_BINS  # {0..7, 8+}, 9 bins
TIME_BANDS: Tuple[str, ...] = psrc.TIME_BANDS  # five frozen bands
_FAMILY_WIDTH = {"trips_per_day": 9, "mode_shares": 5, "time_bands": 5}

#: Sealed E1 bars (A2.1).
POOLED_BAR = 0.10
CELL_BAR = 0.20  # 2 x pooled
EPSILON = 0.00655  # paired match tolerance (A2.1, §7 of the fold investigation)

#: Guard-merge (A2.1): the three car-free / remainder cells collapse into one.
_MERGED_SRC: Tuple[str, ...] = (
    "low|car0|remainder",
    "mid|car0|remainder",
    "high|car0|remainder",
)
MERGED_CELL = "car0|remainder"
#: The ten protected cells, in a frozen order.
CELLS10: Tuple[str, ...] = tuple(c for c in SEGMENT_CELLS if c not in _MERGED_SRC) + (
    MERGED_CELL,
)

#: Seed derivation offset for the paired bootstrap (mirrors the fold
#: investigation's "paired = base + 44").
SEED_OFFSET_PAIRED = 44
DEFAULT_BASE_SEED = 20260717


# ---------------------------------------------------------------------------
# Small numeric kernels (identical semantics to the sealed reference machinery)
# ---------------------------------------------------------------------------

def tvd(p, q) -> float:
    """Total variation distance = 0.5 * sum |p - q|."""
    return float(0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def normalize(v) -> np.ndarray:
    """Normalize a nonnegative mass vector; an all-zero vector -> NaNs."""
    v = np.asarray(v, float)
    s = v.sum()
    if s <= 0:
        return np.full_like(v, np.nan)
    return v / s


def _trips_bin_index(n: int) -> int:
    """Index into the {0..7, 8+} bins for a per-day trip count."""
    return n if n < 8 else 8


# ---------------------------------------------------------------------------
# Re-pinned protected-cell membership (docs/M2_RING_REPIN.md; A2.6)
# ---------------------------------------------------------------------------

def load_home_tracts(
    dataset, data_dir=psrc.DEFAULT_DATA_DIR
) -> Dict[str, object]:
    """household_id -> raw home_tract_2020, from the raw households CSV.

    The promoted adapter's cache omits the home-tract column, so the re-pin
    reads it directly from the (gitignored) raw survey households file,
    restricted to the pooled waves with positive weight — exactly the household
    set the cache carries.
    """
    path = Path(data_dir) / psrc._HOUSEHOLDS_CSV
    raw = pd.read_csv(
        path,
        usecols=["household_id", "survey_year", "home_tract_2020", "hh_weight"],
        dtype={"household_id": str},
        low_memory=False,
    )
    raw = raw[raw.survey_year.isin(psrc.WAVES) & (raw.hh_weight > 0)]
    return dict(zip(raw.household_id.astype(str), raw.home_tract_2020))


def repinned_cell_of_household(
    dataset,
    ring_of_household: Optional[Callable[[object], Optional[str]]] = None,
    home_tracts: Optional[Mapping[str, object]] = None,
) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """Map each household to its guard-merged protected cell, or ``None``.

    Residence banding is injected from the committed tract->zone map
    (``grounding.zone_map.ring_of_household``, the A2.6 re-pin); the adapter's
    built-in proxy is superseded for all scoring. Returns ``(cell_of, drops)``
    where ``cell_of[household_id]`` is one of :data:`CELLS10` or ``None`` and
    ``drops`` logs how many households fell out of the cells and why:

      * ``pna_income`` — income "prefer not to answer": excluded from cells,
        still included in the pooled statistic.
      * ``unmapped_ring`` — home tract missing / not in the committed map:
        dropped-from-cells with a logged count (M2_RING_REPIN §5).
    """
    if ring_of_household is None:
        ring_of_household = zone_map.ring_of_household
    if home_tracts is None:
        home_tracts = load_home_tracts(dataset)

    cell_of: Dict[str, Optional[str]] = {}
    drops = {"pna_income": 0, "unmapped_ring": 0}
    for r in dataset.households.itertuples(index=False):
        hid = str(r.household_id)
        inc = r.income_class
        if inc is None or (isinstance(inc, float) and np.isnan(inc)):
            cell_of[hid] = None
            drops["pna_income"] += 1
            continue
        ring = ring_of_household(home_tracts.get(hid))
        if ring is None:
            cell_of[hid] = None
            drops["unmapped_ring"] += 1
            continue
        cell = segment_cell(int(inc), int(r.household_cars), ring)
        cell_of[hid] = MERGED_CELL if cell in _MERGED_SRC else cell
    return cell_of, drops


def households_by_cell(
    cell_of_household: Mapping[str, Optional[str]]
) -> Dict[str, frozenset]:
    """Invert the household->cell map into cell -> frozenset(household_id)."""
    out: Dict[str, set] = {c: set() for c in CELLS10}
    for hid, cell in cell_of_household.items():
        if cell is not None:
            out.setdefault(cell, set()).add(hid)
    return {c: frozenset(out.get(c, set())) for c in CELLS10}


# ---------------------------------------------------------------------------
# persona <-> person <-> household plumbing (harness-side only)
# ---------------------------------------------------------------------------

def persona_of_person_map(persona_index: pd.DataFrame) -> Dict[str, str]:
    """person_id -> persona_id (from ``grounding.seeding.persona_index``)."""
    return {
        str(p): str(pa)
        for p, pa in zip(persona_index["person_id"], persona_index["persona_id"])
    }


def persona_cell_map(
    persona_index: pd.DataFrame, cell_of_household: Mapping[str, Optional[str]]
) -> Dict[str, Optional[str]]:
    """persona_id -> guard-merged cell (or None) via the household re-pin."""
    return {
        str(pa): cell_of_household.get(str(hh))
        for pa, hh in zip(persona_index["persona_id"], persona_index["household_id"])
    }


def day_slots_by_persona(
    dataset, persona_of_person: Mapping[str, str]
) -> Dict[str, List[Tuple[int, float]]]:
    """persona_id -> [(day_index, day_weight), ...], one slot per observed
    weighted weekday person-day (zero-trip weekdays included). ``day_index`` is
    the observed ``daynum`` so CRN keys are stable and a person's multiple days
    draw independently; slots are sorted by daynum for determinism.
    """
    pd_tab = dataset.person_days
    out: Dict[str, List[Tuple[int, float]]] = {}
    for pid, dn, w in zip(
        pd_tab.person_id.astype(str), pd_tab.daynum, pd_tab.w_day
    ):
        persona = persona_of_person.get(str(pid))
        if persona is None:
            continue
        out.setdefault(persona, []).append((int(dn), float(w)))
    for slots in out.values():
        slots.sort()
    return out


# ---------------------------------------------------------------------------
# Truth distributions (FIXED; bit-match the sealed measurement record)
# ---------------------------------------------------------------------------

@dataclass
class TruthDistributions:
    """Full-sample truth: pooled three-family distributions plus per-cell."""

    pooled: Dict[str, np.ndarray]
    cells: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)


def truth_distributions(
    dataset, cell_of_household: Optional[Mapping[str, Optional[str]]] = None
) -> TruthDistributions:
    """The frozen full-sample three-family distributions via the committed
    adapter builders. When ``cell_of_household`` is given, also the per-cell
    full-sample distributions over the re-pinned ten protected cells.
    """
    pooled = {
        "trips_per_day": dataset.trips_per_day_distribution(),
        "mode_shares": dataset.mode_share_distribution(),
        "time_bands": dataset.departure_band_distribution(),
    }
    cells: Dict[str, Dict[str, np.ndarray]] = {}
    if cell_of_household is not None:
        by_cell = households_by_cell(cell_of_household)
        for cell in CELLS10:
            ids = by_cell[cell]
            cells[cell] = {
                "trips_per_day": dataset.trips_per_day_distribution(ids),
                "mode_shares": dataset.mode_share_distribution(ids),
                "time_bands": dataset.departure_band_distribution(ids),
            }
    return TruthDistributions(pooled=pooled, cells=cells)


# ---------------------------------------------------------------------------
# Per-household family-mass matrices (the paired bootstrap's truth machinery)
# ---------------------------------------------------------------------------

def household_family_matrices(
    dataset, households: Optional[Sequence[str]] = None
):
    """Per-household weighted category-mass matrices for the three families.

    Mirrors ``docs/internal/m0_bars/m0_common.household_family_matrices``: the
    distribution of any household multiset S = ``normalize(colsum(M[S]))``. A
    paired-bootstrap truth resample is ``normalize(counts @ M)`` for
    multinomial household counts.
    """
    pd_tab = dataset.person_days
    trips = dataset.weekday_trips
    if households is None:
        households = sorted(set(pd_tab.household_id.astype(str)))
    hidx = {h: i for i, h in enumerate(households)}
    H = len(households)

    Ma = np.zeros((H, 9))
    bi = {b: j for j, b in enumerate(TRIPS_PER_DAY_BINS)}
    for h, n, w in zip(pd_tab.household_id.astype(str), pd_tab.n_collapsed, pd_tab.w_day):
        Ma[hidx[h], bi[psrc.trips_bin(int(n))]] += w

    mi = {m: j for j, m in enumerate(MODES)}
    ci = {b: j for j, b in enumerate(TIME_BANDS)}
    Mb = np.zeros((H, len(MODES)))
    Mc = np.zeros((H, len(TIME_BANDS)))
    for h, m, w in zip(trips.household_id.astype(str), trips["mode"], trips.w_trip):
        Mb[hidx[h], mi[m]] += w
    for h, b, w in zip(trips.household_id.astype(str), trips.band, trips.w_trip):
        Mc[hidx[h], ci[b]] += w

    return list(households), hidx, Ma, Mb, Mc


# ---------------------------------------------------------------------------
# Realized-days -> distribution families (the ONE scoring path for both arms)
# ---------------------------------------------------------------------------

def _accumulate_masses(
    days_by_persona: Mapping[str, Sequence],
    persona_cell: Mapping[str, Optional[str]],
):
    """One run's pooled + per-cell family MASS vectors from realized days.

    trips/day: day_weight into the trip-count bin, EVERY day (zero-trip days
    land in bin 0). mode/band: each trip contributes its day's day_weight
    (D5: simulated trips carry the day_weight as trip weight).
    """
    mi = {m: j for j, m in enumerate(MODES)}
    ci = {b: j for j, b in enumerate(TIME_BANDS)}
    pooled = [np.zeros(9), np.zeros(len(MODES)), np.zeros(len(TIME_BANDS))]
    cell_mass = {
        c: [np.zeros(9), np.zeros(len(MODES)), np.zeros(len(TIME_BANDS))]
        for c in CELLS10
    }
    for pid, days in days_by_persona.items():
        cell = persona_cell.get(pid)
        cm = cell_mass[cell] if cell in cell_mass else None
        for day in days:
            w = day.day_weight
            b = _trips_bin_index(len(day.trips))
            pooled[0][b] += w
            if cm is not None:
                cm[0][b] += w
            for t in day.trips:
                mj = mi[t.mode]
                cj = ci[t.depart_band]
                pooled[1][mj] += w
                pooled[2][cj] += w
                if cm is not None:
                    cm[1][mj] += w
                    cm[2][cj] += w
    return pooled, cell_mass


@dataclass
class ArmDistributions:
    """An arm's FIXED ensemble-mean distributions (pooled + per cell)."""

    pooled: Dict[str, np.ndarray]
    cells: Dict[str, Dict[str, np.ndarray]]
    n_runs: int


def ensemble_arm(
    producer: Callable[[str], Mapping[str, Sequence]],
    persona_cell: Mapping[str, Optional[str]],
    n_runs: int,
    namespace_prefix: str,
) -> ArmDistributions:
    """Average the N runs' NORMALIZED distributions before TVD (A2.1).

    ``producer(namespace)`` returns persona_id -> [RealizedDay] for one run;
    it is called with namespace ``f"{namespace_prefix}run{k}"`` for k in 0..N-1
    so the CRN streams are independent across runs but paired across arms.
    """
    pooled_acc = [np.zeros(9), np.zeros(len(MODES)), np.zeros(len(TIME_BANDS))]
    cell_acc = {
        c: [np.zeros(9), np.zeros(len(MODES)), np.zeros(len(TIME_BANDS))]
        for c in CELLS10
    }
    cell_valid = {c: np.zeros(3) for c in CELLS10}
    for k in range(n_runs):
        ns = f"{namespace_prefix}run{k}"
        days = producer(ns)
        pooled, cell_mass = _accumulate_masses(days, persona_cell)
        for j in range(3):
            pooled_acc[j] += normalize(pooled[j])
        for c in CELLS10:
            for j in range(3):
                d = normalize(cell_mass[c][j])
                if not np.isnan(d).any():
                    cell_acc[c][j] += d
                    cell_valid[c][j] += 1

    pooled_dist = {FAMILIES[j]: pooled_acc[j] / n_runs for j in range(3)}
    cells_dist: Dict[str, Dict[str, np.ndarray]] = {}
    for c in CELLS10:
        cells_dist[c] = {}
        for j in range(3):
            cnt = cell_valid[c][j]
            width = _FAMILY_WIDTH[FAMILIES[j]]
            cells_dist[c][FAMILIES[j]] = (
                cell_acc[c][j] / cnt if cnt > 0 else np.full(width, np.nan)
            )
    return ArmDistributions(pooled=pooled_dist, cells=cells_dist, n_runs=n_runs)


def simulate_arm(
    cards: Sequence[dict],
    dataset,
    persona_cell: Mapping[str, Optional[str]],
    day_slots: Mapping[str, Sequence[Tuple[int, float]]],
    n_runs: int = 20,
    namespace_prefix: str = "method_",
) -> ArmDistributions:
    """The METHOD arm: execute persona cards into realized days and score them
    through :func:`ensemble_arm` (the ONE scoring path). Nothing is fitted
    across personas — cards are per-person compressions of own records (A2.1).
    """
    from agents.card_executor import execute_days  # local: keep import graph light

    def producer(namespace: str):
        return execute_days(cards, day_slots, namespace, update_habits=False)

    return ensemble_arm(producer, persona_cell, n_runs, namespace_prefix)


# ---------------------------------------------------------------------------
# Scoring: pooled + per-cell TVDs
# ---------------------------------------------------------------------------

def pooled_tvd(arm_pooled: Mapping[str, np.ndarray], truth_pooled: Mapping[str, np.ndarray]) -> float:
    """Pooled headline TVD = max over the three families (A2.1 convention)."""
    return max(tvd(arm_pooled[f], truth_pooled[f]) for f in FAMILIES)


def per_family_tvd(
    arm_pooled: Mapping[str, np.ndarray], truth_pooled: Mapping[str, np.ndarray]
) -> Dict[str, float]:
    return {f: tvd(arm_pooled[f], truth_pooled[f]) for f in FAMILIES}


def cell_tvds(
    arm_cells: Mapping[str, Mapping[str, np.ndarray]],
    truth_cells: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, float]:
    """Per-cell headline TVD (max over families) for each of the ten cells."""
    out: Dict[str, float] = {}
    for c in CELLS10:
        out[c] = max(tvd(arm_cells[c][f], truth_cells[c][f]) for f in FAMILIES)
    return out


# ---------------------------------------------------------------------------
# Paired falsification bootstrap (A2.1 / M2 D5)
# ---------------------------------------------------------------------------

def paired_bootstrap(
    method_pooled: Mapping[str, np.ndarray],
    mnl_pooled: Mapping[str, np.ndarray],
    dataset=None,
    B: int = 500,
    base_seed: int = DEFAULT_BASE_SEED,
    matrices=None,
) -> dict:
    """Household-atomic paired falsification bootstrap (A2.1).

    Both arms' FIXED ensemble-mean pooled distributions are scored against the
    SAME resampled truth in every replicate; Delta_b = pooled TVD_method -
    pooled TVD_MNL. Pass iff the empirical 95% CI of Delta lies entirely below
    +epsilon (epsilon = 0.00655). ``matrices`` may be a pre-built
    ``(households, hidx, Ma, Mb, Mc)`` tuple (from
    :func:`household_family_matrices`); otherwise it is built from ``dataset``.
    Seed = ``base_seed + SEED_OFFSET_PAIRED`` (documented, not a magic number).
    """
    if matrices is None:
        matrices = household_family_matrices(dataset)
    households, _hidx, Ma, Mb, Mc = matrices
    Ms = (Ma, Mb, Mc)
    H = len(households)
    probs = np.full(H, 1.0 / H)
    seed = base_seed + SEED_OFFSET_PAIRED
    rng = np.random.default_rng(seed)

    method_arr = [np.asarray(method_pooled[f], float) for f in FAMILIES]
    mnl_arr = [np.asarray(mnl_pooled[f], float) for f in FAMILIES]

    deltas = np.empty(B)
    m_abs = np.empty(B)
    n_abs = np.empty(B)
    for b in range(B):
        ch = rng.multinomial(H, probs).astype(float)
        truth_b = [normalize(ch @ Ms[j]) for j in range(3)]
        mp = max(tvd(method_arr[j], truth_b[j]) for j in range(3))
        npd = max(tvd(mnl_arr[j], truth_b[j]) for j in range(3))
        m_abs[b] = mp
        n_abs[b] = npd
        deltas[b] = mp - npd

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "B": int(B),
        "seed": int(seed),
        "seed_derivation": f"base_seed({base_seed}) + SEED_OFFSET_PAIRED({SEED_OFFSET_PAIRED})",
        "epsilon": EPSILON,
        "delta_mean": float(deltas.mean()),
        "delta_sd": float(deltas.std(ddof=1)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "pass": bool(hi < EPSILON),
        "method_bootstrap_tvd_mean": float(m_abs.mean()),
        "mnl_bootstrap_tvd_mean": float(n_abs.mean()),
    }
