"""E2 — variance preservation scoring, per sealed A2.2 (01_PREREGISTRATION.md §7)
and M2 architecture spec D8.

E2(i) — spread ratios, ONE-ARM PRESERVATION READING (sealed A2.2(i)):
    ratio(dim) = simulated between-agent variance / between-individual
    variance of the SEEDING records in force, per dimension, pass band
    [0.8, 1.2]. Dimensions, exactly: person-mean weekday trips/day, person
    car-share, person ride-share. Both sides are day-weighted person means,
    mirroring the real-side construction of the M0 measurement script
    (docs/internal/m0_bars/run_e2.py — read-only reference, never imported):

    * mean trips/day per person = sum(w_day * n_trips_day) / sum(w_day) over
      that person's weekday person-days (zero-trip weekdays included);
    * car-share / ride-share per person = trip-weight share of that mode over
      that person's weekday trips. On the simulated side each realized trip
      carries its day's slot weight as trip weight (spec D5), so the share is
      sum(day_weight over car trips) / sum(day_weight over all trips).
    * persons/personas with no weighted days are excluded from all dimensions;
      persons/personas with no (collapsed/realized) trips are excluded from
      the share dimensions — both WITH logged counts, as in the M0 script.
    * between-individual variance is person_weight-weighted on both sides
      (w_person; personas inherit their seeding person's weight 1:1).

    "The seeding records in force" = the diary records of exactly the persons
    whose personas are being executed (record-level 1:1 seeding; the caller
    passes the dataset those cards were seeded from). Ensemble handling: the
    ratio is computed per run and the headline ratio is the mean over the
    N runs (the denominator is constant, so this equals the ratio of the
    mean simulated variance); per-run ratios are reported alongside.

E2(ii) — independence of agent-level prediction errors (sealed A2.2(ii)):
    mean pairwise cross-persona correlation of prediction errors <= 0.20,
    measured on REALIZED discrete choices (the CRN draw layer), never on
    expected-value loads.

    ERROR-VECTOR CONSTRUCTION (documented per the E2 build mandate):
    for ensemble run k (CRN namespace ``run{k}``) and persona i, the error is

        e_ik(dim) = s_ik(dim) - d_i(dim)

    where s_ik is persona i's day-weighted person statistic over the realized
    days of run k (same three dimensions as E2(i)) and d_i is the person's own
    diary expectation (the same real-side person statistic used in E2(i)'s
    denominator). d_i is constant in k, so the correlation across runs is
    numerically invariant to it; it is kept so e_ik is a genuine prediction
    error and the construction stays parallel to M0.

    GROUPING MAP (M0 -> M2, documented per the mandate): the M0 floor
    (0.0105, sealed in A2.2) is a calendar-date ICC — survey-world residuals
    of multi-day persons grouped by the calendar DATE they share, measuring
    the common daily shock. The simulated analogue has no calendar dates;
    what personas share is the RUN: within namespace ``run{k}`` every persona
    draws from the same run-level CRN stream and (from M3 on) lives through
    the same world feedback. So the run index k replaces the calendar date
    as the grouping along which common shocks are measured: the statistic is
    the mean pairwise CROSS-PERSONA correlation of e_i. across the run axis.

    COMPUTATION — VARIANCE-OF-SUMS IDENTITY (never an O(N^2) pair loop):
    standardize each persona's error series across runs, z_i = (e_i - mean_k
    e_ik) / sd_k(e_i); then with S_k = sum_i z_ik,

        Var_k(S) = sum_i Var(z_i) + sum_{i != j} Cov(z_i, z_j)
                 = N + N (N - 1) * rho_bar
        rho_bar  = (Var_k(S) / N - 1) / (N - 1)

    which equals the unweighted mean pairwise Pearson correlation EXACTLY
    (sample-moment identity; any consistent ddof), verified against a brute
    force O(N^2) computation in tests. Personas whose error series has zero
    variance across runs (fully deterministic repertoires) carry no
    correlation information and are dropped WITH a logged count, as are
    personas whose statistic is undefined in any run (incomplete series).
    Reported per dimension plus the max; bar: max rho_bar <= 0.20.

This module is pure scoring machinery: frames/realized structures in, plain
dicts out. The CLI wiring lives in evaluation/run_e2.py.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from agents.card_executor import RealizedDay, execute_days
# _persona_id_map is the exact masked-reindex rule the seeding step used to
# mint the persona ids on the cards; reusing it (read-only) is what guarantees
# the diary<->card join here can never drift from the seeding join.
from grounding.seeding import _persona_id_map

DIMENSIONS = ("mean_trips_per_day", "car_share", "ride_share")
SPREAD_BAND = (0.8, 1.2)
CORRELATION_BAR = 0.20
DEFAULT_RUNS = 20


def namespaces_for(n_runs: int, seed: int = 0) -> List[str]:
    """The ensemble CRN namespaces: ``run{seed}`` .. ``run{seed+n_runs-1}``.

    ``seed`` shifts the starting run index so a re-run can request a fresh,
    independent ensemble while staying inside the D3 ``run{k}`` namespace
    doctrine (arms pair by sharing these exact namespaces)."""
    return [f"run{seed + k}" for k in range(n_runs)]


def weighted_var(x, w) -> float:
    """Weighted population variance sum(w (x-m)^2)/sum(w) — the exact
    real-side estimator of the M0 measurement script. NaN on zero mass."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = w.sum()
    if sw <= 0:
        return float("nan")
    m = (w * x).sum() / sw
    return float((w * (x - m) ** 2).sum() / sw)


# ---------------------------------------------------------------------------
# per-person statistics, both sides (day-weighted person means)
# ---------------------------------------------------------------------------

def person_stats_from_diary(person_days: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """Real-side per-person statistics, mirroring the M0 construction.

    ``person_days`` needs person_id / w_day / n_collapsed (zero-trip weekdays
    included); ``trips`` needs person_id / mode / w_trip. A ``w_person``
    column on ``person_days`` is carried through (defaults to 1.0 when the
    frame carries none, e.g. synthetic fixtures). Persons with no collapsed
    trips get NaN shares (excluded from the share dimensions downstream)."""
    pdays = person_days.copy()
    pdays["person_id"] = pdays["person_id"].astype(str)
    if "w_person" not in pdays.columns:
        pdays["w_person"] = 1.0

    g = pdays.groupby("person_id")
    mean_trips = g.apply(
        lambda d: float((d.w_day * d.n_collapsed).sum() / d.w_day.sum()),
        include_groups=False,
    )
    w_person = g["w_person"].first()

    tr = trips.copy()
    tr["person_id"] = tr["person_id"].astype(str)
    tg = tr.groupby("person_id")
    car_share = tg.apply(
        lambda d: float(d.loc[d["mode"] == "car", "w_trip"].sum() / d.w_trip.sum()),
        include_groups=False,
    )
    ride_share = tg.apply(
        lambda d: float(d.loc[d["mode"] == "ride", "w_trip"].sum() / d.w_trip.sum()),
        include_groups=False,
    )

    out = pd.DataFrame(
        {
            "mean_trips_per_day": mean_trips,
            "car_share": car_share.reindex(mean_trips.index),
            "ride_share": ride_share.reindex(mean_trips.index),
            "w_person": w_person,
        }
    )
    out.index.name = "person_id"
    return out


def person_stats_from_realized(realized: Mapping[str, Sequence[RealizedDay]]) -> pd.DataFrame:
    """Simulated-side per-persona statistics from one run's realized days.

    Day-weighted person means, mirroring the diary side: each realized trip
    carries its day's slot weight (spec D5). Personas with zero total day
    weight are omitted; personas with no realized trips get NaN shares."""
    rows: Dict[str, dict] = {}
    for persona_id, days in realized.items():
        w_sum = sum(d.day_weight for d in days)
        if w_sum <= 0:
            continue
        trip_mass = 0.0
        car_mass = 0.0
        ride_mass = 0.0
        weighted_trip_count = 0.0
        for d in days:
            n = len(d.trips)
            weighted_trip_count += d.day_weight * n
            trip_mass += d.day_weight * n
            for t in d.trips:
                if t.mode == "car":
                    car_mass += d.day_weight
                elif t.mode == "ride":
                    ride_mass += d.day_weight
        rows[persona_id] = {
            "mean_trips_per_day": weighted_trip_count / w_sum,
            "car_share": (car_mass / trip_mass) if trip_mass > 0 else float("nan"),
            "ride_share": (ride_mass / trip_mass) if trip_mass > 0 else float("nan"),
        }
    out = pd.DataFrame.from_dict(rows, orient="index", dtype=float)
    if out.empty:
        out = pd.DataFrame(columns=list(DIMENSIONS), dtype=float)
    out.index.name = "persona_id"
    return out[list(DIMENSIONS)]


# ---------------------------------------------------------------------------
# E2(i) spread ratios
# ---------------------------------------------------------------------------

def spread_ratios(
    real_stats: pd.DataFrame, sim_stats_by_run: Sequence[pd.DataFrame]
) -> dict:
    """Per-dimension spread ratios (sim between-agent var / real
    between-individual var), one entry per dimension plus the band verdict.

    ``real_stats`` is the diary-side frame (dimensions + ``w_person``)
    indexed by the SAME ids as the simulated frames (the caller reindexes the
    diary frame to persona ids). The simulated variance is weighted by the
    same ``w_person`` (personas inherit their person's weight 1:1)."""
    out: dict = {"band": list(SPREAD_BAND), "dimensions": {}}
    all_pass = True
    for dim in DIMENSIONS:
        real_sub = real_stats[real_stats[dim].notna()]
        real_var = weighted_var(real_sub[dim].to_numpy(), real_sub["w_person"].to_numpy())
        per_run = []
        for sim in sim_stats_by_run:
            sub = sim[sim[dim].notna()]
            w = real_stats["w_person"].reindex(sub.index).fillna(1.0)
            sim_var = weighted_var(sub[dim].to_numpy(), w.to_numpy())
            per_run.append(sim_var / real_var if real_var > 0 else float("nan"))
        ratio = float(np.mean(per_run)) if per_run else float("nan")
        ok = bool(SPREAD_BAND[0] <= ratio <= SPREAD_BAND[1])
        all_pass = all_pass and ok
        out["dimensions"][dim] = {
            "real_var": real_var,
            "per_run_ratios": [float(r) for r in per_run],
            "ratio": ratio,
            "n_real_persons": int(len(real_sub)),
            "pass": ok,
        }
    out["pass"] = all_pass
    return out


# ---------------------------------------------------------------------------
# E2(ii) mean pairwise error correlation (variance-of-sums identity)
# ---------------------------------------------------------------------------

def mean_pairwise_correlation(errors) -> dict:
    """Mean pairwise cross-persona Pearson correlation of error series.

    ``errors`` is a personas x runs matrix (rows = personas, columns = runs).
    Rows containing NaN (incomplete series) and rows with zero variance
    across runs are dropped WITH logged counts; the survivors are
    standardized and the mean pairwise correlation is read off the
    variance-of-sums identity (see module docstring) — never an O(N^2) pair
    loop. Exactly equals the unweighted mean of all pairwise Pearson
    correlations (tested against brute force)."""
    e = np.asarray(errors, dtype=float)
    if e.ndim != 2:
        raise ValueError("errors must be a 2-D personas x runs matrix")
    n_rows, n_runs = e.shape

    complete = ~np.isnan(e).any(axis=1)
    n_incomplete = int(n_rows - complete.sum())
    e = e[complete]

    if n_runs >= 2 and len(e):
        sd = e.std(axis=1, ddof=1)
    else:
        sd = np.zeros(len(e))
    nonzero = sd > 0
    n_zero_variance = int(len(e) - nonzero.sum())
    e = e[nonzero]
    sd = sd[nonzero]

    n = len(e)
    if n < 2 or n_runs < 2:
        rho = float("nan")
    else:
        z = (e - e.mean(axis=1, keepdims=True)) / sd[:, None]
        s = z.sum(axis=0)
        var_s = float(s.var(ddof=1))
        rho = (var_s / n - 1.0) / (n - 1)
    return {
        "rho": float(rho) if rho == rho else float("nan"),
        "n_personas": int(n),
        "n_runs": int(n_runs),
        "n_incomplete_dropped": n_incomplete,
        "n_zero_variance_dropped": n_zero_variance,
    }


def error_correlations(
    diary_stats: pd.DataFrame, sim_stats_by_run: Sequence[pd.DataFrame]
) -> dict:
    """E2(ii): per-dimension mean pairwise error correlations plus the max.

    Builds, per dimension, the personas x runs error matrix
    e_ik = s_ik - d_i (see module docstring) over the personas whose diary
    expectation is defined, and hands it to the variance-of-sums identity.
    Personas with an undefined simulated statistic in any run enter as NaN
    rows and are dropped (logged) inside :func:`mean_pairwise_correlation`."""
    out: dict = {"bar": CORRELATION_BAR, "dimensions": {}}
    max_rho = float("-inf")
    for dim in DIMENSIONS:
        d = diary_stats[dim].dropna()
        ids = d.index
        cols = [sim[dim].reindex(ids).to_numpy(dtype=float) for sim in sim_stats_by_run]
        matrix = np.column_stack(cols) if cols else np.empty((len(ids), 0))
        e = matrix - d.to_numpy(dtype=float)[:, None]
        res = mean_pairwise_correlation(e)
        out["dimensions"][dim] = res
        if res["rho"] == res["rho"]:  # not NaN
            max_rho = max(max_rho, res["rho"])
    out["max_rho"] = max_rho if max_rho > float("-inf") else float("nan")
    out["pass"] = bool(out["max_rho"] == out["max_rho"] and out["max_rho"] <= CORRELATION_BAR)
    return out


# ---------------------------------------------------------------------------
# end-to-end scoring: cards + dataset -> E2 verdict
# ---------------------------------------------------------------------------

def day_slots_of(person_days: pd.DataFrame, persona_of_person: Mapping[str, str]) -> dict:
    """persona_id -> [(day_index, day_weight), ...]: one simulated day per
    observed weighted weekday person-day slot; the simulated day inherits the
    slot's day weight (spec D5). ``day_index`` is the diary daynum."""
    slots: Dict[str, list] = {}
    pdays = person_days.copy()
    pdays["person_id"] = pdays["person_id"].astype(str)
    for person_id, grp in pdays.groupby("person_id"):
        persona_id = persona_of_person.get(person_id)
        if persona_id is None:
            continue
        slots[persona_id] = [
            (int(r.daynum), float(r.w_day)) for r in grp.itertuples(index=False)
        ]
    return slots


def score_e2(
    cards: Sequence[dict],
    dataset,
    n_runs: int = DEFAULT_RUNS,
    seed: int = 0,
    producer: Optional[Callable[[str], Mapping[str, Sequence[RealizedDay]]]] = None,
) -> dict:
    """Score E2(i)+(ii) for a card population against its seeding dataset.

    ``dataset`` is the PSRCDataset (or a synthetic stand-in with the same
    ``persons`` / ``person_days`` / ``weekday_trips`` frames) whose records
    seeded the cards — the "seeding records in force" of sealed A2.2(i).
    Executes the cards for ``n_runs`` ensemble runs under the D3 ``run{k}``
    CRN namespaces (``seed`` shifts the starting index) and returns the full
    E2 verdict dict. Never mutates the cards (habit updating is off:
    scoring is a measurement, not lived time).

    ``producer`` is an OPTIONAL injection seam (M3 D6). When ``None`` (the
    default) each ensemble run is produced by ``card_executor.execute_days`` —
    byte-identical to the pre-M3 behaviour. When supplied,
    ``producer(namespace)`` provides ``persona_id -> [RealizedDay]`` in its
    place (e.g. the M3 loop's scoring window) so a world-coupled dynamic arm is
    scored through the UNCHANGED E2 path."""
    id_map = _persona_id_map(dataset.persons["person_id"].astype(str))
    diary = person_stats_from_diary(dataset.person_days, dataset.weekday_trips)
    diary.index = diary.index.map(id_map)
    diary = diary[diary.index.notna()]

    card_ids = [c["persona_id"] for c in cards]
    unmatched = sorted(set(card_ids) - set(diary.index))
    diary_in_force = diary.loc[diary.index.intersection(card_ids)]

    slots = day_slots_of(dataset.person_days, id_map)
    namespaces = namespaces_for(n_runs, seed)

    sim_by_run = []
    for ns in namespaces:
        if producer is None:
            realized = execute_days(cards, slots, ns, update_habits=False)
        else:
            realized = producer(ns)
        sim_by_run.append(person_stats_from_realized(realized))

    spread = spread_ratios(diary_in_force, sim_by_run)
    correlation = error_correlations(diary_in_force, sim_by_run)

    return {
        "eval": "e2",
        "n_cards": len(cards),
        "n_seeding_persons_in_force": int(len(diary_in_force)),
        "n_cards_without_diary_match": len(unmatched),
        "n_runs": int(n_runs),
        "seed": int(seed),
        "namespaces": namespaces,
        "spread_ratios": spread,
        "error_correlation": correlation,
        "e2_pass": bool(spread["pass"] and correlation["pass"]),
        "protocol": {
            "reading": "one-arm preservation (sealed A2.2(i)); denominator = "
                       "between-individual variance of the seeding records in "
                       "force, numerator = simulated between-agent variance "
                       "from realized trips; day-weighted person means both "
                       "sides, person_weight-weighted variances",
            "error_grouping": "run-index grouping (simulated analogue of the "
                              "M0 calendar-date ICC; see evaluation/e2.py "
                              "docstring for the documented mapping)",
            "realized": "realized discrete choices via the CRN draw layer; "
                        "expected-value loads are never scored here",
        },
    }
