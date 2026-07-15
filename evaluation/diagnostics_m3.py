"""M3 diagnostics — REPORTED ONLY, never sealed (M3 design D7).

WHY this file exists: the M2 gate record (docs/M2_GATE_RECORD.md, "Findings
and open items") surfaced two soft-audit subpopulations inside the deployed
card set that are numerically faithful (they pass every hard gate, including
the per-card fidelity gate) but structurally suspicious for a milestone that
starts measuring day-to-day dynamics:

* **duplicate-pattern cards** ("FIT-CHECK gaming", ~6.8% of accepted LLM
  cards): a card with 2+ patterns whose trip content is IDENTICAL at every
  pattern, just spread across different integer weights. Numerically this
  reproduces the person's mean trips/day and mode mix fine; structurally it
  zeroes the person's simulated day-to-day variance (every drawn day is the
  same day).
* **tiling / near-enumeration cards** (~8.7% of multi-day persons, a SOFT
  check beyond the hard ``replay_smell`` gate): a multi-day person's card
  patterns exactly tile their observed day sequences — an expected side
  effect of fidelity pressure, not necessarily gaming, but still a person
  whose simulated repertoire is a literal replay of the diary rather than a
  compression of it.

M3 design D7 requires these be RECOMPUTED against whatever card set is
actually deployed (never trusted from a stale manifest count), reported
alongside within-person variance (sim vs observed), E1/E2 scores split by
card provenance (LLM vs fallback), and a sensitivity ("does it distort M3
dynamics") table that recomputes with each flagged subpopulation excluded.
None of this seals anything or gates a build — no pass/fail bar appears
anywhere in this module; every function/output below is a REPORTED number
feeding the M3 gate record, exactly as D7 specifies. If a distortion shows,
the fix is a DATED PROPOSAL to the owner, never a silent regeneration.

Reuse discipline: this module calls only PUBLIC functions of
``evaluation.e1`` / ``evaluation.e2`` (both owned by another agent today,
read-only here) plus ``agents.card_executor``, ``grounding.seeding``, and
``grounding.card_validation``. It never imports ``evaluation.truth`` (the
quarantined answer key, pre-registration §2, "the wall").

Provenance of the two soft-audit definitions ported here: the M2 generation
harness's post-gate audit scripts that produced
``runs/m2_cards_r2/manifest.json``'s ``audits`` block
(``llm_cards_with_duplicate_pattern_sequences`` /
``replay.n_near_enumeration`` / ``near_enumeration_by_source``) were
scratch-only driver scripts, never committed to this repo (they lived
outside the tracked tree and are gone) — the logic is PORTED here faithfully
from their exact algorithm, verified byte-for-byte against the manifest's own
recorded counts (see the reproduction tests in
``tests/test_diagnostics_m3.py``): ``flag_duplicate_pattern_cards`` matches
the manifest's 577/8,496 = 6.8% figure exactly; ``flag_tiling_cards`` matches
the manifest's 260/2,975 = 8.7% figure exactly when run against the ROUND-2
card set the manifest was computed from. Recomputed against the currently
deployed round-2b card set (``data/cards/cards_m2_masked_r2b.jsonl`` — the
fallback builder was revised between rounds 2 and 2b, M2_GATE_RECORD.md), the
near-enumeration count has DRIFTED (see the reproduction test's comment and
this module's report to the M3 gate record) — exactly the kind of drift D7
asks this module to catch and surface, not silently absorb.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from agents.card_executor import execute_days
from evaluation import e1, e2
from grounding import seeding
from grounding.card_validation import day_signatures

# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _card_source(card: Mapping) -> Optional[str]:
    """"llm" / "fallback" / None (missing or unrecognized provenance) —
    never miscounts an unrecognized value into either bucket (mirrors
    evaluation/run_e1.py's ``_fallback_share`` convention: two independent
    checks, not an exhaustive partition assumption)."""
    prov = card.get("provenance") or {}
    src = prov.get("card_source")
    return src if src in ("llm", "fallback") else None


def _pattern_signature(pattern: Mapping) -> tuple:
    """Content signature of one pattern: its trips only (purpose, mode,
    depart_band per trip, in order) — id and weight are deliberately
    excluded, since "duplicate pattern" means identical CONTENT at
    different weight."""
    return tuple(
        (t.get("purpose"), t.get("mode"), t.get("depart_band"))
        for t in pattern.get("trips", [])
    )


def _card_active_pattern_signatures(card: Mapping) -> set:
    """The set of a card's UNIQUE non-empty (active) pattern signatures —
    quiet/no-trip patterns are excluded, exactly as the M2 replay-smell gate
    excludes them (grounding.card_validation.replay_smell): a mandatory
    no-trip pattern is never evidence of enumeration."""
    out = set()
    for p in card.get("patterns", []):
        sig = _pattern_signature(p)
        if sig:
            out.add(sig)
    return out


def _to_native(obj):
    """Coerce numpy scalars / NaN to JSON-native types (mirrors
    evaluation/run_e1.py's ``_to_native``, ported here rather than imported
    since run_e1.py's helper is not part of e1's public scoring surface)."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v  # NaN -> null
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and obj != obj:
        return None  # NaN -> null
    return obj


# ---------------------------------------------------------------------------
# 1. duplicate-pattern cards (M2 "FIT-CHECK gaming" audit, ported)
# ---------------------------------------------------------------------------

_DUP_DEFINITION = (
    "a card with 2+ patterns whose trip-content signatures "
    "((purpose, mode, depart_band) per trip; id/weight ignored) are ALL "
    "identical -- the same day written twice (or more) at different "
    "integer weights instead of compressing genuine day-to-day variety. "
    "A single-pattern card has nothing to duplicate and never qualifies. "
    "Ported from the M2 generation harness's post-gate audit (manifest "
    "key llm_cards_with_duplicate_pattern_sequences.all_patterns_identical, "
    "577/8,496 = 6.8% of accepted LLM cards; M2_GATE_RECORD.md "
    "'FIT-CHECK gaming')."
)


def flag_duplicate_pattern_cards(cards: Sequence[Mapping]) -> dict:
    """LLM/fallback cards whose patterns are all content-identical.

    Reported, never sealed (D7): split by ``provenance.card_source``, each
    side carrying its sorted persona-id list, flagged count, population
    total, and share. See :data:`_DUP_DEFINITION` for the exact rule and its
    M2 provenance; ``tests/test_diagnostics_m3.py`` reproduces the 577/8,496
    figure against the deployed card set.
    """
    flagged: Dict[str, List[str]] = {"llm": [], "fallback": []}
    totals: Dict[str, int] = {"llm": 0, "fallback": 0}
    for card in cards:
        src = _card_source(card)
        if src is None:
            continue
        totals[src] += 1
        patterns = card.get("patterns", [])
        if len(patterns) < 2:
            continue
        sigs = {_pattern_signature(p) for p in patterns}
        if len(sigs) == 1:
            flagged[src].append(card["persona_id"])

    out: dict = {"diagnostic_only": True, "definition": _DUP_DEFINITION}
    for src in ("llm", "fallback"):
        ids = sorted(flagged[src])
        n_total = totals[src]
        out[src] = {
            "persona_ids": ids,
            "n_flagged": len(ids),
            "n_total": n_total,
            "share": (len(ids) / n_total) if n_total else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# 2. tiling / near-enumeration cards (M2 soft replay audit, ported)
# ---------------------------------------------------------------------------

_TILING_DEFINITION = (
    "among multi-day persons (>1 observed weekday day), the share whose "
    "card's set of UNIQUE active (non-quiet) pattern trip-sequences equals "
    "the set of unique observed active-day sequences -- a broader/softer "
    "check than the hard replay_smell gate (which additionally requires "
    "equal COUNTS, not just equal sets, and never fires on cards whose "
    "pattern count differs from the observed active-day count). Ported "
    "from the M2 generation harness's post-gate near-enumeration audit "
    "(manifest key replay.n_near_enumeration / near_enumeration_by_source, "
    "260/2,975 = 8.7% of multi-day persons on the round-2 card set)."
)


def flag_tiling_cards(
    cards: Sequence[Mapping],
    observed_day_sequences: Mapping[str, Sequence[Sequence]],
) -> dict:
    """Near-enumeration soft audit: multi-day persons whose card patterns
    tile their observed day sequences.

    ``observed_day_sequences`` is persona_id -> one entry per observed
    weekday day (each a sequence of (purpose, mode, band) triples, empty for
    a no-trip day) -- exactly :func:`grounding.card_validation.day_signatures`
    output, keyed by persona_id (see :func:`observed_day_sequences_of` for
    the canonical way to build this from a seeding dataset, the same route
    the M2 evidence cache used). See :data:`_TILING_DEFINITION` for the
    exact rule; ``tests/test_diagnostics_m3.py`` reproduces the manifest's
    260/2,975 figure against the round-2 card set and reports (as a
    documented finding, not a bug) how the round-2b fallback-builder
    revision shifted the fallback-side count on the currently deployed set.
    """
    cards_by_id = {c["persona_id"]: c for c in cards}
    multi_ids = sorted(
        pid
        for pid, seqs in observed_day_sequences.items()
        if len(seqs) > 1 and pid in cards_by_id
    )

    flagged: Dict[str, List[str]] = {"llm": [], "fallback": []}
    for pid in multi_ids:
        obs_active = {
            tuple(tuple(t) for t in day) for day in observed_day_sequences[pid] if day
        }
        if not obs_active:
            continue
        card = cards_by_id[pid]
        src = _card_source(card)
        if src is None:
            continue
        if _card_active_pattern_signatures(card) == obs_active:
            flagged[src].append(pid)

    n_flagged_total = len(flagged["llm"]) + len(flagged["fallback"])
    out: dict = {
        "diagnostic_only": True,
        "definition": _TILING_DEFINITION,
        "n_multi_day_persons": len(multi_ids),
        "n_flagged_total": n_flagged_total,
        "share": (n_flagged_total / len(multi_ids)) if multi_ids else float("nan"),
    }
    for src in ("llm", "fallback"):
        ids = sorted(flagged[src])
        out[src] = {"persona_ids": ids, "n_flagged": len(ids)}
    return out


# ---------------------------------------------------------------------------
# building the two "observed" inputs from a seeding dataset (convenience;
# write_diagnostics uses these by default, tests build small dicts by hand)
# ---------------------------------------------------------------------------


def observed_day_sequences_of(
    dataset, persona_of_person: Mapping[str, str]
) -> Dict[str, List[tuple]]:
    """persona_id -> :func:`grounding.card_validation.day_signatures` output,
    built from the SAME committed-adapter tables the M2 evidence cache used
    (``grounding.seeding.enriched_trips`` + ``dataset.person_days``) -- the
    "psrc adapter" route :func:`flag_tiling_cards`'s docstring points at.
    """
    trips = seeding.enriched_trips(dataset)
    trips = trips.copy()
    trips["person_id"] = trips["person_id"].astype(str)
    trips_by_person = {pid: grp for pid, grp in trips.groupby("person_id")}

    person_days = dataset.person_days.copy()
    person_days["person_id"] = person_days["person_id"].astype(str)
    days_by_person = {pid: grp for pid, grp in person_days.groupby("person_id")}

    empty_days = dataset.person_days.iloc[0:0]
    empty_trips = trips.iloc[0:0]

    out: Dict[str, List[tuple]] = {}
    for person_id, persona_id in persona_of_person.items():
        pdays = days_by_person.get(person_id, empty_days)
        ptrips = trips_by_person.get(person_id, empty_trips)
        out[persona_id] = day_signatures(pdays, ptrips)
    return out


def observed_person_day_stats(
    dataset, persona_of_person: Mapping[str, str]
) -> Dict[str, List[Tuple[float, int]]]:
    """persona_id -> [(day_weight, n_collapsed), ...] over that person's
    observed weekday person-days -- the OBSERVED-side input
    :func:`within_person_variance` compares the simulated side against.
    Reads the same ``dataset.person_days`` columns
    (:func:`evaluation.e1.day_slots_by_persona`) does, just carrying the
    trip count instead of the day index.
    """
    pd_tab = dataset.person_days
    out: Dict[str, list] = {}
    for pid, w, n in zip(
        pd_tab.person_id.astype(str), pd_tab.w_day, pd_tab.n_collapsed
    ):
        persona = persona_of_person.get(str(pid))
        if persona is None:
            continue
        out.setdefault(persona, []).append((float(w), int(n)))
    return out


# ---------------------------------------------------------------------------
# 3. within-person day-to-day variance (sim vs observed), REPORTED
# ---------------------------------------------------------------------------

_WITHIN_DEFINITION = (
    "per multi-slot persona (>=2 observed weekday slots), the weighted "
    "variance (evaluation.e2.weighted_var) of realized trips/day across "
    "their scoring-window realized days (weighted by each day's own slot "
    "weight) vs the same weighted variance of the person's OWN observed "
    "day-to-day trip counts; the population ratio is the ratio of the two "
    "POOLED means (mean simulated variance / mean observed variance) over "
    "the personas with a defined, positive observed variance -- never a "
    "mean-of-per-person-ratios, which would over-weight thin persons."
)


def within_person_variance(
    realized_days_by_persona: Mapping[str, Sequence],
    day_slots: Mapping[str, Sequence[Tuple[int, float]]],
    observed_person_days: Mapping[str, Sequence[Tuple[float, int]]],
) -> dict:
    """Simulated vs observed within-person day-to-day variance, REPORTED.

    ``realized_days_by_persona`` is persona_id -> list of
    ``agents.card_executor.RealizedDay`` (produced elsewhere -- the M3
    baseline loop's scoring-window days, or a static ``execute_days`` call;
    this function is pure over it, D7). ``day_slots`` is persona_id -> list
    of ``(day_index, day_weight)`` (:func:`evaluation.e1.day_slots_by_persona`
    shape) -- it both selects the "multi-slot" population (>=2 entries) and
    restricts ``realized_days_by_persona`` to the named ``day_index`` values
    (so extra warm-up/non-scoring days present in the realized list are
    never counted). ``observed_person_days`` is
    :func:`observed_person_day_stats` output.
    """
    persona_ids = sorted(pid for pid, slots in day_slots.items() if len(slots) >= 2)

    per_persona: Dict[str, dict] = {}
    qualifying: List[Tuple[float, float]] = []
    n_insufficient_sim_days = 0
    n_insufficient_obs_days = 0
    n_zero_or_undefined_obs_variance = 0

    for pid in persona_ids:
        slot_day_indices = {d for d, _ in day_slots[pid]}
        days = [
            d for d in realized_days_by_persona.get(pid, []) if d.day_index in slot_day_indices
        ]
        if len(days) < 2:
            n_insufficient_sim_days += 1
            continue
        sim_x = [float(len(d.trips)) for d in days]
        sim_w = [float(d.day_weight) for d in days]
        sim_var = e2.weighted_var(sim_x, sim_w)

        obs_rows = observed_person_days.get(pid, [])
        if len(obs_rows) < 2:
            n_insufficient_obs_days += 1
            continue
        obs_x = [float(n) for _, n in obs_rows]
        obs_w = [float(w) for w, _ in obs_rows]
        obs_var = e2.weighted_var(obs_x, obs_w)

        if not (obs_var == obs_var) or obs_var <= 0:  # NaN or zero/undefined
            n_zero_or_undefined_obs_variance += 1
            per_persona[pid] = {
                "sim_variance": sim_var,
                "obs_variance": obs_var,
                "ratio": float("nan"),
            }
            continue

        ratio = sim_var / obs_var
        per_persona[pid] = {
            "sim_variance": sim_var,
            "obs_variance": obs_var,
            "ratio": ratio,
        }
        qualifying.append((sim_var, obs_var))

    if qualifying:
        sim_mean = sum(q[0] for q in qualifying) / len(qualifying)
        obs_mean = sum(q[1] for q in qualifying) / len(qualifying)
        population_ratio = (sim_mean / obs_mean) if obs_mean > 0 else float("nan")
    else:
        sim_mean = obs_mean = population_ratio = float("nan")

    return {
        "diagnostic_only": True,
        "definition": _WITHIN_DEFINITION,
        "n_multi_slot_personas": len(persona_ids),
        "n_insufficient_sim_days": n_insufficient_sim_days,
        "n_insufficient_obs_days": n_insufficient_obs_days,
        "n_zero_or_undefined_obs_variance": n_zero_or_undefined_obs_variance,
        "n_in_population_ratio": len(qualifying),
        "sim_mean_variance": sim_mean,
        "obs_mean_variance": obs_mean,
        "population_ratio_sim_over_obs": population_ratio,
        "per_persona": per_persona,
    }


# ---------------------------------------------------------------------------
# shared E1/E2 subset-scoring plumbing (provenance_split_scores +
# sensitivity_split both need "score this card subset against truth
# restricted to the same persons" -- factored once here)
# ---------------------------------------------------------------------------


@dataclass
class _E1Context:
    persona_index: object
    persona_of_person: Dict[str, str]  # person_id -> persona_id
    person_of_persona: Dict[str, str]  # persona_id -> person_id (1:1 inverse)
    cell_of_household: Dict[str, Optional[str]]
    persona_cell: Dict[str, Optional[str]]
    hh_of_persona: Dict[str, str]
    personas_of_hh: Dict[str, frozenset]
    day_slots: Dict[str, List[Tuple[int, float]]]


def _build_e1_context(dataset) -> _E1Context:
    persona_index = seeding.persona_index(dataset)
    persona_of_person = e1.persona_of_person_map(persona_index)
    person_of_persona = {pa: p for p, pa in persona_of_person.items()}
    cell_of_household, _drops = e1.repinned_cell_of_household(dataset)
    persona_cell = e1.persona_cell_map(persona_index, cell_of_household)
    hh_of_persona = {
        str(pa): str(hh)
        for pa, hh in zip(persona_index["persona_id"], persona_index["household_id"])
    }
    hh_sets: Dict[str, set] = {}
    for pa, hh in hh_of_persona.items():
        hh_sets.setdefault(hh, set()).add(pa)
    personas_of_hh = {hh: frozenset(pas) for hh, pas in hh_sets.items()}
    day_slots = e1.day_slots_by_persona(dataset, persona_of_person)
    return _E1Context(
        persona_index, persona_of_person, person_of_persona, cell_of_household,
        persona_cell, hh_of_persona, personas_of_hh, day_slots,
    )


class _PersonFilteredView:
    """Duck-typed dataset view restricting the three E1 distribution
    families to a fixed PERSON-id subset.

    Composes ONLY the public ``PSRCDataset.trips_per_day_distribution`` /
    ``mode_share_distribution`` / ``departure_band_distribution`` methods
    (grounding/adapters/psrc.py's public surface, through their additive
    ``person_ids`` kwarg -- default ``None`` there is regression-tested
    byte-identical to the frozen household-only path). Passed to
    :func:`evaluation.e1.truth_distributions` so its POOLED call -- which
    takes no id argument of its own, only the per-cell calls do -- is
    restricted too: this view IS "filter person_days/persona map before
    truth_distributions" (M3 design D7), without editing evaluation/e1.py.

    WHY person-level, not household-level (review finding, 2026-07-15):
    card provenance is assigned per PERSON while households mix provenances
    -- on the deployed r2b population roughly a quarter of households are
    mixed-provenance and the majority of fallback personas share a household
    with an LLM-card persona -- and the fidelity gate concentrated
    heavy-trip persons into fallback, so a household-level truth restriction
    would systematically dilute the fallback split's truth side with its
    lighter LLM household-mates: exactly the comparison this diagnostic
    exists to make clean. Person-level restriction makes truth and sim cover
    IDENTICAL persona sets, with full coverage (no dropping of mixed
    households); the per-split ``n_mixed_household_personas`` count reports
    how many personas the household-level approach would have contaminated.
    """

    def __init__(self, base_dataset, person_ids: Iterable[str]):
        self._base = base_dataset
        self._person_ids = frozenset(str(p) for p in person_ids)

    def trips_per_day_distribution(self, household_ids=None):
        return self._base.trips_per_day_distribution(
            household_ids, person_ids=self._person_ids
        )

    def mode_share_distribution(self, household_ids=None):
        return self._base.mode_share_distribution(
            household_ids, person_ids=self._person_ids
        )

    def departure_band_distribution(self, household_ids=None):
        return self._base.departure_band_distribution(
            household_ids, person_ids=self._person_ids
        )


def _restricted_truth(dataset, person_ids: Iterable[str], cell_of_household):
    view = _PersonFilteredView(dataset, person_ids)
    return e1.truth_distributions(view, cell_of_household=cell_of_household)


def _score_e1_subset(
    cards_subset: Sequence[Mapping],
    dataset,
    ctx: _E1Context,
    n_runs: int,
    namespace_prefix: str,
    producer: Optional[Callable[[str], Mapping[str, Sequence]]] = None,
) -> dict:
    """E1 pooled TVD + worst cell for one card subset, truth restricted to
    EXACTLY the subset personas' persons (person-level, never their whole
    households -- see :class:`_PersonFilteredView` on why household-level
    truth is biased for provenance splits). ``producer`` (when given) is an
    :func:`evaluation.e1.ensemble_arm`-shaped callable (namespace ->
    persona_id -> list[RealizedDay] for one run, e.g. the M3 loop) filtered
    here to this subset's persona ids and passed straight through to
    ``evaluation.e1.simulate_arm``'s own ``producer`` seam (M3 D6: the
    producer kwarg's default reproduces the static
    ``agents.card_executor.execute_days`` path byte-identically).

    ``n_mixed_household_personas`` in the output is the transparency count
    the review asked for: subset personas living in a household that also
    contains at least one persona OUTSIDE the subset -- each one a persona
    whose truth side the old household-level restriction would have
    contaminated with household-mates' diaries.
    """
    ids = sorted(c["persona_id"] for c in cards_subset)
    if not ids:
        return {
            "n_cards": 0, "persona_ids": [],
            "truth_restriction": "person_level", "n_truth_persons": 0,
            "n_mixed_household_personas": 0,
            "pooled_tvd": float("nan"), "cell_tvds": {}, "worst_cell": None,
            "worst_cell_tvd": float("nan"),
        }
    id_set = set(ids)
    person_ids = {
        ctx.person_of_persona[pid] for pid in ids if pid in ctx.person_of_persona
    }
    persona_cell = {pid: ctx.persona_cell.get(pid) for pid in ids}
    day_slots = {pid: ctx.day_slots.get(pid, []) for pid in ids}

    n_mixed = sum(
        1
        for pid in ids
        if any(
            other not in id_set
            for other in ctx.personas_of_hh.get(ctx.hh_of_persona.get(pid), ())
        )
    )

    truth = _restricted_truth(dataset, person_ids, ctx.cell_of_household)

    wrapped_producer = None
    if producer is not None:
        def wrapped_producer(namespace, _ids=id_set, _base=producer):
            days = _base(namespace)
            return {pid: d for pid, d in days.items() if pid in _ids}

    arm = e1.simulate_arm(
        cards_subset, dataset, persona_cell, day_slots,
        n_runs=n_runs, namespace_prefix=namespace_prefix, producer=wrapped_producer,
    )

    pooled = e1.pooled_tvd(arm.pooled, truth.pooled)
    cells = e1.cell_tvds(arm.cells, truth.cells)
    finite_cells = {c: v for c, v in cells.items() if v == v}  # drop NaN cells
    worst_cell = max(finite_cells, key=finite_cells.get) if finite_cells else None
    return {
        "n_cards": len(cards_subset),
        "persona_ids": ids,
        "truth_restriction": "person_level",
        "n_truth_persons": len(person_ids),
        "n_mixed_household_personas": n_mixed,
        "pooled_tvd": pooled,
        "cell_tvds": cells,
        "worst_cell": worst_cell,
        "worst_cell_tvd": cells.get(worst_cell) if worst_cell is not None else float("nan"),
    }


def _score_e2_subset(
    cards_subset: Sequence[Mapping],
    dataset,
    ctx: _E1Context,
    n_runs: int,
    seed: int,
    producer: Optional[Callable[[str], Mapping[str, Sequence]]] = None,
) -> dict:
    """E2 spread ratios (+ error correlation) for one card subset, truth
    (diary) restricted to the SAME persons. Scores through the UNMODIFIED
    ``evaluation.e2.score_e2`` (its own ``producer`` kwarg, M3 D6, already
    filters the sim side to the given ``cards_subset``'s persona ids via its
    internal ``card_ids`` join); this helper additionally recomputes the
    matched diary-side persona-id set independently (public functions only)
    so it can be reported and asserted on (``score_e2`` itself only returns
    counts, not the ids) -- purely for reporting, never for scoring.
    """
    ids = sorted(c["persona_id"] for c in cards_subset)
    if not ids:
        return {
            "n_cards": 0, "persona_ids": [], "truth_persona_ids": [],
            "n_cards_without_diary_match": 0, "spread_ratios": {}, "error_correlation": {},
        }
    id_set = set(ids)

    wrapped_producer = None
    if producer is not None:
        def wrapped_producer(namespace, _ids=id_set, _base=producer):
            days = _base(namespace)
            return {pid: d for pid, d in days.items() if pid in _ids}

    result = e2.score_e2(cards_subset, dataset, n_runs=n_runs, seed=seed, producer=wrapped_producer)

    diary = e2.person_stats_from_diary(dataset.person_days, dataset.weekday_trips)
    diary.index = diary.index.map(ctx.persona_of_person)
    diary = diary[diary.index.notna()]
    diary_in_force = diary.loc[diary.index.intersection(list(id_set))]

    return {
        "n_cards": len(cards_subset),
        "persona_ids": ids,
        "truth_persona_ids": sorted(diary_in_force.index),
        "n_cards_without_diary_match": result["n_cards_without_diary_match"],
        "spread_ratios": result["spread_ratios"],
        "error_correlation": result["error_correlation"],
    }


# ---------------------------------------------------------------------------
# 4. provenance split (LLM-only vs fallback-only), diagnostic only
# ---------------------------------------------------------------------------


def provenance_split_scores(
    cards: Sequence[Mapping],
    dataset,
    n_runs: int,
    seed: int,
    producer: Optional[Callable[[str], Mapping[str, Sequence]]] = None,
) -> dict:
    """E1 pooled TVD + worst cell, and E2 spread ratios, scored SEPARATELY
    on the LLM-only and fallback-only subpopulations of ``cards``, with the
    truth side (both E1's distribution families and E2's diary) filtered
    PERSON-LEVEL to exactly the SAME persons in each split (D7: "both
    simulated and observed arms must cover identical persona sets"; see
    :class:`_PersonFilteredView` on why a household-level restriction would
    bias the fallback split). ``producer`` (optional) is an
    :func:`evaluation.e1.ensemble_arm`-shaped callable the M3 runner injects
    for loop-realized days; default reproduces the static
    ``execute_days``-based path via the unmodified public
    ``evaluation.e1`` / ``evaluation.e2`` functions. Always
    ``"diagnostic_only": True`` -- this never feeds a sealed verdict.
    """
    ctx = _build_e1_context(dataset)
    llm_cards = [c for c in cards if _card_source(c) == "llm"]
    fallback_cards = [c for c in cards if _card_source(c) == "fallback"]

    out: dict = {
        "diagnostic_only": True,
        "n_runs": int(n_runs),
        "seed": int(seed),
        "splits": {},
    }
    for label, subset in (("llm", llm_cards), ("fallback", fallback_cards)):
        out["splits"][label] = {
            "e1": _score_e1_subset(
                subset, dataset, ctx, n_runs, f"m3diag_prov_{label}_", producer=producer,
            ),
            "e2": _score_e2_subset(subset, dataset, ctx, n_runs, seed, producer=producer),
        }
    return out


# ---------------------------------------------------------------------------
# 5. sensitivity split: exclude each flagged subpopulation, recompute
# ---------------------------------------------------------------------------


def sensitivity_split(
    cards: Sequence[Mapping],
    dataset,
    realized_days_by_persona: Mapping[str, Sequence],
    day_slots: Mapping[str, Sequence[Tuple[int, float]]],
    observed_person_days: Mapping[str, Sequence[Tuple[float, int]]],
    duplicate_pattern_ids: Iterable[str],
    tiling_ids: Iterable[str],
    n_runs: int = 20,
    seed: int = 0,
    producer: Optional[Callable[[str], Mapping[str, Sequence]]] = None,
) -> dict:
    """The "does it distort M3 dynamics" table (D7): E1 pooled TVD + the
    within-person variance population ratio, recomputed EXCLUDING each
    flagged subpopulation in turn (duplicate-pattern excluded; tiling
    excluded; both excluded) alongside an unexcluded baseline row, so an
    owner can compare side by side. No bar, no verdict: a material shift is
    a finding to propose a dated fix for, never to act on silently.
    """
    ctx = _build_e1_context(dataset)
    dup_ids = set(duplicate_pattern_ids)
    til_ids = set(tiling_ids)
    cuts: Dict[str, set] = {
        "baseline": set(),
        "exclude_duplicate_pattern": dup_ids,
        "exclude_tiling": til_ids,
        "exclude_both": dup_ids | til_ids,
    }

    out: dict = {"diagnostic_only": True, "cuts": {}}
    for cut_name, excluded in cuts.items():
        remaining = [c for c in cards if c["persona_id"] not in excluded]
        ids_remaining = {c["persona_id"] for c in remaining}

        e1_res = _score_e1_subset(
            remaining, dataset, ctx, n_runs, f"m3diag_sens_{cut_name}_", producer=producer,
        )
        wv = within_person_variance(
            {pid: d for pid, d in realized_days_by_persona.items() if pid in ids_remaining},
            {pid: d for pid, d in day_slots.items() if pid in ids_remaining},
            {pid: d for pid, d in observed_person_days.items() if pid in ids_remaining},
        )
        out["cuts"][cut_name] = {
            "n_excluded": len(excluded),
            "n_remaining_cards": len(remaining),
            "e1_pooled_tvd": e1_res["pooled_tvd"],
            "e1_worst_cell": e1_res["worst_cell"],
            "e1_worst_cell_tvd": e1_res["worst_cell_tvd"],
            "within_person_variance_population_ratio": wv["population_ratio_sim_over_obs"],
            "within_person_variance_n_multi_slot_personas": wv["n_multi_slot_personas"],
        }
    return out


# ---------------------------------------------------------------------------
# 6. write_diagnostics: compute everything above, write runs/<out>/... json
# ---------------------------------------------------------------------------


def write_diagnostics(
    out_dir,
    cards: Sequence[Mapping],
    dataset,
    n_runs: int = 20,
    seed: int = 0,
    producer: Optional[Callable[[str], Mapping[str, Sequence]]] = None,
    realized_days_by_persona: Optional[Mapping[str, Sequence]] = None,
    observed_day_sequences: Optional[Mapping[str, Sequence[Sequence]]] = None,
) -> dict:
    """Compute every D7 diagnostic against ``cards``/``dataset`` and write
    ``<out_dir>/diagnostics_m3.json`` (deterministic: sorted id lists,
    ``sort_keys=True`` on write). Returns the same dict that was written.

    ``realized_days_by_persona`` / ``observed_day_sequences`` default to a
    fresh static computation (``execute_days`` / :func:`observed_day_sequences_of`)
    when omitted -- the M3 runner passes its own loop-realized scoring-window
    days and cached evidence instead when calling this end-to-end.
    """
    persona_index = seeding.persona_index(dataset)
    persona_of_person = e1.persona_of_person_map(persona_index)
    day_slots = e1.day_slots_by_persona(dataset, persona_of_person)
    observed_person_days = observed_person_day_stats(dataset, persona_of_person)

    if observed_day_sequences is None:
        observed_day_sequences = observed_day_sequences_of(dataset, persona_of_person)
    if realized_days_by_persona is None:
        realized_days_by_persona = execute_days(
            cards, day_slots, "m3diag_within0", update_habits=False,
        )

    dup = flag_duplicate_pattern_cards(cards)
    tiling = flag_tiling_cards(cards, observed_day_sequences)

    dup_ids = set(dup["llm"]["persona_ids"]) | set(dup["fallback"]["persona_ids"])
    til_ids = set(tiling["llm"]["persona_ids"]) | set(tiling["fallback"]["persona_ids"])
    cards_by_id = {c["persona_id"]: c for c in cards}

    within: Dict[str, dict] = {
        "overall": within_person_variance(realized_days_by_persona, day_slots, observed_person_days),
    }
    for src in ("llm", "fallback"):
        src_ids = {pid for pid, c in cards_by_id.items() if _card_source(c) == src}
        cuts = {
            "duplicate_pattern": dup_ids & src_ids,
            "tiling": til_ids & src_ids,
            "clean": src_ids - dup_ids - til_ids,
        }
        for cut_name, ids in cuts.items():
            within[f"{cut_name}_{src}"] = within_person_variance(
                {pid: d for pid, d in realized_days_by_persona.items() if pid in ids},
                {pid: d for pid, d in day_slots.items() if pid in ids},
                {pid: d for pid, d in observed_person_days.items() if pid in ids},
            )

    prov = provenance_split_scores(cards, dataset, n_runs, seed, producer=producer)
    sens = sensitivity_split(
        cards, dataset, realized_days_by_persona, day_slots, observed_person_days,
        dup_ids, til_ids, n_runs=n_runs, seed=seed, producer=producer,
    )

    result = {
        "diagnostic_only": True,
        "duplicate_pattern": dup,
        "tiling": tiling,
        "within_person_variance": within,
        "provenance_split_scores": prov,
        "sensitivity_split": sens,
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "diagnostics_m3.json").write_text(
        json.dumps(_to_native(result), indent=2, sort_keys=True) + "\n"
    )
    return result
