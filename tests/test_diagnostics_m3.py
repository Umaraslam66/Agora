"""evaluation/diagnostics_m3.py — M3 design D7 REPORTED diagnostics.

Covered: the two ported soft-audit flaggers against hand-built cards AND
against the reproduction targets recorded in
``runs/m2_cards_r2/manifest.json`` (duplicate-pattern 577/8,496 = 6.8%;
near-enumeration 260/2,975 = 8.7% on the round-2 set the manifest was itself
computed from -- plus a documented finding on how that count has DRIFTED on
the currently deployed round-2b set, per the round-2b fallback-builder
revision); within-person variance math against hand-computed values;
a provenance-split smoke test on a small synthetic population asserting
persona-set consistency between the truth and simulated sides (D7); a
sensitivity-split smoke test; and a write_diagnostics round-trip.

Also covered (review finding, 2026-07-15 — person-level truth restriction):
the additive ``person_ids`` kwarg on the three PSRC distribution builders is
regression-tested BYTE-IDENTICAL at its ``None`` default against the
pre-change household-only algorithm (that path feeds the sealed E1 truth);
its filtering behaviour is hand-checked; and a dedicated mixed-provenance-
household fixture proves the provenance split's truth side is restricted to
exactly the subset personas' persons (pooled TVD hits 0 for a
diary-replaying card, which household-level truth would contaminate with
household-mates' diaries), with the per-split ``n_mixed_household_personas``
transparency count asserted.

All synthetic fixtures here are MASKED synthetic data (style of
tests/test_card_executor.py's ``make_card`` / tests/test_e1.py's ``_dataset``
/ tests/test_e2.py's ``make_dataset``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.card_executor import RealizedDay, RealizedTrip, execute_days
from evaluation import diagnostics_m3 as dm
from evaluation import e1
from grounding import seeding
from grounding.adapters import psrc

REPO_ROOT = Path(__file__).resolve().parent.parent
CARDS_R2B = REPO_ROOT / "data" / "cards" / "cards_m2_masked_r2b.jsonl"
CARDS_R2 = REPO_ROOT / "data" / "cards" / "cards_m2_masked_r2.jsonl"


def _load_jsonl(path: Path) -> list:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# small card builders (style of tests/test_card_executor.py's make_card)
# ---------------------------------------------------------------------------

def make_card(persona_id, patterns, source="llm", cars=1, can_drive=True) -> dict:
    return {
        "persona_id": persona_id,
        "skeleton": {"household_cars": cars, "can_drive": can_drive},
        "patterns": patterns,
        "rules": [],
        "habit_counters": {},
        "provenance": {"card_source": source},
    }


WORK_TRIPS = [
    {"purpose": "work", "mode": "car", "depart_band": "am_peak"},
    {"purpose": "home", "mode": "car", "depart_band": "pm_peak"},
]
ERRAND_TRIPS = [{"purpose": "shop_daily", "mode": "walk", "depart_band": "midday"}]


# ---------------------------------------------------------------------------
# 1. flag_duplicate_pattern_cards -- hand checks
# ---------------------------------------------------------------------------

def test_flag_duplicate_pattern_cards_hand_check():
    dup_llm = make_card("P00001", [
        {"id": "a", "weight": 3, "trips": list(WORK_TRIPS)},
        {"id": "b", "weight": 7, "trips": list(WORK_TRIPS)},
    ])
    clean_llm = make_card("P00002", [
        {"id": "a", "weight": 5, "trips": list(WORK_TRIPS)},
        {"id": "b", "weight": 5, "trips": list(ERRAND_TRIPS)},
    ])
    single_pattern_llm = make_card("P00003", [{"id": "a", "weight": 1, "trips": list(WORK_TRIPS)}])
    dup_fallback = make_card("P00004", [
        {"id": "a", "weight": 2, "trips": list(ERRAND_TRIPS)},
        {"id": "b", "weight": 8, "trips": list(ERRAND_TRIPS)},
    ], source="fallback")

    result = dm.flag_duplicate_pattern_cards(
        [dup_llm, clean_llm, single_pattern_llm, dup_fallback]
    )

    assert result["diagnostic_only"] is True
    assert result["llm"]["persona_ids"] == ["P00001"]
    assert result["llm"]["n_flagged"] == 1
    assert result["llm"]["n_total"] == 3
    assert result["llm"]["share"] == pytest.approx(1 / 3)
    assert result["fallback"]["persona_ids"] == ["P00004"]
    assert result["fallback"]["n_flagged"] == 1
    assert result["fallback"]["n_total"] == 1


def test_flag_duplicate_pattern_cards_ignores_unrecognized_provenance():
    no_prov = {"persona_id": "P00009", "patterns": [
        {"id": "a", "weight": 1, "trips": list(WORK_TRIPS)},
        {"id": "b", "weight": 1, "trips": list(WORK_TRIPS)},
    ]}
    result = dm.flag_duplicate_pattern_cards([no_prov])
    assert result["llm"]["n_total"] == 0
    assert result["fallback"]["n_total"] == 0
    assert result["llm"]["persona_ids"] == []


# ---------------------------------------------------------------------------
# 2. flag_tiling_cards -- hand checks
# ---------------------------------------------------------------------------

def test_flag_tiling_cards_hand_check():
    day_a = (("work", "car", "am_peak"), ("home", "car", "pm_peak"))
    day_b = (("shop_daily", "walk", "midday"),)

    def _pattern(pid, weight, day):
        return {"id": pid, "weight": weight,
                "trips": [{"purpose": p, "mode": m, "depart_band": b} for p, m, b in day]}

    tiling_llm = make_card("P00001", [_pattern("a", 5, day_a), _pattern("b", 2, day_b)])
    undercompressed_llm = make_card("P00002", [_pattern("a", 10, day_a)])  # misses day_b
    single_day_llm = make_card("P00003", [_pattern("a", 10, day_a)])
    tiling_fallback = make_card(
        "P00004", [_pattern("a", 5, day_a), _pattern("b", 2, day_b)], source="fallback",
    )

    observed = {
        "P00001": [day_a, day_b],
        "P00002": [day_a, day_b],
        "P00003": [day_a],  # single observed day -> excluded from the denominator
        "P00004": [day_a, day_b],
    }

    result = dm.flag_tiling_cards(
        [tiling_llm, undercompressed_llm, single_day_llm, tiling_fallback], observed,
    )

    assert result["diagnostic_only"] is True
    assert result["n_multi_day_persons"] == 3  # P1, P2, P4 (P3 excluded)
    assert result["llm"]["persona_ids"] == ["P00001"]
    assert result["fallback"]["persona_ids"] == ["P00004"]
    assert result["n_flagged_total"] == 2
    assert result["share"] == pytest.approx(2 / 3)


def test_flag_tiling_cards_ignores_persons_without_a_card():
    day_a = (("work", "car", "am_peak"),)
    day_b = (("shop_daily", "walk", "midday"),)
    observed = {"P00999": [day_a, day_b]}  # no matching card at all
    result = dm.flag_tiling_cards([], observed)
    assert result["n_multi_day_persons"] == 0
    assert result["n_flagged_total"] == 0


# ---------------------------------------------------------------------------
# 3. within_person_variance -- hand checks
# ---------------------------------------------------------------------------

def test_within_person_variance_hand_check():
    # sim: trip counts 2 and 4, weights 1 and 1 -> mean 3, var = ((2-3)^2+(4-3)^2)/2 = 1.0
    days = [
        RealizedDay(day_index=0, day_weight=1.0, trips=[
            RealizedTrip("work", "car", "am_peak"), RealizedTrip("home", "car", "pm_peak"),
        ]),
        RealizedDay(day_index=1, day_weight=1.0, trips=[
            RealizedTrip("work", "car", "am_peak") for _ in range(4)
        ]),
    ]
    realized = {"P00001": days}
    day_slots = {"P00001": [(0, 1.0), (1, 1.0)]}
    # obs: trip counts 1 and 3, weights 1 and 1 -> mean 2, var = ((1-2)^2+(3-2)^2)/2 = 1.0
    observed = {"P00001": [(1.0, 1), (1.0, 3)]}

    result = dm.within_person_variance(realized, day_slots, observed)

    assert result["diagnostic_only"] is True
    assert result["n_multi_slot_personas"] == 1
    assert result["per_persona"]["P00001"]["sim_variance"] == pytest.approx(1.0)
    assert result["per_persona"]["P00001"]["obs_variance"] == pytest.approx(1.0)
    assert result["per_persona"]["P00001"]["ratio"] == pytest.approx(1.0)
    assert result["population_ratio_sim_over_obs"] == pytest.approx(1.0)
    assert result["n_in_population_ratio"] == 1


def test_within_person_variance_filters_extra_realized_days_by_day_slots():
    # day_index 0 is NOT a scoring slot (e.g. a warm-up day) -- must be
    # excluded from the sim variance even though it is present in the
    # realized-days list.
    days = [
        RealizedDay(day_index=0, day_weight=1.0, trips=[]),
        RealizedDay(day_index=1, day_weight=1.0, trips=[RealizedTrip("work", "car", "am_peak")]),
        RealizedDay(day_index=2, day_weight=1.0, trips=[
            RealizedTrip("work", "car", "am_peak") for _ in range(3)
        ]),
    ]
    realized = {"P00001": days}
    day_slots = {"P00001": [(1, 1.0), (2, 1.0)]}
    observed = {"P00001": [(1.0, 1), (1.0, 3)]}

    result = dm.within_person_variance(realized, day_slots, observed)
    # sim x restricted to day_index in {1, 2} -> [1, 3] -> var 1.0, matches obs
    assert result["per_persona"]["P00001"]["sim_variance"] == pytest.approx(1.0)
    assert result["population_ratio_sim_over_obs"] == pytest.approx(1.0)


def test_within_person_variance_excludes_single_slot_personas_and_handles_nan():
    day_slots = {"P00002": [(0, 1.0)]}  # only one slot -> not multi-slot
    result = dm.within_person_variance({"P00002": []}, day_slots, {"P00002": [(1.0, 1)]})
    assert result["n_multi_slot_personas"] == 0
    assert result["population_ratio_sim_over_obs"] != result["population_ratio_sim_over_obs"]  # NaN


def test_within_person_variance_zero_observed_variance_is_reported_not_divided():
    days = [
        RealizedDay(day_index=0, day_weight=1.0, trips=[RealizedTrip("work", "car", "am_peak")]),
        RealizedDay(day_index=1, day_weight=1.0, trips=[
            RealizedTrip("work", "car", "am_peak"), RealizedTrip("home", "car", "pm_peak"),
        ]),
    ]
    realized = {"P00001": days}
    day_slots = {"P00001": [(0, 1.0), (1, 1.0)]}
    observed = {"P00001": [(1.0, 2), (1.0, 2)]}  # zero observed variance
    result = dm.within_person_variance(realized, day_slots, observed)
    assert result["per_persona"]["P00001"]["obs_variance"] == pytest.approx(0.0)
    assert result["per_persona"]["P00001"]["ratio"] != result["per_persona"]["P00001"]["ratio"]  # NaN
    assert result["n_zero_or_undefined_obs_variance"] == 1
    assert result["n_in_population_ratio"] == 0


# ---------------------------------------------------------------------------
# reproduction checks against the deployed / manifest-source card sets
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_reproduce_duplicate_pattern_count_on_deployed_r2b_set():
    """runs/m2_cards_r2/manifest.json audits.llm_cards_with_duplicate_pattern_sequences
    .all_patterns_identical = 577 (6.8% of 8,496 LLM cards; M2_GATE_RECORD.md
    "FIT-CHECK gaming"). Fallback cards never trigger this definition by
    construction (grounding.card_validation.fallback_card collapses
    identical-content days into ONE weighted pattern via a Counter, so two
    DISTINCT patterns with identical content never coexist on a fallback
    card) -- verified empirically at 0/3,444 here, not merely assumed.
    """
    cards = _load_jsonl(CARDS_R2B)
    result = dm.flag_duplicate_pattern_cards(cards)
    assert result["llm"]["n_total"] == 8496
    assert result["llm"]["n_flagged"] == 577
    assert result["llm"]["share"] == pytest.approx(577 / 8496, abs=1e-9)
    assert result["fallback"]["n_total"] == 3444
    assert result["fallback"]["n_flagged"] == 0


@pytest.fixture(scope="module")
def _psrc_dataset():
    return psrc.load_or_build()


@pytest.fixture(scope="module")
def _persona_of_person(_psrc_dataset):
    persona_index = seeding.persona_index(_psrc_dataset)
    return e1.persona_of_person_map(persona_index)


@pytest.mark.slow
def test_reproduce_near_enumeration_count_matches_manifest_on_round2_set(
    _psrc_dataset, _persona_of_person,
):
    """flag_tiling_cards, applied to the ROUND-2 card set
    (data/cards/cards_m2_masked_r2.jsonl) that runs/m2_cards_r2/manifest.json
    was itself computed from, reproduces the recorded audit counts exactly:
    260/2,975 = 8.7% (165 llm + 95 fallback; manifest audits.replay /
    .near_enumeration_by_source). This is the fidelity check on the PORTED
    logic itself, independent of any later card-set revision.
    """
    cards = _load_jsonl(CARDS_R2)
    observed = dm.observed_day_sequences_of(_psrc_dataset, _persona_of_person)
    result = dm.flag_tiling_cards(cards, observed)

    assert result["n_multi_day_persons"] == 2975
    assert result["llm"]["n_flagged"] == 165
    assert result["fallback"]["n_flagged"] == 95
    assert result["n_flagged_total"] == 260
    assert result["share"] == pytest.approx(260 / 2975, abs=1e-9)


@pytest.mark.slow
def test_near_enumeration_on_deployed_r2b_set_has_drifted_from_round2(
    _psrc_dataset, _persona_of_person,
):
    """DOCUMENTED FINDING (not a bug): the round-2b fallback-builder revision
    (16-trip cap + closest-trip-count anti-enumeration fold; M2_GATE_RECORD.md
    / memory 12438, "Repaired E2 Variance from 0.767 to 1.000") changed
    FALLBACK card CONTENT for the persona set generation fell back on -- LLM
    cards are byte-identical between round 2 and round 2b. Recomputing the
    SAME near-enumeration audit against the currently DEPLOYED round-2b set
    therefore reproduces the LLM-side count exactly (165, unchanged) but the
    fallback-side count has moved from 95 (round-2, the figure cited in
    docs/M2_GATE_RECORD.md and docs/internal/M3_DESIGN.md D7 as "8.7%") to
    105 on round-2b -- 270/2,975 = 9.08% actual on the set M3 diagnostics
    runs against day to day. Surfaced here per D7's own mandate ("Tiling set
    ... recomputed per card"; this module never trusts a stale manifest
    count) -- flagged to the orchestrator for the M3 gate record rather than
    silently amended or hidden.
    """
    cards = _load_jsonl(CARDS_R2B)
    observed = dm.observed_day_sequences_of(_psrc_dataset, _persona_of_person)
    result = dm.flag_tiling_cards(cards, observed)

    assert result["n_multi_day_persons"] == 2975
    assert result["llm"]["n_flagged"] == 165
    assert result["fallback"]["n_flagged"] == 105
    assert result["n_flagged_total"] == 270
    assert result["share"] == pytest.approx(270 / 2975, abs=1e-9)


# ---------------------------------------------------------------------------
# synthetic population for the provenance-split / sensitivity-split /
# write_diagnostics smoke tests (style of tests/test_e1.py's _dataset +
# tests/test_e2.py's make_dataset/make_cards)
# ---------------------------------------------------------------------------

# persona ids under the masked reindex of the four synthetic person ids
# (sorted person_id -> P00001..): 1101, 1102, 1201, 1301
P1101, P1102, P1201, P1301 = "P00001", "P00002", "P00003", "P00004"


def _synthetic_dataset():
    households = pd.DataFrame([
        {"household_id": "H1", "income_class": 3, "household_cars": 1},
        {"household_id": "H2", "income_class": 2, "household_cars": 0},
        {"household_id": "H3", "income_class": 4, "household_cars": 2},
    ])
    persons = pd.DataFrame([
        {"person_id": "1101", "household_id": "H1"},
        {"person_id": "1102", "household_id": "H1"},
        {"person_id": "1201", "household_id": "H2"},
        {"person_id": "1301", "household_id": "H3"},
    ])
    person_days = pd.DataFrame([
        {"person_id": "1101", "household_id": "H1", "daynum": 1, "w_day": 1.0, "n_collapsed": 2},
        {"person_id": "1101", "household_id": "H1", "daynum": 2, "w_day": 1.0, "n_collapsed": 1},
        {"person_id": "1102", "household_id": "H1", "daynum": 1, "w_day": 2.0, "n_collapsed": 2},
        {"person_id": "1201", "household_id": "H2", "daynum": 1, "w_day": 1.0, "n_collapsed": 1},
        {"person_id": "1301", "household_id": "H3", "daynum": 1, "w_day": 1.0, "n_collapsed": 0},
    ])
    weekday_trips = pd.DataFrame([
        {"person_id": "1101", "household_id": "H1", "daynum": 1, "tripnum": 1,
         "mode": "car", "band": "am_peak", "w_trip": 1.0},
        {"person_id": "1101", "household_id": "H1", "daynum": 1, "tripnum": 2,
         "mode": "car", "band": "pm_peak", "w_trip": 1.0},
        {"person_id": "1101", "household_id": "H1", "daynum": 2, "tripnum": 1,
         "mode": "walk", "band": "midday", "w_trip": 1.0},
        {"person_id": "1102", "household_id": "H1", "daynum": 1, "tripnum": 1,
         "mode": "ride", "band": "midday", "w_trip": 2.0},
        {"person_id": "1102", "household_id": "H1", "daynum": 1, "tripnum": 2,
         "mode": "transit", "band": "evening", "w_trip": 2.0},
        {"person_id": "1201", "household_id": "H2", "daynum": 1, "tripnum": 1,
         "mode": "car", "band": "am_peak", "w_trip": 1.0},
    ])
    return psrc.PSRCDataset(
        households=households, persons=persons, person_days=person_days,
        weekday_trips=weekday_trips, build_log={},
    )


def _synthetic_cards():
    commute = {"id": "commute", "weight": 6, "trips": [
        {"purpose": "work", "mode": "car", "depart_band": "am_peak"},
        {"purpose": "home", "mode": "car", "depart_band": "pm_peak"},
    ]}
    errand = {"id": "errand", "weight": 3, "trips": [
        {"purpose": "shop_daily", "mode": "walk", "depart_band": "midday"},
    ]}
    transit_day = {"id": "transit_day", "weight": 5, "trips": [
        {"purpose": "work", "mode": "transit", "depart_band": "am_peak"},
        {"purpose": "home", "mode": "transit", "depart_band": "evening"},
    ]}
    quiet = {"id": "quiet", "weight": 2, "trips": []}
    return [
        make_card(P1101, [dict(commute), dict(errand), dict(quiet)], source="llm", cars=1),
        make_card(P1102, [dict(transit_day), dict(quiet)], source="fallback", cars=1),
        make_card(P1201, [dict(commute), dict(quiet)], source="llm", cars=0),
        make_card(P1301, [dict(errand), dict(quiet)], source="fallback", cars=2),
    ]


# ---------------------------------------------------------------------------
# 4. provenance_split_scores -- smoke: persona-set consistency (D7)
# ---------------------------------------------------------------------------

def test_provenance_split_scores_smoke_persona_set_consistency():
    ds = _synthetic_dataset()
    cards = _synthetic_cards()

    result = dm.provenance_split_scores(cards, ds, n_runs=2, seed=0)

    assert result["diagnostic_only"] is True
    assert set(result["splits"]) == {"llm", "fallback"}
    for label in ("llm", "fallback"):
        split = result["splits"][label]
        e1_ids = set(split["e1"]["persona_ids"])
        e2_ids = set(split["e2"]["persona_ids"])
        e2_truth_ids = set(split["e2"]["truth_persona_ids"])
        assert e1_ids == e2_ids  # same card population feeds both scorers
        # D7: "both simulated and observed arms must cover identical persona
        # sets" -- the truth (diary) side must match the sim side exactly.
        assert e2_truth_ids == e2_ids
        assert split["e2"]["n_cards_without_diary_match"] == 0
        assert isinstance(split["e1"]["pooled_tvd"], float)
        assert split["e1"]["pooled_tvd"] == split["e1"]["pooled_tvd"]  # not NaN
        # review finding: truth restriction is PERSON-level, sim and truth
        # cover identical persona sets (n_truth_persons == subset size)
        assert split["e1"]["truth_restriction"] == "person_level"
        assert split["e1"]["n_truth_persons"] == len(split["e1"]["persona_ids"])

    assert set(result["splits"]["llm"]["e1"]["persona_ids"]) == {P1101, P1201}
    assert set(result["splits"]["fallback"]["e1"]["persona_ids"]) == {P1102, P1301}
    # H1 is the one mixed-provenance household (P1101 llm + P1102 fallback):
    # each split contains exactly one persona whose household-mate is outside it
    assert result["splits"]["llm"]["e1"]["n_mixed_household_personas"] == 1
    assert result["splits"]["fallback"]["e1"]["n_mixed_household_personas"] == 1


def test_provenance_split_scores_accepts_injected_producer():
    """The M3 runner injects loop-realized days instead of the static
    execute_days path (D6/D7): a ``producer(namespace) -> persona_id ->
    [RealizedDay]`` callable for the WHOLE population must be filterable to
    each split's own persona ids and flow straight through to
    ``evaluation.e1.simulate_arm`` / ``evaluation.e2.score_e2``'s own
    ``producer`` seam (added to those modules during this build) -- proven
    here by injecting an all-zero-trip producer and checking the pooled TVD
    changes accordingly (a real effect, not silently ignored) while
    persona-set consistency still holds.
    """
    ds = _synthetic_dataset()
    cards = _synthetic_cards()
    day_slots = e1.day_slots_by_persona(ds, e1.persona_of_person_map(seeding.persona_index(ds)))

    def zero_trip_producer(namespace):
        return {
            c["persona_id"]: [
                RealizedDay(day_index=d, day_weight=w, trips=[])
                for d, w in day_slots.get(c["persona_id"], [])
            ]
            for c in cards
        }

    default_result = dm.provenance_split_scores(cards, ds, n_runs=2, seed=0)
    injected_result = dm.provenance_split_scores(
        cards, ds, n_runs=2, seed=0, producer=zero_trip_producer,
    )

    for label in ("llm", "fallback"):
        split = injected_result["splits"][label]
        assert set(split["e1"]["persona_ids"]) == set(split["e2"]["truth_persona_ids"])
        # the injected all-quiet producer must actually be used (not silently
        # dropped back to the static path): pooled TVD differs from default
        assert (
            split["e1"]["pooled_tvd"]
            != default_result["splits"][label]["e1"]["pooled_tvd"]
        )


# ---------------------------------------------------------------------------
# 4b. person-level truth restriction (review finding, 2026-07-15)
# ---------------------------------------------------------------------------

def _old_household_only_distributions(person_days, trips, household_ids):
    """The PRE-CHANGE psrc distribution algorithm, replicated verbatim
    (household-only ``_select``): the byte-identity reference the additive
    ``person_ids=None`` default is regression-tested against. This copy is
    the test's frozen spec of the sealed truth path -- if the adapter's
    default path ever drifts from it, the sealed E1 truth has drifted."""
    def select(df):
        if household_ids is None:
            return df
        ids = household_ids if isinstance(household_ids, (set, frozenset)) else set(household_ids)
        return df[df["household_id"].isin(ids)]

    df = select(person_days)
    bins = df["n_collapsed"].map(psrc.trips_bin)
    tpd = psrc.normalize(
        df.groupby(bins)["w_day"].sum()
        .reindex(psrc.TRIPS_PER_DAY_BINS, fill_value=0.0).to_numpy(dtype=float)
    )
    tf = select(trips)
    modes = psrc.normalize(
        tf.groupby("mode")["w_trip"].sum()
        .reindex(e1.MODES, fill_value=0.0).to_numpy(dtype=float)
    )
    bands = psrc.normalize(
        tf.groupby("band")["w_trip"].sum()
        .reindex(psrc.TIME_BANDS, fill_value=0.0).to_numpy(dtype=float)
    )
    return tpd, modes, bands


def test_psrc_person_ids_default_is_byte_identical_to_prechange_path():
    """The additive ``person_ids`` kwarg (grounding/adapters/psrc.py) at its
    ``None`` default must be BYTE-IDENTICAL to the pre-change household-only
    algorithm on every household-filter case -- this code path feeds the
    sealed E1 truth (evaluation.e1.truth_distributions), which must not move.
    (tests/test_e1.py::test_truth_distributions_bit_match_sealed_record
    additionally pins the real-data outputs against the sealed record.)"""
    ds = _synthetic_dataset()
    for hh_ids in (None, {"H1"}, {"H1", "H2"}, {"H3"}, set()):
        exp_tpd, exp_modes, exp_bands = _old_household_only_distributions(
            ds.person_days, ds.weekday_trips, hh_ids,
        )
        np.testing.assert_array_equal(
            psrc.trips_per_day_distribution(ds.person_days, hh_ids), exp_tpd)
        np.testing.assert_array_equal(
            psrc.mode_share_distribution(ds.weekday_trips, hh_ids), exp_modes)
        np.testing.assert_array_equal(
            psrc.departure_band_distribution(ds.weekday_trips, hh_ids), exp_bands)
        # the dataset methods thread the default identically
        np.testing.assert_array_equal(ds.trips_per_day_distribution(hh_ids), exp_tpd)
        np.testing.assert_array_equal(ds.mode_share_distribution(hh_ids), exp_modes)
        np.testing.assert_array_equal(ds.departure_band_distribution(hh_ids), exp_bands)


def test_psrc_person_ids_filter_hand_check():
    """person_ids restricts to exactly those persons' rows (intersecting
    with any household filter), matched as strings."""
    ds = _synthetic_dataset()
    # person 1102 alone: one day of 2 trips -> bin "2" carries all mass
    tpd = ds.trips_per_day_distribution(person_ids={"1102"})
    expected = np.zeros(9)
    expected[2] = 1.0
    np.testing.assert_allclose(tpd, expected)
    # modes: 1102's trips are one ride + one transit at equal w_trip
    modes = ds.mode_share_distribution(person_ids={"1102"})
    assert modes[e1.MODES.index("ride")] == pytest.approx(0.5)
    assert modes[e1.MODES.index("transit")] == pytest.approx(0.5)
    assert modes[e1.MODES.index("car")] == pytest.approx(0.0)
    # intersection with a household filter: 1102 is in H1, so H2 & {1102} = empty -> NaN
    empty = ds.trips_per_day_distribution({"H2"}, person_ids={"1102"})
    assert np.isnan(empty).all()


def test_provenance_split_truth_is_person_level_not_household_level():
    """THE review-finding proof: provenance is per PERSON, households mix
    provenances (on the deployed r2b set, 24.3% of households are mixed and
    55.2% of fallback personas share a household with an LLM persona whose
    diary is systematically lighter), so the split's truth side must be the
    subset personas' OWN person-days only -- never their households'.

    Construction: mixed household H1 = heavy LLM person 1101 (one 3-trip
    day) + fallback person 1102 (one 2-trip day, distinct modes/bands); the
    fallback card replays 1102's diary EXACTLY, so against person-level
    truth the fallback split's pooled TVD is 0. Household-level truth would
    mix 1101's 3-trip car day into the fallback truth side (trips/day mass
    at bin 3, car mode mass, am_peak band mass) and force pooled TVD ~0.5 --
    the dilution bias this fix removes.
    """
    households = pd.DataFrame([
        {"household_id": "H1", "income_class": 3, "household_cars": 1},
        {"household_id": "H2", "income_class": 2, "household_cars": 1},
    ])
    persons = pd.DataFrame([
        {"person_id": "1101", "household_id": "H1"},
        {"person_id": "1102", "household_id": "H1"},
        {"person_id": "1201", "household_id": "H2"},
    ])
    person_days = pd.DataFrame([
        {"person_id": "1101", "household_id": "H1", "daynum": 1, "w_day": 1.0, "n_collapsed": 3},
        {"person_id": "1102", "household_id": "H1", "daynum": 1, "w_day": 1.0, "n_collapsed": 2},
        {"person_id": "1201", "household_id": "H2", "daynum": 1, "w_day": 1.0, "n_collapsed": 1},
    ])
    weekday_trips = pd.DataFrame(
        [
            {"person_id": "1101", "household_id": "H1", "daynum": 1, "tripnum": t,
             "mode": "car", "band": "am_peak", "w_trip": 1.0}
            for t in (1, 2, 3)
        ]
        + [
            {"person_id": "1102", "household_id": "H1", "daynum": 1, "tripnum": 1,
             "mode": "transit", "band": "midday", "w_trip": 1.0},
            {"person_id": "1102", "household_id": "H1", "daynum": 1, "tripnum": 2,
             "mode": "ride", "band": "evening", "w_trip": 1.0},
            {"person_id": "1201", "household_id": "H2", "daynum": 1, "tripnum": 1,
             "mode": "car", "band": "am_peak", "w_trip": 1.0},
        ]
    )
    ds = psrc.PSRCDataset(
        households=households, persons=persons, person_days=person_days,
        weekday_trips=weekday_trips, build_log={},
    )
    # sorted person ids 1101, 1102, 1201 -> P00001, P00002, P00003
    cards = [
        make_card("P00001", [{"id": "day", "weight": 1, "trips": [
            {"purpose": "work", "mode": "car", "depart_band": "am_peak"} for _ in range(3)
        ]}], source="llm"),
        make_card("P00002", [{"id": "day", "weight": 1, "trips": [
            {"purpose": "shop_daily", "mode": "transit", "depart_band": "midday"},
            {"purpose": "home", "mode": "ride", "depart_band": "evening"},
        ]}], source="fallback"),
        make_card("P00003", [{"id": "day", "weight": 1, "trips": [
            {"purpose": "work", "mode": "car", "depart_band": "am_peak"},
        ]}], source="llm"),
    ]

    result = dm.provenance_split_scores(cards, ds, n_runs=2, seed=0)

    fb = result["splits"]["fallback"]["e1"]
    assert fb["persona_ids"] == ["P00002"]
    assert fb["truth_restriction"] == "person_level"
    assert fb["n_truth_persons"] == 1
    # the single-pattern card replays the person's own diary exactly ->
    # person-level truth gives pooled TVD 0; household-level truth would mix
    # 1101's heavier car day into it (pooled TVD ~0.5)
    assert fb["pooled_tvd"] == pytest.approx(0.0, abs=1e-12)
    # P00002's household-mate P00001 is outside the fallback subset
    assert fb["n_mixed_household_personas"] == 1

    llm = result["splits"]["llm"]["e1"]
    assert llm["persona_ids"] == ["P00001", "P00003"]
    assert llm["pooled_tvd"] == pytest.approx(0.0, abs=1e-12)
    assert llm["n_mixed_household_personas"] == 1  # P00001 (mate P00002); P00003 pure


# ---------------------------------------------------------------------------
# 5. sensitivity_split -- smoke
# ---------------------------------------------------------------------------

def test_sensitivity_split_smoke():
    ds = _synthetic_dataset()
    cards = _synthetic_cards()

    persona_index = seeding.persona_index(ds)
    persona_of_person = e1.persona_of_person_map(persona_index)
    day_slots = e1.day_slots_by_persona(ds, persona_of_person)
    realized = execute_days(cards, day_slots, "sens_smoke", update_habits=False)
    observed_person_days = dm.observed_person_day_stats(ds, persona_of_person)
    observed_seqs = dm.observed_day_sequences_of(ds, persona_of_person)

    dup = dm.flag_duplicate_pattern_cards(cards)
    tiling = dm.flag_tiling_cards(cards, observed_seqs)
    dup_ids = set(dup["llm"]["persona_ids"]) | set(dup["fallback"]["persona_ids"])
    til_ids = set(tiling["llm"]["persona_ids"]) | set(tiling["fallback"]["persona_ids"])

    result = dm.sensitivity_split(
        cards, ds, realized, day_slots, observed_person_days, dup_ids, til_ids,
        n_runs=2, seed=0,
    )

    assert result["diagnostic_only"] is True
    assert set(result["cuts"]) == {
        "baseline", "exclude_duplicate_pattern", "exclude_tiling", "exclude_both",
    }
    assert result["cuts"]["baseline"]["n_remaining_cards"] == 4
    assert result["cuts"]["baseline"]["n_excluded"] == 0
    # nothing in this tiny synthetic population is flagged (no duplicate
    # patterns, no multi-day near-enumeration by construction) -> every cut
    # is a no-op on the card count
    for cut in result["cuts"].values():
        assert cut["n_remaining_cards"] == 4


# ---------------------------------------------------------------------------
# 6. write_diagnostics -- round-trip
# ---------------------------------------------------------------------------

def test_write_diagnostics_round_trip(tmp_path):
    ds = _synthetic_dataset()
    cards = _synthetic_cards()
    out_dir = tmp_path / "m3_smoke"

    result = dm.write_diagnostics(str(out_dir), cards, ds, n_runs=2, seed=0)

    assert result["diagnostic_only"] is True
    assert set(result) == {
        "diagnostic_only", "duplicate_pattern", "tiling", "within_person_variance",
        "provenance_split_scores", "sensitivity_split",
    }

    out_file = out_dir / "diagnostics_m3.json"
    assert out_file.is_file()
    written = json.loads(out_file.read_text())
    assert written["diagnostic_only"] is True
    assert set(written) == set(result)

    # determinism: persona-id lists are sorted on write
    llm_ids = written["duplicate_pattern"]["llm"]["persona_ids"]
    assert llm_ids == sorted(llm_ids)
    fb_ids = written["tiling"]["fallback"]["persona_ids"]
    assert fb_ids == sorted(fb_ids)

    # re-writing is byte-stable (same inputs -> same file content)
    dm.write_diagnostics(str(out_dir), cards, ds, n_runs=2, seed=0)
    written_again = json.loads(out_file.read_text())
    assert written_again == written
