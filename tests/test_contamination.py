"""E5(i) machinery: quarantined unmasked vocabulary, the unmasked prompt
builder (inverted mask-lint assertion), paired-probe arithmetic + flag
threshold, CRN namespace pairing, and the run_e5 CLI on synthetic card sets.

The unmasked vocabulary is quarantined (evaluation/truth/); every test that
touches it sets AGORA_EVAL_CONTEXT=1 first, exactly as the E5 harness does.
This test file itself contains no real place-name literals: real strings are
only ever read out of the quarantined module or the forbidden-token list.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agents.card_executor import execute_days
from evaluation import contamination, run_e5
from grounding import seeding
from grounding.masking.mask_lint import (
    default_token_path,
    lint_text,
    load_forbidden_tokens,
)
from world.config import get_config
from world.network import ERA_LABELS
from world.tolling import PERIODS

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = load_forbidden_tokens(default_token_path())


@pytest.fixture
def uv(monkeypatch):
    """The quarantined unmasked vocabulary, imported the harness way."""
    monkeypatch.setenv("AGORA_EVAL_CONTEXT", "1")
    from evaluation.truth import unmasked_vocabulary

    return unmasked_vocabulary


# ---------------------------------------------------------------------------
# synthetic fixtures (masked; mirror tests/test_seeding.py's shapes)
# ---------------------------------------------------------------------------

def make_dataset() -> SimpleNamespace:
    persons = pd.DataFrame(
        [
            {"person_id": "1101", "household_id": "H1", "survey_year": 0,
             "can_drive_bool": True},
            {"person_id": "1102", "household_id": "H2", "survey_year": 0,
             "can_drive_bool": True},
            {"person_id": "1201", "household_id": "H9", "survey_year": 0,
             "can_drive_bool": False},
        ]
    )
    households = pd.DataFrame(
        [
            {"household_id": "H1", "income_class": 4, "household_cars": 1,
             "hhsize": 3, "home_tract_2020": "990001"},
            {"household_id": "H2", "income_class": 3, "household_cars": 2,
             "hhsize": 2, "home_tract_2020": "990002"},
            {"household_id": "H9", "income_class": 1, "household_cars": 0,
             "hhsize": 1, "home_tract_2020": "990009"},
        ]
    )
    person_days = pd.DataFrame(
        [
            {"person_id": "1101", "daynum": 1, "w_day": 1.0, "n_collapsed": 2},
            {"person_id": "1101", "daynum": 2, "w_day": 1.0, "n_collapsed": 0},
            {"person_id": "1102", "daynum": 1, "w_day": 1.0, "n_collapsed": 1},
            {"person_id": "1201", "daynum": 1, "w_day": 1.0, "n_collapsed": 1},
        ]
    )
    weekday_trips = pd.DataFrame(
        [
            {"trip_id": "t1", "person_id": "1101", "daynum": 1, "tripnum": 1,
             "purpose": "work", "mode": "car", "band": "am_peak", "w_trip": 1.0},
            {"trip_id": "t2", "person_id": "1101", "daynum": 1, "tripnum": 2,
             "purpose": "home", "mode": "car", "band": "pm_peak", "w_trip": 1.0},
            {"trip_id": "t3", "person_id": "1102", "daynum": 1, "tripnum": 1,
             "purpose": "shop_daily", "mode": "walk", "band": "midday", "w_trip": 1.0},
            {"trip_id": "t4", "person_id": "1201", "daynum": 1, "tripnum": 1,
             "purpose": "leisure", "mode": "transit", "band": "evening", "w_trip": 1.0},
        ]
    )
    return SimpleNamespace(
        persons=persons, households=households,
        person_days=person_days, weekday_trips=weekday_trips,
    )


# persona ids under the seeding reindex (sorted person ids 1101, 1102, 1201)
P1, P2, P3 = "P00001", "P00002", "P00003"


def make_card(persona_id, patterns, cars=1, can_drive=True):
    return {
        "persona_id": persona_id,
        "skeleton": {"household_cars": cars, "can_drive": can_drive},
        "patterns": patterns,
        "rules": [],
    }


def faithful_cards():
    """Cards that reproduce the synthetic diary distributions closely."""
    return [
        make_card(P1, [
            {"id": "commute", "weight": 1, "trips": [
                {"purpose": "work", "mode": "car", "depart_band": "am_peak"},
                {"purpose": "home", "mode": "car", "depart_band": "pm_peak"},
            ]},
            {"id": "quiet", "weight": 1, "trips": []},
        ]),
        make_card(P2, [
            {"id": "errand", "weight": 1, "trips": [
                {"purpose": "shop_daily", "mode": "walk", "depart_band": "midday"},
            ]},
        ]),
        make_card(P3, [
            {"id": "social", "weight": 1, "trips": [
                {"purpose": "leisure", "mode": "transit", "depart_band": "evening"},
            ]},
        ], cars=0, can_drive=False),
    ]


def distorted_cards():
    """Same personas, deliberately wrong behavior (everything bike/night)."""
    pattern = {"id": "off", "weight": 1, "trips": [
        {"purpose": "other", "mode": "bike", "depart_band": "night"},
        {"purpose": "other", "mode": "bike", "depart_band": "night"},
        {"purpose": "other", "mode": "bike", "depart_band": "night"},
    ]}
    return [
        make_card(P1, [dict(pattern)]),
        make_card(P2, [dict(pattern)]),
        make_card(P3, [dict(pattern)], cars=0, can_drive=False),
    ]


# ---------------------------------------------------------------------------
# the quarantined vocabulary itself
# ---------------------------------------------------------------------------

def test_zone_names_cover_all_thirty_zones(uv):
    expected = {f"Z{i:02d}" for i in range(1, 31)}
    assert set(uv.ZONE_NAMES) == expected
    values = list(uv.ZONE_NAMES.values())
    assert all(isinstance(v, str) and v for v in values)
    assert len(set(values)) == 30  # all distinct


def test_vocabulary_deliberately_carries_forbidden_tokens(uv):
    """INVERTED assertion: the unmasked vocabulary EXISTS to name the real
    arena, so it must hit the forbidden-token list — heavily."""
    corpus = "\n".join(
        list(uv.ZONE_NAMES.values())
        + list(uv.FACILITY_NAMES.values())
        + list(uv.ERA_DATES.values())
        + [uv.UNKNOWN_ZONE_NAME]
    )
    hit_tokens = {v.token for v in lint_text(corpus, FORBIDDEN)}
    assert len(hit_tokens) >= 5, (
        f"unmasked vocabulary hits only {sorted(hit_tokens)} — an unmasked "
        "arm that does not name the arena cannot probe contamination"
    )


def test_facility_names_cover_world_facility_codes(uv):
    cfg = get_config("cityk_corridor")
    world_codes = set(cfg.facilities) | {cfg.water_facility.code}
    assert world_codes <= set(uv.FACILITY_NAMES)


def test_prices_eras_currency_match_world_vocabulary(uv):
    assert set(uv.TOLL_PRICES_DOLLARS) == set(PERIODS)
    assert set(uv.TOLL_PRICE_STRINGS) == set(PERIODS)
    assert set(uv.ERA_DATES) == set(ERA_LABELS)
    assert uv.CURRENCY == "dollars"
    assert uv.NONPASS_SURCHARGE_DOLLARS > 0
    # peak must price above off-peak, as in the real schedule
    assert uv.TOLL_PRICES_DOLLARS["pm_peak"] > uv.TOLL_PRICES_DOLLARS["offpeak"]


def test_unmask_text_applies_documented_mapping(uv):
    masked, real = next(iter(uv.EVIDENCE_LINE_MAP.items()))
    assert uv.unmask_text(f"prefix {masked} suffix") == f"prefix {real} suffix"


def test_unmask_skeleton_substitutes_zones_only(uv):
    skeleton = {"home_zone": "Z07", "work_zone": None, "age": 39, "employed": True}
    out = uv.unmask_skeleton(skeleton)
    assert out["home_zone"] == uv.ZONE_NAMES["Z07"]
    assert out["work_zone"] is None
    assert out["age"] == 39 and out["employed"] is True
    assert skeleton["home_zone"] == "Z07"  # input not mutated


# ---------------------------------------------------------------------------
# the unmasked prompt builder
# ---------------------------------------------------------------------------

def test_unmasked_prompts_carry_tokens_masked_pipeline_stays_clean(
    uv, tmp_path, monkeypatch
):
    ds = make_dataset()
    zones = {"990001": "Z07", "990002": "Z21", "990009": "Z30"}

    # masked arm: the seeding gate passes and no prompt carries a token
    masked_path = tmp_path / "masked.jsonl"
    summary = seeding.build_prompts(
        ds, masked_path, data_dir=None, zone_of_tract=zones.get
    )
    assert summary["n_offenders"] == 0
    for line in masked_path.read_text().splitlines():
        assert not lint_text(json.loads(line)["prompt"], FORBIDDEN)

    # unmasked arm: EVERY prompt must carry at least one forbidden token
    unmasked_path = tmp_path / "unmasked.jsonl"
    result = contamination.build_unmasked_prompts(
        ds, unmasked_path, data_dir=None, zone_of_tract=zones.get
    )
    assert result["n_prompts"] == 3
    assert result["lint_exempt"] is True
    assert result["n_prompts_with_unmasked_tokens"] == 3
    records = [json.loads(line) for line in unmasked_path.read_text().splitlines()]
    assert len(records) == 3
    for rec in records:
        assert rec["arm"] == "unmasked"
        hits = lint_text(rec["prompt"], FORBIDDEN)
        assert hits, f"unmasked prompt for {rec['persona_id']} carries no real tokens"


def test_unmasked_prompt_substitutes_zone_names_same_render_path(uv, tmp_path):
    ds = make_dataset()
    zones = {"990001": "Z07", "990002": "Z21", "990009": "Z30"}
    out_path = tmp_path / "unmasked.jsonl"
    contamination.build_unmasked_prompts(
        ds, out_path, data_dir=None, zone_of_tract=zones.get
    )
    by_id = {
        json.loads(line)["persona_id"]: json.loads(line)["prompt"]
        for line in out_path.read_text().splitlines()
    }
    # zone code replaced by its real name (read from the quarantined module,
    # never a literal here)
    assert uv.ZONE_NAMES["Z07"] in by_id[P1]
    assert "Z07" not in by_id[P1]
    # SAME single render path: the masked template envelope is unchanged
    assert "=== RESIDENT ===" in by_id[P1]
    assert "=== TRAVEL EVIDENCE" in by_id[P1]


def test_unmasked_prompt_output_path_is_restricted(tmp_path):
    # outside the repo tree: allowed
    contamination._assert_out_path_allowed(tmp_path / "ok.jsonl")
    # under data/ (gitignored): allowed
    contamination._assert_out_path_allowed(REPO_ROOT / "data" / "e5" / "u.jsonl")
    # under data/synthetic/ (tracked): refused
    with pytest.raises(ValueError, match="forbidden tokens"):
        contamination._assert_out_path_allowed(
            REPO_ROOT / "data" / "synthetic" / "u.jsonl"
        )
    # anywhere else in the repo tree: refused
    for bad in ("evaluation", "grounding", "runs", "docs"):
        with pytest.raises(ValueError, match="forbidden tokens"):
            contamination._assert_out_path_allowed(REPO_ROOT / bad / "u.jsonl")


def test_grounding_seeding_has_no_lint_escape_hatch():
    """The exemption must live HERE, not as a flag on the masked pipeline:
    grounding/seeding.py exposes no way to skip its gate."""
    import inspect

    for fn in (seeding.build_prompts, seeding.build_retry_prompts):
        params = inspect.signature(fn).parameters
        assert not any("lint" in name.lower() for name in params), (
            f"{fn.__name__} grew a lint bypass parameter — the E5 exemption "
            "must stay quarantined in evaluation/contamination.py"
        )


# ---------------------------------------------------------------------------
# paired-probe arithmetic + flag threshold
# ---------------------------------------------------------------------------

def _dists(a, b, c):
    return {
        "trips_per_day": np.array(a, dtype=float),
        "mode_shares": np.array(b, dtype=float),
        "time_bands": np.array(c, dtype=float),
    }


def test_tvd_hand_check():
    assert contamination.tvd([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert contamination.tvd([0.5, 0.5], [0.3, 0.7]) == pytest.approx(0.2)
    assert contamination.tvd([0.25, 0.75], [0.25, 0.75]) == pytest.approx(0.0)


def test_paired_probe_arithmetic_and_pooling():
    ref = _dists([0.5, 0.5], [0.5, 0.5], [0.5, 0.5])
    masked = _dists([0.7, 0.3], [0.6, 0.4], [0.55, 0.45])    # TVDs 0.2, 0.1, 0.05
    unmasked = _dists([0.6, 0.4], [0.55, 0.45], [0.5, 0.5])  # TVDs 0.1, 0.05, 0.0
    probe = contamination.paired_probe(masked, unmasked, ref)
    assert probe["pooled_tvd_masked"] == pytest.approx(0.2)   # max over families
    assert probe["pooled_tvd_unmasked"] == pytest.approx(0.1)
    assert probe["relative_improvement"] == pytest.approx(0.5)
    assert probe["flag"] is True
    assert probe["per_family"]["trips_per_day"]["tvd_masked"] == pytest.approx(0.2)


def test_paired_probe_flag_threshold_boundary():
    ref = _dists([0.5, 0.5], [0.5, 0.5], [0.5, 0.5])
    masked = _dists([0.7, 0.3], [0.5, 0.5], [0.5, 0.5])  # pooled 0.2

    # exactly at the threshold: 25% improvement is NOT a flag (flag if > 25%)
    at = _dists([0.65, 0.35], [0.5, 0.5], [0.5, 0.5])    # pooled 0.15 -> rel 0.25
    probe = contamination.paired_probe(masked, at, ref)
    assert probe["relative_improvement"] == pytest.approx(0.25)
    assert probe["flag"] is False

    just_over = _dists([0.649, 0.351], [0.5, 0.5], [0.5, 0.5])
    assert contamination.paired_probe(masked, just_over, ref)["flag"] is True

    # unmasked WORSE than masked: negative improvement, never a flag
    worse = _dists([0.8, 0.2], [0.5, 0.5], [0.5, 0.5])
    probe = contamination.paired_probe(masked, worse, ref)
    assert probe["relative_improvement"] < 0
    assert probe["flag"] is False


def test_paired_probe_zero_masked_tvd_is_defined():
    ref = _dists([0.5, 0.5], [0.5, 0.5], [0.5, 0.5])
    perfect = _dists([0.5, 0.5], [0.5, 0.5], [0.5, 0.5])
    probe = contamination.paired_probe(perfect, perfect, ref)
    assert probe["relative_improvement"] == 0.0
    assert probe["flag"] is False


# ---------------------------------------------------------------------------
# CRN namespace pairing + end-to-end scoring
# ---------------------------------------------------------------------------

def test_identical_arms_share_draws_and_never_flag():
    ds = make_dataset()
    results = contamination.score_e5(
        faithful_cards(), faithful_cards(), ds, n_runs=4, seed=0
    )
    assert results["namespaces"] == [f"run{k}" for k in range(4)]
    probe = results["probe"]
    assert probe["pooled_tvd_masked"] == pytest.approx(probe["pooled_tvd_unmasked"])
    assert probe["relative_improvement"] == pytest.approx(0.0)
    assert results["contamination_flag"] is False


def test_crn_pairing_same_key_same_draw_across_arms():
    """Two arms that build the same CRN keys realize the same choices —
    the pairing the sealed 25% threshold relies on (D3)."""
    ds = make_dataset()
    id_map = seeding._persona_id_map(ds.persons["person_id"].astype(str))
    from evaluation.e2 import day_slots_of

    slots = day_slots_of(ds.person_days, id_map)
    arm_a = execute_days(faithful_cards(), slots, "run0", update_habits=False)
    arm_b = execute_days(faithful_cards(), slots, "run0", update_habits=False)
    assert arm_a == arm_b
    # a different namespace is an INDEPENDENT stream: across several other
    # namespaces at least one realization must differ (deterministic fact —
    # the CRN is a pure function of the key strings)
    others = [
        execute_days(faithful_cards(), slots, f"run{k}", update_habits=False)
        for k in range(1, 6)
    ]
    assert any(other != arm_a for other in others)


def test_score_e5_flags_memorizing_unmasked_arm():
    """Masked arm distorted, unmasked arm faithful to the reference: a large
    relative improvement must raise the contamination flag."""
    ds = make_dataset()
    results = contamination.score_e5(
        distorted_cards(), faithful_cards(), ds, n_runs=4, seed=0
    )
    probe = results["probe"]
    assert probe["pooled_tvd_unmasked"] < probe["pooled_tvd_masked"]
    assert probe["relative_improvement"] > 0.25
    assert results["contamination_flag"] is True


def test_score_e5_requires_identical_populations():
    ds = make_dataset()
    with pytest.raises(ValueError, match="IDENTICAL agent populations"):
        contamination.score_e5(
            faithful_cards(), faithful_cards()[:2], ds, n_runs=2, seed=0
        )


def test_realized_distributions_hand_check():
    from agents.card_executor import RealizedDay, RealizedTrip

    realized = {
        "PA": [RealizedDay(1, 2.0, [RealizedTrip("work", "car", "am_peak")])],
        "PB": [RealizedDay(1, 2.0, [])],
    }
    dists = contamination.realized_distributions(realized)
    # trips/day: one day with 1 trip (mass 2), one with 0 (mass 2)
    assert dists["trips_per_day"][0] == pytest.approx(0.5)
    assert dists["trips_per_day"][1] == pytest.approx(0.5)
    # all trip mass on car / am_peak
    from grounding.taxonomy import MODES

    assert dists["mode_shares"][MODES.index("car")] == pytest.approx(1.0)
    assert dists["time_bands"][1] == pytest.approx(1.0)  # am_peak is band 2


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def _write_synthetic_cache(cache_dir):
    ds = make_dataset()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds.households.to_pickle(cache_dir / "households.pkl")
    ds.persons.to_pickle(cache_dir / "persons.pkl")
    ds.person_days.to_pickle(cache_dir / "person_days.pkl")
    ds.weekday_trips.to_pickle(cache_dir / "weekday_trips_collapsed.pkl")


def test_run_e5_cli_end_to_end(tmp_path):
    cache_dir = tmp_path / "cache"
    _write_synthetic_cache(cache_dir)
    masked_path = tmp_path / "masked_cards.json"
    unmasked_path = tmp_path / "unmasked_cards.json"
    masked_path.write_text(json.dumps(distorted_cards()))
    unmasked_path.write_text(json.dumps(faithful_cards()))
    out_dir = tmp_path / "out" / "e5-test"

    rc = run_e5.main(
        [
            "--masked-cards", str(masked_path),
            "--unmasked-cards", str(unmasked_path),
            "--out", str(out_dir),
            "--runs", "4",
            "--seed", "0",
            "--cache-dir", str(cache_dir),
        ]
    )
    assert rc == 0
    results = json.loads((out_dir / "results.json").read_text())
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert results["eval"] == "e5i"
    assert results["contamination_flag"] is True
    assert results["probe"]["threshold"] == 0.25
    assert manifest["namespaces"] == [f"run{k}" for k in range(4)]
    assert len(manifest["masked_cards_sha256"]) == 64
    assert len(manifest["unmasked_cards_sha256"]) == 64
    assert manifest["masked_cards_sha256"] != manifest["unmasked_cards_sha256"]
    assert "pairing" in manifest
