"""grounding/seeding.py — evidence determinism, skeleton, reindex, prompt gate.

All fixtures are MASKED synthetic data (zone codes, generic labels). One
integration test loads the real adapter via load_or_build() and skips
gracefully when the gitignored data/psrc CSVs are absent.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from grounding import seeding
from grounding.adapters import psrc
from grounding.masking.mask_lint import (
    default_token_path,
    lint_text,
    load_forbidden_tokens,
)
from grounding.render import render_seed_prompt, render_seed_retry_prompt
from world import crn

WAVE_A, WAVE_B = psrc.WAVES  # never spell wave years as literals

FORBIDDEN = load_forbidden_tokens(default_token_path())

# A token from the versioned forbidden list, used to test the prompt gate.
PLANTED_TOKEN = "gamla stan"
assert PLANTED_TOKEN in FORBIDDEN


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------

def make_dataset() -> SimpleNamespace:
    """A tiny PSRCDataset-shaped object: three persons, two households."""
    persons = pd.DataFrame(
        [
            {"person_id": "1102", "household_id": "H2", "survey_year": WAVE_A,
             "can_drive_bool": True, "segment": "mid|car1p|remainder"},
            {"person_id": "1101", "household_id": "H1", "survey_year": WAVE_A,
             "can_drive_bool": True, "segment": "mid|car1p|catchment"},
            {"person_id": "1201", "household_id": "H9", "survey_year": WAVE_B,
             "can_drive_bool": False, "segment": "low|car0|remainder"},
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
            {"person_id": "1101", "daynum": 1, "n_collapsed": 2},
            {"person_id": "1101", "daynum": 2, "n_collapsed": 0},
            {"person_id": "1102", "daynum": 1, "n_collapsed": 1},
            {"person_id": "1201", "daynum": 1, "n_collapsed": 1},
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


def person_1101_frames():
    ds = make_dataset()
    pdays = ds.person_days[ds.person_days.person_id == "1101"]
    trips = ds.weekday_trips[ds.weekday_trips.person_id == "1101"]
    return pdays, trips


# ---------------------------------------------------------------------------
# evidence lines
# ---------------------------------------------------------------------------

def test_evidence_lines_deterministic_and_order_insensitive():
    pdays, trips = person_1101_frames()
    prow = {"transit_freq": "5 days a week"}
    a = seeding.evidence_lines_of(pdays, trips, prow)
    b = seeding.evidence_lines_of(pdays, trips, prow)
    assert a == b
    # shuffling input row order must not change the lines
    c = seeding.evidence_lines_of(
        pdays.iloc[::-1].reset_index(drop=True),
        trips.iloc[::-1].reset_index(drop=True),
        prow,
    )
    assert a == c


def test_evidence_lines_content_multi_day():
    pdays, trips = person_1101_frames()
    lines = seeding.evidence_lines_of(pdays, trips, {"transit_freq": "5 days a week"})
    text = "\n".join(lines)
    assert "Contributed 2 recorded weekday days." in lines
    assert any(l.startswith("Trips per recorded weekday:") for l in lines)
    assert "1 day with no trips" in text and "1 day with two trips" in text
    assert "Purpose work: 1 trip; modes car x1; usually departs in the am_peak." in lines
    assert "Overall weekday mode use: car x2." in lines
    assert "Overall departure timing: am_peak x1, pm_peak x1." in lines
    # multi-day persons get neutral per-day sequence lines (sequences ARE evidence)
    assert "Recorded day sequences:" in lines
    assert "One recorded day: work am_peak by car, then home pm_peak by car." in lines
    assert "One recorded day: no trips." in lines
    # weekend days mentioned but never summarized
    assert any("weekend" in l for l in lines)
    # masked self-reported frequency wording
    assert "Self-reports getting around by shared transit most weekdays." in lines


def test_evidence_lines_are_masked_no_times_no_tokens():
    pdays, trips = person_1101_frames()
    lines = seeding.evidence_lines_of(pdays, trips, {"transit_freq": "5 days a week"})
    for line in lines:
        assert not re.search(r"\d{1,2}:\d{2}", line), line  # no clock times
        assert not lint_text(line, FORBIDDEN), line  # no forbidden tokens


def test_evidence_single_day_person_has_no_sequence_block():
    ds = make_dataset()
    pdays = ds.person_days[ds.person_days.person_id == "1102"]
    trips = ds.weekday_trips[ds.weekday_trips.person_id == "1102"]
    lines = seeding.evidence_lines_of(pdays, trips, {})
    assert "Recorded day sequences:" not in lines
    assert "Contributed 1 recorded weekday day." in lines


def test_evidence_freq_lines_skipped_when_missing():
    pdays, trips = person_1101_frames()
    lines = seeding.evidence_lines_of(
        pdays, trips, {"transit_freq": "Missing Response"}
    )
    assert not any(l.startswith("Self-reports") for l in lines)
    lines2 = seeding.evidence_lines_of(pdays, trips, {})  # columns absent entirely
    assert not any(l.startswith("Self-reports") for l in lines2)


# ---------------------------------------------------------------------------
# observed_stats_of — the fidelity gate's reference (must be consistent with
# what evidence_lines_of tells the model, on the SAME frames)
# ---------------------------------------------------------------------------

def test_observed_stats_deterministic_and_order_insensitive():
    pdays, trips = person_1101_frames()
    a = seeding.observed_stats_of(pdays, trips)
    b = seeding.observed_stats_of(pdays, trips)
    assert a == b
    c = seeding.observed_stats_of(
        pdays.iloc[::-1].reset_index(drop=True),
        trips.iloc[::-1].reset_index(drop=True),
    )
    assert a == c


def test_observed_stats_values_multi_day():
    # person 1101: day1 has two car trips, day2 has zero trips.
    pdays, trips = person_1101_frames()
    obs = seeding.observed_stats_of(pdays, trips)
    assert obs["n_observed_weekdays"] == 2
    assert obs["n_observed_trips"] == 2
    assert obs["mean_trips_per_weekday"] == 1.0  # (2 + 0) / 2
    assert obs["mode_counts"] == {"car": 2}
    assert obs["mode_shares"] == {"car": 1.0}
    assert obs["has_quiet_weekday"] is True
    assert obs["quiet_share"] == 0.5  # one of two weekdays quiet


def test_observed_stats_consistent_with_evidence_lines():
    # The gate's mode reference must use the SAME counts the evidence line
    # ("Overall weekday mode use: ...") shows the model.
    pdays, trips = person_1101_frames()
    obs = seeding.observed_stats_of(pdays, trips)
    lines = seeding.evidence_lines_of(pdays, trips, {})
    mode_line = next(l for l in lines if l.startswith("Overall weekday mode use:"))
    # parse "Overall weekday mode use: car x2." -> {"car": 2}
    body = mode_line.split(":", 1)[1].strip().rstrip(".")
    parsed = {
        seg.strip().split(" x")[0]: int(seg.strip().split(" x")[1])
        for seg in body.split(",")
    }
    assert parsed == obs["mode_counts"]
    # the trips-per-weekday histogram the model sees is the same day set /
    # quiet split the gate scores against
    assert any("1 day with no trips" in l for l in lines)  # the one quiet day
    assert obs["has_quiet_weekday"] is True


def test_observed_stats_day_weighted_when_w_day_present():
    # Non-uniform day weights: the quiet day carries triple weight, so the
    # day-weighted mean and quiet share diverge from the raw (uniform) figures.
    pdays = pd.DataFrame(
        [
            {"person_id": "x", "daynum": 1, "n_collapsed": 4, "w_day": 1.0},
            {"person_id": "x", "daynum": 2, "n_collapsed": 0, "w_day": 3.0},
        ]
    )
    trips = pd.DataFrame(
        [
            {"daynum": 1, "tripnum": 1, "purpose": "work", "mode": "car", "band": "am_peak"},
            {"daynum": 1, "tripnum": 2, "purpose": "work", "mode": "car", "band": "am_peak"},
            {"daynum": 1, "tripnum": 3, "purpose": "leisure", "mode": "walk", "band": "midday"},
            {"daynum": 1, "tripnum": 4, "purpose": "home", "mode": "car", "band": "pm_peak"},
        ]
    )
    obs = seeding.observed_stats_of(pdays, trips)
    # day-weighted mean = (1*4 + 3*0) / (1 + 3) = 1.0
    assert obs["mean_trips_per_weekday"] == 1.0
    # quiet share = 3 / 4
    assert obs["quiet_share"] == 0.75
    # mode shares are day-weighted over the active day's trips (all weight 1):
    assert obs["mode_shares"] == {"car": 0.75, "walk": 0.25}
    # raw counts still echo the evidence line
    assert obs["mode_counts"] == {"car": 3, "walk": 1}


def test_observed_stats_no_observed_days_is_empty():
    obs = seeding.observed_stats_of(
        pd.DataFrame(columns=["daynum", "n_collapsed"]),
        pd.DataFrame(columns=["daynum", "tripnum", "purpose", "mode", "band"]),
    )
    assert obs["n_observed_weekdays"] == 0
    assert obs["has_quiet_weekday"] is False
    assert obs["mean_trips_per_weekday"] == 0.0


# ---------------------------------------------------------------------------
# persona_id reindex
# ---------------------------------------------------------------------------

def test_persona_id_reindex_is_rank_of_sorted_person_id():
    mapping = seeding._persona_id_map(["30", "100", "2"])
    # sorted lexicographically as strings: "100" < "2" < "30"
    assert mapping == {"100": "P00001", "2": "P00002", "30": "P00003"}


def test_persona_index_stable_under_row_order():
    ds = make_dataset()
    idx1 = seeding.persona_index(ds, data_dir=None)
    ds2 = make_dataset()
    ds2.persons = ds2.persons.iloc[::-1].reset_index(drop=True)
    idx2 = seeding.persona_index(ds2, data_dir=None)
    m1 = dict(zip(idx1.person_id, idx1.persona_id))
    m2 = dict(zip(idx2.person_id, idx2.persona_id))
    assert m1 == m2
    assert all(re.fullmatch(r"P\d{5,}", p) for p in m1.values())


def test_persona_index_carries_keys_and_skeleton_fields():
    ds = make_dataset()
    idx = seeding.persona_index(ds, data_dir=None)
    assert len(idx) == 3
    assert idx.persona_id.is_unique
    for col in ("person_id", "household_id", "wave", "fold", "segment"):
        assert col in idx.columns
    for col in seeding.SKELETON_FIELDS:
        assert col in idx.columns
    row = idx[idx.person_id == "1101"].iloc[0]
    assert row["household_cars"] == 1
    assert row["household_size"] == 3
    assert row["income_class"] == 4
    assert row["fold"] == psrc.fold_id("H1")
    assert bool(row["can_drive"]) is True


# ---------------------------------------------------------------------------
# skeleton
# ---------------------------------------------------------------------------

def test_skeleton_of_maps_demographics():
    person = {
        "persona_id": "P00001",
        "can_drive_bool": True,
        "employment": "Employed full time (35+ hours/week, paid)",
        "adult_student": "No, not a student",
        "schooltype": "Missing Response",
        "age": "35-44 years",
        "work_tract_2020": "990101",
    }
    household = {"hhsize": 3, "household_cars": 1, "income_class": 4,
                 "home_tract_2020": "990001"}
    sk = seeding.skeleton_of(person, household)
    assert sk["age"] == 39  # band midpoint
    assert sk["employed"] is True
    assert sk["student"] is False
    assert sk["can_drive"] is True
    assert sk["household_size"] == 3
    assert sk["household_cars"] == 1
    assert sk["income_class"] == 4
    assert set(sk) == set(seeding.SKELETON_FIELDS)


def test_skeleton_zone_placeholder_and_injection_seam():
    person = {"persona_id": "P00001", "can_drive_bool": True,
              "employment": "Employed full time (35+ hours/week, paid)",
              "work_tract_2020": "990101"}
    household = {"household_cars": 1, "income_class": 3, "hhsize": 2,
                 "home_tract_2020": "990001"}
    sk = seeding.skeleton_of(person, household)
    # tracts unknown to the committed map fall back to the placeholder
    assert sk["home_zone"] == "Z00"
    assert sk["work_zone"] == "Z00"
    # the one-line injection: a committed tract->zone callable
    zone_of = {"990001": "Z04", "990101": "Z11"}.get
    sk2 = seeding.skeleton_of(person, household, zone_of_tract=zone_of)
    assert sk2["home_zone"] == "Z04"
    assert sk2["work_zone"] == "Z11"
    # not employed, no injection -> no work zone
    sk3 = seeding.skeleton_of(
        {"persona_id": "P00002", "can_drive_bool": True, "employment": "Retired"},
        household,
    )
    assert sk3["work_zone"] is None


def test_skeleton_has_pass_is_crn_drawn_from_world_prior():
    person = {"persona_id": "P00042", "can_drive_bool": True}
    household = {"household_cars": 1, "income_class": 3}
    sk1 = seeding.skeleton_of(person, household)
    sk2 = seeding.skeleton_of(person, household)
    assert sk1["has_pass"] == sk2["has_pass"]  # deterministic
    # the documented key contract, against the frozen world prior
    expected = crn.draw("seed:P00042:has_pass") < seeding.pass_prior()
    assert sk1["has_pass"] == expected
    assert seeding.pass_prior() == 0.75  # frozen world-config value


# ---------------------------------------------------------------------------
# prompt building + lint gate
# ---------------------------------------------------------------------------

def test_build_prompts_writes_gated_jsonl(tmp_path):
    ds = make_dataset()
    out = tmp_path / "prompts.jsonl"
    stats = seeding.build_prompts(ds, out, data_dir=None)
    assert stats["n_prompts"] == 3
    assert stats["n_offenders"] == 0
    records = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(records) == 3
    for rec in records:
        assert set(rec) == {"persona_id", "prompt", "attempt"}
        assert rec["attempt"] == 1
        assert rec["prompt"].startswith("You are writing a compact behavior card")
        assert "=== RESIDENT ===" in rec["prompt"]
        assert "=== TRAVEL EVIDENCE" in rec["prompt"]
    # prompts route through the ONE render path
    idx = seeding.persona_index(ds, data_dir=None)
    row = idx[idx.person_id == "1102"].iloc[0].to_dict()
    skeleton = {f: row[f] for f in seeding.SKELETON_FIELDS}
    pdays = ds.person_days[ds.person_days.person_id == "1102"]
    trips = seeding.enriched_trips(ds, data_dir=None, person_ids=["1102"])
    evidence = seeding.evidence_lines_of(pdays, trips, {})
    # the prompt surface renders an absent work zone as "none", never "None"
    render_skeleton = {k: ("none" if v is None else v) for k, v in skeleton.items()}
    expected = render_seed_prompt(render_skeleton, evidence, len(pdays), "serve")
    by_id = {r["persona_id"]: r["prompt"] for r in records}
    assert by_id[row["persona_id"]] == expected


def test_build_prompts_lint_gate_catches_planted_token(tmp_path):
    ds = make_dataset()
    out = tmp_path / "prompts.jsonl"
    with pytest.raises(ValueError) as excinfo:
        # plant a forbidden token through the zone seam: it lands in the
        # rendered skeleton block, and the gate must fail loud
        seeding.build_prompts(ds, out, data_dir=None, zone_of_tract=lambda t: PLANTED_TOKEN)
    msg = str(excinfo.value)
    assert "MASK-LINT" in msg
    assert PLANTED_TOKEN in msg
    assert "P00001" in msg  # offending personas are listed


def test_build_retry_prompts_appends_failure_block(tmp_path):
    skeleton = {"home_zone": "Z04", "age": 39, "can_drive": True}
    evidence = ["Contributed 1 recorded weekday day."]
    failures = [
        {
            "persona_id": "P00007",
            "skeleton": skeleton,
            "evidence_lines": evidence,
            "n_observed_days": 1,
            "failure_reasons": ["patterns[0].weight: 0 < minimum 1"],
            "attempt": 1,
        }
    ]
    out = tmp_path / "retry.jsonl"
    stats = seeding.build_retry_prompts(failures, out)
    assert stats["n_prompts"] == 1
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["persona_id"] == "P00007"
    assert rec["attempt"] == 2  # bumped
    base = render_seed_prompt(skeleton, evidence, 1, "serve")
    assert rec["prompt"].startswith(base.rstrip("\n"))
    assert "PREVIOUS ATTEMPT FAILED VALIDATION" in rec["prompt"]
    assert "patterns[0].weight: 0 < minimum 1" in rec["prompt"]
    assert rec["prompt"] == render_seed_retry_prompt(
        skeleton, evidence, 1, failures[0]["failure_reasons"], "serve"
    )


def test_fidelity_feedback_feeds_retry_prompt_verbatim(tmp_path):
    # the numeric fidelity feedback strings must survive the retry-prompt
    # mask-lint gate and land verbatim in the rendered retry prompt
    from grounding import card_validation as cv

    pdays = pd.DataFrame([{"person_id": "x", "daynum": 1, "n_collapsed": 3}])
    trips = pd.DataFrame(
        [
            {"daynum": 1, "tripnum": 1, "purpose": "work", "mode": "car", "band": "am_peak"},
            {"daynum": 1, "tripnum": 2, "purpose": "home", "mode": "car", "band": "pm_peak"},
            {"daynum": 1, "tripnum": 3, "purpose": "shop_daily", "mode": "car", "band": "midday"},
        ]
    )
    obs = seeding.observed_stats_of(pdays, trips)
    # a card that undershoots the mean and distorts the mode mix
    card = {
        "patterns": [{"id": "p0", "weight": 5,
                      "trips": [{"purpose": "work", "mode": "walk", "depart_band": "am_peak"}]}],
        "rules": [],
        "voice": "x",
    }
    reasons = cv.fidelity(card, obs)
    assert reasons  # it does fail
    failures = [{
        "persona_id": "P00009",
        "skeleton": {"home_zone": "Z01"},
        "evidence_lines": [],
        "n_observed_days": 1,
        "failure_reasons": reasons,
        "attempt": 1,
    }]
    out = tmp_path / "retry.jsonl"
    seeding.build_retry_prompts(failures, out)  # must NOT raise MASK-LINT
    rendered = json.loads(out.read_text().splitlines()[0])["prompt"]
    for r in reasons:
        assert r in rendered  # verbatim in the rendered retry prompt


def test_build_retry_prompts_lint_gate(tmp_path):
    failures = [
        {
            "persona_id": "P00008",
            "skeleton": {"home_zone": "Z01"},
            "evidence_lines": [],
            "n_observed_days": 0,
            "failure_reasons": [f"voice mentions {PLANTED_TOKEN}"],
            "attempt": 1,
        }
    ]
    with pytest.raises(ValueError, match="MASK-LINT"):
        seeding.build_retry_prompts(failures, tmp_path / "retry.jsonl")


# ---------------------------------------------------------------------------
# real-adapter integration (skips gracefully without data/psrc)
# ---------------------------------------------------------------------------

_RAW_PERSONS = psrc.DEFAULT_DATA_DIR / "hts_persons_2017_2025_v2026.1.csv"


@pytest.mark.skipif(
    not _RAW_PERSONS.exists(), reason="gitignored data/psrc CSVs absent"
)
def test_integration_real_adapter_three_persons_lint_clean():
    dataset = psrc.load_or_build()
    idx = seeding.persona_index(dataset)
    # one persona per person, all of them
    assert len(idx) == len(dataset.persons)
    assert idx.persona_id.is_unique
    assert idx.person_id.is_unique

    person_ids = sorted(idx.person_id)[:3]
    trips = seeding.enriched_trips(dataset, person_ids=person_ids)
    person_days = dataset.person_days
    rows = idx.set_index("person_id")
    for pid in person_ids:
        row = rows.loc[pid].to_dict()
        skeleton = {f: row[f] for f in seeding.SKELETON_FIELDS}
        pdays = person_days[person_days.person_id.astype(str) == pid]
        ptrips = trips[trips.person_id == pid]
        evidence = seeding.evidence_lines_of(pdays, ptrips, {})
        prompt = render_seed_prompt(skeleton, evidence, len(pdays), "serve")
        assert not lint_text(prompt, FORBIDDEN), f"prompt for {row['persona_id']} not lint-clean"
        assert not re.search(r"\d{1,2}:\d{2}", "\n".join(evidence))
