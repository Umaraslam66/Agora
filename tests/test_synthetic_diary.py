"""M0 tests for the synthetic diary stand-in (grounding/synthetic_diary.py +
grounding/adapters/rvu_schema.py).

Covers, per the M0 spec:
  - schema JSONL round-trip (records -> file -> records, lossless);
  - determinism (same seed -> byte-identical output file);
  - household atomicity (members share household_id; no person in two
    households; the E1 80/20 split is household-atomic);
  - marginal sanity (trips/day in [1.5, 4], all four modes present,
    synthetic flag on 100% of records, DEV header line present);
  - persistent heterogeneity (per-person mode shares are over-dispersed
    versus an iid-choice baseline — the property E2's spread ratio needs
    the stand-in to exercise).

Run:  .venv/bin/python -m pytest tests/test_synthetic_diary.py -v
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounding.adapters.rvu_schema import (
    MODES,
    SCHEMA_VERSION,
    crosses_cordon,
    household_atomicity_violations,
    household_split,
    is_within_cordon,
    read_jsonl,
    record_from_dict,
    record_to_dict,
    validate_record,
    write_jsonl,
    zone_ring,
)
from grounding.synthetic_diary import generate, summarize

SEED = 42
N = 200  # enough for stable marginals, small enough to stay fast


def _records():
    return generate(SEED, N)


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------

def test_records_validate_clean():
    records = _records()
    violations = [v for r in records for v in validate_record(r)]
    assert violations == []


def test_dict_round_trip_is_lossless():
    records = _records()
    for rec in records:
        assert record_from_dict(record_to_dict(rec)) == rec


def test_jsonl_round_trip_is_lossless(tmp_path):
    records = _records()
    path = tmp_path / "diaries.jsonl"
    n = write_jsonl(records, path)
    assert n == len(records)
    back = list(read_jsonl(path))
    assert back == records


def test_jsonl_header_line_declares_dev_status(tmp_path):
    path = tmp_path / "diaries.jsonl"
    write_jsonl(_records(), path)
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["record_type"] == "meta"
    assert first["dev_only"] is True
    assert first["synthetic"] is True
    assert first["schema_version"] == SCHEMA_VERSION
    assert "never cited" in first["note"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_same_records():
    assert generate(SEED, N) == generate(SEED, N)


def test_same_seed_byte_identical_file(tmp_path):
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(generate(SEED, N), p1)
    write_jsonl(generate(SEED, N), p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_different_seed_differs():
    assert generate(SEED, N) != generate(SEED + 1, N)


# ---------------------------------------------------------------------------
# Household atomicity (E1 splits are household-atomic)
# ---------------------------------------------------------------------------

def test_household_atomicity():
    records = _records()
    assert household_atomicity_violations(records) == []
    # Every individual carries its household's id.
    for r in records:
        assert r.individual.household_id == r.household.household_id


def test_multi_person_households_exist_and_share_household():
    """The atomicity guarantee is vacuous unless some households actually
    have several members in the sample."""
    records = _records()
    by_hh = {}
    for r in records:
        by_hh.setdefault(r.household.household_id, set()).add(r.individual.person_id)
    assert any(len(members) >= 2 for members in by_hh.values())


def test_split_is_household_atomic_and_deterministic():
    records = _records()
    split_by_hh = {}
    for r in records:
        hid = r.household.household_id
        s = household_split(hid)
        assert s in ("train", "holdout")
        assert split_by_hh.setdefault(hid, s) == s  # stable per household
        # keyed on household only: every member gets the household's split
        assert household_split(r.individual.household_id) == s
    # Both sides of the split are populated and the ratio is roughly 80/20.
    n_hh = len(split_by_hh)
    holdout = sum(1 for s in split_by_hh.values() if s == "holdout")
    assert 0.05 < holdout / n_hh < 0.40


# ---------------------------------------------------------------------------
# Marginal sanity
# ---------------------------------------------------------------------------

def test_trips_per_day_in_sane_band():
    stats = summarize(_records())
    assert 1.5 <= stats["trips_per_person_day"] <= 4.0


def test_all_four_modes_present():
    stats = summarize(_records())
    assert set(stats["mode_shares"]) == set(MODES)
    for share in stats["mode_shares"].values():
        assert share > 0.0


def test_synthetic_flag_on_every_record():
    for r in _records():
        assert r.synthetic is True
        assert r.household.synthetic is True
        assert r.individual.synthetic is True
        assert r.schema_version == SCHEMA_VERSION


def test_departure_peaks_exist():
    """AM (07-09) and PM (16-18) two-hour bands must both be clearly denser
    than a flat departure profile would make them (2h of an ~18h travel day
    ~= 11%)."""
    stats = summarize(_records())
    assert stats["share_departures_am_peak_0709"] > 0.13
    assert stats["share_departures_pm_peak_1618"] > 0.13


def test_zone_codes_only_no_place_names():
    """Contamination masking: agent-facing geography is zone codes only."""
    records = _records()
    for r in records:
        zones = [r.household.home_zone, r.individual.home_zone]
        if r.individual.work_zone:
            zones.append(r.individual.work_zone)
        for t in r.trips:
            zones += [t.origin_zone, t.destination_zone]
        for z in zones:
            assert len(z) == 3 and z.startswith("Z") and z[1:].isdigit()


def test_cordon_structure():
    """Some, but not all, trips cross the inner-ring cordon; the flag agrees
    with the ring taxonomy."""
    records = _records()
    crossing = total = 0
    for r in records:
        for t in r.trips:
            total += 1
            x = crosses_cordon(t.origin_zone, t.destination_zone)
            assert x == (is_within_cordon(t.origin_zone) != is_within_cordon(t.destination_zone))
            crossing += x
    assert 0 < crossing < total


def test_car_ownership_correlates_with_outer_ring():
    """The documented correlation direction: outer-ring households own cars
    more often than inner-ring households."""
    records = _records()
    by_hh = {r.household.household_id: r.household for r in records}
    def rate(ring):
        hhs = [h for h in by_hh.values() if zone_ring(h.home_zone) == ring]
        assert hhs, f"no households in ring {ring}"
        return sum(1 for h in hhs if h.household_cars > 0) / len(hhs)
    assert rate("outer") > rate("inner")


# ---------------------------------------------------------------------------
# Persistent heterogeneity (the E2 mechanism)
# ---------------------------------------------------------------------------

def test_person_level_mode_choice_is_overdispersed():
    """Persistent person-level propensities must make individuals sticky in
    their mode use: the variance of per-person car shares must exceed what
    iid per-trip draws at the pooled car share would produce. If someone
    replaces the latents with per-trip noise, this test fails."""
    records = generate(SEED, 400)
    per_person = []  # (n_trips, car_share)
    car_total = trip_total = 0
    for r in records:
        n = len(r.trips)
        c = sum(1 for t in r.trips if t.main_mode == "car")
        per_person.append((n, c / n))
        car_total += c
        trip_total += n
    p = car_total / trip_total
    shares = [s for _, s in per_person]
    mean_share = sum(shares) / len(shares)
    observed_var = sum((s - mean_share) ** 2 for s in shares) / len(shares)
    # iid baseline: Var(share_i) = p(1-p)/n_i, averaged over persons.
    iid_var = sum(p * (1 - p) / n for n, _ in per_person) / len(per_person)
    assert observed_var > 1.5 * iid_var, (
        f"per-person car shares look iid (observed {observed_var:.4f} vs "
        f"iid baseline {iid_var:.4f}); persistent latents are not reaching choices"
    )


def test_trip_counts_vary_between_persons():
    """Mobility latent must spread trip counts across persons (not everyone
    makes the same number of trips)."""
    counts = [len(r.trips) for r in _records()]
    assert len(set(counts)) >= 3


# ---------------------------------------------------------------------------
# The committed sample file (data/synthetic/diaries_dev.jsonl)
# ---------------------------------------------------------------------------

_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "diaries_dev.jsonl"


def test_committed_sample_exists_under_1mb_and_is_all_synthetic():
    assert _SAMPLE.exists(), "run: python grounding/synthetic_diary.py --seed 42 --n-individuals 500 --out data/synthetic/diaries_dev.jsonl"
    assert _SAMPLE.stat().st_size < 1_000_000
    lines = _SAMPLE.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["record_type"] == "meta" and header["dev_only"] is True
    records = list(read_jsonl(_SAMPLE))
    assert len(records) == len(lines) - 1
    assert all(r.synthetic for r in records)  # 100% of records
    assert household_atomicity_violations(records) == []


def test_committed_sample_matches_seed_42_regeneration():
    """The committed file must be exactly what seed 42 / 500 individuals
    produces — no hand edits."""
    regenerated = generate(42, 500)
    on_disk = list(read_jsonl(_SAMPLE))
    assert on_disk == regenerated
