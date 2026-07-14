"""Frozen-taxonomy consistency (M0 owner decision, pre-registration §5).

The five-mode vocabulary and its ORDER are sealed: the order is the
temperature-0 tie-break order, so serving, agents and grounding must all
agree on one tuple. The E1 protected-segment definitions (income band x
car ownership x residence band) are sealed alongside. If any of these
tests fail after M0, someone edited a frozen surface — that requires a
dated pre-registration §7 amendment, not a code fix.
"""
from agents.logit_chooser import MODES_ALL
from grounding.adapters.rvu_schema import MODES as SCHEMA_MODES
from grounding.taxonomy import (
    CATCHMENT_RINGS,
    KNOWN_MODE_CLASSES,
    MODES,
    SEGMENT_CELLS,
    car_band,
    collapse_mode,
    income_band,
    residence_band,
    segment_cell,
)
from serving.gateway import DEFAULT_MODES_ORDER


# ---------------------------------------------------------------------------
# The frozen mode tuple is single-sourced everywhere
# ---------------------------------------------------------------------------

def test_mode_tuple_is_frozen_and_single_sourced():
    assert MODES == ("walk", "transit", "ride", "car", "bike")
    assert tuple(DEFAULT_MODES_ORDER) == MODES
    assert tuple(MODES_ALL) == MODES
    assert tuple(SCHEMA_MODES) == MODES


# ---------------------------------------------------------------------------
# Survey collapse rules (harness-side)
# ---------------------------------------------------------------------------

def test_sole_driver_is_car():
    assert collapse_mode("Drive SOV") == "car"
    # ...even for someone who reports not driving (data noise): SOV wins.
    assert collapse_mode("Drive SOV", can_drive=False) == "car"


def test_shared_drive_splits_on_driver_flag():
    assert collapse_mode("Drive HOV2", driver="Driver") == "car"
    assert collapse_mode("Drive HOV2", driver="Passenger") == "ride"
    assert collapse_mode("Drive HOV3+", driver="Passenger") == "ride"
    # switched drivers mid-trip counts as driving
    assert collapse_mode("Drive HOV3+", driver="Both (switched drivers)") == "car"


def test_shared_drive_missing_flag_falls_back_on_licence():
    assert collapse_mode("Drive HOV2", driver=None, can_drive=True) == "car"
    assert collapse_mode("Drive HOV2", driver=None, can_drive=False) == "ride"


def test_scheduled_shared_services_are_transit():
    assert collapse_mode("Transit") == "transit"
    assert collapse_mode("School Bus") == "transit"


def test_hired_rides_and_micro_vehicles():
    assert collapse_mode("Ride Hail") == "ride"
    assert collapse_mode("Micromobility") == "bike"
    assert collapse_mode("Walk") == "walk"
    assert collapse_mode("Bike") == "bike"


def test_homeless_classes_drop_to_none():
    assert collapse_mode("Other") is None
    assert collapse_mode("Missing Response") is None


def test_unknown_labels_are_not_silently_known():
    # Adapters must consult KNOWN_MODE_CLASSES and fail loud on drift.
    assert "Hoverboard Deluxe" not in KNOWN_MODE_CLASSES
    assert collapse_mode("Hoverboard Deluxe") is None


# ---------------------------------------------------------------------------
# E1 protected segments
# ---------------------------------------------------------------------------

def test_twelve_protected_cells():
    assert len(SEGMENT_CELLS) == 12
    assert len(set(SEGMENT_CELLS)) == 12


def test_income_banding_is_the_frozen_grouping():
    assert [income_band(c) for c in (1, 2, 3, 4, 5)] == [
        "low", "low", "mid", "mid", "high"]


def test_car_band_is_binary():
    assert car_band(0) == "car0"
    assert car_band(1) == car_band(3) == "car1p"


def test_residence_band_covers_both_zone_taxonomies():
    # grounding's current ring names and the City K world's ring names
    # must both resolve; only membership of CATCHMENT_RINGS is frozen.
    assert "inner" in CATCHMENT_RINGS and "core" in CATCHMENT_RINGS
    assert residence_band("inner") == "catchment"
    assert residence_band("core") == "catchment"
    assert residence_band("inner_suburb") == "remainder"
    assert residence_band("outer") == "remainder"


def test_segment_cell_id_shape():
    assert segment_cell(1, 0, "inner") == "low|car0|catchment"
    assert segment_cell(5, 2, "outer") == "high|car1p|remainder"
    assert segment_cell(3, 1, "core") in SEGMENT_CELLS
