"""Shared fixtures for the Agora day-one test suite.

Everything here is MASKED sample data: zone codes, City K, generic occupations,
perturbed currency units. Nothing in this file (or any fixture) may name the
real anchoring natural experiment — see grounding/masking/forbidden_tokens.txt.
"""

from __future__ import annotations

import pytest

REPO_ROOT_HINT = "run pytest from the repository root (pyproject sets pythonpath=['.'])"


@pytest.fixture
def sample_persona() -> dict:
    """A masked persona seeded the way a real diary record would be."""
    return {
        "persona_id": "PK-000173",
        "home_zone": "Z07",
        "work_zone": "Z02",
        "household_size": 3,
        "occupation": "clinic receptionist",
        "income_band": "B3",
        "car_owner": True,
        # per-rule habit-strength counters: days each standing rule has been
        # followed (the memory substrate tested by E6)
        "habits": {
            "commute_by_car": 412,
            "friday_grocery_run_z05": 96,
            "summer_cycle_commute": 38,
        },
        "rules": [
            "Drive to work in Z02 on weekdays, leaving between 07:15 and 07:40.",
            "Do the weekly grocery run to Z05 on Friday after work.",
            "Cycle to work when the season allows and no child drop-off is needed.",
        ],
    }


@pytest.fixture
def sample_world_state() -> dict:
    """A masked world snapshot: City K, zone-pair times, cordon flag, fee."""
    return {
        "city": "City K",
        "day_index": 118,
        "toll_cordon_active": True,
        "toll_fee_units": 22,
        "currency": "credits",
        # zone-pair travel times in minutes, per mode
        "travel_times_min": {
            "Z07->Z02": {"car": 26, "transit": 34, "bike": 41},
            "Z07->Z05": {"car": 14, "transit": 22, "bike": 19},
            "Z02->Z07": {"car": 29, "transit": 35, "bike": 42},
        },
    }


@pytest.fixture
def sample_options() -> list:
    """Masked discrete-choice options for a choice gateway query."""
    return [
        {"id": "CAR", "label": "drive (crosses the toll cordon)"},
        {"id": "TRANSIT", "label": "take transit"},
        {"id": "BIKE", "label": "cycle"},
        {"id": "STAY", "label": "skip the trip today"},
    ]
