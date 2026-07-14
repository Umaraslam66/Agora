#!/usr/bin/env python3
"""M0 synthetic travel-diary generator — the DEV stand-in for Trafa RVU/RES
microdata (01_PREREGISTRATION.md §6: "results on stand-in data are labeled
DEV and never cited").

WHAT THIS IS: a seeded generator of one-person-day diary records in the
canonical schema of `grounding/adapters/rvu_schema.py` — a synthetic
population for a generic large Nordic metro. It exists so the full pipeline
(seeding -> personas -> E1/E2 scoring, household-atomic splits, JSONL I/O)
can be built and tested honestly before real microdata is licensed. It is
NOT a calibrated population model: every numeric target below is a DEV
placeholder chosen to be plausible, not fitted to anything.

WHAT THIS IS NOT: real data, a model of the real anchor city, or a
calibration target.
No real place names, district names, or real calendar dates appear in any
field — zones are codes Z01-Z25 (three rings: inner / inner_suburb /
outer, with a cordon between `inner` and the rest) and days are relative
indices 0-6. Every record carries `synthetic: True` and the output file
starts with a DEV-status header line.

DESIGN: PERSISTENT PERSON-LEVEL HETEROGENEITY, NOT IID NOISE.
E2 (variance preservation) tests whether the simulated population keeps
real between-INDIVIDUAL spread. A generator whose randomness is all
per-trip iid noise would produce individuals who are exchangeable given
their demographics — between-agent variance would collapse toward the
segment mean, and E2's spread ratio would be meaningless as a pipeline
check. So each individual gets latent, persistent propensities drawn ONCE
at creation and reused for every choice that person makes:

    mobility_propensity   ~ Gamma-ish positive multiplier on trip count
                            (some people just travel more, every day)
    car_affinity          ~ persistent taste shift toward car vs pt
    active_affinity       ~ persistent taste shift toward walk/bike
    schedule_offset_min   ~ persistent personal clock shift (early birds /
                            late risers), applied to every departure time

These latents correlate with structure (car_affinity rises with car
ownership, licence, and outer-ring residence; mobility falls at high age)
but retain independent variance, so two demographically identical agents
still differ persistently — which is exactly the property E2 measures.

CORRELATION STRUCTURE (matters more than exact levels, per M0 scope):
  - car ownership increases with income class and with ring
    (outer > inner_suburb > inner) — set in _household();
  - mode choice depends on car availability per adult, distance band,
    ring, and the persistent latents — set in _choose_mode();
  - work/education trips dominate the AM peak; PM peak is the mirror
    (return-home) plus evening leisure — set in the purpose sequencing
    of _make_trips() and the per-purpose departure-time distributions;
  - workplaces concentrate toward the center (inner ring has the largest
    workplace weight), so outer-ring car commuters cross the cordon —
    giving M4's toll shock a real population to bite on.

DEV MARGINAL TARGETS (placeholders — NOT calibration; do not tune agents
against them; real bars are frozen at end of M0 per the pre-registration):
see the TARGET_* constants below. Rough intent: ~2.4-2.8 trips/person-day,
mode shares plausible for a large Nordic metro (car ~40-50%, pt ~20-30%,
walk ~20-25%, bike ~5-10%), AM peak 07-09, PM peak 16-18.

Usage:
    python grounding/synthetic_diary.py --seed 42 --n-individuals 500 \
        --out data/synthetic/diaries_dev.jsonl

Determinism: same (--seed, --n-individuals) -> byte-identical output file.
Stdlib only (random, json, argparse) — no numpy dependency here, so the
generator stays runnable anywhere the repo checks out.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running both as a module and as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from grounding.adapters.rvu_schema import (  # noqa: E402
    SCHEMA_VERSION,
    RING_ZONES,
    DiaryRecord,
    Household,
    Individual,
    Trip,
    validate_record,
    write_jsonl,
    zone_distance_km,
    zone_ring,
)

# ---------------------------------------------------------------------------
# DEV MARGINAL TARGETS — placeholders, NOT calibration (see module docstring)
# ---------------------------------------------------------------------------

TARGET_TRIPS_PER_DAY = (2.4, 2.8)      # DEV placeholder band, person-day
TARGET_MODE_SHARES = {                  # DEV placeholders, large Nordic metro
    "car": 0.40,                        # (high-pt metro: car well under half)
    "transit": 0.35,
    "walk": 0.18,
    "bike": 0.07,
}
TARGET_AM_PEAK = ("07:00", "09:00")    # DEV placeholder: work departures cluster here
TARGET_PM_PEAK = ("16:00", "18:00")    # DEV placeholder: return-home departures

# Ring population weights (DEV placeholders): most residents live outside
# the inner ring, as in any large metro.
RING_POPULATION_WEIGHTS = {"inner": 0.22, "inner_suburb": 0.44, "outer": 0.34}

# Workplace attraction weights (DEV placeholders): jobs concentrate toward
# the center — this is what routes outer-ring commuters across the cordon.
RING_WORKPLACE_WEIGHTS = {"inner": 0.52, "inner_suburb": 0.33, "outer": 0.15}

# P(household has >= 1 car | ring, income_class), DEV placeholders.
# Correlation direction is the point: outer ring and higher income both
# raise ownership; the inner ring stays car-light even when rich.
CAR_OWNERSHIP_BASE = {"inner": 0.35, "inner_suburb": 0.66, "outer": 0.85}
CAR_OWNERSHIP_INCOME_SLOPE = 0.06      # added per income class above class 3

# Second-car probability given a first car (DEV placeholder).
SECOND_CAR_P = {"inner": 0.06, "inner_suburb": 0.16, "outer": 0.30}

# Per-purpose departure-time mixture (minutes since midnight): each entry is
# (mean, sd, weight). Work/education pull the AM peak; return-home pulls the
# PM peak. DEV placeholders.
DEPART_MIX: Dict[str, List[Tuple[int, int, float]]] = {
    "work": [(465, 40, 0.8), (540, 60, 0.2)],           # ~07:45 dominant
    "education": [(475, 30, 0.9), (720, 90, 0.1)],      # ~07:55
    "home": [(1020, 55, 0.7), (780, 90, 0.3)],          # ~17:00 dominant
    "shop_daily": [(1050, 90, 0.6), (660, 100, 0.4)],
    "shop_other": [(750, 130, 1.0)],
    "leisure": [(1110, 80, 0.7), (840, 120, 0.3)],
    "personal_business": [(660, 120, 1.0)],
    "pickup_dropoff": [(480, 45, 0.5), (990, 45, 0.5)],
    "other": [(780, 150, 1.0)],
}

# Mode speeds (km/h) used only to derive arrive_time from distance.
MODE_SPEED_KMH = {"car": 30.0, "transit": 22.0, "walk": 4.8, "bike": 14.0}

GENERATOR_VERSION = "synthetic-diary-m0-0.1"


# ---------------------------------------------------------------------------
# Latent person-level propensities (the E2 heterogeneity mechanism)
# ---------------------------------------------------------------------------

class PersonLatents:
    """Drawn once per individual; reused for every trip. See module
    docstring, "PERSISTENT PERSON-LEVEL HETEROGENEITY"."""

    __slots__ = ("mobility", "car_affinity", "active_affinity", "clock_offset_min")

    def __init__(self, mobility: float, car_affinity: float,
                 active_affinity: float, clock_offset_min: int) -> None:
        self.mobility = mobility
        self.car_affinity = car_affinity
        self.active_affinity = active_affinity
        self.clock_offset_min = clock_offset_min


def _draw_latents(rng: random.Random, ind: Individual, hh: Household) -> PersonLatents:
    # Positive multiplicative mobility propensity, lognormal-ish, mean ~1.
    mobility = rng.lognormvariate(0.0, 0.35)
    if ind.age >= 70:
        mobility *= 0.75
    if not ind.employed and not ind.student:
        mobility *= 0.85

    # Car taste: structural push from ownership/licence/ring + own variance.
    car_affinity = rng.gauss(0.0, 0.8)
    if hh.household_cars > 0 and ind.driving_licence:
        car_affinity += 0.9
    if hh.household_cars >= 2:
        car_affinity += 0.4
    if zone_ring(hh.home_zone) == "outer":
        car_affinity += 0.5
    elif zone_ring(hh.home_zone) == "inner":
        car_affinity -= 0.6

    # Active-mode taste: falls with age, has its own persistent variance.
    active_affinity = rng.gauss(0.0, 0.7)
    if ind.age < 35:
        active_affinity += 0.3
    if ind.age >= 65:
        active_affinity -= 0.5

    # Personal clock: persistent early/late shift applied to every departure.
    clock_offset_min = int(rng.gauss(0.0, 25.0))

    return PersonLatents(mobility, car_affinity, active_affinity, clock_offset_min)


# ---------------------------------------------------------------------------
# Population synthesis
# ---------------------------------------------------------------------------

def _weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _pick_zone(rng: random.Random, ring_weights: Dict[str, float]) -> str:
    ring = _weighted_choice(rng, ring_weights)
    return rng.choice(RING_ZONES[ring])


_ALL_ZONES: List[str] = [z for zones in RING_ZONES.values() for z in zones]


def _gravity_zone(rng: random.Random, from_zone: str,
                  ring_attraction: Dict[str, float], beta_km: float) -> str:
    """Distance-decayed destination sampling: P(z) ∝ attraction(ring(z)) *
    exp(-d(from,z)/beta). Without the decay, "same ring" destinations can be
    diametrically opposite (30+ km) and the trip-length distribution loses
    its short-trip mass entirely — which is what real diaries are made of."""
    weights = []
    for z in _ALL_ZONES:
        d = zone_distance_km(from_zone, z)
        weights.append(ring_attraction[zone_ring(z)]
                       * pow(2.718281828, -d / beta_km))
    return rng.choices(_ALL_ZONES, weights=weights, k=1)[0]


def _household(rng: random.Random, hh_id: str) -> Household:
    home_zone = _pick_zone(rng, RING_POPULATION_WEIGHTS)
    ring = zone_ring(home_zone)

    household_type = _weighted_choice(rng, {
        "single": 0.36,
        "couple_no_children": 0.26,
        "couple_with_children": 0.22,
        "single_parent": 0.08,
        "other": 0.08,
    })
    size = {
        "single": 1,
        "couple_no_children": 2,
        "couple_with_children": rng.choice([3, 4, 4, 5]),
        "single_parent": rng.choice([2, 3]),
        "other": rng.choice([2, 3]),
    }[household_type]

    income_class = rng.choices([1, 2, 3, 4, 5], weights=[0.15, 0.22, 0.28, 0.22, 0.13])[0]

    p_car = CAR_OWNERSHIP_BASE[ring] + CAR_OWNERSHIP_INCOME_SLOPE * (income_class - 3)
    cars = 0
    if rng.random() < max(0.02, min(0.95, p_car)):
        cars = 1
        if size >= 2 and rng.random() < SECOND_CAR_P[ring] + 0.03 * (income_class - 3):
            cars = 2

    return Household(
        household_id=hh_id,
        household_size=size,
        household_type=household_type,
        household_cars=cars,
        home_zone=home_zone,
        income_class=income_class,
    )


def _adult_ages(rng: random.Random, household_type: str) -> List[int]:
    if household_type == "single":
        return [rng.randint(20, 84)]
    if household_type == "couple_no_children":
        a = rng.randint(24, 82)
        return [a, max(20, min(84, a + rng.randint(-6, 6)))]
    if household_type == "couple_with_children":
        a = rng.randint(30, 55)
        return [a, max(25, min(60, a + rng.randint(-5, 5)))]
    if household_type == "single_parent":
        return [rng.randint(26, 55)]
    return [rng.randint(20, 80), rng.randint(20, 80)]


def _individuals(rng: random.Random, hh: Household) -> List[Individual]:
    """Adults of the household (children under 16 are skipped at M0 — the
    seeding target is decision-making agents; a real adapter can widen
    coverage to RVU's 6-84 span later)."""
    ages = _adult_ages(rng, hh.household_type)
    people: List[Individual] = []
    for i, age in enumerate(ages):
        employed = False
        student = False
        if age < 26 and rng.random() < 0.42:
            student = True
        elif age <= 66:
            employed = rng.random() < 0.80
        elif age <= 70:
            employed = rng.random() < 0.20

        licence = age >= 18 and rng.random() < (
            0.62 if zone_ring(hh.home_zone) == "inner" else
            0.80 if zone_ring(hh.home_zone) == "inner_suburb" else 0.90
        )

        work_zone: Optional[str] = None
        if employed:
            # Center-weighted attraction with a long commute decay length.
            work_zone = _gravity_zone(rng, hh.home_zone, RING_WORKPLACE_WEIGHTS,
                                      beta_km=14.0)
        elif student:
            # Education places: mildly center-weighted, shorter decay.
            work_zone = _gravity_zone(
                rng, hh.home_zone,
                {"inner": 0.40, "inner_suburb": 0.40, "outer": 0.20},
                beta_km=9.0)

        people.append(Individual(
            person_id=f"{hh.household_id}_P{i + 1}",
            household_id=hh.household_id,
            age=age,
            sex=rng.choice(["m", "f"]),
            driving_licence=licence,
            employed=employed,
            student=student,
            home_zone=hh.home_zone,
            work_zone=work_zone,
            person_weight=1.0,  # DEV: no sampling design; see rvu_schema TODO 4
        ))
    return people


# ---------------------------------------------------------------------------
# Trip generation
# ---------------------------------------------------------------------------

def _depart_minutes(rng: random.Random, purpose: str, latents: PersonLatents,
                    not_before: int) -> int:
    mean, sd, _ = rng.choices(
        DEPART_MIX[purpose], weights=[w for _, _, w in DEPART_MIX[purpose]], k=1)[0]
    m = int(rng.gauss(mean, sd)) + latents.clock_offset_min
    m = max(m, not_before)
    return max(5 * 60, min(23 * 60 + 30, m))


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _choose_mode(rng: random.Random, ind: Individual, hh: Household,
                 latents: PersonLatents, distance_km: float,
                 car_taken_by_other: bool) -> str:
    """Soft-max over persistent-taste-shifted utilities. DEV placeholder
    utilities: shaped for correlation direction (distance bands, car access,
    ring), not fitted levels."""
    cars_free = hh.household_cars > 0 and ind.driving_licence and not car_taken_by_other

    u = {"walk": 1.6, "bike": -0.1, "transit": -0.9, "car": 0.2}

    # Distance shaping: walk owns the short band, pt/car need length to pay
    # off, pt is unattractive for very short hops (access time dominates).
    u["walk"] -= 1.1 * max(0.0, distance_km - 1.2)
    u["bike"] -= 0.35 * max(0.0, distance_km - 2.5)
    u["transit"] += 0.13 * min(distance_km, 14.0)
    u["transit"] -= 0.9 * max(0.0, 2.0 - distance_km)
    u["car"] += 0.26 * min(distance_km, 18.0)

    # Persistent tastes (the E2 mechanism).
    u["car"] += latents.car_affinity
    u["walk"] += 0.7 * latents.active_affinity
    u["bike"] += latents.active_affinity

    # Ring/context shaping: inner-ring pt is good, outer-ring pt is sparse.
    ring = zone_ring(hh.home_zone)
    if ring == "inner":
        u["transit"] += 0.5
        u["car"] -= 0.4
    elif ring == "outer":
        u["transit"] -= 0.4

    if not cars_free:
        u["car"] -= 6.0  # near-hard availability constraint
    if ind.age >= 75:
        u["bike"] -= 1.2

    mx = max(u.values())
    exps = {m: pow(2.718281828, v - mx) for m, v in u.items()}
    total = sum(exps.values())
    r = rng.random() * total
    acc = 0.0
    for m, e in exps.items():
        acc += e
        if r <= acc:
            return m
    return "transit"


def _day_purposes(rng: random.Random, ind: Individual,
                  latents: PersonLatents) -> List[str]:
    """Ordered out-of-home purpose sequence for the day (each entry becomes
    one outbound trip; the return-home trip is appended automatically).
    Trip count scales with the persistent mobility latent."""
    purposes: List[str] = []
    if ind.employed:
        purposes.append("work")
    elif ind.student:
        purposes.append("education")

    # Extra discretionary stops: expected count scales with mobility latent.
    extra_menu = ["shop_daily", "shop_daily", "leisure", "personal_business",
                  "shop_other", "pickup_dropoff", "other"]
    lam = 0.9 * latents.mobility if purposes else 1.3 * latents.mobility
    n_extra = 0
    # Cheap Poisson-ish draw without numpy.
    p = rng.random()
    threshold = pow(2.718281828, -lam)
    while p > threshold and n_extra < 4:
        n_extra += 1
        p *= rng.random()
    for _ in range(n_extra):
        purposes.append(rng.choice(extra_menu))

    if not purposes and rng.random() < 0.75:
        purposes.append(rng.choice(["shop_daily", "leisure", "personal_business"]))

    return purposes


def _destination_zone(rng: random.Random, ind: Individual, purpose: str,
                      current_zone: str) -> str:
    if purpose == "work" and ind.work_zone:
        return ind.work_zone
    if purpose == "education" and ind.work_zone:
        return ind.work_zone
    if purpose in ("shop_daily", "pickup_dropoff"):
        # Local: mostly own zone, else strongly distance-decayed.
        if rng.random() < 0.60:
            return current_zone
        return _gravity_zone(rng, current_zone,
                             {"inner": 1.0, "inner_suburb": 1.0, "outer": 1.0},
                             beta_km=3.0)
    # Other discretionary: mildly center-weighted, moderate decay.
    return _gravity_zone(rng, current_zone,
                         {"inner": 1.6, "inner_suburb": 1.2, "outer": 0.8},
                         beta_km=6.0)


def _make_trips(rng: random.Random, ind: Individual, hh: Household,
                latents: PersonLatents, car_taken_by_other: bool) -> List[Trip]:
    purposes = _day_purposes(rng, ind, latents)
    if not purposes:
        return []

    trips: List[Trip] = []
    current_zone = ind.home_zone
    clock = 0
    for purpose in purposes:
        dest = _destination_zone(rng, ind, purpose, current_zone)
        depart = _depart_minutes(rng, purpose, latents, not_before=clock + 10)
        dist = max(0.4, zone_distance_km(current_zone, dest) * rng.uniform(0.85, 1.25))
        dist = round(dist, 1)
        mode = _choose_mode(rng, ind, hh, latents, dist, car_taken_by_other)
        travel_min = max(4, int(dist / MODE_SPEED_KMH[mode] * 60))
        arrive = min(23 * 60 + 55, depart + travel_min)
        trips.append(Trip(
            trip_index=len(trips),
            purpose=purpose,
            main_mode=mode,
            origin_zone=current_zone,
            destination_zone=dest,
            depart_time=_fmt(depart),
            arrive_time=_fmt(arrive),
            distance_km=dist,
        ))
        current_zone = dest
        # Activity dwell before the next departure.
        clock = arrive + (
            rng.randint(300, 540) if purpose in ("work", "education")
            else rng.randint(25, 150)
        )

    # Return home.
    if current_zone != ind.home_zone or trips:
        dest = ind.home_zone
        depart = _depart_minutes(rng, "home", latents, not_before=clock)
        dist = max(0.4, zone_distance_km(current_zone, dest) * rng.uniform(0.85, 1.25))
        dist = round(dist, 1)
        mode = _choose_mode(rng, ind, hh, latents, dist, car_taken_by_other)
        travel_min = max(4, int(dist / MODE_SPEED_KMH[mode] * 60))
        arrive = min(23 * 60 + 59, depart + travel_min)
        trips.append(Trip(
            trip_index=len(trips),
            purpose="home",
            main_mode=mode,
            origin_zone=current_zone,
            destination_zone=dest,
            depart_time=_fmt(depart),
            arrive_time=_fmt(arrive),
            distance_km=dist,
        ))
    return trips


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------

def generate(seed: int, n_individuals: int) -> List[DiaryRecord]:
    """Generates diary records for ~n_individuals individuals (grouped into
    households; the last household is always completed, so the exact count
    may overshoot by one household member). Deterministic in (seed,
    n_individuals)."""
    rng = random.Random(seed)
    records: List[DiaryRecord] = []
    n_people = 0
    hh_serial = 0

    while n_people < n_individuals:
        hh_serial += 1
        hh = _household(rng, f"H{hh_serial:05d}")
        people = _individuals(rng, hh)

        # Within-household car contention: with 1 car and 2+ licensed
        # adults, at most one gets it for the day (crude but directional).
        licensed = [p for p in people if p.driving_licence]
        car_winner: Optional[str] = None
        if hh.household_cars == 1 and len(licensed) >= 2:
            car_winner = rng.choice(licensed).person_id

        day_index = rng.randint(0, 6)
        day_type = "weekend" if day_index >= 5 else "weekday"

        for ind in people:
            latents = _draw_latents(rng, ind, hh)
            car_taken = car_winner is not None and ind.person_id != car_winner
            trips = _make_trips(rng, ind, hh, latents, car_taken_by_other=car_taken)
            if not trips:
                continue  # zero-trip days exist in RVU too, but E1 seeds need trips
            records.append(DiaryRecord(
                record_id=f"{ind.person_id}_D{day_index}",
                household=hh,
                individual=ind,
                trips=trips,
                day_index=day_index,
                day_type=day_type,
            ))
            n_people += 1

    return records


def summarize(records: List[DiaryRecord]) -> dict:
    """Marginals of a generated sample — for eyeballing and for the README.
    These are DESCRIPTIVE of the DEV sample, never calibration targets."""
    n = len(records)
    trips = [t for r in records for t in r.trips]
    mode_counts: Dict[str, int] = {}
    am = pm = 0
    cordon_crossings = 0
    for t in trips:
        mode_counts[t.main_mode] = mode_counts.get(t.main_mode, 0) + 1
        hh, mm = t.depart_time.split(":")
        minutes = int(hh) * 60 + int(mm)
        if 7 * 60 <= minutes < 9 * 60:
            am += 1
        if 16 * 60 <= minutes < 18 * 60:
            pm += 1
        from grounding.adapters.rvu_schema import crosses_cordon
        if crosses_cordon(t.origin_zone, t.destination_zone):
            cordon_crossings += 1

    n_trips = len(trips)
    households = {r.household.household_id for r in records}
    car_hh = {r.household.household_id for r in records if r.household.household_cars > 0}
    return {
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "individuals": n,
        "households": len(households),
        "trips": n_trips,
        "trips_per_person_day": round(n_trips / n, 3) if n else 0.0,
        "mode_shares": {m: round(c / n_trips, 4) for m, c in sorted(mode_counts.items())},
        "share_departures_am_peak_0709": round(am / n_trips, 4),
        "share_departures_pm_peak_1618": round(pm / n_trips, 4),
        "share_trips_crossing_cordon": round(cordon_crossings / n_trips, 4),
        "share_households_with_car": round(len(car_hh) / len(households), 4),
        "synthetic": True,
        "dev_only": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-individuals", type=int, required=True)
    ap.add_argument("--out", required=True, help="output JSONL path")
    args = ap.parse_args(argv)

    records = generate(args.seed, args.n_individuals)

    violations: List[str] = []
    for r in records:
        violations.extend(validate_record(r))
    if violations:
        print(f"SCHEMA VIOLATIONS ({len(violations)}):", file=sys.stderr)
        for v in violations[:20]:
            print("  " + v, file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(records, out)

    stats = summarize(records)
    stats["seed"] = args.seed
    stats["records_written"] = n
    stats["out"] = str(out)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
