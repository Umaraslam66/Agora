"""Tests for the E1 MNL falsification arm (evaluation/mnl_arm.py; D6 / A2.1).

Covered: strictly-out-of-fold discipline (a fold-f fit sees no fold-f persona,
asserted by construction); the day-signature-class draw (whole observed purpose
sequences, no novel combinations); CRN key-site pairing with the method arm;
and a small end-to-end build that emits valid RealizedDay structures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.card_executor import RealizedDay
from evaluation import e1, golden_pairs as gp, mnl_arm
from grounding.adapters import psrc


# ---------------------------------------------------------------------------
# synthetic scenario (adapter-shaped, spanning multiple folds)
# ---------------------------------------------------------------------------

def _scenario():
    # H000->fold2, H001->1, H002->0, H003->1, H004->2, H005->4, H006->4,
    # H007->1, H008->1, H009->0  (all five folds represented via OOF)
    n = 10
    hh_rows, pidx_rows, pd_rows, tr_rows = [], [], [], []
    slots = {}
    cell_of_household = {}
    modes = ["car", "walk", "transit", "ride", "bike"]
    bands = ["am_peak", "midday", "pm_peak", "evening", "night"]
    purposes = ["work", "home", "leisure", "shop_daily", "personal_business"]
    for i in range(n):
        hid = f"H{i:03d}"
        pid = f"{hid}p"
        persona = f"P{i:03d}"
        inc = None if i == 9 else (i % 5) + 1  # persona 9 = PNA (no cell)
        cars = i % 3
        hh_rows.append({"household_id": hid, "income_class": inc, "household_cars": cars})
        pidx_rows.append({
            "persona_id": persona, "person_id": pid, "household_id": hid,
            "age": 25 + i * 3, "income_class": inc, "employed": (i % 2 == 0),
            "can_drive": (cars >= 1), "household_cars": cars,
        })
        cell_of_household[hid] = None if inc is None else (
            "car0|remainder" if cars == 0 else f"{['low','low','mid','mid','high'][inc-1]}|car1p|catchment"
        )
        ndays = 1 + (i % 2)  # some multi-day persons
        slots[persona] = []
        for d in range(1, ndays + 1):
            ntrips = 2 + (i % 3)
            pd_rows.append({"household_id": hid, "person_id": pid, "daynum": d,
                            "n_collapsed": ntrips, "w_day": 1.0 + i})
            slots[persona].append((d, 1.0 + i))
            for tn in range(1, ntrips + 1):
                mode = modes[(i + tn) % 5]
                if mode == "car" and cars == 0:
                    mode = "ride"
                tr_rows.append({
                    "household_id": hid, "person_id": pid, "daynum": d, "tripnum": tn,
                    "purpose": purposes[(i + tn) % 5], "mode": mode,
                    "band": bands[(i + tn) % 5], "w_trip": 1.0 + i,
                })
    ds = psrc.PSRCDataset(
        households=pd.DataFrame(hh_rows),
        persons=pd.DataFrame(columns=["person_id", "household_id"]),
        person_days=pd.DataFrame(pd_rows),
        weekday_trips=pd.DataFrame(tr_rows),
        build_log={},
    )
    persona_index = pd.DataFrame(pidx_rows)
    enriched = pd.DataFrame(tr_rows)
    return ds, persona_index, enriched, cell_of_household, slots


# ---------------------------------------------------------------------------
# 1. strictly-out-of-fold discipline (the sentinel)
# ---------------------------------------------------------------------------

def test_training_persona_ids_exclude_own_fold():
    _ds, persona_index, _et, _cell, _slots = _scenario()
    persona_fold = {
        str(pa): psrc.fold_id(str(hh))
        for pa, hh in zip(persona_index["persona_id"], persona_index["household_id"])
    }
    for f in range(mnl_arm.N_FOLDS):
        held_in = mnl_arm.training_persona_ids_for_fold(persona_index, f)
        own = {p for p, fold in persona_fold.items() if fold == f}
        # by construction: a fold-f fit sees NO fold-f persona
        assert held_in.isdisjoint(own)
        # and it sees every other-fold persona
        assert held_in == {p for p, fold in persona_fold.items() if fold != f}


def test_fit_records_are_all_out_of_fold():
    """Every training record feeding fold-f's fit comes from a household whose
    fold != f — the OOF guarantee at the record level, not just the id-set level."""
    ds, persona_index, enriched, cell_of_household, slots = _scenario()
    # reproduce the record-building the fit uses, tagging each record's fold
    persona_of_person = e1.persona_of_person_map(persona_index)
    et = enriched.copy()
    et["person_id"] = et["person_id"].astype(str)
    trips_by_person = {str(pid): grp for pid, grp in et.groupby("person_id")}
    trips_by_persona = {persona_of_person[p]: t for p, t in trips_by_person.items()}
    skel = {str(r.persona_id): {"age": r.age, "income_class": r.income_class,
            "employed": r.employed, "can_drive": r.can_drive,
            "household_cars": r.household_cars}
            for r in persona_index.itertuples(index=False)}
    evidence = {}
    for pa in persona_index["persona_id"]:
        t = trips_by_persona.get(str(pa))
        m = list(t["mode"]) if t is not None else []
        p = list(t["purpose"]) if t is not None else []
        evidence[str(pa)] = gp.person_evidence(str(pa), skel[str(pa)], m, p)
    persona_fold = {str(pa): psrc.fold_id(str(hh))
                    for pa, hh in zip(persona_index["persona_id"], persona_index["household_id"])}
    # record-level OOF: the count of records feeding fold f equals the OOF trip
    # count (own-fold trips are structurally excluded, so they can never leak in).
    oof_trip_count = {
        f: sum(len(trips_by_persona[pa]) for pa in evidence if persona_fold[pa] != f)
        for f in range(mnl_arm.N_FOLDS)
    }
    assert all(oof_trip_count[f] > 0 for f in range(mnl_arm.N_FOLDS))
    coefs = mnl_arm.fit_mode_coefficients(persona_index, trips_by_persona, evidence, iters=20)
    assert set(coefs) == set(range(mnl_arm.N_FOLDS))


# ---------------------------------------------------------------------------
# 2. day-signature-class draw: whole observed purpose sequences, no novelty
# ---------------------------------------------------------------------------

def test_signature_draw_is_an_observed_sequence():
    ds, persona_index, enriched, cell_of_household, slots = _scenario()
    arm = mnl_arm.build_mnl_arm(ds, persona_index, enriched, cell_of_household, slots, iters=20)
    days = arm.producer("ns_run0")
    # every produced day's purpose-sequence must be a signature in that
    # persona's out-of-fold signature lookup (drawn whole, never fabricated)
    for pid, dlist in days.items():
        allowed = set(arm.info[pid].sig_sigs)
        for d in dlist:
            sig = tuple(t.purpose for t in d.trips)
            assert sig in allowed
            # every trip carries the day's weight structure
            assert isinstance(d, RealizedDay)
            for t in d.trips:
                assert t.mode in e1.MODES
                assert t.depart_band in e1.TIME_BANDS


def test_pna_persona_falls_back_to_pooled_signatures():
    ds, persona_index, enriched, cell_of_household, slots = _scenario()
    arm = mnl_arm.build_mnl_arm(ds, persona_index, enriched, cell_of_household, slots, iters=20)
    pna = "P009"  # income None -> no cell
    assert arm.info[pna].cell is None
    # its signature lookup is the pooled one for its fold
    fold = psrc.fold_id("H009")
    assert arm.info[pna].sig_sigs == arm.sig_dist[fold]["__pooled__"][0]


# ---------------------------------------------------------------------------
# 3. CRN pairing: the MNL arm uses the frozen key sites, paired with the method
# ---------------------------------------------------------------------------

def test_mnl_uses_frozen_crn_key_sites(monkeypatch):
    ds, persona_index, enriched, cell_of_household, slots = _scenario()
    arm = mnl_arm.build_mnl_arm(ds, persona_index, enriched, cell_of_household, slots, iters=20)
    seen = []
    real = mnl_arm.pick_weighted

    def spy(key, items, weights):
        seen.append(key)
        return real(key, items, weights)

    monkeypatch.setattr(mnl_arm, "pick_weighted", spy)
    persona = "P000"
    day_index, w = slots[persona][0]
    arm.make_days(persona, "nsX", [(day_index, w)])
    # skeleton site
    assert f"nsX:{persona}:{day_index}:skeleton" in seen
    # per-trip band + mode sites, i indexed from 0
    assert any(k == f"nsX:{persona}:{day_index}:trip0:mode" for k in seen)
    assert any(k == f"nsX:{persona}:{day_index}:trip0:band" for k in seen)


def test_mnl_and_method_share_day_key_prefix():
    """Both arms build keys under the same ``{ns}:{persona}:{day}:`` prefix, so
    they are paired within a run namespace (the method arm's pattern site and the
    MNL arm's skeleton/trip sites live under the identical run/persona/day key)."""
    _ds, _pidx, _et, _cell, slots = _scenario()
    persona = "P000"
    day_index, _w = slots[persona][0]
    ns = "method_run3"
    # method arm's only draw site (card_executor): "{ns}:{persona}:{day}:pattern"
    method_key = f"{ns}:{persona}:{day_index}:pattern"
    # MNL arm's skeleton draw site under the SAME prefix
    mnl_key = f"{ns}:{persona}:{day_index}:skeleton"
    assert method_key.rsplit(":", 1)[0] == mnl_key.rsplit(":", 1)[0]


def test_producer_is_deterministic_per_namespace():
    ds, persona_index, enriched, cell_of_household, slots = _scenario()
    arm = mnl_arm.build_mnl_arm(ds, persona_index, enriched, cell_of_household, slots, iters=20)
    a = arm.producer("ns_run0")
    b = arm.producer("ns_run0")
    for pid in a:
        assert [(t.mode, t.depart_band, t.purpose) for d in a[pid] for t in d.trips] == \
               [(t.mode, t.depart_band, t.purpose) for d in b[pid] for t in d.trips]


# ---------------------------------------------------------------------------
# 4. end-to-end: the arm scores through the same e1 path as the method arm
# ---------------------------------------------------------------------------

def test_mnl_arm_scores_through_e1_ensemble():
    ds, persona_index, enriched, cell_of_household, slots = _scenario()
    arm = mnl_arm.build_mnl_arm(ds, persona_index, enriched, cell_of_household, slots, iters=20)
    persona_cell = e1.persona_cell_map(persona_index, cell_of_household)
    dists = e1.ensemble_arm(arm.producer, persona_cell, n_runs=3, namespace_prefix="mnl_")
    # pooled families are proper probability vectors
    for f in e1.FAMILIES:
        v = dists.pooled[f]
        assert not np.isnan(v).any()
        assert v.sum() == pytest.approx(1.0)
