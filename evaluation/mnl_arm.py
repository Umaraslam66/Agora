"""E1 MNL falsification arm (pre-registration §3 E1, A2.1; M2 spec D6).

The method's grounding claim only stands if the LLM-written persona cards
beat-or-match a calibrated MNL scored identically. This module IS that MNL arm.
Everything fitted here is STRICTLY out-of-fold (A2.1): for each of the five
household-atomic folds f, every component is trained on the OTHER four folds and
applied to fold-f personas, then all folds are pooled.

Two fitted components, both out-of-fold:

* Day-structure model (D6). Per guard-merged protected cell (the same ten cells
  E1 scores; the twelve-cell definition collapsed for sparse safety), the
  day_weight-weighted distribution over observed weekday DAY-SIGNATURE CLASSES,
  where a class is the ORDERED PURPOSE SEQUENCE of the day (empty tuple = a
  zero-trip weekday). Drawing a whole observed purpose-sequence (rather than
  factorising trips/day x purpose independently) is the sparse-robust choice:
  every drawn skeleton is a real observed day-shape, so no zero-probability
  novel combination can appear, and the joint (n_trips, purpose composition) is
  preserved. A cell whose out-of-fold signature list is empty falls back to the
  pooled (all-training-fold) distribution. Departure bands are then drawn per
  (cell, purpose) from the out-of-fold band distribution.
* Mode choice (D6). ``agents.logit_chooser`` fit per fold on out-of-fold observed
  trips, through the harness-side GOLDEN-format builder
  (``evaluation.golden_pairs``) that encodes the same person-level evidence the
  LLM saw. At simulation the realized mode is drawn from the fitted softmax
  utilities via a CRN inverse-CDF over the frozen five-mode order
  (``world.crn.pick_weighted`` on the softmax probabilities) — never
  ``choose(temperature>0)`` — so the arm pairs draw-for-draw with the method arm
  at the same CRN key sites (``trip{i}:mode``).

CRN key sites (paired with the method arm within a run namespace):
  * ``{ns}:{persona_id}:{day_index}:skeleton``   — the day purpose-sequence draw
  * ``{ns}:{persona_id}:{day_index}:trip{i}:band`` — the depart-band draw
  * ``{ns}:{persona_id}:{day_index}:trip{i}:mode`` — the mode draw

The arm emits the same ``RealizedDay`` / ``RealizedTrip`` structures the method
arm emits, so ``evaluation.e1`` scores both arms through one code path.
"""
from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agents import logit_chooser
from agents.card_executor import RealizedDay, RealizedTrip
from evaluation import golden_pairs as gp
from evaluation.e1 import CELLS10, MERGED_CELL, _MERGED_SRC
from grounding.adapters import psrc
from grounding.taxonomy import MODES
from world.crn import pick_weighted

N_FOLDS = 5
_POOLED_KEY = "__pooled__"


# ---------------------------------------------------------------------------
# fold assignment (household-atomic; the committed adapter rule)
# ---------------------------------------------------------------------------

def _guard_cell(segment: Optional[str]) -> Optional[str]:
    """Collapse a 12-cell segment string to the guard-merged 10-cell key, or
    None for PNA-income households (no cell)."""
    if segment is None or (isinstance(segment, float) and np.isnan(segment)):
        return None
    return MERGED_CELL if segment in _MERGED_SRC else segment


# ---------------------------------------------------------------------------
# per-persona evidence + parsed feature vector (constant across a persona's trips)
# ---------------------------------------------------------------------------

class _PersonaInfo:
    """Precomputed per-persona simulation state (constant across runs and days):
    fold, guard-merged cell, the parsed GOLDEN feature vector, the feasible mode
    set, and the fold's softmax mode probabilities + signature lookup."""

    __slots__ = (
        "fold", "cell", "parsed", "feasible", "mode_order", "mode_probs",
        "sig_sigs", "sig_weights",
    )

    def __init__(self, fold, cell, parsed, feasible, mode_order, mode_probs,
                 sig_sigs, sig_weights):
        self.fold = fold
        self.cell = cell
        self.parsed = parsed
        self.feasible = feasible
        self.mode_order = mode_order
        self.mode_probs = mode_probs
        self.sig_sigs = sig_sigs
        self.sig_weights = sig_weights


def _softmax_probs(parsed, feasible, coef) -> Tuple[List[str], List[float]]:
    """Softmax over the fitted MNL utilities, ordered by the frozen five-mode
    order (the inverse-CDF order the CRN draw walks)."""
    utils = logit_chooser.utilities(parsed, feasible, coef)
    order = [m for m in MODES if m in feasible]
    u = np.array([utils[m] for m in order], dtype=float)
    u -= u.max()
    p = np.exp(u)
    p /= p.sum()
    return order, [float(x) for x in p]


# ---------------------------------------------------------------------------
# The arm
# ---------------------------------------------------------------------------

class MNLArm:
    """A fully out-of-fold MNL arm ready to simulate realized days.

    Build it with :func:`build_mnl_arm`; call :meth:`producer` to get a
    ``producer(namespace) -> {persona_id: [RealizedDay]}`` for :func:`e1.ensemble_arm`.
    """

    def __init__(
        self,
        persona_ids: Sequence[str],
        day_slots: Mapping[str, Sequence[Tuple[int, float]]],
        info: Mapping[str, _PersonaInfo],
        sig_dist: Mapping[int, Mapping[str, Tuple[List[tuple], List[float]]]],
        band_dist: Mapping[int, Mapping[Tuple[str, str], Tuple[List[str], List[float]]]],
        band_global: Mapping[int, Tuple[List[str], List[float]]],
        diagnostics: dict,
    ):
        self.persona_ids = list(persona_ids)
        self.day_slots = day_slots
        self.info = info
        self.sig_dist = sig_dist
        self.band_dist = band_dist
        self.band_global = band_global
        self.diagnostics = diagnostics

    # -- draws --------------------------------------------------------------

    def _sig_lookup(self, fold: int, cell: Optional[str]):
        d = self.sig_dist[fold]
        if cell is not None and cell in d and d[cell][0]:
            return d[cell]
        return d[_POOLED_KEY]

    def _draw_band(self, fold: int, cell: Optional[str], purpose: str, key: str) -> str:
        bd = self.band_dist[fold]
        for k in ((cell, purpose), (_POOLED_KEY, purpose)):
            entry = bd.get(k)
            if entry is not None and entry[0]:
                return pick_weighted(key, entry[0], entry[1])
        bands, weights = self.band_global[fold]
        return pick_weighted(key, bands, weights)

    def make_days(
        self, persona_id: str, namespace: str, slots: Sequence[Tuple[int, float]]
    ) -> List[RealizedDay]:
        pinfo = self.info[persona_id]
        fold = pinfo.fold
        cell = pinfo.cell
        sigs, weights = pinfo.sig_sigs, pinfo.sig_weights
        order, probs = pinfo.mode_order, pinfo.mode_probs
        days: List[RealizedDay] = []
        for day_index, w in slots:
            sig = pick_weighted(
                f"{namespace}:{persona_id}:{day_index}:skeleton", sigs, weights
            )
            trips: List[RealizedTrip] = []
            for i, purpose in enumerate(sig):
                band = self._draw_band(
                    fold, cell, purpose,
                    f"{namespace}:{persona_id}:{day_index}:trip{i}:band",
                )
                mode = pick_weighted(
                    f"{namespace}:{persona_id}:{day_index}:trip{i}:mode", order, probs
                )
                trips.append(RealizedTrip(purpose, mode, band, None))
            days.append(RealizedDay(int(day_index), float(w), trips))
        return days

    def producer(self, namespace: str) -> Dict[str, List[RealizedDay]]:
        return {
            pid: self.make_days(pid, namespace, self.day_slots.get(pid, []))
            for pid in self.persona_ids
        }


# ---------------------------------------------------------------------------
# Fitting (all strictly out-of-fold)
# ---------------------------------------------------------------------------

def _fold_of_household(household_id: str) -> int:
    return psrc.fold_id(str(household_id))


def fit_mode_coefficients(
    persona_index: pd.DataFrame,
    trips_by_persona: Mapping[str, pd.DataFrame],
    evidence: Mapping[str, gp.PersonEvidence],
    iters: int = 250,
    workdir: Optional[str] = None,
) -> Dict[int, dict]:
    """Fit one logit coefficient set per fold on that fold's OUT-OF-FOLD trips
    (folds != f), via ``agents.logit_chooser.fit`` through the GOLDEN builder.

    Returns ``{fold: coef_dict}``. Out-of-fold discipline is enforced by
    construction: a fold-f record is built ONLY from personas whose household
    fold != f (asserted by :func:`training_persona_ids_for_fold`).
    """
    persona_fold = {
        str(pa): _fold_of_household(hh)
        for pa, hh in zip(persona_index["persona_id"], persona_index["household_id"])
    }
    # Build all per-trip records once, tagged with the persona's fold.
    all_records: List[Tuple[int, dict]] = []
    for pid, ev in evidence.items():
        fold = persona_fold[pid]
        tdf = trips_by_persona.get(pid)
        if tdf is None or not len(tdf):
            continue
        # deterministic split tag (used only for the fit's accuracy print)
        for purpose, mode in zip(tdf["purpose"], tdf["mode"]):
            rec = gp.training_record(ev, str(purpose), str(mode), split="train")
            if rec is not None:
                all_records.append((fold, rec))

    coefs: Dict[int, dict] = {}
    tmp = workdir or tempfile.mkdtemp(prefix="mnl_fit_")
    tmp_path = Path(tmp)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for f in range(N_FOLDS):
        recs = [r for fold, r in all_records if fold != f]
        pairs_path = tmp_path / f"pairs_oof_fold{f}.jsonl"
        with pairs_path.open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        coef_path = tmp_path / f"coef_oof_fold{f}.json"
        logit_chooser.fit(str(pairs_path), str(coef_path), iters=iters, lr=0.5, l2=1e-3)
        coefs[f] = logit_chooser.load_coef(str(coef_path))
    return coefs


def training_persona_ids_for_fold(
    persona_index: pd.DataFrame, fold: int
) -> frozenset:
    """The persona ids whose OOF fit for ``fold`` may see (households not in
    ``fold``). Exposed for the out-of-fold sentinel test."""
    return frozenset(
        str(pa)
        for pa, hh in zip(persona_index["persona_id"], persona_index["household_id"])
        if _fold_of_household(hh) != fold
    )


def fit_day_structure(
    person_days: pd.DataFrame,
    trips_by_person: Mapping[str, pd.DataFrame],
    persona_of_person: Mapping[str, str],
    cell_of_persona: Mapping[str, Optional[str]],
):
    """Fit the out-of-fold day-signature and depart-band models.

    Returns ``(sig_dist, band_dist, band_global)`` where, for each fold f:
      * ``sig_dist[f][cell]`` = (signatures, weights): day_weight-weighted
        distribution over ordered purpose-sequences for that cell, aggregated
        over the OTHER folds; ``sig_dist[f]['__pooled__']`` over all other-fold
        person-days.
      * ``band_dist[f][(cell, purpose)]`` and ``[('__pooled__', purpose)]`` =
        (bands, weights): trip_weight-weighted depart-band distribution.
      * ``band_global[f]`` = the other-fold global band distribution (ultimate
        fallback).
    """
    # -- per (fold, cell) signature mass, and per (fold, cell, purpose) band mass
    sig_by_cell_fold: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
    sig_by_fold: Dict[int, Counter] = defaultdict(Counter)
    band_by_cp_fold: Dict[Tuple[str, str, int], Counter] = defaultdict(Counter)
    band_by_p_fold: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
    band_global_fold: Dict[int, Counter] = defaultdict(Counter)

    # signature per person-day: ordered purposes (empty tuple for a zero-trip day)
    seq_by_person: Dict[str, Dict[int, tuple]] = {}
    for pid, tdf in trips_by_person.items():
        ordered = tdf.sort_values(["daynum", "tripnum"]) if "tripnum" in tdf else tdf.sort_values(["daynum"])
        by_day: Dict[int, list] = defaultdict(list)
        for daynum, purpose in zip(ordered["daynum"], ordered["purpose"]):
            by_day[int(daynum)].append(str(purpose))
        seq_by_person[str(pid)] = {d: tuple(v) for d, v in by_day.items()}

    for hh, person_id, daynum, w_day in zip(
        person_days.household_id.astype(str),
        person_days.person_id.astype(str),
        person_days.daynum,
        person_days.w_day,
    ):
        fold = _fold_of_household(hh)
        persona = persona_of_person.get(str(person_id))
        cell = cell_of_persona.get(persona) if persona is not None else None
        sig = seq_by_person.get(str(person_id), {}).get(int(daynum), tuple())
        sig_by_fold[fold][sig] += float(w_day)
        if cell is not None:
            sig_by_cell_fold[(cell, fold)][sig] += float(w_day)

    for pid, tdf in trips_by_person.items():
        persona = persona_of_person.get(str(pid))
        cell = cell_of_persona.get(persona) if persona is not None else None
        hh_fold = _fold_of_household(_household_of_person(pid, tdf))  # constant per person
        for purpose, band, w in zip(tdf["purpose"], tdf["band"], tdf["w_trip"]):
            band_by_p_fold[(str(purpose), hh_fold)][str(band)] += float(w)
            band_global_fold[hh_fold][str(band)] += float(w)
            if cell is not None:
                band_by_cp_fold[(cell, str(purpose), hh_fold)][str(band)] += float(w)

    def _combine_counters(counters):
        acc = Counter()
        for c in counters:
            acc.update(c)
        return acc

    def _to_lists(counter: Counter):
        items = sorted(counter.items(), key=lambda kv: (repr(kv[0]),))
        return [k for k, _ in items], [float(v) for _, v in items]

    sig_dist: Dict[int, Dict[str, Tuple[list, list]]] = {}
    band_dist: Dict[int, Dict[Tuple[str, str], Tuple[list, list]]] = {}
    band_global: Dict[int, Tuple[list, list]] = {}
    for f in range(N_FOLDS):
        other = [g for g in range(N_FOLDS) if g != f]
        sig_dist[f] = {}
        for cell in CELLS10:
            acc = _combine_counters(sig_by_cell_fold.get((cell, g), Counter()) for g in other)
            sig_dist[f][cell] = _to_lists(acc)
        sig_dist[f][_POOLED_KEY] = _to_lists(
            _combine_counters(sig_by_fold.get(g, Counter()) for g in other)
        )
        band_dist[f] = {}
        # per (cell, purpose): aggregate this fold's OTHER folds
        seen_cp = set()
        for (cell, purpose, _g) in band_by_cp_fold.keys():
            if (cell, purpose) in seen_cp:
                continue
            seen_cp.add((cell, purpose))
            acc = _combine_counters(
                band_by_cp_fold.get((cell, purpose, g), Counter()) for g in other
            )
            band_dist[f][(cell, purpose)] = _to_lists(acc)
        seen_p = set()
        for (purpose, _g) in band_by_p_fold.keys():
            if purpose in seen_p:
                continue
            seen_p.add(purpose)
            acc = _combine_counters(band_by_p_fold.get((purpose, g), Counter()) for g in other)
            band_dist[f][(_POOLED_KEY, purpose)] = _to_lists(acc)
        band_global[f] = _to_lists(
            _combine_counters(band_global_fold.get(g, Counter()) for g in other)
        )
    return sig_dist, band_dist, band_global


def _household_of_person(person_id: str, tdf: pd.DataFrame) -> str:
    if "household_id" in tdf.columns and len(tdf):
        return str(tdf["household_id"].iloc[0])
    return str(person_id)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_mnl_arm(
    dataset,
    persona_index: pd.DataFrame,
    enriched_trips: pd.DataFrame,
    cell_of_household: Mapping[str, Optional[str]],
    day_slots: Mapping[str, Sequence[Tuple[int, float]]],
    iters: int = 250,
    workdir: Optional[str] = None,
) -> MNLArm:
    """Assemble a fully out-of-fold :class:`MNLArm`.

    ``enriched_trips`` must carry columns person_id, daynum, tripnum, purpose,
    mode, band, w_trip (``grounding.seeding.enriched_trips``). ``day_slots`` is
    the shared per-persona slot table both arms simulate.
    """
    persona_index = persona_index.copy()
    persona_index["persona_id"] = persona_index["persona_id"].astype(str)
    persona_index["person_id"] = persona_index["person_id"].astype(str)
    persona_index["household_id"] = persona_index["household_id"].astype(str)

    persona_of_person = {
        str(p): str(pa)
        for p, pa in zip(persona_index["person_id"], persona_index["persona_id"])
    }
    cell_of_persona = {
        str(pa): cell_of_household.get(str(hh))
        for pa, hh in zip(persona_index["persona_id"], persona_index["household_id"])
    }
    skeleton_of_persona = {
        str(r.persona_id): {
            "age": r.age,
            "income_class": r.income_class,
            "employed": r.employed,
            "can_drive": r.can_drive,
            "household_cars": r.household_cars,
        }
        for r in persona_index.itertuples(index=False)
    }

    et = enriched_trips.copy()
    et["person_id"] = et["person_id"].astype(str)
    # attach household_id (for band-model fold assignment) if absent
    if "household_id" not in et.columns:
        p2h = {str(p): str(h) for p, h in zip(persona_index["person_id"], persona_index["household_id"])}
        et["household_id"] = et["person_id"].map(p2h)
    trips_by_person = {str(pid): grp for pid, grp in et.groupby("person_id")}

    # per-persona GOLDEN evidence
    evidence: Dict[str, gp.PersonEvidence] = {}
    for persona_id, person_id in zip(persona_index["persona_id"], persona_index["person_id"]):
        skel = skeleton_of_persona[str(persona_id)]
        tdf = trips_by_person.get(str(person_id))
        if tdf is not None and len(tdf):
            modes = list(tdf["mode"])
            purposes = list(tdf["purpose"])
        else:
            modes, purposes = [], []
        evidence[str(persona_id)] = gp.person_evidence(str(persona_id), skel, modes, purposes)

    # trips keyed by persona for the mode fit
    trips_by_persona = {
        persona_of_person[pid]: tdf for pid, tdf in trips_by_person.items() if pid in persona_of_person
    }

    # -- fit both components strictly out-of-fold ---------------------------
    coefs = fit_mode_coefficients(persona_index, trips_by_persona, evidence, iters=iters, workdir=workdir)
    sig_dist, band_dist, band_global = fit_day_structure(
        dataset.person_days, trips_by_person, persona_of_person, cell_of_persona
    )

    # -- precompute per-persona info (softmax probs + signature lookup) -----
    info: Dict[str, _PersonaInfo] = {}
    for r in persona_index.itertuples(index=False):
        persona_id = str(r.persona_id)
        fold = _fold_of_household(r.household_id)
        cell = cell_of_persona.get(persona_id)
        ev = evidence[persona_id]
        parsed = logit_chooser.parse_prompt(gp.build_prompt(ev))
        order, probs = _softmax_probs(parsed, ev.feasible, coefs[fold])
        sigs, weights = (sig_dist[fold][cell] if (cell is not None and sig_dist[fold][cell][0])
                         else sig_dist[fold][_POOLED_KEY])
        info[persona_id] = _PersonaInfo(
            fold=fold, cell=cell, parsed=parsed, feasible=ev.feasible,
            mode_order=order, mode_probs=probs, sig_sigs=sigs, sig_weights=weights,
        )

    diagnostics = {
        "n_personas": len(info),
        "fit_iters": iters,
        "n_coef_per_fold": {f: len(c) for f, c in coefs.items()},
    }
    return MNLArm(
        persona_ids=[str(p) for p in persona_index["persona_id"]],
        day_slots=day_slots,
        info=info,
        sig_dist=sig_dist,
        band_dist=band_dist,
        band_global=band_global,
        diagnostics=diagnostics,
    )
