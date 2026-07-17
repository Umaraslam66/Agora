"""A4.1 E7 tier-population build driver (harness side).

Builds, for each tier T1..T5 + T4-noclaims + T4-nofidelity, the generation
inputs and (after cluster generation) the gated tier card population:

  prompts   per-tier ``prompts.jsonl`` (persona_id, prompt, attempt — the
            serving/batch_gen input shape, every prompt mask-lint-gated) plus
            a ``tier_context.json`` sidecar (per-persona skeleton, the
            tier-visible fidelity reference, visible day sequences, and the
            CRN-selected T4 day) and the E7 manifest with every A4.1 pin.
  retry     after a generation round, re-run the SAME gates on the raw
            output and render attempt-N+1 prompts (tier evidence + numeric
            gate feedback, via `grounding.seeding.build_retry_prompts`) for
            the failures — the M2 deployed population's attempt-round
            structure (up to 3 attempts), mirrored per A4.1's "SAME
            generation pipeline" pin. Symmetric across tiers by
            construction: every tier gets the same attempt budget.
  assemble  join a tier's cluster ``cards_raw.jsonl`` round outputs (later
            rounds overwrite earlier for the same persona) with the sidecar,
            run the gates (fidelity anchored to the tier-visible diary;
            T1-T3 and T4-nofidelity validate with an empty observed
            reference), and assemble accepted cards; terminal failures take
            the deterministic tier fallback
            (`grounding.e7_tiers.tier_fallback_card`).

CRN pairing across tiers (A4.1): identical persona set and generation seeds;
prompts differ only by the evidence bundle. The T4 day-selection rule and the
ESS scoring protocol pins are recorded in the manifest BEFORE any scoring.

The raw stated-typical-mode survey label is mapped to the masked five-mode
token HERE (harness side, via the frozen `calibration.e3_recall_fit`
WORK_MODE_MAP) so agent-facing code never sees real vocabulary.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

from calibration.e3_recall_fit import WORK_MODE_MAP
from grounding import e7_tiers, seeding
from grounding.adapters import psrc
from grounding.card_validation import (
    assemble_card,
    day_signatures,
    validate_card,
    validate_card_structural,
)
from grounding.render import render_seed_prompt
from serving.batch_gen import prompt_sha256

PR2_HH_CSV = Path("data/psrc/codebook_2019/2017-2019-pr2-1-Household.csv")


def _person_rows(dataset) -> pd.DataFrame:
    """persona_index + masked stated-typical-mode token + PR2 residence
    factors, keyed by persona_id."""
    idx = seeding.persona_index(dataset)
    idx["household_id"] = idx["household_id"].astype(str)

    def masked_typical(row):
        m = WORK_MODE_MAP.get(str(row.get("work_mode")))
        if m == "__carpool__":
            return "car" if bool(row.get("can_drive", True)) else "ride"
        return m

    idx["stated_typical_mode"] = idx.apply(masked_typical, axis=1)

    if PR2_HH_CSV.exists():
        hh = pd.read_csv(
            PR2_HH_CSV, dtype=str, low_memory=False,
            usecols=lambda c: c in ("household_id", "hhid", "res_factors_transit",
                                    "res_factors_walk"),
        )
        id_col = "household_id" if "household_id" in hh.columns else "hhid"
        hh = hh.rename(columns={id_col: "household_id"})
        keep = [c for c in ("res_factors_transit", "res_factors_walk") if c in hh.columns]
        idx = idx.merge(hh[["household_id"] + keep], on="household_id", how="left")
    return idx


def _diary_frames(dataset):
    enriched = seeding.enriched_trips(dataset)
    enriched["person_id"] = enriched["person_id"].astype(str)
    pdays = dataset.person_days.copy()
    pdays["person_id"] = pdays["person_id"].astype(str)
    return pdays, enriched


def cmd_prompts(args) -> int:
    dataset = psrc.load_or_build()
    rows = _person_rows(dataset)
    pdays, enriched = _diary_frames(dataset)
    days_by = {pid: g for pid, g in pdays.groupby("person_id")}
    trips_by = {pid: g for pid, g in enriched.groupby("person_id")}
    empty_days, empty_trips = pdays.iloc[0:0], enriched.iloc[0:0]

    tokens = seeding._forbidden_tokens()
    out_root = Path(args.out)
    tiers = args.tiers or list(e7_tiers.TIERS)

    manifest = {
        "amendment": "01_PREREGISTRATION.md section 7 A4.1",
        "date": date.today().isoformat(),
        "tiers": tiers,
        "block_order": "[world][stated][diary]",
        "t4_day_selection": {
            "rule": "world.crn.pick_weighted over the persona's observed "
                    "weekday daynums, weighted by day weight",
            "site_key": e7_tiers.T4_DAY_SITE,
        },
        "fidelity_applicability": sorted(e7_tiers.FIDELITY_TIERS),
        "fallback_semantics": "tier-blind deterministic backstop "
                              "(grounding.e7_tiers.tier_fallback_card)",
        "t5_superset_note": (
            "T5 = T4 + remaining diary; the stated-claims block renders ALL "
            "A3.1 items (typical mode, telework, residence importance added), "
            "a strict superset of the M2 bundle's self-report block — required "
            "by A4.1 nesting (T3 must be contained in T4 and T5)"
        ),
        "ess_protocol_pins": {
            "units": "equivalent sample size (Gao, Han & Liang 2026, "
                     "arXiv 2601.12343)",
            "flexible_baseline": "the E1 MNL falsification arm's day-structure "
                                 "+ mode-choice model, trained on n real diary "
                                 "records under the A2.1 fold structure",
            "cv_protocol": "A2.1 five household-atomic folds, pooled "
                           "out-of-fold, ensemble N >= 20",
            "headline_discipline": "T5 is the SOLE headline; T1-T4 are the "
                                   "diagnostic information-value curve",
        },
        "counts": {},
    }

    for tier in tiers:
        records: List[dict] = []
        context: Dict[str, dict] = {}
        n_lint_fail = 0
        for r in rows.itertuples(index=False):
            row = r._asdict()
            pid = str(row["persona_id"])
            person_id = str(row["person_id"])
            skeleton = {f: row.get(f) for f in seeding.SKELETON_FIELDS}
            pdd = days_by.get(person_id, empty_days)
            ptr = trips_by.get(person_id, empty_trips)

            selected = None
            if tier in e7_tiers.ONE_DAY_TIERS and len(pdd):
                daynums = [int(d) for d in pdd["daynum"]]
                weights = [float(w) for w in pdd["day_weight"]]
                if sum(weights) <= 0:
                    weights = [1.0] * len(daynums)
                selected = e7_tiers.t4_day_of(pid, daynums, weights)

            lines, vis_days, vis_trips, n_obs = e7_tiers.tier_evidence(
                tier, skeleton, pdd, ptr, row, selected
            )
            skeleton_view = {k: ("none" if v is None else v) for k, v in skeleton.items()}
            prompt = render_seed_prompt(skeleton_view, lines, n_obs, mode="serve")
            offending = seeding._gate_prompt(pid, prompt, tokens)
            if offending:
                n_lint_fail += 1
                continue
            records.append({"persona_id": pid, "prompt": prompt, "attempt": 1})
            context[pid] = {
                "skeleton": skeleton,
                "observed": e7_tiers.tier_fidelity_observed(tier, vis_days, vis_trips),
                "observed_day_sequences": day_signatures(vis_days, vis_trips),
                "selected_daynum": selected,
                "prompt_sha256": prompt_sha256(prompt),
            }

        tier_dir = out_root / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        with (tier_dir / "prompts.jsonl").open("w") as f:
            for rec in records:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        (tier_dir / "tier_context.json").write_text(
            json.dumps(context, default=str)
        )
        manifest["counts"][tier] = {
            "prompts": len(records), "lint_failures": n_lint_fail,
        }
        print(f"{tier}: {len(records)} prompts ({n_lint_fail} lint failures)")
        if n_lint_fail:
            print(f"  WARNING: {tier} had lint failures — inspect before shipping")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "e7_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


def _read_rounds(paths) -> Dict[str, dict]:
    """Read one or more generation-round outputs in order; a later round's
    record overwrites an earlier one for the same persona (retry rounds only
    contain previously-failed personas, so acceptance is monotone)."""
    raw_by_pid: Dict[str, dict] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                raw_by_pid[str(rec["persona_id"])] = rec
    return raw_by_pid


def _gate_generated(tier: str, ctx: Mapping, obj) -> List[str]:
    """The tier gate, ONE code path for retry and assemble: fidelity-gated
    tiers run the full five-gate compose against the tier-visible diary;
    T1-T3 / T4-nofidelity are fidelity-exempt at generation (A4.1)."""
    if obj is None:
        return ["no generation record"]
    if tier in e7_tiers.FIDELITY_TIERS:
        return validate_card(
            obj, ctx["skeleton"], ctx["observed"], ctx["observed_day_sequences"]
        )
    return validate_card_structural(
        obj, ctx["skeleton"], ctx["observed"] or None,
        ctx["observed_day_sequences"],
    )


def cmd_retry(args) -> int:
    """Emit attempt-N+1 prompts for a tier's gate failures (M2 round-trip)."""
    from grounding.seeding import build_retry_prompts

    dataset = psrc.load_or_build()
    rows = _person_rows(dataset)
    row_of = {str(r["persona_id"]): r for r in rows.to_dict("records")}
    pdays, enriched = _diary_frames(dataset)
    days_by = {pid: g for pid, g in pdays.groupby("person_id")}
    trips_by = {pid: g for pid, g in enriched.groupby("person_id")}
    empty_days, empty_trips = pdays.iloc[0:0], enriched.iloc[0:0]

    tier = args.tier
    tier_dir = Path(args.out) / tier
    context = json.loads((tier_dir / "tier_context.json").read_text())
    raw_by_pid = _read_rounds(args.generated)

    failures: List[dict] = []
    for pid, ctx in context.items():
        rec = raw_by_pid.get(pid)
        obj = rec.get("raw_json") if rec else None
        errs = _gate_generated(tier, ctx, obj)
        if not errs:
            continue
        row = row_of[pid]
        person_id = str(row["person_id"])
        selected = ctx.get("selected_daynum")
        lines, _vd, _vt, n_obs = e7_tiers.tier_evidence(
            tier, ctx["skeleton"], days_by.get(person_id, empty_days),
            trips_by.get(person_id, empty_trips), row,
            int(selected) if selected is not None else None,
        )
        failures.append({
            "persona_id": pid,
            "skeleton": ctx["skeleton"],
            "evidence_lines": lines,
            "n_observed_days": n_obs,
            "failure_reasons": errs,
            "attempt": rec.get("attempt", 1) if rec else 1,
        })

    out_path = tier_dir / f"prompts_attempt{args.attempt}.jsonl"
    info = build_retry_prompts(failures, out_path)
    print(f"{tier}: {info['n_prompts']} retry prompts -> {out_path}")
    return 0


def cmd_assemble(args) -> int:
    dataset = psrc.load_or_build()
    rows = _person_rows(dataset)
    row_of = {str(r["persona_id"]): r for r in rows.to_dict("records")}
    pdays, enriched = _diary_frames(dataset)
    days_by = {pid: g for pid, g in pdays.groupby("person_id")}
    trips_by = {pid: g for pid, g in enriched.groupby("person_id")}
    empty_days, empty_trips = pdays.iloc[0:0], enriched.iloc[0:0]

    tier = args.tier
    tier_dir = Path(args.out) / tier
    context = json.loads((tier_dir / "tier_context.json").read_text())
    raw_by_pid = _read_rounds(args.generated)

    cards: List[dict] = []
    stats = {"accepted": 0, "fallback": 0, "gate_failures": {}}
    for pid, ctx in context.items():
        row = row_of[pid]
        person_id = str(row["person_id"])
        skeleton = ctx["skeleton"]
        rec = raw_by_pid.get(pid)
        obj = rec.get("raw_json") if rec else None
        errs = _gate_generated(tier, ctx, obj)
        if not errs:
            card = assemble_card(pid, skeleton, obj, {
                "card_source": "llm", "e7_tier": tier,
                "model": rec.get("model"), "attempt": rec.get("attempt", 1),
                "prompt_sha": rec.get("prompt_sha256"), "mask_lint": "v0.3",
            })
            cards.append(card)
            stats["accepted"] += 1
        else:
            sel = ctx.get("selected_daynum")
            pdd = days_by.get(person_id, empty_days)
            ptr = trips_by.get(person_id, empty_trips)
            if tier in e7_tiers.ONE_DAY_TIERS and sel is not None:
                pdd = pdd[pdd["daynum"].astype(int) == int(sel)]
                ptr = ptr[ptr["daynum"].astype(int) == int(sel)]
            elif tier not in ("T5",) and tier not in e7_tiers.ONE_DAY_TIERS:
                pdd, ptr = empty_days, empty_trips
            card = e7_tiers.tier_fallback_card(tier, pid, skeleton, pdd, ptr, row)
            cards.append(card)
            stats["fallback"] += 1
            key = errs[0].split(":")[0][:40] if errs else "unknown"
            stats["gate_failures"][key] = stats["gate_failures"].get(key, 0) + 1

    out_path = tier_dir / f"cards_{tier}.jsonl"
    with out_path.open("w") as f:
        for c in cards:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    (tier_dir / "assemble_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"{tier}: {stats['accepted']} accepted, {stats['fallback']} fallback "
          f"-> {out_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--out", default="runs/e7_tiers")
    p.add_argument("--tiers", nargs="*", default=None)
    p.set_defaults(func=cmd_prompts)
    r = sub.add_parser("retry")
    r.add_argument("--out", default="runs/e7_tiers")
    r.add_argument("--tier", required=True)
    r.add_argument("--generated", required=True, nargs="+",
                   help="generation-round outputs so far, in round order")
    r.add_argument("--attempt", type=int, required=True,
                   help="the attempt number the emitted prompts carry (2 or 3)")
    r.set_defaults(func=cmd_retry)
    a = sub.add_parser("assemble")
    a.add_argument("--out", default="runs/e7_tiers")
    a.add_argument("--tier", required=True)
    a.add_argument("--generated", required=True, nargs="+",
                   help="generation-round outputs, in round order (later "
                        "rounds overwrite earlier for the same persona)")
    a.set_defaults(func=cmd_assemble)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
