"""Transfer-arena (BT2) population build driver (harness side).

Builds the SINGLE full-information persona population for the Stockholm
transfer arena (A1.2/A1.3, A8): the frozen generation pipeline applied
unchanged — T5-shape evidence (world context + all stated claims + full
diary) over the same 11,940 PSRC personas, rendered through the ONE seed
prompt path — with exactly one arena-ruled data difference:

  has_pass    The corridor transponder-pass concept does not exist in the
              cordon world; per the sealed ruling restated in A8.5(ii) the
              household transponder-pass gate does NOT carry to arena 2
              (the cordon analog — exemption structure — is world
              structure, not method). Every persona therefore carries the
              no-instrument value ``has_pass = False`` instead of a draw
              from the corridor world's ``pass_prior``. Recorded in the
              manifest; nothing else about the skeleton changes.

The E7 tiers are NOT rebuilt here (A4.1 was BT1-scoped). Scoring weights
are the OWNER-ADOPTED adults-only-M1 raking weights
(`runs/transfer_reweight_adultsM1/weights.csv`); they enter at scoring
time as aggregation weights, never at generation, and their sha256 is
pinned in this build's manifest so the firing driver can refuse a swap.

Subcommands mirror `evaluation.build_e7_tiers` (the SAME generation
pipeline, same attempt-round structure, same gates — full five-gate
compose incl. fidelity against the full observed diary, as for T5):

  prompts    render attempt-1 prompts + pop_context.json sidecar + manifest
  retry      re-gate a round's raw output, emit attempt-N+1 prompts
  assemble   gate all rounds, assemble accepted cards, deterministic
             fallback (`grounding.e7_tiers.tier_fallback_card`, T5 shape)
             for terminal failures

Every prompt is mask-lint-gated at render time (both arenas' vocabularies,
v0.3). Every derived number downstream carries the METHOD-TRANSFER label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Dict, List

from grounding import e7_tiers, seeding
from grounding.adapters import psrc
from grounding.card_validation import (
    assemble_card,
    day_signatures,
    validate_card,
)
from grounding.render import render_seed_prompt
from serving.batch_gen import prompt_sha256

from evaluation.build_e7_tiers import _diary_frames, _person_rows

#: The evidence shape of this population: T5 = full information.
EVIDENCE_TIER = "T5"

ADOPTED_WEIGHTS = Path("runs/transfer_reweight_adultsM1/weights.csv")
ADOPTED_MANIFEST = Path("runs/transfer_reweight_adultsM1/manifest.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _transfer_rows(dataset):
    """persona rows with the arena-2 ruled skeleton correction applied."""
    rows = _person_rows(dataset)
    # A8.5(ii): no transponder instrument exists in the cordon world; the
    # no-instrument value replaces the corridor pass_prior CRN draw.
    rows["has_pass"] = False
    return rows


def cmd_prompts(args) -> int:
    dataset = psrc.load_or_build()
    rows = _transfer_rows(dataset)
    pdays, enriched = _diary_frames(dataset)
    days_by = {pid: g for pid, g in pdays.groupby("person_id")}
    trips_by = {pid: g for pid, g in enriched.groupby("person_id")}
    empty_days, empty_trips = pdays.iloc[0:0], enriched.iloc[0:0]

    tokens = seeding._forbidden_tokens()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

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

        lines, vis_days, vis_trips, n_obs = e7_tiers.tier_evidence(
            EVIDENCE_TIER, skeleton, pdd, ptr, row, None
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
            "observed": e7_tiers.tier_fidelity_observed(
                EVIDENCE_TIER, vis_days, vis_trips
            ),
            "observed_day_sequences": day_signatures(vis_days, vis_trips),
            "prompt_sha256": prompt_sha256(prompt),
        }

    with (out_root / "prompts.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    (out_root / "pop_context.json").write_text(json.dumps(context, default=str))

    manifest = {
        "eval": "METHOD-TRANSFER population build (A1.3, A8)",
        "date": date.today().isoformat(),
        "arena": "cityk_cordon",
        "evidence_tier": EVIDENCE_TIER,
        "pipeline": "evaluation.build_e7_tiers evidence path, applied unchanged",
        "has_pass_rule": (
            "forced False for every persona — the transponder-pass concept is "
            "corridor-instrument-specific and does not carry (A8.5(ii)); the "
            "cordon exemption structure is world structure, not method"
        ),
        "scoring_weights": {
            "variant": "adults-only M1 (owner ruling 2026-07-19; "
                       "runs/transfer_reweight_adultsM1/ADOPTION_RULING.md)",
            "weights_csv_sha256": _sha256(ADOPTED_WEIGHTS),
            "reweight_manifest_sha256": _sha256(ADOPTED_MANIFEST),
            "applied_at": "scoring time only (aggregation weights); "
                          "generation is unweighted and per-persona",
        },
        "counts": {"prompts": len(records), "lint_failures": n_lint_fail},
        "label": "METHOD-TRANSFER",
    }
    (out_root / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"transfer_pop: {len(records)} prompts ({n_lint_fail} lint failures)")
    if n_lint_fail:
        print("  WARNING: lint failures — inspect before shipping")
    return 0


def _read_rounds(paths) -> Dict[str, dict]:
    raw_by_pid: Dict[str, dict] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                raw_by_pid[str(rec["persona_id"])] = rec
    return raw_by_pid


def _gate(ctx, obj) -> List[str]:
    """Full five-gate compose (schema, mask-lint, replay-smell, feasibility,
    fidelity) against the full observed diary — the T5 gate, unchanged."""
    if obj is None:
        return ["no generation record"]
    return validate_card(
        obj, ctx["skeleton"], ctx["observed"], ctx["observed_day_sequences"]
    )


def cmd_retry(args) -> int:
    from grounding.seeding import build_retry_prompts

    dataset = psrc.load_or_build()
    rows = _transfer_rows(dataset)
    row_of = {str(r["persona_id"]): r for r in rows.to_dict("records")}
    pdays, enriched = _diary_frames(dataset)
    days_by = {pid: g for pid, g in pdays.groupby("person_id")}
    trips_by = {pid: g for pid, g in enriched.groupby("person_id")}
    empty_days, empty_trips = pdays.iloc[0:0], enriched.iloc[0:0]

    out_root = Path(args.out)
    context = json.loads((out_root / "pop_context.json").read_text())
    raw_by_pid = _read_rounds(args.generated)

    failures: List[dict] = []
    for pid, ctx in context.items():
        rec = raw_by_pid.get(pid)
        obj = rec.get("raw_json") if rec else None
        errs = _gate(ctx, obj)
        if not errs:
            continue
        row = row_of[pid]
        person_id = str(row["person_id"])
        lines, _vd, _vt, n_obs = e7_tiers.tier_evidence(
            EVIDENCE_TIER, ctx["skeleton"], days_by.get(person_id, empty_days),
            trips_by.get(person_id, empty_trips), row, None,
        )
        failures.append({
            "persona_id": pid,
            "skeleton": ctx["skeleton"],
            "evidence_lines": lines,
            "n_observed_days": n_obs,
            "failure_reasons": errs,
            "attempt": rec.get("attempt", 1) if rec else 1,
        })

    out_path = out_root / f"prompts_attempt{args.attempt}.jsonl"
    info = build_retry_prompts(failures, out_path)
    print(f"transfer_pop: {info['n_prompts']} retry prompts -> {out_path}")
    return 0


def cmd_assemble(args) -> int:
    dataset = psrc.load_or_build()
    rows = _transfer_rows(dataset)
    row_of = {str(r["persona_id"]): r for r in rows.to_dict("records")}
    pdays, enriched = _diary_frames(dataset)
    days_by = {pid: g for pid, g in pdays.groupby("person_id")}
    trips_by = {pid: g for pid, g in enriched.groupby("person_id")}
    empty_days, empty_trips = pdays.iloc[0:0], enriched.iloc[0:0]

    out_root = Path(args.out)
    context = json.loads((out_root / "pop_context.json").read_text())
    raw_by_pid = _read_rounds(args.generated)

    cards: List[dict] = []
    stats = {"accepted": 0, "fallback": 0, "gate_failures": {}}
    for pid, ctx in context.items():
        row = row_of[pid]
        person_id = str(row["person_id"])
        skeleton = ctx["skeleton"]
        rec = raw_by_pid.get(pid)
        obj = rec.get("raw_json") if rec else None
        errs = _gate(ctx, obj)
        if not errs:
            card = assemble_card(pid, skeleton, obj, {
                "card_source": "llm", "transfer_pop": True,
                "model": rec.get("model"), "attempt": rec.get("attempt", 1),
                "prompt_sha": rec.get("prompt_sha256"), "mask_lint": "v0.3",
            })
            cards.append(card)
            stats["accepted"] += 1
        else:
            card = e7_tiers.tier_fallback_card(
                EVIDENCE_TIER, pid, skeleton,
                days_by.get(person_id, empty_days),
                trips_by.get(person_id, empty_trips), row,
            )
            cards.append(card)
            stats["fallback"] += 1
            key = errs[0].split(":")[0][:40] if errs else "unknown"
            stats["gate_failures"][key] = stats["gate_failures"].get(key, 0) + 1

    out_path = out_root / "cards_transfer.jsonl"
    with out_path.open("w") as f:
        for c in cards:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    (out_root / "assemble_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"transfer_pop: {stats['accepted']} accepted, {stats['fallback']} "
          f"fallback -> {out_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--out", default="runs/transfer_pop")
    p.set_defaults(func=cmd_prompts)
    r = sub.add_parser("retry")
    r.add_argument("--out", default="runs/transfer_pop")
    r.add_argument("--generated", required=True, nargs="+")
    r.add_argument("--attempt", type=int, required=True)
    r.set_defaults(func=cmd_retry)
    a = sub.add_parser("assemble")
    a.add_argument("--out", default="runs/transfer_pop")
    a.add_argument("--generated", required=True, nargs="+")
    a.set_defaults(func=cmd_assemble)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
