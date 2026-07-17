"""Offline slow-brain rewrite round-trip driver (M3 rehearsal; M4 shock runs).

WHY THIS FILE EXISTS — the scored loop batches rewrite requests at day
boundaries, but the model lives on the GPU cluster: requests must leave the
loop as a prompts file, run through serving/batch_gen there (the SAME driver,
schema, and guided decoding as card generation), and come back through the
SAME five-gate acceptance the in-process client applies. This driver is the
local half of that round trip, plus the jam-injection rehearsal mode that
manufactures a real request batch on ordinary-day data so the rewrite path
can be validated end-to-end with the real model BEFORE any shock milestone
depends on it.

Three subcommands:

* ``collect`` — run the baseline loop on a corridor-driver subset with an
  injected facility closure (the jam), collect the organic RewriteRequests
  (client=None), dedupe to the first request per persona, and write
  ``requests_round1.json`` + ``prompts_round1.jsonl`` (batch_gen input shape).
* ``gate`` — join a round's requests with the cluster's ``cards_raw.jsonl``,
  run the full acceptance path (five validate_card gates + strong-rule
  immutability + apply_rewrite), and write accepted cards, gate stats, and —
  after round 1 — ``requests_round2.json`` + ``prompts_round2.jsonl`` with the
  failure feedback embedded (the attempt-2 mechanic, byte-identical to the
  in-process client's retry prompt).

Rounds are driven externally (ship prompts, sbatch jobs/gen_persona_cards.sbatch,
fetch outputs, gate) because the cluster is offline from this process.

Masking discipline: requests, prompts, and gate reasons are masked-clean by
construction (the prompt path is grounding.render's; the gate re-lints).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from agents.baseline_loop import run_baseline_loop
from agents.slow_brain import (
    MAX_REWRITE_ATTEMPTS,
    StandardSurprisePolicy,
    apply_rewrite,
    restore_strong_rules,  # the ONE acceptance definition (D4 revision)
)
from agents.two_brain import RewriteRequest, SurpriseEvent
from grounding.card_validation import validate_card, validate_card_structural
from grounding.render import build_rewrite_prompt_records
from serving.batch_gen import prompt_sha256
from world.bridge import population_from_cards


# ---------------------------------------------------------------------------
# request (de)serialization — mirrors agents.baseline_loop._request_to_dict
# ---------------------------------------------------------------------------

def request_from_dict(d: dict, attempt: Optional[int] = None) -> RewriteRequest:
    return RewriteRequest(
        persona_id=d["persona_id"],
        day_index=int(d["day_index"]),
        card=d["card"],
        surprises=tuple(
            SurpriseEvent(
                persona_id=s["persona_id"],
                day_index=int(s["day_index"]),
                context_key=s["context_key"],
                expected_minutes=float(s["expected_minutes"]),
                realized_minutes=float(s["realized_minutes"]),
                z=float(s["z"]),
            )
            for s in d["surprises"]
        ),
        strong_rule_ids=tuple(d["strong_rule_ids"]),
        attempt=int(attempt if attempt is not None else d.get("attempt", 1)),
        reason=d.get("reason", "surprise"),
        announcement=d.get("announcement"),
        shock_mode=bool(d.get("shock_mode", False)),
    )


def _load_json(path: str):
    return json.loads(Path(path).read_text())


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# shared context construction (reuses run_m3's builders — one definition)
# ---------------------------------------------------------------------------

def _contexts_for(card_personas: set):
    from evaluation.run_m3 import _validation_context
    from grounding import seeding
    from grounding.adapters import psrc

    dataset = psrc.load_or_build()
    persona_index = seeding.persona_index(dataset)
    enriched = seeding.enriched_trips(dataset)
    vctx = _validation_context(persona_index, dataset, enriched, card_personas)
    observed_ctx = {pid: c["observed"] for pid, c in vctx.items()}
    return dataset, vctx, observed_ctx


# ---------------------------------------------------------------------------
# collect: jam-injected loop on a corridor-driver subset
# ---------------------------------------------------------------------------

def cmd_collect(args) -> int:
    from evaluation import e1
    from evaluation.run_e2 import load_cards
    from grounding import seeding
    from world.config import cityk_corridor

    out = Path(args.out)
    cards = load_cards(args.cards)
    config = cityk_corridor()

    pop = population_from_cards(cards, config, namespace=args.namespace)
    corridor_cards = [c for c, on in zip(cards, pop.is_corridor) if on]
    drivers = [
        c for c in corridor_cards
        if any(t.get("mode") == "car" for p in c.get("patterns", []) for t in p.get("trips", []))
        and (c["skeleton"].get("household_cars") or 0) >= 1
        and c["skeleton"].get("can_drive", False)
    ]
    drivers = sorted(drivers, key=lambda c: c["persona_id"])[: args.subset]
    if not drivers:
        print("no corridor car-driver personas matched the subset filter", file=sys.stderr)
        return 2

    dataset, _, observed_ctx = _contexts_for({c["persona_id"] for c in drivers})
    persona_index = seeding.persona_index(dataset)
    pop_map = e1.persona_of_person_map(persona_index)
    all_slots = e1.day_slots_by_persona(dataset, pop_map)
    slots = {c["persona_id"]: all_slots.get(c["persona_id"], []) for c in drivers}

    # The jam: from --jam-day on, every corridor facility's capacity is cut by
    # --jam-capacity-factor (a physical incident/works shock, not a closure —
    # at these toy-corridor loads a closure only shifts travelers ~1 minute,
    # below the surprise threshold; a capacity cut makes the BPR term bite).
    # Implemented as a two-segment run through the loop's own checkpoint seam:
    # segment 1 lives days [0, jam_day) on the normal config, segment 2 resumes
    # bit-identically on a jammed-config copy.
    from dataclasses import replace as dc_replace

    policy = StandardSurprisePolicy(warmup_days=args.warmup)
    jammed_facilities = {
        code: dc_replace(f, capacity=f.capacity * args.jam_capacity_factor)
        for code, f in config.facilities.items()
    }
    jammed_config = dc_replace(config, facilities=jammed_facilities)

    seg1 = run_baseline_loop(
        drivers, config, slots, namespace=args.namespace, n_days=args.days,
        warmup_days=args.warmup, policy=policy, client=None,
        run_through_day=args.jam_day - 1,
    )
    res = run_baseline_loop(
        drivers, jammed_config, slots, namespace=args.namespace, n_days=args.days,
        warmup_days=args.warmup, policy=policy, client=None,
        resume_state=seg1.state,
    )

    first_per_persona: Dict[str, dict] = {}
    for req in res.pending_rewrites:
        first_per_persona.setdefault(req["persona_id"], req)
    requests = sorted(first_per_persona.values(), key=lambda r: r["persona_id"])

    reqs = [request_from_dict(d, attempt=1) for d in requests]
    prompts = build_rewrite_prompt_records(reqs, observed_context=observed_ctx)

    _write_json(out / "requests_round1.json", requests)
    _write_jsonl(out / "prompts_round1.jsonl", prompts)
    _write_json(out / "collect_manifest.json", {
        "cards_path": args.cards, "subset": args.subset,
        "n_corridor_driver_candidates": len(drivers),
        "jam": {"capacity_factor": args.jam_capacity_factor, "from_day": args.jam_day},
        "loop": {"namespace": args.namespace, "n_days": args.days, "warmup": args.warmup},
        "n_raw_requests": len(res.pending_rewrites),
        "n_deduped_requests": len(requests),
        "surprise_counts_by_day": res.surprise_counts
        if isinstance(res.surprise_counts, dict) else list(res.surprise_counts),
    })
    print(f"collect: {len(requests)} request(s) "
          f"({len(res.pending_rewrites)} raw) -> {out}/prompts_round1.jsonl")
    return 0


# ---------------------------------------------------------------------------
# gate: cluster outputs -> five-gate acceptance (+ attempt-2 prompts)
# ---------------------------------------------------------------------------

def cmd_gate(args) -> int:
    out = Path(args.out)
    round_no = args.round
    requests = _load_json(args.requests)
    raw_by_persona: Dict[str, dict] = {}
    with open(args.generated) as f:
        for line in f:
            rec = json.loads(line)
            raw_by_persona[str(rec["persona_id"])] = rec

    _, vctx, observed_ctx = _contexts_for({r["persona_id"] for r in requests})

    accepted_cards: List[dict] = []
    audit: List[dict] = []
    failures: Dict[str, List[str]] = {}
    rejected_requests: List[dict] = []

    n_with_restorations = 0
    for rd in requests:
        pid = rd["persona_id"]
        req = request_from_dict(rd, attempt=round_no)
        rec = raw_by_persona.get(pid)
        ctx = vctx.get(pid)
        errs: List[str]
        restored: List[str] = []
        obj = rec.get("raw_json") if rec else None
        if rec is None:
            errs = ["no generation record returned for this persona"]
        elif not rec.get("gen_ok") or obj is None:
            errs = [f"generation not ok (finish_reason={rec.get('finish_reason')})"]
        elif ctx is None:
            errs = ["no validation context for this persona; cannot gate the rewrite"]
        else:
            # D4 revision: strong-rule drift is mechanically REPAIRED
            # (restored verbatim, shadow-guarded), never a rejection; the
            # gates run on the repaired object. Shock-mode requests (A4.2(i))
            # take the structural-only compose, mirroring GatedSlowBrain.
            obj, restored = restore_strong_rules(obj, req.card, req.strong_rule_ids)
            if req.shock_mode:
                errs = validate_card_structural(
                    obj, ctx["skeleton"], ctx.get("observed"),
                    ctx["observed_day_sequences"],
                )
            else:
                errs = validate_card(
                    obj, ctx["skeleton"], ctx["observed"], ctx["observed_day_sequences"]
                )

        if restored:
            n_with_restorations += 1
        if not errs:
            new_card = apply_rewrite(
                req.card, obj, req.day_index, round_no,
                rec.get("model", "unknown"), rec.get("prompt_sha256"),
            )
            accepted_cards.append(new_card)
            audit.append({"persona_id": pid, "round": round_no, "accepted": True,
                          "strong_rules_restored": restored})
        else:
            failures[pid] = list(errs)
            rejected_requests.append(rd)
            audit.append({"persona_id": pid, "round": round_no, "accepted": False,
                          "gate_failures": errs, "strong_rules_restored": restored})

    _write_jsonl(out / f"accepted_cards_round{round_no}.jsonl", accepted_cards)
    stats = {
        "round": round_no,
        "n_requests": len(requests),
        "n_accepted": len(accepted_cards),
        "n_rejected": len(rejected_requests),
        "acceptance_rate": (len(accepted_cards) / len(requests)) if requests else None,
        "n_with_strong_rule_restorations": n_with_restorations,
        "failure_reason_counts": _reason_counts(failures),
        "audit": audit,
    }
    _write_json(out / f"gate_stats_round{round_no}.json", stats)

    if rejected_requests and round_no < MAX_REWRITE_ATTEMPTS:
        next_round = round_no + 1
        for rd in rejected_requests:
            rd["attempt"] = next_round
        reqs2 = [request_from_dict(d) for d in rejected_requests]
        prompts2 = build_rewrite_prompt_records(
            reqs2, observed_context=observed_ctx, failures=failures
        )
        _write_json(out / f"requests_round{next_round}.json", rejected_requests)
        _write_jsonl(out / f"prompts_round{next_round}.jsonl", prompts2)
        print(f"gate round {round_no}: {len(accepted_cards)} accepted, "
              f"{len(rejected_requests)} rejected -> prompts_round{next_round}.jsonl")
    else:
        print(f"gate round {round_no}: {len(accepted_cards)} accepted, "
              f"{len(rejected_requests)} terminally rejected (old cards stand)")
    return 0


def _reason_counts(failures: Dict[str, List[str]]) -> Dict[str, int]:
    """Coarse failure taxonomy: count by the leading word-group of each
    masked failure string (enough to see WHICH gate rejects without
    parsing free text downstream)."""
    counts: Dict[str, int] = {}
    for errs in failures.values():
        for e in errs:
            key = " ".join(str(e).split()[:3])
            counts[key] = counts.get(key, 0) + 1
    return counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="jam-injected loop -> round-1 requests + prompts")
    c.add_argument("--cards", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--subset", type=int, default=200)
    c.add_argument("--jam-day", type=int, default=12)
    c.add_argument("--jam-capacity-factor", type=float, default=0.12,
                   help="corridor capacity multiplier from --jam-day on (the jam)")
    c.add_argument("--warmup", type=int, default=10)
    c.add_argument("--days", type=int, default=17)
    c.add_argument("--namespace", default="m3_rehearsal")
    c.set_defaults(func=cmd_collect)

    g = sub.add_parser("gate", help="cluster outputs -> gated acceptance")
    g.add_argument("--requests", required=True)
    g.add_argument("--generated", required=True, help="cluster cards_raw.jsonl")
    g.add_argument("--out", required=True)
    g.add_argument("--round", type=int, required=True)
    g.set_defaults(func=cmd_gate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
