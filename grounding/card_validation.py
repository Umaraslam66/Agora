"""Persona-card validation, linting, replay-smell, feasibility, and assembly.

Every card is checked before use through four independent deterministic gates
(M2 architecture spec D1/D7), all pure stdlib:

* :func:`validate_card_json` — structural validation implementing the frozen
  grounding/card_schema.json semantics directly (types, enums, bounds,
  patterns, additionalProperties) with no jsonschema dependency. The
  harness-owned top-level fields (persona_id, skeleton, ...) are attached
  outside the LLM output and are stripped before validation, so both a raw
  generation object and a fully assembled card validate through the same call.
* :func:`lint_card_text` — mask-lint every string in the object against the
  versioned forbidden-token list (voice and ids can leak nothing).
* :func:`replay_smell` — flag enumeration-instead-of-compression: day/date/time
  references anywhere, exact reproduction of a multi-day person's per-day
  sequences, and pattern-count overflow (double-guarding the schema).
* :func:`feasibility` — reject car trips a person physically cannot make
  (no household vehicle, or not licensed).

Plus :func:`assemble_card` (attach the harness-owned fields, seed the habit
counters empty) and :func:`fallback_card` (deterministic template compression
of the evidence when generation terminally fails).

Masking discipline: no real place name, agency, date, or bare wave-year appears
in any literal or comment here (mask-lint gate).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from agents.habit_memory import HabitCounter
from grounding.masking.mask_lint import (
    default_token_path,
    lint_text,
    load_forbidden_tokens,
)

CARD_VERSION = "m2-1.0"

_SCHEMA_PATH = Path(__file__).resolve().parent / "card_schema.json"

# Harness-owned top-level fields attached outside the LLM output; stripped
# before schema validation so a full card and a raw generation both validate.
_HARNESS_KEYS = frozenset(
    {"card_version", "persona_id", "skeleton", "surprise_log", "habit_counters", "provenance"}
)


def _load_schema() -> dict:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_SCHEMA = _load_schema()


# ---------------------------------------------------------------------------
# a small deterministic subset json-schema validator (stdlib only)
# ---------------------------------------------------------------------------

def _type_ok(value, want: str) -> bool:
    if want == "object":
        return isinstance(value, dict)
    if want == "array":
        return isinstance(value, (list, tuple))
    if want == "integer":
        # JSON integers are not booleans (bool is an int subclass in Python).
        return isinstance(value, int) and not isinstance(value, bool)
    if want == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if want == "string":
        return isinstance(value, str)
    if want == "boolean":
        return isinstance(value, bool)
    return True


def _validate(instance, schema: Mapping, path: str, errors: List[str]) -> None:
    want = schema.get("type")
    if want is not None and not _type_ok(instance, want):
        errors.append(f"{path}: expected type {want}, got {type(instance).__name__}")
        return

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value {instance!r} not in allowed set {schema['enum']}")

    if want == "integer" or want == "number":
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")

    if want == "string":
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: string length {len(instance)} > maxLength {schema['maxLength']}")
        if "pattern" in schema and not re.match(schema["pattern"], instance):
            errors.append(f"{path}: {instance!r} does not match pattern {schema['pattern']}")

    if want == "array":
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: {len(instance)} items < minItems {schema['minItems']}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: {len(instance)} items > maxItems {schema['maxItems']}")
        item_schema = schema.get("items")
        if item_schema is not None:
            for i, item in enumerate(instance):
                _validate(item, item_schema, f"{path}[{i}]", errors)

    if want == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                errors.append(f"{path}: missing required property '{key}'")
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            errors.append(f"{path}: {len(instance)} properties < minProperties {schema['minProperties']}")
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props:
                _validate(value, props[key], f"{path}.{key}", errors)
            elif additional is False:
                errors.append(f"{path}: additional property '{key}' is not allowed")


def validate_card_json(obj) -> list[str]:
    """Structural validation against the frozen card schema. Empty list = valid.

    A full assembled card is accepted (harness-owned top-level fields are
    stripped first); a genuinely unknown top-level key is still rejected.
    """
    if not isinstance(obj, dict):
        return ["<root>: card must be a JSON object"]
    work = {k: v for k, v in obj.items() if k not in _HARNESS_KEYS}
    errors: List[str] = []
    _validate(work, _SCHEMA, "<card>", errors)
    return errors


# ---------------------------------------------------------------------------
# text lint
# ---------------------------------------------------------------------------

def _iter_strings(obj) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, Mapping):
        for value in obj.values():
            out.extend(_iter_strings(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            out.extend(_iter_strings(value))
    return out


def lint_card_text(obj) -> list[str]:
    """Mask-lint every string in the object (voice, ids, and every other string
    value) against the versioned forbidden-token list. Empty list = clean."""
    tokens = load_forbidden_tokens(default_token_path())
    out: List[str] = []
    for s in _iter_strings(obj):
        for v in lint_text(s, tokens):
            out.append(f"forbidden token {v.token!r} in string {s!r}")
    return out


# ---------------------------------------------------------------------------
# replay smell
# ---------------------------------------------------------------------------

_DAY_INDEX_RE = re.compile(r"day\s*\d", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
# No \b anchors: "_" is a word character, so a boundary-anchored pattern would
# miss a clock time embedded in an id like "leave_08:15".
_TIME_RE = re.compile(r"\d{1,2}:\d{2}")


def _pattern_sequences(patterns: Sequence[Mapping]) -> List[tuple]:
    seqs = []
    for p in patterns:
        seqs.append(
            tuple(
                (t.get("purpose"), t.get("mode"), t.get("depart_band"))
                for t in p.get("trips", [])
            )
        )
    return seqs


def replay_smell(obj, observed_day_sequences: Sequence[Sequence]) -> list[str]:
    """Flag enumeration-instead-of-compression (M2 spec D1).

    ``observed_day_sequences`` is one entry per observed weekday day, each a
    sequence of ``(purpose, mode, band)`` triples (empty for a no-trip day).
    """
    errors: List[str] = []
    patterns = obj.get("patterns", []) if isinstance(obj, Mapping) else []

    # (a) day/date/time references anywhere in the card's strings
    for s in _iter_strings(obj):
        if _DAY_INDEX_RE.search(s):
            errors.append(f"day-index reference in string {s!r}")
        if _TIME_RE.search(s):
            errors.append(f"clock-time (HH:MM) reference in string {s!r}")
        if _DATE_RE.search(s):
            errors.append(f"date-like reference in string {s!r}")

    # (c) >6 patterns (schema also catches this; double-guard)
    if len(patterns) > 6:
        errors.append(f"{len(patterns)} patterns exceed the cap of 6")

    # (b) multi-day exact reproduction (enumeration, not compression).
    # No-trip days are excluded from BOTH sides of the comparison: the seed
    # template itself MANDATES a no-trip pattern whenever quiet days are
    # observed, so its presence can never be evidence of memorization — the
    # enumeration smell is about the ACTIVE-day repertoire. Without this, a
    # two-day person with one quiet day and one active day could never be
    # represented faithfully (their unique faithful compression would always
    # be flagged).
    observed = [tuple((t[0], t[1], t[2]) for t in seq) for seq in observed_day_sequences]
    observed_active = [seq for seq in observed if seq]
    if len(observed_active) > 1:
        pat_seqs = [s for s in _pattern_sequences(patterns) if s]
        if len(pat_seqs) == len(observed_active) and sorted(pat_seqs) == sorted(observed_active):
            errors.append(
                "pattern set reproduces every observed day sequence one-for-one "
                "(enumeration instead of compression)"
            )
    return errors


# ---------------------------------------------------------------------------
# feasibility
# ---------------------------------------------------------------------------

def _car_allowed(skeleton: Mapping) -> bool:
    """Whether this person can physically make a car (driver) trip."""
    cars = skeleton.get("household_cars")
    can_drive = skeleton.get("can_drive", True)
    return (cars is None or cars >= 1) and bool(can_drive)


def feasibility(obj, skeleton: Mapping) -> list[str]:
    """Reject car trips the person cannot physically make: no household vehicle,
    or not licensed. This is the validation-time REJECT reason for the retry
    loop; the executor additionally coerces car->ride at run time as a belt."""
    if _car_allowed(skeleton):
        return []
    errors: List[str] = []
    for pi, p in enumerate(obj.get("patterns", [])):
        for ti, t in enumerate(p.get("trips", [])):
            if t.get("mode") == "car":
                errors.append(
                    f"patterns[{pi}].trips[{ti}]: car trip but household has no "
                    f"vehicle or person cannot drive"
                )
    for ri, r in enumerate(obj.get("rules", [])):
        if r.get("then", {}).get("mode") == "car":
            errors.append(
                f"rules[{ri}]: rule sets mode car but household has no vehicle "
                f"or person cannot drive"
            )
    return errors


# ---------------------------------------------------------------------------
# fidelity (the fifth gate — does the card reproduce the person's own diary?)
# ---------------------------------------------------------------------------
#
# Round-1 cards passed the four structural gates yet systematically distorted
# each person's OWN records (mean weekday trips/day -24.7%, mode-share drift,
# spurious quiet-day weight) — enough to fail E1/E2. This gate compares the
# card's weighted self-implied behaviour against a deterministic reference
# computed from the SAME frames the evidence builder showed the model
# (grounding.seeding.observed_stats_of), and its failure strings are the exact
# numeric feedback fed back verbatim through render_seed_retry_prompt.
#
# Tolerances (chosen against integer-weight granularity, weights 1..10, <=6
# patterns, so a faithful card can always pass — see tests/test_card_validation):
#   (a) mean trips/day : max(0.40, 0.15 x observed)
#   (b) mode-share TVD : 0.20   (skipped when the person made zero weekday trips)
#   (c) quiet share    : max(0.12, 0.60 x observed); a no-trip pattern is
#                        forbidden outright when no quiet weekday was observed.

_MODE_TVD_BAR = 0.20
_MEAN_TOL_FLOOR = 0.40
_MEAN_TOL_REL = 0.15
_QUIET_TOL_FLOOR = 0.12
_QUIET_TOL_REL = 0.60


def _n2(x: float) -> str:
    """Round to 2 decimals for masked-clean feedback. Every emitted magnitude
    (trips/day <= ~8+, shares/TVD in [0,1]) is a single- or two-digit value, so
    the rendered token can never be a 4-digit year-like string."""
    return f"{float(x):.2f}"


def _pattern_weight(p: Mapping) -> float:
    w = p.get("weight")
    if isinstance(w, bool) or not isinstance(w, (int, float)):
        return 0.0
    return float(w)


def _pattern_trips(p: Mapping) -> list:
    t = p.get("trips")
    return t if isinstance(t, list) else []


def _implied_mode_mass(patterns: Sequence[Mapping]) -> tuple:
    """(mode -> weighted trip mass, total weighted trip mass) over pattern
    trips, each trip weighted by its pattern weight."""
    mass: dict = {}
    total = 0.0
    for p in patterns:
        w = _pattern_weight(p)
        for t in _pattern_trips(p):
            m = t.get("mode")
            mass[m] = mass.get(m, 0.0) + w
            total += w
    return mass, total


def _shares_tvd(implied: Mapping, observed: Mapping) -> float:
    keys = set(implied) | set(observed)
    return 0.5 * sum(abs(implied.get(k, 0.0) - observed.get(k, 0.0)) for k in keys)


def _mode_drift_hint(implied: Mapping, observed: Mapping) -> str:
    """Name the single most over- and under-weighted mode (masked; mode names
    are frozen-taxonomy tokens, always safe to emit)."""
    keys = sorted(set(implied) | set(observed))
    diffs = [(k, implied.get(k, 0.0) - observed.get(k, 0.0)) for k in keys]
    over = max(diffs, key=lambda kv: kv[1])
    under = min(diffs, key=lambda kv: kv[1])
    parts = []
    if over[1] > 0.01:
        parts.append(f"you over-weight {over[0]}")
    if under[1] < -0.01:
        parts.append(f"you under-weight {under[0]}")
    return "; ".join(parts) if parts else "the mode mix is off"


def fidelity(obj, observed: Mapping) -> list[str]:
    """Fifth gate: does the card reproduce the person's OWN observed weekday
    diary? ``observed`` is :func:`grounding.seeding.observed_stats_of` output.
    Empty list = faithful. Failure strings carry the numbers in masked-clean
    form (2 decimals) so they feed render_seed_retry_prompt verbatim.

    A person with no observed weekday days yields no constraint (nothing to be
    faithful to); the mode check is additionally skipped for a person who made
    zero weekday trips.
    """
    errors: List[str] = []
    if not isinstance(obj, Mapping):
        return errors
    if not observed or not observed.get("n_observed_weekdays"):
        return errors

    patterns = obj.get("patterns", []) or []
    total_w = sum(_pattern_weight(p) for p in patterns)
    if total_w <= 0:  # schema gate owns missing/invalid weights
        return errors

    # (a) card-implied mean trips per weekday
    implied_mean = sum(_pattern_weight(p) * len(_pattern_trips(p)) for p in patterns) / total_w
    obs_mean = float(observed.get("mean_trips_per_weekday", 0.0))
    if abs(implied_mean - obs_mean) > max(_MEAN_TOL_FLOOR, _MEAN_TOL_REL * obs_mean):
        hint = (
            "shift weight toward fuller days"
            if implied_mean < obs_mean
            else "shift weight toward lighter days or trim trips from your patterns"
        )
        errors.append(
            f"card implies {_n2(implied_mean)} trips per weekday; the evidence "
            f"shows {_n2(obs_mean)} — {hint}"
        )

    # (b) card-implied mode shares vs observed (skip when no observed trips)
    if observed.get("n_observed_trips", 0) > 0 and observed.get("mode_shares"):
        mass, mtot = _implied_mode_mass(patterns)
        implied_shares = {m: mass[m] / mtot for m in mass} if mtot > 0 else {}
        obs_shares = dict(observed["mode_shares"])
        t = _shares_tvd(implied_shares, obs_shares)
        if t > _MODE_TVD_BAR:
            errors.append(
                f"card's weekday mode mix is off (variation distance {_n2(t)} "
                f"exceeds the allowed {_n2(_MODE_TVD_BAR)}); "
                f"{_mode_drift_hint(implied_shares, obs_shares)} — match the "
                f"observed mode counts without adding variety the evidence does "
                f"not show or shortchanging the dominant mode"
            )

    # (c) quiet-day discipline
    implied_quiet = sum(
        _pattern_weight(p) for p in patterns if not _pattern_trips(p)
    ) / total_w
    if not observed.get("has_quiet_weekday", False):
        if any(not _pattern_trips(p) for p in patterns):
            errors.append(
                "card includes a no-trip pattern but every recorded weekday had "
                "trips — remove the empty pattern"
            )
    else:
        obs_quiet = float(observed.get("quiet_share", 0.0))
        if abs(implied_quiet - obs_quiet) > max(_QUIET_TOL_FLOOR, _QUIET_TOL_REL * obs_quiet):
            errors.append(
                f"card's no-trip share is {_n2(implied_quiet)}; the evidence "
                f"shows {_n2(obs_quiet)} quiet weekdays — adjust the no-trip "
                f"pattern weight"
            )
    return errors


# ---------------------------------------------------------------------------
# composed validation entry point (the ops pipeline's single gate call)
# ---------------------------------------------------------------------------

def _is_fallback(obj) -> bool:
    return (
        isinstance(obj, Mapping)
        and isinstance(obj.get("provenance"), Mapping)
        and obj["provenance"].get("card_source") == "fallback"
    )


def validate_card(
    obj,
    skeleton: Mapping,
    observed: Mapping,
    observed_day_sequences: Sequence[Sequence],
) -> list[str]:
    """Run all five gates in order and return the concatenated reasons (empty =
    the card passes). This is the single entry point the generation ops compose
    for the D7 validation loop; the returned strings are the machine-readable
    failure block the retry prompt echoes verbatim.

    Order: schema -> mask-lint -> replay-smell -> feasibility -> fidelity.

    Fallback cards (``provenance.card_source == "fallback"``) are EXEMPT from
    the fifth (fidelity) gate. The fidelity gate exists only to drive LLM
    retries with numeric feedback, and a fallback card is the terminal safety
    net with no retry behind it: it is a deterministic empirical resampling of
    the person's own day-signatures whose only residual infidelity is (i) the
    anti-enumeration fold that compresses all-distinct multi-day repertoires and
    (ii) the >8-trip signature truncation — both inherent to its construction
    and both flagged (``card_source="fallback"``, ``signature_truncated``). It
    still passes the four structural gates.
    """
    errors = validate_card_json(obj)
    if not isinstance(obj, Mapping):
        return errors
    errors = (
        errors
        + lint_card_text(obj)
        + replay_smell(obj, observed_day_sequences)
        + feasibility(obj, skeleton)
    )
    if not _is_fallback(obj):
        errors = errors + fidelity(obj, observed)
    return errors


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def assemble_card(persona_id: str, skeleton: Mapping, llm_obj: Mapping, provenance: Mapping) -> dict:
    """Attach the harness-owned fields to a validated LLM output object, seeding
    an empty HabitCounter for every pattern id AND rule id (D1)."""
    patterns = list(llm_obj.get("patterns", []))
    rules = list(llm_obj.get("rules", []))
    ids = [p["id"] for p in patterns] + [r["id"] for r in rules]
    habit_counters = {i: HabitCounter().to_dict() for i in ids}
    return {
        "card_version": CARD_VERSION,
        "persona_id": persona_id,
        "skeleton": dict(skeleton),
        "patterns": patterns,
        "rules": rules,
        "voice": llm_obj.get("voice", ""),
        "surprise_log": [],
        "habit_counters": habit_counters,
        "provenance": dict(provenance),
    }


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------

_FALLBACK_VOICE = "I get around in the plain, steady way my usual routine calls for."


def day_signatures(person_days, trips) -> List[tuple]:
    """One signature per observed weekday day, in day order: a tuple of
    ``(purpose, mode, band)`` triples (empty tuple for a no-trip day). Shared by
    the fallback compression and the replay check."""
    import pandas as pd  # local import keeps the module import light

    pd_df = person_days if isinstance(person_days, pd.DataFrame) else pd.DataFrame(list(person_days))
    tr_df = trips if isinstance(trips, pd.DataFrame) else pd.DataFrame(list(trips))

    daynums = sorted({int(d) for d in pd_df["daynum"]}) if "daynum" in pd_df and len(pd_df) else []
    by_day: dict = {d: [] for d in daynums}
    if len(tr_df):
        ordered = tr_df.sort_values(["daynum", "tripnum"]) if "tripnum" in tr_df else tr_df.sort_values(["daynum"])
        for r in ordered.itertuples(index=False):
            d = int(r.daynum)
            by_day.setdefault(d, []).append((r.purpose, r.mode, r.band))
    return [tuple(by_day[d]) for d in daynums]


def _weight_from_count(count: int, max_count: int) -> int:
    if max_count <= 0:
        return 1
    return max(1, min(10, round(count / max_count * 10)))


def fallback_card(persona_id: str, skeleton: Mapping, person_days, trips) -> dict:
    """Deterministic template compression of the evidence when generation
    terminally fails (D7). Clusters the person's observed weekday day-signatures
    into <=6 most-frequent signatures with integer weights 1..10 proportional to
    frequency (always keeping a no-trip pattern if one was observed), no rules,
    a fixed neutral voice, provenance ``card_source="fallback"``.
    """
    sigs = day_signatures(person_days, trips)

    # Trip-cap belt: a pattern's trips list is schema-capped at 8, but a person
    # can record a weekday with more than 8 trips. Truncate any over-long
    # day-signature to its first 8 trips deterministically (day_signatures
    # already orders trips by tripnum) so the fallback card stays schema-valid;
    # flag it in provenance. The residual infidelity from dropping trips 9+ is
    # the documented cost of the terminal safety net.
    signature_truncated = any(len(sig) > 8 for sig in sigs)
    if signature_truncated:
        sigs = [sig[:8] for sig in sigs]

    # Feasibility belt: the diary can contain car trips a person's skeleton
    # says they cannot make (e.g. a zero-vehicle household reporting a borrowed
    # car). The fallback is terminal — there is no retry behind it — so coerce
    # those to ride here, exactly as the executor would at run time, and the
    # fallback card passes every validation gate including feasibility.
    if not _car_allowed(skeleton):
        sigs = [
            tuple((p, "ride" if m == "car" else m, b) for (p, m, b) in sig)
            for sig in sigs
        ]

    counts = Counter(sigs)

    if not counts:
        patterns = [{"id": "quiet_day", "weight": 1, "trips": []}]
    else:
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

        # Anti-enumeration guard: a multi-day person whose every ACTIVE day is
        # a distinct signature would otherwise reproduce the observed active
        # days one-for-one, which the replay lint flags as enumeration instead
        # of compression. Fold the least-frequent non-empty signature into the
        # most-frequent non-empty one (deterministic tie-break by signature
        # sort order) so the card genuinely compresses — fewer active patterns
        # than active days. The no-trip pattern is never the one dropped, and
        # a single-active-signature card is left alone (the replay lint
        # likewise ignores no-trip days, so it is not enumeration).
        active_days = [s for s in sigs if s]
        if len(active_days) > 1 and len(set(active_days)) == len(active_days):
            non_empty = [i for i, (sig, _) in enumerate(ordered) if sig != tuple()]
            if len(non_empty) >= 2:
                drop_i = non_empty[-1]
                keep_i = non_empty[0]
                sig_drop, c_drop = ordered[drop_i]
                sig_keep, c_keep = ordered[keep_i]
                ordered[keep_i] = (sig_keep, c_keep + c_drop)
                ordered.pop(drop_i)
                ordered = sorted(ordered, key=lambda kv: (-kv[1], kv[0]))

        selected = ordered[:6]
        empty = tuple()
        if empty in counts and all(sig != empty for sig, _ in selected):
            selected = selected[:5] + [(empty, counts[empty])]
        max_count = max(c for _, c in selected)
        patterns = []
        for idx, (sig, c) in enumerate(selected):
            pid = "quiet_day" if sig == empty else f"pattern{idx + 1}"
            trips_list = [
                {"purpose": p, "mode": m, "depart_band": b} for (p, m, b) in sig
            ]
            patterns.append({"id": pid, "weight": _weight_from_count(c, max_count), "trips": trips_list})

    llm_obj = {"patterns": patterns, "rules": [], "voice": _FALLBACK_VOICE}
    provenance = {"card_source": "fallback", "attempt": None}
    if signature_truncated:
        provenance["signature_truncated"] = True
    return assemble_card(persona_id, skeleton, llm_obj, provenance)
