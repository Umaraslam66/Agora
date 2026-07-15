"""The single render path for Agora persona/world prompts.

Render-parity is a core doctrine (see 00_PROJECT_BRIEF.md and the frozen
01_PREREGISTRATION.md). ONE function renders any persona and any world state,
and it is used unchanged for BOTH training and serving. There is never a second
prompt-construction code path. In the predecessor project a train/serve
rendering distance confound silently flipped a verdict; this module exists so
that failure mode can never recur.

Guarantees:
  * Deterministic. Dict keys are emitted in canonical (sorted) order; there are
    no timestamps, no randomness, and no network access.
  * Template-sourced. Every human-readable line comes from a file in
    grounding/templates/; this module only substitutes masked, structured data.
  * Mode-neutral. ``mode`` selects an output envelope only; it must never change
    the rendered persona/world content. Today "train" and "serve" are identical.

The train and serve transplants MUST route their prompt construction through
this module (re-export ``render_persona_prompt`` as ``render_prompt``); the
render-parity test enforces that identity.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "render_persona_prompt",
    "render_choice_prompt",
    "render_seed_prompt",
    "render_seed_retry_prompt",
    "render_rewrite_prompt",
    "rewrite_render_inputs",
    "build_rewrite_prompt_records",
    "VALID_MODES",
]

VALID_MODES = ("train", "serve")

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _check_mode(mode: str) -> None:
    if mode not in VALID_MODES:
        raise ValueError(
            "render mode must be one of "
            + repr(VALID_MODES)
            + "; got "
            + repr(mode)
        )


def _load_template(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _fmt_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _render_block(mapping: dict) -> str:
    """Render a mapping as sorted ``  key: value`` lines (canonical order)."""
    if not mapping:
        return "  (none)"
    lines = []
    for key in sorted(mapping, key=str):
        value = mapping[key]
        if isinstance(value, dict):
            inner = ", ".join(
                f"{sub}={_fmt_scalar(value[sub])}" for sub in sorted(value, key=str)
            )
            lines.append(f"  {key}: {inner}")
        elif isinstance(value, (list, tuple)):
            inner = ", ".join(_fmt_scalar(v) for v in value)
            lines.append(f"  {key}: {inner}")
        else:
            lines.append(f"  {key}: {_fmt_scalar(value)}")
    return "\n".join(lines)


def _render_rules(rules) -> str:
    if not rules:
        return "  (none)"
    return "\n".join(f"  - {rule}" for rule in rules)


def _option_code(index: int, option: object) -> str:
    if isinstance(option, dict) and "id" in option:
        return str(option["id"])
    return chr(ord("A") + index) if index < 26 else str(index)


def _render_options(options) -> str:
    if not options:
        return "  (no options)"
    lines = []
    for index, option in enumerate(options):
        code = _option_code(index, option)
        if isinstance(option, dict):
            label = option.get("label", option.get("id", _fmt_scalar(option)))
        else:
            label = _fmt_scalar(option)
        lines.append(f"  {code}. {label}")
    return "\n".join(lines)


def _render_persona_card(persona: dict) -> str:
    fields = {
        "persona_id": persona.get("persona_id", "n/a"),
        "home_zone": persona.get("home_zone", "n/a"),
        "work_zone": persona.get("work_zone", "n/a"),
        "household_size": persona.get("household_size", "n/a"),
        "occupation": persona.get("occupation", "n/a"),
        "car_owner": _fmt_scalar(persona.get("car_owner", False)),
        "income_band": persona.get("income_band", "n/a"),
        "habits_block": _render_block(persona.get("habits", {})),
        "rules_block": _render_rules(persona.get("rules", [])),
    }
    return _load_template("persona_card.txt").format_map(fields).rstrip("\n")


def _render_world_state(world_state: dict) -> str:
    fields = {
        "city": world_state.get("city", "n/a"),
        "day_index": world_state.get("day_index", "n/a"),
        "toll_cordon_active": _fmt_scalar(world_state.get("toll_cordon_active", False)),
        "toll_fee_units": world_state.get("toll_fee_units", "n/a"),
        "currency": world_state.get("currency", "units"),
        "travel_times_block": _render_block(world_state.get("travel_times_min", {})),
    }
    return _load_template("world_state.txt").format_map(fields).rstrip("\n")


def render_persona_prompt(persona: dict, world_state: dict, mode: str) -> str:
    """Render a persona and world state into one deterministic prompt string.

    ``mode`` must be "train" or "serve". It selects an output envelope only and
    must never alter the persona/world content: for a given persona and world
    state, the "train" and "serve" outputs are byte-identical today.
    """
    _check_mode(mode)
    persona_text = _render_persona_card(persona)
    world_text = _render_world_state(world_state)
    return persona_text + "\n\n" + world_text + "\n"


def render_seed_prompt(
    skeleton: dict, evidence_lines: list, n_observed_days: int, mode: str
) -> str:
    """Render the slow brain's ONE persona-card-writing prompt.

    ``skeleton`` is the harness-computed masked demographic block (never
    LLM-invented); ``evidence_lines`` are pre-built deterministic habit
    summary strings from the seeding module (masked vocabulary only — the
    caller is responsible for the data, this module only formats). Same
    render-parity contract as every other prompt: ``mode`` selects an
    envelope only, "train" and "serve" outputs are byte-identical today.
    """
    _check_mode(mode)
    fields = {
        "skeleton_block": _render_block(skeleton),
        "evidence_block": _render_rules(evidence_lines),
        "n_observed_days": n_observed_days,
    }
    return _load_template("persona_seed.txt").format_map(fields)


def render_seed_retry_prompt(
    skeleton: dict,
    evidence_lines: list,
    n_observed_days: int,
    failure_reasons: list,
    mode: str,
) -> str:
    """Render the retry prompt for a persona whose first card failed validation.

    Same render-parity contract as :func:`render_seed_prompt`: this reuses that
    one path unchanged (there is still exactly one seed-prompt body) and appends
    a machine-readable failure block from a template, so a retry attempt differs
    from the first attempt ONLY by the appended list of rejection reasons (D7's
    validation-loop retry). ``failure_reasons`` are pre-built masked strings from
    the validators; this module only formats them.
    """
    _check_mode(mode)
    base = render_seed_prompt(skeleton, evidence_lines, n_observed_days, mode).rstrip("\n")
    retry = _load_template("persona_seed_retry.txt").format_map(
        {"failure_reasons": _render_rules(list(failure_reasons))}
    ).rstrip("\n")
    return base + "\n\n" + retry + "\n"


def render_choice_prompt(
    persona: dict, options: list, world_state: dict, mode: str
) -> str:
    """Render a discrete-choice query: persona card, world state, then options.

    This is the exact string a choice gateway would send to the model. It reuses
    ``render_persona_prompt`` so there is still exactly one render path.
    """
    _check_mode(mode)
    base = render_persona_prompt(persona, world_state, mode).rstrip("\n")
    choice_text = _load_template("choice_query.txt").format_map(
        {"options_block": _render_options(options)}
    ).rstrip("\n")
    return base + "\n\n" + choice_text + "\n"


# ---------------------------------------------------------------------------
# slow-brain REWRITE prompt (M3 D4) — a surprised persona's card is revised
# through this SAME single render path, so a rewrite prompt can never diverge
# from the seed prompt the model was trained/served on (render-parity).
# ---------------------------------------------------------------------------

# The trip/card fields the model owns; everything else on an assembled card
# (habit_counters, surprise_log, provenance, persona_id, skeleton, card_version)
# is harness state and is stripped before it is ever shown back to the model —
# the M2 lesson where persona-id digits and log fields tripped the mask/replay
# lints. The skeleton is shown separately in the RESIDENT block.
_CARD_VIEW_KEYS = ("patterns", "rules", "voice")

# One neutral evidence line for the record-builder's convenience path, when a
# caller has no pre-built seeding evidence to hand (the loop passes the real
# evidence lines through ``render_context``).
_DEFAULT_REWRITE_EVIDENCE = (
    "The current card is the working summary of this person's ordinary weekdays.",
)


def _card_view(obj: dict) -> dict:
    """The model-owned view of a card: patterns/rules/voice only."""
    return {k: obj[k] for k in _CARD_VIEW_KEYS if k in obj}


def _render_card_json(obj: dict) -> str:
    """Deterministic JSON of the model-owned card view (sorted keys).

    ensure_ascii=False is load-bearing: with the default ASCII escaping, a
    curly apostrophe in a card's voice text renders as a unicode escape
    sequence whose four digits happen to spell a forbidden bare wave-year
    token — the prompt would trip mask-lint on its own quoting artifact."""
    return json.dumps(_card_view(obj), sort_keys=True, indent=2, ensure_ascii=False)


def _fmt_minutes(value: object) -> str:
    return f"{float(value):.1f}"


def _fmt_share(value: object) -> str:
    return f"{float(value):.2f}"


def _render_surprise_block(entries) -> str:
    """Render the masked surprise block: context key, expected vs realized
    minutes, and the surprise magnitude z. Entries are plain dicts so the
    block is a pure function of masked scalars (no truth, no clock times)."""
    if not entries:
        return "  (none)"
    lines = []
    for e in entries:
        lines.append(
            "  - context {key}: expected {exp} min, experienced {real} min, "
            "gap z {z}".format(
                key=e["context_key"],
                exp=_fmt_minutes(e["expected_minutes"]),
                real=_fmt_minutes(e["realized_minutes"]),
                z=_fmt_share(e["z"]),
            )
        )
    return "\n".join(lines)


def _render_immutable_rules(rule_ids) -> str:
    ids = [str(rid) for rid in rule_ids]
    return ", ".join(ids) if ids else "(none)"


def _render_fit_check(fit: dict) -> str:
    """Render the FIT CHECK reference figures (the same numeric targets the
    seed prompt's fidelity gate enforces) from an ``observed_stats_of``-shaped
    mapping. Shares/means are two-decimal, so every emitted token is a single-
    or two-digit magnitude — never a year-like string."""
    lines = []
    if fit.get("mean_trips_per_weekday") is not None:
        lines.append(
            f"  mean trips per recorded weekday: "
            f"{float(fit['mean_trips_per_weekday']):.2f}"
        )
    shares = fit.get("mode_shares") or {}
    counts = fit.get("mode_counts") or {}
    if shares:
        mix = ", ".join(f"{m}={_fmt_share(shares[m])}" for m in sorted(shares))
        lines.append(f"  weekday mode mix (shares): {mix}")
    elif counts:
        mix = ", ".join(f"{m}={int(counts[m])}" for m in sorted(counts))
        lines.append(f"  weekday mode-use counts: {mix}")
    if fit.get("quiet_share") is not None:
        lines.append(f"  quiet weekday share: {float(fit['quiet_share']):.2f}")
    return "\n".join(lines) if lines else "  (no reference figures available)"


def render_rewrite_prompt(
    skeleton: dict,
    evidence_lines: list,
    current_obj: dict,
    surprise_block: list,
    immutable_rule_ids,
    fit_check_numbers: dict,
    mode: str = "serve",
    failure_reasons=(),
) -> str:
    """Render the slow brain's ONE card-REWRITE prompt (M3 D4).

    Reuses the seed prompt's evidence + FIT CHECK structure (same numeric
    targets, so a rewrite must still satisfy the card fidelity gate) and adds
    the current card (model-owned fields only — harness state stripped), a
    masked surprise block (``surprise_block`` is a list of dicts with keys
    ``context_key``, ``expected_minutes``, ``realized_minutes``, ``z``), and an
    explicit immutable-rules list. Same render-parity contract as every other
    prompt: ``mode`` selects an envelope only, "train" and "serve" are
    byte-identical today.

    ``failure_reasons`` (masked validator strings) appends the SAME retry block
    the seed path uses (persona_seed_retry.txt), so a retry attempt differs from
    the first ONLY by the appended rejection list — the D4 attempt-2 mechanic.
    """
    _check_mode(mode)
    fields = {
        "skeleton_block": _render_block(skeleton),
        "evidence_block": _render_rules(list(evidence_lines)),
        "current_card_json": _render_card_json(current_obj),
        "surprise_block": _render_surprise_block(surprise_block),
        "immutable_rules_block": _render_immutable_rules(immutable_rule_ids),
        "fit_check_block": _render_fit_check(fit_check_numbers),
    }
    base = _load_template("persona_rewrite.txt").format_map(fields).rstrip("\n")
    if failure_reasons:
        retry = _load_template("persona_seed_retry.txt").format_map(
            {"failure_reasons": _render_rules(list(failure_reasons))}
        ).rstrip("\n")
        return base + "\n\n" + retry + "\n"
    return base + "\n"


def _implied_stats(obj: dict) -> dict:
    """Fallback FIT CHECK figures derived from a card's own patterns, for the
    record-builder's convenience path when no observed reference is supplied.
    Mirrors the fidelity gate's weight-normalized implied quantities."""
    patterns = obj.get("patterns", []) or []
    total_w = 0.0
    trip_mass = 0.0
    quiet_mass = 0.0
    mode_mass: dict = {}
    mode_total = 0.0
    for p in patterns:
        w = p.get("weight")
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            continue
        w = float(w)
        trips = p.get("trips") or []
        total_w += w
        trip_mass += w * len(trips)
        if not trips:
            quiet_mass += w
        for t in trips:
            m = t.get("mode")
            mode_mass[m] = mode_mass.get(m, 0.0) + w
            mode_total += w
    if total_w <= 0:
        return {
            "mean_trips_per_weekday": 0.0,
            "mode_shares": {},
            "quiet_share": 0.0,
            "has_quiet_weekday": False,
        }
    shares = {m: mode_mass[m] / mode_total for m in mode_mass} if mode_total > 0 else {}
    return {
        "mean_trips_per_weekday": trip_mass / total_w,
        "mode_shares": shares,
        "quiet_share": quiet_mass / total_w,
        "has_quiet_weekday": quiet_mass > 0,
    }


def rewrite_render_inputs(request, render_context=None, observed=None) -> dict:
    """Assemble the six :func:`render_rewrite_prompt` arguments from one
    ``agents.two_brain.RewriteRequest`` plus optional per-persona render context.

    A request carries the current card (skeleton + model-owned fields), the
    surprises backing the trigger, and the immutable rule ids; the seeding
    evidence lines and the observed FIT CHECK figures are NOT on the request, so
    the loop supplies them through ``render_context`` (persona_id -> {"evidence_
    lines": [...], "fit_check_numbers": {...}}). When a persona is absent the
    FIT CHECK falls back to ``observed`` (an ``observed_stats_of`` mapping) and
    then to the card's own implied stats, so the builder stays callable with
    requests alone. Returns a kwargs dict for ``render_rewrite_prompt``.
    """
    card = request.card
    skeleton = {
        k: ("none" if v is None else v)
        for k, v in dict(card.get("skeleton", {})).items()
    }
    surprises = [
        {
            "context_key": s.context_key,
            "expected_minutes": s.expected_minutes,
            "realized_minutes": s.realized_minutes,
            "z": s.z,
        }
        for s in request.surprises
    ]
    ctx = (render_context or {}).get(request.persona_id, {})
    evidence_lines = list(ctx.get("evidence_lines") or _DEFAULT_REWRITE_EVIDENCE)
    fit = ctx.get("fit_check_numbers")
    if fit is None:
        fit = observed if observed is not None else _implied_stats(_card_view(card))
    return {
        "skeleton": skeleton,
        "evidence_lines": evidence_lines,
        "current_obj": _card_view(card),
        "surprise_block": surprises,
        "immutable_rule_ids": list(request.strong_rule_ids),
        "fit_check_numbers": fit,
    }


def build_rewrite_prompt_records(
    requests,
    render_context=None,
    mode: str = "serve",
    failures=None,
    observed_context=None,
) -> list:
    """Build ``{"persona_id", "prompt", "attempt"}`` records for the offline
    generation driver (serving/batch_gen) from a batch of rewrite requests.

    Lives here rather than in agents/ deliberately: the render-parity AST guard
    forbids any ``(render|build|format|make).*(prompt|persona)`` def outside
    grounding/, and prompt construction is grounding's alone. agents/slow_brain
    re-exports this name. ``render_context`` supplies each persona's seeding
    evidence + observed FIT CHECK figures (see :func:`rewrite_render_inputs`).

    ``failures`` (persona_id -> gate-failure strings from the prior attempt)
    and ``observed_context`` (persona_id -> observed stats) make the offline
    retry round render the SAME attempt-2 prompt the in-process client renders:
    bump each request's ``attempt`` and pass the terminal failures here.
    """
    records = []
    for request in requests:
        observed = (observed_context or {}).get(request.persona_id)
        inputs = rewrite_render_inputs(request, render_context, observed=observed)
        failure_reasons = tuple((failures or {}).get(request.persona_id, ()))
        prompt = render_rewrite_prompt(mode=mode, failure_reasons=failure_reasons, **inputs)
        records.append(
            {"persona_id": request.persona_id, "prompt": prompt, "attempt": request.attempt}
        )
    return records
