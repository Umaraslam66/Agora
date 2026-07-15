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

from pathlib import Path

__all__ = [
    "render_persona_prompt",
    "render_choice_prompt",
    "render_seed_prompt",
    "render_seed_retry_prompt",
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
