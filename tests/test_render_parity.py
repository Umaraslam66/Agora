"""Day-one doctrine test 1: RENDER PARITY.

Doctrine (00_PROJECT_BRIEF.md, "Transplants"/"What this project is"): in the
predecessor project, a train/serve rendering distance confound (route vs
beeline km) silently flipped a verdict. Therefore Agora has exactly ONE render
path — grounding.render — used identically for training and serving. No second
prompt-construction code path may ever exist.

Three enforcement layers:
  (a) the renderer exists, is deterministic, and mode never changes content;
  (b) serving and training re-export THE SAME function object (identity, not
      equality) — a copy-paste "equal" renderer is exactly the confound;
  (c) a static AST guard: no prompt/persona-building function may be DEFINED
      outside grounding/ (re-export/assignment is fine; ``def`` is not).

Part (b) is EXPECTED TO FAIL until the serving/training transplants route
their prompt construction through grounding.render. That loud red is the
required day-one state — do not skip, xfail, or soften it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import grounding.render as gr

REPO_ROOT = Path(__file__).resolve().parent.parent

PARITY_DOCTRINE = (
    "RENDER-PARITY VIOLATION: {pkg} does not route prompt construction through "
    "grounding.render — {pkg}.render_prompt must be grounding.render."
    "render_persona_prompt ITSELF (object identity, not an equal-looking copy). "
    "The train/serve distance confound in the predecessor project silently "
    "flipped a verdict; one render path, ever."
)


# ---------------------------------------------------------------------------
# (a) the single renderer exists, is deterministic, and mode-neutral
# ---------------------------------------------------------------------------

def test_renderer_exists_and_is_deterministic(sample_persona, sample_world_state):
    assert callable(getattr(gr, "render_persona_prompt", None)), (
        "grounding.render.render_persona_prompt is missing — this function IS "
        "the single render path; without it there is nothing for training and "
        "serving to share."
    )
    first = gr.render_persona_prompt(sample_persona, sample_world_state, mode="serve")
    second = gr.render_persona_prompt(sample_persona, sample_world_state, mode="serve")
    assert isinstance(first, str) and first
    assert first == second, (
        "DETERMINISM VIOLATION: rendering the same persona/world twice produced "
        "different strings. The renderer must be pure — canonical key order, no "
        "timestamps, no randomness — or training rows and serving prompts drift "
        "apart in ways no diff will ever show."
    )


def test_mode_does_not_change_content(sample_persona, sample_world_state):
    train = gr.render_persona_prompt(sample_persona, sample_world_state, mode="train")
    serve = gr.render_persona_prompt(sample_persona, sample_world_state, mode="serve")
    assert train == serve, (
        "RENDER-PARITY VIOLATION: mode='train' and mode='serve' rendered "
        "different persona/world content. Mode may only ever select an output "
        "envelope; the moment content depends on mode, the model trains on one "
        "world and is served another — the exact confound that silently flipped "
        "a verdict in the predecessor project."
    )


def test_choice_prompt_is_deterministic(sample_persona, sample_world_state, sample_options):
    a = gr.render_choice_prompt(sample_persona, sample_options, sample_world_state, "train")
    b = gr.render_choice_prompt(sample_persona, sample_options, sample_world_state, "serve")
    assert a == b, (
        "RENDER-PARITY VIOLATION: render_choice_prompt differs between train "
        "and serve mode. The choice gateway must send byte-identical context to "
        "what the model was trained on."
    )


# ---------------------------------------------------------------------------
# (b) serving and training must re-export the SAME function object
#     (EXPECTED RED today — transplants in flight)
# ---------------------------------------------------------------------------

def test_serving_reexports_the_one_renderer():
    import serving

    assert getattr(serving, "render_prompt", None) is gr.render_persona_prompt, (
        PARITY_DOCTRINE.format(pkg="serving")
    )


def test_training_reexports_the_one_renderer():
    import training

    assert getattr(training, "render_prompt", None) is gr.render_persona_prompt, (
        PARITY_DOCTRINE.format(pkg="training")
    )


# ---------------------------------------------------------------------------
# (c) static guard: nobody may DEFINE a prompt/persona builder outside grounding/
# ---------------------------------------------------------------------------

_FORBIDDEN_DEF = re.compile(r"(render|build|format|make).*(prompt|persona)", re.IGNORECASE)
_GUARDED_PACKAGES = ("agents", "serving", "training", "world", "calibration")


def _offending_defs(root: Path) -> list[str]:
    hits: list[str] = []
    for pkg in _GUARDED_PACKAGES:
        pkg_dir = root / pkg
        if not pkg_dir.is_dir():
            continue
        for path in sorted(pkg_dir.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError:
                continue  # in-flight file; it will be scanned once it parses
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _FORBIDDEN_DEF.search(node.name):
                        hits.append(f"{path.relative_to(root)}:{node.lineno}:def {node.name}")
    return hits


def test_no_prompt_builder_defined_outside_grounding():
    hits = _offending_defs(REPO_ROOT)
    assert not hits, (
        "RENDER-PARITY VIOLATION: prompt/persona-building function(s) DEFINED "
        "outside grounding/:\n  " + "\n  ".join(hits) + "\n"
        "Re-exporting grounding.render is fine; defining a second renderer is "
        "not. Two render paths is how the predecessor project trained on one "
        "world and served another without anyone noticing. Delete the def and "
        "route through grounding.render."
    )
