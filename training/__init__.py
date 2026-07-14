# RENDER-PARITY: training data must be rendered by the same single render
# path that serves agents at runtime. This re-export is identity-checked by
# tests/test_render_parity.py; training code must never build prompt text.
from grounding.render import render_persona_prompt as render_prompt

__all__ = ["render_prompt"]
