# RENDER-PARITY: serving has no prompt construction of its own. The one and
# only render path is grounding.render; this re-export is identity-checked by
# tests/test_render_parity.py (the predecessor project's train/serve prompt
# divergence silently flipped a verdict — one render path, ever).
from grounding.render import render_persona_prompt as render_prompt

__all__ = ["render_prompt"]
