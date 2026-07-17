# Contributing to Agora

Thanks for contributing. This is a pre-registered research codebase; the rules
below are not conventions, they are the experiment's integrity. CI enforces
most of them.

## Never, under any circumstances

- **Never edit `01_PREREGISTRATION.md`.** It is frozen. Changes are append-only
  dated amendments in its §7 — the original text is never rewritten.
- **Never touch `evaluation/truth/`** from agent, world, grounding,
  calibration, serving, or training code — no imports, no path reads, no
  copies. The import-boundary test and a runtime tripwire enforce this
  (pre-registration §2). Blind results are scored once and sealed.
- **Never add a second render path.** All prompt construction goes through
  `grounding/render.py`. Re-export it; never re-implement it — the
  render-parity test fails any prompt/persona-building `def` outside
  `grounding/`.
- **Never put real-world identifiers in agent-visible text** — templates and
  agent-facing string literals must pass mask-lint against the versioned
  forbidden-token list. Mask the text; never weaken the list to get green.
- **Never commit personal usernames or emails** in any file, including job
  scripts and notebooks.

## Always

- **Test-first.** New behavior lands with tests; the day-one doctrine tests
  (`tests/test_render_parity.py`, `tests/test_mask_lint.py`,
  `tests/test_truth_import_boundary.py`) must never be skipped, xfailed, or
  softened to make a build green.
- **Mask-lint runs in CI** on every push; a forbidden-token hit fails the
  build (E5.iii). Changes to the token list require a dated version bump in
  the file header.
- **Negative results are kept.** If a gate fails or an ablation shows a
  mechanism is not load-bearing, that is written up in `docs/` — maintained
  privately and published with the results (this is a public repo) — with the
  same prominence as a pass. Deleting a negative result is falsification.
- **No unbacked numbers.** Any quantitative claim in README or docs must be
  reproducible from a file in `runs/`.

## Practicalities

- Python ≥ 3.11. `pip install -e ".[dev]"`, then `pytest -v` and
  `python -m grounding.masking.mask_lint` before pushing.
- `ruff check .` (line length 100) keeps style noise out of review.
