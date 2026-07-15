"""Day-one doctrine test 3: THE TRUTH IMPORT BOUNDARY (pre-registration §2).

Pre-registration §2 ("The wall"): "nothing measured in P2/P3 may influence any
tuning, prompt, correction, threshold, or model choice. The truth series lives
in `evaluation/truth/`, which agent/serving code cannot import (enforced by
test)."

If agent or serving code can read the truth series, every blind score is
worthless — the prediction could have been fitted to its own answer key. Two
enforcement layers: a static AST scan of all agent-side packages, and a
runtime tripwire in evaluation/truth/__init__.py that refuses to import
outside the evaluation harness (AGORA_EVAL_CONTEXT=1).
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BAD_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bad_import_fixture.py"
BAD_UNMASKED_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "bad_unmasked_vocab_fixture.py"
)

_GUARDED_PACKAGES = ("world", "agents", "grounding", "calibration", "serving", "training")

WALL_DOCTRINE = (
    'TRUTH-BOUNDARY VIOLATION (pre-registration §2, "The wall"): "nothing '
    "measured in P2/P3 may influence any tuning, prompt, correction, threshold, "
    "or model choice. The truth series lives in `evaluation/truth/`, which "
    'agent/serving code cannot import (enforced by test)." The truth series may '
    "never influence agent/serving code — a single import makes every blind "
    "score unfalsifiable. Offending reference(s):\n  "
)


def scan_file_for_truth_refs(path: Path) -> list[str]:
    """Return 'file:line:detail' hits for any reference to evaluation.truth.

    Catches: ``import evaluation.truth``, ``from evaluation.truth import ...``,
    ``from evaluation import truth``, and any string literal containing
    ``evaluation.truth`` or ``evaluation/truth`` (which covers importlib and
    path-based dodges).
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []  # in-flight file; it will be scanned once it parses
    hits: list[str] = []

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "evaluation.truth" or alias.name.startswith(
                    "evaluation.truth."
                ):
                    hits.append(f"{rel(path)}:{node.lineno}:import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "evaluation.truth" or module.startswith("evaluation.truth."):
                hits.append(f"{rel(path)}:{node.lineno}:from {module} import ...")
            elif module == "evaluation" and any(a.name == "truth" for a in node.names):
                hits.append(f"{rel(path)}:{node.lineno}:from evaluation import truth")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "evaluation.truth" in node.value or "evaluation/truth" in node.value:
                hits.append(
                    f"{rel(path)}:{node.lineno}:string literal referencing "
                    "evaluation.truth"
                )
    return hits


# ---------------------------------------------------------------------------
# (a) static scan: no agent-side package touches the truth
# ---------------------------------------------------------------------------

def test_no_agent_side_code_references_truth():
    hits: list[str] = []
    for pkg in _GUARDED_PACKAGES:
        pkg_dir = REPO_ROOT / pkg
        if not pkg_dir.is_dir():
            continue
        for path in sorted(pkg_dir.rglob("*.py")):
            hits.extend(scan_file_for_truth_refs(path))
    assert not hits, WALL_DOCTRINE + "\n  ".join(hits)


# ---------------------------------------------------------------------------
# (b) self-test: the scanner must catch a planted violation
# ---------------------------------------------------------------------------

def test_scanner_has_teeth_on_bad_fixture():
    hits = scan_file_for_truth_refs(BAD_FIXTURE)
    assert hits, (
        "TRUTH-BOUNDARY SCANNER IS BLIND: tests/fixtures/bad_import_fixture.py "
        "contains `from evaluation.truth import series` and the scanner "
        "reported nothing. A wall nobody checks is not a wall — pre-registration "
        "§2 is unenforced until this passes."
    )
    assert any("from evaluation.truth import" in h for h in hits)


# ---------------------------------------------------------------------------
# (c) runtime tripwire in evaluation/truth/__init__.py, both directions
# ---------------------------------------------------------------------------
# Fresh subprocesses avoid sys.modules caching and env leakage between the two
# directions.

def _import_truth_in_subprocess(extra_env: dict) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, **extra_env}
    env.pop("AGORA_EVAL_CONTEXT", None)
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", "import evaluation.truth"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_truth_import_blocked_outside_eval_context():
    proc = _import_truth_in_subprocess({})
    assert proc.returncode != 0 and "quarantined" in proc.stderr, (
        "TRUTH TRIPWIRE DISARMED: `import evaluation.truth` succeeded without "
        "AGORA_EVAL_CONTEXT=1. The runtime tripwire is the last line of defense "
        "when static scanning is dodged (pre-registration §2). Expected "
        "RuntimeError('evaluation.truth is quarantined (pre-registration §2). "
        "Set AGORA_EVAL_CONTEXT=1 in the evaluation harness only.').\n"
        f"stderr was:\n{proc.stderr}"
    )
    assert "RuntimeError" in proc.stderr


# ---------------------------------------------------------------------------
# (d) the E5 UNMASKED VOCABULARY shares the quarantine (sealed A2.3 / spec D9)
# ---------------------------------------------------------------------------
# evaluation/truth/unmasked_vocabulary.py deliberately names the real arena
# (real place/facility names, dates, prices) for the E5(i) unmasked arm. It is
# a submodule of evaluation.truth, so the static scan above and the runtime
# tripwire both already cover it — these tests pin that coverage explicitly so
# a future refactor cannot silently move the vocabulary out from under the
# wall: grounding/, agents/, serving/, world/ (and calibration/, training/)
# must never be able to import it.


def test_scanner_catches_unmasked_vocabulary_import():
    hits = scan_file_for_truth_refs(BAD_UNMASKED_FIXTURE)
    assert hits, (
        "UNMASKED-VOCABULARY QUARANTINE IS BLIND: tests/fixtures/"
        "bad_unmasked_vocab_fixture.py imports "
        "evaluation.truth.unmasked_vocabulary and the scanner reported "
        "nothing. If agent-side code can read the unmasked vocabulary, the "
        "masked arm of E5(i) is contaminated by construction (A2.3 / D9)."
    )
    assert any("unmasked_vocabulary" in h for h in hits)


def test_no_agent_side_code_references_unmasked_vocabulary():
    hits: list[str] = []
    for pkg in _GUARDED_PACKAGES:
        pkg_dir = REPO_ROOT / pkg
        if not pkg_dir.is_dir():
            continue
        for path in sorted(pkg_dir.rglob("*.py")):
            hits.extend(scan_file_for_truth_refs(path))
    offenders = [h for h in hits if "unmasked" in h]
    # any evaluation.truth reference is already fatal above; this pins the
    # unmasked-vocabulary case by name so its failure message is unmissable
    assert not offenders, WALL_DOCTRINE + "\n  ".join(offenders)


def _import_unmasked_in_subprocess(extra_env: dict) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ}
    env.pop("AGORA_EVAL_CONTEXT", None)
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", "import evaluation.truth.unmasked_vocabulary"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_unmasked_vocabulary_blocked_outside_eval_context():
    proc = _import_unmasked_in_subprocess({})
    assert proc.returncode != 0 and "quarantined" in proc.stderr, (
        "UNMASKED-VOCABULARY TRIPWIRE DISARMED: "
        "`import evaluation.truth.unmasked_vocabulary` succeeded without "
        "AGORA_EVAL_CONTEXT=1. The package tripwire must fire for every "
        "submodule of evaluation.truth (pre-registration §2; A2.3/D9).\n"
        f"stderr was:\n{proc.stderr}"
    )


def test_unmasked_vocabulary_importable_inside_eval_context():
    proc = _import_unmasked_in_subprocess({"AGORA_EVAL_CONTEXT": "1"})
    assert proc.returncode == 0, (
        "UNMASKED-VOCABULARY TRIPWIRE OVER-ARMED: the E5 harness itself "
        "(AGORA_EVAL_CONTEXT=1) cannot import the unmasked vocabulary — the "
        "wall must keep agents out, not lock the contamination probe out.\n"
        f"stderr was:\n{proc.stderr}"
    )


def test_truth_import_allowed_inside_eval_context(monkeypatch):
    proc = _import_truth_in_subprocess({"AGORA_EVAL_CONTEXT": "1"})
    assert proc.returncode == 0, (
        "TRUTH TRIPWIRE OVER-ARMED: the evaluation harness itself (with "
        "AGORA_EVAL_CONTEXT=1) cannot import evaluation.truth — the wall must "
        "keep agents out, not lock the scorer out.\n"
        f"stderr was:\n{proc.stderr}"
    )
    # and in-process, for the harness path actually used by evaluation code
    monkeypatch.setenv("AGORA_EVAL_CONTEXT", "1")
    sys.modules.pop("evaluation.truth", None)
    module = importlib.import_module("evaluation.truth")
    assert module is not None
    sys.modules.pop("evaluation.truth", None)
