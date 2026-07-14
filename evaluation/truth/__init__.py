"""Quarantined truth series (pre-registration §2).

This package holds the published outcome data for the anchoring natural
experiment. Nothing in world/, agents/, grounding/, calibration/, serving/, or
training/ may import it — enforced statically by
tests/test_truth_import_boundary.py and at runtime by the tripwire below.

The evaluation harness (and only the evaluation harness) sets
AGORA_EVAL_CONTEXT=1 before importing.
"""

import os as _os

if _os.environ.get("AGORA_EVAL_CONTEXT") != "1":
    raise RuntimeError(
        "evaluation.truth is quarantined (pre-registration §2). "
        "Set AGORA_EVAL_CONTEXT=1 in the evaluation harness only."
    )

del _os
