# docs/ — gate records, in the repo

Discipline inherited from the predecessor project (its specs lived off-repo
and drifted; that never happens here):

- **Gate records live IN the repo.** Every milestone ends with a scored eval;
  the record of that score — inputs, run manifests, verdict — is committed
  here. No off-repo spec, wiki, or doc may govern a decision.
- **Sealed verdicts are never overwritten.** Blind tests are scored once
  (pre-registration §2). Later re-runs are appended as clearly-labeled
  post-hoc analyses; the original verdict stays, verbatim.
- **Negative results are product.** A failed gate or a "mechanism not
  load-bearing" finding is written up with the same care and prominence as a
  pass, and kept forever. Deleting one is falsification.
- Any number cited in a doc must be reproducible from a file in `runs/`.
- Amendments to the pre-registration are append-only and dated, in
  `01_PREREGISTRATION.md` §7 — never edits to its body, and never here.
