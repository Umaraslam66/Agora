# METHOD-TRANSFER population campaign — 2026-07-19 (public record)

Single full-information population (A1.3, A8), 11,940 personas, built by the
frozen generation pipeline (T5-shape evidence; has_pass forced to the
no-instrument value per A8.5(ii)). Scoring weights: the owner-adopted
adults-only-M1 raking weights (see ADOPTION_RULING in the reweight run dir),
sha-pinned in build_manifest.json.

- Leonardo jobs (full-node, 4x vLLM shards, Qwen/Qwen3-8B): 49844506
  (attempt 1, 11,940 prompts), 49846568 (attempt 2, 4,308), 49847143
  (attempt 3, 2,201); per-job manifests in runs/transfer_pop_gen{1,2,3}/.
- Assembly: 9,911 accepted (83.0%), 2,029 deterministic fallback (17.0%)
  — the sealed BT1 T5 population's profile was 9,930 / 2,010 (83.2%), i.e.
  the frozen pipeline reproduced its acceptance behavior in arena 2.
- OD-crossing (cordon) personas: 1,333 (11.2% unweighted; 8.7% weighted);
  LLM-card share among crossers 82.4% vs 83.0% overall.
- cards_transfer.jsonl sha256: e1de36c63d8803aa23d57f5e72d77fbb1f1f5ca43cea54a1e1622cd5c0809468
  (card set itself is record-derived and lives local + private mirror only)
- pop_context.json sha256: 804ece35fb57480be7e72f235a6d42c09e4f4e8201da07755805b91bc02cb4a4
