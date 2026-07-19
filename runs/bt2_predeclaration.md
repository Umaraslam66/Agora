# BT2 pre-firing declaration — 2026-07-20, project owner ruling

Recorded and pushed BEFORE the firing (wall discipline):

1. **Walltime rule (owner ruling 2026-07-20, verbatim intent):** if the
   BT2 job is killed by walltime BEFORE the scoring phase reads any truth
   quantity, no blind quantity exists; resubmission is permitted and will
   be recorded openly. (The driver imports the quarantined truth module
   only after every simulation arm completes; a walltime kill before that
   point leaves the answer key untouched.)
2. **QOS:** the firing runs under `boost_qos_lprod` (4-day max wall;
   available to the project account) at `--time=48:00:00`, within the
   sealed one-node full-node compute discipline (A8.7(iv) analog). The
   phase timeline is NOT re-pinned.
3. Pre-go review completed by the owner: the real charge-ladder table in
   the harness-side derivation note verified against the institutional
   record; the A8.1 window constants reviewed and APPROVED.
4. Go given 2026-07-20, inline: `AGORA_BT1_AUTHORIZED=1 — fire BT2.`
