#!/usr/bin/env bash
# BT1 blind-shock firing guard — see CLAUDE.md > "Blind tests fire ONCE".
#
# PreToolUse(Bash) hook. Reads the tool-call JSON on stdin and BLOCKS any command
# that runs the BT1 / M4 blind-shock scoring entrypoint UNLESS the command carries
# the project owner's explicit inline authorization flag AGORA_BT1_AUTHORIZED=1.
#
# Scoring the blind arena against evaluation/truth/ seals the verdict permanently
# (pre-registration §2 wall); it must happen once, only on the owner's word, never
# as a side effect. This guard is the enforcement CLAUDE.md itself cannot provide.
#
# Deliberately narrow — it matches only BT1/blind-shock *entrypoints*, so it does
# NOT block `pytest` (incl. tests/test_blind_shock.py), the non-blind SR 520
# rehearsal, or ordinary evals (run_e1/run_e2/run_e5/run_m3). Pure bash: no jq or
# python dependency, cheap enough to run on every Bash call.
#
# WHEN THE BT1 DRIVER IS WRITTEN: add its module/path to ENTRYPOINT_RE below.
set -euo pipefail

payload="$(cat)"

# BT1/blind-shock scoring entrypoints (module form `-m evaluation.X` or path
# `evaluation/X.py`). run_m4/run_bt1/run_e4 are the anticipated driver names.
# run_bt2 = the transfer-arena (Stockholm) driver, pre-registered here on the
# owner's ruling 2026-07-19 BEFORE any driver code exists (same discipline as
# BT1: the name enters the matcher first).
ENTRYPOINT_RE='(-m[[:space:]]+evaluation\.(run_m4|run_bt1|run_bt2|run_e4|blind_shock)|evaluation/(run_m4|run_bt1|run_bt2|run_e4|blind_shock)\.py)'

# Fast path: not a BT1 entrypoint -> allow instantly.
printf '%s' "$payload" | grep -Eq "$ENTRYPOINT_RE" || exit 0

# Explicit owner authorization present inline -> allow the one-time firing.
printf '%s' "$payload" | grep -q 'AGORA_BT1_AUTHORIZED=1' && exit 0

# Otherwise deny via a PreToolUse permission decision (exit 0; the JSON blocks).
cat <<'JSON'
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"BT1 / M4 blind-shock scoring is HELD. It fires ONCE, only on the project owner's explicit authorization, never automatically (pre-registration §2 wall; §7). Scoring against evaluation/truth/ seals the verdict permanently — there is no re-run. To fire it deliberately as the owner, re-run with the inline flag: AGORA_BT1_AUTHORIZED=1 <command>. See CLAUDE.md > 'Blind tests fire ONCE'."}}
JSON
exit 0
