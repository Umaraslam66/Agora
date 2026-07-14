#!/usr/bin/env python3
"""Held-out mode-choice accuracy by candidate log-likelihood ranking.

For each held-out trip, score every available mode by the summed logprob
of its tokens given the chat prompt (one teacher-forced forward per
candidate, batched per example) and predict the argmax. Deterministic —
no sampling, no constrained-decoding dependency — and works identically
for a base model or base+LoRA adapter, so the same script produces the
"baseline to beat" numbers (base Qwen3-8B) and the fine-tuned numbers.

Metrics: top-1 accuracy (unweighted + survey-trip-weighted), majority-
class floor, per-mode recall, predicted vs. actual mode shares.

  python3 eval_choice.py --model Qwen/Qwen3-8B [--adapter DIR] \
      --pairs pairs.jsonl --split test --out metrics.json

RENDER-PARITY: prompt text must originate from grounding.render. This
module never constructs persona/world prompt text itself: evaluation
prompts arrive fully rendered in the pairs JSONL (rendered upstream by
the single render path), and the synthetic probe/fare text used by
--habit-probe / --probe-fare must be supplied by an injectable renderer
(--renderer module[:attr]) backed by that same render path. This file
deliberately does NOT import grounding.render.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict

import torch


# ---------------------------------------------------------------------------
# Injectable renderer seam
# ---------------------------------------------------------------------------
# RENDER-PARITY: prompt text must originate from grounding.render. The
# predecessor project imported its probe-line and fare-line builders from
# sibling pipeline modules; here they are injected instead, so there is
# exactly one render path shared by training-pair generation, serving and
# these probes. The renderer object (module or attribute) may provide:
#   probe_lines(condition, target_mode, other_mode, employed) -> [str]
#       REQUIRED for --habit-probe / --probe-habit (incl. blend sweep).
#   fare_line(amount) -> str          e.g. " A monthly transit pass costs
#       N <currency units>."  REQUIRED for --fare / --probe-fare.
#   fare_marker: str                  substring present in every fare_line
#       output, used to refuse double insertion. REQUIRED with fare_line.
#   belief_lines(condition, m0_phrase, a_phrase) -> [str]   optional
#       override of the in-file default below.
#   mode_phrase: dict mode -> phrase  optional override of MODE_PHRASE.
#   mode_order: list of modes         optional override of MODE_ORDER.
_RENDERER = None


def set_renderer(obj):
    global _RENDERER
    _RENDERER = obj


def load_renderer(spec):
    """--renderer 'pkg.mod' (module is the renderer) or 'pkg.mod:attr'."""
    import importlib
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr) if attr else mod


def _renderer_attr(name, feature):
    if _RENDERER is None or not hasattr(_RENDERER, name):
        raise RuntimeError(
            "%s requires --renderer providing %r; prompt text must "
            "originate from the project's single render path "
            "(RENDER-PARITY)" % (feature, name))
    return getattr(_RENDERER, name)


def probe_lines(condition, target_mode, other_mode, employed):
    # RENDER-PARITY: prompt text must originate from grounding.render.
    fn = _renderer_attr("probe_lines", "--habit-probe/--probe-habit")
    return fn(condition, target_mode, other_mode, employed)


def fare_line(amount):
    # RENDER-PARITY: prompt text must originate from grounding.render.
    fn = _renderer_attr("fare_line", "--fare/--probe-fare")
    return fn(amount)


def fare_marker():
    return _renderer_attr("fare_marker", "--fare/--probe-fare")


def load_model_bf16(name):
    """Same loader as lora_sft: CausalLM auto-class first, then the VL
    auto-class for *ForConditionalGeneration archs."""
    import transformers
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            try:
                model = cls.from_pretrained(name, dtype=torch.bfloat16)
            except TypeError:
                model = cls.from_pretrained(name, torch_dtype=torch.bfloat16)
            print(f"[load] {name} via {cls_name}")
            return model
        except ValueError as e:
            print(f"[load] {cls_name} rejected {name}: {str(e)[:120]}")
    raise RuntimeError(f"no auto class could load {name}")


@torch.no_grad()
def score_candidates(model, tok, prompt_ids, candidates):
    """Sum logprob of each candidate's tokens after the prompt. One padded
    batch (n_candidates rows) -> one forward."""
    rows, spans = [], []
    for cand in candidates:
        cand_ids = tok(cand, add_special_tokens=False)["input_ids"]
        rows.append(list(prompt_ids) + cand_ids)
        spans.append((len(prompt_ids), len(cand_ids)))
    width = max(len(r) for r in rows)
    pad = tok.pad_token_id
    input_ids = torch.tensor([r + [pad] * (width - len(r)) for r in rows]).cuda()
    attn = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]).cuda()
    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)
    scores = []
    for i, (start, n) in enumerate(spans):
        s = 0.0
        for j in range(n):
            pos = start + j
            tok_id = input_ids[i, pos].item()
            s += logprobs[i, pos - 1, tok_id].item()  # logits are shifted
        scores.append(s)
    return scores


# Generic mode vocabulary (canonical choice tokens -> natural phrases).
# Overridable via renderer.mode_phrase for extended vocabularies; unknown
# modes fall back to the token itself.
MODE_PHRASE = {"car": "driving", "transit": "public transport",
               "walk": "walking", "bike": "cycling"}

# Canonical mode order for deterministic few-shot exemplar selection.
MODE_ORDER = ["walk", "bike", "car", "transit"]


def _phrase(mode):
    table = MODE_PHRASE
    if _RENDERER is not None and hasattr(_RENDERER, "mode_phrase"):
        table = _RENDERER.mode_phrase
    return table.get(mode, mode)


def _mode_order():
    if _RENDERER is not None and hasattr(_RENDERER, "mode_order"):
        return list(_RENDERER.mode_order)
    return list(MODE_ORDER)


def softmax_probs(scores):
    """Softmax over summed log-likelihoods, temperature 1. Shared by
    --expected-shares and --belief-probe."""
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def _cap(phrase):
    return phrase[0].upper() + phrase[1:]


def belief_lines(condition, m0_phrase, a_phrase):
    """Experience-block lines for the belief-sensitivity probe: pro supports
    M0 and knocks the runner-up A, anti is the mirror image, neutral carries
    no mode-relevant signal.

    RENDER-PARITY: these are synthetic probe lines appended to prompts; a
    renderer providing belief_lines overrides this default so probe wording
    stays under the single render path's control."""
    if _RENDERER is not None and hasattr(_RENDERER, "belief_lines"):
        return _RENDERER.belief_lines(condition, m0_phrase, a_phrase)
    if condition == "pro":
        return [f"{_cap(m0_phrase)} worked really well for me yesterday",
                f"{_cap(a_phrase)} was slow and frustrating last time"]
    if condition == "anti":
        return [f"{_cap(m0_phrase)} was miserable and slow for me yesterday",
                f"{_cap(a_phrase)} worked really well last time"]
    return ["The weather was pleasant yesterday",
            "I listened to a good podcast on the way"]


def with_experience(messages, lines):
    """Append the experience block to the user turn, byte-identical to the
    upstream render path's experience-block rendering.
    # RENDER-PARITY: prompt text must originate from grounding.render --
    # this wrapping (' Your experience: ' + '; '.join + '.') must match the
    # single render path exactly; render-parity tests own that check."""
    user = dict(messages[1])
    user["content"] = user["content"] + " Your experience: " + "; ".join(lines) + "."
    return [messages[0], user]


def run_belief_probe(model, tok, rows, limit):
    """Does an appended memory/experience block move the argmax choice?
    4x score_candidates cost per probed example (none/pro/anti/neutral);
    skips examples with fewer than 2 available modes."""
    conditions = ("pro", "anti", "neutral")
    flips = {c: 0 for c in conditions}
    deltas = {c: [] for c in conditions}
    argmax_counts = {c: Counter() for c in conditions}
    expected_sums = {c: defaultdict(float) for c in conditions}
    n_used = 0
    for r in rows[:limit]:
        cands = r["meta"]["available_modes"]
        if len(cands) < 2:
            continue
        prompt_text = tok.apply_chat_template(
            r["messages"], add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        scores_none = score_candidates(model, tok, prompt_ids, cands)
        idx_m0 = max(range(len(cands)), key=lambda k: scores_none[k])
        idx_a = max((k for k in range(len(cands)) if k != idx_m0),
                    key=lambda k: scores_none[k])
        m0, a = cands[idx_m0], cands[idx_a]
        margin_none = scores_none[idx_m0] - scores_none[idx_a]
        n_used += 1

        for cond in conditions:
            lines = belief_lines(cond, _phrase(m0), _phrase(a))
            msgs = with_experience(r["messages"], lines)
            p_text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
            p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
            scores_c = score_candidates(model, tok, p_ids, cands)
            margin_c = scores_c[idx_m0] - max(
                scores_c[k] for k in range(len(cands)) if k != idx_m0)
            deltas[cond].append(margin_c - margin_none)
            idx_pred = max(range(len(cands)), key=lambda k: scores_c[k])
            pred = cands[idx_pred]
            if pred != m0:
                flips[cond] += 1
            argmax_counts[cond][pred] += 1
            probs_c = softmax_probs(scores_c)
            for k, mode in enumerate(cands):
                expected_sums[cond][mode] += probs_c[k]

    def mean_std(xs):
        if not xs:
            return 0.0, 0.0
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / len(xs)
        return mu, var ** 0.5

    out = {"n_examples": n_used}
    for cond in conditions:
        mu, sd = mean_std(deltas[cond])
        out[cond] = {
            "flip_rate": round(flips[cond] / n_used, 4) if n_used else 0.0,
            "margin_delta_mean": round(mu, 4),
            "margin_delta_std": round(sd, 4),
            "argmax_shares": ({m: round(c / n_used, 4)
                               for m, c in sorted(argmax_counts[cond].items())}
                              if n_used else {}),
            "expected_shares": ({m: round(v / n_used, 4)
                                 for m, v in sorted(expected_sums[cond].items())}
                                if n_used else {}),
        }
    swing = out["pro"]["margin_delta_mean"] - out["anti"]["margin_delta_mean"]
    print(f"belief sensitivity: pro-vs-anti margin swing = {swing:.4f} nats, "
          f"flip rate pro={out['pro']['flip_rate'] * 100:.1f}% "
          f"anti={out['anti']['flip_rate'] * 100:.1f}% "
          f"neutral={out['neutral']['flip_rate'] * 100:.1f}%")
    return out


# Trailing " Your experience: <lines>." block, anchored to end-of-string.
# The block's lines never contain '.' (the render path guarantees it), so
# the only period is the terminal one -> the greedy [^\n]* stops there.
# RENDER-PARITY: this regex mirrors the single render path's experience-
# block wrapping; render-parity tests own the byte-level check.
_EXPERIENCE_RE = re.compile(r" Your experience: [^\n]*\.$")


def strip_experience(user_content):
    """Remove a trailing habit/experience block if the prompt already carries
    one (rows rendered with habit blocks do). The probe must REPLACE, never
    stack, so it works from a blockless baseline. No-op on blockless
    prompts."""
    return _EXPERIENCE_RE.sub("", user_content)


def _selfcheck_strip():
    """Cheap invariant check of strip_experience, run once before the probe
    loop (the file has no CLI self-test)."""
    base = ("Persona: age=40. Available modes: walk, transit, car, bike.")
    block = base + " Your experience: You usually drive to work; " \
        "Last week you made 3 walking and 1 cycling trips."
    assert strip_experience(block) == base, "strip did not recover the base"
    assert strip_experience(base) == base, "strip mutated a blockless prompt"
    assert " Your experience: " not in strip_experience(block), \
        "strip left an experience block behind"


def run_habit_probe(model, tok, rows, limit):
    """Habit channel: does an appended travel-habit block move the argmax
    choice? Mirrors run_belief_probe, with three differences:
      * the baseline is the prompt with any existing habit block STRIPPED
        (prompts rendered with habit blocks carry a real one; we replace,
        not stack);
      * habit lines come from the injected renderer's probe_lines (the
        training-format source of truth), keyed on whether the persona is
        employed;
      * the anti condition also reports flip rate broken down by TARGET mode
        (M0) -- under the default renderer contract a car target is
        structurally weaker (car carries no usage-count slot, only the
        usual-mode line), so it should flip less."""
    _selfcheck_strip()
    conditions = ("pro", "anti", "neutral", "none")
    flips = {c: 0 for c in conditions}
    deltas = {c: [] for c in conditions}
    argmax_counts = {c: Counter() for c in conditions}
    expected_sums = {c: defaultdict(float) for c in conditions}
    anti_flips_by_target = defaultdict(int)
    anti_total_by_target = defaultdict(int)
    # Per-target-mode margin deltas, bucketed by M0, for the stratified swing.
    deltas_by_target = {c: defaultdict(list) for c in conditions}
    n_used = 0
    for r in rows[:limit]:
        cands = r["meta"]["available_modes"]
        if len(cands) < 2:
            continue
        # Blockless baseline: strip any real habit block the prompt carries.
        clean_user = strip_experience(r["messages"][1]["content"])
        clean_msgs = [r["messages"][0],
                      {"role": "user", "content": clean_user}]
        # RENDER-PARITY: keys off the render path's persona field format.
        employed = "employed=true" in clean_user
        prompt_text = tok.apply_chat_template(
            clean_msgs, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        scores_none = score_candidates(model, tok, prompt_ids, cands)
        idx_m0 = max(range(len(cands)), key=lambda k: scores_none[k])
        idx_a = max((k for k in range(len(cands)) if k != idx_m0),
                    key=lambda k: scores_none[k])
        m0, a = cands[idx_m0], cands[idx_a]
        margin_none = scores_none[idx_m0] - scores_none[idx_a]
        n_used += 1

        for cond in conditions:
            lines = probe_lines(cond, m0, a, employed)
            # none -> [] -> the blockless baseline itself (no block appended).
            msgs = with_experience(clean_msgs, lines) if lines else clean_msgs
            p_text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
            p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
            scores_c = score_candidates(model, tok, p_ids, cands)
            margin_c = scores_c[idx_m0] - max(
                scores_c[k] for k in range(len(cands)) if k != idx_m0)
            d = margin_c - margin_none
            deltas[cond].append(d)
            deltas_by_target[cond][m0].append(d)
            idx_pred = max(range(len(cands)), key=lambda k: scores_c[k])
            pred = cands[idx_pred]
            if pred != m0:
                flips[cond] += 1
            argmax_counts[cond][pred] += 1
            probs_c = softmax_probs(scores_c)
            for k, mode in enumerate(cands):
                expected_sums[cond][mode] += probs_c[k]
            if cond == "anti":
                # target = M0 (the mode the anti block argues AGAINST).
                anti_total_by_target[m0] += 1
                if pred != m0:
                    anti_flips_by_target[m0] += 1

    def mean_std(xs):
        if not xs:
            return 0.0, 0.0
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / len(xs)
        return mu, var ** 0.5

    out = {"n_examples": n_used}
    for cond in conditions:
        mu, sd = mean_std(deltas[cond])
        out[cond] = {
            "flip_rate": round(flips[cond] / n_used, 4) if n_used else 0.0,
            "margin_delta_mean": round(mu, 4),
            "margin_delta_std": round(sd, 4),
            "argmax_shares": ({m: round(c / n_used, 4)
                               for m, c in sorted(argmax_counts[cond].items())}
                              if n_used else {}),
            "expected_shares": ({m: round(v / n_used, 4)
                                 for m, v in sorted(expected_sums[cond].items())}
                                if n_used else {}),
        }
    out["anti_flip_rate_by_target_mode"] = {
        m: round(anti_flips_by_target[m] / anti_total_by_target[m], 4)
        for m in sorted(anti_total_by_target)}
    swing = out["pro"]["margin_delta_mean"] - out["anti"]["margin_delta_mean"]
    out["swing"] = round(swing, 4)
    # Stratified swing by target mode (M0): count-slot modes each carry a
    # numeric usage slot in probe_lines; car has only the usual-mode line
    # (default renderer contract). If the counted modes clear the pre-
    # registered swing bar while car lags, the aggregate miss is a
    # car-no-count-slot artifact, not a weak channel.
    out["swing_by_target_mode"] = {
        m: {"swing": round(mean_std(deltas_by_target["pro"][m])[0]
                           - mean_std(deltas_by_target["anti"][m])[0], 4),
            "n": len(deltas_by_target["anti"][m])}
        for m in sorted(set(deltas_by_target["anti"]) | set(deltas_by_target["pro"]))}
    print(f"habit sensitivity: pro-vs-anti margin swing = {swing:.4f} nats, "
          f"flip rate pro={out['pro']['flip_rate'] * 100:.1f}% "
          f"anti={out['anti']['flip_rate'] * 100:.1f}% "
          f"neutral={out['neutral']['flip_rate'] * 100:.1f}%")
    strat = " ".join(f"{m}={v['swing']:.2f}(n={v['n']})"
                     for m, v in out["swing_by_target_mode"].items())
    print(f"  swing by target mode (M0): {strat}")
    return out


def build_fewshot_prefix(pairs_path, k):
    """Deterministic few-shot exemplars: from `pairs_path` train split, in
    file order, the first (k // n_modes) examples per mode THAT CARRY A
    HABIT BLOCK, in canonical mode order (see MODE_ORDER / renderer
    override). Returned as alternating user/assistant chat turns; assistant
    turn is the bare mode token (the same token score_candidates ranks). No
    seeds, no tuning — file order is the whole selection rule."""
    order = _mode_order()
    if k % len(order) != 0:
        raise ValueError("--fewshot-k must be a multiple of %d (the mode-"
                         "vocabulary size), got %d" % (len(order), k))
    per_mode = k // len(order)
    picked = {m: [] for m in order}
    need = k
    with open(pairs_path) as f:
        for line in f:
            r = json.loads(line)
            if r["meta"]["split"] != "train":
                continue
            lab = r["assistant"]
            if lab not in picked or len(picked[lab]) >= per_mode:
                continue
            if " Your experience: " not in r["messages"][1]["content"]:
                continue
            picked[lab].append(r)
            need -= 1
            if need == 0:
                break
    if need:
        raise RuntimeError("fewshot: only found %d/%d exemplars" % (k - need, k))
    msgs = []
    for i in range(per_mode):
        for m in order:
            ex = picked[m][i]
            msgs.append({"role": "user", "content": ex["messages"][1]["content"]})
            msgs.append({"role": "assistant", "content": ex["assistant"]})
    return msgs


def with_fewshot(messages, fewshot_msgs):
    """[system, user] -> [system, ex1_u, ex1_a, ..., user]. No-op when the
    prefix is empty, so every zero-shot path is byte-identical to before."""
    if not fewshot_msgs:
        return messages
    return [messages[0]] + fewshot_msgs + messages[1:]


# Trip-distance clause, exactly as the upstream render path renders it
# ("about %.1f km") — every prompt carries exactly one.
# RENDER-PARITY: mirrors the single render path's distance clause format.
_DIST_RE = re.compile(r"about \d+(?:\.\d+)? km")


def set_distance(user_content, km):
    """Rewrite the trip line's distance clause to `km`. Returns None when the
    prompt does not carry exactly one distance clause (caller skips + counts
    the row) rather than silently probing an unmodified prompt."""
    new, n = _DIST_RE.subn("about %.1f km" % km, user_content)
    return new if n == 1 else None


def append_prose(user_content, prose_lines):
    """Merge prose lines INTO the existing experience block (habit lines
    kept, prose appended after them — the serve-realistic stacking the
    rogue-channel invariant worries about), or open a fresh block when the
    row was a dropout/blockless one."""
    m = _EXPERIENCE_RE.search(user_content)
    if m:
        head = user_content[:m.start()]
        inner = m.group(0)[len(" Your experience: "):-1]
        lines = inner.split("; ") + list(prose_lines)
    else:
        head = user_content
        lines = list(prose_lines)
    return head + " Your experience: " + "; ".join(lines) + "."


def insert_fare(user_content, amount):
    """Insert the price-channel sentence (renderer.fare_line) after the
    structured world-state region and before any trailing experience block
    -- world-state belongs in the structured region, before the memory-owned
    trailing slot, exactly as the upstream render path orders them. Appends
    at the end when the prompt carries no experience block. Raises if the
    prompt already carries a fare line -- double insertion is a caller bug,
    not a no-op."""
    if fare_marker() in user_content:
        raise ValueError("user_content already carries a fare line")
    fare = fare_line(amount)
    m = _EXPERIENCE_RE.search(user_content)
    if m:
        return user_content[:m.start()] + fare + user_content[m.start():]
    return user_content + fare


def _selfcheck_fare():
    """Cheap invariant check of insert_fare, run once before --fare /
    --probe-fare do any real work (mirrors _selfcheck_strip's pattern).
    Renderer-generic: works for any fare_line/fare_marker pair as long as
    the marker appears in the rendered line."""
    base = ("Persona: age=40. Available modes: walk, transit, car, bike.")
    block = base + " Your experience: You usually drive to work; " \
        "Last week you made 3 walking and 1 cycling trips."
    fare = fare_line(86)
    assert fare_marker() in fare, \
        "renderer.fare_marker must occur in renderer.fare_line output"
    assert insert_fare(base, 86) == base + fare, \
        "fare insertion at end-of-string (no experience block) failed"
    got = insert_fare(block, 86)
    want = base + fare + block[len(base):]
    assert got == want, "fare insertion before experience block failed"
    assert got.index(fare) < got.index(" Your experience: "), \
        "fare line landed after the experience block"
    try:
        insert_fare(got, 9)
    except ValueError:
        pass
    else:
        raise AssertionError("double fare insertion must raise ValueError")


def run_probe_dump(model, tok, rows, limit, fewshot_msgs, paired,
                   include_habit, include_channel, include_fare,
                   fare_ref, fare_shock, dump_path):
    """Unified probe pass: score every requested probe condition per row and
    dump the raw per-candidate score arrays. NO aggregates here — the
    offline analyzer owns all arithmetic, so the GPU pass never needs
    re-running when a metric changes.

    Families:
      habit   : h_none / h_pro / h_anti / h_neutral on the STRIPPED baseline
                (same construction as run_habit_probe; reference m0 from the
                h_none scores).
      channel : orig (block retained) / p_pro / p_anti (belief_lines merged
                after the habit lines) / d_near / d_far (0.6 / 7.5 km);
                reference m0 from the orig scores.
      fare    : f_ref / f_shock -- the ORIGINAL user content (habit block
                retained) with insert_fare() applied at fare_ref /
                fare_shock; reference m0 from the orig scores, same as
                channel. `orig` is scored whenever include_channel or
                include_fare is set, so the fare family gets its reference
                even when the channel family is off.
    Reference model = adapter when `paired` (matching the existing habit
    dump's convention), else the (possibly few-shot) base itself."""
    _selfcheck_strip()
    n_used = n_dist_skipped = 0
    with open(dump_path, "w") as out_f:
        for r in rows[:limit]:
            cands = r["meta"]["available_modes"]
            if len(cands) < 2:
                continue

            def score_msgs(msgs):
                p_text = tok.apply_chat_template(
                    with_fewshot(msgs, fewshot_msgs),
                    add_generation_prompt=True, tokenize=False,
                    enable_thinking=False)
                p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
                if paired:
                    b, a = _base_adapter_scores(model, tok, p_ids, cands)
                    return [b, a]
                return [score_candidates(model, tok, p_ids, cands)]

            def ref_pick(scored):
                ref = scored[-1]
                i0 = max(range(len(cands)), key=lambda k: ref[k])
                ia = max((k for k in range(len(cands)) if k != i0),
                         key=lambda k: ref[k])
                return i0, ia

            user = r["messages"][1]["content"]
            rec = {"cands": cands, "label": r["assistant"],
                   "weight": r["meta"].get("weight", 1.0), "cond_scores": {}}

            if include_habit:
                clean_user = strip_experience(user)
                clean_msgs = [r["messages"][0],
                              {"role": "user", "content": clean_user}]
                # RENDER-PARITY: keys off the render path's persona format.
                employed = "employed=true" in clean_user
                rec["cond_scores"]["h_none"] = score_msgs(clean_msgs)
                i0, ia = ref_pick(rec["cond_scores"]["h_none"])
                rec["habit_ref"] = {"idx_m0": i0, "idx_a": ia, "m0": cands[i0]}
                for cond in ("pro", "anti", "neutral"):
                    lines = probe_lines(cond, cands[i0], cands[ia], employed)
                    msgs = with_experience(clean_msgs, lines)
                    rec["cond_scores"]["h_" + cond] = score_msgs(msgs)

            if include_channel or include_fare:
                rec["cond_scores"]["orig"] = score_msgs(r["messages"])
                i0, ia = ref_pick(rec["cond_scores"]["orig"])

            if include_channel:
                rec["channel_ref"] = {"idx_m0": i0, "idx_a": ia,
                                      "m0": cands[i0]}
                for cond in ("pro", "anti"):
                    lines = belief_lines(cond, _phrase(cands[i0]),
                                         _phrase(cands[ia]))
                    msgs = [r["messages"][0],
                            {"role": "user",
                             "content": append_prose(user, lines)}]
                    rec["cond_scores"]["p_" + cond] = score_msgs(msgs)
                dist_ok = True
                for name, km in (("d_near", 0.6), ("d_far", 7.5)):
                    du = set_distance(user, km)
                    if du is None:
                        dist_ok = False
                        break
                    msgs = [r["messages"][0], {"role": "user", "content": du}]
                    rec["cond_scores"][name] = score_msgs(msgs)
                if not dist_ok:
                    rec["cond_scores"].pop("d_near", None)
                    n_dist_skipped += 1

            if include_fare:
                rec["fare_ref"] = {"idx_m0": i0, "idx_a": ia, "m0": cands[i0]}
                for name, amount in (("f_ref", fare_ref),
                                     ("f_shock", fare_shock)):
                    fare_user = insert_fare(user, amount)
                    msgs = [r["messages"][0],
                            {"role": "user", "content": fare_user}]
                    rec["cond_scores"][name] = score_msgs(msgs)

            out_f.write(json.dumps(rec) + "\n")
            n_used += 1
            if n_used % 50 == 0:
                print("[probe-dump] %d rows done" % n_used, flush=True)
    print("[probe-dump] wrote %d records to %s (dist-clause missing on %d)"
          % (n_used, dump_path, n_dist_skipped))


def blend_scores(base, adapter, lam):
    """blended[k] = base[k] + lam*(adapter[k]-base[k]) = (1-lam)*base[k] +
    lam*adapter[k]. lam=0 -> pure base, lam=1 -> pure adapter. Pure
    arithmetic on cached per-example score lists -- no model call."""
    return [b + lam * (a - b) for b, a in zip(base, adapter)]


def _selfcheck_blend():
    """Cheap invariant check of blend_scores, run once before the sweep
    functions (mirrors _selfcheck_strip's inline-assert pattern)."""
    base = [1.0, 0.0, 2.0]
    adapter = [0.0, 2.0, 1.0]
    assert blend_scores(base, adapter, 1.0) == adapter, \
        "lambda=1 must equal pure adapter"
    assert blend_scores(base, adapter, 0.0) == base, \
        "lambda=0 must equal pure base"
    got = blend_scores(base, adapter, 0.5)
    want = [0.5, 1.0, 1.5]
    assert all(abs(g - w) < 1e-9 for g, w in zip(got, want)), \
        f"lambda=0.5 blend mismatch: {got} != {want}"


def _parse_lambdas(spec):
    return [float(x) for x in spec.split(",") if x.strip() != ""]


@torch.no_grad()
def _base_adapter_scores(peft_model, tok, prompt_ids, cands):
    """One example's base[] and adapter[] score lists for a fixed prompt +
    candidate set, computed ONCE. adapter[] uses the active LoRA adapter
    (score_candidates' default forward); base[] uses the same forward with
    the adapter disabled via PeftModel.disable_adapter(), which is a
    context manager that temporarily turns adapters off for its block and
    restores them on exit."""
    adapter_scores = score_candidates(peft_model, tok, prompt_ids, cands)
    with peft_model.disable_adapter():
        base_scores = score_candidates(peft_model, tok, prompt_ids, cands)
    return base_scores, adapter_scores


def run_accuracy_blend_sweep(peft_model, tok, rows, lambdas):
    """Inference-only logit-blend lambda-sweep over held-out accuracy.
    Mirrors main()'s accuracy loop, but per example computes base[] and
    adapter[] ONCE (one base forward + one adapter forward), then evaluates
    every lambda in `lambdas` from those cached arrays -- one GPU pass per
    example covers the whole grid instead of one pass per (example, lambda)."""
    n = {lam: 0 for lam in lambdas}
    correct = {lam: 0 for lam in lambdas}
    wsum = {lam: 0.0 for lam in lambdas}
    wcorrect = {lam: 0.0 for lam in lambdas}
    for r in rows:
        prompt_text = tok.apply_chat_template(
            r["messages"], add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        cands = r["meta"]["available_modes"]
        base_scores, adapter_scores = _base_adapter_scores(
            peft_model, tok, prompt_ids, cands)
        label = r["assistant"]
        w = r["meta"].get("weight", 1.0)
        for lam in lambdas:
            blended = blend_scores(base_scores, adapter_scores, lam)
            pred = cands[max(range(len(cands)), key=lambda k: blended[k])]
            hit = pred == label
            n[lam] += 1
            wsum[lam] += w
            correct[lam] += hit
            wcorrect[lam] += w * hit

    out = {}
    for lam in lambdas:
        out[str(lam)] = {
            "weighted_accuracy": round(wcorrect[lam] / wsum[lam], 4) if wsum[lam] else 0.0,
            "accuracy": round(correct[lam] / n[lam], 4) if n[lam] else 0.0,
            "n": n[lam],
        }
    return out


def run_habit_probe_blend_sweep(peft_model, tok, rows, limit, lambdas,
                                dump_scores=None):
    """Inference-only logit-blend lambda-sweep over the habit probe's
    pro-vs-anti margin swing. Reuses run_habit_probe's exact per-example
    logic (blockless baseline via strip_experience, renderer probe_lines
    conditions, margin = top-choice score minus runner-up score), but caches
    base[]/adapter[] ONCE per example per condition (none/pro/anti/neutral)
    and blends per lambda. idx_m0/idx_a (the target mode the pro/anti blocks
    argue about) are FIXED from the adapter-reference argmax and reused at
    every lambda, so block content and measured margin always refer to the
    same mode -- mirroring run_habit_probe with only the scores blended."""
    _selfcheck_strip()
    conditions = ("pro", "anti", "neutral", "none")
    # Per example: {cond: (base[], adapter[])}, plus (m0_idx-free) cand list.
    cached = []
    for r in rows[:limit]:
        cands = r["meta"]["available_modes"]
        if len(cands) < 2:
            continue
        clean_user = strip_experience(r["messages"][1]["content"])
        clean_msgs = [r["messages"][0],
                      {"role": "user", "content": clean_user}]
        # RENDER-PARITY: keys off the render path's persona field format.
        employed = "employed=true" in clean_user
        prompt_text = tok.apply_chat_template(
            clean_msgs, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        base_none, adapter_none = _base_adapter_scores(
            peft_model, tok, prompt_ids, cands)

        # m0/a for constructing the pro/anti/neutral experience blocks are
        # picked from the ADAPTER-only scores (score_candidates' natural
        # default), same as the unblended run_habit_probe would with the
        # adapter active. The per-lambda idx_m0/idx_a used for MARGINS below
        # are recomputed from the blended scores_none instead.
        idx_m0_ref = max(range(len(cands)), key=lambda k: adapter_none[k])
        idx_a_ref = max((k for k in range(len(cands)) if k != idx_m0_ref),
                        key=lambda k: adapter_none[k])
        m0, a = cands[idx_m0_ref], cands[idx_a_ref]

        cond_scores = {"none": (base_none, adapter_none)}
        for cond in ("pro", "anti", "neutral"):
            lines = probe_lines(cond, m0, a, employed)
            msgs = with_experience(clean_msgs, lines) if lines else clean_msgs
            p_text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
            p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
            cond_scores[cond] = _base_adapter_scores(
                peft_model, tok, p_ids, cands)

        cached.append({"cands": cands, "cond_scores": cond_scores,
                       "idx_m0": idx_m0_ref, "idx_a": idx_a_ref, "m0": m0})

    # Optional: persist the per-example cached base/adapter score arrays so any
    # downstream blend arithmetic (incl. bootstrap CIs on per-mode swings) can
    # run OFFLINE with no GPU -- the forwards were already computed once above.
    # Deterministic given (frozen adapter, fixed eval set): a re-run reproduces
    # these exact arrays, so this is a reusable artifact, not a one-off.
    if dump_scores:
        with open(dump_scores, "w") as _f:
            for ex in cached:
                _f.write(json.dumps({
                    "cands": ex["cands"], "idx_m0": ex["idx_m0"],
                    "idx_a": ex["idx_a"], "m0": ex["m0"],
                    "cond_scores": {c: [list(v[0]), list(v[1])]
                                    for c, v in ex["cond_scores"].items()},
                }) + "\n")
        print(f"[blend-sweep] dumped {len(cached)} probe score records "
              f"to {dump_scores}")

    def mean_std(xs):
        if not xs:
            return 0.0, 0.0
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / len(xs)
        return mu, var ** 0.5

    out = {}
    for lam in lambdas:
        flips = {c: 0 for c in conditions}
        deltas = {c: [] for c in conditions}
        deltas_by_target = {c: defaultdict(list) for c in conditions}
        n_used = 0
        for ex in cached:
            cands = ex["cands"]
            base_none, adapter_none = ex["cond_scores"]["none"]
            scores_none = blend_scores(base_none, adapter_none, lam)
            # FIXED reference idx_m0/idx_a (the adapter-argmax the pro/anti
            # blocks were built around), reused at EVERY lambda so the block
            # content and the measured margin refer to the SAME target mode.
            # Recomputing per lambda would let the block argue about one mode
            # while the margin measured another once the blend shifts the
            # argmax -- inverting the pro/anti deltas at low lambda. This
            # mirrors run_habit_probe exactly, with only the scores blended.
            idx_m0 = ex["idx_m0"]
            idx_a = ex["idx_a"]
            m0 = ex["m0"]
            margin_none = scores_none[idx_m0] - scores_none[idx_a]
            n_used += 1

            for cond in conditions:
                base_c, adapter_c = ex["cond_scores"][cond]
                scores_c = blend_scores(base_c, adapter_c, lam)
                margin_c = scores_c[idx_m0] - max(
                    scores_c[k] for k in range(len(cands)) if k != idx_m0)
                deltas[cond].append(margin_c - margin_none)
                deltas_by_target[cond][m0].append(margin_c - margin_none)
                idx_pred = max(range(len(cands)), key=lambda k: scores_c[k])
                pred = cands[idx_pred]
                if pred != m0:
                    flips[cond] += 1

        pro_margin, _ = mean_std(deltas["pro"])
        anti_margin, _ = mean_std(deltas["anti"])
        swing = pro_margin - anti_margin
        anti_flip = flips["anti"] / n_used if n_used else 0.0
        out[str(lam)] = {
            "swing": round(swing, 4),
            "anti_flip": round(anti_flip, 4),
            "pro_margin": round(pro_margin, 4),
            "anti_margin": round(anti_margin, 4),
            "swing_by_target_mode": {
                m: {"swing": round(mean_std(deltas_by_target["pro"][m])[0]
                                   - mean_std(deltas_by_target["anti"][m])[0], 4),
                    "n": len(deltas_by_target["anti"][m])}
                for m in sorted(set(deltas_by_target["anti"])
                                | set(deltas_by_target["pro"]))},
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--renderer", default=None,
                    help="module[:attr] import spec for the injectable "
                         "prompt renderer (probe_lines/fare_line/"
                         "fare_marker, optional belief_lines/mode_phrase/"
                         "mode_order). RENDER-PARITY: must be backed by the "
                         "project's single render path (grounding.render); "
                         "this script never constructs persona/world prompt "
                         "text itself. Required by --habit-probe/"
                         "--probe-habit/--fare/--probe-fare.")
    ap.add_argument("--expected-shares", action="store_true",
                    help="add softmax-expected shares (temp=1) alongside "
                         "the argmax predicted shares")
    ap.add_argument("--belief-probe", action="store_true",
                    help="probe whether an appended experience block flips "
                         "the argmax mode choice")
    ap.add_argument("--habit-probe", action="store_true",
                    help="probe whether an appended travel-habit block "
                         "(renderer.probe_lines) flips the argmax mode "
                         "choice; strips any existing habit block first")
    ap.add_argument("--probe-limit", type=int, default=1000,
                    help="max eval examples scored by --belief-probe / "
                         "--habit-probe (default 1000)")
    ap.add_argument("--logit-blend-sweep", default=None,
                    help="comma-separated lambda list, e.g. "
                         "'0.0,0.5,0.7,0.8,0.85,0.9,0.92,0.95,0.98,1.0'. "
                         "Inference-only: for each example, scores base "
                         "(adapter disabled) and adapter ONCE, then blends "
                         "blended[k]=base[k]+lambda*(adapter[k]-base[k]) "
                         "for every lambda in the list -- no re-running the "
                         "model per lambda. Requires --adapter. With "
                         "--habit-probe, sweeps the habit-probe swing "
                         "(--probe-limit rows); otherwise sweeps held-out "
                         "accuracy over --pairs/--split.")
    ap.add_argument("--dump-probe-scores", default=None,
                    help="path to write per-example cached base/adapter score "
                         "arrays (JSONL) during the habit-probe blend sweep, "
                         "for offline bootstrap CIs on per-mode swings. "
                         "Requires --habit-probe + --logit-blend-sweep.")
    ap.add_argument("--fewshot-k", type=int, default=0,
                    help="prepend k deterministic exemplars (multiple of "
                         "the mode-vocabulary size) as chat turns")
    ap.add_argument("--fewshot-pairs", default=None,
                    help="pairs.jsonl to draw few-shot exemplars from "
                         "(required when --fewshot-k > 0)")
    ap.add_argument("--dump-eval-scores", default=None,
                    help="JSONL path for per-example raw candidate scores "
                         "on the accuracy loop (base-only, or paired "
                         "[base, adapter] when --adapter is given), with the "
                         "true label + weight — feeds offline temperature/"
                         "bias fitting, ECE and AUC")
    ap.add_argument("--probe-dump", default=None,
                    help="JSONL path for the unified probe dump; "
                         "select families with --probe-habit/--probe-channel")
    ap.add_argument("--probe-habit", action="store_true",
                    help="include the habit family (h_none/h_pro/h_anti/"
                         "h_neutral) in --probe-dump")
    ap.add_argument("--probe-channel", action="store_true",
                    help="include the channel family (orig/p_pro/p_anti/"
                         "d_near/d_far) in --probe-dump")
    ap.add_argument("--fare", type=int, default=None,
                    help="insert the price-channel sentence (renderer."
                         "fare_line) at this monthly transit-pass price "
                         "(currency units) into every loaded row's user "
                         "content ONCE, right after the split filter -- "
                         "everything downstream (accuracy loop, probes, "
                         "dumps) inherits it. Incompatible with "
                         "--probe-fare (would double-insert).")
    ap.add_argument("--probe-fare", action="store_true",
                    help="include the fare-contrast family (f_ref/f_shock) "
                         "in --probe-dump; requires --probe-dump, "
                         "incompatible with --fare")
    ap.add_argument("--fare-ref", type=int, default=86,
                    help="reference monthly transit-pass price (currency "
                         "units) for the --probe-fare 'f_ref' condition "
                         "(placeholder default; set from the project's "
                         "pre-registered fare schedule)")
    ap.add_argument("--fare-shock", type=int, default=9,
                    help="shock monthly transit-pass price (currency units) "
                         "for the --probe-fare 'f_shock' condition "
                         "(placeholder default; set from the project's "
                         "pre-registered fare schedule)")
    args = ap.parse_args(argv)

    _selfcheck_blend()

    if args.logit_blend_sweep and not args.adapter:
        ap.error("--logit-blend-sweep requires --adapter (the sweep blends "
                  "base and adapter logits)")
    if args.fewshot_k and not args.fewshot_pairs:
        ap.error("--fewshot-k requires --fewshot-pairs")
    if args.probe_dump and not (args.probe_habit or args.probe_channel
                                or args.probe_fare):
        ap.error("--probe-dump needs --probe-habit and/or --probe-channel "
                  "and/or --probe-fare")
    if args.probe_fare and not args.probe_dump:
        ap.error("--probe-fare requires --probe-dump")
    if args.probe_fare and args.fare is not None:
        ap.error("--probe-fare cannot be combined with --fare "
                  "(the fare line would be inserted twice)")

    needs_probe_lines = args.habit_probe or args.probe_habit
    needs_fare = args.fare is not None or args.probe_fare
    if (needs_probe_lines or needs_fare) and not args.renderer:
        ap.error("--habit-probe/--probe-habit/--fare/--probe-fare require "
                  "--renderer (RENDER-PARITY: probe/fare prompt text must "
                  "come from the project's single render path)")
    if args.renderer:
        set_renderer(load_renderer(args.renderer))

    if needs_fare:
        _selfcheck_fare()

    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in open(args.pairs)]
    rows = [r for r in rows if r["meta"]["split"] == args.split]
    if args.fare is not None:
        # ONE transform, right after the split filter, so every downstream
        # consumer (accuracy loop, probes, dumps) inherits the fare line.
        for r in rows:
            r["messages"][1]["content"] = insert_fare(
                r["messages"][1]["content"], args.fare)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[eval] {len(rows)} pairs, split={args.split}, "
          f"model={args.model}, adapter={args.adapter}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model_bf16(args.model)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.cuda().eval()

    fewshot_msgs = []
    if args.fewshot_k:
        fewshot_msgs = build_fewshot_prefix(args.fewshot_pairs, args.fewshot_k)
        print("[fewshot] k=%d exemplars from %s (labels: %s)"
              % (args.fewshot_k, args.fewshot_pairs,
                 [m["content"] for m in fewshot_msgs[1::2]]))

    if args.probe_dump:
        run_probe_dump(model, tok, rows, args.probe_limit, fewshot_msgs,
                       paired=bool(args.adapter),
                       include_habit=args.probe_habit,
                       include_channel=args.probe_channel,
                       include_fare=args.probe_fare,
                       fare_ref=args.fare_ref,
                       fare_shock=args.fare_shock,
                       dump_path=args.probe_dump)
        with open(args.out, "w") as f:
            json.dump({"probe_dump": args.probe_dump,
                       "fewshot_k": args.fewshot_k,
                       "families": {"habit": args.probe_habit,
                                    "channel": args.probe_channel,
                                    "fare": args.probe_fare}}, f)
        print("EVAL_CHOICE_OK")
        return 0

    if args.logit_blend_sweep:
        lambdas = _parse_lambdas(args.logit_blend_sweep)
        if args.habit_probe:
            print(f"[blend-sweep] habit-probe swing sweep, "
                  f"lambdas={lambdas}, probe_limit={args.probe_limit}")
            sweep = run_habit_probe_blend_sweep(
                model, tok, rows, args.probe_limit, lambdas,
                dump_scores=args.dump_probe_scores)
        else:
            print(f"[blend-sweep] accuracy sweep, lambdas={lambdas}, "
                  f"n_rows={len(rows)}")
            sweep = run_accuracy_blend_sweep(model, tok, rows, lambdas)
        with open(args.out, "w") as f:
            json.dump(sweep, f, indent=2)
        print(json.dumps(sweep, indent=2))
        print("EVAL_CHOICE_OK")
        return 0

    n = correct = 0
    wsum = wcorrect = 0.0
    label_counts = Counter()
    pred_counts = Counter()
    per_mode = defaultdict(lambda: [0, 0])  # mode -> [n, hits]
    expected_share_sum = defaultdict(float)  # --expected-shares
    dump_f = open(args.dump_eval_scores, "w") if args.dump_eval_scores else None
    for i, r in enumerate(rows):
        # tokenize=False + explicit encode: transformers 5's tokenize=True
        # returns a dict, not a token list (see lora_sft.chat_prompt_ids).
        prompt_text = tok.apply_chat_template(
            with_fewshot(r["messages"], fewshot_msgs),
            add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        cands = r["meta"]["available_modes"]
        if dump_f is not None and args.adapter:
            base_s, scores = _base_adapter_scores(model, tok, prompt_ids, cands)
            dumped = [base_s, scores]
        else:
            scores = score_candidates(model, tok, prompt_ids, cands)
            dumped = [scores]
        if dump_f is not None:
            dump_f.write(json.dumps({
                "cands": cands, "label": r["assistant"],
                "weight": r["meta"].get("weight", 1.0),
                "scores": dumped}) + "\n")
        pred = cands[max(range(len(cands)), key=lambda k: scores[k])]
        label = r["assistant"]
        w = r["meta"].get("weight", 1.0)
        n += 1
        wsum += w
        hit = pred == label
        correct += hit
        wcorrect += w * hit
        label_counts[label] += 1
        pred_counts[pred] += 1
        per_mode[label][0] += 1
        per_mode[label][1] += hit
        if args.expected_shares:
            # argmax shares understate minority modes; expected shares are
            # the estimator a sampling simulation actually realizes.
            probs = softmax_probs(scores)
            for k, cand in enumerate(cands):
                expected_share_sum[cand] += probs[k]
        if (i + 1) % 50 == 0:
            print(f"[eval] {i + 1}/{len(rows)} acc so far {correct / n:.3f}",
                  flush=True)

    if dump_f is not None:
        dump_f.close()
        print("[eval-dump] wrote %d records to %s" % (n, args.dump_eval_scores))

    majority_mode, majority_n = label_counts.most_common(1)[0]
    metrics = {
        "model": args.model, "adapter": args.adapter, "split": args.split,
        "n": n,
        "accuracy": round(correct / n, 4),
        "weighted_accuracy": round(wcorrect / wsum, 4),
        "majority_baseline": {"mode": majority_mode,
                              "accuracy": round(majority_n / n, 4)},
        "per_mode_recall": {m: {"n": c[0], "recall": round(c[1] / c[0], 4)}
                            for m, c in sorted(per_mode.items())},
        "actual_shares": {m: round(c / n, 4)
                          for m, c in sorted(label_counts.items())},
        "predicted_shares": {m: round(c / n, 4)
                             for m, c in sorted(pred_counts.items())},
    }
    if args.expected_shares:
        metrics["expected_shares"] = {m: round(v / n, 4)
                                      for m, v in sorted(expected_share_sum.items())}
    if args.belief_probe:
        metrics["belief_probe"] = run_belief_probe(model, tok, rows, args.probe_limit)
    if args.habit_probe:
        metrics["habit_probe"] = run_habit_probe(model, tok, rows, args.probe_limit)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("EVAL_CHOICE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
