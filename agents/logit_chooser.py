#!/usr/bin/env python3
"""Calibrated multinomial-logit (MNL) mode chooser.

Two roles (see 00_PROJECT_BRIEF.md layer 3, 01_PREREGISTRATION.md §3-E1):

  1. Fast-brain fallback — the plain-code executor that runs an agent's
     ordinary daily mode choice from its persona card. The slow-brain LLM
     is only invoked at initialization and on surprises (prediction error
     above threshold); every other day, a chooser like this one decides.

  2. E1 falsification arm — per the pre-registration, "the calibrated MNL
     must be scored identically; the method [the LLM chooser] must beat or
     match it before any architecture claim is made." This module IS that
     control: it is deliberately dumb (a linear-in-features softmax model)
     so that any advantage the LLM shows is attributable to the
     architecture, not to a strawman baseline.

It parses the byte-exact rendered choice prompt (the same persona/trip/
habit text the LLM-based chooser reads — one render function is meant to
serve both training and serving, so there is no train/serve parity gap),
extracts a flat named-feature vector per (prompt, candidate mode), and
scores candidates with a multinomial logit whose coefficients are fit on
paired (prompt, chosen-mode) examples drawn from the same calibration
data as the LLM adapter. The coefficient file is a flat
{feature_name: weight} dict — every weight is individually inspectable,
which is the auditable-chooser property this design asks for.

Provenance: transplanted from the predecessor project's mode chooser
(itself the MNL falsification arm for that project's evals) with all
estimation/choice logic — utility scoring, coefficient fitting, seeded
softmax sampling — preserved unchanged. Stripped and genericized on
transplant: the old project's simulator and scenario names, its
scenario-specific mode vocabulary (the public-transport mode renamed to
"transit"), and hard couplings to that project's specific
prompt-rendering/synthetic-data modules (which do not exist here). The
self-test below was rewritten to build its own synthetic
prompts inline rather than importing sibling modules that were not
transplanted. No LLM calls or prompt-authoring happen anywhere in this
module — it only parses and scores text that was already rendered
upstream; there was no other LLM/prompt coupling in the source to strip.

No GPU, no upstream server, stdlib-only at inference time (numpy is
imported only inside --fit). Deterministic: temperature-0 argmax with a
fixed mode-order tie-break; optional softmax sampling is seeded by a
hash of the prompt text, so a replay is reproducible for a given prompt
and varies across distinct prompts.

Usage:
  --self-test                          parser/scoring/fit invariant checks
  --fit PAIRS.jsonl --out COEF.json    fit coefficients (train split),
                                       report held-out test accuracy
  --eval PAIRS.jsonl --coef COEF.json  evaluate a coefficient file
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys

# The choice modes, in canonical availability order — also the
# deterministic tie-break order used by choose() below.
MODES_ALL = ["walk", "transit", "ride", "car", "bike"]

# Reverse of the habit card's usual-mode phrase rendering ("You usually
# {phrase} to work"). Unknown phrases parse to None (tolerated: the serve
# path may carry free-text reflection beliefs in the same block).
PHRASE_TO_MODE = {
    "drive": "car",
    "take public transport": "transit",
    "cycle": "bike",
    "walk": "walk",
    "get a ride": "ride",
}

# ---------------------------------------------------------------------------
# Prompt parsing (GOLDEN format)
# ---------------------------------------------------------------------------

_PERSONA_RE = re.compile(r"Persona: (.*?)\. Today so far: ")
_TRIP_RE = re.compile(
    r"Next trip: from (\w+) to (\w+)(?:, about ([0-9.]+) km)?, "
    r"departing ([0-9?:]+)(?:, currently via (\w+))?\. ")
_AVAIL_RE = re.compile(r"Available modes: ([a-z, ]+)\.")
_EXPERIENCE_RE = re.compile(r" Your experience: (.*)\.$")
_USUAL_RE = re.compile(r"^You usually (.+) to work$")
_MONTH_RE = re.compile(
    r"^Last month you used (?:public transport (\d+) times)?"
    r"(?: and )?(?:ride services (\d+) times)?$")
_WEEK_RE = re.compile(
    r"^Last week you made (?:(\d+) walking)?(?: and )?"
    r"(?:(\d+) cycling)?(?: trips| and \d+ cycling trips)?")


def _parse_week_line(line):
    """'Last week you made X walking and Y cycling trips' and the
    one-clause forms (the habit card's weekly-activity line)."""
    m = re.match(r"^Last week you made (\d+) walking and (\d+) cycling trips$", line)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^Last week you made (\d+) walking trips$", line)
    if m:
        return int(m.group(1)), None
    m = re.match(r"^Last week you made (\d+) cycling trips$", line)
    if m:
        return None, int(m.group(1))
    return None, None


def _parse_month_line(line):
    """'Last month you used public transport X times and ride services Y
    times' and the one-clause forms (the habit card's monthly-activity
    line)."""
    m = re.match(r"^Last month you used public transport (\d+) times"
                 r" and ride services (\d+) times$", line)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^Last month you used public transport (\d+) times$", line)
    if m:
        return int(m.group(1)), None
    m = re.match(r"^Last month you used ride services (\d+) times$", line)
    if m:
        return None, int(m.group(1))
    return None, None


def parse_prompt(user_content):
    """Parse one GOLDEN-format choice prompt into a flat feature dict.
    Tolerant: missing clauses parse to None; unknown experience lines are
    ignored (serve-time prompts may carry free-text reflection beliefs)."""
    out = {
        "persona": {}, "from_type": None, "to_type": None,
        "distance_km": None, "depart": None, "prior_mode": None,
        "available_modes": None,
        "usual_mode": None, "transit30": None, "ride30": None,
        "walk7": None, "bike7": None,
    }
    m = _PERSONA_RE.search(user_content)
    if m:
        for kv in m.group(1).split(", "):
            if "=" in kv:
                k, v = kv.split("=", 1)
                out["persona"][k] = v
    m = _TRIP_RE.search(user_content)
    if m:
        out["from_type"] = m.group(1)
        out["to_type"] = m.group(2)
        out["distance_km"] = float(m.group(3)) if m.group(3) else None
        out["depart"] = None if m.group(4) == "??:??" else m.group(4)
        out["prior_mode"] = m.group(5)
    m = _AVAIL_RE.search(user_content)
    if m:
        out["available_modes"] = [s.strip() for s in m.group(1).split(",")]
    m = _EXPERIENCE_RE.search(user_content)
    if m:
        for line in m.group(1).split("; "):
            um = _USUAL_RE.match(line)
            if um:
                out["usual_mode"] = PHRASE_TO_MODE.get(um.group(1))
                continue
            transit30, ride30 = _parse_month_line(line)
            if transit30 is not None or ride30 is not None:
                out["transit30"], out["ride30"] = transit30, ride30
                continue
            walk7, bike7 = _parse_week_line(line)
            if walk7 is not None or bike7 is not None:
                out["walk7"], out["bike7"] = walk7, bike7
                continue
            # Unknown line (e.g. free-text belief) -> ignored by design.
    return out


# ---------------------------------------------------------------------------
# Feature map + utilities
# ---------------------------------------------------------------------------
# One flat named-feature vector per (parsed prompt, candidate mode); the
# coefficient file is a flat {feature_name: weight} dict, so fitting is a
# plain softmax regression and every weight is individually inspectable —
# the auditable-chooser property this design asks for.

def _to_float(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def features(parsed, mode):
    p = parsed["persona"]
    d = parsed["distance_km"]
    f = {"asc:%s" % mode: 1.0}
    if d is not None:
        f["lnd:%s" % mode] = math.log(max(d, 0.05))
    else:
        f["dmiss:%s" % mode] = 1.0
    if parsed["prior_mode"] == mode:
        f["prior"] = 1.0
    if parsed["usual_mode"] == mode:
        f["usual"] = 1.0
    if mode == "transit" and parsed["transit30"] is not None:
        f["transit30"] = min(parsed["transit30"], 60) / 30.0
    if mode == "ride" and parsed["ride30"] is not None:
        f["ride30"] = min(parsed["ride30"], 60) / 30.0
    if mode == "walk" and parsed["walk7"] is not None:
        f["walk7"] = min(parsed["walk7"], 21) / 7.0
    if mode == "bike" and parsed["bike7"] is not None:
        f["bike7"] = min(parsed["bike7"], 21) / 7.0
    if mode == "car":
        if p.get("carAvail") == "always":
            f["car_always"] = 1.0
        elif p.get("carAvail") == "sometimes":
            f["car_sometimes"] = 1.0
    if mode == "ride" and p.get("restricted_mobility") == "true":
        f["restricted_ride"] = 1.0
    age = _to_float(p.get("age"), 40.0)
    f["age:%s" % mode] = (age - 40.0) / 40.0
    if p.get("employed") == "true":
        f["emp:%s" % mode] = 1.0
    income = _to_float(p.get("income"), 1800.0)
    f["inc:%s" % mode] = math.log(max(income, 100.0) / 1800.0)
    return f


def utilities(parsed, candidates, coef):
    """Per-candidate utility (log-space score). Unavailable modes are simply
    not in `candidates` — availability gating happens by exclusion, exactly
    like the upstream serving client's availableModes rule."""
    utils = {}
    for m in candidates:
        u = 0.0
        for name, value in features(parsed, m).items():
            u += coef.get(name, 0.0) * value
        utils[m] = u
    return utils


def choose(parsed, candidates, coef, temperature=0.0, seed_text=None):
    """Deterministic argmax at temperature 0 (tie-break: MODES_ALL order).
    temperature > 0 draws from softmax(u/T) seeded by sha256(seed_text) —
    reproducible across runs for the same prompt, varying across distinct
    prompts (deterministic replay)."""
    utils = utilities(parsed, candidates, coef)
    ordered = sorted(candidates, key=lambda m: MODES_ALL.index(m) if m in MODES_ALL else 99)
    if temperature <= 0.0:
        return max(ordered, key=lambda m: utils[m]), utils
    seed = int(hashlib.sha256((seed_text or "").encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)
    mx = max(utils.values())
    weights = [math.exp((utils[m] - mx) / temperature) for m in ordered]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for m, w in zip(ordered, weights):
        acc += w
        if r <= acc:
            return m, utils
    return ordered[-1], utils


def load_coef(path):
    with open(path) as f:
        blob = json.load(f)
    # Coefficient files carry a "_meta" block (fit provenance); weights are
    # every other top-level key.
    return {k: v for k, v in blob.items() if k != "_meta"}


# ---------------------------------------------------------------------------
# Fitting (numpy only here; inference stays stdlib)
# ---------------------------------------------------------------------------

def _read_pairs(path, split):
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if split and rec["meta"].get("split") != split:
                continue
            rows.append(rec)
    return rows


def _design(rows):
    """Build (feature index, per-row candidate feature matrices, labels)."""
    names = {}
    examples = []
    for rec in rows:
        parsed = parse_prompt(rec["messages"][1]["content"])
        cands = rec["meta"]["available_modes"]
        label = rec["assistant"]
        if label not in cands:
            continue
        per_cand = []
        for m in cands:
            f = features(parsed, m)
            for name in f:
                if name not in names:
                    names[name] = len(names)
            per_cand.append(f)
        examples.append((per_cand, cands.index(label), cands))
    return names, examples


def fit(pairs_path, out_path, l2=1e-3, iters=400, lr=0.5, seed=0):
    import numpy as np
    train = _read_pairs(pairs_path, "train")
    test = _read_pairs(pairs_path, "test")
    names, examples = _design(train)
    dim = len(names)
    print("[fit] train=%d test=%d features=%d" % (len(examples), len(test), dim))

    # Dense per-example candidate matrices, padded to max candidate count.
    max_c = max(len(e[2]) for e in examples)
    X = np.zeros((len(examples), max_c, dim))
    mask = np.zeros((len(examples), max_c), dtype=bool)
    y = np.zeros(len(examples), dtype=int)
    for i, (per_cand, label_idx, cands) in enumerate(examples):
        y[i] = label_idx
        for j, f in enumerate(per_cand):
            mask[i, j] = True
            for name, value in f.items():
                X[i, j, names[name]] = value

    w = np.zeros(dim)
    m_adam = np.zeros(dim)
    v_adam = np.zeros(dim)
    rows_idx = np.arange(len(examples))
    last_nll = None
    for t in range(1, iters + 1):
        u = X @ w
        u[~mask] = -1e30
        u -= u.max(axis=1, keepdims=True)
        p = np.exp(u)
        p /= p.sum(axis=1, keepdims=True)
        nll = -np.log(p[rows_idx, y] + 1e-12).mean() + l2 * (w @ w)
        resid = p.copy()
        resid[rows_idx, y] -= 1.0
        grad = np.einsum("ij,ijk->k", resid, X) / len(examples) + 2 * l2 * w
        # Adam
        m_adam = 0.9 * m_adam + 0.1 * grad
        v_adam = 0.999 * v_adam + 0.001 * grad * grad
        mh = m_adam / (1 - 0.9 ** t)
        vh = v_adam / (1 - 0.999 ** t)
        w -= lr * mh / (np.sqrt(vh) + 1e-8)
        if t % 100 == 0 or t == 1:
            print("[fit] iter %4d  nll %.4f" % (t, nll))
        last_nll = nll

    coef = {name: float(w[idx]) for name, idx in names.items()}
    acc, per_mode = _accuracy(test, coef)
    majority = _majority_baseline(test)
    meta = {
        "fit_from": os.path.abspath(pairs_path),
        "n_train": len(examples), "n_test": len(test),
        "final_train_nll": float(last_nll), "l2": l2, "iters": iters,
        "test_accuracy": acc, "test_per_mode_recall": per_mode,
        "test_majority_baseline": majority,
        "note": "Falsification-arm MNL (pre-registration 3-E1); fit on the "
                "same calibration pairs as the LLM adapter.",
    }
    blob = dict(coef)
    blob["_meta"] = meta
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2, sort_keys=True)
    print("[fit] test accuracy %.4f (majority-mode baseline %.4f)" % (acc, majority))
    print("[fit] per-mode recall:", json.dumps(per_mode, sort_keys=True))
    print("[fit] wrote", out_path)
    return acc


def _accuracy(rows, coef):
    n = correct = 0
    per_mode = {}
    for rec in rows:
        parsed = parse_prompt(rec["messages"][1]["content"])
        cands = rec["meta"]["available_modes"]
        label = rec["assistant"]
        if label not in cands:
            continue
        pred, _ = choose(parsed, cands, coef)
        n += 1
        hit = pred == label
        correct += hit
        tot, good = per_mode.get(label, (0, 0))
        per_mode[label] = (tot + 1, good + hit)
    recalls = {m: round(g / t, 4) for m, (t, g) in per_mode.items() if t}
    return (correct / n if n else 0.0), recalls


def _majority_baseline(rows):
    counts = {}
    for rec in rows:
        counts[rec["assistant"]] = counts.get(rec["assistant"], 0) + 1
    return (max(counts.values()) / sum(counts.values())) if counts else 0.0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
# Self-contained: the source module's self-test synthesized prompts via two
# sibling modules (a synthetic-population generator and a habit-card line
# renderer) that were not part of this transplant. Rather than pull those
# in, this version builds minimal GOLDEN-format prompt strings directly —
# it exercises exactly the same parser/scoring/fit invariants, just without
# the external synthetic-data dependency.

def _render_prompt(persona, activity_text, from_type, to_type, distance_km,
                    depart, prior_mode, avail_modes, experience_lines):
    """Build a synthetic GOLDEN-format prompt for self-test purposes only.
    Mirrors the grammar parse_prompt expects; the real renderer lives
    upstream in the serving pipeline, not in this module."""
    persona_str = ", ".join("%s=%s" % kv for kv in persona.items())
    trip = "Next trip: from %s to %s" % (from_type, to_type)
    if distance_km is not None:
        trip += ", about %s km" % distance_km
    trip += ", departing %s" % (depart or "??:??")
    if prior_mode:
        trip += ", currently via %s" % prior_mode
    trip += ". "
    avail = "Available modes: %s." % ", ".join(avail_modes)
    tail = (" Your experience: %s." % "; ".join(experience_lines)) if experience_lines else ""
    return "Persona: %s. Today so far: %s. %s%s%s" % (
        persona_str, activity_text, trip, avail, tail)


def _synth_pairs(n, seed):
    """Self-contained synthetic (prompt, chosen-mode) pairs for the
    --self-test fit smoke check: a simple distance rule (short trips walk,
    longer trips drive) that a correctly-fit MNL should recover well above
    the majority-mode baseline. This is not calibration data — real
    coefficient fitting uses --fit against the actual calibration pairs."""
    rng = random.Random(seed)
    avail = ["walk", "transit", "ride", "car", "bike"]
    pairs = []
    for i in range(n):
        persona = {
            "age": rng.randint(18, 75),
            "income": rng.randint(500, 6000),
            "employed": rng.choice(["true", "false"]),
            "carAvail": rng.choice(["always", "sometimes", "never"]),
        }
        distance = round(rng.uniform(0.1, 3.0), 2)
        depart = "%02d:%02d" % (rng.randint(6, 20), rng.choice([0, 15, 30, 45]))
        prompt = _render_prompt(persona, "ran errands", "home", "work",
                                 distance, depart, None, avail, [])
        label = "walk" if distance < 1.5 else "car"
        pairs.append({
            "messages": [
                {"role": "system", "content": "choose a mode"},
                {"role": "user", "content": prompt},
            ],
            "assistant": label,
            "meta": {
                "available_modes": avail,
                "split": "test" if i % 5 == 0 else "train",
            },
        })
    return pairs


def self_test():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print("[self-test]", ("PASS:" if cond else "FAIL:"), msg)
        ok = ok and cond

    # 1. Parser round-trip on synthetic GOLDEN prompts, with habit blocks.
    rng = random.Random(7)
    n_checked = 0
    round_trip_ok = True
    for i in range(50):
        persona = {
            "age": rng.randint(18, 75),
            "income": rng.randint(500, 6000),
            "employed": rng.choice(["true", "false"]),
            "carAvail": rng.choice(["always", "sometimes", "never"]),
        }
        distance = round(rng.uniform(0.2, 20.0), 1)
        depart = "%02d:%02d" % (rng.randint(6, 20), rng.choice([0, 15, 30, 45]))
        prior = rng.choice([None, "walk", "transit", "car", "bike", "ride"])
        avail = ["walk", "transit", "ride", "car", "bike"]
        experience = [
            "You usually drive to work",
            "Last month you used public transport 4 times and ride services 1 times",
            "Last week you made 3 walking and 2 cycling trips",
        ]
        prompt = _render_prompt(persona, "ran errands", "home", "work",
                                 distance, depart, prior, avail, experience)
        parsed = parse_prompt(prompt)
        round_trip_ok &= parsed["available_modes"] == avail
        round_trip_ok &= parsed["distance_km"] == distance
        round_trip_ok &= parsed["prior_mode"] == prior
        round_trip_ok &= parsed["persona"].get("age") == str(persona["age"])
        round_trip_ok &= parsed["usual_mode"] == "car"
        round_trip_ok &= parsed["transit30"] == 4 and parsed["ride30"] == 1
        round_trip_ok &= parsed["walk7"] == 3 and parsed["bike7"] == 2
        n_checked += 1
    check(round_trip_ok, "GOLDEN prompt round-trip on %d prompts" % n_checked)

    # 1b. One-clause habit lines + unknown (free-text belief) lines tolerated.
    prompt = _render_prompt(
        {"age": 30, "income": 2000, "employed": "true", "carAvail": "sometimes"},
        "worked from a cafe", "home", "work", 3.4, "09:00", None,
        ["walk", "transit", "ride", "car", "bike"],
        ["Transit felt slow on my commute", "Last week you made 5 cycling trips"])
    parsed = parse_prompt(prompt)
    check(parsed["bike7"] == 5 and parsed["walk7"] is None
          and parsed["usual_mode"] is None,
          "one-clause week line parsed; free-text belief line ignored")

    # 2. Availability gating by exclusion + deterministic argmax tie-break.
    coef = {"asc:car": 1.0, "asc:transit": 1.0}
    pred, utils = choose(parsed, ["walk", "transit"], coef)
    check("car" not in utils, "unavailable mode never scored")
    check(pred == "transit", "argmax picks the higher-utility candidate")
    pred_tie, _ = choose(parsed, ["walk", "bike"], {})
    check(pred_tie == "walk", "exact tie broken by fixed mode order")

    # 3. Habit terms move a near-tie (the falsification arm must be
    #    memory-sensitive through the SAME rendered channel as the LLM).
    base_coef = {"asc:transit": 0.05, "prior": 0.5}
    base_persona = {"age": 41, "income": 2500, "employed": "true", "carAvail": "always"}
    prompt_prior_walk = _render_prompt(base_persona, "went to the gym", "home",
                                        "work", 2.1, "08:30", "walk",
                                        ["walk", "transit"], [])
    pred_a, _ = choose(parse_prompt(prompt_prior_walk), ["walk", "transit"], base_coef)
    prompt_no_prior = _render_prompt(base_persona, "went to the gym", "home",
                                      "work", 2.1, "08:30", None,
                                      ["walk", "transit"], [])
    pred_b, _ = choose(parse_prompt(prompt_no_prior), ["walk", "transit"], base_coef)
    check(pred_a == "walk" and pred_b == "transit",
          "prior-mode habit term flips a near-tie")

    # 4. Seeded sampling: deterministic per prompt, varies across prompts.
    picks1 = [choose(parsed, ["walk", "transit", "bike"], {}, temperature=1.0,
                     seed_text="prompt-A")[0] for _ in range(3)]
    pick_other = choose(parsed, ["walk", "transit", "bike"], {}, temperature=1.0,
                        seed_text="prompt-B")[0]
    check(len(set(picks1)) == 1, "temperature sampling deterministic per seed text")
    check(isinstance(pick_other, str), "sampling with different seed text runs")

    # 5. Mini-fit smoke: on synthetic pairs the fitted MNL must beat the
    #    majority baseline (it should pick up the distance rule).
    try:
        import numpy  # noqa: F401
        import tempfile
        pairs = _synth_pairs(400, seed=11)
        with tempfile.TemporaryDirectory() as td:
            pp = os.path.join(td, "pairs.jsonl")
            with open(pp, "w") as f:
                for p in pairs:
                    f.write(json.dumps(p) + "\n")
            cp = os.path.join(td, "coef.json")
            acc = fit(pp, cp, iters=200, lr=0.5)
            test_rows = _read_pairs(pp, "test")
            check(acc > _majority_baseline(test_rows) + 0.10,
                  "fitted MNL beats majority baseline by >10pp on synthetic rule "
                  "(acc %.3f)" % acc)
            loaded = load_coef(cp)
            acc2, _ = _accuracy(test_rows, loaded)
            check(abs(acc2 - acc) < 1e-9, "coefficient file round-trips exactly")
    except ImportError:
        print("[self-test] SKIP: numpy not available, fit smoke skipped")

    print("[self-test]", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--fit", metavar="PAIRS_JSONL")
    mode.add_argument("--eval", metavar="PAIRS_JSONL")
    ap.add_argument("--out", help="coefficient JSON output (with --fit)")
    ap.add_argument("--coef", help="coefficient JSON to evaluate (with --eval)")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.fit:
        if not args.out:
            ap.error("--fit requires --out")
        fit(args.fit, args.out, l2=args.l2, iters=args.iters, lr=args.lr)
        return 0
    if args.eval:
        if not args.coef:
            ap.error("--eval requires --coef")
        rows = _read_pairs(args.eval, "test")
        acc, per_mode = _accuracy(rows, load_coef(args.coef))
        print("test accuracy %.4f (majority baseline %.4f)"
              % (acc, _majority_baseline(rows)))
        print("per-mode recall:", json.dumps(per_mode, sort_keys=True))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
