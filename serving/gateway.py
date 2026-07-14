#!/usr/bin/env python3
"""Serving gateway: OpenAI-compatible discrete-choice proxy in front of vLLM.

Transplanted from a predecessor project's serving gateway; the choice
extraction, blend, and common-random-numbers (CRN) machinery are preserved
verbatim. The gateway is a swappable *chooser* behind one serving interface:
point the decision loop's LLM URL at this process and pick the arm with
--backend.

  blend        two-model chooser: per-mode score = base + lambda*(adapter -
               base), computed from first-token logprobs of two upstream
               vLLM calls (one against --base-model, one against
               --adapter-model). lambda is runtime config (--lam /
               AGORA_BLEND_LAMBDA); the value is a calibration decision, so
               it is never hardcoded — only the mechanism lives here.
  logit        calibrated multinomial logit fast-brain chooser (the
               chooser-invariance / E1 falsification arm). No GPU. Delegates
               to a separate transplant (logit_chooser); imported lazily so
               the other backends run without it.
  passthrough  forward unchanged (optionally overriding the model name):
               adapter-only / base-only arms for A/B runs.

Reflection requests (no guided_choice / structured_outputs) are always
passed through to the upstream untouched (logit backend included, when
--upstream is set; otherwise a fixed neutral belief is returned so GPU-free
runs still work end to end).

Scoring method: per-candidate score = logprob of the candidate's FIRST
generated token (top_logprobs of one forced-decode step). Exact for
single-token mode labels; blend semantics identical to the training/eval
choice harness. Constrained-decoding fields are STRIPPED from upstream calls
(the gateway constrains by scoring candidates itself; also avoids a vLLM
structured-outputs scheduling stall).

Every choice appends one JSONL audit record (--score-log) with per-mode
base/adapter/blended scores — the served chooser is fully auditable.

Deterministic: temperature 0 = argmax with fixed mode-order tie-break;
--temperature > 0 samples seeded by the CRN key (see rng_key / pick), so the
same (agent, day, trip) draws identically across counterfactual twin worlds.

RENDER-PARITY: prompts must be produced by grounding.render — no prompt text
is built in serving/. The gateway only ever *receives* already-rendered
message content; it never constructs a prompt.

Stdlib only. Self-test: `python3 gateway.py --self-test`
(spins up an in-process stub upstream; no GPU, no network beyond localhost).

Usage:
  python3 gateway.py --port 8100 --backend blend \
      --upstream http://localhost:8000/v1 \
      --base-model Qwen3-8B --adapter-model <adapter> [--lam 0.92]
  python3 gateway.py --port 8100 --backend logit --coef <coefs.json>
  # then point the fast/slow brain loop's LLM URL at
  #   http://localhost:8100/v1
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Neutral default model (pre-registration: Qwen3-8B). Overridable by param/env.
DEFAULT_MODEL = os.environ.get("AGORA_GW_BASE_MODEL", "Qwen3-8B")

# Domain-neutral mode set + deterministic tie-break order. The client normally
# supplies its own candidates per request (guided_choice / structured_outputs);
# this order only breaks ties among modes present in it. Override via Config.
DEFAULT_MODES_ORDER = ("walk", "transit", "ride", "car", "bike")


# ---------------------------------------------------------------------------
# Blend math (identical semantics to the training/eval choice harness)
# ---------------------------------------------------------------------------

def blend_scores(base, adapter, lam):
    """blended[k] = base[k] + lam*(adapter[k]-base[k]); dict-keyed."""
    return {k: base[k] + lam * (adapter[k] - base[k]) for k in base}


def pick(scores, temperature, seed_text, bias=None, modes_order=None):
    """Serve softmax(score/T + b), sampled when T>0, argmax otherwise
    (fixed mode-order tie-break). `bias` is the per-mode b from the
    calibration layer (say-do correction); when set with T<=0 the argmax runs
    over score/1 + b so b still acts.

    CRN seed contract: `seed_text` must be the (agent-id, sim-day,
    trip-index) key when the client provides one — common random numbers
    across twin worlds require the SAME draw for the same (agent, day)
    regardless of prompt content. Falling back to prompt text breaks twin
    coupling (counterfactual prompts differ) and is only for legacy runs."""
    bias = bias or {}
    order = modes_order if modes_order is not None else DEFAULT_MODES_ORDER
    t_eff = temperature if temperature > 0.0 else 1.0
    z = {m: scores[m] / t_eff + bias.get(m, 0.0) for m in scores}
    ordered = sorted(z, key=lambda m: order.index(m)
                     if m in order else 99)
    if temperature <= 0.0:
        return max(ordered, key=lambda m: z[m])
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)
    mx = max(z.values())
    weights = [math.exp(z[m] - mx) for m in ordered]
    r = rng.random() * sum(weights)
    acc = 0.0
    for m, w in zip(ordered, weights):
        acc += w
        if r <= acc:
            return m
    return ordered[-1]


def rng_key(request_json, user_content):
    """The RNG stream key: the OpenAI `user` field when the client sets it
    (contract: "<agentId>:<simDay>:<tripIndex>", the decision loop's job),
    else the prompt text. Returns (key, source) so the audit trail records
    which coupling regime a run used — twin worlds are only valid on 'user'."""
    u = request_json.get("user")
    if u:
        return str(u), "user"
    return user_content, "prompt"


# ---------------------------------------------------------------------------
# Upstream calls
# ---------------------------------------------------------------------------

def _post_json(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def first_token_toplogprobs(upstream, model, messages, chat_template_kwargs,
                            timeout, top_n=20):
    """One forced-decode step; returns {normalized_token: logprob} for the
    top-N first-token alternatives."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": top_n,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    resp = _post_json(upstream + "/chat/completions", payload, timeout)
    content = (resp.get("choices") or [{}])[0].get("logprobs") or {}
    steps = content.get("content") or []
    out = {}
    if steps:
        for alt in steps[0].get("top_logprobs") or []:
            tok = alt.get("token", "")
            norm = tok.strip().lower()
            lp = alt.get("logprob")
            if norm and lp is not None and (norm not in out or lp > out[norm]):
                out[norm] = lp
    return out


def candidate_scores(toplogprobs, candidates, floor_gap=5.0):
    """Score each candidate by its first token's logprob. A candidate absent
    from the top-N gets floor = min(seen) - floor_gap (documented
    approximation; with single-token labels and top-20 this is rare)."""
    seen = [lp for lp in toplogprobs.values()]
    floor = (min(seen) if seen else -20.0) - floor_gap
    scores = {}
    for c in candidates:
        first = c.strip().lower()
        scores[c] = toplogprobs.get(first, floor)
    return scores


# ---------------------------------------------------------------------------
# Fast-brain chooser (separate transplant) — lazily / injectably loaded
# ---------------------------------------------------------------------------

def _chooser(cfg):
    """Obtain the calibrated MNL fast-brain chooser used by the logit backend.

    This is a SEPARATE transplant (logit_chooser). It is injected via
    cfg.chooser for tests/wiring, else imported by name (cfg.chooser_module).
    Deferred on purpose so the blend and passthrough backends import and run
    with no such module present.

    Expected interface: load_coef(path)->dict, parse_prompt(text)->dict with
    'available_modes', choose(parsed, candidates, coef, temperature=,
    seed_text=)->(mode, utils_dict)."""
    if cfg.chooser is not None:
        return cfg.chooser
    import importlib
    try:
        return importlib.import_module(cfg.chooser_module)
    except ImportError as e:  # pragma: no cover - integration wiring
        raise RuntimeError(
            "logit backend needs the MNL chooser module %r (a separate "
            "transplant); inject Config(chooser=...) or set "
            "AGORA_GW_CHOOSER_MODULE" % cfg.chooser_module) from e


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

# GPU-free reflection stub: a fixed neutral belief returned when the logit
# backend runs with no upstream. This is a fallback RESPONSE, not a prompt —
# no prompt text is constructed here (see RENDER-PARITY note at module top).
NEUTRAL_BELIEF = "No strong impressions from recent trips"


def parse_lam(spec):
    """--lam accepts a scalar ('0.92') or a per-mode JSON map
    ('{"default": 0.92, "car": 1.0}'). Per-mode lambda is the serving-side
    mitigation surface for any per-mode over-amplification found during
    calibration — the VALUE is a calibration decision, this only provides the
    mechanism."""
    if isinstance(spec, (int, float)):
        return float(spec)
    try:
        return float(spec)
    except (TypeError, ValueError):
        pass
    lam = json.loads(spec)
    if not isinstance(lam, dict) or "default" not in lam:
        raise ValueError("per-mode --lam must be a JSON object with 'default'")
    return {k: float(v) for k, v in lam.items()}


def parse_bias(spec):
    """--bias accepts '' / None (no bias) or a per-mode JSON map
    ('{"transit": 0.4}'); missing modes default to 0.0 in pick()."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        return {k: float(v) for k, v in spec.items()}
    b = json.loads(spec)
    if not isinstance(b, dict):
        raise ValueError("--bias must be a JSON object of mode -> float")
    return {k: float(v) for k, v in b.items()}


class Config:
    def __init__(self, backend, upstream=None, base_model=None,
                 adapter_model=None, lam=0.92, temperature=0.0,
                 coef_path=None, model_override=None, score_log=None,
                 timeout=120, bias=None, chooser=None, chooser_module=None,
                 modes_order=None):
        self.backend = backend
        self.upstream = upstream.rstrip("/") if upstream else None
        self.base_model = base_model
        self.adapter_model = adapter_model
        self.lam = lam
        self.temperature = temperature
        self.bias = bias or {}
        self.model_override = model_override
        self.score_log = score_log
        self.timeout = timeout
        self.modes_order = tuple(modes_order) if modes_order else DEFAULT_MODES_ORDER
        # Fast-brain chooser is a separate transplant; resolve it lazily.
        self.chooser = chooser
        self.chooser_module = chooser_module or os.environ.get(
            "AGORA_GW_CHOOSER_MODULE", "logit_chooser")
        self.coef = _chooser(self).load_coef(coef_path) if coef_path else {}
        self._log_lock = threading.Lock()

    def lam_for(self, mode):
        if isinstance(self.lam, dict):
            return self.lam.get(mode, self.lam["default"])
        return self.lam

    def log_scores(self, record):
        if not self.score_log:
            return
        with self._log_lock:
            with open(self.score_log, "a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")


def extract_candidates(cfg, request_json, user_content):
    """Mode candidates: guided_choice (vLLM<=0.9 dialect) or
    structured_outputs.choice (>=0.24 dialect). Both are domain-agnostic
    constrained-decoding fields — the normal path.

    RENDER-PARITY: prompts must be produced by grounding.render — no prompt
    text is built in serving/. The prompt-parse fallback below reaches into
    the rendered prompt's format, which is the renderer's contract, not the
    gateway's; it delegates to the separate chooser transplant and should be
    avoided by having the client pass candidates explicitly."""
    cands = request_json.get("guided_choice")
    if not cands:
        so = request_json.get("structured_outputs") or {}
        cands = so.get("choice")
    if not cands:
        parsed = _chooser(cfg).parse_prompt(user_content)
        cands = parsed["available_modes"]
    return cands or []


def handle_choice(cfg, request_json):
    """Returns (chosen_mode, audit_dict)."""
    # RENDER-PARITY: prompts must be produced by grounding.render — no prompt
    # text is built in serving/. Here we only READ the already-rendered user
    # message content the client sent.
    user_content = ""
    for msg in request_json.get("messages", []):
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
    candidates = extract_candidates(cfg, request_json, user_content)
    if not candidates:
        raise ValueError("no mode candidates in request or prompt")
    key, key_source = rng_key(request_json, user_content)
    audit = {"backend": cfg.backend, "ts": time.time(),
             "rng_key": key if key_source == "user" else None,
             "rng_key_source": key_source,
             "temperature": cfg.temperature,
             "prompt_sha": hashlib.sha256(user_content.encode()).hexdigest()[:16]}

    if cfg.backend == "logit":
        chooser = _chooser(cfg)
        parsed = chooser.parse_prompt(user_content)
        chosen, utils = chooser.choose(
            parsed, candidates, cfg.coef,
            temperature=cfg.temperature, seed_text=key)
        audit["scores"] = {m: round(u, 4) for m, u in utils.items()}
        return chosen, audit

    if cfg.backend == "blend":
        messages = request_json.get("messages", [])
        ctk = request_json.get("chat_template_kwargs")
        base_top = first_token_toplogprobs(
            cfg.upstream, cfg.base_model, messages, ctk, cfg.timeout)
        adapter_top = first_token_toplogprobs(
            cfg.upstream, cfg.adapter_model, messages, ctk, cfg.timeout)
        base = candidate_scores(base_top, candidates)
        adapter = candidate_scores(adapter_top, candidates)
        blended = {m: base[m] + cfg.lam_for(m) * (adapter[m] - base[m])
                   for m in candidates}
        chosen = pick(blended, cfg.temperature, key, bias=cfg.bias,
                      modes_order=cfg.modes_order)
        audit["lambda"] = cfg.lam
        audit["bias"] = cfg.bias or None
        audit["scores"] = {m: {"base": round(base[m], 4),
                               "adapter": round(adapter[m], 4),
                               "blended": round(blended[m], 4)}
                           for m in candidates}
        return chosen, audit

    raise ValueError("handle_choice called for backend %s" % cfg.backend)


def make_handler(cfg):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet default access log
            pass

        def _reply_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reply_chat(self, content, model, extra=None):
            obj = {
                "id": "gw-chatcmpl-%d" % int(time.time() * 1000),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 1,
                          "total_tokens": 1},
            }
            if extra:
                obj["gateway"] = extra
            self._reply_json(obj)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                request_json = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError:
                self._reply_json({"error": "unparseable JSON body"}, 400)
                return

            constrained = ("structured_outputs" in request_json
                           or "guided_choice" in request_json)
            model = request_json.get("model", "gateway")
            try:
                if constrained and cfg.backend in ("blend", "logit"):
                    chosen, audit = handle_choice(cfg, request_json)
                    cfg.log_scores(audit)
                    self._reply_chat(chosen, model, extra=audit)
                    return
                # passthrough backend, and ALL reflection traffic
                if cfg.upstream:
                    fwd = dict(request_json)
                    if cfg.model_override and constrained:
                        fwd["model"] = cfg.model_override
                    resp = _post_json(cfg.upstream + self._upstream_path(),
                                      fwd, cfg.timeout)
                    self._reply_json(resp)
                    return
                if not constrained:
                    # GPU-free reflection stub (logit backend without
                    # --upstream): a fixed neutral belief keeps the loop alive
                    # without inventing behavior.
                    self._reply_chat(NEUTRAL_BELIEF, model)
                    return
                self._reply_json(
                    {"error": "backend %s needs --upstream" % cfg.backend}, 500)
            except Exception as e:  # noqa: BLE001 — must answer the socket
                self._reply_json({"error": str(e)}, 502)

        def _upstream_path(self):
            # Preserve the endpoint the client hit (…/chat/completions).
            return self.path[self.path.find("/chat/completions"):] \
                if "/chat/completions" in self.path else "/chat/completions"

    return Handler


def serve(cfg, port):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(cfg))
    print("[gateway] backend=%s lam=%s listening on "
          "http://127.0.0.1:%d/v1/chat/completions"
          % (cfg.backend, cfg.lam if cfg.backend == "blend" else "-", port),
          file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Self-test (in-process stub upstream; no GPU, no external transplants)
# ---------------------------------------------------------------------------

class _StubUpstream(BaseHTTPRequestHandler):
    """Returns canned first-token top_logprobs keyed by model name, and a
    canned reflection reply for unconstrained requests. Also records the
    last request body per model for assertion."""
    canned = {}       # model -> {token: logprob}
    seen = []         # list of (path, request_json)

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length).decode("utf-8"))
        _StubUpstream.seen.append((self.path, req))
        model = req.get("model", "?")
        if req.get("logprobs"):
            top = [{"token": t, "logprob": lp}
                   for t, lp in _StubUpstream.canned.get(model, {}).items()]
            obj = {"choices": [{"index": 0,
                                "message": {"role": "assistant", "content": "x"},
                                "logprobs": {"content": [{"token": "x",
                                                          "logprob": -0.1,
                                                          "top_logprobs": top}]},
                                "finish_reason": "stop"}]}
        else:
            obj = {"choices": [{"index": 0,
                                "message": {"role": "assistant",
                                            "content": "stub reflection"},
                                "finish_reason": "stop"}]}
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _StubChooser:
    """Minimal stand-in for the MNL fast-brain chooser transplant, so the
    logit backend is exercised end-to-end without an external module. Mirrors
    the interface _chooser() expects: load_coef / parse_prompt / choose."""
    MODES_ALL = list(DEFAULT_MODES_ORDER)

    @staticmethod
    def load_coef(path):
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def parse_prompt(user_content):
        # RENDER-PARITY: the real chooser parses grounding.render's format;
        # the stub just offers the neutral mode set as available.
        return {"available_modes": list(DEFAULT_MODES_ORDER)}

    @staticmethod
    def choose(parsed, candidates, coef, temperature=0.0, seed_text=None):
        utils = {m: coef.get("asc:" + m, 0.0) for m in candidates}
        chosen = pick(utils, temperature, seed_text or "")
        return chosen, utils


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


def self_test():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print("[self-test]", ("PASS:" if cond else "FAIL:"), msg)
        ok = ok and cond

    # 0. Blend math invariants.
    base = {"car": -1.0, "transit": -2.0}
    adapter = {"car": -3.0, "transit": -0.5}
    check(blend_scores(base, adapter, 1.0) == adapter, "lambda=1 == adapter")
    check(blend_scores(base, adapter, 0.0) == base, "lambda=0 == base")
    mid = blend_scores(base, adapter, 0.5)
    check(abs(mid["car"] - (-2.0)) < 1e-12
          and abs(mid["transit"] - (-1.25)) < 1e-12, "lambda=0.5 midpoint")

    # 0b. Token normalization + missing-candidate floor.
    top = {"walk": -0.2, "transit": -1.5}
    sc = candidate_scores(top, ["walk", "transit", "bike"])
    check(sc["walk"] == -0.2 and sc["bike"] == -1.5 - 5.0,
          "missing candidate floored at min-5")

    # 0c. Serving: bias + (agent, day)-keyed CRN.
    s = {"car": -1.0, "transit": -1.4}
    check(pick(s, 0.0, "x") == "car", "argmax without bias")
    check(pick(s, 0.0, "x", bias={"transit": 0.5}) == "transit",
          "bias flips argmax (softmax(score/T + b) semantics)")
    # Twin-world coupling: same key, different prompts -> same draw.
    draws_a = [pick(s, 1.0, "agent4711:day%d:0" % d) for d in range(40)]
    draws_b = [pick(s, 1.0, "agent4711:day%d:0" % d) for d in range(40)]
    check(draws_a == draws_b, "keyed sampling reproducible across worlds")
    check(len(set(draws_a)) > 1, "T=1 sampling actually mixes modes")
    other = [pick(s, 1.0, "agent0001:day%d:0" % d) for d in range(40)]
    check(other != draws_a, "different agents get different streams")
    k, src = rng_key({"user": "a:1:0"}, "prompt text")
    check(k == "a:1:0" and src == "user", "rng_key prefers the user field")
    k, src = rng_key({}, "prompt text")
    check(k == "prompt text" and src == "prompt", "rng_key prompt fallback")
    check(parse_bias('{"transit": 0.4}') == {"transit": 0.4}
          and parse_bias("") == {}, "parse_bias JSON map + empty")

    # RENDER-PARITY: prompts must be produced by grounding.render — no prompt
    # text is built in serving/. The self-test therefore feeds a STATIC,
    # already-rendered placeholder prompt; production content comes from
    # grounding.render, never from here.
    prompt = "Choose a travel mode for the next trip given the options."
    cands = ["walk", "transit", "ride", "car", "bike"]

    def choice_request(model=DEFAULT_MODEL):
        return {
            "model": model,
            "messages": [{"role": "system", "content": "system placeholder"},
                         {"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 5,
            "chat_template_kwargs": {"enable_thinking": False},
            "guided_choice": cands,
            "structured_outputs": {"choice": cands},
        }

    def post(port, payload, path="/v1/chat/completions"):
        req = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (port, path),
            data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())

    # 1. Stub upstream: base prefers transit, adapter prefers car. At
    #    lambda=0.92 the blend must follow the adapter; at lambda=0.0 the base.
    up_port = _free_port()
    upstream = ThreadingHTTPServer(("127.0.0.1", up_port), _StubUpstream)
    _start(upstream)
    _StubUpstream.canned = {
        "base-m": {"transit": -0.3, "car": -2.0, "walk": -3.0,
                   "bike": -4.0, "ride": -5.0},
        "adapter-m": {"car": -0.2, "transit": -2.5, "walk": -3.5,
                      "bike": -4.5, "ride": -5.5},
    }

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        score_log = os.path.join(td, "scores.jsonl")
        gw_port = _free_port()
        cfg = Config("blend", upstream="http://127.0.0.1:%d/v1" % up_port,
                     base_model="base-m", adapter_model="adapter-m",
                     lam=0.92, score_log=score_log)
        gw = ThreadingHTTPServer(("127.0.0.1", gw_port), make_handler(cfg))
        _start(gw)

        resp = post(gw_port, choice_request())
        content = resp["choices"][0]["message"]["content"]
        check(content == "car", "blend lam=0.92 follows adapter (got %s)" % content)
        check(resp["gateway"]["lambda"] == 0.92, "audit carries lambda")
        check("blended" in resp["gateway"]["scores"]["car"],
              "audit carries per-mode base/adapter/blended scores")
        # Upstream calls must strip constrained-decoding fields + force 1 tok.
        lp_calls = [r for _, r in _StubUpstream.seen if r.get("logprobs")]
        check(len(lp_calls) == 2, "blend makes exactly two upstream calls")
        check(all("guided_choice" not in r and "structured_outputs" not in r
                  and r["max_tokens"] == 1 for r in lp_calls),
              "upstream calls stripped of constraints, max_tokens=1")

        cfg.lam = 0.0
        resp = post(gw_port, choice_request())
        check(resp["choices"][0]["message"]["content"] == "transit",
              "blend lam=0 follows base")

        # Per-mode lambda: default 0 (base, prefers transit) but car pinned to
        # the adapter -> adapter's car (-0.2) beats base's transit (-0.3).
        cfg.lam = {"default": 0.0, "car": 1.0}
        resp = post(gw_port, choice_request())
        check(resp["choices"][0]["message"]["content"] == "car",
              "per-mode lambda overrides one mode's blend weight")
        check(parse_lam("0.92") == 0.92 and
              parse_lam('{"default": 0.92, "ride": 1.0}')["ride"] == 1.0,
              "parse_lam accepts scalar and per-mode JSON map")
        try:
            parse_lam('{"ride": 1.0}')
            check(False, "per-mode lam without 'default' must raise")
        except ValueError:
            check(True, "per-mode lam without 'default' raises")
        cfg.lam = 0.92

        # 2. Reflection passthrough (no constraint fields -> upstream).
        refl = {"model": "base-m",
                "messages": [{"role": "system", "content": "r"},
                             {"role": "user", "content": "recent trips summary"}],
                "temperature": 0, "max_tokens": 120}
        resp = post(gw_port, refl)
        check(resp["choices"][0]["message"]["content"] == "stub reflection",
              "reflection passes through to upstream")

        # 2b. Sampled serving with a user-field CRN key; the audit must record
        # the key + source, and same-key requests repeat.
        cfg.temperature = 1.0
        req_u = choice_request()
        req_u["user"] = "agent4711:12:0"
        ra = post(gw_port, req_u)
        rb = post(gw_port, req_u)
        check(ra["choices"][0]["message"]["content"]
              == rb["choices"][0]["message"]["content"],
              "sampled choice reproducible under the same (agent, day) key")
        check(ra["gateway"]["rng_key_source"] == "user"
              and ra["gateway"]["rng_key"] == "agent4711:12:0",
              "audit records rng key + source")
        cfg.temperature = 0.0

        # 3. Determinism: same prompt, same answer, and audit log written.
        r1 = post(gw_port, choice_request())["choices"][0]["message"]["content"]
        r2 = post(gw_port, choice_request())["choices"][0]["message"]["content"]
        check(r1 == r2 == "car", "blend deterministic at T=0")
        with open(score_log) as f:
            lines = [json.loads(x) for x in f]
        check(len(lines) >= 3 and all("scores" in x for x in lines),
              "score audit log appended per choice")
        gw.server_close()

        # 4. Logit backend end-to-end over HTTP, no upstream (injected stub
        #    chooser stands in for the separate logit_chooser transplant).
        gw2_port = _free_port()
        coef = {"asc:transit": 2.0}  # make it prefer transit deterministically
        coef_path = os.path.join(td, "coef.json")
        with open(coef_path, "w") as f:
            json.dump(coef, f)
        cfg2 = Config("logit", coef_path=coef_path, chooser=_StubChooser)
        gw2 = ThreadingHTTPServer(("127.0.0.1", gw2_port), make_handler(cfg2))
        _start(gw2)
        resp = post(gw2_port, choice_request())
        check(resp["choices"][0]["message"]["content"] == "transit",
              "logit backend serves choices with no upstream/GPU")
        refl_resp = post(gw2_port, refl)
        check(refl_resp["choices"][0]["message"]["content"] == NEUTRAL_BELIEF,
              "logit backend answers reflection with neutral stub")
        gw2.server_close()

        # 5. Passthrough backend with model override.
        gw3_port = _free_port()
        cfg3 = Config("passthrough",
                      upstream="http://127.0.0.1:%d/v1" % up_port,
                      model_override="adapter-m")
        gw3 = ThreadingHTTPServer(("127.0.0.1", gw3_port), make_handler(cfg3))
        _start(gw3)
        _StubUpstream.seen.clear()
        post(gw3_port, choice_request(model="original-m"))
        fwd = _StubUpstream.seen[-1][1]
        check(fwd["model"] == "adapter-m" and "guided_choice" in fwd,
              "passthrough forwards intact with model override")
        gw3.server_close()

    upstream.server_close()
    print("[self-test]", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--backend", choices=["blend", "logit", "passthrough"],
                    default=os.environ.get("AGORA_GW_BACKEND", "blend"))
    ap.add_argument("--upstream", default=os.environ.get("AGORA_GW_UPSTREAM"))
    ap.add_argument("--base-model",
                    default=os.environ.get("AGORA_GW_BASE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--adapter-model",
                    default=os.environ.get("AGORA_GW_ADAPTER_MODEL"))
    ap.add_argument("--lam", type=parse_lam,
                    default=parse_lam(os.environ.get("AGORA_BLEND_LAMBDA", "0.92")),
                    help="scalar (0.92) or per-mode JSON map "
                         "('{\"default\": 0.92, \"ride\": 1.0}')")
    ap.add_argument("--temperature", type=float,
                    default=float(os.environ.get("AGORA_CHOICE_TEMPERATURE", "0")))
    ap.add_argument("--coef", help="coefficient JSON (logit backend)")
    ap.add_argument("--model-override",
                    help="model name override (passthrough backend)")
    ap.add_argument("--score-log", help="JSONL audit log of per-choice scores")
    ap.add_argument("--bias", type=parse_bias,
                    default=parse_bias(os.environ.get("AGORA_CHOICE_BIAS", "")),
                    help='per-mode bias b as JSON, e.g. \'{"transit": 0.4, '
                         '"car": -0.1}\' — the calibrated b_mode from the '
                         'say-do correction; served as softmax(score/T + b)')
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.backend == "blend" and not (args.upstream and args.base_model
                                        and args.adapter_model):
        ap.error("blend backend needs --upstream, --base-model, --adapter-model")
    if args.backend == "passthrough" and not args.upstream:
        ap.error("passthrough backend needs --upstream")

    cfg = Config(args.backend, upstream=args.upstream,
                 base_model=args.base_model, adapter_model=args.adapter_model,
                 lam=args.lam, temperature=args.temperature,
                 coef_path=args.coef, model_override=args.model_override,
                 score_log=args.score_log, timeout=args.timeout,
                 bias=args.bias)
    serve(cfg, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
