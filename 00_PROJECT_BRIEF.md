# SAGA — Say-do Aligned Generative Agents
*(working name; "saga" is Swedish for tale — rename freely)*

## What this project is
A domain-agnostic **method** for simulating human populations with LLM agents, validated blind against a real natural experiment: the Stockholm congestion charge (introduced Jan 2006, removed Aug 2006, reinstated permanently Aug 2007). Transport is the testbed; the contribution is the method. Successor to `enact` (Berlin/MATSim), whose sealed negative findings motivate this design:

- Narrative memory never reached decisions (decision-inert beliefs) → here, memory must earn its place via a measurable behavioral signature (hysteresis, E6).
- Native price response ~10× too weak; aggregate response had to be imposed → here, calibration is a separate, pre-registered layer with a hard wall (E4).
- Habit channel failed its 2.0-nat gate twice → here, the cheap brain *is* the habit substrate; the LLM is not in the daily loop.
- Train/serve distance confound (route vs beeline km) silently flipped a verdict → here, **render-parity is a day-one test**: one function renders any persona/world state for both training and serving. No second code path, ever.

## The five layers (frozen)
1. **Grounding** — every agent is seeded from **one real anonymous individual diary record** (Swedish travel survey), never a segment average. Persona = skeleton (demographics, home/work zones, car ownership) + habits (observed trip patterns) + voice (reasoning material from published qualitative studies of the charge).
2. **Population/variance** — the simulated population must preserve real human spread. Diversity is inherited from record-level seeding, not injected. Measured by E2.
3. **Decision (two-brain)** — a **slow brain** (LLM) writes and rewrites a compact *persona card* of rules, called only at initialization and on **surprises** (prediction error above threshold: new toll, jam, budget break). A **fast brain** (plain code) executes daily choices by following the card. Cards carry per-rule **habit-strength counters** (days followed); when the world reverts, strong habits resist — this is the memory mechanism, tested by E6. Surprise log capped at 5 entries; resolved surprises fold into rules or drop.
4. **Say-do calibration** — every agent has *survey mode* (asked hypotheticals, no consequences) and *life mode* (acts under budget/time consequences). The gap between them is measured on small calibration events (fuel price moves, fare changes, seasons), corrected once, frozen, and tested for **transfer** to the big event (E3).
5. **Evaluation** — distributional metrics only (never point estimates), strict calibration/validation wall on the on→off→on timeline, contamination masking, ensemble uncertainty with coverage checks. Governed by `01_PREREGISTRATION.md`, which outranks any later idea.

## Contamination masking (non-negotiable)
Agents must never see: "Stockholm", "congestion tax/charge", "trängselskatt", real district names, real 2006/2007 dates, or exact historical toll prices. The world is an anonymized city ("City K"), zones are codes, dates are shifted, prices perturbed ±10% preserving relative structure. Agents experience raw events ("a new toll appears on your route"), not named history. Every prompt template passes a mask-lint check (forbidden-token list) in CI.

## The world (replaces MATSim)
A deliberately light custom world: zones with zone-to-zone travel times per mode, a volume-delay function so car times worsen with load (consequences are real), transit times derived from GTFS, a toll cordon with a fee schedule, per-agent daily time/money budgets. One simulated day = every agent's fast brain runs; aggregate loads feed back into travel times. Target: 10k agents/day in seconds on CPU.

## Transplants from enact (proven, world-agnostic)
- `chooser_gateway.py` — first-token-logit choice API with blend + CRN determinism
- `training/lora_sft.py`, `eval_choice.py` — LoRA SFT + choice eval harness
- `AgentMemory.java` → port to Python — EMA/shock/trailing-counter substrate becomes the card's habit counters
- JSONL analysis scripts, `bootstrap_permode_ci.py` (bootstrap CIs, permutation placebos)
- `logit_chooser.py` — calibrated MNL as fast-brain fallback and falsification arm
- Adapter schema (`nhts_adapter.py`/`srv_pipeline.py`) as template for Swedish diary data
- The docs/ discipline: pre-registration wall, sealed gates, negative results preserved

## Data
- **SCB** neighborhood statistics — agent skeletons (open)
- **Trafa RVU** national travel survey + Stockholm County travel surveys (1986/2004/2015) — habits; microdata likely **on application**: file early, build against published aggregates + a synthetic stand-in with identical schema meanwhile
- **GTFS / Trafiklab** — transit layer (open)
- **OpenStreetMap** — zones/network (open)
- Published cordon counts 2005–2008 + attitude time series — the truth, quarantined in `evaluation/truth/`, never readable by agent code (enforced by import-boundary test)
- Published qualitative interview studies of the charge — voice material

## Repo skeleton
```
saga/
├── 01_PREREGISTRATION.md   # first commit, frozen
├── world/                  # zones, congestion, toll, GTFS ingest
├── agents/                 # persona cards, fast brain, slow brain client
├── grounding/              # diary adapters, seeding, masking + mask-lint
├── calibration/            # say-do measurement + correction
├── evaluation/             # E1–E6 harness; truth/ quarantined
├── serving/                # gateway (transplant), vLLM configs
├── training/               # LoRA (transplant)
├── jobs/                   # Slurm (Leonardo) — no usernames in files
├── tests/                  # render-parity, mask-lint, import-boundary from day one
└── docs/                   # gate records, IN THE REPO (enact lesson: no off-repo specs)
```

## Build order (each milestone ends with a scored eval, nothing unscored accumulates)
- **M0 Data audit** — acquire/inspect data; write the synthetic stand-in; **freeze numeric bars in the pre-registration** (metrics are already frozen; bars are set once real data quality is known, before any agent runs).
- **M1 World** — City K runs with scripted (non-LLM) agents; congestion feedback sane.
- **M2 Grounding** — record-seeded agents + persona cards; score **E1, E2**.
- **M3 Baseline dynamics** — ordinary-day loop with two brains; E1/E2 re-scored end-to-end.
- **M4 Shock** — toll on; score **E4** blind test 1 (off) then blind test 2 (return).
- **M5 Say-do** — survey mode + correction; score **E3**.
- **M6 Memory** — habit-strength ablation; score **E6**. **E5 runs continuously from M2.**

## Working rules
- Pre-registration outranks everything; changing it requires a dated amendment note, never an edit.
- Negative results are kept and written up in docs/ — they are product, not failure.
- All GPU work uses full allocated nodes (DDP/torchrun or packed parallel runs); billing is per node.
- No personal usernames/emails in any committed file.
- Every claim in README/docs must be reproducible from a file in `runs/` — no unbacked headline numbers (enact lesson).
