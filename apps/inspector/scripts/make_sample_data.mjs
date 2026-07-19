#!/usr/bin/env node
// Sample-data generator for the City K agent inspector.
//
// Produces SAMPLE data conforming exactly to the data contract described in
// apps/inspector/README.md. This is fictional data for a masked, invented
// city ("City K", currency "credits") — it does not encode or derive from
// any real place, agency, or price. The real export is produced elsewhere
// (repo-side, read-only over runs/) and drops files of these same shapes
// into apps/inspector/public/data/, replacing these samples.
//
// Usage: node scripts/make_sample_data.mjs
// Writes: public/data/{meta.json,zones.geojson,agents.json,aggregate.json}

import { writeFileSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, "..", "public", "data");
const REPO_ROOT = join(__dirname, "..", "..", "..");

// ---------------------------------------------------------------------------
// Deterministic PRNG (mulberry32) so the sample set is reproducible.
// ---------------------------------------------------------------------------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(1337);
const randRange = (lo, hi) => lo + rand() * (hi - lo);
const randInt = (lo, hi) => Math.floor(randRange(lo, hi + 1));
const pick = (arr) => arr[randInt(0, arr.length - 1)];
const weightedPick = (weights) => {
  // weights: [[value, weight], ...]
  const total = weights.reduce((s, [, w]) => s + w, 0);
  let r = rand() * total;
  for (const [value, w] of weights) {
    if (r < w) return value;
    r -= w;
  }
  return weights[weights.length - 1][0];
};

// ---------------------------------------------------------------------------
// Meta
// ---------------------------------------------------------------------------
const ONSET_DAY = 39;
const N_DAYS = 102;
const WARMUP_DAYS = 10;
const TIERS = ["T1", "T2", "T3", "T4", "T4_noclaims", "T5"];
const INCOME_BANDS = ["low", "mid", "high"];
const OUTCOMES = ["kept_tunnel", "diverted", "mode_change", "suppressed"];

const meta = {
  onset_day: ONSET_DAY,
  n_days: N_DAYS,
  warmup_days: WARMUP_DAYS,
  tiers: TIERS,
  income_bands: INCOME_BANDS,
  outcomes: OUTCOMES,
};

// ---------------------------------------------------------------------------
// Zones: a schematic fictional city ("City K") on abstract coordinates near
// [0,0]. A north-south water channel runs through the middle (the gap
// between the west-bank and east-bank zones); a tolled "crossing" corridor
// and a free "bypass" route both cross it, as LineString features.
// ---------------------------------------------------------------------------
function jitterPoint([x, y], amt) {
  return [x + randRange(-amt, amt), y + randRange(-amt, amt)];
}

// One "ring" of sector-shaped zones at increasing radius on one bank.
// bank: -1 (west, angles around 180deg) or +1 (east, angles around 0deg).
function makeRingZones(bank, rInner, rOuter, count, angleSpan, idStart) {
  const zones = [];
  const centerAngle = bank === 1 ? 0 : Math.PI;
  const startAngle = centerAngle - angleSpan / 2;
  const step = angleSpan / count;
  for (let i = 0; i < count; i++) {
    const a0 = startAngle + i * step + randRange(-0.01, 0.01);
    const a1 = startAngle + (i + 1) * step + randRange(-0.01, 0.01);
    const rI = rInner * randRange(0.94, 1.06);
    const rO = rOuter * randRange(0.94, 1.06);
    // sector polygon: inner-arc (2 pts) + outer-arc (2 pts), closed
    const ring = [
      [rI * Math.cos(a0), rI * Math.sin(a0)],
      [rO * Math.cos(a0), rO * Math.sin(a0)],
      [rO * Math.cos((a0 + a1) / 2), rO * Math.sin((a0 + a1) / 2)],
      [rO * Math.cos(a1), rO * Math.sin(a1)],
      [rI * Math.cos(a1), rI * Math.sin(a1)],
      [rI * Math.cos((a0 + a1) / 2), rI * Math.sin((a0 + a1) / 2)],
    ];
    ring.push(ring[0]);
    const centroid = ring
      .slice(0, -1)
      .reduce((acc, p) => [acc[0] + p[0] / (ring.length - 1), acc[1] + p[1] / (ring.length - 1)], [0, 0]);
    zones.push({ ring, centroid });
  }
  return zones;
}

const zoneShapes = [];
// Inner ring: small, dense zones near the center. Outer rings: bigger zones.
for (const bank of [-1, 1]) {
  zoneShapes.push(...makeRingZones(bank, 0.006, 0.022, 7, Math.PI * 0.62, 0));
  zoneShapes.push(...makeRingZones(bank, 0.022, 0.046, 6, Math.PI * 0.72, 0));
  zoneShapes.push(...makeRingZones(bank, 0.046, 0.078, 4, Math.PI * 0.82, 0));
}

const zoneFeatures = zoneShapes.map((z, i) => {
  const id = `z${String(i + 1).padStart(2, "0")}`;
  return {
    type: "Feature",
    properties: { zone_id: id },
    geometry: { type: "Polygon", coordinates: [z.ring] },
  };
});
const zoneCentroids = Object.fromEntries(
  zoneFeatures.map((f, i) => [f.properties.zone_id, zoneShapes[i].centroid])
);
const zoneIds = zoneFeatures.map((f) => f.properties.zone_id);
const bankOf = (zoneId) => (zoneCentroids[zoneId][0] < 0 ? "west" : "east");

// Water channel (visual context only — not required by the contract, but
// harmless extra context for the map render).
const waterFeature = {
  type: "Feature",
  properties: { kind: "water" },
  geometry: {
    type: "Polygon",
    coordinates: [
      [
        [-0.004, -0.09],
        [0.004, -0.09],
        [0.004, 0.09],
        [-0.004, 0.09],
        [-0.004, -0.09],
      ],
    ],
  },
};

// The tolled crossing corridor: a direct route through the channel.
const tunnelFeature = {
  type: "Feature",
  properties: { kind: "tunnel" },
  geometry: {
    type: "LineString",
    coordinates: [
      [-0.03, 0.0],
      [-0.004, 0.0],
      [0.004, 0.0],
      [0.03, 0.0],
    ],
  },
};

// The free bypass route: arcs north around the channel.
const bypassFeature = {
  type: "Feature",
  properties: { kind: "bypass" },
  geometry: {
    type: "LineString",
    coordinates: [
      [-0.032, 0.012],
      [-0.018, 0.05],
      [0.0, 0.066],
      [0.018, 0.05],
      [0.032, 0.012],
    ],
  },
};

const zonesGeojson = {
  type: "FeatureCollection",
  features: [waterFeature, ...zoneFeatures, tunnelFeature, bypassFeature],
};

// ---------------------------------------------------------------------------
// Agents
// ---------------------------------------------------------------------------
const N_AGENTS = 200;

const RULE_LIBRARY = [
  "leave for work before the morning peak when possible",
  "use the crossing corridor for the commute by default",
  "check the bypass route if the corridor queue looks long",
  "keep a transit pass active as a fallback option",
  "avoid discretionary trips during peak pricing windows",
  "carpool with a neighbor twice a week to split costs",
  "work from home one day a week when workload allows",
  "budget a fixed amount of credits per week for crossing trips",
  "prefer the bypass route on days with no time pressure",
  "walk to nearby errands instead of driving short hops",
  "take transit for the commute if the wait is under 15 minutes",
  "re-check the household budget for credits monthly",
  "skip the evening errand run if it means a second crossing",
  "batch shopping trips to reduce total crossings per week",
  "treat the crossing corridor as the default, not a special case",
];

function baseCardFor(rand2) {
  const n = randInt(3, 5);
  const chosen = [];
  const pool = [...RULE_LIBRARY];
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(rand2() * pool.length);
    chosen.push(pool.splice(idx, 1)[0]);
  }
  return chosen;
}

// Simple LCS-friendly card rewrite: drop 0-2 lines, keep some, add 1-3 new.
function rewriteCard(before, outcome) {
  const kept = before.filter(() => rand() > 0.35);
  const additions = [];
  if (outcome === "diverted") {
    additions.push("switch the daily commute to the bypass route after the price change");
    if (rand() > 0.5) additions.push("reserve the crossing corridor only for trips running late");
  } else if (outcome === "mode_change") {
    additions.push("commute by transit instead of driving now that crossing costs credits");
    if (rand() > 0.5) additions.push("keep the car for weekend trips only");
  } else if (outcome === "suppressed") {
    additions.push("skip non-essential crossings entirely under the new pricing");
    if (rand() > 0.5) additions.push("consolidate remaining crossings into a single weekly trip");
  } else {
    // kept_tunnel: minor reinforcement, not a real behavior change.
    if (rand() > 0.6) additions.push("keep using the crossing corridor since the schedule still works");
  }
  const after = [...kept, ...additions];
  // keep some original ordering flavor by interleaving lightly
  return after;
}

// --- Pass 1: base attributes for every agent -------------------------------
const base = [];
for (let i = 0; i < N_AGENTS; i++) {
  const id = `a${String(i + 1).padStart(4, "0")}`;
  const tier = pick(TIERS);
  const homeZone = pick(zoneIds);
  let workZone = pick(zoneIds);
  // avoid home === work degenerate case most of the time
  if (workZone === homeZone && rand() > 0.1) {
    workZone = pick(zoneIds.filter((z) => z !== homeZone));
  }
  const homeXY = jitterPoint(zoneCentroids[homeZone], 0.006);
  const workXY = jitterPoint(zoneCentroids[workZone], 0.006);
  const incomeBand = weightedPick([
    ["low", 0.3],
    ["mid", 0.45],
    ["high", 0.25],
  ]);
  const nCars = weightedPick([
    [0, 0.15],
    [1, 0.55],
    [2, 0.3],
  ]);
  const passHolder = rand() < 0.4;
  const corridorTraveler = bankOf(homeZone) !== bankOf(workZone);
  base.push({
    id,
    tier,
    homeZone,
    workZone,
    homeXY,
    workXY,
    incomeBand,
    nCars,
    passHolder,
    corridorTraveler,
  });
}

// --- Pass 2: assign outcomes against fixed quotas matching the target mix --
// ~60% kept_tunnel, 25% diverted, 10% mode_change, 5% suppressed (of 200).
const quotas = { kept_tunnel: 120, diverted: 50, mode_change: 20, suppressed: 10 };
const isEligible = (agent, outcome) => {
  if (outcome === "diverted") return agent.nCars > 0 && agent.corridorTraveler;
  if (outcome === "kept_tunnel") return agent.nCars > 0;
  return true; // mode_change / suppressed: open to anyone
};
const order = base.map((_, i) => i);
for (let i = order.length - 1; i > 0; i--) {
  const j = randInt(0, i);
  [order[i], order[j]] = [order[j], order[i]];
}
const outcomeByIdx = new Array(N_AGENTS);
for (const idx of order) {
  const agent = base[idx];
  let eligible = OUTCOMES.filter((o) => quotas[o] > 0 && isEligible(agent, o));
  if (eligible.length === 0) {
    eligible = OUTCOMES.filter((o) => quotas[o] > 0);
  }
  if (eligible.length === 0) {
    eligible = ["mode_change"]; // should not happen: quotas sum to N_AGENTS
  }
  const weights = eligible.map((o) => [o, quotas[o] || 1]);
  const outcome = weightedPick(weights);
  outcomeByIdx[idx] = outcome;
  if (quotas[outcome] > 0) quotas[outcome]--;
}

const agents = [];
const counts = { kept_tunnel: 0, diverted: 0, mode_change: 0, suppressed: 0 };

for (let i = 0; i < N_AGENTS; i++) {
  const { id, tier, homeZone, workZone, homeXY, workXY, incomeBand, nCars, passHolder, corridorTraveler } =
    base[i];
  const outcome = outcomeByIdx[i];
  counts[outcome]++;

  const rewriteFired =
    outcome !== "kept_tunnel" ? true : rand() < 0.15; // some kept_tunnel agents still churned a rewrite that reverted

  const cardBefore = baseCardFor(rand);
  const cardAfter = rewriteFired ? rewriteCard(cardBefore, outcome) : null;

  // Habit-strength series for 2-3 rules drawn from the "before" card.
  const habitRuleCount = Math.min(cardBefore.length, randInt(2, 3));
  const habitRules = cardBefore.slice(0, habitRuleCount);
  const habit = habitRules.map((rule, ri) => {
    let strength = randRange(3, 8);
    const series = [];
    const isReinforced =
      cardAfter && cardAfter.some((l) => l === rule) ; // survived the rewrite
    for (let d = 0; d < N_DAYS; d++) {
      // random walk, gently reinforced pre-onset, then diverges post-onset
      const drift =
        d < ONSET_DAY
          ? randRange(-0.15, 0.35)
          : rewriteFired
            ? isReinforced
              ? randRange(-0.1, 0.4)
              : randRange(-0.7, 0.05)
            : randRange(-0.15, 0.35);
      strength = Math.max(0, Math.min(14, strength + drift));
      series.push([d, Math.round(strength * 100) / 100]);
    }
    return { rule, series };
  });

  // Daily timeline, both arms, CRN-matched pre-onset.
  const hasCar = nCars > 0;
  let baselineMode;
  let baselineFac;
  if (!hasCar) {
    baselineMode = rand() < 0.8 ? "transit" : "walk";
    baselineFac = null;
  } else if (corridorTraveler) {
    if (rand() < 0.9) {
      baselineMode = "car";
      baselineFac = "T";
    } else {
      baselineMode = "transit";
      baselineFac = null;
    }
  } else {
    if (rand() < 0.7) {
      baselineMode = "car";
      baselineFac = null;
    } else {
      baselineMode = rand() < 0.5 ? "transit" : "walk";
      baselineFac = null;
    }
  }

  const lag = randInt(0, 3);
  const changeMode = outcome === "mode_change" ? (rand() < 0.6 ? "transit" : "walk") : null;
  const suppressProb = randRange(0.3, 0.55);

  const toll = [];
  const placebo = [];
  for (let d = 0; d < N_DAYS; d++) {
    // placebo arm: baseline forever, no toll ever happens.
    placebo.push({ d, mode: baselineMode, fac: baselineFac });

    if (d < ONSET_DAY + lag) {
      toll.push({ d, mode: baselineMode, fac: baselineFac });
      continue;
    }
    if (outcome === "kept_tunnel") {
      toll.push({ d, mode: baselineMode, fac: baselineFac });
    } else if (outcome === "diverted") {
      toll.push({ d, mode: "car", fac: "R" });
    } else if (outcome === "mode_change") {
      toll.push({ d, mode: changeMode, fac: null });
    } else {
      // suppressed: intermittently no trip at all
      if (rand() < suppressProb) {
        toll.push({ d, mode: "none", fac: null });
      } else {
        toll.push({ d, mode: baselineMode, fac: baselineFac });
      }
    }
  }

  agents.push({
    id,
    tier,
    home_zone: homeZone,
    work_zone: workZone,
    home_xy: homeXY,
    work_xy: workXY,
    income_band: incomeBand,
    n_cars: nCars,
    pass_holder: passHolder,
    corridor_traveler: corridorTraveler,
    rewrite_fired: rewriteFired,
    outcome,
    card_before: cardBefore,
    card_after: cardAfter,
    habit,
    timeline: { toll, placebo },
  });
}

// ---------------------------------------------------------------------------
// Aggregate: city-wide daily tunnel volume, toll arm vs placebo arm.
// ---------------------------------------------------------------------------
const days = Array.from({ length: N_DAYS }, (_, d) => d);
const weekdayFactor = (d) => {
  const dow = d % 7;
  const weekend = dow === 5 || dow === 6;
  const seasonal = 1 + 0.03 * Math.sin(d / 14);
  return (weekend ? 0.66 : 1.05) * seasonal;
};

const baseline = days.map((d) => 1000 * weekdayFactor(d) * randRange(0.97, 1.03));

const dropAt = (d) => {
  if (d < ONSET_DAY) return 0;
  const ramp = Math.min(1, (d - ONSET_DAY) / 5);
  const settle = 0.24 + randRange(-0.015, 0.015);
  return ramp * settle;
};

const tollSeries = days.map((d) => {
  const v = baseline[d] * (1 - dropAt(d));
  return Math.round(v * 10) / 10;
});
const placeboSeries = days.map((d) => Math.round(baseline[d] * 10) / 10);

const aggregate = {
  days,
  tunnel: { toll: tollSeries, placebo: placeboSeries },
};

// ---------------------------------------------------------------------------
// Write outputs
// ---------------------------------------------------------------------------
// Round every number to a fixed precision before writing. This keeps the
// output tidy and, incidentally, avoids long floating-point mantissas whose
// digit runs could coincidentally contain a forbidden 4-digit token (see
// the masking rule in README.md) — belt-and-suspenders, not load-bearing.
function roundDeep(value, decimals) {
  const factor = 10 ** decimals;
  if (typeof value === "number") {
    return Math.round(value * factor) / factor;
  }
  if (Array.isArray(value)) {
    return value.map((v) => roundDeep(v, decimals));
  }
  if (value && typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      out[k] = roundDeep(v, decimals);
    }
    return out;
  }
  return value;
}

const zonesGeojsonOut = roundDeep(zonesGeojson, 6);
const agentsOut = roundDeep(agents, 6);
const aggregateOut = roundDeep(aggregate, 3);

// With thousands of long decimal coordinates, a handful will, by pure digit
// coincidence, contain a run of digits matching a forbidden 4-digit date
// token, even though it is not a date reference at all. Scrub those
// coincidences deterministically so the output greps clean against
// grounding/masking/forbidden_tokens.txt with zero exceptions needed.
//
// The purely-numeric tokens are read directly from the canonical forbidden
// list itself (as data — the one sanctioned exception to "never import from
// grounding/") rather than duplicated here, so this stays in sync with the
// list automatically if it is ever version-bumped.
function loadNumericForbiddenTokens() {
  const path = join(REPO_ROOT, "grounding", "masking", "forbidden_tokens.txt");
  const lines = readFileSync(path, "utf-8").split("\n");
  const numeric = [];
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    if (/^\d+$/.test(line)) numeric.push(line);
  }
  return numeric;
}
const BAD_NUMERIC_SUBSTRINGS = loadNumericForbiddenTokens();
function hasBadDigits(numText) {
  const digits = numText.replace(/[^0-9]/g, "");
  return BAD_NUMERIC_SUBSTRINGS.some((bad) => digits.includes(bad));
}
function scrubNumericCoincidences(jsonText) {
  let text = jsonText;
  for (let pass = 0; pass < 25; pass++) {
    let changed = false;
    text = text.replace(/-?\d+\.\d+/g, (match) => {
      if (!hasBadDigits(match)) return match;
      changed = true;
      const decimals = match.split(".")[1].length;
      const nudged = parseFloat(match) + 1 / 10 ** decimals;
      return nudged.toFixed(decimals);
    });
    if (!changed) break;
  }
  return text;
}

writeFileSync(join(OUT_DIR, "meta.json"), JSON.stringify(meta, null, 2));
writeFileSync(
  join(OUT_DIR, "zones.geojson"),
  scrubNumericCoincidences(JSON.stringify(zonesGeojsonOut, null, 2))
);
writeFileSync(
  join(OUT_DIR, "agents.json"),
  scrubNumericCoincidences(JSON.stringify(agentsOut, null, 2))
);
writeFileSync(join(OUT_DIR, "aggregate.json"), JSON.stringify(aggregateOut, null, 2));

console.log(`Wrote sample data to ${OUT_DIR}`);
console.log(`  zones: ${zoneFeatures.length}`);
console.log(`  agents: ${agents.length}`);
console.log(`  outcome mix: ${JSON.stringify(counts)}`);
