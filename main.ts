// gta6-trophy-rarity-forecast-01 — E/N decomposition forecast (Node 22, stdlib only).
//
// Forecast each GTA VI trophy's Exophase earn rate at the case target by forecasting
// earners E and players-tracked N separately (rate = E/N), fitting only on refresh
// events, shrinking per-trophy slopes toward their tier/story group, and spreading
// quantiles on the logit scale with a horizon-aware sigma clamped to [0.5, 1.5].
//
// Why E/N, not logit(rate): a rate moves for two opposite reasons — N grows
// (dilution: pushes rates down) and E grows (player progress: pushes rates up).
// Extrapolating logit(rate) directly mixes the two and misses the rebound after a
// refresh (launch night: N jumped 638 -> 3,868 at the first refresh while earners
// barely moved, so rates fell ~6x; afterwards story trophies rebounded as the new
// cohort played through the early chapters). Fitting ln N and ln E separately
// against x = ln(1 + hours since first read) captures both forces.
//
// Refresh events: the page is a step function — reads only change when Exophase
// refreshes. Unchanged reads are duplicates, not new observations; fitting on all
// reads biases slopes toward zero. So history is compressed to its first read plus
// every read where players_tracked or any rate changed, and all fits run on events.
//
// Flat history (a single event, as in the practice cases): no slope is observable.
// Instead of persistence (the house baseline, which ignores the coming refresh),
// a launch-night influx prior is used for N: players accumulate after release at
// a roughly constant hourly rate, each hour adding about as many players as the
// whole pre-release tracked cohort, so N(t) = N_first * (1 + hours since release).
// E grows as the new cohort progresses: new players earn each trophy at a
// fraction alpha of the veteran per-player rate, alpha growing with hours since
// release (calibrated on the public 04:30 refresh and the live page — see
// PROGRESS_PER_HOUR). The median is then E(target)/N(target).
//
// No randomness, no system clock, no network: every time comes from the case
// file (history[].at, at, target, game.release_date). Output is deterministic
// byte-for-byte across runs.
//
// Usage: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Shrinkage strength: w = n / (n + K_SHRINK) weights a trophy's own slope vs its group. */
const K_SHRINK = 4;
/** Per-trophy log-earner slope is clamped to this range (earners never fall, never explode). */
const SLOPE_MIN = -0.2;
const SLOPE_MAX = 1.5;
/** Level uncertainty on the logit scale (rate rounding + staleness of the last read). */
const SIGMA_LEVEL = 0.25;
/** Assumed lag (hours) between the target and the last refresh before it. */
const REFRESH_LAG_HOURS = 6;
/** Spread clamp on the logit scale; the 100%-trophy spread is separate (SIGMA_FULL). */
const SIGMA_MIN = 0.5;
const SIGMA_MAX = 1.5;
/** Slope-uncertainty fallback when a fit has too few events to estimate an SE. */
const SIGMA_SLOPE_FALLBACK = 0.5;
/** Spread for a trophy every tracked player holds (rate 100% at every event). */
const SIGMA_FULL = 0.05;
/** Spread for a trophy with no usable history at all. */
const SIGMA_UNKNOWN = 1.5;
/** Cap on the launch-night prior's N jump (x N_last). */
const PRIOR_N_CAP = 20;
/** New-player progression rate (per hour since release, after the first 2 h):
 *  new players earn each trophy at a fraction alpha of the veteran per-player
 *  rate, because they progress through the early content in their first hours.
 *  Calibrated on the two public anchors: the 04:30 refresh (103 new non-intro
 *  earners over 3,230 new players ~ alpha 0.075 at 4.5 h) and the live page at
 *  11:35 UTC (~30% of veteran rates at 11.6 h). Only used on the flat-history
 *  (single-event) path, where no earner slope can be fitted. */
const PROGRESS_PER_HOUR = 0.03;
const PROGRESS_DELAY_HOURS = 2;
const PROGRESS_ALPHA_CAP = 0.6;

interface Trophy {
  id: number;
  description: string;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id?: string;
  levels: number[];
  horizon_minutes: number;
  at: string;
  target: string;
  game?: { release_date?: string };
  trophies: Trophy[];
  history: Read[];
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const logit = (pct: number): number => {
  const p = clampRate(pct);
  return Math.log(p / (100 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Hours between two ISO timestamps. */
const hoursBetween = (a: string, b: string): number =>
  (Date.parse(b) - Date.parse(a)) / 3_600_000;

/** Story/mission trophies progress fast (players reach them in their first hours);
 *  collection/grind trophies progress slowly. Used only to group slopes for shrinkage. */
const MISSION_NAMES =
  /^(Complete (Breakfast Run|The Cartel Heist|Child's Play|Headliners|VCIA Takedown|Operation Savage Glory)\.)/;
const isStory = (d: string): boolean =>
  /Complete Chapter \d+\./.test(d) ||
  /Complete the final mission/.test(d) ||
  MISSION_NAMES.test(d);

/** Ordinary least squares y ~ a + b*x. Returns slope, intercept, SE(slope) or null,
 *  residual sum of squares, mean of x, and n. SE is null when n < 3 or x is constant. */
function ols(xs: number[], ys: number[]): {
  b: number;
  a: number;
  se: number | null;
  xMean: number;
  n: number;
} {
  const n = xs.length;
  const xMean = xs.reduce((s, v) => s + v, 0) / n;
  const yMean = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - xMean) ** 2;
    sxy += (xs[i] - xMean) * (ys[i] - yMean);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  const a = yMean - b * xMean;
  let se: number | null = null;
  if (n >= 3 && sxx > 0) {
    let sse = 0;
    for (let i = 0; i < n; i++) sse += (ys[i] - (a + b * xs[i])) ** 2;
    const dof = n - 2;
    if (dof > 0) se = Math.sqrt(sse / dof / sxx);
  }
  return { b, a, se, xMean, n };
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const nT = c.trophies.length;
  const history = [...c.history].sort((r, s) => Date.parse(r.at) - Date.parse(s.at));
  const lastRead = history[history.length - 1];

  // No history at all: emit a valid, maximally-spread forecast rather than crashing.
  if (!lastRead) {
    return c.trophies.map((t) => ({
      id: t.id,
      q: Z.map((z) => round4(clampRate(invLogit(logit(0.5) + z * SIGMA_UNKNOWN)))),
    }));
  }

  // ---- 1. Compress history to refresh events. ----
  const events: Read[] = [];
  for (const r of history) {
    const prev = events[events.length - 1];
    if (!prev) {
      events.push(r);
      continue;
    }
    let changed = r.players_tracked !== prev.players_tracked;
    if (!changed) {
      for (let i = 0; i < nT; i++) {
        if (r.rates[i] !== prev.rates[i]) {
          changed = true;
          break;
        }
      }
    }
    if (changed) events.push(r);
  }

  const t0 = events[0].at;
  // x = ln(1 + hours since first read); only differences of x are ever used.
  const xOf = (at: string): number => Math.log(1 + Math.max(0, hoursBetween(t0, at)));
  const xEvents = events.map((e) => xOf(e.at));
  const xTarget = xOf(c.target);
  const xLast = xEvents[xEvents.length - 1];
  const nLast = lastRead.players_tracked;
  const horizonDays = c.horizon_minutes / 1440;

  // ---- 2. Forecast N(target). ----
  let nTarget: number;
  let bN: number; // log-slope of N, for the refresh-lag spread component.
  let usePrior = false; // single event: N from the influx prior, E from the progression bump.
  let progressAlpha = 0;
  if (events.length >= 2) {
    const fit = ols(
      xEvents,
      events.map((e) => Math.log(Math.max(1, e.players_tracked))),
    );
    bN = clamp(fit.b, 0, 3);
    if (bN <= 0) {
      nTarget = nLast;
    } else {
      // Refit the intercept at the clamped slope so the level stays anchored.
      const aN =
        events.reduce(
          (s, e, i) => s + Math.log(Math.max(1, e.players_tracked)) - bN * xEvents[i],
          0,
        ) / events.length;
      nTarget = Math.exp(aN + bN * xTarget);
      // Guardrails: N rises monotonically but never explodes.
      nTarget = clamp(nTarget, nLast, nLast * (1 + horizonDays) ** 2);
    }
  } else {
    // Single event: no slope is observable. Launch-night influx prior — the page
    // is a stale pre-release snapshot and a refresh before the target will reveal
    // the players who have started since release. Assume a roughly constant
    // hourly influx after release, each hour adding about as many players as the
    // whole pre-release cohort: N(t) = N_first * (1 + hours since release).
    const rel = c.game?.release_date;
    const tRel = rel ? hoursBetween(`${rel}T00:00:00Z`, c.target) : NaN;
    if (Number.isFinite(tRel) && tRel > 0 && tRel <= 24 * 30) {
      nTarget = Math.min(nLast * (1 + tRel), nLast * PRIOR_N_CAP);
      // New players progress through early content in their first hours, so
      // they earn each trophy at a growing fraction of the veteran rate.
      progressAlpha = Math.min(
        PROGRESS_ALPHA_CAP,
        PROGRESS_PER_HOUR * Math.max(0, tRel - PROGRESS_DELAY_HOURS),
      );
      usePrior = true;
    } else {
      nTarget = nLast;
    }
    // Implied log-slope of the prior, for the refresh-lag component.
    bN = xTarget > xLast ? Math.log(Math.max(1, nTarget / nLast)) / (xTarget - xLast) : 0;
  }

  // ---- 3. Per-trophy earner fits, with group shrinkage. ----
  const story = c.trophies.map((t) => isStory(t.description ?? ''));
  // Earners per event (rounded: 2-decimal rates make earners integers).
  const earners: number[][] = c.trophies.map((_, i) =>
    events.map((e) => (e.rates[i] == null ? NaN : Math.round(((e.rates[i] as number) / 100) * e.players_tracked))),
  );
  const lastEarners: number[] = c.trophies.map((_, i) => {
    for (let k = events.length - 1; k >= 0; k--) {
      if (!Number.isNaN(earners[i][k])) return earners[i][k];
    }
    return NaN;
  });

  // Raw OLS slopes where a trophy has >= 2 positive-earner events.
  const raw = c.trophies.map((_, i) => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let k = 0; k < events.length; k++) {
      const e = earners[i][k];
      if (!Number.isNaN(e) && e > 0) {
        xs.push(xEvents[k]);
        ys.push(Math.log(e));
      }
    }
    if (xs.length < 2) return null;
    const f = ols(xs, ys);
    return { ...f, xs, ys };
  });

  // Group slope = median raw slope in (tier x story); fall back to tier, then all.
  const groupSlope = (i: number): number | null => {
    const tier = c.trophies[i].tier;
    const st = story[i];
    const pool = (pred: (j: number) => boolean): number[] => {
      const out: number[] = [];
      for (let j = 0; j < nT; j++) if (raw[j] && pred(j)) out.push(raw[j]!.b);
      return out;
    };
    let p = pool((j) => c.trophies[j].tier === tier && story[j] === st);
    if (p.length < 2) p = pool((j) => c.trophies[j].tier === tier);
    if (p.length < 2) p = pool(() => true);
    return p.length > 0 ? median(p) : null;
  };

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < nT; i++) {
    const t = c.trophies[i];
    const eLast = lastEarners[i];

    // Trophy every tracked player holds at every event (e.g. the 100% intro trophy):
    // E = N always, so its rate stays 100% with a minimal spread.
    const alwaysFull = events.every((e) => e.rates[i] != null && (e.rates[i] as number) >= 99.99);
    if (alwaysFull) {
      rows.push({ id: t.id, q: Z.map((z) => round4(clampRate(invLogit(logit(99.99) + z * SIGMA_FULL)))) });
      continue;
    }

    // No usable history: widest spread around a low rate.
    if (Number.isNaN(eLast)) {
      rows.push({ id: t.id, q: Z.map((z) => round4(clampRate(invLogit(logit(0.5) + z * SIGMA_UNKNOWN)))) });
      continue;
    }

    // Shrink the slope toward the group, refit the intercept at the shrunk slope.
    const r = raw[i];
    const g = groupSlope(i);
    let bStar = 0;
    let seB: number | null = null;
    let xMean = xLast;
    if (r) {
      const w = r.n / (r.n + K_SHRINK);
      bStar = g == null ? r.b : w * r.b + (1 - w) * g;
      bStar = clamp(bStar, SLOPE_MIN, SLOPE_MAX);
      seB = r.se == null ? null : w * r.se;
      xMean = r.xMean;
    }
    let eTarget: number;
    if (r) {
      const aStar = r.ys.reduce((s, y, k) => s + (y - bStar * r.xs[k]), 0) / r.n;
      eTarget = Math.exp(aStar + bStar * xTarget);
    } else if (usePrior && progressAlpha > 0 && eLast > 0) {
      // Flat history: earners still grow, because the new cohort progresses.
      // New earners = new players x alpha x the trophy's veteran per-player rate.
      const lastRate = lastRead.rates[i];
      const dN = Math.max(0, nTarget - nLast);
      eTarget = eLast + dN * progressAlpha * ((lastRate ?? 0) / 100);
    } else {
      // Earners constant (or a single positive read): hold them — the median then
      // moves only through the denominator N(target).
      eTarget = eLast;
    }
    // Earners rise monotonically and never exceed the players tracked.
    eTarget = clamp(eTarget, eLast, nTarget);

    const medRate = clampRate((100 * eTarget) / nTarget);

    // ---- 4. Spread on the logit scale. ----
    const dX = Math.max(0, xTarget - xMean);
    const sigmaSlope = seB == null ? SIGMA_SLOPE_FALLBACK : seB * dX;
    const sigmaRefresh = Math.abs(bStar - bN) * (REFRESH_LAG_HOURS / 24);
    let sigma = Math.sqrt(sigmaSlope ** 2 + SIGMA_LEVEL ** 2 + sigmaRefresh ** 2);
    sigma = clamp(sigma, SIGMA_MIN, SIGMA_MAX);
    // A trophy nobody has earned yet gets the widest spread.
    if (eLast === 0) sigma = SIGMA_MAX;

    const l = logit(medRate);
    let prev = -Infinity;
    const q = Z.map((z) => {
      const v = clampRate(round4(invLogit(l + z * sigma)));
      const out = Math.max(v, prev);
      prev = out;
      return out;
    });
    rows.push({ id: t.id, q });
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
