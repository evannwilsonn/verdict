/**
 * The whole experiment, in order:
 *   1. hold out 2,000 training reviews as validation (the other 23,000 train naive Bayes)
 *   2. baselines and the naive Bayes settings, chosen on validation
 *   3. the pretrained model three ways (how it handles long reviews), on validation
 *   4. improvements chosen on validation: long-review strategy, decision threshold, blending with naive Bayes
 *   5. one evaluation on all 25,000 test reviews
 * Model outputs are cached in results/cache/ so a re-run only scores what's missing.
 *
 *   npm run experiment
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { load, stratifiedSplit, type Review } from "./data.js";
import * as bayes from "./bayes.js";
import { bestThreshold, evaluate, mcnemar, type Scores } from "./metrics.js";
import * as transformer from "./transformer.js";

const SEED = 2026;
const DATA = "data/aclImdb";
mkdirSync("results/cache", { recursive: true });

const r4 = (x: number) => Math.round(x * 1e4) / 1e4;
const pct = (x: number) => (100 * x).toFixed(2) + "%";
const line = (name: string, s: Scores) =>
  console.log(`  ${name.padEnd(46)} acc ${pct(s.accuracy)} [${pct(s.accuracyLow)}–${pct(s.accuracyHigh)}]  P ${s.precision.toFixed(3)}  R ${s.recall.toFixed(3)}  F1 ${s.f1.toFixed(3)}`);
const row = (name: string, step: string, s: Scores, extra: Record<string, unknown> = {}) => ({
  name, step, n: s.n, accuracy: r4(s.accuracy), accuracyLow: r4(s.accuracyLow), accuracyHigh: r4(s.accuracyHigh),
  precision: r4(s.precision), recall: r4(s.recall), f1: r4(s.f1),
  truePos: s.truePos, falsePos: s.falsePos, falseNeg: s.falseNeg, trueNeg: s.trueNeg, ...extra,
});
const save = (file: string, value: unknown) => writeFileSync(`results/${file}`, JSON.stringify(value, null, 2) + "\n");

// ---- 1. data -----------------------------------------------------------------------------
const trainAll = load(DATA, "train"), test = load(DATA, "test");
const [valid, fit] = stratifiedSplit(trainAll, 2000, SEED);
const yValid = valid.map(r => r.label), yTest = test.map(r => r.label);
console.log(`train ${trainAll.length} (fit ${fit.length} + validation ${valid.length}), test ${test.length}`);
console.log(`positive share: fit ${pct(fit.filter(r => r.label).length / fit.length)}, validation ${pct(yValid.filter(Boolean).length / valid.length)}, test ${pct(yTest.filter(Boolean).length / test.length)}`);

// ---- 2. baselines --------------------------------------------------------------------------
const validRows: object[] = [];
console.log("\nBaselines (validation)");
const majority = evaluate(valid.map(() => 1), yValid, 0.5);
line("Always positive", majority); validRows.push(row("Always positive", "baseline", majority));

const nbPlain = bayes.train(fit.map(r => r.text), fit.map(r => r.label));
const nbPlainVal = evaluate(valid.map(r => bayes.score(nbPlain, r.text)), yValid);
line("Naive Bayes, words, α = 1", nbPlainVal); validRows.push(row("Naive Bayes, words, α = 1", "baseline", nbPlainVal));

console.log("\nNaive Bayes settings (validation accuracy)");
const grid: object[] = [];
let bestNb = { acc: 0, options: bayes.defaults };
for (const alpha of [0.1, 0.5, 1]) for (const bigrams of [false, true]) for (const binary of [false, true]) for (const negation of [false, true]) {
  const options = { alpha, bigrams, binary, negation };
  const m = bayes.train(fit.map(r => r.text), fit.map(r => r.label), options);
  const acc = evaluate(valid.map(r => bayes.score(m, r.text)), yValid).accuracy;
  grid.push({ ...options, accuracy: r4(acc) });
  if (acc > bestNb.acc + 1e-9) bestNb = { acc, options };
}
console.log(`  best: ${JSON.stringify(bestNb.options)} → ${pct(bestNb.acc)}`);
save("bayes-grid.json", grid);
const nb = bayes.train(fit.map(r => r.text), fit.map(r => r.label), bestNb.options);
const nbValid = valid.map(r => bayes.score(nb, r.text));
const nbBestVal = evaluate(nbValid, yValid);
line("Naive Bayes, tuned", nbBestVal); validRows.push(row("Naive Bayes, tuned", "baseline", nbBestVal, { options: bestNb.options }));

// ---- 3. pretrained model ---------------------------------------------------------------------
const clf = await transformer.load();

async function scored(reviews: Review[], strategy: transformer.Strategy, tag: string): Promise<number[]> {
  const file = `results/cache/${tag}-${strategy}.json`;
  const cache: Record<string, number> = existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : {};
  let done = 0; const t0 = Date.now();
  for (const r of reviews) {
    if (!(r.id in cache)) {
      cache[r.id] = Math.round((await clf.logOdds(r.text, strategy)) * 1e5) / 1e5;
      if (++done % 1000 === 0) {
        writeFileSync(file, JSON.stringify(cache));
        console.log(`    ${tag}/${strategy}: ${done} scored, ${((Date.now() - t0) / done).toFixed(0)} ms each`);
      }
    }
  }
  writeFileSync(file, JSON.stringify(cache));
  return reviews.map(r => cache[r.id]);
}

const longValid = valid.map(r => clf.tokenCount(r.text) > 510);
console.log(`\nPretrained DistilBERT-SST2, no IMDB training (validation; ${pct(longValid.filter(Boolean).length / valid.length)} of reviews exceed 510 tokens)`);
const validBy: Record<string, number[]> = {};
for (const s of transformer.STRATEGIES) {
  validBy[s] = await scored(valid, s, "valid");
  const sc = evaluate(validBy[s], yValid);
  const longOnly = evaluate(validBy[s].filter((_, i) => longValid[i]), yValid.filter((_, i) => longValid[i]));
  line(`DistilBERT, ${s}`, sc);
  console.log(`    on long reviews only: ${pct(longOnly.accuracy)} (${longOnly.n})`);
  validRows.push(row(`DistilBERT, ${s}`, s === "head" ? "model" : "improvement", sc, { longAccuracy: r4(longOnly.accuracy), longN: longOnly.n }));
}

// ---- 4. improvements chosen on validation ----------------------------------------------------
const strategy = transformer.STRATEGIES.reduce((a, b) => evaluate(validBy[b], yValid).accuracy > evaluate(validBy[a], yValid).accuracy + 1e-9 ? b : a);
const threshold = bestThreshold(validBy[strategy], yValid);
const thrVal = evaluate(validBy[strategy], yValid, threshold);
console.log(`\nChosen long-review strategy: ${strategy}; best threshold ${threshold.toFixed(3)} (0 = the model's own 50/50 point)`);
line(`DistilBERT, ${strategy}, threshold ${threshold.toFixed(2)}`, thrVal);
validRows.push(row(`DistilBERT, ${strategy}, tuned threshold`, "improvement", thrVal, { threshold: r4(threshold) }));

// Blend: DistilBERT log-odds + w × naive Bayes log-odds. Naive Bayes log-odds run much larger
// (it adds up hundreds of word weights), so w is searched on a log scale.
const ws = [0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5];
const blendSearch = ws.map(w => {
  const s = validBy[strategy].map((x, i) => x + w * nbValid[i]);
  const t = bestThreshold(s, yValid);
  return { w, threshold: t, accuracy: evaluate(s, yValid, t).accuracy };
});
const bestBlend = blendSearch.reduce((a, b) => b.accuracy > a.accuracy + 1e-9 ? b : a);
const blendVal = evaluate(validBy[strategy].map((x, i) => x + bestBlend.w * nbValid[i]), yValid, bestBlend.threshold);
console.log(`Blend weight search: ${blendSearch.map(b => `${b.w}→${pct(b.accuracy)}`).join(", ")}`);
line(`Blend: DistilBERT + ${bestBlend.w} × naive Bayes`, blendVal);
validRows.push(row(`Blend: DistilBERT + ${bestBlend.w} × naive Bayes`, "improvement", blendVal, { weight: bestBlend.w, threshold: r4(bestBlend.threshold) }));
save("validation.json", validRows);
save("blend-search.json", blendSearch.map(b => ({ weight: b.w, threshold: r4(b.threshold), accuracy: r4(b.accuracy) })));

// ---- 5. test, once -----------------------------------------------------------------------
console.log(`\nTest set: all ${test.length} reviews`);
const nbTest = test.map(r => bayes.score(nb, r.text));
const headTest = await scored(test, "head", "test");
const chosenTest = strategy === "head" ? headTest : await scored(test, strategy, "test");
const blendTest = chosenTest.map((x, i) => x + bestBlend.w * nbTest[i]);
const longTest = test.map(r => clf.tokenCount(r.text) > 510);

const finals: [string, number[], number][] = [
  ["Always positive", test.map(() => 1), 0.5],
  ["Naive Bayes, tuned", nbTest, 0],
  ["DistilBERT, head (default)", headTest, 0],
  [`DistilBERT, ${strategy}`, chosenTest, 0],
  [`DistilBERT, ${strategy}, tuned threshold`, chosenTest, threshold],
  [`Blend: DistilBERT + ${bestBlend.w} × naive Bayes`, blendTest, bestBlend.threshold],
];
const testRows = finals.map(([name, s, t]) => {
  const sc = evaluate(s, yTest, t);
  const lg = evaluate(s.filter((_, i) => longTest[i]), yTest.filter((_, i) => longTest[i]), t);
  const sh = evaluate(s.filter((_, i) => !longTest[i]), yTest.filter((_, i) => !longTest[i]), t);
  line(name, sc);
  console.log(`    short reviews ${pct(sh.accuracy)} (${sh.n}), long reviews ${pct(lg.accuracy)} (${lg.n})`);
  return row(name, "test", sc, { threshold: r4(t), shortAccuracy: r4(sh.accuracy), shortN: sh.n, longAccuracy: r4(lg.accuracy), longN: lg.n });
});
save("test.json", testRows);

const correct = (s: number[], t: number) => s.map((x, i) => (x >= t) === (yTest[i] === 1));
const comparisons = [
  ["Naive Bayes vs DistilBERT head", correct(nbTest, 0), correct(headTest, 0)],
  [`DistilBERT head vs ${strategy}`, correct(headTest, 0), correct(chosenTest, 0)],
  [`DistilBERT ${strategy} vs blend`, correct(chosenTest, threshold), correct(blendTest, bestBlend.threshold)],
] as const;
console.log("\nPaired comparisons on the test set (McNemar's exact test)");
const tests = comparisons.map(([name, a, b]) => {
  const m = mcnemar(a, b);
  console.log(`  ${name}: only first right ${m.onlyA}, only second right ${m.onlyB}, p = ${m.p.toExponential(2)}`);
  return { comparison: name, onlyFirstRight: m.onlyA, onlySecondRight: m.onlyB, p: Number(m.p.toPrecision(3)) };
});
save("significance.json", tests);
save("settings.json", {
  seed: SEED, validation: valid.length, test: test.length, model: transformer.MODEL_ID, dtype: "q8",
  bayes: bestNb.options, strategy, threshold: r4(threshold), blend: { weight: bestBlend.w, threshold: r4(bestBlend.threshold) },
  longReviewTokens: 510, testLongShare: r4(longTest.filter(Boolean).length / test.length),
});
console.log("\nWrote results/");
