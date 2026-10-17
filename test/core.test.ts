import { test } from "node:test";
import assert from "node:assert/strict";
import { clean, rng, stratifiedSplit, type Review } from "../src/data.js";
import * as bayes from "../src/bayes.js";
import { bestThreshold, evaluate, mcnemar, wilson } from "../src/metrics.js";

test("clean turns <br /> into line breaks and squeezes spaces", () => {
  assert.equal(clean("Great.<br /><br />Really   great."), "Great.\n\nReally great.");
});

test("the PRNG is reproducible", () => {
  const a = rng(7), b = rng(7);
  assert.deepEqual([a(), a(), a()], [b(), b(), b()]);
});

test("stratified split takes equal positives and negatives and loses nothing", () => {
  const reviews: Review[] = Array.from({ length: 100 }, (_, i) => ({ id: String(i), split: "train", label: (i % 4 === 0 ? 1 : 0) as 0 | 1, stars: 1, text: "" }));
  const [sample, rest] = stratifiedSplit(reviews, 20, 3);
  assert.equal(sample.length, 20);
  assert.equal(sample.filter(r => r.label === 1).length, 10);
  assert.equal(sample.length + rest.length, 100);
  assert.equal(new Set([...sample, ...rest].map(r => r.id)).size, 100);
});

test("tokenizer splits contractions and keeps punctuation for negation scope", () => {
  assert.deepEqual(bayes.tokenize("I didn't love it."), ["i", "did", "not", "love", "it", "."]);
  assert.deepEqual(bayes.features("not good, but fine", { bigrams: false, negation: true }), ["not", "NOT_good", "but", "fine"]);
});

test("naive Bayes learns the obvious words and matches the hand-computed weight", () => {
  const texts = ["great fun", "great acting", "dull plot", "dull and slow"];
  const labels: (0 | 1)[] = [1, 1, 0, 0];
  const m = bayes.train(texts, labels, { alpha: 1 });
  assert.ok(bayes.score(m, "great") > 0);
  assert.ok(bayes.score(m, "dull") < 0);
  // vocabulary {great, fun, acting, dull, plot, and, slow} = 7; positive tokens 4, negative 5
  const expected = Math.log((2 + 1) / (4 + 7)) - Math.log((0 + 1) / (5 + 7));
  assert.ok(Math.abs(m.weights.get("great")! - expected) < 1e-12);
});

test("binary naive Bayes counts a word once per review", () => {
  const m = bayes.train(["good good good", "bad"], [1, 0], { binary: true, alpha: 1 });
  assert.equal(bayes.score(m, "good good"), bayes.score(m, "good"));
});

test("evaluate counts the confusion matrix", () => {
  const s = evaluate([2, 1, -1, -2, 0.5], [1, 0, 1, 0, 1]);
  assert.deepEqual([s.truePos, s.falsePos, s.falseNeg, s.trueNeg], [2, 1, 1, 1]);
  assert.equal(s.accuracy, 0.6);
});

test("the Wilson interval matches a published value", () => {
  // 81 successes out of 263: 95% interval 0.2553 – 0.3662 (Newcombe 1998, example)
  const [lo, hi] = wilson(81, 263);
  assert.ok(Math.abs(lo - 0.2553) < 5e-4 && Math.abs(hi - 0.3662) < 5e-4, `${lo} ${hi}`);
});

test("bestThreshold finds the cut that separates the classes", () => {
  const t = bestThreshold([-3, -2, 1, 2, 3], [0, 0, 0, 1, 1]);
  assert.ok(t > 1 && t < 2, String(t));
  assert.equal(evaluate([-3, -2, 1, 2, 3], [0, 0, 0, 1, 1], t).accuracy, 1);
});

test("McNemar's exact test on a hand-checked case", () => {
  // 6 discordant pairs all favouring B: p = 2 × 0.5^6 = 0.03125
  const a = [false, false, false, false, false, false, true];
  const b = [true, true, true, true, true, true, true];
  const m = mcnemar(a, b);
  assert.deepEqual([m.onlyA, m.onlyB], [0, 6]);
  assert.ok(Math.abs(m.p - 0.03125) < 1e-12);
});
