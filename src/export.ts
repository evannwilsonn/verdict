/**
 * Exports what the browser demo needs from naive Bayes: its settings, prior and a pruned set of
 * feature weights (features seen in at least MIN_DOCS training reviews). Pruning keeps the file
 * small; the script reports how often the pruned model agrees with the full one on validation.
 *
 *   npm run export
 */
import { readFileSync, writeFileSync } from "node:fs";
import { load, stratifiedSplit } from "./data.js";
import * as bayes from "./bayes.js";

const MIN_DOCS = Number(process.argv[2] ?? 10);
const settings = JSON.parse(readFileSync("results/settings.json", "utf8"));
const trainAll = load("data/aclImdb", "train");
const [valid, fit] = stratifiedSplit(trainAll, 2000, settings.seed);
const full = bayes.train(fit.map(r => r.text), fit.map(r => r.label), settings.bayes);

const docs = new Map<string, number>();
for (const r of fit) for (const f of new Set(bayes.features(r.text, settings.bayes))) docs.set(f, (docs.get(f) ?? 0) + 1);
const weights: Record<string, number> = {};
for (const [f, w] of full.weights) if ((docs.get(f) ?? 0) >= MIN_DOCS) weights[f] = Math.round(w * 1e3) / 1e3;
const pruned: bayes.BayesModel = { options: full.options, prior: full.prior, weights: new Map(Object.entries(weights)) };

let agree = 0;
for (const r of valid) if ((bayes.score(full, r.text) >= 0) === (bayes.score(pruned, r.text) >= 0)) agree++;
const out = { options: full.options, prior: Math.round(full.prior * 1e6) / 1e6, minDocs: MIN_DOCS, weights };
const json = JSON.stringify(out);
writeFileSync("web/bayes.json", json);
console.log(`kept ${Object.keys(weights).length} of ${full.weights.size} features (seen in ${MIN_DOCS}+ reviews), ${(json.length / 1e6).toFixed(2)} MB`);
console.log(`pruned vs full naive Bayes agree on ${agree} of ${valid.length} validation reviews`);
