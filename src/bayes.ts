/**
 * Multinomial naive Bayes, written from scratch: the in-domain baseline. It learns from the
 * 23,000 IMDB training reviews, unlike the pretrained model, which has never seen IMDB.
 */

export interface BayesOptions {
  alpha: number;      // add-α smoothing
  bigrams: boolean;   // also count adjacent word pairs
  binary: boolean;    // count each feature once per review (Wang & Manning 2012)
  negation: boolean;  // mark words after "not", "no", "never"… until the next punctuation
}

export const defaults: BayesOptions = { alpha: 1, bigrams: false, binary: false, negation: false };

const NEGATORS = new Set(["not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "cannot", "without", "hardly"]);

/** Lowercased word tokens; contractions like "didn't" become "did" + "not". */
export function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/n't\b/g, " not").match(/[a-z0-9]+(?:'[a-z]+)?|[.!?,;:]/g) ?? [];
}

export function features(text: string, o: Pick<BayesOptions, "bigrams" | "negation">): string[] {
  const toks = tokenize(text);
  const words: string[] = [];
  let negated = false;
  for (const t of toks) {
    if (/^[.!?,;:]$/.test(t)) { negated = false; continue; }
    words.push(o.negation && negated ? "NOT_" + t : t);
    if (o.negation && NEGATORS.has(t)) negated = true;
  }
  if (!o.bigrams) return words;
  const out = words.slice();
  for (let i = 1; i < words.length; i++) out.push(words[i - 1] + " " + words[i]);
  return out;
}

export interface BayesModel {
  options: BayesOptions;
  prior: number;                 // log P(pos) − log P(neg)
  weights: Map<string, number>;  // log P(f | pos) − log P(f | neg) for every feature seen in training
}

export function train(texts: readonly string[], labels: readonly (0 | 1)[], options: Partial<BayesOptions> = {}): BayesModel {
  const o = { ...defaults, ...options };
  const counts = [new Map<string, number>(), new Map<string, number>()];
  const totals = [0, 0], docs = [0, 0];
  texts.forEach((text, i) => {
    const y = labels[i];
    docs[y]++;
    let fs = features(text, o);
    if (o.binary) fs = [...new Set(fs)];
    for (const f of fs) { counts[y].set(f, (counts[y].get(f) ?? 0) + 1); totals[y]++; }
  });
  const vocab = new Set([...counts[0].keys(), ...counts[1].keys()]);
  const v = vocab.size;
  const weights = new Map<string, number>();
  for (const f of vocab) {
    const pPos = ((counts[1].get(f) ?? 0) + o.alpha) / (totals[1] + o.alpha * v);
    const pNeg = ((counts[0].get(f) ?? 0) + o.alpha) / (totals[0] + o.alpha * v);
    weights.set(f, Math.log(pPos) - Math.log(pNeg));
  }
  return { options: o, prior: Math.log(docs[1]) - Math.log(docs[0]), weights };
}

/** Log-odds that the review is positive. Features never seen in training are ignored. */
export function score(model: BayesModel, text: string): number {
  let fs = features(text, model.options);
  if (model.options.binary) fs = [...new Set(fs)];
  let s = model.prior;
  for (const f of fs) s += model.weights.get(f) ?? 0;
  return s;
}
