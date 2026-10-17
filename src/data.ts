import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

/** One IMDB review. `stars` is the reviewer's own 1–10 score, taken from the file name. */
export interface Review {
  id: string;
  split: "train" | "test";
  label: 0 | 1; // 1 = positive (7–10 stars), 0 = negative (1–4 stars)
  stars: number;
  text: string;
}

/** The raw files use <br /> for line breaks. */
export function clean(text: string): string {
  return text.replace(/<br\s*\/?>/gi, "\n").replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

/** Reads aclImdb/{train,test}/{pos,neg}/<id>_<stars>.txt. */
export function load(dir: string, split: "train" | "test"): Review[] {
  const out: Review[] = [];
  for (const [folder, label] of [["neg", 0], ["pos", 1]] as const) {
    const path = join(dir, split, folder);
    for (const file of readdirSync(path).sort()) {
      const m = file.match(/^(\d+)_(\d+)\.txt$/);
      if (!m) continue;
      out.push({ id: `${split}/${folder}/${m[1]}`, split, label, stars: Number(m[2]), text: clean(readFileSync(join(path, file), "utf8")) });
    }
  }
  return out;
}

/** Small, fast, seedable PRNG (mulberry32), so every sample is reproducible. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffle<T>(items: readonly T[], seed: number): T[] {
  const out = items.slice(), r = rng(seed);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** Splits off `n` reviews with equal positives and negatives; returns [sample, rest]. */
export function stratifiedSplit(reviews: readonly Review[], n: number, seed: number): [Review[], Review[]] {
  const picked = new Set<string>();
  for (const label of [0, 1] as const) {
    shuffle(reviews.filter(r => r.label === label), seed + label).slice(0, n / 2).forEach(r => picked.add(r.id));
  }
  return [reviews.filter(r => picked.has(r.id)), reviews.filter(r => !picked.has(r.id))];
}
