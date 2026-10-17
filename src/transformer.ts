import { AutoModelForSequenceClassification, AutoTokenizer, Tensor } from "@huggingface/transformers";

/**
 * DistilBERT fine-tuned on SST-2 (single sentences from movie reviews), run with
 * Transformers.js, the same library and weights the browser demo loads. It has never seen IMDB.
 */
export const MODEL_ID = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";

/**
 * How to fit a review into the model's 512-token window (510 plus [CLS] and [SEP]):
 * - head: keep the first 510 tokens (what the pipeline does by default)
 * - headTail: keep the first 128 and last 382 (Sun et al. 2019 found this best for long reviews)
 * - chunks: score overlapping 510-token windows and average them
 */
export type Strategy = "head" | "headTail" | "chunks";
export const STRATEGIES: Strategy[] = ["head", "headTail", "chunks"];
const BODY = 510;

export interface Classifier {
  /** Positive-minus-negative logit, i.e. the model's log-odds that the review is positive. */
  logOdds(text: string, strategy: Strategy): Promise<number>;
  tokenCount(text: string): number;
}

export async function load(): Promise<Classifier> {
  const tok = await AutoTokenizer.from_pretrained(MODEL_ID);
  const model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID, { dtype: "q8" });
  const cls = 101, sep = 102; // [CLS] and [SEP] in bert-base-uncased's vocabulary
  const encode = (text: string): number[] => tok.encode(text, { add_special_tokens: false });

  async function run(body: number[]): Promise<number> {
    const ids = [cls, ...body, sep];
    const input_ids = new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const attention_mask = new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]);
    const { logits } = await model({ input_ids, attention_mask });
    const [neg, pos] = logits.data as Float32Array;
    return pos - neg;
  }

  return {
    tokenCount: text => encode(text).length,
    async logOdds(text, strategy) {
      const body = encode(text);
      if (body.length <= BODY || strategy === "head") return run(body.slice(0, BODY));
      if (strategy === "headTail") return run([...body.slice(0, 128), ...body.slice(body.length - (BODY - 128))]);
      const stride = BODY / 2;
      const scores: number[] = [];
      for (let start = 0; ; start += stride) {
        scores.push(await run(body.slice(start, start + BODY)));
        if (start + BODY >= body.length) break;
      }
      return scores.reduce((a, b) => a + b, 0) / scores.length;
    },
  };
}
