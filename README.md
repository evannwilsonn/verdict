# Verdict

**Did they like it?** Paste a movie review and a DistilBERT sentiment model reads it in your browser, stamps a verdict, and marks up the sentences that decided it. There's no server: the model runs in the tab through [Transformers.js](https://huggingface.co/docs/transformers.js), and nothing you paste leaves the browser.

Demo: `docs/index.html` (served with GitHub Pages).

Project 7 of my AI project run. It's written in TypeScript and follows the same seven steps as the others: problem, data, baseline, model, evaluation, improvement, deployment.

## 1. Problem

Given the text of a movie review, say whether the writer liked the film. Success means accuracy on reviews the system never saw while it was being built, reported with an interval. It also has to run fast enough on a laptop CPU to be used live in a browser.

## 2. Data

The IMDB Large Movie Review Dataset (Maas et al., 2011) has 25,000 training and 25,000 test reviews, each split half positive and half negative. See [`data/README.md`](data/README.md) for the download script and citation. The reviews aren't committed.

The training split was divided once, with a fixed seed (2026), into:

- 2,000 validation reviews, stratified. Every choice below was made on these.
- 23,000 reviews used to fit naive Bayes.

The test split was scored once, at the end.

About 12% of test reviews run longer than DistilBERT's 512-token window. That matters later.

## 3. Baselines

| | Validation | Test |
|---|---|---|
| Always say "positive" | 50.0% | 50.0% |
| Naive Bayes, single words, α = 1 | 84.6% | — |
| Naive Bayes, tuned (bigrams, binary counts, α = 0.5) | 88.5% | 86.2% |

Naive Bayes is written from scratch in [`src/bayes.ts`](src/bayes.ts). The tuned settings came from a 24-combination grid on validation (`results/bayes-grid.json`). Negation marking didn't help once bigrams were on.

## 4. Model

`Xenova/distilbert-base-uncased-finetuned-sst-2-english`, quantised to 8 bits. It was fine-tuned on SST-2 (short movie-review sentences) and was **never trained on IMDB**. On CPU it takes about 50 ms per review.

Out of the box it sees only the first 510 tokens. I compared three ways to handle long reviews on validation:

| Strategy | All | Long reviews only |
|---|---|---|
| head: first 510 tokens | 88.1% | 78.8% |
| headTail: first 128 + last 382 | 88.9% | 84.6% |
| chunks: 510-token windows, averaged | 88.9% | 84.6% |

headTail ties chunks and costs one forward pass instead of several, so it's the one used. Reviews tend to put the verdict at the end.

## 5. Evaluation

Test set, all 25,000 reviews, with 95% Wilson intervals:

| Model | Accuracy | 95% interval | F1 | Short reviews | Long reviews |
|---|---|---|---|---|---|
| Naive Bayes, tuned | 86.15% | 85.71–86.57% | 0.855 | 86.16% | 86.03% |
| DistilBERT, head (default) | 89.02% | 88.63–89.41% | 0.887 | 89.93% | 82.60% |
| DistilBERT, headTail | 89.46% | 89.07–89.83% | 0.892 | 89.93% | 86.10% |
| DistilBERT, headTail, tuned threshold | 89.68% | 89.30–90.05% | 0.896 | 90.05% | 87.10% |
| **Blend: DistilBERT + 0.15 × naive Bayes** | **91.32%** | 90.96–91.66% | 0.914 | 91.58% | 89.47% |

McNemar's exact test on the same reviews:

- **headTail vs head:** 269 reviews only headTail got right, 161 only head got right, p = 2.2e-7.
- **Blend vs headTail:** 849 vs 440, p = 2.1e-30.

## 6. Improvement

These steps were taken in order, each chosen on validation:

1. **Long reviews:** headTail instead of head. +0.4 points on test, nearly all of it from the long 12%.
2. **Threshold:** SST-2 is a different domain. Moving the decision line to log-odds −0.98 fixed a lean toward "negative".
3. **Blend:** naive Bayes and DistilBERT make different mistakes. Naive Bayes counts IMDB-specific words; DistilBERT reads sentences. The blend adds 0.15 × the naive Bayes log-odds to DistilBERT's log-odds, with the weight picked from 11 values on validation (`results/blend-search.json`). It was the largest single gain.

So "never trained on IMDB" holds for DistilBERT alone (89.0% out of the box). The 91.3% blend includes a small model that was trained on IMDB.

## 7. Deployment

`docs/index.html` is a single static page:

- It loads Transformers.js 3.8.1 from jsDelivr and the model from the Hugging Face Hub, then caches both in the browser.
- It runs the deployed configuration: headTail, the blend, and the tuned threshold.
- It explains a verdict by **occlusion**: drop each sentence in turn, re-score, and shade the sentence by how far the verdict moved.

**Pruning:** the browser copy of naive Bayes keeps only features seen in at least 5 training reviews. That's 147,071 of 1.4 million features, about 2.8 MB. The pruned model agrees with the full one on 1,931 of 2,000 validation reviews, and the blend scores 90.95% on validation with it vs 91.5% with the full model. The test numbers above are for the full model.

## Where it still fails

Sarcasm. "Truly the cinematic event of the decade" reads as praise to a model that learned praise words, and the demo has a sample that shows this. Mixed reviews that hedge to the end are also close calls.

## Run it

```sh
npm install
./scripts/get-data.sh                 # or scripts/get-data.ps1 on Windows
npm test                              # 10 unit tests
npm run experiment                    # about an hour on CPU; writes results/*.json
npm run export -- 5                   # writes web/bayes.json (pruned naive Bayes)
python3 scripts/build-demo.py         # writes docs/index.html from the results
```

## Layout

```
src/data.ts         loading, cleaning, seeded split
src/bayes.ts        naive Bayes from scratch
src/transformer.ts  DistilBERT via Transformers.js, three long-review strategies
src/metrics.ts      accuracy, F1, Wilson interval, threshold search, McNemar
src/experiment.ts   the whole run, validation then test
src/export.ts       pruned naive Bayes for the browser
web/template.html   the demo page (design notes in web/DESIGN.md)
results/            every number on the page and in this README
```
