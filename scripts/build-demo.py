#!/usr/bin/env python3
"""Builds docs/index.html from web/template.html. It embeds the test results, the settings the
experiment chose, and (when the blend is the final model) the pruned naive Bayes weights, so
the page works on its own. The deployed configuration is whichever improvement scored best on
validation; the notes on the page are written from the actual numbers."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
load = lambda p: json.loads((root / p).read_text(encoding="utf-8"))
import sys
preview = "--preview" in sys.argv
s, valid = load("results/settings.json"), load("results/validation.json")
if preview:
    # Before the test run finishes: show validation numbers, clearly labelled, and no test claims.
    test = [dict(r, name=("DistilBERT, head (default)" if r["name"] == "DistilBERT, head" else r["name"]), shortAccuracy=None,
                 longAccuracy=r.get("longAccuracy")) for r in valid if r["name"] != "Naive Bayes, words, α = 1"]
    sig = [{"onlyFirstRight": 0, "onlySecondRight": 0, "p": 1}] * 3
    s.setdefault("testLongShare", 0.13)
else:
    test, sig = load("results/test.json"), load("results/significance.json")
pct = lambda x: f"{100 * x:.1f}%"
by = {r["name"]: r for r in test}

strategy, thr, blend = s["strategy"], s["threshold"], s["blend"]
names = {
    "strategy": f"DistilBERT, {strategy}",
    "threshold": f"DistilBERT, {strategy}, tuned threshold",
    "blend": f"Blend: DistilBERT + {blend['weight']} × naive Bayes",
}
val_by = {r["name"]: r["accuracy"] for r in valid}
val_names = {"strategy": f"DistilBERT, {strategy}", "threshold": f"DistilBERT, {strategy}, tuned threshold", "blend": names["blend"]}
final = max(val_names, key=lambda k: (val_by[val_names[k]], -list(val_names).index(k)))
if final == "blend" and blend["weight"] == 0:
    final = "threshold"

head, nb, chosen = by["DistilBERT, head (default)"], by["Naive Bayes, tuned"], by[names["strategy"]]
fin = by[names[final]]
label = {"strategy": "DistilBERT, long reviews handled", "threshold": "DistilBERT, tuned", "blend": "DistilBERT + naive Bayes"}[final]
note = {
    "strategy": f"Long reviews read as {'first 128 + last 382 tokens' if strategy == 'headTail' else 'overlapping windows'}",
    "threshold": "Long-review handling plus a decision line tuned on validation",
    "blend": f"DistilBERT's log-odds plus {blend['weight']} × naive Bayes's",
}[final]
strategy_words = {"head": "only its first 510 tokens", "headTail": "its first 128 and last 382 tokens", "chunks": "overlapping 510-token windows, averaged"}
long_note = (f"{pct(s['testLongShare'])} of test reviews are longer than the model's 512-token window. Out of the box it reads only the start "
             f"and gets {pct(head['longAccuracy'])} of those right. Reading {strategy_words[strategy]} instead gets {pct(chosen['longAccuracy'])}.")
blend_row = by[names["blend"]]
blend_note = (f"Naive Bayes counts words; DistilBERT reads sentences. They make different mistakes, so adding a small share of naive Bayes "
              f"({blend['weight']} × its log-odds) to DistilBERT's moved test accuracy to {pct(blend_row['accuracy'])}." if blend["weight"] > 0 else
              "Blending in naive Bayes didn't help on validation, so the deployed model is DistilBERT alone.")
pair = sig[-1] if final == "blend" else sig[1]
sig_note = (f"On the same 25,000 reviews, {pair['onlySecondRight']:,} were right only after the change and {pair['onlyFirstRight']:,} only before it. "
            f"McNemar's test puts the chance of a gap that size by luck at p = {pair['p']:.2g}.")
row_notes = {
    "Always positive": "Calls every review positive",
    "Naive Bayes, tuned": "From scratch, " + ", ".join(k for k, v in [("bigrams", s["bayes"]["bigrams"]), ("binary counts", s["bayes"]["binary"]), ("negation marks", s["bayes"]["negation"])] if v) + f", α = {s['bayes']['alpha']}",
    "DistilBERT, head (default)": "Pretrained on SST-2, first 510 tokens",
    names["strategy"]: "Same model, " + strategy_words[strategy],
    names["threshold"]: f"Decision line moved to log-odds {thr:.2f}",
    names["blend"]: f"Plus naive Bayes, line at {blend['threshold']:.2f}",
}
if preview:
    long_note = "13% of reviews are longer than the model's 512-token window, and reading only their start costs accuracy. Validation numbers for now."
    blend_note = f"Blending in a small share of naive Bayes ({blend['weight']} × its log-odds) scored best on validation."
    sig_note = "The significance test runs on the full 25,000-review test set, which is still being scored."
settings = {"preview": "Validation numbers (2,000 reviews held out from training) while the 25,000-review test set is scored. Long-review column shows DistilBERT only." if preview else None,
            "strategy": strategy, "threshold": thr, "blend": blend, "final": final, "finalName": names[final],
            "finalLabel": label, "finalNote": note, "notes": {"long": long_note, "blend": blend_note, "sig": sig_note}, "rowNotes": row_notes}

page = (root / "web" / "template.html").read_text(encoding="utf-8")
bayes = (root / "web" / "bayes.json").read_text(encoding="utf-8") if final == "blend" else "null"
for marker, data in [("SETTINGS", json.dumps(settings, ensure_ascii=False)), ("RESULTS", json.dumps(test, ensure_ascii=False)), ("BAYES", bayes)]:
    page = page.replace(f"/*{marker}*/null", data.replace("</", "<\\/"))
assert "/*SETTINGS*/" not in page and "/*RESULTS*/" not in page and "/*BAYES*/" not in page
(root / "docs").mkdir(exist_ok=True)
(root / "docs" / "index.html").write_text(page, encoding="utf-8")
print(f"Deployed configuration: {final} ({names[final]}, test {pct(fin['accuracy'])}). Wrote docs/index.html ({len(page.encode()):,} bytes)")
