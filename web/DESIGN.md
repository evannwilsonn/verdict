# Verdict demo: design notes

The page is a **marked-up newspaper review**, the same piece as the project thumbnail (`.github/thumbnail.jpg`): a masthead on newsprint, a review an editor has gone over with two pens, and a rubber stamp with the verdict.

## Tokens (defined at the top of `template.html`)

- **Paper:** newsprint (`#f2eee3`) with a faint ink speckle tile. Dark mode is "night edition": dark newsprint (`#16140f`), light ink type, the same speckle in light ink.
- **Ink:** near-black `#1f1d1a` for type (light ink `#ece6d6` in dark mode).
- **Two pens only:**
  - Green (`#2f7d4a`; `#5fb87e` in dark mode) means liked it.
  - Red (`#b23a2c`; `#e8735f` in dark mode) means didn't.
  - Small green text uses the deeper `--green-ink` so it passes AA on newsprint.
- **Type** (self-hosted in `docs/fonts`, SIL OFL):
  - Gloock for the masthead, section heads and the stamp.
  - IBM Plex Serif for reading, figures and letter-spaced capital labels.
  - Nothing You Could Do for handwritten margin notes only.
- **Signatures:**
  - The thick + thin double rule under the masthead and above each section.
  - Hand-drawn wavy underlines on the review, green or red, heavier and darker the more a sentence moved the verdict, with margin notes ("+ decided it", "− against").
  - A tilted, double-bordered rubber stamp (SVG, ink-texture filter) that thumps down: green "LIKED IT"/"LOVED IT", red "DIDN'T LIKE IT"/"HATED IT".
- **Layout:** dateline, masthead, deck; then the desk (a ruled letter box with sample clippings, and the stamp); the marked-up review; the scoreboard (dot plot with 95% intervals, short-vs-long dumbbell, three handwritten notes, table in a `<details>`); a colophon.

## Things to avoid

- A third pen colour.
- Gradients, shadows or rounded cards.
- Any studio's posters or stills.

## Voice

Plain and exact: "Did they like it?", "Marked up". Numbers are written from the results files, never by hand.
