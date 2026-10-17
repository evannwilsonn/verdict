# Verdict demo: design notes

The page is a **two-colour risograph film zine**: issue 07 of a small, cheaply printed magazine about whether people liked the movie.

## Tokens (defined at the top of `template.html`)

- **Paper:** bright white stock (`#fbfbf8`) with a fixed grain layer. Dark mode is "dark stock" (`#121318`).
- **Two inks only:**
  - Riso blue (`#0078bf`) means liked it.
  - Fluorescent pink (`#ff48b0`) means didn't.
  - Where they overlap they multiply, which in dark mode becomes screen. Body text is a dark blue ink, not black.
- **Type:**
  - Anybody, wide and black, for headlines and the verdict stamp.
  - Karla for reading.
  - Martian Mono for figures and labels.
  - Caveat for handwritten margin notes only.
- **One signature:** the misregistered two-ink headline, with the pink layer offset a few pixels from the blue. The halftone dot patch beside it is part of the same print feel.
- **Layout: a cut-and-paste collage**, not a column of cards.
  - Poster headline across the top, corner stickers (source link, ink switch, "issue 07" badge), vertical spine text.
  - The desk: a taped, lined review slip with torn-paper sample strips, and a rubber stamp (SVG, ring text, ink-texture filter) that thumps down with the verdict.
  - The scoreboard: a dot plot with 95% interval lines, a short-vs-long dumbbell chart, and three handwritten notes (Caveat). The full table sits in a `<details>`.
  - A colophon closes the page.
- **The marked-up review** is the core visual: sentences highlighted in blue or pink, with opacity set by how far the verdict moved without them.

## Things to avoid

- A third ink.
- Gradients other than the halftone.
- Card shadows (the offset solid shadow is a print overprint, not elevation).
- Any studio's posters or stills.

## Voice

Zine-casual but exact: "Did they like it?", "Marked up". Numbers are written from the results files, never by hand.
