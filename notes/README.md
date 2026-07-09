# CS224n Study Notes

Maintained by my TA (Claude) as I work through the course.

- **[mistakes-log.md](mistakes-log.md)** — dated log of every misconception + correction.
- **concepts/** — distilled, revision-ready notes by topic (numbered; start with the map).
  - [00-concept-map.md](concepts/00-concept-map.md) — index + how the notes relate.
  - [01-word-embeddings.md](concepts/01-word-embeddings.md) — one-hot → dense embeddings; why "embedding".
  - [02-count-based-svd.md](concepts/02-count-based-svd.md) — co-occurrence + SVD (eigenvalues, Truncated SVD) for a1 Q1.3.
  - [03-word2vec-and-glove.md](concepts/03-word2vec-and-glove.md) — word2vec (neural) & GloVe (hybrid).
  - [04-neural-nets-ner.md](concepts/04-neural-nets-ner.md) — using the vectors: NER window-classification + why non-linearities.
  - [05-backprop-matrix-calculus.md](concepts/05-backprop-matrix-calculus.md) — training the net: matrix calculus (Jacobians) + backpropagation.

How it works: I ask questions / do assignments → mistakes get logged → recurring ones
get distilled into `concepts/` and turned into teaching guidelines in this repo's `CLAUDE.md`
(general personal habits live one level up in `../CLAUDE.md` at the `~/github` workspace root).
