# Mistakes Log

A running, dated log of misconceptions and errors — raw material for the concept notes.
Newest entries at the top.

**Entry template:**
```
## YYYY-MM-DD — <short title>
- **Context:** <lecture / assignment / question>
- **What I thought / did:** <the mistake>
- **Correct understanding:** <the fix, with the why>
- **Pattern tag:** <e.g. tensor-shapes, gradient-math, broadcasting, terminology>
```

---

<!-- entries go below -->

## 2026-07-03 — Word vector = rows of U·S, not U alone
- **Context:** a1 Q1.3 `reduce_to_k_dim` + SVD concept note. (User caught an imprecise definition in the note by checking against `Singular_Value_Decomposition_Tutorial.pdf`.)
- **What was wrong:** The note said "rows of U are the word vectors."
- **Correct understanding:** The word vector is a row of **U·S**, not U alone. U gives only the *direction* on each latent-topic axis; multiplying by S scales each axis by that topic's importance (σ). Kirk Baker's tutorial computes word similarity on rows of **US** (documents on rows of VS). Confirmed by a1's docstring: `reduce_to_k_dim` "actually returns U * S", and `TruncatedSVD.fit_transform` returns U·S.
- **Pattern tag:** svd, definitions-precision, assignment-detail
