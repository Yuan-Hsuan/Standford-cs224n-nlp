# Daily Review Log 每日復盤紀錄

Tracks the end-of-day quiz so weak answers get revisited the next day.
記錄收工考的表現,弱項隔天重問。Newest on top.

**Entry template:**
```
## YYYY-MM-DD — <concept>
- Pushed today? ✅ / ❌
- Quizzed on: <questions asked>
- Strong 答得好: <...>
- Weak / revisit tomorrow 要復盤: <...>
```

---

<!-- entries go below -->

## 2026-07-04 — Word vectors (SVD, word2vec, GloVe, polysemy)
- Pushed today? ⏳ batch pending at wrap-up (notes reorg + mastery system)
- Quizzed on: Q1 why word vec = U·S; Q2 GloVe crucial insight + how it combines count/prediction; Q3 why "mouse" shows both senses but many polysemes don't
- Strong 答得好: Q2 big picture — knew GloVe combines count-based + prediction and uses global counts; "relative vector" idea was near the ratios insight
- Weak / revisit tomorrow 要復盤:
  - **Q1**: confused **singular value with eigenvalue** — S holds singular values σ = √λ (of AᵀA), NOT eigenvalues. Also missed the "why U·S": U = direction only, S = importance scaling.
  - **Q3**: polysemy not yet absorbed — the dominant-sense-frequency reason. Do it properly tomorrow.
