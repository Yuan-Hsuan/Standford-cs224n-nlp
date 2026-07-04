# 00 · Concept Map — how the notes relate

## The notes 筆記清單

| # | Note | Base | Description |
|---|------|------------|------|
| **01** | [[01-word-embeddings]] | foundation 基礎 | What a word embedding is and why. |
| **02** | [[02-count-based-svd]] | count-based 計數式(傳統） | Co-occurrence matrix + SVD. |
| **03** | [[03-word2vec-and-glove]] | neural + hybrid 神經+混合 | word2vec (predictive) and GloVe (combines count + prediction). |

---

## How they relate 關係

- **word representation is the goal.** All the others are just different ways to build the vectors.
  01 是目標,其他都是「做出這種向量」的不同方法。
- **Co-occurrence matrix is the traditional way (count-based).** Count global co-occurrence stats, then compress with SVD.
  02 是傳統計數法:數全域共現,再用 SVD 壓縮。
- **word2vec is the neural way (prediction-based).** Slide a window and learn by predicting context words.
  word2vec 是神經預測法:滑窗預測上下文來學。
- **GloVe combines both.** It uses global co-occurrence counts (like 02) but learns the vectors
  (like word2vec) — so it gets the strengths of each. GloVe 結合兩者:用全域統計 + 用學習。

**Order 順序:** traditional (02) → neural (word2vec) → hybrid (GloVe). 傳統 → 神經 → 混合。

---

## Map to assignments 對應作業

- **a1 Part 1** → 02 count-based SVD
- **a1 Part 2** → 03 GloVe pretrained vectors 
- **a2** → 03 word2vec skip-gram + negative sampling