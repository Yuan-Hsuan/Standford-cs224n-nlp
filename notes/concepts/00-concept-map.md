# 00 · Concept Map 概念地圖 — how the notes relate

> **Related 關聯**: [[01-word-embeddings]] · [[02-count-based-svd]] · [[03-word2vec-and-glove]] · [[04-neural-nets-ner]].

---

## 1. The notes 筆記清單

| # | Note | Base | Description |
|---|------|------------|------|
| **01** | [[01-word-embeddings]] | foundation 基礎 | What a word embedding is and why. |
| **02** | [[02-count-based-svd]] | count-based 計數式(傳統） | Co-occurrence matrix + SVD. |
| **03** | [[03-word2vec-and-glove]] | neural + hybrid 神經+混合 | word2vec (predictive) and GloVe (combines count + prediction). |
| **04** | [[04-neural-nets-ner]] | neural classifier 神經分類器 | *Using* the vectors: NER window-classification + why non-linearities. |
| **05** | [[05-backprop-matrix-calculus]] | training 訓練 | How the net *learns*: matrix calculus (Jacobians) + backpropagation. |

---

## 2. How they relate 關係

- **word representation is the goal.** All the others are just different ways to build the vectors.
  01 是目標,其他都是「做出這種向量」的不同方法。
- **Co-occurrence matrix is the traditional way (count-based).** Count global co-occurrence stats, then compress with SVD.
  02 是傳統計數法:數全域共現,再用 SVD 壓縮。
- **word2vec is the neural way (prediction-based).** Slide a window and learn by predicting context words.
  word2vec 是神經預測法:滑窗預測上下文來學。
- **GloVe combines both.** It uses global co-occurrence counts (like 02) but learns the vectors
  (like word2vec) — so it gets the strengths of each. GloVe 結合兩者:用全域統計 + 用學習。

- **Then we *use* the vectors.** 04 drops them into a classifier: window-based **NER**, and why a neural
  net (with **non-linearities**) beats a linear one. 01–03 做出向量,04 拿去用在分類器。
- **And we *train* the net.** 05 is the learning machinery: **matrix calculus** + **backpropagation** to
  compute $\nabla_\theta J$ for SGD. 05 是訓練機制:矩陣微積分 + 反向傳播算梯度。

**Order 順序:** traditional (02) → neural (word2vec) → hybrid (GloVe) → use them (04) → train them (05).
傳統 → 神經 → 混合 → 拿去用 → 訓練。

---

## 3. Map to assignments 對應作業

- **a1 Part 1** → 02 count-based SVD
- **a1 Part 2** → 03 GloVe pretrained vectors 
- **a2** → 03 word2vec skip-gram + negative sampling