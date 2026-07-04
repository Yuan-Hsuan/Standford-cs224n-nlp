# Word Embeddings 詞嵌入

> Foundational concept for the whole course. 全課程的基礎概念。
> Everything from a1 (co-occurrence + SVD) to a5 (BERT) is about *learning good vector
> representations of meaning*. 從 a1 到 a5 本質都在學「怎麼用向量表示詞義」。
> Related: [[svd-dimensionality-reduction]]

---

## 1. The problem: represent a word as numbers 問題:如何用數字表示一個詞

A model can only do math on numbers, not on the string `"cat"`. So each word must become a
vector. The question is *which* vector.
模型只能對數字做運算,不能直接算字串 `"cat"`,所以每個詞必須變成一個向量——問題是**用哪個向量**。

---

## 2. The naive answer: one-hot vectors 獨熱編碼

Give every word in the vocabulary its **own dimension**. 每個詞佔一個維度。
With a 10,000-word vocabulary, each word is a length-10,000 vector: `1` in its own slot, `0`
elsewhere.

- `cat` = `[0, 0, 1, 0, …, 0]`  (1 in position 3)
- `dog` = `[0, 1, 0, 0, …, 0]`  (1 in position 2)

This is **"one dimension per word"** — vocab size = number of dimensions.
「一個詞一個維度」,詞典多大維度就多大。

### Two fatal problems 兩個致命問題
1. **High-dimensional & sparse** 高維、稀疏 — 10,000 dims, almost all zeros. Wasteful.
2. **No sense of similarity** 無法表達相似 — any two distinct one-hot vectors are *orthogonal*,
   so their dot product is 0. Mathematically `distance(cat, dog) == distance(cat, table)`.
   任意兩個詞向量都垂直、點積為 0 → 模型看不出「貓和狗」比「貓和桌子」更像。

---

## 3. The fix: embed into a low-dim, dense space 嵌入低維稠密空間

An **embedding** maps each word from the huge sparse one-hot space into a much smaller
**continuous** space — typically **50 / 100 / 300** real-valued dimensions.
把每個詞從上萬維的稀疏空間,映射到一個低得多、每維都是連續實數的稠密空間。

- `cat` ≈ `[0.21, -0.53, 0.88, …]`  (100 real numbers)
- `dog` ≈ `[0.19, -0.49, 0.91, …]`

**Key payoff** 關鍵好處: semantically similar words get similar vectors (high cosine similarity).
語義相近的詞,向量也相近。*cat* sits near *dog*, far from *table*. This is the math behind Firth's:

> *"You shall know a word by the company it keeps."*

| | One-hot | Embedding |
|---|---|---|
| Dimensions 維度 | vocab size (~10⁴) | small (50–300) |
| Values 取值 | 0 / 1 | continuous reals 連續實數 |
| Density 密度 | sparse 稀疏 | dense 稠密 |
| Encodes similarity? 表達相似? | ❌ no | ✅ yes |

---

## 4. Why is it called "embedding"? 為什麼叫「嵌入」 ⭐ (key intuition)

In mathematics, an **embedding** means placing the *structure* of one space **inside** another.
數學上「嵌入」= 把一個空間的結構裝進另一個空間。

Here: we take the giant **"one dimension per word"** space and **compress it into** a much
smaller space **while preserving the semantic relationships** between words.
這裡就是把「一個詞一維」的巨大空間,壓縮裝進一個小得多、但**保留了語義關係**的空間裡。

The word *embed* captures this: not just shrinking, but **fitting the meaningful structure into**
a lower-dimensional home. 不只是縮小,而是把「有意義的結構」裝進低維空間。

---

## 5. Terminology 術語

"**Word vectors**" and "**word embeddings**" are used interchangeably. 兩者通用。
"Embedding" emphasizes the *mapping from high-dim → meaningful low-dim space*.

---

## 6. How this threads through the course 本概念貫穿全課程

| Where 位置 | How embeddings appear 如何出現 |
|---|---|
| **a1** Count-based | Build co-occurrence matrix (high-dim), then **SVD** to *manually* reduce to low-dim. 手動降維。|
| **a1** GloVe | Load embeddings *learned automatically* from a training objective. 直接載入學好的。|
| **a2** word2vec | *Learn* embeddings from scratch via skip-gram + negative sampling. 從零訓練。|
| **a3–a4** | Word embeddings feed RNNs / seq2seq as the input layer. 作為輸入層。|
| **a5** BERT | **Contextual** embeddings — a word's vector depends on its sentence. 依賴上下文。|

**Mental model of the arc** 主線:
static hand-counted (a1) → static learned (a2) → contextual learned (a5).
靜態手算 → 靜態學習 → 上下文相關。

---

## 7. One-line summary 一句話總結

> **Word embedding = turning "high-dim, sparse, meaning-less" one-hot vectors into
> "low-dim, dense, similarity-aware" vectors.**
> 把「高維、稀疏、無語義」的獨熱向量,變成「低維、稠密、能表達詞義相似」的向量。
