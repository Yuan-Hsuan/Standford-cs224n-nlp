# 03 · word2vec & GloVe 預測式與混合詞向量

> **Source**: CS224n Lecture 2 (wordvecs2), p.10–23. [GloVe: Pennington, Socher, Manning, EMNLP 2014]
## Part A — word2vec (prediction-based, neural) 預測式

### A1. The idea
Slide a window over the corpus. For each **center word**, predict its **context words**
(skip-gram). Learn the vectors that make good predictions.
滑動窗口,用中心詞預測上下文詞(skip-gram),學出好向量。

- Each word has **two vectors**: $v_w$ as center, $u_w$ as context (easier optimization; average at the end).
  每個詞有兩個向量:當中心詞 $v_w$、當上下文 $u_w$。
- Variant 變體: **CBOW** predicts the center word from context (the reverse). skip-gram 的反向是 CBOW。

### A2. The naive softmax is too expensive 樸素 softmax 太貴
$$P(o\mid c)=\frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}$$
The denominator sums over the **whole vocabulary** $V$ — huge. 分母要對整個詞典加總,太慢。

### A3. Negative sampling (what word2vec actually uses) 負採樣
Instead of softmax, turn it into **binary logistic regressions**: push the true (center, context) pair
up, push $K$ random "noise" pairs down. 把它變成二元分類:真的配對推高,K 個隨機配對壓低。

$$J_{\text{neg}}=-\log\sigma(u_o^\top v_c)-\sum_{k=1}^{K}\log\sigma(-u_k^\top v_c)$$

- $\sigma$ = sigmoid (not softmax). 用 sigmoid,不是 softmax。
- Negative samples drawn from $P(w)=U(w)^{3/4}/Z$ — the unigram distribution raised to 3/4, which lets
  rarer words be sampled a bit more. 3/4 次方讓罕見詞多被抽一點。
- Gradients are **sparse**: each window only touches $2m+1$ real words + $Km$ negatives. 梯度稀疏。

---

## Part B — GloVe (hybrid: count + prediction) 混合

### B1. The crucial insight — ratios of co-occurrence probabilities
Meaning is encoded not in raw counts, but in the **ratio** $P(x\mid \text{ice})/P(x\mid \text{steam})$:

| x | ratio $P(x\mid ice)/P(x\mid steam)$ | meaning 語義 |
|---|---|---|
| solid 固體 | **large 大** | relates to ice, not steam |
| gas 氣體 | **small 小** | relates to steam, not ice |
| water 水 | ≈ 1 | related to both 都有關 |
| fashion 時尚 | ≈ 1 | related to neither 都無關 |

The "large / small / ≈1" pattern captures the **solid–gas** meaning axis. 這個比值抓到「固態 vs 氣態」的語義維度。

### B2. Log-bilinear model 對數雙線性模型
GloVe designs vectors so their **dot product ≈ log co-occurrence count**:
$$w_i^\top w_j + b_i + b_j \approx \log X_{ij}$$
Then **vector differences capture the ratios** → meaning becomes linear directions (enables analogies
`king − man + woman ≈ queen`). 於是向量相減對應比值,語義變成線性方向。

### B2.1 From ratio to log-count — the derivation 從比值到 log 次數(推導)

GloVe does **not** optimize the ratio directly. It uses an algebra trick to turn "ratio of
probabilities" into "dot product of vectors". GloVe 不直接優化比值,而是把比值轉成點積。

1. Want a function $F$ of the vectors to equal the ratio:
   $$F(w_i, w_j, w_k) = \frac{P(k\mid i)}{P(k\mid j)}$$
2. Compare $i,j$ with a **vector difference**, and reduce to a scalar with a **dot product**:
   $$F\big((w_i-w_j)^\top w_k\big) = \frac{P(k\mid i)}{P(k\mid j)}$$
3. Pick $F=\exp$, because it turns **subtraction into division** 把減法變除法:
   $$\exp\big((w_i-w_j)^\top w_k\big)=\frac{\exp(w_i^\top w_k)}{\exp(w_j^\top w_k)}=\frac{P(k\mid i)}{P(k\mid j)}$$
4. So it suffices that $\exp(w_i^\top w_k)=P(k\mid i)$, i.e. take $\log$:
   $$w_i^\top w_k=\log P(k\mid i)=\log X_{ik}-\log X_i$$
5. $\log X_i$ is independent of $k$ → absorb into a bias $b_i$ (add $b_k$ for symmetry):
   $$w_i^\top w_k+b_i+b_k=\log X_{ik}$$

**Punchline 重點**: the ratio never appears in the final objective — it fits dot products to
`log X`. But because of the derivation, fitting log-counts **automatically** makes
$\log\frac{P(k\mid i)}{P(k\mid j)}=(w_i-w_j)^\top w_k$, so the **ratio is encoded in the vector
difference**. That is why analogies work. 比值被藏進「向量差」,所以能做類比。

### B3. The loss
$$J=\sum_{i,j} f(X_{ij})\,\big(w_i^\top w_j + b_i + b_j - \log X_{ij}\big)^2$$
- $X_{ij}$ = global co-occurrence count of word $j$ in word $i$'s context. 全域共現次數。
- $f(X_{ij})$ = weighting that **caps very frequent pairs** (so "the", "of" don't dominate). 抑制高頻配對。
- **Fast training, scalable to huge corpora** — trains on the nonzero entries of $X$. 只算非零項,快、可擴展。

---

## Part C — word2vec vs GloVe 對照

| | word2vec (skip-gram) | GloVe |
|---|---|---|
| Base | prediction / neural | hybrid |
| Data | local windows, one at a time 局部窗口 | global co-occurrence matrix 全域矩陣 |
| Learns by | predicting context (neg. sampling) 預測 | fitting dot product to log-counts 擬合對數次數 |
| Loss | $-\log\sigma(u_o^\top v_c)-\sum_k\log\sigma(-u_k^\top v_c)$ | $\sum f(X_{ij})(w_i^\top w_j+b_i+b_j-\log X_{ij})^2$ |
| Uses global stats? | ✗ (iterates corpus) | ✓ |
| Speed | scales with corpus size | fast, scales with nonzeros |
| Performance | comparable; GloVe slightly better on some tasks (e.g. NER, p.29) 差不多,GloVe 某些任務略勝 |

---

## One-line summary 一句話總結

> **word2vec** learns embeddings by *predicting* context with a neural objective (skip-gram + negative
> sampling). **GloVe** starts from *global co-occurrence counts* and *learns* vectors whose dot products
> match log-counts, so it combines count-based statistics with predictive learning.
> word2vec 用神經預測學向量;GloVe 從全域共現統計出發、用學習擬合對數次數,結合了計數與預測兩者的優點。