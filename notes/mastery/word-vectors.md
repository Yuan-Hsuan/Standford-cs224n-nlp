# Mastery · Word Vectors 精熟:詞向量

> A study/quiz layer (separate from the reference notes in `concepts/`). 為了檢驗吸收,不是查閱用。
> Structure per concept: **① mental models ② core architectures ③ questions to truly understand**.

---

## ① Most important mental models 最重要的思維模型

1. **Distributional hypothesis 分佈假說** — "You shall know a word by the company it keeps." A word's
   meaning = the distribution of contexts it appears in. 詞義 = 它出現的上下文分佈。
2. **Embedding = structure-preserving compression 嵌入=保結構的壓縮** — map sparse one-hot (10⁴ dims)
   → dense low-dim (50–300), keeping semantic similarity. 高維稀疏 → 低維稠密,保住相似度。
3. **Meaning is geometry 語義即幾何** — similarity = cosine / dot product; nearby vectors ⇒ similar words.
4. **Two philosophies + a hybrid 兩種哲學 + 混合** — count-based (global stats, SVD) vs prediction-based
   (local windows, word2vec) vs hybrid (GloVe combines both).
5. **Static vs contextual 靜態 vs 情境** — one vector per word (a frequency-weighted average of senses)
   vs one vector per context (BERT). Static can't disambiguate senses. 靜態抓不到多義。

## ② Core architectures 核心架構

1. **Co-occurrence + SVD** (count-based): build word×context counts, factor $A=USV^T$, word vector = rows of $U\cdot S$.
2. **word2vec skip-gram + negative sampling** (prediction): predict context from center; sigmoid, not softmax.
3. **GloVe** (hybrid): log-bilinear objective so dot product ≈ log co-occurrence; captures probability **ratios**.
4. **Contextual (ELMo/BERT)** (a5): vector depends on the sentence. 依語境。

## ③ Questions to truly understand 真正理解要回答的問題 (quiz bank 題庫)

**Foundations**
- Why can't one-hot vectors express similarity? (orthogonal → dot product 0)
- What does the distributional hypothesis mean, with a concrete example?

**Count-based / SVD**
- In $A=USV^T$, what do $U$, $S$, $V$ represent physically?
- Why is the word vector $U\cdot S$, not $U$ alone?
- Why is $\sigma_i=\sqrt{\lambda_i}$ (of $A^TA$)? Why is $AA^T$ symmetric?
- Why can a 2-D SVD plot come out mirrored/rotated?

**Prediction / word2vec**
- Why negative sampling instead of full softmax? What is the loss?

**Hybrid / GloVe**
- What is GloVe's "crucial insight" (co-occurrence probability ratios)?
- How does GloVe combine the strengths of count-based and prediction-based?

**Limits / polysemy**
- Why does one GloVe vector for a polysemous word still reveal multiple senses via neighbors?
- Why do many polysemous words fail (only one sense shows)? (dominant-sense frequency)
- Why might synonyms (large/massive/huge) NOT cluster? (frequency, 2-D compression, local context)

**Big picture**
- Count-based vs prediction-based: trade-offs? Where does GloVe sit?
- Static vs contextual embeddings — what problem does BERT solve that GloVe can't?
