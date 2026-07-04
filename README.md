# Stanford CS224n (Spring/Winter 2024 Offerings)

This repository is designed for self-studying Stanford's **CS224n: Natural Language Processing with Deep Learning**. It hosts a structured timeline, lectures directory, assignment sandboxes, and virtual environment configs.

---

## 🎯 Learning Goals

1. **Master NLP Fundamentals & Math**: Understand how semantic meaning is represented (Word2Vec, GloVe), parse syntax structures (dependency parsing), and study information bottleneck mechanics in neural networks.
2. **Build Deep NLP Models from Scratch**: Implement custom versions of Word2Vec (NumPy), a Transition-based Dependency Parser (PyTorch), a Sequence-to-Sequence translation system with Attention (PyTorch), and a minBERT encoder model.
3. **Build Fluency in PyTorch**: Learn tensor manipulation, dimension tracking, custom neural network modules, loss functions, custom training loops, and optimization routines.
4. **Understand Modern Large Language Models (LLMs)**: Master the Transformer architecture, pretraining/fine-tuning dynamics, Reinforcement Learning from Human Feedback (RLHF), context extending techniques (RAG), and parameter-efficient tuning (LoRA).

---

## 🛠️ Environment Setup (venv & pip)

Follow these steps to initialize your isolated Python environment and install the required dependencies:

```bash
# 1. Create a virtual environment named 'venv'
python3 -m venv venv

# 2. Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate

# 3. Upgrade pip to the latest version
pip install --upgrade pip

# 4. Install all course dependencies
pip install -r requirements.txt
```

To verify the installation succeeded and PyTorch can access your GPU/M-series chip:
```bash
python -c "import torch; print('PyTorch Version:', torch.__version__); print('MPS Available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'N/A')"
```

---

## 📅 10-Week Study Schedule

Below is a weekly tracking template structured for a 10-week course layout.

| Week | Lecture Topics | Materials & Slides | Deliverable | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Week 1** | **Lectures 01 & 02**: Intro to NLP & Word Vectors, Word2Vec, GloVe, Word Embeddings | [Archive Web](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) | Start **Assignment 1** (Word Vectors) | `[~]` |
| **Week 2** | **Lectures 03 & 04**: Subword Models & Neural Networks, Word Window Classification, Backpropagation | [Archive Web](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) | Submit **Assignment 1** | `[ ]` |
| **Week 3** | **Lectures 05 & 06**: Backprop Math, Computations, Dependency Parsing | [Archive Web](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) | Start **Assignment 2** (Word2Vec Implementation) | `[ ]` |
| **Week 4** | **Lectures 07 & 08**: Recurrent Neural Networks (RNNs) & LSTMs/GRUs | [Archive Web](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) | Submit **Assignment 2** & Start **Assignment 3** (Dependency Parser) | `[ ]` |
| **Week 5** | **Lectures 09 & 10**: Neural Machine Translation (Seq2Seq), Attention Mechanisms | [Archive Web](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) | Submit **Assignment 3** | `[ ]` |
| **Week 6** | **Lectures 11 & 12**: Self-Attention & Transformers, Pretraining, BERT | [Archive Web](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) | Start **Assignment 4** (NMT with Attention) | `[ ]` |
| **Week 7** | **Lectures 13 & 14**: Prompting, RAG, Context Windows, LLMs | [Archive Web](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) | Submit **Assignment 4** & Start **Assignment 5** (minBERT) | `[ ]` |
| **Week 8** | **Lectures 15 & 16**: RLHF, DPO, Reasoning, Agents, Tool Use | [Archive Web](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) | Begin working on the **Final Project** baseline | `[ ]` |
| **Week 9** | **Lectures 17 & 18**: Model Efficiency (Quantization, LoRA), Interpretability & Probe analysis | [Archive Web](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) | Submit **Assignment 5** | `[ ]` |
| **Week 10** | **Lectures 19 & 20**: Safety/Bias in NLP, Multimodal VLMs, Future of NLP | [Archive Web](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) | Final Project evaluations and wrap-up | `[ ]` |

---

## 📌 Current Progress (updated 2026-07-03)

**Assignment 1 — Exploring Word Vectors**

- **Part 1 · Count-Based Vectors (10 pts) — ✅ done**
  - Q1.1 `distinct_words` · Q1.2 `compute_co_occurrence_matrix` · Q1.3 `reduce_to_k_dim`
  - Q1.4 `plot_embeddings` · Q1.5 co-occurrence plot analysis (written)
- **Part 2 · Prediction-Based Vectors — GloVe (15 pts) — 🔄 in progress**
  - Environment fixed: macOS python.org SSL certs (ran `Install Certificates.command`) so `gensim`
    can download `glove-wiki-gigaword-200`
  - Next: Q2.1 GloVe plot analysis → synonyms/antonyms, analogies, bias

**Study notes** (`notes/`) — maintained by my TA (Claude), English + 繁中

- `notes/concepts/00-concept-map.md` — index + how the concept notes relate
- `notes/concepts/01-word-embeddings.md` — one-hot → dense embeddings; why "embedding"
- `notes/concepts/02-count-based-svd.md` — co-occurrence + SVD (eigenvalues, Truncated SVD)
- `notes/concepts/03-word2vec-and-glove.md` — word2vec (neural) & GloVe (hybrid)
- `notes/mistakes-log.md` — dated log of misconceptions + corrections

---

## 🔗 Key Learning Resources

*   **Syllabus & Lecture Archives**: [Stanford CS224N Portal](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/)
*   **Video Lectures**: [CS224N 2024 YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)
*   **Recommended Book**: *Speech and Language Processing (3rd ed. draft)* by Dan Jurafsky and James H. Martin