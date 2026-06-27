# CS224n Assignments (2024 Term)

This directory hosts the programming assignments for the Stanford CS224n (Natural Language Processing with Deep Learning) course. Each folder contains Python files and Jupyter notebooks implementing core NLP algorithms.

## Assignment Overview

1. **[a1_word_vectors](file:///Users/yuan/github/Standford-cs224n-nlp/assignments/a1_word_vectors/)**: **Exploration of Word Vectors**
   - **Concepts**: Co-occurrence matrices, Singular Value Decomposition (SVD), cosine similarity, Word2Vec embedding visualization.
   - **Key files**: `exploring_word_vectors.ipynb`.

2. **[a2_word2vec](file:///Users/yuan/github/Standford-cs224n-nlp/assignments/a2_word2vec/)**: **Implementing Word2Vec**
   - **Concepts**: Skip-gram, Negative Sampling, mathematical gradients derivation, NumPy implementation of training loops.
   - **Key files**: `word2vec.py`, `sgd.py`, `run.py`.

3. **[a3_dependency_parser](file:///Users/yuan/github/Standford-cs224n-nlp/assignments/a3_dependency_parser/)**: **Neural Transition-Based Dependency Parsing**
   - **Concepts**: Dependency syntax tree, shift-reduce transition parsing, neural network classifier in PyTorch.
   - **Key files**: `parser_transitions.py`, `parser_model.py`, `run.py`.

4. **[a4_nmt_attention](file:///Users/yuan/github/Standford-cs224n-nlp/assignments/a4_nmt_attention/)**: **Neural Machine Translation with Attention**
   - **Concepts**: Sequence-to-Sequence (Seq2Seq) models, Bidirectional LSTMs, Luong Attention, BLEU score evaluation.
   - **Key files**: `nmt_model.py`, `model_embeddings.py`, `run.sh`.

5. **[a5_pretrain_bert](file:///Users/yuan/github/Standford-cs224n-nlp/assignments/a5_pretrain_bert/)**: **BERT and Pretraining/Fine-Tuning**
   - **Concepts**: Multi-head self-attention, Transformer encoder, pretraining masking tasks, fine-tuning BERT on downstream applications (sentiment classification, NLI/QA).
   - **Key files**: `bert.py`, `classifier.py`, `optimizer.py`.

---

## Assignment Setup

You can download the starter code packages automatically by running the setup script from the root directory:

```bash
bash scripts/setup_assignments.sh
```

This script retrieves official starter zip packages from Stanford's archives, extracts them into their respective assignment folders, and cleans up the zip files automatically.
