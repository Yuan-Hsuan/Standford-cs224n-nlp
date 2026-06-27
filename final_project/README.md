# CS224n Final Project Sandbox

The CS224n final project allows you to apply what you have learned to a significant NLP problem. You can choose to implement the **Default Final Project** or a **Custom Project**.

## Final Project Options

### Option A: The Default Project (minBERT & Downstream Tasks)
This is the recommended path for self-study. It consists of:
1. **Implementing minBERT**:
   - Write the core components of the BERT model (Self-Attention, Transformer Layer, BERT Model).
   - Implement the optimizer (AdamW).
2. **Fine-Tuning on Downstream Tasks**:
   - Sentiment Classification (SST dataset).
   - Paraphrase Detection (QQP dataset).
   - Semantic Textual Similarity (STS dataset).
3. **Extensions**:
   - Implement advanced techniques to improve performance (e.g., Cosine Warmup, Multi-task pretraining, Contrastive learning, Adapter layers).

### Option B: Custom Project
Choose a unique research direction or task. Typical areas include:
- Question Answering (e.g., SQuAD 2.0).
- Text Generation / Summarization.
- Multi-modal applications (Vision + Text, Audio + Text).
- Parameter Efficient Fine-Tuning (PEFT) comparisons.

---

## Project Milestones

Use these goals to structure your project timeline:
- [ ] **Milestone 1**: Project Proposal - Define task, dataset, and baseline model.
- [ ] **Milestone 2**: Setup and Baseline - Get the default model/dataset working and establish baseline metrics.
- [ ] **Milestone 3**: Core Architecture - Implement the custom features, extensions, or modules.
- [ ] **Milestone 4**: Evaluation & Analysis - Run hyperparameter searches, produce error analysis, and write the final report.
