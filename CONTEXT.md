# Mental Health Text Classifier

NLP course term project that detects emotional risk signals from short-to-medium user-written text and maps them to a four-tier risk scale.

## Language

**Risk Level**:
One of four ordinal categories — Positive, Neutral, Negative, Crisis — derived from emotion probabilities. Not a clinical diagnosis.
_Avoid_: severity, label, class

**Risk Score**:
A scalar in [0, 2] computed as P(sadness) + P(fear). Drives the Risk Level mapping.
_Avoid_: danger score, threat score

**Emotion Scores**:
A probability distribution over six emotions (sadness, joy, love, anger, fear, surprise) output by the classifier.
_Avoid_: emotion probabilities, softmax output

**Core Classifier**:
The primary model that maps text → Emotion Scores → Risk Level. In TP1 this is Ch.4 (frozen encoder + LogReg). In TP2 this is Ch.11 (fine-tuned MentalBERT).
_Avoid_: main model, base model

**Frozen Encoder**:
A SentenceTransformer used as a fixed feature extractor with no gradient updates during training (TP1 / Ch.4 approach).
_Avoid_: pre-trained encoder, static encoder

**Fine-tuned Encoder**:
A transformer whose weights are updated end-to-end during supervised training on the task dataset (TP2 / Ch.11 approach).
_Avoid_: trained model, updated encoder

**Sentence Chunking**:
Splitting multi-sentence input into individual sentences, encoding each separately, then mean-pooling the resulting vectors into one representation. Enables long-text input beyond single-sentence training distribution.
_Avoid_: text splitting, paragraph chunking, windowing

**Trend Direction**:
A three-value signal (improving / stable / deteriorating) computed from the half-split delta of Risk Weights across the seven-entry memory window.
_Avoid_: trend, mood trend, direction

**Session**:
A per-user context containing the sliding memory window and conversation buffer. Persisted to JSON in `sessions/`.
_Avoid_: user state, conversation history

## Module Map

| Module | Chapter | Role |
|---|---|---|
| `ch1_zero_shot.py` | Ch.1 | Zero-shot LLM baseline (no training) |
| `ch2_anchor.py` | Ch.2 | Anchor cosine similarity baseline (no training) |
| `ch4_classifier.py` | Ch.4 | **TP1 Core Classifier** — Frozen Encoder + LogReg |
| `ch5_clustering.py` | Ch.5 | UMAP + HDBSCAN + BERTopic topic clustering |
| `ch6_prompt.py` | Ch.6 | Prompt-engineered LLM reasoning + recommendation |
| `ch7_memory.py` | Ch.7 | Sliding-window memory + empathy chain |
| `ch11_finetuned.py` | Ch.11 | **TP2 Core Classifier** — Fine-tuned MentalBERT + Sentence Chunking |
