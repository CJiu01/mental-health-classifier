"""
Ch.11 — Fine-tuned MentalBERT Classifier  (TP2 upgrade)

Upgrades Ch.4 in two ways:
  1. End-to-end fine-tuning (Ch.11): mental/mental-bert-base-uncased trained on
     dair-ai/emotion instead of a frozen encoder + LogisticRegression.
  2. Long-text support (Ch.10 bi-encoder concept): sentences are split and encoded
     individually, then mean-pooled before classification.

Training is done in notebooks/ch11_training.ipynb (Google Colab).
This module handles inference only, loading the saved fine-tuned weights.
"""

import os
import sys
import numpy as np
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    EMOTION_LABELS, VALID_RISK_LEVELS, RISK_THRESHOLDS,
    RISK_WEIGHT, CH11_BASE_MODEL, CH11_MAX_LENGTH, CH11_SAVED_DIR,
)

# Lazy NLTK import — only downloaded once
_NLTK_READY = False


def _ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    _NLTK_READY = True


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK. Falls back to period-split."""
    _ensure_nltk()
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.replace("!",".|").replace("?",".|").split(".") if s.strip()]
    return [s for s in sentences if s.strip()] or [text]


def compute_risk(emotion_scores: dict) -> tuple:
    """Return (risk_level, risk_score) from emotion probability dict."""
    risk_score = emotion_scores["sadness"] + emotion_scores["fear"]
    joy_score  = emotion_scores["joy"]     + emotion_scores["love"]

    if   risk_score > RISK_THRESHOLDS["crisis"]  : risk_level = "Crisis"
    elif risk_score > RISK_THRESHOLDS["negative"] : risk_level = "Negative"
    elif joy_score  > RISK_THRESHOLDS["positive"] : risk_level = "Positive"
    else                                           : risk_level = "Neutral"

    return risk_level, round(float(risk_score), 4)


class FinetunedClassifier:
    """
    Fine-tuned MentalBERT classifier with sentence chunking for long text.

    Usage
    -----
    clf = FinetunedClassifier()          # loads from CH11_SAVED_DIR
    out = clf.predict("I feel hopeless.")
    out = clf.predict("Line1. Line2. Line3. Line4. Line5.")  # long text
    """

    def __init__(self, model_dir: str = CH11_SAVED_DIR):
        self.model_dir = model_dir
        if torch.cuda.is_available():
            _dev = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _dev = "mps"
        else:
            _dev = "cpu"
        self.device = torch.device(_dev)
        self._loaded   = False
        self.tokenizer = None
        self.model     = None
        self._load()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.model_dir):
            print(f"[Ch.11] Model not found at '{self.model_dir}'.")
            print("[Ch.11] Run notebooks/ch11_training.ipynb on Colab first.")
            return

        print(f"[Ch.11] Loading fine-tuned MentalBERT from '{self.model_dir}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        ).to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"[Ch.11] Ready (device={self.device}).")

    # ── Sentence chunking (Ch.10 bi-encoder concept) ──────────────────────────

    def _encode_sentences(self, text: str) -> torch.Tensor:
        """
        Split text into sentences, encode each individually, return mean-pooled
        logits tensor of shape [num_labels].

        Single-sentence texts skip chunking entirely.
        """
        sentences = _split_sentences(text)

        all_logits = []
        for sent in sentences:
            inputs = self.tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=CH11_MAX_LENGTH,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits  # [1, num_labels]
            all_logits.append(logits.squeeze(0))      # [num_labels]

        # Mean-pool across sentences → single [num_labels] vector
        return torch.stack(all_logits).mean(dim=0)

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        assert self._loaded, "Model not loaded. Run ch11_training.ipynb first."

        mean_logits   = self._encode_sentences(text)
        probs         = F.softmax(mean_logits, dim=-1).cpu().numpy()
        emotion_scores = {
            name: round(float(p), 4)
            for name, p in zip(EMOTION_LABELS, probs)
        }
        primary_emotion          = max(emotion_scores, key=emotion_scores.get)
        risk_level, risk_score   = compute_risk(emotion_scores)

        return {
            "risk_level"      : risk_level,
            "risk_score"      : risk_score,
            "primary_emotion" : primary_emotion,
            "emotion_scores"  : emotion_scores,
            "reasoning"       : None,
            "recommendation"  : None,
            "empathy_response": None,
            "trend"           : [],
            "trend_direction" : None,
            "cluster_id"      : None,
            "cluster_keywords": None,
            "timestamp"       : datetime.now(timezone.utc).isoformat(),
            "mode"            : "ch11",
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    def predict_batch_fast(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Batched inference without sentence chunking — for evaluation speed."""
        assert self._loaded, "Model not loaded."
        from tqdm import tqdm

        all_results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="[Ch.11] Batches"):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=CH11_MAX_LENGTH,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits  # [B, num_labels]

            probs_batch = F.softmax(logits, dim=-1).cpu().numpy()
            for probs in probs_batch:
                emotion_scores  = {n: round(float(p), 4) for n, p in zip(EMOTION_LABELS, probs)}
                primary_emotion = max(emotion_scores, key=emotion_scores.get)
                risk_level, risk_score = compute_risk(emotion_scores)
                all_results.append({
                    "risk_level"      : risk_level,
                    "risk_score"      : risk_score,
                    "primary_emotion" : primary_emotion,
                    "emotion_scores"  : emotion_scores,
                })
        return all_results


# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clf = FinetunedClassifier()

    samples = [
        # Short (single sentence)
        "I feel so happy and grateful today!",
        "I am really worried and can't stop thinking about the worst.",
        "I want to disappear and stop being a burden to everyone.",
        # Long (multi-sentence — Ch.10 chunking path)
        (
            "Today was really hard. I couldn't get out of bed. "
            "Everything feels pointless. I don't see any way forward. "
            "I just want it all to stop."
        ),
    ]

    print("=== Ch.11 Fine-tuned MentalBERT ===")
    for s in samples:
        out = clf.predict(s)
        print(f"  [{out['risk_level']:8}]  score={out['risk_score']:.3f}"
              f"  '{s[:60]}'")
