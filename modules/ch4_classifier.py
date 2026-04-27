"""
Ch.4 — Core Classifier
SentenceTransformer embeddings + LogisticRegression → 6-class emotion → 4-tier risk
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    EMBEDDING_MODEL, EMOTION_LABELS, RISK_THRESHOLDS,
    LOGREG_PARAMS, SAVED_MODELS_DIR
)


def compute_risk(emotion_scores: dict) -> tuple:
    """Return (risk_level, risk_score) from emotion probability dict."""
    risk_score = emotion_scores["sadness"] + emotion_scores["fear"]
    joy_score  = emotion_scores["joy"]     + emotion_scores["love"]

    if   risk_score > RISK_THRESHOLDS["crisis"]  : risk_level = "Crisis"
    elif risk_score > RISK_THRESHOLDS["negative"] : risk_level = "Negative"
    elif joy_score  > RISK_THRESHOLDS["positive"] : risk_level = "Positive"
    else                                          : risk_level = "Neutral"

    return risk_level, round(float(risk_score), 4)


class MentalHealthClassifier:

    def __init__(self):
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.clf     = LogisticRegression(**LOGREG_PARAMS)
        self._fitted = False

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, train_texts: list, train_labels: list,
              show_progress: bool = True) -> None:
        print("[Ch.4] Encoding training texts...")
        embeddings = self.encoder.encode(
            train_texts, show_progress_bar=show_progress, batch_size=64
        )
        print("[Ch.4] Fitting LogisticRegression...")
        self.clf.fit(embeddings, train_labels)
        self._fitted = True
        print("[Ch.4] Training complete.")

    # ── Inference ──────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        return self.encoder.encode(text)

    def predict(self, text: str) -> dict:
        return self.predict_from_vector(self.encode(text), text)

    def predict_from_vector(self, vector: np.ndarray, text: str = "") -> dict:
        """Shared encoder path: accepts pre-computed embedding."""
        assert self._fitted, "Model not trained. Call train() or load() first."

        proba         = self.clf.predict_proba(vector.reshape(1, -1))[0]
        emotion_scores = {name: round(float(p), 4)
                          for name, p in zip(EMOTION_LABELS, proba)}
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        risk_level, risk_score = compute_risk(emotion_scores)

        return {
            "risk_level"     : risk_level,
            "risk_score"     : risk_score,
            "primary_emotion": primary_emotion,
            "emotion_scores" : emotion_scores,
            "reasoning"      : None,
            "recommendation" : None,
            "empathy_response"  : None,
            "trend"             : [],
            "trend_direction"   : None,
            "cluster_id"        : None,
            "cluster_keywords"  : None,
            "timestamp"         : datetime.now(timezone.utc).isoformat(),
            "mode"              : "quick",
        }

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, test_texts: list, test_labels: list,
                 show_progress: bool = True) -> dict:
        print("[Ch.4] Encoding test texts...")
        embeddings  = self.encoder.encode(
            test_texts, show_progress_bar=show_progress, batch_size=64
        )
        predictions = self.clf.predict(embeddings)
        report      = classification_report(
            test_labels, predictions,
            target_names=EMOTION_LABELS, output_dict=True
        )
        print(classification_report(test_labels, predictions,
                                    target_names=EMOTION_LABELS))
        return report

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: str = SAVED_MODELS_DIR) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.clf, os.path.join(directory, "ch4_logreg.pkl"))
        with open(os.path.join(directory, "ch4_encoder_config.json"), "w") as f:
            json.dump({"embedding_model": EMBEDDING_MODEL}, f)
        print(f"[Ch.4] Model saved to {directory}/")

    def load(self, directory: str = SAVED_MODELS_DIR) -> None:
        self.clf     = joblib.load(os.path.join(directory, "ch4_logreg.pkl"))
        self._fitted = True
        print(f"[Ch.4] Model loaded from {directory}/")


# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.loader import load_emotion_dataset

    data = load_emotion_dataset()
    clf  = MentalHealthClassifier()
    clf.train(data["train"]["texts"], data["train"]["labels"])
    clf.save()

    print("\n── Evaluation on test set ──")
    clf.evaluate(data["test"]["texts"], data["test"]["labels"])

    print("\n── Sample predictions ──")
    samples = [
        "I feel so happy and grateful today!",
        "I am really worried and can't stop thinking about the worst.",
        "I want to disappear and stop being a burden to everyone.",
    ]
    for s in samples:
        out = clf.predict(s)
        print(f"  [{out['risk_level']:8}]  {s[:60]}")
