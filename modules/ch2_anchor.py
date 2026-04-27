"""
Ch.2 — Anchor Embedding Classifier
Cosine similarity between input and pre-defined anchor sentence embeddings.
No training required.
"""

import os
import sys
import numpy as np
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ANCHOR_MODEL

EMOTION_ANCHORS = {
    "Positive": [
        "I feel so happy and grateful today.",
        "Everything is going well and I feel loved.",
        "I am excited, proud, and full of positive energy.",
    ],
    "Neutral": [
        "Today was an ordinary day, nothing special happened.",
        "I feel okay, not particularly good or bad.",
        "Things are just normal lately, going through the routine.",
    ],
    "Negative": [
        "I feel sad and a bit hopeless about the future.",
        "I am scared and constantly worried about what might happen.",
        "I feel anxious, tired, and emotionally drained all the time.",
    ],
    "Crisis": [
        "I want to disappear and stop existing forever.",
        "I feel completely worthless and see no reason to go on.",
        "I can't take it anymore and want to end everything.",
    ],
}


class AnchorClassifier:

    def __init__(self):
        print("[Ch.2] Loading anchor embedding model...")
        self._model = SentenceTransformer(ANCHOR_MODEL)
        self._anchors = {
            label: np.mean(self._model.encode(sentences), axis=0)
            for label, sentences in EMOTION_ANCHORS.items()
        }
        print("[Ch.2] Anchor vectors ready.")

    def predict(self, text: str) -> dict:
        vec    = self._model.encode(text).reshape(1, -1)
        scores = {
            label: float(cosine_similarity(vec, av.reshape(1, -1))[0][0])
            for label, av in self._anchors.items()
        }
        risk_level = max(scores, key=scores.get)

        return {
            "risk_level"     : risk_level,
            "risk_score"     : round(scores[risk_level], 4),
            "primary_emotion": None,
            "emotion_scores" : {k: round(v, 4) for k, v in scores.items()},
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


if __name__ == "__main__":
    clf = AnchorClassifier()
    samples = [
        "I feel so happy and grateful today!",
        "I am really worried and can't stop thinking about the worst.",
        "I want to disappear and stop being a burden to everyone.",
    ]
    for s in samples:
        out = clf.predict(s)
        print(f"  [{out['risk_level']:8}]  score={out['risk_score']:.3f}  {s[:55]}")
        print(f"            Anchor scores: {out['emotion_scores']}\n")
