"""
Ch.5 — Unsupervised Emotional Pattern Discovery
UMAP dimensionality reduction + HDBSCAN clustering + BERTopic topic modeling.
Reuses embeddings from ch4 (shared encoder).
"""

import os
import sys
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EMBEDDING_MODEL

RISK_QUERY = "depression suicide hopeless worthless"


class EmotionClusterer:

    def __init__(self):
        self._umap_model = UMAP(
            n_components=5, min_dist=0.0,
            metric="cosine", random_state=42
        )
        self._hdbscan_model = HDBSCAN(
            min_cluster_size=100,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        self._topic_model  = None
        self._risk_topic_ids: list = []
        self.is_fitted     = False

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, texts: list, embeddings: np.ndarray) -> None:
        """Fit BERTopic using pre-computed embeddings (from ch4 encoder)."""
        print("[Ch.5] Fitting BERTopic (UMAP + HDBSCAN)...")
        self._topic_model = BERTopic(
            embedding_model   = EMBEDDING_MODEL,
            umap_model        = self._umap_model,
            hdbscan_model     = self._hdbscan_model,
            representation_model = KeyBERTInspired(),
            verbose           = True,
        )
        self._topic_model.fit(texts, embeddings)
        self.is_fitted = True

        self._risk_topic_ids = self.find_risk_clusters(RISK_QUERY)
        print(f"[Ch.5] Fitting complete. "
              f"Clusters={len(set(self._topic_model.topics_))}, "
              f"Risk clusters={self._risk_topic_ids}")

    # ── Inference ──────────────────────────────────────────────────────────────

    def transform(self, text: str) -> dict:
        assert self.is_fitted, "Call fit() first."
        topic_ids, probs = self._topic_model.transform([text])
        topic_id         = int(topic_ids[0])
        keywords         = self._get_keywords(topic_id)

        return {
            "cluster_id"      : topic_id,
            "cluster_keywords": keywords,
            "is_risk_cluster" : topic_id in self._risk_topic_ids,
        }

    def transform_from_vector(self, vector: np.ndarray) -> dict:
        """Accept pre-computed embedding to avoid double encoding."""
        return self.transform_vector(vector)

    # ── Risk Cluster Detection ─────────────────────────────────────────────────

    def find_risk_clusters(self, query: str = RISK_QUERY, top_n: int = 5) -> list:
        assert self.is_fitted, "Call fit() first."
        topic_ids, scores = self._topic_model.find_topics(query, top_n=top_n)
        return [int(t) for t in topic_ids]

    # ── Visualization helpers ──────────────────────────────────────────────────

    def get_topic_info(self):
        assert self.is_fitted
        return self._topic_model.get_topic_info()

    def _get_keywords(self, topic_id: int) -> list:
        if topic_id == -1:
            return []
        topic = self._topic_model.get_topic(topic_id)
        return [word for word, _ in topic[:5]] if topic else []


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.loader import load_emotion_dataset
    from modules.ch4_classifier import MentalHealthClassifier

    data = load_emotion_dataset()
    texts = data["train"]["texts"][:5000]   # subset for demo speed

    print("[Ch.5] Encoding with ch4 shared encoder...")
    clf        = MentalHealthClassifier()
    clf.load()
    embeddings = clf.encoder.encode(texts, show_progress_bar=True)

    clusterer = EmotionClusterer()
    clusterer.fit(texts, embeddings)

    sample = "I feel so hopeless and don't see any reason to continue."
    result = clusterer.transform(sample)
    print(f"\nSample : '{sample}'")
    print(f"Cluster: {result['cluster_id']}  "
          f"Keywords: {result['cluster_keywords']}  "
          f"IsRisk: {result['is_risk_cluster']}")
