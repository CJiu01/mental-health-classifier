"""
Main pipeline orchestrator.
Quick mode  : ch4 only (~0.1s)
Full mode   : ch4 → ch6 → ch7 (~3-5s on GPU)
Clustering  : ch5 optional (use_cluster=True)
"""

import os
import uuid
from datetime import datetime, timezone

from modules.ch4_classifier import MentalHealthClassifier
from modules.ch5_clustering  import EmotionClusterer
from modules.ch6_prompt      import PromptClassifier
from modules.ch7_memory      import EmotionalMemorySystem
from config import SAVED_MODELS_DIR


class MentalHealthPipeline:

    def __init__(self, mode: str = "quick", session_id: str = None,
                 load_classifier: bool = True):
        """
        Parameters
        ----------
        mode            : "quick" | "full"
        session_id      : reuse an existing session (None = new session)
        load_classifier : load saved ch4 model if available
        """
        assert mode in ("quick", "full"), "mode must be 'quick' or 'full'"
        self.mode       = mode
        self.session_id = session_id or str(uuid.uuid4())[:8]

        # Ch.4 — always initialised
        self.classifier = MentalHealthClassifier()
        if load_classifier and os.path.exists(
            os.path.join(SAVED_MODELS_DIR, "ch4_logreg.pkl")
        ):
            self.classifier.load()
        else:
            print("[Pipeline] ch4 model not found — call train() first.")

        # Ch.5 — optional, initialised on first use
        self.clusterer: EmotionClusterer = None

        # Full-mode components — lazy initialised
        self._prompter: PromptClassifier      = None
        self._memory  : EmotionalMemorySystem = None

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, train_texts: list, train_labels: list) -> None:
        self.classifier.train(train_texts, train_labels)
        self.classifier.save()

    def fit_clusterer(self, texts: list) -> None:
        """Fit ch5 clustering on provided texts using ch4 embeddings."""
        print("[Pipeline] Encoding for clustering...")
        embeddings = self.classifier.encoder.encode(
            texts, show_progress_bar=True, batch_size=64
        )
        self.clusterer = EmotionClusterer()
        self.clusterer.fit(texts, embeddings)

    # ── Inference ──────────────────────────────────────────────────────────────

    def run(self, text: str, use_cluster: bool = False) -> dict:
        text = self._validate(text)

        # Shared encoding (one pass for ch4 + ch5)
        vector = self.classifier.encode(text)

        # Ch.4 — core classification
        result = self.classifier.predict_from_vector(vector, text)

        # Ch.5 — optional clustering
        if use_cluster and self.clusterer and self.clusterer.is_fitted:
            cluster_info             = self.clusterer.transform(text)
            result["cluster_id"]     = cluster_info["cluster_id"]
            result["cluster_keywords"] = cluster_info["cluster_keywords"]

        # quick mode exit
        if self.mode == "quick":
            result["trend"] = self._get_trend()
            result["trend_direction"] = self._get_trend_direction()
            self._update_memory_log(text, result)
            return result

        # full mode — ch6 (independent) + ch7
        if self._prompter is None:
            self._prompter = PromptClassifier()

        prompt_out              = self._prompter.predict(text)
        result["reasoning"]     = prompt_out.get("reasoning")
        result["recommendation"]= prompt_out.get("recommendation")

        memory = self._get_memory()
        memory.set_llm(self._prompter._llm)   # reuse ch6 LLM, avoid double loading
        result["empathy_response"] = memory.generate_empathy(
            text, result["risk_level"]
        )
        self._update_memory_log(text, result)
        result["trend"]          = memory.get_trend()
        result["trend_direction"]= memory.get_trend_direction()
        result["mode"]           = "full"
        return result

    # ── Session helpers ────────────────────────────────────────────────────────

    def save_session(self) -> None:
        if self._memory:
            self._memory.save()

    def load_session(self) -> None:
        self._get_memory().load()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _validate(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("EMPTY_INPUT: text must not be empty.")
        if len(text) > 2000:
            raise ValueError("TOO_LONG: text exceeds 2000 characters.")
        return text.strip()

    def _get_memory(self) -> EmotionalMemorySystem:
        if self._memory is None:
            self._memory = EmotionalMemorySystem(session_id=self.session_id)
        return self._memory

    def _get_trend(self) -> list:
        if self._memory:
            return self._memory.get_trend()
        return []

    def _get_trend_direction(self) -> str:
        if self._memory:
            return self._memory.get_trend_direction()
        return "stable"

    def _update_memory_log(self, text: str, result: dict) -> None:
        mem = self._get_memory()
        mem.add_entry(
            text           = text,
            risk_level     = result["risk_level"],
            risk_score     = result["risk_score"],
            primary_emotion= result["primary_emotion"],
        )


# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipe = MentalHealthPipeline(mode="quick")

    samples = [
        "I feel so happy and grateful today!",
        "I am really worried and can't stop thinking about the worst.",
        "I want to disappear and stop being a burden to everyone.",
    ]
    print("=== Quick Mode ===")
    for s in samples:
        out = pipe.run(s)
        print(f"  [{out['risk_level']:8}]  score={out['risk_score']:.3f}"
              f"  trend={out['trend']}  '{s[:45]}'")
