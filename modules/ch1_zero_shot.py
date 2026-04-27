"""
Ch.1 — Zero-Shot Baseline
Phi-3-mini-4k-instruct via HuggingFace pipeline, no training required.
"""

import gc
import os
import sys
from datetime import datetime, timezone

import torch
from transformers import pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LLM_HF_MODEL, VALID_RISK_LEVELS, VALID_EMOTIONS

ZERO_SHOT_PROMPT = (
    "You are a mental health text analysis assistant.\n"
    "Classify the following text into exactly one category: "
    "Positive / Neutral / Negative / Crisis\n\n"
    'Text: "{text}"\n\n'
    "Respond with only: <category> | <one-sentence reason>"
)


class ZeroShotClassifier:

    def __init__(self):
        # Ch.1 runs on CPU: it is a baseline-only model.
        # GPU is reserved for Ch.4 (SentenceTransformer) which remains loaded.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[Ch.1] Loading Phi-3 pipeline on CPU (baseline mode)...")
        self._pipe = pipeline(
            "text-generation",
            model=LLM_HF_MODEL,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=False,
            return_full_text=False,
            max_new_tokens=60,
            do_sample=False,
        )
        print("[Ch.1] Ready.")

    def predict(self, text: str) -> dict:
        prompt  = ZERO_SHOT_PROMPT.format(text=text)
        raw     = self._pipe([{"role": "user", "content": prompt}])
        output  = raw[0]["generated_text"].strip()
        risk_level, reasoning = self._parse(output)

        return {
            "risk_level"     : risk_level,
            "risk_score"     : None,
            "primary_emotion": None,
            "emotion_scores" : None,
            "reasoning"      : reasoning,
            "recommendation" : None,
            "empathy_response"  : None,
            "trend"             : [],
            "trend_direction"   : None,
            "cluster_id"        : None,
            "cluster_keywords"  : None,
            "timestamp"         : datetime.now(timezone.utc).isoformat(),
            "mode"              : "quick",
        }

    def _parse(self, output: str) -> tuple:
        """Parse '<category> | <reason>' format."""
        parts = output.split("|", 1)
        raw_level = parts[0].strip()
        reasoning = parts[1].strip() if len(parts) > 1 else output

        # Match to valid label (case-insensitive)
        for level in VALID_RISK_LEVELS:
            if level.lower() in raw_level.lower():
                return level, reasoning
        return "Neutral", reasoning   # fallback


if __name__ == "__main__":
    clf = ZeroShotClassifier()
    samples = [
        "I feel so happy and grateful today!",
        "I am really worried and can't stop thinking about the worst.",
        "I want to disappear and stop being a burden to everyone.",
    ]
    for s in samples:
        out = clf.predict(s)
        print(f"  [{out['risk_level']:8}]  {s[:60]}")
        print(f"            Reason: {out['reasoning']}\n")
