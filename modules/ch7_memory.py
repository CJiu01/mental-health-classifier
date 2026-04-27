"""
Ch.7 — Conversational Memory & Longitudinal Tracking
Layer 1: Custom EmotionLog (trend tracking, 7-day window)
Layer 2: Sliding-window conversation buffer + LlamaCpp empathy chain
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    LLM_REPO_ID, LLM_FILENAME, MEMORY_WINDOW_K,
    RISK_WEIGHT, SESSIONS_DIR
)

EMPATHY_TEMPLATE = """\
You are a compassionate mental health support assistant.

Conversation history:
{chat_history}

User's recent emotional trend: {trend_summary}
Current risk level assessed: {risk_level}

User's message: "{user_text}"

Provide a brief (2-3 sentences) empathetic response that:
- Acknowledges the user's current emotional state warmly
- Is calibrated to the {risk_level} risk level
- For "Crisis": always include a professional help encouragement

Response:"""

TONE_GUIDE = {
    "Positive": "Reinforce positive momentum warmly.",
    "Neutral" : "Show gentle curiosity and interest.",
    "Negative": "Offer warm comfort and active listening.",
    "Crisis"  : "Immediately encourage professional help — you are not alone.",
}


class _WindowBuffer:
    """Sliding-window conversation buffer (replaces ConversationBufferWindowMemory)."""

    def __init__(self, k: int = 7):
        self.k = k
        self._messages: list = []

    def save_context(self, user_msg: str, ai_msg: str) -> None:
        self._messages.append({"role": "user",      "content": user_msg})
        self._messages.append({"role": "assistant", "content": ai_msg})
        if len(self._messages) > self.k * 2:
            self._messages = self._messages[-(self.k * 2):]

    def format_history(self) -> str:
        if not self._messages:
            return ""
        lines = []
        for msg in self._messages:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)


class EmotionalMemorySystem:

    def __init__(self, session_id: str, k: int = MEMORY_WINDOW_K):
        self.session_id = session_id
        self.k          = k

        # Layer 1: custom emotion log
        self._log: list           = []
        self._total_entries: int  = 0
        self._archive_summary: str = ""

        # Layer 2: sliding window buffer + LLM (lazy)
        self._window_buffer = _WindowBuffer(k=k)
        self._llm           = None
        self._llm_ready     = False

    # ── LLM lazy loader ────────────────────────────────────────────────────────

    def set_llm(self, llm) -> None:
        """Accept a pre-loaded Llama instance (e.g. from ch6) to avoid double loading."""
        self._llm       = llm
        self._llm_ready = True

    def _ensure_llm(self) -> None:
        if self._llm_ready:
            return
        print("[Ch.7] Loading LlamaCpp for empathy chain...")
        from llama_cpp import Llama
        self._llm = Llama.from_pretrained(
            repo_id      = LLM_REPO_ID,
            filename     = LLM_FILENAME,
            n_gpu_layers = -1,
            n_ctx        = 2048,
            verbose      = False,
        )
        self._llm_ready = True
        print("[Ch.7] LLM ready.")

    # ── Layer 1: EmotionLog ────────────────────────────────────────────────────

    def add_entry(self, text: str, risk_level: str,
                  risk_score: float = 0.0, primary_emotion: str = "") -> None:
        self._total_entries += 1
        entry = {
            "day_index"      : self._total_entries,
            "text"           : text,
            "risk_level"     : risk_level,
            "risk_score"     : risk_score,
            "primary_emotion": primary_emotion,
            "timestamp"      : datetime.now(timezone.utc).isoformat(),
        }
        self._log.append(entry)

        if len(self._log) > self.k:
            overflow = self._log.pop(0)
            self._archive_to_summary(overflow)

    def get_trend(self) -> list:
        return [e["risk_level"] for e in self._log]

    def get_trend_direction(self) -> str:
        trend = self.get_trend()
        if len(trend) < 3:
            return "stable"
        scores      = [RISK_WEIGHT[r] for r in trend]
        half        = len(scores) // 2
        first_half  = np.mean(scores[:half])
        second_half = np.mean(scores[half:])
        delta       = second_half - first_half
        if   delta >  0.5: return "deteriorating"
        elif delta < -0.5: return "improving"
        else:              return "stable"

    def _archive_to_summary(self, entry: dict) -> None:
        msg = (f"Day {entry['day_index']}: risk={entry['risk_level']}, "
               f"emotion={entry['primary_emotion']}, text='{entry['text'][:60]}'")
        self._archive_summary = (
            self._archive_summary + " | " + msg if self._archive_summary else msg
        )

    def _format_trend_summary(self) -> str:
        if not self._log:
            return "No entries yet."
        return " → ".join(f"Day {e['day_index']}: {e['risk_level']}" for e in self._log)

    # ── Layer 2: Empathy Response ──────────────────────────────────────────────

    def generate_empathy(self, text: str, risk_level: str) -> str:
        self._ensure_llm()
        prompt = EMPATHY_TEMPLATE.format(
            chat_history  = self._window_buffer.format_history(),
            trend_summary = self._format_trend_summary(),
            risk_level    = risk_level,
            user_text     = text,
        )
        output   = self._llm(prompt, max_tokens=150, temperature=0.7, stop=["\n\n"])
        response = output["choices"][0]["text"].strip()
        self._window_buffer.save_context(text, response)
        return response

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: str = SESSIONS_DIR) -> None:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.session_id}.json")
        payload = {
            "session_id"       : self.session_id,
            "created_at"       : datetime.now(timezone.utc).isoformat(),
            "total_entries"    : self._total_entries,
            "window_entries"   : self._log,
            "long_term_summary": self._archive_summary,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, directory: str = SESSIONS_DIR) -> None:
        path = os.path.join(directory, f"{self.session_id}.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            payload = json.load(f)
        self._log             = payload.get("window_entries", [])
        self._total_entries   = payload.get("total_entries", len(self._log))
        self._archive_summary = payload.get("long_term_summary", "")


# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Scenario A: Deteriorating trend ===")
    mem = EmotionalMemorySystem(session_id="demo_a")
    for text, risk in [
        ("I had a great day today!", "Positive"),
        ("Feeling pretty good, got things done.", "Positive"),
        ("Normal day, nothing special.", "Neutral"),
        ("Feeling a bit low today.", "Neutral"),
        ("Really tired and anxious lately.", "Negative"),
        ("Can't sleep, everything feels hopeless.", "Negative"),
        ("I just want to disappear.", "Crisis"),
    ]:
        mem.add_entry(text, risk)
    print("Trend          :", mem.get_trend())
    print("Trend direction:", mem.get_trend_direction())

    print("\n=== Scenario B: Improving trend ===")
    mem2 = EmotionalMemorySystem(session_id="demo_b")
    for text, risk in [
        ("I want to disappear.", "Crisis"),
        ("Can't stop crying.", "Crisis"),
        ("Feeling a bit less hopeless.", "Negative"),
        ("Had a small win today.", "Neutral"),
        ("Things are looking up.", "Positive"),
    ]:
        mem2.add_entry(text, risk)
    print("Trend          :", mem2.get_trend())
    print("Trend direction:", mem2.get_trend_direction())
