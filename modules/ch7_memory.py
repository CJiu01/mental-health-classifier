"""
Ch.7 — Conversational Memory & Longitudinal Tracking
Layer 1: Custom EmotionLog (trend tracking, 7-day window)
Layer 2: LangChain ConversationBufferWindowMemory + ConversationSummaryMemory
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timezone
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    LLM_REPO_ID, LLM_FILENAME, MEMORY_WINDOW_K,
    RISK_WEIGHT, SESSIONS_DIR
)

# ── Empathy Prompt ─────────────────────────────────────────────────────────────

EMPATHY_TEMPLATE = PromptTemplate(
    input_variables=["user_text", "risk_level", "trend_summary", "chat_history"],
    template="""\
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

Response:""",
)

TONE_GUIDE = {
    "Positive": "Reinforce positive momentum warmly.",
    "Neutral" : "Show gentle curiosity and interest.",
    "Negative": "Offer warm comfort and active listening.",
    "Crisis"  : "Immediately encourage professional help — you are not alone.",
}


class EmotionalMemorySystem:

    def __init__(self, session_id: str, k: int = MEMORY_WINDOW_K):
        self.session_id = session_id
        self.k          = k

        # Layer 1: custom emotion log
        self._log: list = []          # list of MemoryEntry dicts
        self._total_entries: int = 0

        # Layer 2: LangChain memory objects (initialised lazily)
        self._window_memory  = None
        self._summary_memory = None
        self._empathy_chain  = None
        self._llm            = None
        self._llm_ready      = False

    # ── LLM lazy loader ────────────────────────────────────────────────────────

    def _ensure_llm(self) -> None:
        if self._llm_ready:
            return
        print("[Ch.7] Loading LlamaCpp for empathy chain...")
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=LLM_REPO_ID, filename="Phi-3-mini-4k-instruct-fp16.gguf")
        self._llm  = LlamaCpp(
            model_path    = model_path,
            n_gpu_layers  = -1,
            n_ctx         = 2048,
            temperature   = 0.7,
            max_tokens    = 150,
            verbose       = False,
        )
        self._window_memory  = ConversationBufferWindowMemory(
            k=self.k, memory_key="chat_history"
        )
        self._summary_memory = ConversationSummaryMemory(
            llm=self._llm, memory_key="long_term_summary"
        )
        self._empathy_chain  = LLMChain(
            prompt=EMPATHY_TEMPLATE, llm=self._llm, memory=self._window_memory
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

        # Overflow → archive to SummaryMemory
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
        if not self._llm_ready:
            return
        msg = (f"Day {entry['day_index']}: risk={entry['risk_level']}, "
               f"emotion={entry['primary_emotion']}, text='{entry['text'][:60]}'")
        self._summary_memory.save_context({"input": msg}, {"output": ""})

    def _format_trend_summary(self) -> str:
        if not self._log:
            return "No entries yet."
        parts = [f"Day {e['day_index']}: {e['risk_level']}" for e in self._log]
        return " → ".join(parts)

    # ── Layer 2: Empathy Response ──────────────────────────────────────────────

    def generate_empathy(self, text: str, risk_level: str) -> str:
        self._ensure_llm()
        trend_summary = self._format_trend_summary()
        history       = self._window_memory.load_memory_variables({}).get(
                            "chat_history", ""
                        )
        response = self._empathy_chain.predict(
            user_text     = text,
            risk_level    = risk_level,
            trend_summary = trend_summary,
            chat_history  = history,
        )
        self._window_memory.save_context(
            {"input": text}, {"output": response.strip()}
        )
        return response.strip()

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: str = SESSIONS_DIR) -> None:
        os.makedirs(directory, exist_ok=True)
        path    = os.path.join(directory, f"{self.session_id}.json")
        payload = {
            "session_id"      : self.session_id,
            "created_at"      : datetime.now(timezone.utc).isoformat(),
            "total_entries"   : self._total_entries,
            "window_entries"  : self._log,
            "long_term_summary": (
                self._summary_memory.load_memory_variables({})
                    .get("long_term_summary", "")
                if self._llm_ready else ""
            ),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, directory: str = SESSIONS_DIR) -> None:
        path = os.path.join(directory, f"{self.session_id}.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            payload = json.load(f)
        self._log           = payload.get("window_entries", [])
        self._total_entries = payload.get("total_entries", len(self._log))


# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Scenario A: Deteriorating trend ===")
    mem = EmotionalMemorySystem(session_id="demo_a")
    scenario_a = [
        ("I had a great day today!", "Positive"),
        ("Feeling pretty good, got things done.", "Positive"),
        ("Normal day, nothing special.", "Neutral"),
        ("Feeling a bit low today.", "Neutral"),
        ("Really tired and anxious lately.", "Negative"),
        ("Can't sleep, everything feels hopeless.", "Negative"),
        ("I just want to disappear.", "Crisis"),
    ]
    for text, risk in scenario_a:
        mem.add_entry(text, risk)

    print("Trend          :", mem.get_trend())
    print("Trend direction:", mem.get_trend_direction())

    print("\n=== Scenario B: Improving trend ===")
    mem2 = EmotionalMemorySystem(session_id="demo_b")
    scenario_b = [
        ("I want to disappear.", "Crisis"),
        ("Can't stop crying.", "Crisis"),
        ("Feeling a bit less hopeless.", "Negative"),
        ("Had a small win today.", "Neutral"),
        ("Things are looking up.", "Positive"),
    ]
    for text, risk in scenario_b:
        mem2.add_entry(text, risk)

    print("Trend          :", mem2.get_trend())
    print("Trend direction:", mem2.get_trend_direction())
