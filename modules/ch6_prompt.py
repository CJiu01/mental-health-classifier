"""
Ch.6 — Prompt Engineering Pipeline
6-component system prompt + 2-shot examples + Chain-of-Thought reasoning.
Fully independent: takes raw text only, no ch4 results injected.
"""

import os
import sys
import json
from datetime import datetime, timezone
from llama_cpp import Llama

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LLM_REPO_ID, LLM_FILENAME, VALID_RISK_LEVELS

# ── Prompt Components ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
[PERSONA]
You are a professional mental health text analysis assistant trained to detect \
emotional signals in everyday written language.

[INSTRUCTION]
Analyze the given text and classify the writer's mental health risk level \
into exactly one of four categories: Positive, Neutral, Negative, or Crisis.
Use step-by-step reasoning before reaching your conclusion.

[CONTEXT]
Daily texts such as diary entries and social media posts contain subtle emotional \
cues that may indicate early signs of depression, anxiety, or crisis. \
Accurate detection enables timely support before critical moments are missed.

[OUTPUT FORMAT]
Respond ONLY with a valid JSON object matching this exact schema:
{
  "risk_level"     : "Positive" | "Neutral" | "Negative" | "Crisis",
  "primary_emotion": "sadness" | "joy" | "love" | "anger" | "fear" | "surprise",
  "reasoning"      : "<step-by-step explanation, 2-3 sentences>",
  "recommendation" : "<one actionable suggestion tailored to the risk level>"
}

[AUDIENCE]
This analysis is read by the user themselves for personal self-awareness. \
It is not a clinical diagnosis.

[TONE]
Be empathetic, non-judgmental, and professional. \
For Crisis level, always encourage seeking professional help.\
"""

FEW_SHOT_MESSAGES = [
    # Shot 1 — Positive
    {
        "role": "user",
        "content": (
            'Analyze this text: "Finally submitted my thesis. '
            'I feel so relieved and proud of myself. Going out to celebrate tonight!"\n\n'
            "Think step by step before classifying."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "risk_level"     : "Positive",
            "primary_emotion": "joy",
            "reasoning"      : (
                "The text expresses achievement and relief following task completion. "
                "Words like 'relieved', 'proud', and 'celebrate' indicate strong "
                "positive affect with no signs of distress."
            ),
            "recommendation" : (
                "Keep celebrating your achievements. "
                "Maintain this positive momentum by setting your next goal."
            ),
        }, indent=2),
    },
    # Shot 2 — Crisis
    {
        "role": "user",
        "content": (
            'Analyze this text: "I am so tired of everything. '
            'I just want to disappear and stop being a burden to everyone around me."\n\n'
            "Think step by step before classifying."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "risk_level"     : "Crisis",
            "primary_emotion": "sadness",
            "reasoning"      : (
                "The phrase 'want to disappear' and 'burden to everyone' are "
                "clinically recognized warning signals for suicidal ideation. "
                "Combined with expressions of exhaustion, this indicates a "
                "high-risk state requiring immediate attention."
            ),
            "recommendation" : (
                "Please reach out to a trusted person or contact a mental health "
                "crisis line immediately. You are not alone."
            ),
        }, indent=2),
    },
]


def _build_messages(text: str) -> list:
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + FEW_SHOT_MESSAGES
        + [{
            "role": "user",
            "content": f'Analyze this text: "{text}"\n\nThink step by step before classifying.',
        }]
    )


# ── Classifier ─────────────────────────────────────────────────────────────────

class PromptClassifier:

    def __init__(self):
        print("[Ch.6] Loading Phi-3 GGUF model...")
        self._llm = Llama.from_pretrained(
            repo_id      = LLM_REPO_ID,
            filename     = LLM_FILENAME,
            n_gpu_layers = -1,
            n_ctx        = 2048,
            verbose      = False,
        )
        print("[Ch.6] Ready.")

    def predict(self, text: str) -> dict:
        messages = _build_messages(text)
        raw = self._llm.create_chat_completion(
            messages        = messages,
            response_format = {"type": "json_object"},
            temperature     = 0,
            max_tokens      = 256,
        )["choices"][0]["message"]["content"]

        result = self._parse(raw)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        return result

    def _parse(self, raw: str) -> dict:
        try:
            result = json.loads(raw)
            assert result.get("risk_level") in VALID_RISK_LEVELS
            return result
        except (json.JSONDecodeError, KeyError, AssertionError):
            return {
                "risk_level"     : "Neutral",
                "primary_emotion": "surprise",
                "reasoning"      : f"Parsing failed. Raw: {raw[:100]}",
                "recommendation" : "Please try again.",
            }


if __name__ == "__main__":
    clf = PromptClassifier()
    samples = [
        "I feel so happy and grateful today!",
        "I am really worried and can't stop thinking about the worst.",
        "I want to disappear and stop being a burden to everyone.",
    ]
    for s in samples:
        out = clf.predict(s)
        print(f"\n[{out['risk_level']:8}]  '{s[:55]}'")
        print(f"  Emotion : {out.get('primary_emotion')}")
        print(f"  Reason  : {out.get('reasoning', '')[:100]}")
        print(f"  Advice  : {out.get('recommendation', '')[:100]}")
