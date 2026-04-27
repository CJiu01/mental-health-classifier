"""
Mental Health Text Classifier — Gradio Web Demo
Run locally : python app.py
Run in Colab: !python app.py   (generates a public share link)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

# ── Auto-train if saved model is missing ──────────────────────────────────────
def _ensure_model():
    from config import SAVED_MODELS_DIR
    model_path = os.path.join(SAVED_MODELS_DIR, "ch4_logreg.pkl")
    if not os.path.exists(model_path):
        print("[App] ch4 model not found — auto-training on dair-ai/emotion …")
        print("[App] This takes ~2–3 min on first run (downloading dataset + encoding).")
        from data.loader import load_emotion_dataset
        from modules.ch4_classifier import MentalHealthClassifier
        data = load_emotion_dataset()
        clf  = MentalHealthClassifier()
        clf.train(data["train"]["texts"], data["train"]["labels"])
        clf.save()
        print("[App] Training complete. Model saved.")

_ensure_model()

# ── Lazy pipeline loader ───────────────────────────────────────────────────────
_pipelines: dict = {}

def _get_pipe(mode: str):
    if mode not in _pipelines:
        print(f"[App] Initialising pipeline ({mode} mode)…")
        from pipeline import MentalHealthPipeline
        _pipelines[mode] = MentalHealthPipeline(mode=mode)
    return _pipelines[mode]

# ── Style constants ────────────────────────────────────────────────────────────
RISK_COLOR = {
    "Positive": "#27ae60",
    "Neutral":  "#7f8c8d",
    "Negative": "#e67e22",
    "Crisis":   "#c0392b",
}
RISK_EMOJI = {
    "Positive": "🟢",
    "Neutral":  "⚪",
    "Negative": "🟠",
    "Crisis":   "🔴",
}

# ── Core inference function ────────────────────────────────────────────────────
def classify(text: str, use_full_mode: bool):
    if not text or not text.strip():
        empty = "<p style='color:#aaa;text-align:center;padding:20px'>텍스트를 입력하세요.</p>"
        return empty, {}, "<p style='color:#aaa'>—</p>", ""

    mode = "full" if use_full_mode else "quick"
    try:
        out = _get_pipe(mode).run(text.strip())
    except Exception as exc:
        return f"<p style='color:red'><b>Error:</b> {exc}</p>", {}, "", ""

    risk      = out.get("risk_level", "Neutral")
    score     = out.get("risk_score") or 0.0
    emotion   = out.get("primary_emotion") or "—"
    em_scores = out.get("emotion_scores") or {}
    trend     = out.get("trend", [])
    trend_dir = out.get("trend_direction", "stable")
    color     = RISK_COLOR.get(risk, "#7f8c8d")
    emoji     = RISK_EMOJI.get(risk, "")

    # ── Risk level badge ────────────────────────────────────────────────────────
    risk_html = f"""
<div style="text-align:center;padding:20px 10px">
  <div style="display:inline-block;background:{color};color:#fff;
              padding:14px 44px;border-radius:14px;font-size:26px;
              font-weight:700;box-shadow:0 4px 12px rgba(0,0,0,.18)">
    {emoji}&nbsp; {risk}
  </div>
  <p style="font-size:13px;color:#555;margin-top:10px">
    Risk Score: <b>{score:.3f}</b>&nbsp;|&nbsp;Primary Emotion: <b>{emotion}</b>
  </p>
</div>"""

    # ── Trend ───────────────────────────────────────────────────────────────────
    arrows    = {"deteriorating": "📉", "improving": "📈", "stable": "➡️"}
    arrow     = arrows.get(trend_dir, "")
    trend_str = " → ".join(trend) if trend else "(첫 번째 입력)"
    trend_html = f"""
<div style="padding:10px 6px;font-size:14px">
  <p style="margin:4px 0"><b>Trend Direction:</b> {arrow} <b>{trend_dir}</b></p>
  <p style="margin:6px 0;color:#666;font-size:12px"><b>History:</b> {trend_str}</p>
</div>"""

    # ── LLM output (full mode only) ─────────────────────────────────────────────
    llm_html = ""
    if use_full_mode:
        reasoning      = (out.get("reasoning")       or "—").strip()
        recommendation = (out.get("recommendation")  or "—").strip()
        empathy        = (out.get("empathy_response") or "—").strip()
        llm_html = f"""
<div style="background:#f8f9fa;border-left:4px solid {color};
            border-radius:6px;padding:14px 18px;font-size:14px;line-height:1.6">
  <p style="margin:0 0 10px"><b>🧠 Reasoning</b><br>
    <span style="color:#444">{reasoning}</span></p>
  <hr style="border:none;border-top:1px solid #ddd;margin:10px 0">
  <p style="margin:0 0 10px"><b>💡 Recommendation</b><br>
    <span style="color:#444">{recommendation}</span></p>
  <hr style="border:none;border-top:1px solid #ddd;margin:10px 0">
  <p style="margin:0"><b>💙 Empathy Response</b><br>
    <span style="color:#444">{empathy}</span></p>
</div>"""

    return risk_html, em_scores, trend_html, llm_html


# ── Gradio UI ──────────────────────────────────────────────────────────────────
CSS = """
footer { display:none !important; }
.submit-btn { font-size:15px !important; }
"""

with gr.Blocks(title="Mental Health Text Classifier", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown("""
# 🧠 Mental Health Text Classifier
**Prototype Demo** — 짧은 텍스트(일기, SNS 글 등)를 입력하면 감정 리스크 수준을 분석합니다.

| 🟢 Positive | ⚪ Neutral | 🟠 Negative | 🔴 Crisis |
|---|---|---|---|
| Joy / Love 우세 | 특별한 신호 없음 | Sadness / Fear 감지 | 즉각 주의 필요 |

> ⚠️ 본 시스템은 연구 프로토타입이며 **임상 진단 도구가 아닙니다.**
""")

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                lines=5,
                placeholder="예: I feel so tired and hopeless lately...",
                label="Input Text",
            )
            with gr.Row():
                full_mode_check = gr.Checkbox(
                    label="Full Mode  (Ch.6 LLM reasoning + Ch.7 empathy  |  ~3–5s)",
                    value=False,
                )
                submit_btn = gr.Button("Analyze →", variant="primary", elem_classes="submit-btn")

        with gr.Column(scale=2):
            risk_output    = gr.HTML(label="Risk Level")
            emotion_output = gr.Label(num_top_classes=6, label="Emotion Scores")

    with gr.Row():
        trend_output = gr.HTML(label="Trend")
        llm_output   = gr.HTML(label="LLM Output (Full Mode)")

    gr.Examples(
        examples=[
            ["I feel so happy and grateful today! Life is wonderful.",             False],
            ["Just a regular day, nothing much happened.",                          False],
            ["I've been really anxious and can't stop worrying about everything.", False],
            ["I want to disappear and stop being a burden to everyone.",            False],
        ],
        inputs=[text_input, full_mode_check],
        label="예시 텍스트",
    )

    # Wire up events
    submit_btn.click(
        fn=classify,
        inputs=[text_input, full_mode_check],
        outputs=[risk_output, emotion_output, trend_output, llm_output],
    )
    text_input.submit(
        fn=classify,
        inputs=[text_input, full_mode_check],
        outputs=[risk_output, emotion_output, trend_output, llm_output],
    )


if __name__ == "__main__":
    demo.launch(
        share=True,        # generates a public URL (works in Colab too)
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860,
    )
