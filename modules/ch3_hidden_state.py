"""
Ch.3 — Hidden State Feature Extractor
Extracts mean-pooled transformer hidden states as rich feature vectors.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LLM_HF_MODEL

_tokenizer = None
_model     = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        print("[Ch.3] Loading Phi-3 for hidden state extraction...")
        _tokenizer = AutoTokenizer.from_pretrained(LLM_HF_MODEL)
        _model     = AutoModelForCausalLM.from_pretrained(
            LLM_HF_MODEL,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=False,
        )
        _model.eval()
        print("[Ch.3] Model ready.")


def extract_hidden_state(text: str) -> torch.Tensor:
    """
    Returns mean-pooled last hidden state of shape [hidden_dim].
    For Phi-3-mini: hidden_dim = 3072
    """
    _load_model()
    device    = next(_model.parameters()).device
    input_ids = _tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).input_ids.to(device)

    with torch.no_grad():
        hidden_states = _model.model(input_ids)[0]   # [1, seq_len, hidden_dim]

    return hidden_states[0].mean(dim=0).cpu()        # [hidden_dim]


if __name__ == "__main__":
    samples = [
        "I feel so happy and grateful today!",
        "I want to disappear and stop being a burden to everyone.",
    ]
    for s in samples:
        vec = extract_hidden_state(s)
        print(f"  shape={tuple(vec.shape)}  norm={vec.norm():.3f}  '{s[:50]}'")
