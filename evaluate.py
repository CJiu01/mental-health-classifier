"""
evaluate.py — Experiment runner (Exp 1, 2, 3 from Step 5 design)
Compares ch1 / ch2 / ch4 on the same test set and generates report figures.
"""

import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no GUI windows)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

from modules.ch4_classifier import MentalHealthClassifier, compute_risk
from config import EMOTION_LABELS, VALID_RISK_LEVELS, SAVED_MODELS_DIR, CH11_SAVED_DIR


# ── Risk label helpers ─────────────────────────────────────────────────────────

def emotion_label_to_risk(emotion_label: str) -> str:
    """Map a single emotion label string → risk level (for ch1/ch2 comparison)."""
    mapping = {
        "joy"     : "Positive",
        "love"    : "Positive",
        "surprise": "Neutral",
        "anger"   : "Neutral",
        "fear"    : "Negative",
        "sadness" : "Negative",
    }
    return mapping.get(emotion_label, "Neutral")


def int_labels_to_risk(int_labels: list) -> list:
    """Convert dair-ai/emotion integer labels → risk level strings."""
    return [emotion_label_to_risk(EMOTION_LABELS[i]) for i in int_labels]


# ── Exp 1: Core Classifier Evaluation ─────────────────────────────────────────

def exp1_core_evaluation(clf: MentalHealthClassifier,
                          test_texts: list, test_labels: list) -> dict:
    """6-class emotion classification report + confusion matrix."""
    print("\n" + "="*60)
    print("Exp 1 — Core Classifier (Ch.4) Evaluation")
    print("="*60)
    report = clf.evaluate(test_texts, test_labels)
    plot_confusion_matrix(
        "Ch.4 LogisticRegression",
        test_labels,
        _predict_int_labels(clf, test_texts),
        EMOTION_LABELS,
    )
    return report


def _predict_int_labels(clf: MentalHealthClassifier, texts: list) -> list:
    embeddings  = clf.encoder.encode(texts, show_progress_bar=True, batch_size=64)
    return list(clf.clf.predict(embeddings))


# ── Exp 2: Model Comparison ────────────────────────────────────────────────────

def exp2_model_comparison(test_texts: list, test_labels: list,
                           ch4_clf: MentalHealthClassifier,
                           ch1_clf=None, ch2_clf=None) -> pd.DataFrame:
    """Compare ch1 / ch2 / ch4 on 4-tier risk level classification."""
    print("\n" + "="*60)
    print("Exp 2 — Model Comparison (4-tier risk level)")
    print("="*60)

    true_risk = int_labels_to_risk(test_labels)
    rows      = []

    # Ch.4 predictions
    print("[Exp 2] Running Ch.4...")
    ch4_preds = []
    embeddings = ch4_clf.encoder.encode(test_texts, show_progress_bar=True, batch_size=64)
    for vec in embeddings:
        proba = ch4_clf.clf.predict_proba(vec.reshape(1, -1))[0]
        scores = {n: float(p) for n, p in zip(EMOTION_LABELS, proba)}
        risk, _ = compute_risk(scores)
        ch4_preds.append(risk)
    rows.append(_summary_row("Ch.4 LogisticReg (trained)", true_risk, ch4_preds))

    # Ch.2 predictions (optional)
    if ch2_clf is not None:
        print("[Exp 2] Running Ch.2...")
        ch2_preds = [ch2_clf.predict(t)["risk_level"] for t in test_texts]
        rows.append(_summary_row("Ch.2 Anchor Cosine (no-train)", true_risk, ch2_preds))

    # Ch.1 predictions (optional, slow)
    if ch1_clf is not None:
        print("[Exp 2] Running Ch.1 (this may take a while)...")
        ch1_preds = [ch1_clf.predict(t)["risk_level"] for t in test_texts]
        rows.append(_summary_row("Ch.1 Zero-Shot LLM (no-train)", true_risk, ch1_preds))

    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))
    plot_comparison_bar(df)
    return df


def _summary_row(name: str, y_true: list, y_pred: list) -> dict:
    rep = classification_report(y_true, y_pred,
                                 labels=VALID_RISK_LEVELS, output_dict=True,
                                 zero_division=0)
    return {
        "Model"      : name,
        "Accuracy"   : round(accuracy_score(y_true, y_pred), 4),
        "Macro F1"   : round(rep["macro avg"]["f1-score"], 4),
        "Macro Prec.": round(rep["macro avg"]["precision"], 4),
        "Macro Rec." : round(rep["macro avg"]["recall"], 4),
    }


# ── Exp 3: Risk Distribution ───────────────────────────────────────────────────

def exp3_risk_distribution(test_texts: list, test_labels: list,
                            ch4_clf: MentalHealthClassifier) -> None:
    print("\n" + "="*60)
    print("Exp 3 — Risk Level Distribution")
    print("="*60)

    embeddings = ch4_clf.encoder.encode(test_texts, show_progress_bar=True, batch_size=64)
    risk_preds = []
    for vec in embeddings:
        proba  = ch4_clf.clf.predict_proba(vec.reshape(1, -1))[0]
        scores = {n: float(p) for n, p in zip(EMOTION_LABELS, proba)}
        risk, _ = compute_risk(scores)
        risk_preds.append(risk)

    counts = {r: risk_preds.count(r) for r in VALID_RISK_LEVELS}
    print("Risk distribution:", counts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    axes[0].pie(
        counts.values(), labels=counts.keys(), autopct="%1.1f%%",
        colors=["#2ecc71","#95a5a6","#e67e22","#e74c3c"]
    )
    axes[0].set_title("Risk Level Distribution (test set)")

    # Bar chart
    axes[1].bar(counts.keys(), counts.values(),
                color=["#2ecc71","#95a5a6","#e67e22","#e74c3c"])
    axes[1].set_title("Risk Level Counts")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("risk_distribution.png", dpi=150)
    plt.show()


# ── Plot helpers ───────────────────────────────────────────────────────────────

def plot_confusion_matrix(title: str, y_true, y_pred, labels: list,
                          save_path: str = "confusion_matrix_ch4.png") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_comparison_bar(df: pd.DataFrame, save_path: str = "model_comparison.png") -> None:
    x   = np.arange(len(df))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, df["Accuracy"], w, label="Accuracy", color="#3498db")
    ax.bar(x + w/2, df["Macro F1"], w, label="Macro F1", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=10, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Accuracy & Macro F1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ── Exp 4: Ch.11 Fine-tuned MentalBERT ────────────────────────────────────────

def exp4_ch11_evaluation(test_texts: list, test_labels: list) -> dict:
    """6-class emotion report for Ch.11 fine-tuned model."""
    from modules.ch11_finetuned import FinetunedClassifier

    if not os.path.exists(CH11_SAVED_DIR):
        print("[Exp 4] Ch.11 model not found. Run notebooks/ch11_training.ipynb first.")
        return {}

    print("\n" + "="*60)
    print("Exp 4 — Ch.11 Fine-tuned MentalBERT Evaluation")
    print("="*60)

    clf     = FinetunedClassifier()
    print("[Exp 4] Running batched inference on test set...")
    results = clf.predict_batch_fast(test_texts, batch_size=32)
    preds   = [EMOTION_LABELS.index(r["primary_emotion"]) for r in results]

    report = classification_report(
        test_labels, preds,
        target_names=EMOTION_LABELS, output_dict=True
    )
    print(classification_report(test_labels, preds, target_names=EMOTION_LABELS))

    plot_confusion_matrix(
        "Ch.11 Fine-tuned MentalBERT",
        test_labels, preds,
        EMOTION_LABELS,
        save_path="confusion_matrix_ch11.png",
    )
    return report


def exp2_extended(test_texts: list, test_labels: list,
                  ch4_clf: MentalHealthClassifier, ch2_clf=None) -> pd.DataFrame:
    """3-way comparison: Ch.2 vs Ch.4 vs Ch.11 on 4-tier risk level."""
    from modules.ch11_finetuned import FinetunedClassifier

    print("\n" + "="*60)
    print("Exp 2 Extended — Ch.2 vs Ch.4 vs Ch.11 (4-tier risk)")
    print("="*60)

    true_risk = int_labels_to_risk(test_labels)
    rows      = []

    # Ch.4
    print("[Exp 2] Running Ch.4...")
    ch4_preds  = []
    embeddings = ch4_clf.encoder.encode(test_texts, show_progress_bar=True, batch_size=64)
    for vec in embeddings:
        proba  = ch4_clf.clf.predict_proba(vec.reshape(1, -1))[0]
        scores = {n: float(p) for n, p in zip(EMOTION_LABELS, proba)}
        risk, _ = compute_risk(scores)
        ch4_preds.append(risk)
    rows.append(_summary_row("Ch.4 LogisticReg — TP1 (frozen encoder)", true_risk, ch4_preds))

    # Ch.2 (optional)
    if ch2_clf is not None:
        print("[Exp 2] Running Ch.2...")
        ch2_preds = [ch2_clf.predict(t)["risk_level"] for t in test_texts]
        rows.append(_summary_row("Ch.2 Anchor Cosine (no-train)", true_risk, ch2_preds))

    # Ch.11
    if os.path.exists(CH11_SAVED_DIR):
        print("[Exp 2] Running Ch.11...")
        ch11_clf    = FinetunedClassifier()
        ch11_results = ch11_clf.predict_batch_fast(test_texts, batch_size=32)
        ch11_preds   = [r["risk_level"] for r in ch11_results]
        rows.append(_summary_row("Ch.11 Fine-tuned MentalBERT — TP2", true_risk, ch11_preds))
    else:
        print("[Exp 2] Ch.11 model not found — skipping.")

    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))
    plot_comparison_bar(df, save_path="model_comparison_tp2.png")
    return df


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.loader import load_emotion_dataset

    data = load_emotion_dataset()
    clf  = MentalHealthClassifier()
    clf.load()

    exp1_core_evaluation(clf, data["test"]["texts"], data["test"]["labels"])
    exp2_model_comparison(data["test"]["texts"], data["test"]["labels"], clf)
    exp3_risk_distribution(data["test"]["texts"], data["test"]["labels"], clf)
    exp4_ch11_evaluation(data["test"]["texts"], data["test"]["labels"])
    exp2_extended(data["test"]["texts"], data["test"]["labels"], clf)
