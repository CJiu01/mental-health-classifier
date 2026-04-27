# Shared constants for all modules

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
ANCHOR_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID     = "microsoft/Phi-3-mini-4k-instruct-gguf"
LLM_FILENAME    = "*fp16.gguf"
LLM_HF_MODEL    = "microsoft/Phi-3-mini-4k-instruct"
DATASET_NAME    = "dair-ai/emotion"

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

VALID_RISK_LEVELS  = ["Positive", "Neutral", "Negative", "Crisis"]
VALID_EMOTIONS     = EMOTION_LABELS

RISK_THRESHOLDS = {
    "crisis"  : 0.70,
    "negative": 0.40,
    "positive": 0.50,   # joy + love combined
}

# Numeric weight per risk level (used for trend_direction calculation)
RISK_WEIGHT = {"Positive": 0, "Neutral": 1, "Negative": 2, "Crisis": 3}

LOGREG_PARAMS = {
    "C"           : 1.0,
    "solver"      : "lbfgs",
    "multi_class" : "auto",
    "max_iter"    : 1000,
    "random_state": 42,
    "class_weight": "balanced",
}

MEMORY_WINDOW_K = 7

SAVED_MODELS_DIR = "saved_models"
SESSIONS_DIR     = "sessions"
