from typing import TypedDict, Optional


class EmotionScores(TypedDict):
    sadness : float
    joy     : float
    love    : float
    anger   : float
    fear    : float
    surprise: float


class ClassifierOutput(TypedDict):
    # Core (always present)
    risk_level      : str           # Positive | Neutral | Negative | Crisis
    risk_score      : float         # 0.0 ~ 1.0
    primary_emotion : str
    emotion_scores  : EmotionScores

    # Full mode only (Ch.6)
    reasoning       : Optional[str]
    recommendation  : Optional[str]

    # Full mode only (Ch.7)
    empathy_response  : Optional[str]
    trend             : list          # list[str] — last k risk_levels
    trend_direction   : Optional[str] # improving | stable | deteriorating

    # Optional (Ch.5)
    cluster_id       : Optional[int]
    cluster_keywords : Optional[list]

    # Metadata
    timestamp: str
    mode     : str                  # quick | full


class ErrorOutput(TypedDict):
    error    : str
    code     : str   # EMPTY_INPUT | TOO_LONG | MODEL_ERROR
    timestamp: str
