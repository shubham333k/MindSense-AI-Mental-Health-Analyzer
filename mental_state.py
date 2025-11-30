# mental_state.py
from typing import Dict, Tuple

WEIGHTS = {
    "sentiment_negative": 0.4,
    "sadness": 0.25,
    "fear": 0.2,
    "anger": 0.15,
    "sarcasm": 0.1,
    "rumination_keywords": 0.25,
}

RUMINATION_KEYWORDS = {"can't", "never", "always", "should", "must", "hopeless", "ruined"}

def score_rumination(text: str) -> float:
    tokens = text.lower().split()
    count = sum(1 for t in tokens if t in RUMINATION_KEYWORDS)
    return min(1.0, count / 3)

def infer_mental_state(signals: Dict) -> Tuple[str, float]:
    sent_neg = 1.0 if signals.get("sentiment") == "negative" else 0.0

    sadness = 1.0 if signals.get("emotion") == "sadness" else 0.0
    fear = 1.0 if signals.get("emotion") == "fear" else 0.0
    anger = 1.0 if signals.get("emotion") == "anger" else 0.0
    sarcasm = 1.0 if signals.get("sarcasm") == "sarcastic" else 0.0

    rum = score_rumination(signals.get("text", ""))

    score_low_mood = WEIGHTS["sentiment_negative"]*sent_neg + WEIGHTS["sadness"]*sadness + WEIGHTS["rumination_keywords"]*rum
    score_anxiety = WEIGHTS["sentiment_negative"]*sent_neg + WEIGHTS["fear"]*fear
    score_irritated = WEIGHTS["anger"]*anger + WEIGHTS["sarcasm"]*sarcasm
    score_ruminate = rum * 0.8

    scores = {
        "low_mood": score_low_mood,
        "high_anxiety": score_anxiety,
        "irritated": score_irritated,
        "ruminating": score_ruminate
    }

    state = max(scores, key=scores.get)
    confidence = float(min(1.0, scores[state]))

    if confidence < 0.25:
        return "neutral_ok", confidence

    return state, confidence
