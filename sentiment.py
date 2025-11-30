# sentiment.py
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure NLTK VADER is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Sentiment (VADER)
def get_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.25:
        label = "positive"
    elif compound <= -0.25:
        label = "negative"
    else:
        label = "neutral"
    return label, abs(compound), scores

# Emotion model (DistilBERT)
emotion_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]

def get_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = emotion_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    conf, idx = torch.max(probs, 0)
    return emotion_labels[idx], float(conf)

# Simple sarcasm detector (keyword/emoji heuristics)
sarcasm_keywords = [
    "yeah right", "sure", "of course", "amazing", "wow",
    "great job", "nice", "brilliant", "as if", "obviously",
    "totally", "fantastic", "right...", "ok sure"
]

def get_sarcasm(text):
    text_lower = text.lower()
    score = 0
    for k in sarcasm_keywords:
        if k in text_lower:
            score += 1
    if "🙄" in text_lower or "🤦" in text_lower:
        score += 1
    if score > 0:
        return "sarcastic", 0.7
    else:
        return "not_sarcastic", 0.3

# Combined analyzer
def analyze_text_full(text):
    sentiment, sent_conf, sent_scores = get_sentiment(text)
    try:
        emotion, emo_conf = get_emotion(text)
    except Exception as e:
        # fail-safe: return neutral emotion on model issues
        emotion, emo_conf = "joy", 0.0
    sarcasm, sar_conf = get_sarcasm(text)
    return {
        "sentiment": sentiment,
        "sentiment_confidence": sent_conf,
        "sentiment_scores": sent_scores,
        "emotion": emotion,
        "emotion_confidence": emo_conf,
        "sarcasm": sarcasm,
        "sarcasm_confidence": sar_conf
    }
