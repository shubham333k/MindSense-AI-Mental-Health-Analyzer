# utils.py
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Clean Text
# ---------------------------------------------------------
def clean_text(text):
    text = re.sub(r'http\S+', '', text)                # URLs
    text = re.sub(r'@\w+', '', text)                  # @mentions
    text = re.sub(r'[^A-Za-z\s]', '', text)           # symbols
    text = re.sub(r'\s+', ' ', text).strip()          # extra spaces
    return text

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
def load_data():
    """Load CSV dataset from project folder (common locations)."""
    # Try local file in project folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "Mental_Health_and_Social_Media_Balance_Dataset.csv"),
        os.path.join(current_dir, "data", "Mental_Health_and_Social_Media_Balance_Dataset.csv")
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    # not found
    raise FileNotFoundError("Dataset not found in project folder.")

# ---------------------------------------------------------
# Analyze Data (safe summary + insight extraction)
# ---------------------------------------------------------
def analyze_data(df):
    insights = {
        "avg_screen_time": None,
        "avg_sleep_quality": None,
        "avg_stress": None,
        "avg_happiness": None,
        "top_platform": None,
        "screen_happiness_corr": None,
        "error": None
    }
    if df is None:
        insights["error"] = "Dataset not found."
        return insights
    try:
        if "Daily_Screen_Time(hrs)" in df.columns:
            col = df["Daily_Screen_Time(hrs)"].dropna()
            if not col.empty:
                insights["avg_screen_time"] = round(col.mean(), 2)
        if "Sleep_Quality(1-10)" in df.columns:
            col = df["Sleep_Quality(1-10)"].dropna()
            if not col.empty:
                insights["avg_sleep_quality"] = round(col.mean(), 2)
        if "Stress_Level(1-10)" in df.columns:
            col = df["Stress_Level(1-10)"].dropna()
            if not col.empty:
                insights["avg_stress"] = round(col.mean(), 2)
        if "Happiness_Index(1-10)" in df.columns:
            col = df["Happiness_Index(1-10)"].dropna()
            if not col.empty:
                insights["avg_happiness"] = round(col.mean(), 2)
        if "Social_Media_Platform" in df.columns:
            col = df["Social_Media_Platform"].dropna()
            if not col.empty:
                insights["top_platform"] = col.value_counts().idxmax()
        if ("Daily_Screen_Time(hrs)" in df.columns and "Happiness_Index(1-10)" in df.columns):
            x = df["Daily_Screen_Time(hrs)"].dropna()
            y = df["Happiness_Index(1-10)"].dropna()
            if not x.empty and not y.empty:
                insights["screen_happiness_corr"] = round(x.corr(y), 2)
    except Exception as e:
        insights["error"] = f"Error while computing insights: {e}"
    return insights

# ---------------------------------------------------------
# Predefined supportive text responses
# ---------------------------------------------------------
def therapist_style_response(user_text, sentiment, insights):
    text = (user_text or "").lower()

    positive_responses = [
        "It sounds like you're experiencing something uplifting. It's great that you can recognize these moments. What else has been bringing you joy lately?",
        "That’s wonderful to hear. Staying connected to the things that motivate you can really support emotional well-being.",
        "It feels like you’re in a good place right now. Hold onto that positivity and let it guide your daily actions."
    ]

    negative_responses = [
        "I’m sorry you're going through this. It's okay to feel this way sometimes. Want to talk about what's been weighing on you?",
        "It sounds like something has been affecting your emotional balance. You’re not alone in this — I’m here to listen.",
        "It seems like you're dealing with something tough. Recognizing this feeling is an important step. What do you think might help ease this moment?"
    ]

    neutral_responses = [
        "I’m hearing you. If you'd like, you can share a bit more about what's on your mind.",
        "Thanks for sharing. Even neutral moments carry meaning. Want to explore what you’re feeling under the surface?",
        "I appreciate you expressing this. Feel free to share anything that might help us understand your situation better."
    ]

    extra = ""
    avg_stress = insights.get("avg_stress")
    avg_sleep = insights.get("avg_sleep_quality")

    if avg_stress is not None and avg_stress > 6:
        extra += "\n\nBy the way, I noticed higher stress in the data. Small grounding routines can help, like slow breaths or stepping away for a moment."
    if avg_sleep is not None and avg_sleep < 6:
        extra += "\n\nSleep quality can shape emotional balance. Even small improvements can make a difference."

    if sentiment == "positive":
        base = positive_responses
    elif sentiment == "negative":
        base = negative_responses
    else:
        base = neutral_responses

    import random
    return random.choice(base) + extra

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
def plot_visuals(df):
    charts = {}
    # 1. Screen time vs happiness
    fig1, ax1 = plt.subplots()
    ax1.scatter(df['Daily_Screen_Time(hrs)'], df['Happiness_Index(1-10)'])
    ax1.set_xlabel("Daily Screen Time (hrs)")
    ax1.set_ylabel("Happiness Index (1-10)")
    ax1.set_title("Screen Time vs Happiness Index")
    charts['screen_vs_happiness'] = fig1
    # 2. Stress vs sleep
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['Stress_Level(1-10)'], df['Sleep_Quality(1-10)'])
    ax2.set_xlabel("Stress Level (1-10)")
    ax2.set_ylabel("Sleep Quality (1-10)")
    ax2.set_title("Stress Level vs Sleep Quality")
    charts['stress_vs_sleep'] = fig2
    # 3. Average metrics
    fig3, ax3 = plt.subplots()
    averages = {
        "Screen Time": df['Daily_Screen_Time(hrs)'].mean(),
        "Sleep Quality": df['Sleep_Quality(1-10)'].mean(),
        "Stress": df['Stress_Level(1-10)'].mean(),
        "Happiness": df['Happiness_Index(1-10)'].mean()
    }
    ax3.bar(averages.keys(), averages.values())
    ax3.set_title("Average Well-being Indicators")
    ax3.set_ylabel("Score / Hours")
    charts['averages'] = fig3
    return charts

# Small helper tips and followups
def coping_tips(emotion):
    tips = {
        "sadness": "Try a grounding activity—take a short walk, stretch, or write down what you're feeling.",
        "anger": "Take a deep breath… sometimes counting backwards from 10 helps calm the body.",
        "fear": "Fear feels heavy, but you’re safe right now. Slow breathing may help.",
        "joy": "Hold onto this feeling—maybe take a moment to reflect on what made you feel good.",
        "love": "That’s beautiful—expressing appreciation or gratitude can make it even stronger.",
        "surprise": "Unexpected moments can be confusing. Take your time to process it."
    }
    return tips.get(emotion, "Take a moment, breathe deeply, and be kind to yourself.")

def followup_question(emotion):
    questions = {
        "sadness": "What do you think triggered these feelings?",
        "anger": "What part of the situation felt most upsetting?",
        "fear": "Is there a specific worry on your mind?",
        "joy": "What made you feel happy recently?",
        "love": "Would you like to share what made you feel this connection?",
        "surprise": "Did something unexpected happen today?"
    }
    return questions.get(emotion, "Tell me more about how you're feeling.")
