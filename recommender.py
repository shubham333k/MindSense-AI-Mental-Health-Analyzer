# recommender.py
import random

RECOMMENDATIONS = {
    "low_mood": [
        {"id": "walk_10", "text": "Take a short 10-minute walk."},
        {"id": "music", "text": "Play one song that makes you feel better."},
        {"id": "small_task", "text": "Do one small, easy task right now."},
    ],
    "high_anxiety": [
        {"id": "breathing_478", "text": "Try 4-7-8 breathing for 3 cycles."},
        {"id": "ground_5", "text": "Try grounding: notice 5 things you can see."},
    ],
    "irritated": [
        {"id": "pause_30", "text": "Pause for 30 seconds and breathe slowly."},
        {"id": "write_2", "text": "Write down what bothered you for 2 minutes."},
    ],
    "ruminating": [
        {"id": "schedule_thoughts", "text": "Set aside 10 minutes later to think about this instead of now."},
        {"id": "short_distraction", "text": "Take a short break and focus on something else for a moment."},
    ],
    "neutral_ok": [
        {"id": "check_in", "text": "You're doing okay. Want a wellbeing tip?"},
    ]
}

def get_recommendations(state, top_k=2):
    pool = RECOMMENDATIONS.get(state, RECOMMENDATIONS["neutral_ok"])
    random.shuffle(pool)
    return pool[:top_k]
