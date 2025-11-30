# therapist_model.py
from utils import therapist_style_response

def generate_therapy_response(user_text, sentiment, emotion, sarcasm, insights):
    """
    Smarter therapist response:
    - Uses base reply from utils
    - Adds short emotion-aware lines
    - Notes sarcasm if present
    - Adds dataset-based tips from insights
    """
    base_reply = therapist_style_response(user_text, sentiment, insights)
    final = base_reply + "\n\n"
    response = ""

    # Emotion-aware responses
    if emotion == "sadness":
        response += "I can hear some sadness in your words. It's okay to feel low. You're not alone in this.\n"
    elif emotion == "anger":
        response += "It sounds like something has frustrated you. Anger often shows that something important to you was affected.\n"
    elif emotion == "fear":
        response += "There seems to be some worry or fear in what you're expressing. It's natural to feel this way.\n"
    elif emotion == "joy":
        response += "I'm glad you're feeling some positivity. Moments like these matter.\n"
    elif emotion == "love":
        response += "There’s warmth in what you're expressing. Love and care add meaning to life.\n"
    elif emotion == "surprise":
        response += "It sounds like something unexpected happened. Surprises can bring mixed feelings.\n"

    # Sarcasm-aware
    if sarcasm == "sarcastic":
        response += "I notice a bit of sarcasm. Sometimes it can be a way to hide frustration or tiredness.\n"

    # Dataset insights additions
    avg_stress = insights.get("avg_stress")
    avg_sleep = insights.get("avg_sleep_quality")
    avg_screen = insights.get("avg_screen_time")

    if avg_stress and avg_stress > 6:
        response += "\nA lot of people in the dataset report higher stress. Short breaks and breathing can help.\n"
    if avg_sleep and avg_sleep < 6:
        response += "\nSleep seems low in the data. Small sleep improvements can make a difference.\n"
    if avg_screen and avg_screen > 5:
        response += "\nHigh screen time appears in the dataset — short screen breaks can help reset your mind.\n"

    final_response = final + response
    return final_response.strip()
