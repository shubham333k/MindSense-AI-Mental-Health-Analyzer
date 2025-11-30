# app.py
import time
import streamlit as st
import pandas as pd
import math
from collections import Counter, defaultdict

# existing modules (your files)
from utils import clean_text, load_data, analyze_data, plot_visuals, coping_tips, followup_question
from sentiment import analyze_text_full, get_sentiment, get_emotion  # analyze_text_full uses these; see file.
from therapist_model import generate_therapy_response
from mental_state import infer_mental_state, score_rumination  # mental_state utilities
from recommender import get_recommendations  # base pool
from history_store import init_db, log_event, fetch_recent, record_feedback  # sqlite persistence

st.set_page_config(page_title="Mental Health TRIO + Advanced Analytics", layout="centered")
init_db()  # ensure DB present

# ------------------------------
# Load dataset insights (safe)
# ------------------------------
try:
    df = load_data()
    insights = analyze_data(df)
except FileNotFoundError:
    df = None
    insights = {
        "avg_screen_time": None,
        "avg_sleep_quality": None,
        "avg_stress": None,
        "avg_happiness": None,
        "top_platform": None,
        "screen_happiness_corr": None,
        "error": "Dataset not found - running in demo mode."
    }

# ------------------------------
# Session state
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "stage" not in st.session_state:
    st.session_state.stage = "initial"

if "initial_text" not in st.session_state:
    st.session_state.initial_text = ""

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "q_index" not in st.session_state:
    st.session_state.q_index = 0

if "results" not in st.session_state:
    st.session_state.results = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "mental_state" not in st.session_state:
    st.session_state.mental_state = None

if "mental_confidence" not in st.session_state:
    st.session_state.mental_confidence = 0.0

# ------------------------------
# Helpers: rerun and follow-up questions
# ------------------------------
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            return

def generate_followup_questions(user_text):
    q1 = "When did you first notice feeling like this?"
    q2 = "On a scale of 1-10, how strong is this feeling right now?"
    q3 = "Has anything changed in your routine or relationships recently?"
    q4 = "Who do you usually talk to about this, or how do you usually cope?"
    q5 = "What would help you feel a little better right now?"
    return [q1, q2, q3, q4, q5]

# ------------------------------
# New: Thinking style profiler
# ------------------------------
def profile_thinking_style(text: str):
    text = (text or "").lower()
    styles = {
        "self_critical": ["my fault", "i'm useless", "worthless", "i am the problem", "i'm the problem"],
        "catastrophic": ["ruined", "never", "always", "everything is", "nothing will", "it will never"],
        "avoidant": ["i don't want to", "i can't face", "avoid", "leave it", "i'll ignore"],
        "anxious_future": ["what if", "i'm scared", "panic", "i'm worried", "i'm afraid"],
        "solution_focused": ["i can try", "maybe i should", "i will", "i'll try", "plan to"]
    }
    scores = {k: 0 for k in styles}
    for style, words in styles.items():
        for w in words:
            if w in text:
                scores[style] += 1
    best_style = max(scores, key=scores.get)
    return best_style, scores

# ------------------------------
# New: Trigger detector
# ------------------------------
NEG_TRIGGERS = ["exam", "deadline", "family", "relationship", "breakup", "lonely", "job", "interview", "failure", "boss", "arguments"]

def detect_triggers(text: str):
    t = (text or "").lower()
    found = [w for w in NEG_TRIGGERS if w in t]
    return found

# ------------------------------
# New: Adaptive recommendation ranking using history DB
# ------------------------------
def compute_rec_success_rates(rows):
    """
    rows: output of fetch_recent -> list of tuples (id, ts, text, state, confidence, rec_ids, feedback)
    returns: dict rec_id -> (shown_count, success_count)
    """
    stats = defaultdict(lambda: {"shown": 0, "success": 0})
    for r in rows:
        rec_ids = r[5] or ""
        feedback = r[6] or ""
        for rec in rec_ids.split(","):
            rec = rec.strip()
            if not rec:
                continue
            stats[rec]["shown"] += 1
            # treat "tried" or "helpful" as success signal
            if feedback in ("tried", "helpful"):
                stats[rec]["success"] += 1
    return stats

def rank_recommendations(pool, history_rows):
    stats = compute_rec_success_rates(history_rows)
    def score_rec(r):
        s = stats.get(r["id"], {"shown": 0, "success": 0})
        shown = max(1, s["shown"])
        rate = s["success"] / shown
        # combine success_rate with slight exploration noise
        return rate + (0.15 * (1.0 - (1.0 / (1 + shown)))) + (0.05 * st.session_state.get("random_seed", 0))
    pool_sorted = sorted(pool, key=score_rec, reverse=True)
    return pool_sorted

# ------------------------------
# New: Wellness score (simple fusion)
# ------------------------------
def compute_wellness_score(sent_conf, emo_conf, rumination_score, state_conf):
    """
    Returns a normalized wellness score between 0 and 1 (higher = better).
    We invert some components since higher rumination -> worse.
    """
    # ensure inputs in 0..1
    sent_conf = float(min(max(sent_conf, 0.0), 1.0))
    emo_conf = float(min(max(emo_conf, 0.0), 1.0))
    rum = float(min(max(rumination_score, 0.0), 1.0))
    state_conf = float(min(max(state_conf, 0.0), 1.0))
    # formula: weighted combination (tunable)
    raw = (0.25 * sent_conf) + (0.25 * emo_conf) + (0.2 * (1 - rum)) + (0.3 * (1 - state_conf))
    # clamp
    score = max(0.0, min(1.0, raw))
    return round(score, 3)

# ------------------------------
# Page UI
# ------------------------------
st.title("🧠MindSense")
st.write("Type a short statement. The app will ask follow-ups, analyze your mental state, recommend safe micro-actions and track progress over time.")

# ------------------------------
# Initial statement form
# ------------------------------
if st.session_state.stage == "initial":
    with st.form("initial_form"):
        user_text = st.text_area("💬 Enter one statement about how you feel or what's happened:", height=140)
        submitted = st.form_submit_button("Start Follow-up")
    if submitted and user_text.strip():
        st.session_state.initial_text = user_text.strip()
        st.session_state.questions = generate_followup_questions(user_text)
        st.session_state.answers = ["" for _ in st.session_state.questions]
        st.session_state.q_index = 0
        st.session_state.stage = "questioning"
        safe_rerun()

# ------------------------------
# Questioning stage
# ------------------------------
if st.session_state.stage == "questioning":
    idx = st.session_state.q_index
    question = st.session_state.questions[idx]
    st.write(f"### Question {idx+1} of {len(st.session_state.questions)}")
    with st.form(f"qform_{idx}"):
        answer = st.text_area(question, value=st.session_state.answers[idx], height=120)
        next_btn = st.form_submit_button("Next")
        back_btn = st.form_submit_button("Back")
    if back_btn:
        st.session_state.answers[idx] = answer
        if idx > 0:
            st.session_state.q_index = idx - 1
        safe_rerun()
    if next_btn:
        st.session_state.answers[idx] = answer
        if idx + 1 < len(st.session_state.questions):
            st.session_state.q_index = idx + 1
            safe_rerun()
        else:
            # combine and analyze
            combined_text = st.session_state.initial_text + " " + " ".join(st.session_state.answers)
            cleaned = clean_text(combined_text)

            # ------------------------------
            # Crisis detection (must run BEFORE main analysis)
            # ------------------------------
            CRISIS_WORDS = [
                "kill myself", "end my life", "suicide", "want to die", "hurt myself"
            ]
            if any(word in cleaned.lower() for word in CRISIS_WORDS):
                st.error(
                    "You may be feeling extremely overwhelmed. Please reach out to someone immediately. "
                    "If you are in immediate danger, contact local emergency services now."
                )
                log_event(cleaned, "crisis", 1.0, [])
                st.session_state.stage = "result"
                st.session_state.results = {
                    "sentiment": "negative",
                    "emotion": "fear",
                    "sarcasm": "not_sarcastic",
                    "sentiment_confidence": 1.0,
                    "emotion_confidence": 1.0,
                    "sarcasm_confidence": 0.0,
                    "sentiment_scores": {}
                }
                safe_rerun()
            else:
                # ------------------------------
                # Main analysis (existing analyzer)
                # analyze_text_full provides sentiment + emotion + sarcasm (see sentiment.py).
                # If you later expand sentiment.py to return emotion_probs, prefer that.
                # ------------------------------
                try:
                    results = analyze_text_full(cleaned)
                except Exception as e:
                    results = {
                        "sentiment": "neutral",
                        "sentiment_confidence": 0.0,
                        "sentiment_scores": {},
                        "emotion": "joy",
                        "emotion_confidence": 0.0,
                        "sarcasm": "not_sarcastic",
                        "sarcasm_confidence": 0.0,
                        "error": f"analysis error: {e}"
                    }

                # Build an emotion_probs fallback (one-hot-ish) if not provided
                emotion_probs = results.get("emotion_probs")
                if not emotion_probs:
                    # fallback: take label and confidence -> place in prob dict
                    label = results.get("emotion", "joy")
                    conf = results.get("emotion_confidence", 0.0)
                    emotion_probs = {k: 0.0 for k in ["anger", "fear", "joy", "love", "sadness", "surprise"]}
                    if label in emotion_probs:
                        emotion_probs[label] = conf
                    else:
                        # assign to joy default
                        emotion_probs["joy"] = conf

                # ------------------------------
                # Mental state inference (uses your mental_state.py logic)
                # ------------------------------
                signals = {
                    "sentiment": results.get("sentiment"),
                    "emotion": results.get("emotion"),
                    "emotion_probs": emotion_probs,
                    "sarcasm": results.get("sarcasm"),
                    "text": cleaned
                }
                state, confidence = infer_mental_state(signals)

                # thinking-style + triggers
                thinking_style, style_scores = profile_thinking_style(cleaned)
                triggers_found = detect_triggers(cleaned)

                # ------------------------------
                # Adaptive recommendations: get pool, then rank based on history
                # ------------------------------
                base_pool = get_recommendations(state, top_k=10)  # ask pool (we'll rank & pick top 2)
                recent_rows = fetch_recent(500)  # compute stats on recent history
                ranked_pool = rank_recommendations(base_pool, recent_rows)
                recs = ranked_pool[:2] if ranked_pool else base_pool[:2]

                # log event and store recs
                log_event(cleaned, state, confidence, [r["id"] for r in recs])

                # store session info
                st.session_state.results = results
                st.session_state.mental_state = state
                st.session_state.mental_confidence = confidence
                st.session_state.recommendations = recs
                st.session_state.thinking_style = thinking_style
                st.session_state.triggers_found = triggers_found

                # update history (for quick in-memory trend chart)
                st.session_state.history.append({
                    "text": cleaned,
                    "sentiment": results.get("sentiment"),
                    "emotion": results.get("emotion"),
                    "state": state
                })
                st.session_state.stage = "result"
                safe_rerun()

# ------------------------------
# Results + therapist reply + dashboard
# ------------------------------
if st.session_state.stage == "result" and st.session_state.results:
    results = st.session_state.results

    st.subheader("🔍Final Analysis")
    st.markdown("**Sentiment:**")
    st.info(f"{results.get('sentiment','unknown').upper()} (Confidence: {results.get('sentiment_confidence',0):.2f})")

    st.markdown("**Emotion:**")
    st.warning(f"{results.get('emotion','unknown').upper()} (Confidence: {results.get('emotion_confidence',0):.2f})")

    st.markdown("**Sarcasm:**")
    emoji = "🙃" if results.get("sarcasm") == "sarcastic" else "🙂"
    st.error(f"{emoji} {results.get('sarcasm','unknown').upper()} (Confidence: {results.get('sarcasm_confidence',0):.2f})")

    st.markdown("**Raw Sentiment Scores (VADER):**")
    st.json(results.get("sentiment_scores", {}))

    # Transcript
    st.subheader("🗒 Transcript")
    st.write("**Initial statement:**")
    st.write(st.session_state.initial_text)
    st.write("**Follow-up Q&A:**")
    for i, q in enumerate(st.session_state.questions):
        st.write(f"**Q{i+1}.** {q}")
        st.write(f"**A{i+1}.** {st.session_state.answers[i] or '—'}")

    # Therapist reply
    try:
        bot_reply = generate_therapy_response(
            st.session_state.initial_text,
            results.get("sentiment"),
            results.get("emotion"),
            results.get("sarcasm"),
            insights
        )
    except TypeError:
        bot_reply = generate_therapy_response(
            st.session_state.initial_text,
            results.get("sentiment"),
            insights
        )
    st.subheader("🤖 Therapist Reply")
    st.success(bot_reply)

    # Thinking style & triggers
    st.subheader("🧠 Cognitive Profile")
    think_style = st.session_state.get("thinking_style", "unknown")
    st.info(f"Thinking style (detected): {think_style.replace('_',' ').title()}")

    st.subheader("⚠ Potential Triggers")
    tf = st.session_state.get("triggers_found", [])
    if tf:
        st.warning(", ".join(tf))
    else:
        st.write("No obvious triggers detected in this input.")

    # Recommendations (adaptive)
    st.subheader("🎯Recommendations")
    recs = st.session_state.get("recommendations", [])
    if recs:
        for r in recs:
            st.write(f"- {r['text']}")
            if st.button(f"👍 Tried: {r['id']}"):
                recent = fetch_recent(1)
                if recent:
                    event_id = recent[0][0]
                    record_feedback(event_id, "tried")
                st.success("Thanks — noted that you tried it.")

    # Wellness score + short progress metric
    sent_conf = results.get("sentiment_confidence", 0.0)
    emo_conf = results.get("emotion_confidence", 0.0)
    rum = score_rumination(clean_text(st.session_state.initial_text + " " + " ".join(st.session_state.answers)))
    state_conf = st.session_state.get("mental_confidence", 0.0)

    score = compute_wellness_score(sent_conf, emo_conf, rum, state_conf)
    st.metric("Overall Wellness Score", f"{score} / 1.0")

    # Progress chart (simple)
    if len(st.session_state.history) > 0:
        st.subheader("📈 Recent Mood Trend")
        df_hist = pd.DataFrame(st.session_state.history)
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        df_hist["sentiment_score"] = df_hist["sentiment"].map(sentiment_map).fillna(0)
        st.line_chart(df_hist["sentiment_score"])

    # Coping tip + followup question
    st.subheader("💡 Coping Tip")
    st.info(coping_tips(results.get("emotion")))

    st.subheader("❓ Follow-Up Prompt")
    st.write(followup_question(results.get("emotion")))

    # Small interactive breathing exercise
    if st.button("Start 10-second breathing exercise"):
        placeholder = st.empty()
        for i in range(10, 0, -1):
            placeholder.markdown(f"**Breathe... {i}**")
            time.sleep(1)
        placeholder.markdown("**Done — nice!**")

    # Sidebar: analytics from DB (longer-term)
    with st.sidebar:
        st.header("Dataset Insights")
        st.write(insights)
        if df is not None:
            if st.button("Show dataset visuals"):
                charts = plot_visuals(df)
                st.pyplot(charts["screen_vs_happiness"])
                st.pyplot(charts["stress_vs_sleep"])
                st.pyplot(charts["averages"])

        st.subheader("📚 Interaction Analytics (DB)")
        recent = fetch_recent(500)
        if recent:
            df_db = pd.DataFrame(recent, columns=["id","ts","text","state","confidence","rec_ids","feedback"])
            st.write("Events stored:", len(df_db))
            state_counts = df_db['state'].value_counts().to_dict()
            st.write("State counts:", state_counts)
            st.bar_chart(pd.Series(state_counts))
            # recommendation effectiveness
            stats = compute_rec_success_rates(recent)
            if stats:
                df_stats = pd.DataFrame([(k, v["shown"], v["success"], (v["success"]/max(1,v["shown"]))) for k,v in stats.items()],
                                        columns=["rec_id","shown","success","success_rate"])
                st.subheader("Recommendation stats (recent)")
                st.dataframe(df_stats.sort_values("success_rate", ascending=False).head(10))

    # Restart button
    if st.button("Start Over"):
        st.session_state.stage = "initial"
        st.session_state.initial_text = ""
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.q_index = 0
        st.session_state.results = None
        st.session_state.recommendations = []
        st.session_state.mental_state = None
        st.session_state.mental_confidence = 0.0
        st.session_state.history = []
        safe_rerun()
