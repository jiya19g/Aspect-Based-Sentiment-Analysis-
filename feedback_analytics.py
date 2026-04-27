import streamlit as st
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
import io
from collections import Counter

st.set_page_config(
    page_title="Student Feedback Analytics System",
    page_icon="📊",
    layout="wide"
)

# ========== SESSION STATE INITIALIZATION ==========

for key, default in [
    ('feedback_input', ''),
    ('bulk_analyzed', False),
    ('bulk_results', None),
    ('sample_df', None),
    ('bulk_df', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ========== NLP ENGINE ==========

SLANG_MAP = {
    "prof": "professor", "chill": "good", "kinda": "somewhat", "tbh": "",
    "ngl": "", "bruh": "", "lol": "", "trash": "terrible", "sucks": "bad",
    "awesome": "excellent", "amazing": "excellent", "horrible": "terrible",
    "useless": "bad", "brutal": "very bad", "lit": "excellent", "mid": "average",
    "meh": "average", "heck": "bad", "damn": "bad", "crazy": "intense",
}

ASPECTS = {
    "Teaching Quality": {
        "triggers": ["professor", "teacher", "instructor", "faculty", "lecture", "teaches",
                     "teaching", "explains", "prof"],
        "positive": ["excellent", "good", "great", "clear", "knowledgeable", "explains well",
                     "helpful", "nice", "chill", "great teaching"],
        "negative": ["poor", "bad", "confusing", "unclear", "boring", "terrible", "awful",
                     "horrible", "rushed", "very bad"],
        "strong_negative": ["terrible", "awful", "horrible", "very bad"],
    },
    "Assessment & Evaluation": {
        "triggers": ["assignment", "exam", "exams", "grade", "test", "quiz", "grading",
                     "assessment", "deadline", "hard", "difficult", "easy", "very bad"],
        "positive": ["fair", "reasonable", "clear", "easy", "manageable", "good", "great"],
        "negative": ["unfair", "hard", "difficult", "lengthy", "confusing", "very bad", "tough",
                     "too much", "overwhelming", "terrible"],
        "strong_negative": ["unfair", "very bad", "terrible"],
    },
    "Practical & Lab": {
        "triggers": ["lab", "labs", "practical", "hands-on", "equipment", "workshop",
                     "dirty", "hectic", "messy", "disorganized", "clean"],
        "positive": ["good", "great", "organized", "helpful", "clean", "functional",
                     "well-equipped", "useful"],
        "negative": ["bad", "poor", "dirty", "hectic", "messy", "disorganized", "broken",
                     "outdated", "useless", "terrible", "awful", "horrible"],
        "strong_negative": ["terrible", "awful", "horrible", "useless", "dirty"],
    },
    "Mentoring & Support": {
        "triggers": ["ta", "tas", "mentor", "support", "guidance", "office hours",
                     "tutor", "helpful tas"],
        "positive": ["helpful", "supportive", "available", "responsive", "approachable",
                     "friendly", "good", "great"],
        "negative": ["unhelpful", "unavailable", "unresponsive", "poor", "bad", "terrible"],
        "strong_negative": ["unhelpful", "unavailable", "terrible"],
    },
    "Course Design & Workload": {
        "triggers": ["workload", "heavy", "deadline", "pace", "structure", "organization",
                     "syllabus", "overwhelming", "manageable"],
        "positive": ["manageable", "balanced", "reasonable", "well-structured", "good", "great"],
        "negative": ["overwhelming", "heavy", "tight", "rushed", "excessive", "too much", "very bad"],
        "strong_negative": ["overwhelming", "excessive", "very bad"],
    },
    "Teaching Methodology": {
        "triggers": ["teaching style", "slides", "presentation", "engagement", "delivery",
                     "interactive", "boring", "engaging"],
        "positive": ["interactive", "engaging", "clear slides", "good examples", "dynamic", "effective"],
        "negative": ["boring", "monotonous", "reads slides", "unengaging", "passive", "dull"],
        "strong_negative": ["boring", "monotonous", "unengaging"],
    },
    "Learning Outcomes": {
        "triggers": ["learned", "learning", "skills", "knowledge", "understanding", "useful",
                     "valuable", "relevant"],
        "positive": ["learned a lot", "useful", "valuable", "practical", "relevant", "good", "great"],
        "negative": ["waste", "nothing", "useless", "irrelevant", "pointless"],
        "strong_negative": ["useless", "irrelevant", "pointless", "waste"],
    },
}

NEGATION_PATTERN = re.compile(r'\b(not|never|no|doesn\'?t|isn\'?t|wasn\'?t|won\'?t|can\'?t|hardly)\s+')


def normalise_text(text: str) -> str:
    t = text.lower().strip()
    for slang, replacement in SLANG_MAP.items():
        t = re.sub(r'\b' + re.escape(slang) + r'\b', replacement, t)
    return t


def split_into_clauses(text: str) -> list:
    clauses = re.split(r'[.!?;]\s+', text)
    final_clauses = []
    for clause in clauses:
        sub = re.split(r'\s+but\s+|\s+however\s+|\s+although\s+', clause)
        final_clauses.extend([c.strip() for c in sub if c.strip()])
    return final_clauses if final_clauses else [text.strip()]


def score_sentiment(clause: str, pos_words: list, neg_words: list,
                    strong_neg_words: list) -> tuple:
    """
    Returns (pos_score, neg_score).
    Strong negative words count double.
    Negation reduces positive signal and does not blindly amplify negative.
    """
    pos_score = 0
    neg_score = 0

    for w in pos_words:
        if re.search(r'\b' + re.escape(w) + r'\b', clause):
            pos_score += 1

    for w in neg_words:
        if re.search(r'\b' + re.escape(w) + r'\b', clause):
            weight = 2 if w in strong_neg_words else 1
            neg_score += weight

    if NEGATION_PATTERN.search(clause):
        # Negation weakens positive signals significantly
        pos_score *= 0.2
        # Avoid blindly flipping — only bump neg if there was a positive to negate
        if pos_score > 0:
            neg_score = max(neg_score, 0.5)

    return pos_score, neg_score


def analyze_all_aspects(text: str) -> dict:
    if not text or len(text.strip()) < 3:
        return {}

    norm_text = normalise_text(text)
    clauses = split_into_clauses(norm_text)

    aspect_scores = {}

    for clause in clauses:
        if len(clause.split()) < 2:
            continue

        triggered_aspects = []
        for aspect, config in ASPECTS.items():
            if any(trigger in clause for trigger in config["triggers"]):
                triggered_aspects.append(aspect)

        if not triggered_aspects:
            for aspect, config in ASPECTS.items():
                pos_check = any(word in clause for word in config["positive"])
                neg_check = any(word in clause for word in config["negative"])
                if pos_check or neg_check:
                    triggered_aspects.append(aspect)
                    break

        for aspect in triggered_aspects:
            config = ASPECTS[aspect]
            strong_neg = config.get("strong_negative", [])
            pos_score, neg_score = score_sentiment(
                clause, config["positive"], config["negative"], strong_neg
            )

            if neg_score > 0 and neg_score >= pos_score:
                sentiment = "Negative"
                confidence = min(0.92, 0.75 + neg_score * 0.08)
                score_val = max(10, 40 - neg_score * 7)
            elif pos_score > 0 and pos_score > neg_score:
                sentiment = "Positive"
                confidence = min(0.92, 0.75 + pos_score * 0.08)
                score_val = min(95, 65 + pos_score * 8)
            else:
                sentiment = "Neutral"
                confidence = 0.65
                score_val = 50

            if aspect not in aspect_scores:
                aspect_scores[aspect] = {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "score": score_val,
                    "count": 1,
                }
            else:
                aspect_scores[aspect]["count"] += 1
                aspect_scores[aspect]["confidence"] = (
                    aspect_scores[aspect]["confidence"] + confidence
                ) / 2
                aspect_scores[aspect]["score"] = (
                    aspect_scores[aspect]["score"] + score_val
                ) / 2
                if sentiment != aspect_scores[aspect]["sentiment"]:
                    aspect_scores[aspect]["sentiment"] = "Neutral"

    results = {
        aspect: {
            "sentiment": data["sentiment"],
            "confidence": round(data["confidence"], 3),
            "score": int(data["score"]),
        }
        for aspect, data in aspect_scores.items()
    }

    # Fallback for short unmatched feedback
    if not results and len(norm_text.split()) > 3:
        pos_words = ["good", "great", "excellent", "nice", "helpful", "useful", "chill"]
        neg_words = ["bad", "poor", "terrible", "awful", "horrible", "useless", "very bad"]
        pos = sum(1 for w in pos_words if w in norm_text)
        neg = sum(1 for w in neg_words if w in norm_text)

        if pos > neg:
            results["General Feedback"] = {"sentiment": "Positive", "confidence": 0.75, "score": 70}
        elif neg > pos:
            results["General Feedback"] = {"sentiment": "Negative", "confidence": 0.75, "score": 30}
        elif pos > 0 or neg > 0:
            results["General Feedback"] = {"sentiment": "Neutral", "confidence": 0.65, "score": 50}

    return results


def generate_insights(results: dict) -> dict:
    insights = {"strengths": [], "weaknesses": [], "neutral": [], "recommendations": []}
    for aspect, data in results.items():
        if data["sentiment"] == "Positive":
            insights["strengths"].append(aspect)
            insights["recommendations"].append(f"Maintain current standards for {aspect}")
        elif data["sentiment"] == "Negative":
            insights["weaknesses"].append(aspect)
            insights["recommendations"].append(f"Address issues with {aspect}")
        else:
            insights["neutral"].append(aspect)
    return insights


def calculate_overall_score(results: dict) -> int | None:
    """Returns None when there are no results, so callers can distinguish no-data from score=50."""
    if not results:
        return None
    scores = [data["score"] for data in results.values()]
    return sum(scores) // len(scores)


# ========== BULK ANALYSIS ==========

@st.cache_data(show_spinner=False)
def analyze_bulk_feedback(df_json: str, text_column: str = "feedback") -> list:
    """Cached bulk analysis. Accepts JSON-serialised DataFrame to satisfy cache hashing."""
    df = pd.read_json(io.StringIO(df_json))
    results_list = []
    for idx, row in df.iterrows():
        feedback_text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        if len(feedback_text.strip()) <= 10:
            continue

        aspect_results = analyze_all_aspects(feedback_text)
        overall_score = calculate_overall_score(aspect_results)
        if overall_score is None:
            overall_score = 50

        pos_count = sum(1 for r in aspect_results.values() if r["sentiment"] == "Positive")
        neg_count = sum(1 for r in aspect_results.values() if r["sentiment"] == "Negative")
        neutral_count = sum(1 for r in aspect_results.values() if r["sentiment"] == "Neutral")

        # Start from original row to avoid column name collisions
        row_data = row.to_dict()
        row_data.update({
            "feedback_id": idx,
            "feedback_text": feedback_text[:200] + ("..." if len(feedback_text) > 200 else ""),
            "full_text": feedback_text,
            "aspects_detected": len(aspect_results),
            "positive_aspects": pos_count,
            "negative_aspects": neg_count,
            "neutral_aspects": neutral_count,
            "overall_score": overall_score,
            "aspect_details": aspect_results,
        })
        results_list.append(row_data)
    return results_list


def generate_bulk_insights(bulk_results: list) -> dict:
    total = len(bulk_results)
    if total == 0:
        return {}

    aspect_aggregator = {}
    all_scores = []

    for result in bulk_results:
        all_scores.append(result["overall_score"])
        for aspect, data in result["aspect_details"].items():
            if aspect not in aspect_aggregator:
                aspect_aggregator[aspect] = {
                    "positive": 0, "negative": 0, "neutral": 0,
                    "total_score": 0, "count": 0,
                }
            aspect_aggregator[aspect][data["sentiment"].lower()] += 1
            aspect_aggregator[aspect]["total_score"] += data["score"]
            aspect_aggregator[aspect]["count"] += 1

    for aspect in aspect_aggregator:
        count = aspect_aggregator[aspect]["count"]
        aspect_aggregator[aspect]["avg_score"] = (
            aspect_aggregator[aspect]["total_score"] / count if count > 0 else 0
        )

    avg_score = sum(all_scores) / total
    score_std = pd.Series(all_scores).std() if len(all_scores) > 1 else 0.0

    feedback_sentiments = []
    for result in bulk_results:
        pos = result["positive_aspects"]
        neg = result["negative_aspects"]
        feedback_sentiments.append(
            "Positive" if pos > neg else ("Negative" if neg > pos else "Neutral")
        )
    sentiment_counts = Counter(feedback_sentiments)

    priority_matrix = sorted(
        [
            {
                "aspect": aspect,
                "negative_percentage": (agg["negative"] / total) * 100,
                "avg_score": agg["avg_score"],
            }
            for aspect, agg in aspect_aggregator.items()
        ],
        key=lambda x: x["negative_percentage"],
        reverse=True,
    )

    stop_words = {
        "the", "a", "an", "and", "or", "but", "so", "for", "of", "to", "in",
        "on", "at", "with", "by", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "this", "that", "these", "those", "it", "they",
        "we", "you", "he", "she",
    }
    all_words = []
    for result in bulk_results:
        words = result["full_text"].lower().split()
        all_words.extend(w for w in words if len(w) > 3 and w not in stop_words)
    word_freq = Counter(all_words).most_common(20)

    positive_feedbacks = sorted(
        [r for r in bulk_results if r["positive_aspects"] > r["negative_aspects"]],
        key=lambda x: x["overall_score"],
        reverse=True,
    )[:3]
    negative_feedbacks = sorted(
        [r for r in bulk_results if r["negative_aspects"] > r["positive_aspects"]],
        key=lambda x: x["overall_score"],
    )[:3]

    return {
        "total_feedbacks": total,
        "avg_overall_score": avg_score,
        "score_std": score_std,
        "total_positive_aspects": sum(r["positive_aspects"] for r in bulk_results),
        "total_negative_aspects": sum(r["negative_aspects"] for r in bulk_results),
        "sentiment_distribution": sentiment_counts,
        "aspect_aggregator": aspect_aggregator,
        "priority_matrix": priority_matrix,
        "word_frequency": word_freq,
        "top_positive_feedbacks": positive_feedbacks,
        "top_negative_feedbacks": negative_feedbacks,
        "bulk_results": bulk_results,
    }


def filter_bulk_results(bulk_results: list, df: pd.DataFrame,
                        faculty: str | None, course: str | None,
                        semester: str | None) -> list:
    filtered = bulk_results
    if faculty and faculty != "All":
        filtered = [r for r in filtered if r.get("faculty") == faculty]
    if course and course != "All":
        filtered = [r for r in filtered if r.get("course") == course]
    if semester and semester != "All":
        filtered = [r for r in filtered if r.get("semester") == semester]
    return filtered


# ========== SAMPLE DATA ==========

def get_sample_csv() -> pd.DataFrame:
    return pd.DataFrame({
        "feedback": [
            "Professor Smith explains concepts very clearly. The assignments are fair and workload is manageable.",
            "Dr. Johnson's teaching style is engaging. However, the exams are too difficult and grading is harsh.",
            "Professor Williams is very knowledgeable but the lab sessions are disorganized. TAs are helpful.",
            "Dr. Brown's course structure is excellent. The mentor was very supportive throughout the semester.",
            "Professor Davis is good but sometimes rushed. Assignments are fair but deadlines are too tight.",
            "Dr. Wilson's labs are well-organized. The professor could be more engaging though.",
            "Professor Miller: Worst course ever. Unfair grading and useless labs. No support from faculty.",
            "Dr. Taylor's course content is relevant and practical. Workload is manageable. Would recommend!",
            "Professor Anderson is knowledgeable but the exam was too difficult. Lab sessions were helpful.",
            "Dr. Thomas: Great support from TAs but the course structure needs improvement. Too many deadlines.",
            "Professor Martinez explains well but the workload is overwhelming. Labs are okay.",
            "Dr. Garcia is an amazing teacher. Very helpful and always available. Best course ever!",
            "Professor Rodriguez: Poor teaching quality. Confusing lectures. Will not recommend.",
            "Dr. Lee's practical sessions are very useful. The assignments are challenging but fair.",
        ],
        "faculty": [
            "Prof. Smith", "Dr. Johnson", "Prof. Williams", "Dr. Brown", "Prof. Davis",
            "Dr. Wilson", "Prof. Miller", "Dr. Taylor", "Prof. Anderson", "Dr. Thomas",
            "Prof. Martinez", "Dr. Garcia", "Prof. Rodriguez", "Dr. Lee",
        ],
        "course": [
            "CS101", "CS201", "CS101", "CS301", "CS201",
            "CS101", "CS301", "CS201", "CS101", "CS301",
            "CS201", "CS101", "CS301", "CS201",
        ],
        "semester": [
            "Fall 2024", "Fall 2024", "Spring 2024", "Spring 2024", "Fall 2024",
            "Spring 2024", "Fall 2024", "Spring 2024", "Fall 2024", "Spring 2024",
            "Fall 2024", "Spring 2024", "Fall 2024", "Spring 2024",
        ],
    })


# ========== CUSTOM CSS ==========

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    h1 { font-size: 1.75rem; font-weight: 600; }
    .subheader { font-size: 0.85rem; margin-bottom: 1.5rem;
                 border-bottom: 1px solid rgba(148,163,184,0.3); padding-bottom: 0.75rem; }
    .metric-card { border: 1px solid rgba(148,163,184,0.3); border-radius: 0.5rem;
                   padding: 1rem; text-align: center; }
    .metric-value { font-size: 1.75rem; font-weight: 600; }
    .metric-label { font-size: 0.7rem; text-transform: uppercase; opacity: 0.6; }
    .aspect-card { border: 1px solid rgba(148,163,184,0.3); border-radius: 0.5rem;
                   padding: 0.75rem; margin: 0.5rem 0; }
    .aspect-title { font-weight: 600; }
    .sentiment-badge-positive { background: #dcfce7; color: #14532d; padding: 0.2rem 0.6rem;
                                border-radius: 9999px; font-size: 0.7rem; font-weight: 600; }
    .sentiment-badge-negative { background: #fee2e2; color: #7f1d1d; padding: 0.2rem 0.6rem;
                                border-radius: 9999px; font-size: 0.7rem; font-weight: 600; }
    .sentiment-badge-neutral  { background: #fef3c7; color: #78350f; padding: 0.2rem 0.6rem;
                                border-radius: 9999px; font-size: 0.7rem; font-weight: 600; }
    .progress-container { background: rgba(148,163,184,0.2); border-radius: 9999px;
                          height: 0.25rem; margin-top: 0.5rem; }
    .progress-bar-positive { background: #15803d; height: 100%; border-radius: 9999px; }
    .progress-bar-negative { background: #b91c1c; height: 100%; border-radius: 9999px; }
    .progress-bar-neutral   { background: #d97706; height: 100%; border-radius: 9999px; }
    .insight-positive { background: rgba(34,197,94,0.12); border-left: 3px solid #16a34a;
                        color: #14532d; padding: 0.6rem 1rem; margin: 0.4rem 0; border-radius: 0; }
    .insight-negative { background: rgba(239,68,68,0.12); border-left: 3px solid #dc2626;
                        color: #7f1d1d; padding: 0.6rem 1rem; margin: 0.4rem 0; border-radius: 0; }
    .insight-neutral  { background: rgba(234,179,8,0.12); border-left: 3px solid #d97706;
                        color: #78350f; padding: 0.6rem 1rem; margin: 0.4rem 0; border-radius: 0; }
    @media (prefers-color-scheme: dark) {
        .insight-positive { color: #86efac; }
        .insight-negative { color: #fca5a5; }
        .insight-neutral  { color: #fcd34d; }
        .sentiment-badge-positive { background: rgba(34,197,94,0.2); color: #86efac; }
        .sentiment-badge-negative { background: rgba(239,68,68,0.2); color: #fca5a5; }
        .sentiment-badge-neutral  { background: rgba(234,179,8,0.2); color: #fcd34d; }
    }
    .stButton button { background: #2563eb; color: white; border: none;
                       border-radius: 0.375rem; font-weight: 500; }
    .stButton button:hover { background: #1e40af; }
    .footer { text-align: center; opacity: 0.5; font-size: 0.7rem; margin-top: 2rem;
              padding-top: 1rem; border-top: 1px solid rgba(148,163,184,0.3); }
    .filter-box { background: rgba(148,163,184,0.08); padding: 1rem; border-radius: 0.5rem;
                  margin-bottom: 1rem; border: 1px solid rgba(148,163,184,0.2); }
</style>
""", unsafe_allow_html=True)

# ========== TABS ==========

tab1, tab2 = st.tabs(["📝 Single Feedback Analysis", "📊 Bulk Feedback Analysis"])

# ========== TAB 1: SINGLE FEEDBACK ==========

with tab1:
    st.markdown("<h1>Student Feedback Analytics System</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="subheader">Multi-aspect sentiment analysis with contrast detection | '
        "Enhanced Rule-based Engine</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Sample Test Cases")
        test_cases = {
            "Mixed Sentiment": "The prof is chill but exams are brutal and labs are kinda useless",
            "Teaching + Labs + TAs": "Great teaching, terrible labs, helpful TAs",
            "Professor + Assignments": "The professor explains well but assignments are too hard",
        }
        for name, text in test_cases.items():
            if st.button(name, use_container_width=True):
                st.session_state["feedback_input"] = text
                st.rerun()

        st.markdown("---")
        st.markdown("**Dimensions Analyzed**")
        for aspect in ASPECTS:
            st.markdown(f"- {aspect}")

    feedback = st.text_area(
        "Feedback Text",
        value=st.session_state["feedback_input"],
        height=120,
        placeholder="Enter student feedback here...",
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze = st.button("Run Analysis", type="primary", use_container_width=True)

    if analyze and feedback.strip():
        with st.spinner("Analyzing feedback..."):
            results = analyze_all_aspects(feedback)
            insights = generate_insights(results)
            overall_score = calculate_overall_score(results)

        if results:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{len(results)}</div>'
                    '<div class="metric-label">ASPECTS</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                pos = sum(1 for r in results.values() if r["sentiment"] == "Positive")
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value" style="color:#15803d;">'
                    f'{pos}</div><div class="metric-label">POSITIVE</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                neg = sum(1 for r in results.values() if r["sentiment"] == "Negative")
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value" style="color:#b91c1c;">'
                    f'{neg}</div><div class="metric-label">NEGATIVE</div></div>',
                    unsafe_allow_html=True,
                )
            with c4:
                score_display = overall_score if overall_score is not None else "—"
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{score_display}'
                    '<span style="font-size:0.8rem;">/100</span></div>'
                    '<div class="metric-label">SCORE</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("#### Aspect-wise Analysis")

            for aspect, data in results.items():
                badge_class = f"sentiment-badge-{data['sentiment'].lower()}"
                bar_class = f"progress-bar-{data['sentiment'].lower()}"
                st.markdown(
                    f"""
                    <div class="aspect-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="aspect-title">{aspect}</span>
                            <span class="{badge_class}">{data['sentiment']}</span>
                        </div>
                        <div style="font-size:0.75rem; color:#64748b;">
                            Confidence: {data['confidence']:.1%}
                        </div>
                        <div class="progress-container">
                            <div class="{bar_class}" style="width:{data['confidence']*100}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("#### Key Insights")

            if insights["strengths"]:
                st.markdown(
                    f'<div class="insight-positive"><strong>✅ Strengths:</strong> '
                    f'{", ".join(insights["strengths"])}</div>',
                    unsafe_allow_html=True,
                )
            if insights["weaknesses"]:
                st.markdown(
                    f'<div class="insight-negative"><strong>⚠️ Needs Improvement:</strong> '
                    f'{", ".join(insights["weaknesses"])}</div>',
                    unsafe_allow_html=True,
                )
            if insights["neutral"]:
                st.markdown(
                    f'<div class="insight-neutral"><strong>➖ Neutral:</strong> '
                    f'{", ".join(insights["neutral"])}</div>',
                    unsafe_allow_html=True,
                )
    elif analyze:
        st.warning("Please enter feedback.")

# ========== TAB 2: BULK FEEDBACK ==========

with tab2:
    st.markdown("## Bulk Feedback Analysis")
    st.markdown(
        "Upload a CSV file with columns: **feedback**, **faculty**, **course**, **semester** (optional)"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="CSV must contain a 'feedback' column. Optional: 'faculty', 'course', 'semester'",
        )
    with col2:
        if st.button("📋 Load Sample Dataset", use_container_width=True):
            st.session_state["sample_df"] = get_sample_csv()
            st.session_state["bulk_analyzed"] = False
            st.session_state["bulk_results"] = None

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="warn")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="latin-1", on_bad_lines="warn")

        if "feedback" not in df.columns:
            st.error("CSV must contain a 'feedback' column.")
        else:
            st.success(f"✅ Loaded {len(df)} feedback entries")
            st.dataframe(df.head(5), use_container_width=True)

            if st.button("📊 Analyze Bulk Feedback", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing {len(df)} feedback entries..."):
                    bulk_results = analyze_bulk_feedback(df.to_json(), "feedback")
                    st.session_state["bulk_results"] = bulk_results
                    st.session_state["bulk_df"] = df
                    st.session_state["bulk_analyzed"] = True

    elif st.session_state["sample_df"] is not None:
        df = st.session_state["sample_df"]
        st.info("📋 Sample Dataset Loaded (14 feedback entries with faculty, course, and semester data)")
        st.dataframe(df, use_container_width=True)

        if st.button("📊 Analyze Sample Data", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {len(df)} feedback entries..."):
                bulk_results = analyze_bulk_feedback(df.to_json(), "feedback")
                st.session_state["bulk_results"] = bulk_results
                st.session_state["bulk_df"] = df
                st.session_state["bulk_analyzed"] = True

    # ---- Results ----
    if st.session_state.get("bulk_analyzed") and st.session_state.get("bulk_results"):
        bulk_results = st.session_state["bulk_results"]
        df = st.session_state["bulk_df"]

        faculties = list(df["faculty"].unique()) if "faculty" in df.columns else []
        courses = list(df["course"].unique()) if "course" in df.columns else []
        semesters = list(df["semester"].unique()) if "semester" in df.columns else []

        st.markdown("---")
        st.markdown("### 🔍 Filter Analysis")

        with st.container():
            st.markdown('<div class="filter-box">', unsafe_allow_html=True)
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                selected_faculty = st.selectbox("Filter by Faculty", ["All"] + faculties) if faculties else None
            with fc2:
                selected_course = st.selectbox("Filter by Course", ["All"] + courses) if courses else None
            with fc3:
                selected_semester = st.selectbox("Filter by Semester", ["All"] + semesters) if semesters else None
            st.markdown("</div>", unsafe_allow_html=True)

        filtered_results = filter_bulk_results(
            bulk_results, df, selected_faculty, selected_course, selected_semester
        )
        insights = generate_bulk_insights(filtered_results)

        if insights.get("total_feedbacks", 0) == 0:
            st.warning("No results found for the selected filters.")
        else:
            # Executive Dashboard
            st.markdown("---")
            st.markdown("## 📊 Executive Dashboard")

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">'
                    f'{insights["total_feedbacks"]}</div>'
                    '<div class="metric-label">FEEDBACKS</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value" style="color:#15803d;">'
                    f'{insights["total_positive_aspects"]}</div>'
                    '<div class="metric-label">POSITIVE MENTIONS</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value" style="color:#b91c1c;">'
                    f'{insights["total_negative_aspects"]}</div>'
                    '<div class="metric-label">NEGATIVE MENTIONS</div></div>',
                    unsafe_allow_html=True,
                )
            with m4:
                ratio = insights["total_positive_aspects"] / max(insights["total_negative_aspects"], 1)
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{ratio:.1f}x</div>'
                    '<div class="metric-label">POS/NEG RATIO</div></div>',
                    unsafe_allow_html=True,
                )
            with m5:
                avg = insights["avg_overall_score"]
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{avg:.0f}'
                    '<span style="font-size:0.8rem;">/100</span></div>'
                    '<div class="metric-label">AVG SCORE</div></div>',
                    unsafe_allow_html=True,
                )

            # Satisfaction gauge
            st.markdown("---")
            st.markdown("#### Average Satisfaction Score")
            avg = insights["avg_overall_score"]
            gauge_color = "#15803d" if avg >= 70 else ("#92400e" if avg >= 50 else "#b91c1c")
            st.markdown(
                f"""
                <div style="background:#e2e8f0; border-radius:9999px; height:0.5rem; overflow:hidden;">
                    <div style="background:{gauge_color}; width:{avg}%; height:0.5rem;
                                border-radius:9999px;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:0.25rem;">
                    <span style="font-size:0.7rem;">Needs Improvement (0–40)</span>
                    <span style="font-size:0.7rem;">Moderate (40–60)</span>
                    <span style="font-size:0.7rem;">Good (60–80)</span>
                    <span style="font-size:0.7rem;">Excellent (80–100)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Sentiment + Score distribution
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Sentiment Distribution")
                sentiment_df = pd.DataFrame({
                    "Sentiment": list(insights["sentiment_distribution"].keys()),
                    "Count": list(insights["sentiment_distribution"].values()),
                })
                fig = px.pie(
                    sentiment_df, values="Count", names="Sentiment",
                    color="Sentiment",
                    color_discrete_map={"Positive": "#15803d", "Negative": "#b91c1c",
                                        "Neutral": "#92400e"},
                    hole=0.4,
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Score Distribution")
                scores = [r["overall_score"] for r in insights["bulk_results"]]
                fig = px.histogram(
                    x=scores, nbins=20,
                    labels={"x": "Satisfaction Score", "y": "Frequency"},
                    color_discrete_sequence=["#2563eb"],
                )
                fig.add_vline(
                    x=avg, line_dash="dash", line_color="red",
                    annotation_text=f"Avg: {avg:.0f}",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Aspect sentiment stacked bar
            if insights["aspect_aggregator"]:
                st.markdown("---")
                st.markdown("#### Sentiment Distribution by Dimension")

                aspect_data = [
                    {
                        "Dimension": aspect,
                        "Positive": agg["positive"],
                        "Negative": agg["negative"],
                        "Neutral": agg["neutral"],
                        "Avg Score": round(agg["avg_score"], 1),
                    }
                    for aspect, agg in insights["aspect_aggregator"].items()
                ]
                aspect_df = pd.DataFrame(aspect_data)

                fig = go.Figure(data=[
                    go.Bar(name="Positive", x=aspect_df["Dimension"],
                           y=aspect_df["Positive"], marker_color="#15803d"),
                    go.Bar(name="Negative", x=aspect_df["Dimension"],
                           y=aspect_df["Negative"], marker_color="#b91c1c"),
                    go.Bar(name="Neutral", x=aspect_df["Dimension"],
                           y=aspect_df["Neutral"], marker_color="#92400e"),
                ])
                fig.update_layout(barmode="stack", height=450, title="Mentions by Dimension")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Dimension Performance Scores")
                st.dataframe(
                    aspect_df.sort_values("Avg Score", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

            # Faculty-wise
            if "faculty" in df.columns:
                st.markdown("---")
                st.markdown("#### 👨‍🏫 Faculty-wise Analysis")

                faculty_stats: dict = {}
                for result in bulk_results:
                    faculty = result.get("faculty", "Unknown")
                    if faculty not in faculty_stats:
                        faculty_stats[faculty] = {
                            "scores": [], "positive": 0, "negative": 0, "neutral": 0
                        }
                    faculty_stats[faculty]["scores"].append(result["overall_score"])
                    faculty_stats[faculty]["positive"] += result["positive_aspects"]
                    faculty_stats[faculty]["negative"] += result["negative_aspects"]
                    faculty_stats[faculty]["neutral"] += result["neutral_aspects"]

                faculty_data = []
                for faculty, stats in faculty_stats.items():
                    avg_s = sum(stats["scores"]) / len(stats["scores"])
                    total = stats["positive"] + stats["negative"] + stats["neutral"]
                    faculty_data.append({
                        "Faculty": faculty,
                        "Feedbacks": len(stats["scores"]),
                        "Avg Score": round(avg_s, 1),
                        "Positive %": round(stats["positive"] / total * 100, 1) if total else 0,
                        "Negative %": round(stats["negative"] / total * 100, 1) if total else 0,
                        "Total Mentions": total,
                    })
                faculty_df = pd.DataFrame(faculty_data).sort_values("Avg Score", ascending=False)
                fig = px.bar(
                    faculty_df, x="Faculty", y="Avg Score",
                    title="Satisfaction Score by Faculty",
                    color="Avg Score", color_continuous_scale="RdYlGn", text="Avg Score",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(faculty_df, use_container_width=True, hide_index=True)

            # Course-wise
            if "course" in df.columns:
                st.markdown("---")
                st.markdown("#### 📚 Course-wise Analysis")

                course_stats: dict = {}
                for result in bulk_results:
                    course = result.get("course", "Unknown")
                    if course not in course_stats:
                        course_stats[course] = {"scores": [], "positive": 0, "negative": 0}
                    course_stats[course]["scores"].append(result["overall_score"])
                    course_stats[course]["positive"] += result["positive_aspects"]
                    course_stats[course]["negative"] += result["negative_aspects"]

                course_data = []
                for course, stats in course_stats.items():
                    avg_s = sum(stats["scores"]) / len(stats["scores"])
                    total = stats["positive"] + stats["negative"]
                    course_data.append({
                        "Course": course,
                        "Feedbacks": len(stats["scores"]),
                        "Avg Score": round(avg_s, 1),
                        "Positive %": round(stats["positive"] / total * 100, 1) if total else 0,
                        "Negative Count": stats["negative"],
                    })
                course_df = pd.DataFrame(course_data).sort_values("Avg Score", ascending=False)
                fig = px.bar(
                    course_df, x="Course", y="Avg Score",
                    title="Satisfaction Score by Course",
                    color="Avg Score", color_continuous_scale="RdYlGn", text="Avg Score",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(course_df, use_container_width=True, hide_index=True)

            # Priority Matrix
            st.markdown("---")
            st.markdown("#### 🎯 Improvement Priority Matrix")
            for item in insights["priority_matrix"][:7]:
                pct = item["negative_percentage"]
                avg_s = item["avg_score"]
                label = item["aspect"]
                if pct > 30:
                    st.markdown(
                        f'<div class="insight-negative"><strong>🔴 HIGH PRIORITY:</strong> '
                        f'{label} — {pct:.1f}% negative feedback (Avg Score: {avg_s:.0f})</div>',
                        unsafe_allow_html=True,
                    )
                elif pct > 15:
                    st.markdown(
                        f'<div class="insight-neutral"><strong>🟡 MEDIUM PRIORITY:</strong> '
                        f'{label} — {pct:.1f}% negative feedback (Avg Score: {avg_s:.0f})</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="insight-positive"><strong>🟢 LOW PRIORITY:</strong> '
                        f'{label} — {pct:.1f}% negative feedback (Avg Score: {avg_s:.0f})</div>',
                        unsafe_allow_html=True,
                    )

            # Keywords
            st.markdown("---")
            st.markdown("#### 🔤 Key Themes from Feedback")
            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                keyword_df = pd.DataFrame(insights["word_frequency"][:10],
                                          columns=["Keyword", "Frequency"])
                st.dataframe(keyword_df, use_container_width=True, hide_index=True)
            with kw_col2:
                keywords = [k for k, _ in insights["word_frequency"][:10]]
                freqs = [v for _, v in insights["word_frequency"][:10]]
                fig = px.bar(
                    x=freqs, y=keywords, orientation="h",
                    labels={"x": "Frequency", "y": ""},
                    color=freqs, color_continuous_scale="Blues",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Representative examples
            st.markdown("---")
            st.markdown("#### 📝 Representative Feedback Examples")
            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                st.markdown("**📈 Best Feedback Examples**")
                for fb in insights["top_positive_feedbacks"]:
                    fac = f" ({fb.get('faculty', 'Unknown')})" if "faculty" in fb else ""
                    with st.expander(f"Score: {fb['overall_score']}/100{fac}"):
                        st.write(fb["full_text"])
            with ex_col2:
                st.markdown("**📉 Needs Improvement Examples**")
                for fb in insights["top_negative_feedbacks"]:
                    fac = f" ({fb.get('faculty', 'Unknown')})" if "faculty" in fb else ""
                    with st.expander(f"Score: {fb['overall_score']}/100{fac}"):
                        st.write(fb["full_text"])

            # Detailed table
            st.markdown("---")
            st.markdown("#### 📋 Detailed Analysis Results")
            results_df = pd.DataFrame([
                {
                    "ID": r["feedback_id"],
                    "Feedback": r["feedback_text"],
                    "Faculty": r.get("faculty", "—"),
                    "Course": r.get("course", "—"),
                    "Aspects": r["aspects_detected"],
                    "Positive": r["positive_aspects"],
                    "Negative": r["negative_aspects"],
                    "Score": r["overall_score"],
                }
                for r in insights["bulk_results"]
            ])
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Export
            st.markdown("---")
            e1, e2, e3 = st.columns([1, 2, 1])
            with e2:
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Export Results to CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"bulk_analysis_results_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            if st.button("Clear Results", use_container_width=True):
                st.session_state["bulk_analyzed"] = False
                st.session_state["bulk_results"] = None
                st.rerun()

st.markdown(
    '<div class="footer">Student Feedback Analytics System | '
    "Multi-aspect Sentiment Analysis | Faculty & Course-wise Analytics</div>",
    unsafe_allow_html=True,
)