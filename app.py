import streamlit as st
import re
from datetime import datetime

st.set_page_config(
    page_title="Student Feedback Analytics System", 
    page_icon="📊", 
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    :root {
        --primary: #2563eb;
        --primary-dark: #1e40af;
        --success: #059669;
        --danger: #dc2626;
        --warning: #d97706;
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    h1 {
        font-size: 1.875rem;
        font-weight: 600;
        color: var(--gray-800);
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    
    .subheader {
        color: var(--gray-600);
        font-size: 0.875rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--gray-200);
        padding-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        border: 1px solid var(--gray-200);
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 1.875rem;
        font-weight: 600;
        color: var(--gray-800);
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--gray-600);
        margin-top: 0.25rem;
    }
    
    .aspect-card {
        border: 1px solid var(--gray-200);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.75rem 0;
        background: white;
        transition: all 0.2s ease;
    }
    
    .aspect-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    
    .aspect-title {
        font-weight: 600;
        font-size: 1rem;
        color: var(--gray-800);
        margin-bottom: 0.5rem;
    }
    
    .sentiment-badge-positive {
        background: #ecfdf5;
        color: var(--success);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .sentiment-badge-negative {
        background: #fef2f2;
        color: var(--danger);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .sentiment-badge-neutral {
        background: #fffbeb;
        color: var(--warning);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .progress-container {
        background: var(--gray-200);
        border-radius: 9999px;
        height: 0.375rem;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .progress-bar-positive {
        background: var(--success);
        height: 100%;
        border-radius: 9999px;
    }
    
    .progress-bar-negative {
        background: var(--danger);
        height: 100%;
        border-radius: 9999px;
    }
    
    .progress-bar-neutral {
        background: var(--warning);
        height: 100%;
        border-radius: 9999px;
    }
    
    .insight-positive {
        background: #ecfdf5;
        border-left: 3px solid var(--success);
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }
    
    .insight-negative {
        background: #fef2f2;
        border-left: 3px solid var(--danger);
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }
    
    .insight-neutral {
        background: #fffbeb;
        border-left: 3px solid var(--warning);
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }
    
    .stButton button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background: var(--primary-dark);
    }
    
    .footer {
        text-align: center;
        color: var(--gray-600);
        font-size: 0.75rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--gray-200);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# ENHANCED ANALYSIS WITH SLANG + MIXED SENTIMENT HANDLING
# =========================

def analyze_all_aspects(text):
    """Enhanced detection with slang, mixed sentiment, and neutral handling"""
    text_lower = text.lower()
    results = {}
    
    # Pre-process for common slang/casual language
    slang_map = {
        "chill": "good",
        "kinda": "somewhat",
        "tbh": "honestly",
        "prof": "professor",
        "nice": "good",
        "okay": "neutral",
        "decent": "neutral",
        "meh": "neutral",
        "sucks": "bad",
        "trash": "bad",
        "awesome": "excellent",
        "amazing": "excellent",
        "horrible": "terrible",
        "useless": "bad"
    }
    
    processed_text = text_lower
    for slang, replacement in slang_map.items():
        processed_text = processed_text.replace(slang, replacement)
    
    # Split into clauses for mixed sentiment detection (using "but", "however", "although", "in some", "in others")
    clauses = re.split(r'\s+but\s+|\s+however\s+|\s+although\s+|\s+though\s+', processed_text)
    
    def get_detailed_sentiment(aspect_keywords, positive_words, negative_words, neutral_words, text_to_check):
        """Check sentiment with support for mixed/neutral"""
        pos_score = 0
        neg_score = 0
        neutral_score = 0
        
        for word in positive_words:
            if word in text_to_check:
                pos_score += 1
        for word in negative_words:
            if word in text_to_check:
                neg_score += 1
        for word in neutral_words:
            if word in text_to_check:
                neutral_score += 1
        
        # Check for mixed sentiment patterns
        if "in some" in text_to_check and "in others" in text_to_check:
            return "Neutral", 0.70  # Mixed means neutral overall
        
        if pos_score > neg_score and pos_score > neutral_score:
            return "Positive", min(0.92, 0.80 + (pos_score * 0.05))
        elif neg_score > pos_score and neg_score > neutral_score:
            return "Negative", min(0.89, 0.80 + (neg_score * 0.05))
        elif neutral_score > 0:
            return "Neutral", 0.70
        else:
            return None, None
    
    # ========== 1. TEACHING QUALITY ==========
    teaching_keywords = ["teacher", "professor", "instructor", "faculty", "lecture", "teaches", "teaching", "prof", "chill"]
    if any(word in processed_text for word in teaching_keywords):
        positive = ["excellent", "good", "great", "clear", "knowledgeable", "explains well", "explains nicely", "amazing", "fantastic", "brilliant", "very helpful", "nice", "chill"]
        negative = ["poor", "bad", "confusing", "unclear", "boring", "terrible", "worst", "awful", "horrible", "doesn't explain", "rushed", "brutal"]
        neutral = ["okay", "decent", "average", "fine", "not bad", "could be better", "sometimes", "in some", "in others"]
        
        sentiment, conf = get_detailed_sentiment(teaching_keywords, positive, negative, neutral, processed_text)
        if sentiment:
            score = 88 if sentiment == "Positive" else (22 if sentiment == "Negative" else 55)
            results["Teaching Quality"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # ========== 2. ASSESSMENT & EVALUATION ==========
    assessment_keywords = ["assignment", "exam", "grade", "test", "quiz", "rubric", "evaluation", "grading", "assessment", "marks", "evaluation system"]
    if any(word in processed_text for word in assessment_keywords):
        positive = ["fair", "easy", "good", "great", "reasonable", "clear", "transparent", "well-designed"]
        negative = ["unfair", "hard", "difficult", "lengthy", "too much", "tough", "impossible", "opaque", "biased", "inconsistent", "too lengthy", "confusing", "brutal", "could be better"]
        neutral = ["okay", "decent", "average", "fine", "not bad"]
        
        sentiment, conf = get_detailed_sentiment(assessment_keywords, positive, negative, neutral, processed_text)
        if sentiment:
            score = 82 if sentiment == "Positive" else (28 if sentiment == "Negative" else 55)
            results["Assessment & Evaluation"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # ========== 3. PRACTICAL & LAB ==========
    lab_keywords = ["lab", "practical", "hands-on", "equipment", "session", "laboratory"]
    if any(word in processed_text for word in lab_keywords):
        positive = ["good", "great", "excellent", "well", "organized", "helpful", "smooth", "modern", "functional", "helpful"]
        negative = ["bad", "poor", "broken", "disorganized", "mess", "terrible", "outdated", "non-functional", "chaos", "kinda useless", "useless", "tbh"]
        neutral = ["okay", "decent", "average", "fine", "could be improved", "not great"]
        
        sentiment, conf = get_detailed_sentiment(lab_keywords, positive, negative, neutral, processed_text)
        if sentiment:
            score = 86 if sentiment == "Positive" else (24 if sentiment == "Negative" else 55)
            results["Practical & Lab"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # ========== 4. MENTORING & SUPPORT ==========
    support_keywords = ["ta", "tas", "mentor", "support", "guidance", "office hours", "assistant", "help", "faculty"]
    if any(word in processed_text for word in support_keywords):
        positive = ["helpful", "supportive", "great", "excellent", "good", "available", "responsive", "approachable", "friendly", "very helpful"]
        negative = ["unhelpful", "bad", "poor", "ignored", "no support", "unavailable", "unresponsive", "dismissive", "sometimes unavailable"]
        neutral = ["okay", "decent", "average", "sometimes", "occasionally"]
        
        sentiment, conf = get_detailed_sentiment(support_keywords, positive, negative, neutral, processed_text)
        if sentiment:
            score = 87 if sentiment == "Positive" else (26 if sentiment == "Negative" else 55)
            results["Mentoring & Support"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # ========== 5. COURSE DESIGN & WORKLOAD ==========
    workload_keywords = ["workload", "lengthy", "heavy", "deadline", "pace", "structure", "organization", "design", "curriculum", "tight", "rushed"]
    if any(word in processed_text for word in workload_keywords):
        positive = ["manageable", "good", "balanced", "well-structured", "perfect", "reasonable", "appropriate"]
        negative = ["overwhelming", "too much", "crushing", "high", "excessive", "unreasonable", "impossible", "lengthy", "tight", "rushed"]
        neutral = ["okay", "decent", "average", "fine", "but manageable"]
        
        sentiment, conf = get_detailed_sentiment(workload_keywords, positive, negative, neutral, processed_text)
        if sentiment:
            score = 84 if sentiment == "Positive" else (30 if sentiment == "Negative" else 55)
            results["Course Design & Workload"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # ========== 6. TEACHING METHODOLOGY ==========
    methodology_keywords = ["slides", "interactive", "examples", "delivery", "method", "engagement", "pedagogy", "approach", "teaching style"]
    if any(word in processed_text for word in methodology_keywords):
        positive = ["interactive", "engaging", "great examples", "practical", "dynamic", "innovative", "effective"]
        negative = ["boring", "monotonous", "just reads", "unengaging", "static", "tedious", "repetitive", "not very engaging"]
        neutral = ["okay", "decent", "average", "fine", "in some", "in others"]
        
        sentiment, conf = get_detailed_sentiment(methodology_keywords, positive, negative, neutral, processed_text)
        if sentiment:
            score = 85 if sentiment == "Positive" else (32 if sentiment == "Negative" else 55)
            results["Teaching Methodology"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # ========== 7. LEARNING OUTCOMES ==========
    outcomes_keywords = ["learn", "skill", "outcome", "knowledge", "useful", "value", "understanding", "competency", "relevant"]
    if any(word in processed_text for word in outcomes_keywords):
        positive = ["learned a lot", "useful", "valuable", "improved", "great learning", "practical knowledge", "applicable", "good and relevant"]
        negative = ["waste", "nothing", "useless", "no improvement", "irrelevant", "forgettable", "kinda useless"]
        neutral = ["okay", "decent", "average", "fine", "somewhat"]
        
        sentiment, conf = get_detailed_sentiment(outcomes_keywords, positive, negative, neutral, processed_text)
        if sentiment:
            score = 86 if sentiment == "Positive" else (25 if sentiment == "Negative" else 55)
            results["Learning Outcomes"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # Fallback: If no aspects detected but text has meaningful content
    if not results and len(processed_text.split()) > 5:
        # Check for general sentiment
        general_positive = ["good", "great", "nice", "excellent", "awesome", "helpful", "useful"]
        general_negative = ["bad", "poor", "terrible", "awful", "horrible", "useless", "waste"]
        
        if any(word in processed_text for word in general_positive):
            results["General Feedback"] = {"sentiment": "Positive", "confidence": 0.75, "score": 70}
        elif any(word in processed_text for word in general_negative):
            results["General Feedback"] = {"sentiment": "Negative", "confidence": 0.75, "score": 30}
        else:
            results["General Feedback"] = {"sentiment": "Neutral", "confidence": 0.65, "score": 50}
    
    return results

def generate_insights(results):
    """Generate actionable insights from results"""
    insights = {"strengths": [], "weaknesses": [], "neutral": [], "recommendations": []}
    
    for aspect, data in results.items():
        if data['sentiment'] == 'Positive':
            insights["strengths"].append(aspect)
            insights["recommendations"].append(f"Maintain current standards for {aspect}")
        elif data['sentiment'] == 'Negative':
            insights["weaknesses"].append(aspect)
            insights["recommendations"].append(f"[URGENT] Address issues with {aspect}")
        elif data['sentiment'] == 'Neutral':
            insights["neutral"].append(aspect)
            insights["recommendations"].append(f"Review and improve {aspect} to convert to positive")
    
    return insights

def calculate_overall_score(results):
    """Calculate overall satisfaction score (0-100)"""
    if not results:
        return 50
    scores = [data['score'] for data in results.values()]
    return sum(scores) // len(scores)

# =========================
# STREAMLIT UI
# =========================

st.markdown('<h1>Student Feedback Analytics System</h1>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Multi-dimensional analysis of student feedback with aspect-wise sentiment detection</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Analysis Configuration")
    
    st.markdown("#### Sample Test Cases")
    test_cases = {
        "Edge: Mixed Teaching": "The teaching style is engaging in some classes but very boring in others",
        "Edge: Casual Language": "The prof is chill and explains nicely but exams are brutal and labs are kinda useless tbh",
        "Edge: Neutral Statement": "The course is decent but not very engaging, and the evaluation system could be better.",
        "Mixed Feedback": "The faculty is very helpful. However, the assignments are too lengthy. The lab sessions are good.",
        "Complex Mixed": "The course content is good and relevant, but the workload is too much and deadlines are very tight. The mentor was helpful but sometimes unavailable.",
        "Mixed with Improvements": "The professor explains well. However, the assignments are confusing and difficult. The lab sessions are okay but could be improved.",
        "Positive with Caveats": "The teaching is good but sometimes rushed, and the workload is high but manageable. The lab sessions are helpful."
    }
    
    for name, text in test_cases.items():
        if st.button(name, use_container_width=True):
            st.session_state['feedback_input'] = text
            st.rerun()
    
    st.markdown("---")
    st.markdown("#### Analytics Framework")
    st.markdown("""
    **7 Pedagogical Dimensions**
    
    - Teaching Quality
    - Assessment & Evaluation
    - Practical & Lab
    - Mentoring & Support
    - Course Design & Workload
    - Teaching Methodology
    - Learning Outcomes
    """)
    
    st.markdown("---")
    st.markdown("#### Features")
    st.info("""
    - Slang/casual language support
    - Mixed sentiment detection
    - Neutral statement handling
    - Fallback general feedback detection
    """)

# Main input
if 'feedback_input' not in st.session_state:
    st.session_state['feedback_input'] = ""

feedback = st.text_area(
    "Feedback Text",
    value=st.session_state['feedback_input'],
    height=120,
    placeholder="Enter student feedback here...",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("Run Analysis", type="primary", use_container_width=True)

if analyze and feedback.strip():
    with st.spinner("Processing feedback..."):
        results = analyze_all_aspects(feedback)
        insights = generate_insights(results)
        overall_score = calculate_overall_score(results)
    
    if results:
        # Key metrics row
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(results)}</div>
                <div class="metric-label">ASPECTS DETECTED</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pos_count = sum(1 for r in results.values() if r['sentiment'] == 'Positive')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #059669;">{pos_count}</div>
                <div class="metric-label">POSITIVE ASPECTS</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            neg_count = sum(1 for r in results.values() if r['sentiment'] == 'Negative')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #dc2626;">{neg_count}</div>
                <div class="metric-label">NEGATIVE ASPECTS</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            neutral_count = sum(1 for r in results.values() if r['sentiment'] == 'Neutral')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #d97706;">{neutral_count}</div>
                <div class="metric-label">NEUTRAL ASPECTS</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall score gauge
        st.markdown("---")
        st.markdown("#### Overall Satisfaction Score")
        
        gauge_color = "#059669" if overall_score >= 70 else ("#d97706" if overall_score >= 50 else "#dc2626")
        st.markdown(f"""
        <div style="background: #e5e7eb; border-radius: 9999px; height: 0.5rem; overflow: hidden;">
            <div style="background: {gauge_color}; width: {overall_score}%; height: 0.5rem; border-radius: 9999px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
            <span style="font-size: 0.75rem; color: #6b7280;">Dissatisfied</span>
            <span style="font-size: 0.75rem; color: #6b7280;">Neutral</span>
            <span style="font-size: 0.75rem; color: #6b7280;">Satisfied</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Aspect-wise analysis
        st.markdown("---")
        st.markdown("#### Aspect-wise Analysis")
        
        for aspect, data in results.items():
            sentiment_class = f"sentiment-badge-{data['sentiment'].lower()}"
            progress_class = f"progress-bar-{data['sentiment'].lower()}"
            
            st.markdown(f"""
            <div class="aspect-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span class="aspect-title">{aspect}</span>
                    <span class="{sentiment_class}">{data['sentiment']}</span>
                </div>
                <div style="font-size: 0.875rem; color: #6b7280;">Confidence: {data['confidence']:.1%}</div>
                <div class="progress-container">
                    <div class="{progress_class}" style="width: {data['confidence']*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Key insights
        st.markdown("---")
        st.markdown("#### Key Insights")
        
        if insights["strengths"]:
            st.markdown(f"""
            <div class="insight-positive">
                <strong>Strengths Identified</strong><br>
                {', '.join(insights['strengths'])} - These aspects received positive feedback.
            </div>
            """, unsafe_allow_html=True)
        
        if insights["weaknesses"]:
            st.markdown(f"""
            <div class="insight-negative">
                <strong>Areas for Improvement</strong><br>
                {', '.join(insights['weaknesses'])} - Student feedback indicates concerns in these areas.
            </div>
            """, unsafe_allow_html=True)
        
        if insights["neutral"]:
            st.markdown(f"""
            <div class="insight-neutral">
                <strong>Neutral Aspects</strong><br>
                {', '.join(insights['neutral'])} - These aspects received mixed or neutral feedback.
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        if insights["recommendations"]:
            st.markdown("#### Recommendations")
            for rec in insights["recommendations"]:
                if "URGENT" in rec:
                    st.error(rec)
                elif "Review" in rec:
                    st.warning(rec)
                else:
                    st.success(rec)
        
        # Export functionality
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "feedback": feedback,
                "overall_score": overall_score,
                "aspects": {k: v['sentiment'] for k, v in results.items()}
            }
            import json
            st.download_button(
                label="Export Report (JSON)",
                data=json.dumps(report_data, indent=2),
                file_name=f"feedback_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
    else:
        st.warning("No specific aspects detected. Please provide more detailed feedback.")

elif analyze:
    st.warning("Please enter feedback to analyze.")

st.markdown('<div class="footer">Student Feedback Analytics System | Enhanced with Slang Support & Mixed Sentiment Detection</div>', unsafe_allow_html=True)