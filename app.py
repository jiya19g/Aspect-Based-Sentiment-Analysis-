import streamlit as st
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import io
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
    
    .bulk-stat-card {
        background: white;
        border: 1px solid var(--gray-200);
        border-radius: 0.5rem;
        padding: 0.75rem;
        text-align: center;
    }
    
    .bulk-stat-value {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .bulk-stat-label {
        font-size: 0.7rem;
        color: var(--gray-600);
    }
    
    .priority-high {
        background: #fef2f2;
        border-left: 3px solid #dc2626;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    
    .priority-medium {
        background: #fffbeb;
        border-left: 3px solid #d97706;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    
    .priority-low {
        background: #f0fdf4;
        border-left: 3px solid #059669;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CORE ANALYSIS FUNCTIONS (COMPLETELY UNCHANGED)
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
    
    def get_detailed_sentiment(positive_words, negative_words, neutral_words, text_to_check):
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
        
        if "in some" in text_to_check and "in others" in text_to_check:
            return "Neutral", 0.70
        
        if pos_score > neg_score and pos_score > neutral_score:
            return "Positive", min(0.92, 0.80 + (pos_score * 0.05))
        elif neg_score > pos_score and neg_score > neutral_score:
            return "Negative", min(0.89, 0.80 + (neg_score * 0.05))
        elif neutral_score > 0:
            return "Neutral", 0.70
        else:
            return None, None
    
    # Teaching Quality
    teaching_keywords = ["teacher", "professor", "instructor", "faculty", "lecture", "teaches", "teaching", "prof"]
    if any(word in processed_text for word in teaching_keywords):
        positive = ["excellent", "good", "great", "clear", "knowledgeable", "explains well", "explains nicely", "amazing", "fantastic", "brilliant", "very helpful", "nice"]
        negative = ["poor", "bad", "confusing", "unclear", "boring", "terrible", "worst", "awful", "horrible", "doesn't explain", "rushed", "brutal"]
        neutral = ["okay", "decent", "average", "fine", "not bad", "could be better", "sometimes", "in some", "in others"]
        sentiment, conf = get_detailed_sentiment(positive, negative, neutral, processed_text)
        if sentiment:
            score = 88 if sentiment == "Positive" else (22 if sentiment == "Negative" else 55)
            results["Teaching Quality"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # Assessment & Evaluation
    assessment_keywords = ["assignment", "exam", "grade", "test", "quiz", "rubric", "evaluation", "grading", "assessment", "marks", "evaluation system"]
    if any(word in processed_text for word in assessment_keywords):
        positive = ["fair", "easy", "good", "great", "reasonable", "clear", "transparent", "well-designed"]
        negative = ["unfair", "hard", "difficult", "lengthy", "too much", "tough", "impossible", "opaque", "biased", "inconsistent", "too lengthy", "confusing", "brutal", "could be better"]
        neutral = ["okay", "decent", "average", "fine", "not bad"]
        sentiment, conf = get_detailed_sentiment(positive, negative, neutral, processed_text)
        if sentiment:
            score = 82 if sentiment == "Positive" else (28 if sentiment == "Negative" else 55)
            results["Assessment & Evaluation"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # Practical & Lab
    lab_keywords = ["lab", "practical", "hands-on", "equipment", "session", "laboratory"]
    if any(word in processed_text for word in lab_keywords):
        positive = ["good", "great", "excellent", "well", "organized", "helpful", "smooth", "modern", "functional"]
        negative = ["bad", "poor", "broken", "disorganized", "mess", "terrible", "outdated", "non-functional", "chaos", "kinda useless", "useless"]
        neutral = ["okay", "decent", "average", "fine", "could be improved", "not great"]
        sentiment, conf = get_detailed_sentiment(positive, negative, neutral, processed_text)
        if sentiment:
            score = 86 if sentiment == "Positive" else (24 if sentiment == "Negative" else 55)
            results["Practical & Lab"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # Mentoring & Support
    support_keywords = ["ta", "tas", "mentor", "support", "guidance", "office hours", "assistant", "help", "faculty"]
    if any(word in processed_text for word in support_keywords):
        positive = ["helpful", "supportive", "great", "excellent", "good", "available", "responsive", "approachable", "friendly", "very helpful"]
        negative = ["unhelpful", "bad", "poor", "ignored", "no support", "unavailable", "unresponsive", "dismissive", "sometimes unavailable"]
        neutral = ["okay", "decent", "average", "sometimes", "occasionally"]
        sentiment, conf = get_detailed_sentiment(positive, negative, neutral, processed_text)
        if sentiment:
            score = 87 if sentiment == "Positive" else (26 if sentiment == "Negative" else 55)
            results["Mentoring & Support"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # Course Design & Workload
    workload_keywords = ["workload", "lengthy", "heavy", "deadline", "pace", "structure", "organization", "design", "curriculum", "tight", "rushed"]
    if any(word in processed_text for word in workload_keywords):
        positive = ["manageable", "good", "balanced", "well-structured", "perfect", "reasonable", "appropriate"]
        negative = ["overwhelming", "too much", "crushing", "high", "excessive", "unreasonable", "impossible", "lengthy", "tight", "rushed"]
        neutral = ["okay", "decent", "average", "fine", "but manageable"]
        sentiment, conf = get_detailed_sentiment(positive, negative, neutral, processed_text)
        if sentiment:
            score = 84 if sentiment == "Positive" else (30 if sentiment == "Negative" else 55)
            results["Course Design & Workload"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # Teaching Methodology
    methodology_keywords = ["slides", "interactive", "examples", "delivery", "method", "engagement", "pedagogy", "approach", "teaching style"]
    if any(word in processed_text for word in methodology_keywords):
        positive = ["interactive", "engaging", "great examples", "practical", "dynamic", "innovative", "effective"]
        negative = ["boring", "monotonous", "just reads", "unengaging", "static", "tedious", "repetitive", "not very engaging"]
        neutral = ["okay", "decent", "average", "fine", "in some", "in others"]
        sentiment, conf = get_detailed_sentiment(positive, negative, neutral, processed_text)
        if sentiment:
            score = 85 if sentiment == "Positive" else (32 if sentiment == "Negative" else 55)
            results["Teaching Methodology"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # Learning Outcomes
    outcomes_keywords = ["learn", "skill", "outcome", "knowledge", "useful", "value", "understanding", "competency", "relevant"]
    if any(word in processed_text for word in outcomes_keywords):
        positive = ["learned a lot", "useful", "valuable", "improved", "great learning", "practical knowledge", "applicable", "good and relevant"]
        negative = ["waste", "nothing", "useless", "no improvement", "irrelevant", "forgettable", "kinda useless"]
        neutral = ["okay", "decent", "average", "fine", "somewhat"]
        sentiment, conf = get_detailed_sentiment(positive, negative, neutral, processed_text)
        if sentiment:
            score = 86 if sentiment == "Positive" else (25 if sentiment == "Negative" else 55)
            results["Learning Outcomes"] = {"sentiment": sentiment, "confidence": conf, "score": score}
    
    # Fallback
    if not results and len(processed_text.split()) > 5:
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
    if not results:
        return 50
    scores = [data['score'] for data in results.values()]
    return sum(scores) // len(scores)

# =========================
# ENHANCED BULK ANALYSIS FUNCTIONS
# =========================

def analyze_bulk_feedback(df, text_column='feedback'):
    results_list = []
    for idx, row in df.iterrows():
        feedback_text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        if len(feedback_text.strip()) > 10:
            aspect_results = analyze_all_aspects(feedback_text)
            overall_score = calculate_overall_score(aspect_results)
            
            pos_count = sum(1 for r in aspect_results.values() if r['sentiment'] == 'Positive')
            neg_count = sum(1 for r in aspect_results.values() if r['sentiment'] == 'Negative')
            neutral_count = sum(1 for r in aspect_results.values() if r['sentiment'] == 'Neutral')
            
            # Extract additional metadata if available
            row_data = {
                'feedback_id': idx,
                'feedback_text': feedback_text[:150] + ('...' if len(feedback_text) > 150 else ''),
                'full_text': feedback_text,
                'aspects_detected': len(aspect_results),
                'positive_aspects': pos_count,
                'negative_aspects': neg_count,
                'neutral_aspects': neutral_count,
                'overall_score': overall_score,
                'aspect_details': aspect_results
            }
            
            # Add any additional columns from original CSV
            for col in df.columns:
                if col not in ['feedback', 'feedback_text', 'full_text']:
                    row_data[col] = row[col]
            
            results_list.append(row_data)
    return results_list

def generate_bulk_insights(bulk_results):
    total = len(bulk_results)
    if total == 0:
        return {}
    
    aspect_aggregator = {}
    all_scores = []
    
    for result in bulk_results:
        all_scores.append(result['overall_score'])
        for aspect, data in result['aspect_details'].items():
            if aspect not in aspect_aggregator:
                aspect_aggregator[aspect] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total_score': 0, 'count': 0}
            aspect_aggregator[aspect][data['sentiment'].lower()] += 1
            aspect_aggregator[aspect]['total_score'] += data['score']
            aspect_aggregator[aspect]['count'] += 1
    
    for aspect in aspect_aggregator:
        count = aspect_aggregator[aspect]['count']
        aspect_aggregator[aspect]['avg_score'] = aspect_aggregator[aspect]['total_score'] / count if count > 0 else 0
        aspect_aggregator[aspect]['positive_pct'] = (aspect_aggregator[aspect]['positive'] / total) * 100
        aspect_aggregator[aspect]['negative_pct'] = (aspect_aggregator[aspect]['negative'] / total) * 100
    
    avg_score = sum(all_scores) / total
    score_std = pd.Series(all_scores).std() if len(all_scores) > 1 else 0
    
    # Calculate sentiment distribution across all feedbacks
    feedback_sentiments = []
    for result in bulk_results:
        pos = result['positive_aspects']
        neg = result['negative_aspects']
        if pos > neg:
            feedback_sentiments.append('Positive')
        elif neg > pos:
            feedback_sentiments.append('Negative')
        else:
            feedback_sentiments.append('Neutral')
    
    sentiment_counts = Counter(feedback_sentiments)
    
    # Generate priority matrix
    priority_matrix = []
    for aspect, agg in aspect_aggregator.items():
        negative_pct = agg['negative_pct']
        avg_score = agg['avg_score']
        
        if negative_pct > 30:
            priority = "High"
        elif negative_pct > 15:
            priority = "Medium"
        else:
            priority = "Low"
        
        priority_matrix.append({
            'aspect': aspect,
            'negative_percentage': negative_pct,
            'avg_score': avg_score,
            'priority': priority
        })
    
    priority_matrix.sort(key=lambda x: x['negative_percentage'], reverse=True)
    
    # Extract keywords from feedbacks for word cloud
    all_words = []
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'so', 'for', 'nor', 'yet', 'of', 'to', 'in', 'on', 'at', 'with', 'without', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those', 'it', 'they', 'we', 'you', 'he', 'she'}
    
    for result in bulk_results:
        words = result['full_text'].lower().split()
        all_words.extend([w for w in words if len(w) > 3 and w not in stop_words])
    
    word_freq = Counter(all_words).most_common(20)
    
    # Find top positive and negative feedback examples
    positive_feedbacks = [r for r in bulk_results if r['positive_aspects'] > r['negative_aspects']]
    negative_feedbacks = [r for r in bulk_results if r['negative_aspects'] > r['positive_aspects']]
    
    top_positive = sorted(positive_feedbacks, key=lambda x: x['overall_score'], reverse=True)[:3]
    top_negative = sorted(negative_feedbacks, key=lambda x: x['overall_score'])[:3]
    
    # Correlation analysis between aspects
    aspect_correlation = {}
    aspects_list = list(aspect_aggregator.keys())
    for aspect in aspects_list:
        aspect_correlation[aspect] = {'positive_corr': [], 'negative_corr': []}
    
    return {
        'total_feedbacks': total,
        'avg_overall_score': avg_score,
        'score_std': score_std,
        'total_positive_aspects': sum(r['positive_aspects'] for r in bulk_results),
        'total_negative_aspects': sum(r['negative_aspects'] for r in bulk_results),
        'sentiment_distribution': sentiment_counts,
        'aspect_aggregator': aspect_aggregator,
        'priority_matrix': priority_matrix,
        'word_frequency': word_freq,
        'top_positive_feedbacks': top_positive,
        'top_negative_feedbacks': top_negative,
        'bulk_results': bulk_results
    }

def get_sample_csv():
    sample_data = pd.DataFrame({
        'feedback': [
            "The professor explains concepts very clearly. However, the assignments are too difficult and the workload is high.",
            "Great teaching methodology with interactive sessions. The TAs are very helpful and always available.",
            "Poor course design. The lab sessions are disorganized with broken equipment. Grading is unfair.",
            "Excellent course structure. The mentor was very supportive throughout the semester.",
            "The teaching is good but sometimes rushed. Assignments are fair but deadlines are too tight.",
            "Labs are well-organized and TAs are knowledgeable. The professor could be more engaging though.",
            "Worst course ever. Unfair grading and useless labs. No support from faculty.",
            "The course content is relevant and practical. Workload is manageable. Would recommend!",
            "Professor is knowledgeable but the exam was too difficult. Lab sessions were helpful.",
            "Great support from TAs but the course structure needs improvement. Too many deadlines."
        ],
        'course': ['CS101', 'CS101', 'CS201', 'CS201', 'CS101', 'CS301', 'CS301', 'CS201', 'CS101', 'CS301'],
        'date': pd.date_range(start='2024-01-01', periods=10, freq='D')
    })
    return sample_data

# =========================
# STREAMLIT UI - TABS
# =========================

# Create tabs
tab1, tab2 = st.tabs(["📝 Single Feedback Analysis", "📊 Bulk Feedback Analysis"])

# ========== TAB 1: SINGLE FEEDBACK ANALYSIS (COMPLETELY UNCHANGED) ==========
with tab1:
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
            if st.button(name, use_container_width=True, key=f"single_{name}"):
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
        label_visibility="collapsed",
        key="single_feedback"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze = st.button("Run Analysis", type="primary", use_container_width=True, key="single_analyze")
    
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
                st.download_button(
                    label="Export Report (JSON)",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"feedback_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="single_export"
                )
            
        else:
            st.warning("No specific aspects detected. Please provide more detailed feedback.")
    
    elif analyze:
        st.warning("Please enter feedback to analyze.")
    
    st.markdown('<div class="footer">Student Feedback Analytics System | Enhanced with Slang Support & Mixed Sentiment Detection</div>', unsafe_allow_html=True)

# ========== TAB 2: BULK FEEDBACK ANALYSIS (ENHANCED) ==========
with tab2:
    st.markdown("## Bulk Feedback Analysis")
    st.markdown("Upload a CSV file containing multiple student feedback entries for comprehensive batch analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="CSV file must contain a 'feedback' column. Optional: 'course', 'date', 'instructor' columns for deeper insights.",
            key="bulk_upload"
        )
    
    with col2:
        st.markdown("#### Sample Data")
        if st.button("Load Sample Dataset", use_container_width=True, key="load_sample"):
            sample_df = get_sample_csv()
            st.session_state['sample_df'] = sample_df
    
    # Display data preview
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'feedback' not in df.columns:
            st.error("CSV must contain a 'feedback' column")
        else:
            st.success(f"✅ Loaded {len(df)} feedback entries")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("📊 Analyze Bulk Feedback", type="primary", use_container_width=True, key="bulk_analyze"):
                with st.spinner(f"Analyzing {len(df)} feedback entries..."):
                    bulk_results = analyze_bulk_feedback(df, 'feedback')
                    insights = generate_bulk_insights(bulk_results)
                    st.session_state['bulk_insights'] = insights
                    st.session_state['bulk_df'] = df
    
    elif 'sample_df' in st.session_state:
        df = st.session_state['sample_df']
        st.info("📋 Sample Dataset Loaded (10 feedback entries with course and date information)")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("📊 Analyze Sample Data", type="primary", use_container_width=True, key="bulk_analyze_sample"):
            with st.spinner(f"Analyzing {len(df)} feedback entries..."):
                bulk_results = analyze_bulk_feedback(df, 'feedback')
                insights = generate_bulk_insights(bulk_results)
                st.session_state['bulk_insights'] = insights
                st.session_state['bulk_df'] = df
    
    # Display bulk analysis results
    if 'bulk_insights' in st.session_state:
        insights = st.session_state['bulk_insights']
        
        st.markdown("---")
        st.markdown("## Executive Dashboard")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="bulk-stat-card">
                <div class="bulk-stat-value">{insights['total_feedbacks']}</div>
                <div class="bulk-stat-label">TOTAL FEEDBACKS</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="bulk-stat-card">
                <div class="bulk-stat-value" style="color: #059669;">{insights['total_positive_aspects']}</div>
                <div class="bulk-stat-label">POSITIVE MENTIONS</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="bulk-stat-card">
                <div class="bulk-stat-value" style="color: #dc2626;">{insights['total_negative_aspects']}</div>
                <div class="bulk-stat-label">NEGATIVE MENTIONS</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            ratio = insights['total_positive_aspects'] / max(insights['total_negative_aspects'], 1)
            st.markdown(f"""
            <div class="bulk-stat-card">
                <div class="bulk-stat-value">{ratio:.1f}<span style="font-size:0.8rem;">x</span></div>
                <div class="bulk-stat-label">POS/NEG RATIO</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            gauge_color = "#059669" if insights['avg_overall_score'] >= 70 else ("#d97706" if insights['avg_overall_score'] >= 50 else "#dc2626")
            st.markdown(f"""
            <div class="bulk-stat-card">
                <div class="bulk-stat-value" style="color: {gauge_color};">{insights['avg_overall_score']:.0f}<span style="font-size:0.8rem;">/100</span></div>
                <div class="bulk-stat-label">AVG SATISFACTION</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall score gauge
        st.markdown("---")
        st.markdown("#### Average Satisfaction Score Across All Feedback")
        gauge_color = "#059669" if insights['avg_overall_score'] >= 70 else ("#d97706" if insights['avg_overall_score'] >= 50 else "#dc2626")
        st.markdown(f"""
        <div style="background: #e5e7eb; border-radius: 9999px; height: 0.5rem; overflow: hidden;">
            <div style="background: {gauge_color}; width: {insights['avg_overall_score']}%; height: 0.5rem; border-radius: 9999px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.25rem;">
            <span style="font-size: 0.7rem; color: #6b7280;">Poor (0-40)</span>
            <span style="font-size: 0.7rem; color: #6b7280;">Fair (40-60)</span>
            <span style="font-size: 0.7rem; color: #6b7280;">Good (60-80)</span>
            <span style="font-size: 0.7rem; color: #6b7280;">Excellent (80-100)</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Two column layout for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            st.markdown("#### Overall Sentiment Distribution")
            sentiment_df = pd.DataFrame({
                'Sentiment': list(insights['sentiment_distribution'].keys()),
                'Count': list(insights['sentiment_distribution'].values())
            })
            fig = px.pie(sentiment_df, values='Count', names='Sentiment', 
                        color='Sentiment', color_discrete_map={'Positive': '#059669', 'Negative': '#dc2626', 'Neutral': '#d97706'},
                        hole=0.4)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Score distribution histogram
            st.markdown("#### Score Distribution")
            scores = [r['overall_score'] for r in insights['bulk_results']]
            fig = px.histogram(x=scores, nbins=20, title="", 
                              labels={'x': 'Satisfaction Score', 'y': 'Frequency'},
                              color_discrete_sequence=['#2563eb'])
            fig.add_vline(x=insights['avg_overall_score'], line_dash="dash", line_color="red", 
                         annotation_text=f"Avg: {insights['avg_overall_score']:.0f}")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Aspect distribution chart
        if insights['aspect_aggregator']:
            st.markdown("---")
            st.markdown("#### Sentiment Distribution by Dimension")
            
            aspect_data = []
            for aspect, agg in insights['aspect_aggregator'].items():
                total = agg['positive'] + agg['negative'] + agg['neutral']
                aspect_data.append({
                    'Dimension': aspect,
                    'Positive': agg['positive'],
                    'Negative': agg['negative'],
                    'Neutral': agg['neutral'],
                    'Avg Score': round(agg['avg_score'], 1),
                    'Total Mentions': total
                })
            
            aspect_df = pd.DataFrame(aspect_data)
            
            fig = go.Figure(data=[
                go.Bar(name='Positive', x=aspect_df['Dimension'], y=aspect_df['Positive'], marker_color='#059669'),
                go.Bar(name='Negative', x=aspect_df['Dimension'], y=aspect_df['Negative'], marker_color='#dc2626'),
                go.Bar(name='Neutral', x=aspect_df['Dimension'], y=aspect_df['Neutral'], marker_color='#d97706')
            ])
            fig.update_layout(barmode='stack', height=450, title="Mentions by Dimension")
            st.plotly_chart(fig, use_container_width=True)
            
            # Dimension scores table
            st.markdown("#### Dimension Performance Scores")
            st.dataframe(aspect_df.sort_values('Avg Score', ascending=False), use_container_width=True, hide_index=True)
        
        # Priority Matrix
        st.markdown("---")
        st.markdown("#### Improvement Priority Matrix")
        st.markdown("Dimensions with high negative percentage need immediate attention")
        
        for item in insights['priority_matrix'][:7]:
            if item['priority'] == 'High':
                st.markdown(f"""
                <div class="priority-high">
                    <strong>🔴 HIGH PRIORITY:</strong> {item['aspect']} - {item['negative_percentage']:.1f}% negative feedback (Avg Score: {item['avg_score']:.0f})
                </div>
                """, unsafe_allow_html=True)
            elif item['priority'] == 'Medium':
                st.markdown(f"""
                <div class="priority-medium">
                    <strong>🟡 MEDIUM PRIORITY:</strong> {item['aspect']} - {item['negative_percentage']:.1f}% negative feedback (Avg Score: {item['avg_score']:.0f})
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="priority-low">
                    <strong>🟢 LOW PRIORITY:</strong> {item['aspect']} - {item['negative_percentage']:.1f}% negative feedback (Avg Score: {item['avg_score']:.0f})
                </div>
                """, unsafe_allow_html=True)
        
        # Word Cloud / Key Themes
        st.markdown("---")
        st.markdown("#### Key Themes from Feedback")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Keywords**")
            keyword_df = pd.DataFrame(insights['word_frequency'], columns=['Keyword', 'Frequency'])
            st.dataframe(keyword_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Simple bar chart of top keywords
            keywords = [k for k, v in insights['word_frequency'][:10]]
            freqs = [v for k, v in insights['word_frequency'][:10]]
            fig = px.bar(x=freqs, y=keywords, orientation='h', title="Most Frequent Terms",
                        labels={'x': 'Frequency', 'y': ''}, color=freqs, color_continuous_scale='Blues')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top Positive and Negative Examples
        st.markdown("---")
        st.markdown("#### Representative Feedback Examples")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Best Feedback Examples**")
            for fb in insights['top_positive_feedbacks']:
                with st.expander(f"Score: {fb['overall_score']}/100"):
                    st.write(fb['full_text'])
        
        with col2:
            st.markdown("**📉 Needs Improvement Examples**")
            for fb in insights['top_negative_feedbacks']:
                with st.expander(f"Score: {fb['overall_score']}/100"):
                    st.write(fb['full_text'])
        
        # Course-wise breakdown (if course column exists)
        if 'course' in df.columns and 'course' in insights['bulk_results'][0]:
            st.markdown("---")
            st.markdown("#### Course-wise Analysis")
            
            course_stats = {}
            for result in insights['bulk_results']:
                course = result.get('course', 'Unknown')
                if course not in course_stats:
                    course_stats[course] = {'scores': [], 'positive': 0, 'negative': 0}
                course_stats[course]['scores'].append(result['overall_score'])
                course_stats[course]['positive'] += result['positive_aspects']
                course_stats[course]['negative'] += result['negative_aspects']
            
            course_data = []
            for course, stats in course_stats.items():
                avg_score = sum(stats['scores']) / len(stats['scores'])
                total = stats['positive'] + stats['negative']
                pos_pct = (stats['positive'] / total * 100) if total > 0 else 0
                course_data.append({
                    'Course': course,
                    'Feedbacks': len(stats['scores']),
                    'Avg Score': round(avg_score, 1),
                    'Positive %': round(pos_pct, 1),
                    'Total Mentions': total
                })
            
            course_df = pd.DataFrame(course_data)
            fig = px.bar(course_df, x='Course', y='Avg Score', title="Satisfaction Score by Course",
                        color='Avg Score', color_continuous_scale='RdYlGn', text='Avg Score')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(course_df, use_container_width=True, hide_index=True)
        
        # Temporal trends (if date column exists)
        if 'date' in df.columns and 'date' in insights['bulk_results'][0]:
            st.markdown("---")
            st.markdown("#### Temporal Trends")
            
            date_stats = {}
            for result in insights['bulk_results']:
                date = pd.to_datetime(result.get('date')).strftime('%Y-%m')
                if date not in date_stats:
                    date_stats[date] = {'scores': [], 'count': 0}
                date_stats[date]['scores'].append(result['overall_score'])
                date_stats[date]['count'] += 1
            
            date_data = []
            for date, stats in sorted(date_stats.items()):
                avg_score = sum(stats['scores']) / len(stats['scores'])
                date_data.append({'Month': date, 'Avg Score': avg_score, 'Feedbacks': stats['count']})
            
            date_df = pd.DataFrame(date_data)
            fig = px.line(date_df, x='Month', y='Avg Score', title="Satisfaction Trend Over Time",
                         markers=True, line_shape='linear')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.markdown("---")
        st.markdown("#### Detailed Analysis Results")
        
        results_df = pd.DataFrame([
            {
                'ID': r['feedback_id'],
                'Feedback': r['feedback_text'],
                'Aspects': r['aspects_detected'],
                'Positive': r['positive_aspects'],
                'Negative': r['negative_aspects'],
                'Neutral': r['neutral_aspects'],
                'Score': r['overall_score']
            }
            for r in insights['bulk_results']
        ])
        
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        st.markdown("---")
        st.markdown("#### Statistical Summary")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Mean Score", f"{insights['avg_overall_score']:.1f}")
        with stat_col2:
            st.metric("Std Deviation", f"{insights['score_std']:.1f}")
        with stat_col3:
            min_score = min(r['overall_score'] for r in insights['bulk_results'])
            st.metric("Min Score", f"{min_score}")
        with stat_col4:
            max_score = max(r['overall_score'] for r in insights['bulk_results'])
            st.metric("Max Score", f"{max_score}")
        
        # Export functionality
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Export Results to CSV",
                data=csv_buffer.getvalue(),
                file_name=f"bulk_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="bulk_export"
            )
        
        # Clear button
        if st.button("Clear Results", use_container_width=True, key="clear_bulk"):
            del st.session_state['bulk_insights']
            del st.session_state['bulk_df']
            st.rerun()

st.markdown('<div class="footer">Student Feedback Analytics System | Enhanced with Slang Support & Mixed Sentiment Detection</div>', unsafe_allow_html=True)