"""
robust_analyzer.py
==================
Completely rule-based multi-aspect sentiment analyzer.
Handles: clause splitting, contrast conjunctions, negation,
         slang, mixed sentiment, neutral detection, edge cases.

Drop-in replacement — same return format as the original:
  {aspect: {"sentiment": str, "confidence": float, "score": int}}
"""

import re
from typing import Dict, List, Tuple, Optional

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: SLANG & NORMALISATION MAP
# ════════════════════════════════════════════════════════════════════════════

SLANG_MAP = {
    # Teaching
    "prof": "professor",
    "profs": "professor",
    "lec": "lecture",
    "lecs": "lectures",
    # Sentiment slang
    "chill": "relaxed good",
    "goated": "excellent",
    "lit": "excellent",
    "fire": "excellent",
    "bussin": "excellent",
    "slaps": "excellent",
    "mid": "mediocre",
    "meh": "mediocre",
    "idk": "",
    "tbh": "",
    "ngl": "",
    "imo": "",
    "fr": "",
    "lowkey": "somewhat",
    "highkey": "very",
    "kinda": "somewhat",
    "sorta": "somewhat",
    "legit": "actually",
    "rn": "",
    "atm": "currently",
    "smh": "disappointing",
    "bruh": "disappointing",
    "ugh": "bad",
    "yikes": "bad",
    "lol": "",
    "lmao": "",
    # Quality slang
    "trash": "terrible",
    "garbage": "terrible",
    "ass": "terrible",
    "sucks": "bad",
    "suck": "bad",
    "sucky": "bad",
    "stinks": "bad",
    "stink": "bad",
    "horrible": "terrible",
    "dope": "excellent",
    "sick": "excellent",
    "banging": "excellent",
    "awesome": "excellent",
    "amazin": "amazing",
    "gr8": "great",
    "gud": "good",
    "gd": "good",
    "ok": "okay",
    "okk": "okay",
    "aight": "okay",
    "useless": "useless",
    "pointless": "useless",
    "waste": "useless",
    "brutal": "very difficult unfair",
    "tough": "difficult",
    "hard af": "very difficult",
    "easy af": "very easy",
    "super": "very",
    "v ": "very ",
}


def normalise(text: str) -> str:
    """Lower-case, expand slang, normalise whitespace."""
    t = text.lower().strip()
    t = re.sub(r"[''']", "'", t)
    t = re.sub(r'[""""]', '"', t)
    t = re.sub(r'\.{2,}', '.', t)
    t = re.sub(r'!+', '!', t)
    t = re.sub(r'\?+', '?', t)
    for slang, replacement in sorted(SLANG_MAP.items(), key=lambda x: -len(x[0])):
        t = re.sub(r'\b' + re.escape(slang) + r'\b', replacement, t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: CLAUSE SPLITTING
# ════════════════════════════════════════════════════════════════════════════

CLAUSE_SPLITTERS = [
    (r'\bbut\b', True),
    (r'\bhowever\b', True),
    (r'\balthough\b', True),
    (r'\bthough\b', True),
    (r'\beven though\b', True),
    (r'\byet\b', True),
    (r'\bwhereas\b', True),
    (r'\bwhile\b', True),
    (r'\bdespite\b', True),
    (r'\bon the other hand\b', True),
    (r'\bunfortunately\b', True),
    (r'\bnevertheless\b', True),
    (r'\bnonetheless\b', True),
    (r'\bstill\b', True),
    (r'\band\b', False),
    (r'\balso\b', False),
    (r'\badditionally\b', False),
    (r'\bmoreover\b', False),
    (r'\bfurthermore\b', False),
    (r'\bas well\b', False),
    (r'\bplus\b', False),
    (r'\bbesides\b', False),
    (r'[.;]', None),
    (r'[,]', None),
]

def split_clauses(text: str) -> List[Tuple[str, Optional[bool]]]:
    annotated = text
    for pattern, is_contrast in CLAUSE_SPLITTERS:
        marker = "<<<C>>>" if is_contrast else ("<<<A>>>" if is_contrast is False else "<<<P>>>")
        annotated = re.sub(pattern, f" {marker} ", annotated, flags=re.IGNORECASE)
    
    parts = re.split(r'(<<<[CAP]>>>)', annotated)
    clauses = []
    current_contrast = None
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part == "<<<C>>>":
            current_contrast = True
        elif part == "<<<A>>>":
            current_contrast = False
        elif part == "<<<P>>>":
            current_contrast = None
        else:
            if len(part.split()) >= 2:
                clauses.append((part, current_contrast))
                current_contrast = None
    
    if not clauses:
        return [(text.lower(), None)]
    return clauses


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: ASPECT DEFINITIONS (ENHANCED)
# ════════════════════════════════════════════════════════════════════════════

ASPECTS = {
    "Teaching Quality": {
        "triggers": ["professor", "teacher", "instructor", "faculty", "lecture", "lectures", "teaches", "teaching", "explains"],
        "positive": [
            ("explains well", 2), ("explains clearly", 2), ("very knowledgeable", 2),
            ("excellent", 2), ("amazing", 2), ("fantastic", 2), ("brilliant", 2),
            ("great", 1), ("good", 1), ("clear", 1), ("knowledgeable", 2),
            ("helpful", 1), ("passionate", 2), ("approachable", 1), ("organized", 1),
            ("effective", 1), ("skilled", 1), ("friendly", 1), ("nice", 1),
        ],
        "negative": [
            ("does not explain", 3), ("doesn't explain", 3), ("never explains", 3),
            ("very poor", 3), ("terrible", 2), ("horrible", 2), ("awful", 2),
            ("worst", 3), ("poor", 2), ("bad", 1), ("confusing", 2), ("unclear", 2),
            ("boring", 2), ("rushed", 2), ("disorganized", 2), ("unhelpful", 2),
            ("dismissive", 2), ("rude", 2),
        ],
        "neutral": [
            ("okay", 1), ("decent", 1), ("average", 1), ("fine", 1), ("not bad", 1),
            ("alright", 1), ("mediocre", 1), ("sometimes", 1), ("could be better", 1),
        ],
    },

    "Assessment & Evaluation": {
        "triggers": ["assignment", "assignments", "exam", "exams", "grade", "grades", "grading", "test", "quiz", "rubric", "evaluation", "assessment", "deadline", "deadlines", "hard", "difficult", "easy"],
        "positive": [
            ("very fair", 3), ("fair", 2), ("reasonable", 2), ("transparent", 2),
            ("well-designed", 2), ("easy", 1), ("manageable", 1), ("good", 1),
            ("great", 2), ("excellent", 2), ("clear", 1), ("consistent", 1),
        ],
        "negative": [
            ("very difficult", 3), ("extremely difficult", 3), ("very hard", 3),
            ("too hard", 2), ("too difficult", 2), ("very unfair", 3),
            ("unfair", 2), ("biased", 3), ("inconsistent", 2), ("confusing", 2),
            ("unclear", 2), ("too many", 1), ("too much", 1), ("excessive", 2),
            ("overwhelming", 2), ("brutal", 2), ("terrible", 2), ("poor", 1),
            ("bad", 1), ("harsh", 2), ("difficult", 1), ("tough", 1), ("hard", 1),
            ("too tight", 2), ("tight deadline", 2),
        ],
        "neutral": [
            ("okay", 1), ("decent", 1), ("average", 1), ("fine", 1), ("not bad", 1),
            ("could be better", 1), ("somewhat", 1),
        ],
    },

    "Practical & Lab": {
        "triggers": ["lab", "labs", "laboratory", "practical", "practicals", "hands-on", "equipment", "workshop", "experiment", "dirty", "hectic", "messy", "disorganized", "clean", "well-organized"],
        "positive": [
            ("well organized", 3), ("well-organized", 3), ("very helpful", 3),
            ("excellent", 2), ("great", 2), ("good", 1), ("modern equipment", 2),
            ("functional equipment", 2), ("clean", 2), ("organized", 2),
            ("smooth", 1), ("effective", 1), ("well-equipped", 2), ("useful", 1),
            ("valuable", 1),
        ],
        "negative": [
            ("very bad", 3), ("terrible", 2), ("horrible", 2), ("awful", 2),
            ("poor", 2), ("bad", 1), ("dirty", 2), ("filthy", 2), ("hectic", 2),
            ("messy", 2), ("broken equipment", 3), ("broken", 2),
            ("outdated equipment", 2), ("outdated", 2), ("disorganized", 3),
            ("chaotic", 2), ("mess", 2), ("useless", 2), ("waste of time", 3),
            ("not helpful", 2), ("unhelpful", 2), ("no equipment", 2),
            ("lack of equipment", 2), ("insufficient", 2),
        ],
        "neutral": [
            ("okay", 1), ("decent", 1), ("average", 1), ("fine", 1),
            ("could be better", 1), ("could be improved", 1),
        ],
    },

    "Mentoring & Support": {
        "triggers": ["ta", "tas", "teaching assistant", "mentor", "support", "guidance", "office hours", "tutor", "help desk"],
        "positive": [
            ("very helpful", 3), ("extremely helpful", 3), ("always available", 3),
            ("helpful", 2), ("supportive", 2), ("great", 2), ("excellent", 2),
            ("good", 1), ("available", 1), ("responsive", 2), ("approachable", 2),
            ("friendly", 1), ("knowledgeable", 2), ("patient", 2),
        ],
        "negative": [
            ("not helpful", 3), ("very unhelpful", 3), ("never available", 3),
            ("unhelpful", 2), ("poor", 2), ("bad", 1), ("terrible", 2),
            ("unavailable", 2), ("unresponsive", 2), ("ignored", 2),
            ("no support", 3), ("lack of support", 3), ("useless", 2),
        ],
        "neutral": [
            ("okay", 1), ("decent", 1), ("average", 1), ("fine", 1),
            ("sometimes available", 2), ("could be better", 1),
        ],
    },

    "Course Design & Workload": {
        "triggers": ["workload", "heavy", "deadline", "pace", "structure", "organization", "syllabus", "course content", "overwhelming", "manageable"],
        "positive": [
            ("well structured", 3), ("well-structured", 3), ("well organized", 3),
            ("manageable", 2), ("balanced", 2), ("reasonable", 2), ("appropriate", 2),
            ("good pace", 2), ("perfect", 2), ("excellent", 2), ("great", 2),
            ("good", 1), ("clear structure", 2), ("well-paced", 2), ("light", 1),
        ],
        "negative": [
            ("extremely heavy", 3), ("way too heavy", 3), ("too heavy", 2),
            ("too much", 2), ("overwhelming", 3), ("excessive", 2),
            ("unmanageable", 3), ("unreasonable", 2), ("poor structure", 2),
            ("disorganized", 2), ("chaotic", 2), ("rushed", 2), ("too fast", 2),
            ("hectic", 2), ("very tight", 2), ("tight deadlines", 2),
            ("impossible deadlines", 3), ("heavy", 1),
        ],
        "neutral": [
            ("okay", 1), ("decent", 1), ("average", 1), ("fine", 1),
            ("could be better", 1), ("somewhat heavy", 2),
        ],
    },

    "Teaching Methodology": {
        "triggers": ["teaching style", "teaching method", "slides", "presentation", "engagement", "delivery", "interactive", "boring", "engaging"],
        "positive": [
            ("very interactive", 3), ("highly interactive", 3), ("interactive", 2),
            ("engaging", 2), ("great examples", 2), ("innovative", 2),
            ("dynamic", 2), ("effective", 1), ("excellent", 2), ("great", 2),
            ("good", 1), ("clear slides", 2), ("practical", 1), ("interesting", 1),
        ],
        "negative": [
            ("just reads slides", 3), ("reads slides", 3), ("monotonous", 2),
            ("very boring", 3), ("boring", 2), ("unengaging", 2),
            ("not engaging", 2), ("no interaction", 2), ("passive", 2),
            ("dull", 2), ("repetitive", 2), ("terrible", 2), ("poor", 2),
            ("ineffective", 2),
        ],
        "neutral": [
            ("okay", 1), ("decent", 1), ("average", 1), ("fine", 1),
            ("mixed", 1), ("could be more engaging", 2), ("not very engaging", 2),
            ("somewhat engaging", 2),
        ],
    },

    "Learning Outcomes": {
        "triggers": ["learned", "learning", "skills", "knowledge", "understanding", "outcome", "useful", "valuable", "relevant"],
        "positive": [
            ("learned a lot", 3), ("great learning", 2), ("very useful", 3),
            ("very valuable", 3), ("applicable", 2), ("practical knowledge", 2),
            ("relevant", 2), ("improved", 2), ("gained skills", 2), ("valuable", 2),
            ("useful", 1), ("good", 1), ("great", 2), ("meaningful", 2),
        ],
        "negative": [
            ("waste of time", 3), ("nothing learned", 3), ("learned nothing", 3),
            ("not useful", 3), ("not relevant", 2), ("irrelevant", 3),
            ("useless", 2), ("pointless", 2), ("forgettable", 2),
            ("no improvement", 2), ("waste", 2),
        ],
        "neutral": [
            ("okay", 1), ("decent", 1), ("average", 1), ("fine", 1),
            ("somewhat useful", 2), ("could be more relevant", 2),
        ],
    },
}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: NEGATION DETECTION
# ════════════════════════════════════════════════════════════════════════════

NEGATION_TRIGGERS = [
    "not", "no", "never", "don't", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "won't", "wouldn't", "can't", "cannot", "couldn't",
    "shouldn't", "hardly", "barely", "scarcely", "without", "lack", "lacks",
]

def has_negation_before(phrase: str, full_clause: str, window: int = 4) -> bool:
    phrase_pos = full_clause.find(phrase)
    if phrase_pos == -1:
        return False
    context = full_clause[max(0, phrase_pos - 60):phrase_pos]
    words = context.split()
    nearby = words[-window:] if len(words) >= window else words
    return any(neg in nearby for neg in NEGATION_TRIGGERS)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: CLAUSE-LEVEL ASPECT SCORING
# ════════════════════════════════════════════════════════════════════════════

def score_clause_for_aspect(clause: str, aspect_config: dict) -> Tuple[float, float, float]:
    pos_score = 0.0
    neg_score = 0.0
    neutral_score = 0.0

    for phrase, weight in sorted(aspect_config["positive"], key=lambda x: -len(x[0])):
        if re.search(r'\b' + re.escape(phrase) + r'\b', clause):
            if has_negation_before(phrase, clause):
                neg_score += weight * 0.85
            else:
                pos_score += weight

    for phrase, weight in sorted(aspect_config["negative"], key=lambda x: -len(x[0])):
        if re.search(r'\b' + re.escape(phrase) + r'\b', clause):
            if has_negation_before(phrase, clause):
                pos_score += weight * 0.70
            else:
                neg_score += weight

    for phrase, weight in sorted(aspect_config["neutral"], key=lambda x: -len(x[0])):
        if re.search(r'\b' + re.escape(phrase) + r'\b', clause):
            neutral_score += weight

    return pos_score, neg_score, neutral_score


def is_aspect_triggered(clause: str, triggers: List[str]) -> bool:
    for trigger in triggers:
        if ' ' in trigger:
            if trigger in clause:
                return True
        else:
            if re.search(r'\b' + re.escape(trigger) + r'\b', clause):
                return True
    return False


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: SENTIMENT DECISION ENGINE
# ════════════════════════════════════════════════════════════════════════════

def decide_sentiment(pos: float, neg: float, neutral: float) -> Tuple[str, float, int]:
    total = pos + neg + neutral
    if total == 0:
        return "Neutral", 0.60, 50

    diff = pos - neg

    # Determine sentiment
    if pos > neg * 1.5 and pos >= 1:
        sentiment = "Positive"
        confidence = min(0.92, 0.75 + (diff / (total + 1)) * 0.20)
        score = int(min(95, 65 + pos * 5))
    elif neg > pos * 1.5 and neg >= 1:
        sentiment = "Negative"
        confidence = min(0.92, 0.75 + ((-diff) / (total + 1)) * 0.20)
        score = int(max(10, 40 - neg * 5))
    elif neutral > max(pos, neg) and neutral >= 1:
        sentiment = "Neutral"
        confidence = min(0.80, 0.65 + neutral * 0.05)
        score = 50
    elif abs(diff) <= 0.8:
        sentiment = "Neutral"
        confidence = 0.65
        score = 50
    elif pos > neg:
        sentiment = "Positive"
        confidence = 0.70
        score = 62
    else:
        sentiment = "Negative"
        confidence = 0.70
        score = 38

    return sentiment, round(confidence, 3), max(5, min(95, score))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN ANALYSIS FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def analyze_all_aspects(text: str) -> Dict[str, Dict]:
    if not text or len(text.strip()) < 3:
        return {}

    norm = normalise(text)
    clauses = split_clauses(norm)
    
    aspect_scores: Dict[str, List[Tuple[float, float, float]]] = {}
    
    for clause_text, is_contrast in clauses:
        clause_text = clause_text.strip()
        if not clause_text:
            continue
        
        # Check which aspects are triggered
        triggered_aspects = []
        for aspect, config in ASPECTS.items():
            if is_aspect_triggered(clause_text, config["triggers"]):
                triggered_aspects.append(aspect)
        
        # If no aspects triggered, try to infer from keywords
        if not triggered_aspects:
            for aspect, config in ASPECTS.items():
                for trigger in config["triggers"][:3]:  # Check first few triggers
                    if trigger in clause_text:
                        triggered_aspects.append(aspect)
                        break
        
        # Score each triggered aspect
        for aspect in triggered_aspects:
            config = ASPECTS[aspect]
            pos, neg, neutral = score_clause_for_aspect(clause_text, config)
            
            # Apply contrast flip if needed
            if is_contrast:
                pos, neg = neg, pos
            
            if aspect not in aspect_scores:
                aspect_scores[aspect] = []
            aspect_scores[aspect].append((pos, neg, neutral))
    
    # Aggregate results
    results = {}
    for aspect, scores in aspect_scores.items():
        total_pos = sum(s[0] for s in scores)
        total_neg = sum(s[1] for s in scores)
        total_neutral = sum(s[2] for s in scores)
        
        sentiment, confidence, score = decide_sentiment(total_pos, total_neg, total_neutral)
        results[aspect] = {"sentiment": sentiment, "confidence": confidence, "score": score}
    
    # Fallback
    if not results:
        # Check for general sentiment
        pos_words = ["good", "great", "excellent", "nice", "helpful", "useful"]
        neg_words = ["bad", "poor", "terrible", "awful", "horrible", "useless"]
        
        pos = sum(1 for w in pos_words if w in norm)
        neg = sum(1 for w in neg_words if w in norm)
        
        if pos > neg:
            results["General Feedback"] = {"sentiment": "Positive", "confidence": 0.75, "score": 70}
        elif neg > pos:
            results["General Feedback"] = {"sentiment": "Negative", "confidence": 0.75, "score": 30}
        elif pos > 0 or neg > 0:
            results["General Feedback"] = {"sentiment": "Neutral", "confidence": 0.65, "score": 50}
    
    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8: HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

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
            insights["recommendations"].append(f"Review and improve {aspect}")
    return insights


def calculate_overall_score(results):
    if not results:
        return 50
    scores = [data['score'] for data in results.values()]
    return sum(scores) // len(scores)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9: TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        ("The professor explains well but assignments are too hard", 
         {"Teaching Quality": "Positive", "Assessment & Evaluation": "Negative"}),
        ("the professor is good and helpful. the labs are hectic and dirty. the exams are very hard",
         {"Teaching Quality": "Positive", "Practical & Lab": "Negative", "Assessment & Evaluation": "Negative"}),
        ("Great teaching, terrible labs, helpful TAs",
         {"Teaching Quality": "Positive", "Practical & Lab": "Negative", "Mentoring & Support": "Positive"}),
    ]
    
    print("=" * 60)
    print("ROBUST ANALYZER TEST")
    print("=" * 60)
    
    for text, expected in test_cases:
        print(f"\nINPUT: {text}")
        results = analyze_all_aspects(text)
        for aspect, data in results.items():
            expected_sent = expected.get(aspect, "?")
            mark = "✓" if expected.get(aspect) == data['sentiment'] else " "
            print(f"  {mark} {aspect}: {data['sentiment']} (conf: {data['confidence']:.2f}, score: {data['score']})")