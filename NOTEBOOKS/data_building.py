# -*- coding: utf-8 -*-
"""
## Data Preparation and Synthesis
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

file_path = '/content/drive/MyDrive/Colab Notebooks/raw_feedback.csv'
df = pd.read_csv(file_path)

df.head()

import pandas as pd
import numpy as np
import random
import hashlib
from datetime import datetime

N = len(df)
faculty_list = [f"Faculty_{i}" for i in range(1, 11)]  # 10 faculties
course_types = ["T", "P", "J"]

# Year distribution
year_probs = [0.22, 0.26, 0.28, 0.24]
years = [1, 2, 3, 4]

# Monthly distribution
months = list(range(1, 13))
month_probs = np.random.dirichlet(np.ones(12), size=1)[0]


def generate_hash_id(text, idx):
    return hashlib.md5(f"{text}_{idx}".encode()).hexdigest()

def get_semester(year):
    return random.choice([year*2 - 1, year*2])

def get_course_type(year):
    if year <= 2:
        return "T"
    elif year == 3:
        return random.choices(["T", "J", "P"], weights=[0.4, 0.3, 0.3])[0]
    else:
        return "P"

def generate_course_code(year, ctype):
    return f"CSE{year}{random.randint(0,9)}{random.randint(0,9)}{ctype}"

def generate_timestamp():
    month = np.random.choice(months, p=month_probs)
    day = random.randint(1, 28)
    return datetime(2024, month, day)


df = df.sample(n=N, replace=True).reset_index(drop=True)

rows = []

for i, row in df.iterrows():
    text = row["text"]

    year = np.random.choice(years, p=year_probs)
    semester = get_semester(year)
    course_type = get_course_type(year)
    course_code = generate_course_code(year, course_type)

    new_row = {
        "feedback_id": generate_hash_id(text, i),
        "feedback_text": text,
        "course_code": course_code,
        "course_type": course_type,
        "faculty_name": random.choice(faculty_list),
        "year": year,
        "semester": semester,
        "timestamp": generate_timestamp()
    }

    rows.append(new_row)

final_df = pd.DataFrame(rows)

print(final_df.head())

"""## Aspect and Sentiment Analysis"""

aspect_dict = {

    "Instructional Quality": [
        "explain","explained","explanation","clarity","clear",
        "confusing","understand","understood","concept","theory",
        "expertise","lecture","lecturer","topic","material"
    ],

    "Teaching Methodology": [
        "example","examples","illustration","ppt","slides",
        "board","presentation","interactive","engaging",
        "discussion","activity","teaching","delivery",
        "video","method"
    ],

    "Assessment & Evaluation": [
        "exam","assessment","test","quiz","grading","marks",
        "marking","evaluation","fair","unfair","transparent",
        "criteria","rubric","midterm","final","assignment",
        "homework","difficulty"
    ],

    "Mentoring & Academic Support": [
        "mentor","guidance","support","helpful","available",
        "approachable","responsive","feedback","suggestion",
        "clarify","supervision","advisor"
    ],

    "Practical & Lab Integration": [
        "lab","laboratory","practical","experiment","equipment",
        "hands-on","session","viva","implementation",
        "project","tool","software","coding","programming"
    ],

    "Course Design & Workload": [
        "workload","structure","organization","pace",
        "fast","slow","rushed","balanced","schedule",
        "timing","week","deadline","duration"
    ],

    "Learning Outcomes & Skill Development": [
        "skill","learning","learned","outcome","knowledge",
        "improvement","growth","development","experience",
        "useful","valuable","recommend"
    ]
}

def get_aspect(sentence):
    sentence = sentence.lower()
    scores = {}

    for aspect, keywords in aspect_dict.items():
        count = sum(1 for word in keywords if word in sentence)
        if count > 0:
            scores[aspect] = count

    if scores:
        return max(scores, key=scores.get)
    else:
        return "General"

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

# initialize
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']

    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

final_df['sentences'] = final_df['feedback_text'].fillna('').apply(nltk.sent_tokenize)

rows = []

for _, row in final_df.iterrows():
    for sent in row['sentences']:

        score = sia.polarity_scores(sent)['compound']

        rows.append({
            "feedback_id": row["feedback_id"],
            "course_type": row["course_type"],
            "sentence": sent,
            "aspect": get_aspect(sent),
            "sentiment": get_sentiment(sent),
            "sentiment_score": score
        })

aspect_df = pd.DataFrame(rows)

aspect_df = aspect_df.groupby("feedback_id").agg({
    "aspect": lambda x: x.mode()[0] if not x.mode().empty else "General",
    "sentiment": lambda x: x.mode()[0] if not x.mode().empty else "Neutral",
    "sentiment_score": "mean"
}).reset_index()

aspect_map = {
    "Instructional Quality": "Teaching",
    "Teaching Methodology": "Teaching",
    "Assessment & Evaluation": "Evaluation",
    "Mentoring & Academic Support": "Support",
    "Practical & Lab Integration": "Lab",
    "Course Design & Workload": "Workload",
    "Learning Outcomes & Skill Development": "Outcome",
    "General": "General"
}

aspect_df["aspect"] = aspect_df["aspect"].map(aspect_map).fillna("General")

print(aspect_df["aspect"].value_counts())

final_df = final_df.merge(
    aspect_df[["feedback_id", "aspect", "sentiment", "sentiment_score"]],
    on="feedback_id",
    how="left"
)

"""## MCQ Generation"""

import pandas as pd
import random

mapping = {
    "T": {
        "Teaching": ["Q1","Q2","Q3"],
        "Interaction": ["Q4","Q5"],
        "Resources": ["Q6"],
        "Evaluation": ["Q7","Q8"],
        "Workload": ["Q9"],
        "Satisfaction": ["Q10"]
    },
    "P": {
        "Teaching": ["Q1","Q2"],
        "Support": ["Q3","Q4"],
        "Feedback": ["Q5"],
        "Innovation": ["Q6"],
        "Planning": ["Q7"],
        "Evaluation": ["Q8"],
        "Outcome": ["Q9"],
        "Satisfaction": ["Q10"]
    },
    "J": {
        "Teaching": ["Q1","Q2","Q3"],
        "Lab": ["Q4","Q5","Q6"],
        "Evaluation": ["Q7","Q8"],
        "Infrastructure": ["Q9"],
        "Workload": ["Q10"]
    }
}


def sentiment_to_mcq(score):

    if score >= 0.6:
        return random.choices(
            ["Strongly Agree", "Agree"],
            weights=[0.8, 0.2]
        )[0]

    elif score >= 0.2:
        return random.choices(
            ["Agree", "Strongly Agree"],
            weights=[0.7, 0.3]
        )[0]

    elif score > -0.2:
        return "Neutral"

    elif score > -0.6:
        return random.choices(
            ["Disagree", "Strongly Disagree"],
            weights=[0.7, 0.3]
        )[0]

    else:
        return random.choices(
            ["Strongly Disagree", "Disagree"],
            weights=[0.8, 0.2]
        )[0]

def generate_mcqs(df):

    for i in range(1, 11):
        df[f"MCQ_Q{i}"] = None

    for idx, row in df.iterrows():
        ctype = row["course_type"]
        aspect = row["aspect"]
        score = row["sentiment_score"]

        if pd.isna(score):
            score = 0

        if ctype in mapping and aspect in mapping[ctype]:
            for q in mapping[ctype][aspect]:
                df.at[idx, f"MCQ_{q}"] = sentiment_to_mcq(score)

        for i in range(1, 11):
            col = f"MCQ_Q{i}"
            if pd.isna(df.at[idx, col]):
                df.at[idx, col] = random.choices(
                    ["Agree", "Neutral", "Disagree"],
                    weights=[0.4, 0.3, 0.3]
                )[0]

    return df

final_df = generate_mcqs(final_df)

final_df.head()

"""## Data Distributions and Analysis"""

final_df["aspect"].value_counts()



mcq_cols = [f"MCQ_Q{i}" for i in range(1, 11)]

all_responses = final_df[mcq_cols].values.flatten()

pd.Series(all_responses).value_counts()

"""## Export ready data"""

analysis_df = final_df.copy()

final_submission_df = final_df.drop(
    columns=["sentences", "aspect", "sentiment", "sentiment_score"]
)

file_name = "final_feedback_dataset.csv"
final_submission_df.to_csv(file_name, index=False)

"""## Further analysis"""

for i in range(1, 11):
    print(f"\nMCQ_Q{i}")
    print(final_df[f"MCQ_Q{i}"].value_counts(normalize=True))

print(aspect_df["aspect"].value_counts())

print(final_df["course_type"].value_counts())

final_df["sentiment_score"].describe()

pd.crosstab(final_df["aspect"], final_df["MCQ_Q1"])

final_df.groupby("aspect")["sentiment_score"].mean()

final_df.groupby("aspect")["sentiment_score"].mean().sort_values()

final_df.groupby(["course_type", "aspect"])["sentiment_score"].mean()

final_df["MCQ_Q9"].value_counts()

