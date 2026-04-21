# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

file_path = '/content/drive/MyDrive/Colab Notebooks/final_feedback_dataset.csv'
df = pd.read_csv(file_path)

df.head()

"""# KEYWORD BASED"""

df_keyword = df.copy()

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

def extract_multi_aspects(text):
    if pd.isna(text):
        return []

    text = str(text).lower()
    aspects_found = []

    for aspect, keywords in aspect_dict.items():
        if any(keyword in text for keyword in keywords):
            aspects_found.append(aspect)

    return aspects_found if aspects_found else ["General"]

df_keyword['aspects'] = df_keyword['feedback_text'].apply(extract_multi_aspects)
df_keyword['num_aspects'] = df_keyword['aspects'].apply(len)

df_exploded = df_keyword.explode('aspects').reset_index(drop=True)
df_exploded = df_exploded.rename(columns={'aspects': 'aspect'})
df_exploded.to_csv('aspect_results.csv', index=False)
print("\n\nSaved to aspect_results.csv")
print(df_exploded[['feedback_text', 'aspect']].head(10))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

"""# LDA BASED

"""

df_lda = df.copy()

import pandas as pd
import re
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logging.info("🚀 Starting LDA pipeline")

logging.info("Step 0: Data copied")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

logging.info("Step 1: Cleaning text...")
df_lda['clean_text'] = df_lda['feedback_text'].apply(clean_text)
logging.info("Step 1 completed")

logging.info("Step 2: Creating document-term matrix...")
vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
X = vectorizer.fit_transform(df_lda['clean_text'])
logging.info(f"Step 2 completed | Shape: {X.shape}")

n_topics = 7
logging.info(f"Step 3: Training LDA with {n_topics} topics...")
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)
logging.info("Step 3 completed")

logging.info("Step 4: Getting topic distribution...")
topic_distribution = lda.transform(X)
logging.info(f"Step 4 completed | Shape: {topic_distribution.shape}")

logging.info("Step 5: Printing topic keywords for manual mapping")

feature_names = vectorizer.get_feature_names_out()

print("\nTOPIC MAPPING - Check what each topic represents:")
print("="*50)

for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-10:]]
    print(f"\nTopic {topic_idx}:")
    print(f"Top words: {', '.join(top_words)}")
    print("Suggested aspect: _______")

logging.info("Step 5 completed - Update mapping manually before proceeding")

topic_to_aspect = {
    0: "Learning Outcomes & Skill Development",
    1: "Instructional Quality",
    2: "Teaching Methodology",
    3: "Practical & Lab Integration",
    4: "Assessment & Evaluation",
    5: "Course Design & Workload",
    6: "Mentoring & Academic Support"
}

logging.info("Step 6: Topic mapping applied")

logging.info("Step 7: Assigning primary aspect...")
df_lda['primary_aspect_raw'] = topic_distribution.argmax(axis=1)
df_lda['primary_aspect'] = df_lda['primary_aspect_raw'].map(topic_to_aspect)
logging.info("Step 7 completed")

logging.info("Step 8: Extracting multiple aspects...")

threshold = 0.15

def get_multiple_aspects(probabilities):
    """Extract multiple aspects above threshold"""
    aspects = []
    for topic_idx, prob in enumerate(probabilities):
        if prob >= threshold:
            aspect_name = topic_to_aspect.get(topic_idx, "General")
            if aspect_name not in aspects:
                aspects.append(aspect_name)
    return aspects if aspects else ["General"]

df_lda['aspects'] = [get_multiple_aspects(row) for row in topic_distribution]
df_lda['num_aspects'] = df_lda['aspects'].apply(len)

logging.info(f"Step 8 completed | Average aspects: {df_lda['num_aspects'].mean():.2f}")

logging.info("Step 9: Displaying results")

print("\n" + "="*50)
print("RESULTS:")
print("="*50)

print("\nSample results with multiple aspects:")
print(df_lda[['feedback_text', 'aspects', 'num_aspects']].head(10))

print("\nAspect distribution (exploded):")
aspect_counts = df_lda.explode('aspects')['aspects'].value_counts()
print(aspect_counts)

print("\nNumber of aspects per feedback:")
print(df_lda['num_aspects'].value_counts().sort_index())

print(f"\nStatistics:")
print(f"  - Total feedbacks: {len(df_lda)}")
print(f"  - Average aspects per feedback: {df_lda['num_aspects'].mean():.2f}")
print(f"  - Max aspects in one feedback: {df_lda['num_aspects'].max()}")
print(f"  - Feedbacks with multiple aspects: {(df_lda['num_aspects'] > 1).sum()}")

print("\nExamples of feedbacks with MULTIPLE aspects:")
multi_aspect_examples = df_lda[df_lda['num_aspects'] > 1][['feedback_text', 'aspects', 'num_aspects']].head(5)
for idx, row in multi_aspect_examples.iterrows():
    print(f"\n  Feedback: {row['feedback_text'][:120]}...")
    print(f"  Aspects ({row['num_aspects']}): {row['aspects']}")

logging.info("Step 10: Saving files...")

df_lda.to_csv('lda_aspect_results.csv', index=False)
print("\nSaved: lda_aspect_results.csv")

df_lda_exploded = df_lda.explode('aspects').reset_index(drop=True)
df_lda_exploded.to_csv('lda_aspect_results_exploded.csv', index=False)
print("Saved: lda_aspect_results_exploded.csv")

aspect_summary = df_lda_exploded.groupby('aspects').size().sort_values(ascending=False)
aspect_summary.to_csv('lda_aspect_summary.csv')
print("Saved: lda_aspect_summary.csv")

logging.info("LDA pipeline completed successfully")

"""# Embeddings + Clustering"""

import pandas as pd
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import re
import torch

df_cluster = df.copy()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

logging.info("Step 1: Cleaning text...")
df_cluster['clean_text'] = df_cluster['feedback_text'].apply(clean_text)
logging.info("Step 1 completed")

logging.info("Step 2: Creating embeddings using Sentence Transformer on GPU...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
embeddings = model.encode(df_cluster['clean_text'].tolist(), show_progress_bar=True, device=device)
logging.info(f"Step 2 completed | Embeddings shape: {embeddings.shape}")

n_clusters = 7
logging.info(f"Step 3: Applying K-Means with {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(embeddings)
logging.info("Step 3 completed")

print("\n" + "="*60)
print("CLUSTER ANALYSIS - Identify what each cluster represents")
print("="*60)

cluster_keywords = {}

for cluster_num in range(n_clusters):

    cluster_indices = df_cluster[df_cluster['cluster'] == cluster_num].index.tolist()

    sample_texts = df_cluster.loc[cluster_indices, 'feedback_text'].head(3).tolist()

    all_text = ' '.join(df_cluster.loc[cluster_indices, 'clean_text'].head(20))
    words = all_text.split()

    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'with', 'by', 'was', 'were', 'is', 'are', 'this', 'that', 'these',
                 'those', 'it', 'they', 'we', 'you', 'he', 'she', 'i', 'me', 'my', 'your'}

    word_freq = {}
    for word in words:
        if word not in stopwords and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1

    top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:8]
    cluster_keywords[cluster_num] = [word for word, freq in top_keywords]

    print(f"\nCLUSTER {cluster_num}:")
    print(f"   Number of feedbacks: {len(cluster_indices)}")
    print(f"   Top keywords: {', '.join(cluster_keywords[cluster_num][:6])}")
    print(f"   Sample feedbacks:")
    for sample in sample_texts[:2]:
        print(f"     - {sample[:100]}...")
    print("   Suggested aspect: _______")


cluster_to_aspect = {
    0: "Instructional Quality",
    1: "Teaching Methodology",
    2: "Assessment & Evaluation",
    3: "Mentoring & Academic Support",
    4: "Practical & Lab Integration",
    5: "Course Design & Workload",
    6: "Learning Outcomes & Skill Development"
}

df_cluster['primary_aspect'] = df_cluster['cluster'].map(cluster_to_aspect)

logging.info("Step 6: Extracting multiple aspects...")

distances_to_centers = kmeans.transform(embeddings)

num_aspects_per_feedback = 2

def get_closest_aspects(distances_row, cluster_map, k=num_aspects_per_feedback):
    """Get k closest clusters as aspects"""
    closest_indices = np.argsort(distances_row)[:k]
    aspects = []
    for idx in closest_indices:
        aspect_name = cluster_map.get(idx, "General")
        if aspect_name != "General" and aspect_name not in aspects:
            aspects.append(aspect_name)
    return aspects if aspects else ["General"]

df_cluster['aspects'] = [
    get_closest_aspects(row, cluster_to_aspect, k=num_aspects_per_feedback)
    for row in distances_to_centers
]
df_cluster['num_aspects'] = df_cluster['aspects'].apply(len)

logging.info(f"Step 6 completed | Average aspects: {df_cluster['num_aspects'].mean():.2f}")

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print("\nSample results:")
print(df_cluster[['feedback_text', 'aspects', 'num_aspects']].head(10))

print("\nAspect distribution (exploded):")
df_cluster_exploded = df_cluster.explode('aspects')
aspect_counts = df_cluster_exploded['aspects'].value_counts()
print(aspect_counts)

print("\nNumber of aspects per feedback:")
print(df_cluster['num_aspects'].value_counts().sort_index())

print(f"\nStatistics:")
print(f"  - Total feedbacks: {len(df_cluster):,}")
print(f"  - Average aspects per feedback: {df_cluster['num_aspects'].mean():.2f}")
print(f"  - Max aspects in one feedback: {df_cluster['num_aspects'].max()}")
print(f"  - Feedbacks with multiple aspects: {(df_cluster['num_aspects'] > 1).sum():,}")

print("\nExamples of feedbacks with MULTIPLE aspects:")
multi_aspect_examples = df_cluster[df_cluster['num_aspects'] > 1][['feedback_text', 'aspects', 'num_aspects']].head(5)
for idx, row in multi_aspect_examples.iterrows():
    print(f"\n  Feedback: {row['feedback_text'][:120]}...")
    print(f"  Aspects ({row['num_aspects']}): {row['aspects']}")

print("\nSaving files...")

df_cluster.to_csv('cluster_aspect_results.csv', index=False)
print("Saved: cluster_aspect_results.csv")

df_cluster_exploded = df_cluster.explode('aspects').reset_index(drop=True)
df_cluster_exploded.to_csv('cluster_aspect_results_exploded.csv', index=False)
print("Saved: cluster_aspect_results_exploded.csv")

cluster_mapping_df = pd.DataFrame([
    {"cluster": k, "aspect": v, "keywords": ', '.join(cluster_keywords.get(k, []))}
    for k, v in cluster_to_aspect.items()
])
cluster_mapping_df.to_csv('cluster_to_aspect_mapping.csv', index=False)
print("Saved: cluster_to_aspect_mapping.csv")

aspect_summary = df_cluster_exploded.groupby('aspects').size().sort_values(ascending=False)
aspect_summary.to_csv('cluster_aspect_summary.csv', index=False)
print("Saved: cluster_aspect_summary.csv")

torch.cuda.empty_cache()

"""# Transformer-Based Aspect Assignmen"""

import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

!pip install -q sentence-transformers

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

df_transformer = df.copy()

model = SentenceTransformer('all-MiniLM-L6-V2', device=device)

aspect_labels = [
    "Instructional Quality",
    "Teaching Methodology",
    "Assessment & Evaluation",
    "Mentoring & Academic Support",
    "Practical & Lab Integration",
    "Course Design & Workload",
    "Learning Outcomes & Skill Development"
]

aspect_descriptions = [
    "teacher explanation clarity concepts easy to understand lectures quality",
    "teaching methods slides examples presentations interactive engaging teaching style",
    "exams assignments grading marks fairness difficulty evaluation tests quiz",
    "teacher support mentoring guidance feedback approachable helpful responsive",
    "labs practical coding programming projects hands on implementation lab sessions",
    "course workload assignments deadlines pace fast slow schedule timing pressure",
    "learning skills knowledge improvement growth useful experience outcomes development"
]

aspect_embeddings = model.encode(aspect_descriptions, convert_to_numpy=True, device=device)

batch_size = 5000
threshold = 0.45

all_aspects = []
all_confidences = []

print(f"Processing {len(df_transformer)} feedbacks on GPU...")
print(f"Batch size: {batch_size}")

for start in tqdm(range(0, len(df_transformer), batch_size)):
    end = min(start + batch_size, len(df_transformer))


    batch_texts = df_transformer['feedback_text'].iloc[start:end].fillna('').tolist()

    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, device=device, show_progress_bar=False)

    similarities = cosine_similarity(batch_embeddings, aspect_embeddings)

    for sim_row in similarities:
        aspects = [aspect_labels[i] for i, sim in enumerate(sim_row) if sim >= threshold]
        if not aspects:
            aspects = ["General"]

        confidences = {aspect_labels[i]: float(sim) for i, sim in enumerate(sim_row) if sim >= threshold}

        all_aspects.append(aspects)
        all_confidences.append(confidences)

df_transformer['aspects'] = all_aspects
df_transformer['aspect_confidences'] = all_confidences
df_transformer['num_aspects'] = df_transformer['aspects'].apply(len)

print("\n" + "="*60)
print("TRANSFORMER MULTI-ASPECT RESULTS (GPU)")
print("="*60)

print("\nSample results:")
print(df_transformer[['feedback_text', 'aspects', 'num_aspects']].head(10))

print("\nAspect distribution:")
aspect_counts = df_transformer.explode('aspects')['aspects'].value_counts()
print(aspect_counts)

print("\nNumber of aspects per feedback:")
print(df_transformer['num_aspects'].value_counts().sort_index())

print(f"\nStatistics:")
print(f"  - Total feedbacks: {len(df_transformer)}")
print(f"  - Average aspects per feedback: {df_transformer['num_aspects'].mean():.2f}")
print(f"  - Max aspects in one feedback: {df_transformer['num_aspects'].max()}")
print(f"  - Feedbacks with multiple aspects: {(df_transformer['num_aspects'] > 1).sum()}")

print("\nExamples of feedbacks with MULTIPLE aspects:")
multi_aspect_examples = df_transformer[df_transformer['num_aspects'] > 1][['feedback_text', 'aspects', 'num_aspects']].head(5)
for idx, row in multi_aspect_examples.iterrows():
    print(f"\n  Feedback: {row['feedback_text'][:120]}...")
    print(f"  Aspects ({row['num_aspects']}): {row['aspects']}")

print("\nSaving files...")

df_transformer.to_csv('transformer_multi_aspect_results.csv', index=False)
print("Saved: transformer_multi_aspect_results.csv")

df_transformer_exploded = df_transformer.explode('aspects').reset_index(drop=True)
df_transformer_exploded.to_csv('transformer_multi_aspect_exploded.csv', index=False)
print("Saved: transformer_multi_aspect_exploded.csv")

aspect_summary = df_transformer_exploded.groupby('aspects').size().sort_values(ascending=False)
aspect_summary.to_csv('transformer_aspect_summary.csv')
print("Saved: transformer_aspect_summary.csv")

print("\n" + "="*60)
print("TRANSFORMER MULTI-ASPECT COMPLETE!")
print("="*60)

torch.cuda.empty_cache()

import pandas as pd
import ast
import re
from tqdm import tqdm
from collections import Counter

df = pd.read_csv('/content/transformer_multi_aspect_results.csv')

def parse_aspects(x):
    if pd.isna(x):
        return ["General"]
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return ["General"]
    return x if isinstance(x, list) else ["General"]

df['aspects_parsed'] = df['aspects'].apply(parse_aspects)

df_general = df[df['aspects_parsed'].apply(lambda x: 'General' in x)].copy()
df_specific = df[~df['aspects_parsed'].apply(lambda x: 'General' in x)].copy()

print("----------------------------------------------------------------------")
print("INITIAL STATUS")
print("----------------------------------------------------------------------")
print(f"Specific feedbacks: {len(df_specific):,} ({len(df_specific)/len(df)*100:.1f}%)")
print(f"General feedbacks: {len(df_general):,} ({len(df_general)/len(df)*100:.1f}%)")

aspect_keywords = {

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

def extract_aspects_fast(text, max_aspects=3):
    """Fast keyword-based aspect extraction"""
    if pd.isna(text) or text == "":
        return ["General"]

    text_lower = text.lower()
    aspects = set()


    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                aspects.add(aspect)
                break

    aspects = list(aspects)[:max_aspects]

    return aspects if aspects else ["General"]

print("\n----------------------------------------------------------------------")
print("REPROCESSING GENERAL FEEDBACKS (CPU OPTIMIZED)")
print("----------------------------------------------------------------------")

reprocessed_results = []

for idx, row in tqdm(df_general.iterrows(), total=len(df_general), desc="Processing"):
    aspects = extract_aspects_fast(row['feedback_text'], max_aspects=3)
    reprocessed_results.append(aspects)

df_general['aspects_reprocessed'] = reprocessed_results
df_general['num_aspects_new'] = df_general['aspects_reprocessed'].apply(len)


df_general['still_general'] = df_general['aspects_reprocessed'].apply(lambda x: 'General' in x)
df_general['now_specific'] = ~df_general['still_general']

still_general = df_general[df_general['still_general'] == True].copy()
now_specific = df_general[df_general['now_specific'] == True].copy()

print(f"\nNow specific: {len(now_specific):,} feedbacks")
print(f"Still General: {len(still_general):,} feedbacks")



reprocessed_export = now_specific.copy()
reprocessed_export['original_aspects'] = reprocessed_export['aspects_parsed']
reprocessed_export['new_aspects'] = reprocessed_export['aspects_reprocessed']
reprocessed_export['num_new_aspects'] = reprocessed_export['num_aspects_new']

excel_columns = [
    'feedback_id', 'feedback_text', 'course_code', 'course_type',
    'original_aspects', 'new_aspects', 'num_new_aspects'
]

reprocessed_export[excel_columns].to_excel('reprocessed_feedbacks.xlsx', index=False)
print("\nSaved: reprocessed_feedbacks.xlsx")


still_general_export = still_general.copy()
still_general_export['original_aspects'] = still_general_export['aspects_parsed']
still_general_export[['feedback_id', 'feedback_text', 'course_code', 'original_aspects']].to_excel(
    'still_general_manual_review.xlsx', index=False
)
print("Saved: still_general_manual_review.xlsx")

df_final_specific = df_specific.copy()
df_final_specific['aspects_final'] = df_final_specific['aspects_parsed']
df_final_specific['method'] = 'transformer_original'

df_final_reprocessed = now_specific.copy()
df_final_reprocessed['aspects_final'] = df_final_reprocessed['aspects_reprocessed']
df_final_reprocessed['method'] = 'reprocessed_keyword'

df_final_still = still_general.copy()
df_final_still['aspects_final'] = [["General"]] * len(df_final_still)
df_final_still['method'] = 'still_general'


df_all = pd.concat([
    df_final_specific[['feedback_id', 'feedback_text', 'course_code', 'aspects_final', 'method']],
    df_final_reprocessed[['feedback_id', 'feedback_text', 'course_code', 'aspects_final', 'method']],
    df_final_still[['feedback_id', 'feedback_text', 'course_code', 'aspects_final', 'method']]
], ignore_index=True)

df_all.to_excel('complete_aspects_summary.xlsx', index=False)
print("Saved: complete_aspects_summary.xlsx")

print("\n----------------------------------------------------------------------")
print("FINAL STATISTICS DASHBOARD")
print("----------------------------------------------------------------------")

print(f"\nTotal feedbacks: {len(df):,}")

print(f"\nFINAL ASPECT DISTRIBUTION:")
df_all_exploded = df_all.explode('aspects_final')
aspect_dist = df_all_exploded['aspects_final'].value_counts()
for aspect, count in aspect_dist.items():
    pct = (count / len(df_all_exploded)) * 100
    print(f"   {aspect}: {count:,} ({pct:.1f}%)")

print(f"\nMETHOD BREAKDOWN:")
print(f"   Transformer original (specific): {len(df_final_specific):,}")
print(f"   Reprocessed (now specific): {len(df_final_reprocessed):,}")
print(f"   Still General: {len(df_final_still):,}")

print(f"\nIMPROVEMENT:")
general_before = len(df_general)
general_after = len(still_general)
improvement = ((general_before - general_after) / general_before) * 100 if general_before > 0 else 0
print(f"   General feedbacks reduced by: {improvement:.1f}%")
print(f"   Before: {general_before:,} -> After: {general_after:,}")


try:
    with pd.ExcelWriter('complete_aspect_analysis.xlsx', engine='openpyxl') as writer:

        pd.DataFrame({
            'Metric': ['Total Feedbacks', 'Original Specific', 'Reprocessed (Now Specific)', 'Still General', 'Improvement %'],
            'Value': [len(df), len(df_final_specific), len(df_final_reprocessed), len(df_final_still), f"{improvement:.1f}%"]
        }).to_excel(writer, sheet_name='Overview', index=False)


        aspect_dist_df = pd.DataFrame({
            'Aspect': aspect_dist.index,
            'Count': aspect_dist.values,
            'Percentage': [(v/len(df_all_exploded))*100 for v in aspect_dist.values]
        })
        aspect_dist_df.to_excel(writer, sheet_name='Aspect_Distribution', index=False)

        reprocessed_export[excel_columns].to_excel(writer, sheet_name='Reprocessed_Feedbacks', index=False)
        still_general_export[['feedback_id', 'feedback_text', 'course_code', 'original_aspects']].to_excel(
            writer, sheet_name='Still_General', index=False
        )

        df_all.to_excel(writer, sheet_name='Complete_Summary', index=False)

    print("\nSaved: complete_aspect_analysis.xlsx (with multiple sheets)")
except Exception as e:
    print(f"\nCould not create multi-sheet Excel: {e}")
    print("   Single sheet files already saved above.")

print("\n----------------------------------------------------------------------")
print("SAMPLE REPROCESSED RESULTS")
print("----------------------------------------------------------------------")
sample_reprocessed = now_specific.head(10)
for idx, row in sample_reprocessed.iterrows():
    print(f"\nFeedback: {row['feedback_text'][:80]}...")
    print(f"   Original: {row['aspects_parsed']}")
    print(f"   New: {row['aspects_reprocessed']}")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

df_keyword = pd.read_csv('aspect_results.csv')
df_lda = pd.read_csv('lda_aspect_results.csv')
df_cluster = pd.read_csv('cluster_aspect_results.csv')
df_transformer = pd.read_csv('transformer_multi_aspect_results.csv')

def safe_convert_to_list(aspect_value):
    if pd.isna(aspect_value):
        return []
    if isinstance(aspect_value, str):
        try:
            if aspect_value.startswith('['):
                return ast.literal_eval(aspect_value)
            else:
                return [aspect_value]
        except:
            return [aspect_value] if aspect_value else []
    if isinstance(aspect_value, list):
        return aspect_value
    return []

print("COMPLETE METHOD COMPARISON (All 4 Methods)")
print("----------------------------------------------------------------------")

total_feedbacks = len(df_keyword)
df_keyword['aspect_clean'] = df_keyword['aspect'].apply(safe_convert_to_list)
keyword_num_aspects = df_keyword['aspect_clean'].apply(len)
keyword_avg = keyword_num_aspects.mean()
keyword_multiple = (keyword_num_aspects > 1).sum()
keyword_general = sum('General' in aspects for aspects in df_keyword['aspect_clean'])

df_lda['aspects_clean'] = df_lda['aspects'].apply(safe_convert_to_list)
lda_num_aspects = df_lda['aspects_clean'].apply(len)
lda_avg = lda_num_aspects.mean()
lda_multiple = (lda_num_aspects > 1).sum()
lda_general = sum('General' in aspects for aspects in df_lda['aspects_clean'])

df_cluster['aspects_clean'] = df_cluster['aspects'].apply(safe_convert_to_list)
cluster_num_aspects = df_cluster['aspects_clean'].apply(len)
cluster_avg = cluster_num_aspects.mean()
cluster_multiple = (cluster_num_aspects > 1).sum()
cluster_general = sum('General' in aspects for aspects in df_cluster['aspects_clean'])

df_transformer['aspects_clean'] = df_transformer['aspects'].apply(safe_convert_to_list)
transformer_num_aspects = df_transformer['aspects_clean'].apply(len)
transformer_avg = transformer_num_aspects.mean()
transformer_multiple = (transformer_num_aspects > 1).sum()
transformer_general = sum('General' in aspects for aspects in df_transformer['aspects_clean'])

print("\nBASIC STATISTICS:")
print("----------------------------------------------------------------------")
print(f"{'Method':<15} {'Total':<10} {'Avg Aspects':<12} {'Multiple %':<12} {'General %':<12}")
print("----------------------------------------------------------------------")
print(f"{'Keyword':<15} {total_feedbacks:<10} {keyword_avg:<12.2f} {(keyword_multiple/total_feedbacks)*100:<11.1f}% {(keyword_general/total_feedbacks)*100:<11.1f}%")
print(f"{'LDA':<15} {len(df_lda):<10} {lda_avg:<12.2f} {(lda_multiple/len(df_lda))*100:<11.1f}% {(lda_general/len(df_lda))*100:<11.1f}%")
print(f"{'Clustering':<15} {len(df_cluster):<10} {cluster_avg:<12.2f} {(cluster_multiple/len(df_cluster))*100:<11.1f}% {(cluster_general/len(df_cluster))*100:<11.1f}%")
print(f"{'Transformer':<15} {len(df_transformer):<10} {transformer_avg:<12.2f} {(transformer_multiple/len(df_transformer))*100:<11.1f}% {(transformer_general/len(df_transformer))*100:<11.1f}%")

print("\nASPECT DISTRIBUTION BY METHOD")
print("----------------------------------------------------------------------")

def count_aspects(df, col_name='aspects_clean'):
    all_aspects = []
    for aspects in df[col_name]:
        if isinstance(aspects, list):
            all_aspects.extend(aspects)
    return pd.Series(all_aspects).value_counts()

aspects_order = ["Instructional Quality", "Teaching Methodology", "Assessment & Evaluation",
                 "Mentoring & Academic Support", "Practical & Lab Integration",
                 "Course Design & Workload", "Learning Outcomes & Skill Development", "General"]

keyword_counts = count_aspects(df_keyword, 'aspect_clean')
keyword_total = keyword_counts.sum()

lda_counts = count_aspects(df_lda, 'aspects_clean')
lda_total = lda_counts.sum()

cluster_counts = count_aspects(df_cluster, 'aspects_clean')
cluster_total = cluster_counts.sum()

transformer_counts = count_aspects(df_transformer, 'aspects_clean')
transformer_total = transformer_counts.sum()

print("\nAspect (% of total mentions):")
print("--------------------------------------------------------------------------------")
print(f"{'Aspect':<35} {'Keyword':<12} {'LDA':<12} {'Clustering':<12} {'Transformer':<12}")
print("--------------------------------------------------------------------------------")

for aspect in aspects_order:
    keyword_pct = (keyword_counts.get(aspect, 0) / keyword_total) * 100 if keyword_total > 0 else 0
    lda_pct = (lda_counts.get(aspect, 0) / lda_total) * 100 if lda_total > 0 else 0
    cluster_pct = (cluster_counts.get(aspect, 0) / cluster_total) * 100 if cluster_total > 0 else 0
    transformer_pct = (transformer_counts.get(aspect, 0) / transformer_total) * 100 if transformer_total > 0 else 0

    print(f"{aspect:<35} {keyword_pct:<11.1f}% {lda_pct:<11.1f}% {cluster_pct:<11.1f}% {transformer_pct:<11.1f}%")

print("\nSAMPLE COMPARISON (First 5 feedbacks)")
print("----------------------------------------------------------------------")

for idx in range(min(5, len(df_keyword))):
    print(f"\nFEEDBACK {idx+1}:")
    print(f"   Text: {df_keyword.iloc[idx]['feedback_text'][:100]}...")
    print(f"   Keyword: {df_keyword.iloc[idx]['aspect_clean']}")
    print(f"   LDA: {df_lda.iloc[idx]['aspects_clean']}")
    print(f"   Clustering: {df_cluster.iloc[idx]['aspects_clean']}")
    print(f"   Transformer: {df_transformer.iloc[idx]['aspects_clean']}")


print("\nGENERATING VISUALIZATIONS...")
print("----------------------------------------------------------------------")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()


keyword_counts_top = keyword_counts.drop('General', errors='ignore').head(7)
if len(keyword_counts_top) > 0:
    axes[0].barh(range(len(keyword_counts_top)), keyword_counts_top.values, color='skyblue')
    axes[0].set_yticks(range(len(keyword_counts_top)))
    axes[0].set_yticklabels(keyword_counts_top.index, fontsize=9)
    axes[0].set_xlabel('Count')
    axes[0].set_title('Keyword Method')
    axes[0].invert_yaxis()
else:
    axes[0].text(0.5, 0.5, 'No data', ha='center', va='center')

lda_counts_top = lda_counts.drop('General', errors='ignore').head(7)
if len(lda_counts_top) > 0:
    axes[1].barh(range(len(lda_counts_top)), lda_counts_top.values, color='lightgreen')
    axes[1].set_yticks(range(len(lda_counts_top)))
    axes[1].set_yticklabels(lda_counts_top.index, fontsize=9)
    axes[1].set_xlabel('Count')
    axes[1].set_title('LDA Method')
    axes[1].invert_yaxis()
else:
    axes[1].text(0.5, 0.5, 'No data', ha='center', va='center')

cluster_counts_top = cluster_counts.drop('General', errors='ignore').head(7)
if len(cluster_counts_top) > 0:
    axes[2].barh(range(len(cluster_counts_top)), cluster_counts_top.values, color='salmon')
    axes[2].set_yticks(range(len(cluster_counts_top)))
    axes[2].set_yticklabels(cluster_counts_top.index, fontsize=9)
    axes[2].set_xlabel('Count')
    axes[2].set_title('Clustering Method')
    axes[2].invert_yaxis()
else:
    axes[2].text(0.5, 0.5, 'No data', ha='center', va='center')

transformer_counts_top = transformer_counts.drop('General', errors='ignore').head(7)
if len(transformer_counts_top) > 0:
    axes[3].barh(range(len(transformer_counts_top)), transformer_counts_top.values, color='purple')
    axes[3].set_yticks(range(len(transformer_counts_top)))
    axes[3].set_yticklabels(transformer_counts_top.index, fontsize=9)
    axes[3].set_xlabel('Count')
    axes[3].set_title('Transformer Method')
    axes[3].invert_yaxis()
else:
    axes[3].text(0.5, 0.5, 'No data', ha='center', va='center')

plt.suptitle('Aspect Distribution by Method', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('method_comparison_aspects.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFINAL RECOMMENDATION")
print("----------------------------------------------------------------------")

print("\nMETHOD SCORING:")
print("----------------------------------------------------------------------")

keyword_multiple_pct = (keyword_multiple/total_feedbacks)*100
lda_multiple_pct = (lda_multiple/len(df_lda))*100
cluster_multiple_pct = (cluster_multiple/len(df_cluster))*100
transformer_multiple_pct = (transformer_multiple/len(df_transformer))*100

keyword_general_pct = (keyword_general/total_feedbacks)*100
lda_general_pct = (lda_general/len(df_lda))*100
cluster_general_pct = (cluster_general/len(df_cluster))*100
transformer_general_pct = (transformer_general/len(df_transformer))*100

print(f"{'Criteria':<30} {'Keyword':<12} {'LDA':<12} {'Clustering':<12} {'Transformer':<12}")
print("----------------------------------------------------------------------")
print(f"{'Avg Aspects':<30} {keyword_avg:<11.2f} {lda_avg:<11.2f} {cluster_avg:<11.2f} {transformer_avg:<11.2f}")
print(f"{'Multiple %':<30} {keyword_multiple_pct:<11.1f}% {lda_multiple_pct:<11.1f}% {cluster_multiple_pct:<11.1f}% {transformer_multiple_pct:<11.1f}%")
print(f"{'General %':<30} {keyword_general_pct:<11.1f}% {lda_general_pct:<11.1f}% {cluster_general_pct:<11.1f}% {transformer_general_pct:<11.1f}%")


summary_data = {
    'Method': ['Keyword', 'LDA', 'Clustering', 'Transformer'],
    'Avg Aspects': [f"{keyword_avg:.2f}", f"{lda_avg:.2f}", f"{cluster_avg:.2f}", f"{transformer_avg:.2f}"],
    'Multiple %': [f"{keyword_multiple_pct:.1f}%", f"{lda_multiple_pct:.1f}%", f"{cluster_multiple_pct:.1f}%", f"{transformer_multiple_pct:.1f}%"],
    'General %': [f"{keyword_general_pct:.1f}%", f"{lda_general_pct:.1f}%", f"{cluster_general_pct:.1f}%", f"{transformer_general_pct:.1f}%"]
}

pd.DataFrame(summary_data).to_csv('method_comparison_summary.csv', index=False)
print("\nSaved: method_comparison_summary.csv")


import gc
gc.collect()

