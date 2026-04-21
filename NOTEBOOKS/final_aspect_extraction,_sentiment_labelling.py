# -*- coding: utf-8 -*-
"""final aspect extraction, sentiment labelling.ipynb
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import ast
import re

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
threshold = 0.3

all_aspects = []
all_confidences = []


print("PART 1: TRANSFORMER ASPECT EXTRACTION")
print(f"Processing {len(df_transformer):,} feedbacks on GPU...")
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

df_transformer.to_csv('transformer_multi_aspect_results.csv', index=False)
print(f"\nSaved: transformer_multi_aspect_results.csv")

print("PART 2: REPROCESSING GENERAL FEEDBACKS")

def parse_aspects(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return ["General"]
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return ["General"]
    return ["General"]

df_transformer['aspects_parsed'] = df_transformer['aspects'].apply(parse_aspects)

df_general = df_transformer[df_transformer['aspects_parsed'].apply(lambda x: 'General' in x)].copy()
df_specific = df_transformer[~df_transformer['aspects_parsed'].apply(lambda x: 'General' in x)].copy()

print(f"Initial Specific: {len(df_specific):,} ({len(df_specific)/len(df_transformer)*100:.1f}%)")
print(f"Initial General: {len(df_general):,} ({len(df_general)/len(df_transformer)*100:.1f}%)")

aspect_keywords = {
    "Instructional Quality": [
        "explain","explained","explanation","clarity","clear","confusing",
        "understand","understood","concept","theory","expertise","lecture",
        "lecturer","topic","material"
    ],
    "Teaching Methodology": [
        "example","examples","illustration","ppt","slides","board",
        "presentation","interactive","engaging","discussion","activity",
        "teaching","delivery","video","method"
    ],
    "Assessment & Evaluation": [
        "exam","assessment","test","quiz","grading","marks","marking",
        "evaluation","fair","unfair","transparent","criteria","rubric",
        "midterm","final","assignment","homework","difficulty","expensive",
        "certificate","price","cost"
    ],
    "Mentoring & Academic Support": [
        "mentor","guidance","support","helpful","available","approachable",
        "responsive","feedback","suggestion","clarify","supervision","advisor"
    ],
    "Practical & Lab Integration": [
        "lab","laboratory","practical","experiment","equipment","hands-on",
        "session","viva","implementation","project","tool","software",
        "coding","programming"
    ],
    "Course Design & Workload": [
        "workload","structure","organization","pace","fast","slow","rushed",
        "balanced","schedule","timing","week","deadline","duration",
        "too much","prerequisite","background"
    ],
    "Learning Outcomes & Skill Development": [
        "skill","learning","learned","outcome","knowledge","improvement",
        "growth","development","experience","useful","valuable","recommend"
    ]
}

def extract_aspects_fast(text, max_aspects=3):
    if pd.isna(text) or text == "":
        return ["General"]
    text_lower = text.lower()
    aspects = set()
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                aspects.add(aspect)
                break
    return list(aspects)[:max_aspects] if aspects else ["General"]

reprocessed_results = []
for idx, row in tqdm(df_general.iterrows(), total=len(df_general), desc="Reprocessing"):
    aspects = extract_aspects_fast(row['feedback_text'], max_aspects=3)
    reprocessed_results.append(aspects)

df_general['aspects_reprocessed'] = reprocessed_results
df_general['still_general'] = df_general['aspects_reprocessed'].apply(lambda x: 'General' in x)
df_general['now_specific'] = ~df_general['still_general']

now_specific = df_general[df_general['now_specific'] == True].copy()
still_general = df_general[df_general['still_general'] == True].copy()

print(f"\nReprocessed -> Now Specific: {len(now_specific):,}")
print(f"Still General (will be dropped): {len(still_general):,}")


print("PART 3: CREATING FINAL DATASET")
df_specific_clean = df_specific.copy()

now_specific_clean = now_specific.copy()
now_specific_clean['aspects'] = now_specific_clean['aspects_reprocessed']
now_specific_clean['num_aspects'] = now_specific_clean['aspects'].apply(len)

df_final = pd.concat([df_specific_clean, now_specific_clean], ignore_index=True)

print(f"\nFINAL DATASET READY:")
print(f"   Total rows: {len(df_final):,}")
print(f"   General dropped: {len(still_general):,} ({len(still_general)/len(df_transformer)*100:.1f}%)")
print(f"   Aspects per feedback (avg): {df_final['num_aspects'].mean():.2f}")

print("PART 4: CREATING EXPLODED DATASET")

df_exploded = df_final.explode('aspects').reset_index(drop=True)
df_exploded.rename(columns={'aspects': 'aspect'}, inplace=True)

print(f"Exploded rows: {len(df_exploded):,}")
print(f"\nAspect distribution:")
print(df_exploded['aspect'].value_counts())

print("PART 5: SAVING FILES")

df_final.to_csv('final_aspect_dataset.csv', index=False)
print("Saved: final_aspect_dataset.csv (all columns with aspects)")

df_exploded.to_csv('final_aspect_exploded.csv', index=False)
print("Saved: final_aspect_exploded.csv (one aspect per row)")

df_final_all_columns = df_final.copy()
df_final_all_columns.to_csv('final_aspect_dataset_full.csv', index=False)
print("Saved: final_aspect_dataset_full.csv (with all MCQs and original columns)")

print("FINAL STATISTICS DASHBOARD")

print(f"\nDATASET OVERVIEW:")
print(f"   Original feedbacks: {len(df_transformer):,}")
print(f"   Final feedbacks (after dropping General): {len(df_final):,}")
print(f"   Dropped: {len(df_transformer) - len(df_final):,} ({((len(df_transformer) - len(df_final))/len(df_transformer))*100:.1f}%)")

print(f"\nASPECT DISTRIBUTION (Final):")
for aspect, count in df_exploded['aspect'].value_counts().items():
    pct = (count / len(df_exploded)) * 100
    print(f"   {aspect}: {count:,} ({pct:.1f}%)")

print(f"\nMULTI-ASPECT STATISTICS:")
df_final['num_aspects'] = df_final['aspects'].apply(len)
print(f"   Average aspects per feedback: {df_final['num_aspects'].mean():.2f}")
print(f"   Single aspect feedbacks: {(df_final['num_aspects'] == 1).sum():,} ({((df_final['num_aspects'] == 1).sum()/len(df_final))*100:.1f}%)")
print(f"   Multiple aspect feedbacks: {(df_final['num_aspects'] > 1).sum():,} ({((df_final['num_aspects'] > 1).sum()/len(df_final))*100:.1f}%)")


print(f"\nSAMPLE FINAL RESULTS:")
sample_aspects = df_exploded.groupby('feedback_text')['aspect'].apply(list).head(5)
for fb, aspects in sample_aspects.items():
    print(f"\n   Feedback: {fb[:80]}...")
    print(f"   Aspects: {aspects}")

print("\nFINAL FILES CREATED:")
print("   1. final_aspect_dataset.csv - Main dataset with aspects column")
print("   2. final_aspect_exploded.csv - One aspect per row (ready for analysis)")
print("   3. final_aspect_dataset_full.csv - Complete dataset with all MCQs")
print("   4. transformer_multi_aspect_results.csv - Raw transformer output")

torch.cuda.empty_cache()

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

file_path = '/content/drive/MyDrive/Colab Notebooks/final_feedback_dataset.csv'
df = pd.read_csv(file_path)

df.head()

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import ast
import re


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
threshold = 0.3

all_aspects = []
all_confidences = []

print("PART 1: TRANSFORMER ASPECT EXTRACTION")


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

df_transformer.to_csv('transformer_multi_aspect_results.csv', index=False)
print(f"\n Saved: transformer_multi_aspect_results.csv")

print("PART 2: REPROCESSING GENERAL FEEDBACKS")

def parse_aspects(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return ["General"]
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return ["General"]
    return ["General"]

df_transformer['aspects_parsed'] = df_transformer['aspects'].apply(parse_aspects)

df_general = df_transformer[df_transformer['aspects_parsed'].apply(lambda x: 'General' in x)].copy()
df_specific = df_transformer[~df_transformer['aspects_parsed'].apply(lambda x: 'General' in x)].copy()

print(f"Initial Specific: {len(df_specific):,} ({len(df_specific)/len(df_transformer)*100:.1f}%)")
print(f"Initial General: {len(df_general):,} ({len(df_general)/len(df_transformer)*100:.1f}%)")

aspect_keywords = {
    "Instructional Quality": [
        "explain","explained","explanation","clarity","clear","confusing",
        "understand","understood","concept","theory","expertise","lecture",
        "lecturer","topic","material"
    ],
    "Teaching Methodology": [
        "example","examples","illustration","ppt","slides","board",
        "presentation","interactive","engaging","discussion","activity",
        "teaching","delivery","video","method"
    ],
    "Assessment & Evaluation": [
        "exam","assessment","test","quiz","grading","marks","marking",
        "evaluation","fair","unfair","transparent","criteria","rubric",
        "midterm","final","assignment","homework","difficulty","expensive",
        "certificate","price","cost"
    ],
    "Mentoring & Academic Support": [
        "mentor","guidance","support","helpful","available","approachable",
        "responsive","feedback","suggestion","clarify","supervision","advisor"
    ],
    "Practical & Lab Integration": [
        "lab","laboratory","practical","experiment","equipment","hands-on",
        "session","viva","implementation","project","tool","software",
        "coding","programming"
    ],
    "Course Design & Workload": [
        "workload","structure","organization","pace","fast","slow","rushed",
        "balanced","schedule","timing","week","deadline","duration",
        "too much","prerequisite","background"
    ],
    "Learning Outcomes & Skill Development": [
        "skill","learning","learned","outcome","knowledge","improvement",
        "growth","development","experience","useful","valuable","recommend"
    ]
}

def extract_aspects_fast(text, max_aspects=3):
    if pd.isna(text) or text == "":
        return ["General"]
    text_lower = text.lower()
    aspects = set()
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                aspects.add(aspect)
                break
    return list(aspects)[:max_aspects] if aspects else ["General"]
reprocessed_results = []
for idx, row in tqdm(df_general.iterrows(), total=len(df_general), desc="Reprocessing"):
    aspects = extract_aspects_fast(row['feedback_text'], max_aspects=3)
    reprocessed_results.append(aspects)

df_general['aspects_reprocessed'] = reprocessed_results
df_general['still_general'] = df_general['aspects_reprocessed'].apply(lambda x: 'General' in x)
df_general['now_specific'] = ~df_general['still_general']

now_specific = df_general[df_general['now_specific'] == True].copy()
still_general = df_general[df_general['still_general'] == True].copy()

print(f"\nReprocessed → Now Specific: {len(now_specific):,}")
print(f"Still General (will be dropped): {len(still_general):,}")

\
print("PART 3: CREATING FINAL DATASET")

df_specific_clean = df_specific.copy()

now_specific_clean = now_specific.copy()
now_specific_clean['aspects'] = now_specific_clean['aspects_reprocessed']
now_specific_clean['num_aspects'] = now_specific_clean['aspects'].apply(len)

df_final = pd.concat([df_specific_clean, now_specific_clean], ignore_index=True)

print(f"\nFINAL DATASET READY:")
print(f"   Total rows: {len(df_final):,}")
print(f"   General dropped: {len(still_general):,} ({len(still_general)/len(df_transformer)*100:.1f}%)")
print(f"   Aspects per feedback (avg): {df_final['num_aspects'].mean():.2f}")

print("PART 4: CREATING EXPLODED DATASET")

df_exploded = df_final.explode('aspects').reset_index(drop=True)
df_exploded.rename(columns={'aspects': 'aspect'}, inplace=True)

print(f"Exploded rows: {len(df_exploded):,}")
print(f"\nAspect distribution:")
print(df_exploded['aspect'].value_counts())

print("PART 5: SAVING FILES")

df_final.to_csv('final_aspect_dataset.csv', index=False)
print("Saved: final_aspect_dataset.csv (all columns with aspects)")

df_exploded.to_csv('final_aspect_exploded.csv', index=False)
print("Saved: final_aspect_exploded.csv (one aspect per row)")

df_final_all_columns = df_final.copy()
df_final_all_columns.to_csv('final_aspect_dataset_full.csv', index=False)
print("Saved: final_aspect_dataset_full.csv (with all MCQs and original columns)")

print("FINAL STATISTICS DASHBOARD")
print(f"\nDATASET OVERVIEW:")
print(f"   Original feedbacks: {len(df_transformer):,}")
print(f"   Final feedbacks (after dropping General): {len(df_final):,}")
print(f"   Dropped: {len(df_transformer) - len(df_final):,} ({((len(df_transformer) - len(df_final))/len(df_transformer))*100:.1f}%)")

print(f"\n ASPECT DISTRIBUTION (Final):")
for aspect, count in df_exploded['aspect'].value_counts().items():
    pct = (count / len(df_exploded)) * 100
    print(f"   {aspect}: {count:,} ({pct:.1f}%)")

print(f"\n MULTI-ASPECT STATISTICS:")
df_final['num_aspects'] = df_final['aspects'].apply(len)
print(f"   Average aspects per feedback: {df_final['num_aspects'].mean():.2f}")
print(f"   Single aspect feedbacks: {(df_final['num_aspects'] == 1).sum():,} ({((df_final['num_aspects'] == 1).sum()/len(df_final))*100:.1f}%)")
print(f"   Multiple aspect feedbacks: {(df_final['num_aspects'] > 1).sum():,} ({((df_final['num_aspects'] > 1).sum()/len(df_final))*100:.1f}%)")

print(f"\n📝 SAMPLE FINAL RESULTS:")
sample_aspects = df_exploded.groupby('feedback_text')['aspect'].apply(list).head(5)
for fb, aspects in sample_aspects.items():
    print(f"\n   Feedback: {fb[:80]}...")
    print(f"   Aspects: {aspects}")

print("\n📁 FINAL FILES CREATED:")
print("   1. final_aspect_dataset.csv - Main dataset with aspects column")
print("   2. final_aspect_exploded.csv - One aspect per row (ready for analysis)")
print("   3. final_aspect_dataset_full.csv - Complete dataset with all MCQs")
print("   4. transformer_multi_aspect_results.csv - Raw transformer output")

torch.cuda.empty_cache()

!pip install transformers
!pip install sentence_transformers
!pip install vaderSentiment

import sys
!{sys.executable} -m pip install vaderSentiment --upgrade --force-reinstall

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
import gc
import os

torch.cuda.empty_cache()
gc.collect()

device = 0 if torch.cuda.is_available() else -1
print(f"Using GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.8)

df = pd.read_csv('final_aspect_exploded.csv')
print(f"Total rows: {len(df):,}")


print("METHOD 1: DistilBERT GPU-Optimized")
tokenizer_distilbert = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model_distilbert = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
).to(device)
model_distilbert.eval()

distilbert = pipeline(
    "sentiment-analysis",
    model=model_distilbert,
    tokenizer=tokenizer_distilbert,
    device=device,
    batch_size=64,
    truncation=True,
    max_length=256,
    padding=True
)

print("Processing with DistilBERT on GPU...")
distilbert_results = []

for i in tqdm(range(0, len(df), 64)):
    batch = df['feedback_text'].iloc[i:i+64].tolist()
    batch = [str(text)[:256] for text in batch]

    try:
        results = distilbert(batch)
        for r in results:
            if r['label'].lower() == 'positive':
                distilbert_results.append(("Positive", r['score']))
            else:
                distilbert_results.append(("Negative", -r['score']))
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"Out of memory at batch {i}, reducing batch size...")
            torch.cuda.empty_cache()
            for j in range(0, len(batch), 32):
                sub_batch = batch[j:j+32]
                sub_results = distilbert(sub_batch)
                for r in sub_results:
                    if r['label'].lower() == 'positive':
                        distilbert_results.append(("Positive", r['score']))
                    else:
                        distilbert_results.append(("Negative", -r['score']))
        else:
            raise e

df['sentiment_distilbert'] = [r[0] for r in distilbert_results]
df['polarity_distilbert'] = [r[1] for r in distilbert_results]

torch.cuda.empty_cache()
gc.collect()

print("METHOD 2: RoBERTa GPU-Optimized")


try:
    tokenizer_roberta = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model_roberta = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ).to(device)
    model_roberta.eval()

    roberta = pipeline(
        "sentiment-analysis",
        model=model_roberta,
        tokenizer=tokenizer_roberta,
        device=device,
        batch_size=48,
        truncation=True,
        max_length=256,
        padding=True
    )

    print("Processing with RoBERTa on GPU...")
    roberta_results = []

    for i in tqdm(range(0, len(df), 48)):
        batch = df['feedback_text'].iloc[i:i+48].tolist()
        batch = [str(text)[:256] for text in batch]

        try:
            results = roberta(batch)
            for r in results:
                if r['label'] == 'positive':
                    roberta_results.append(("Positive", r['score']))
                elif r['label'] == 'negative':
                    roberta_results.append(("Negative", -r['score']))
                else:
                    roberta_results.append(("Neutral", r['score']))
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at batch {i}, skipping to next...")
                torch.cuda.empty_cache()
                for _ in range(len(batch)):
                    roberta_results.append(("Neutral", 0))
            else:
                raise e

    df['sentiment_roberta'] = [r[0] for r in roberta_results]
df['polarity_roberta'] = [r[1] for r in roberta_results]

except Exception as e:
    print(f"RoBERTa failed: {e}")
    df['sentiment_roberta'] = df['sentiment_distilbert']
    df['polarity_roberta'] = df['polarity_distilbert']

torch.cuda.empty_cache()
gc.collect()

print("METHOD 3: BERT GPU-Optimized")
try:
    tokenizer_bert = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model_bert = AutoModelForSequenceClassification.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment"
    ).to(device)
    model_bert.eval()

    bert = pipeline(
        "sentiment-analysis",
        model=model_bert,
        tokenizer=tokenizer_bert,
        device=device,
        batch_size=32,
        truncation=True,
        max_length=256,
        padding=True
    )

    print("Processing with BERT on GPU...")
    bert_results = []

    for i in tqdm(range(0, len(df), 32)):
        batch = df['feedback_text'].iloc[i:i+32].tolist()
        batch = [str(text)[:256] for text in batch]

        try:
            results = bert(batch)
            for r in results:
                rating = int(r['label'].split()[0])
                score = (rating - 3) / 2
                if rating >= 4:
                    bert_results.append(("Positive", score))
                elif rating <= 2:
                    bert_results.append(("Negative", score))
                else:
                    bert_results.append(("Neutral", score))
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at batch {i}, skipping...")
                torch.cuda.empty_cache()
                for _ in range(len(batch)):
                    bert_results.append(("Neutral", 0))
            else:
                raise e

    df['sentiment_bert'] = [r[0] for r in bert_results]
df['polarity_bert'] = [r[1] for r in bert_results]

except Exception as e:
    print(f"BERT failed: {e}")
    df['sentiment_bert'] = df['sentiment_distilbert']
    df['polarity_bert'] = df['polarity_distilbert']

print("RESULTS SUMMARY")

print("\nSentiment Distribution (DistilBERT):")
sent_dist = df['sentiment_distilbert'].value_counts()
for sentiment, count in sent_dist.items():
    print(f"  {sentiment}: {count/len(df)*100:.1f}% ({count:,})")

print("\nAspect-wise Sentiment (DistilBERT):")
aspect_sentiment = df.groupby('aspect').agg({
    'polarity_distilbert': ['mean', 'std', 'count']
}).round(3)
aspect_sentiment.columns = ['avg_polarity', 'std_polarity', 'count']
aspect_sentiment = aspect_sentiment.sort_values('avg_polarity', ascending=False)
print(aspect_sentiment)

print("\nSaving files...")
df.to_csv('aspect_sentiment_gpu_results.csv', index=False)
aspect_sentiment.to_csv('aspect_sentiment_summary.csv')
print("Saved: aspect_sentiment_gpu_results.csv")
print("Saved: aspect_sentiment_summary.csv")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
aspects = aspect_sentiment.index
polarities = aspect_sentiment['avg_polarity'].values
colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in polarities]

plt.barh(range(len(aspects)), polarities, color=colors)
plt.yticks(range(len(aspects)), aspects)
plt.xlabel('Average Sentiment Polarity')
plt.title('Aspect-wise Sentiment Analysis (GPU - DistilBERT)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('aspect_sentiment_gpu.png', dpi=150)
plt.show()

if torch.cuda.is_available():
    print(f"\nGPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")


torch.cuda.empty_cache()

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('aspect_sentiment_gpu_results.csv')

print("\nSENTIMENT DISTRIBUTION COMPARISON:")

distribution_data = []
for method in ['distilbert', 'roberta', 'bert']:
    sent_col = f'sentiment_{method}'
    if sent_col in df.columns:
        dist = df[sent_col].value_counts(normalize=True) * 100
        print(f"\n{method.upper()}:")
        for sentiment, pct in dist.items():
            print(f"   {sentiment}: {pct:.1f}%")
        distribution_data.append({
            'method': method,
            'positive_pct': dist.get('Positive', 0),
            'neutral_pct': dist.get('Neutral', 0),
            'negative_pct': dist.get('Negative', 0)
        })


confidence_data = []
for method in ['distilbert', 'roberta', 'bert']:
    polarity_col = f'polarity_{method}'
    if polarity_col in df.columns:
        avg_conf = df[polarity_col].abs().mean()
        median_conf = df[polarity_col].abs().median()
        std_conf = df[polarity_col].abs().std()
        print(f"\n{method.upper()}:")
        print(f"   Mean Confidence: {avg_conf:.3f}")
        print(f"   Median Confidence: {median_conf:.3f}")
        print(f"   Std Deviation: {std_conf:.3f}")
        confidence_data.append({
            'method': method,
            'mean_confidence': avg_conf,
            'median_confidence': median_conf
        })

print("ASPECT DISCRIMINATION (Higher variation = Better at distinguishing aspects)")

aspect_variation = []
for method in ['distilbert', 'roberta', 'bert']:
    polarity_col = f'polarity_{method}'
    if polarity_col in df.columns:
        aspect_means = df.groupby('aspect')[polarity_col].mean()
        variation = aspect_means.std()
        min_polarity = aspect_means.min()
        max_polarity = aspect_means.max()
        range_polarity = max_polarity - min_polarity

        print(f"\n{method.upper()}:")
        print(f"   Variation (Std): {variation:.3f}")
        print(f"   Range: {range_polarity:.3f} ({min_polarity:.3f} to {max_polarity:.3f})")
        print(f"   Best aspect: {aspect_means.idxmax()} ({aspect_means.max():.3f})")
        print(f"   Worst aspect: {aspect_means.idxmin()} ({aspect_means.min():.3f})")

        aspect_variation.append({
            'method': method,
            'variation': variation,
            'range': range_polarity,
            'best_aspect': aspect_means.idxmax(),
            'worst_aspect': aspect_means.idxmin()
        })


agreement_data = []
methods_list = ['distilbert', 'roberta', 'bert']
for method in methods_list:
    sent_col = f'sentiment_{method}'
    if sent_col in df.columns:
        agreements = []
        for other in methods_list:
            if other != method:
                other_col = f'sentiment_{other}'
                if other_col in df.columns:
                    kappa = cohen_kappa_score(df[sent_col], df[other_col])
                    agreements.append(kappa)
        avg_agreement = np.mean(agreements)
        print(f"\n{method.upper()}:")
        print(f"   Avg agreement with others: {avg_agreement:.3f}")
        agreement_data.append({
            'method': method,
            'avg_agreement': avg_agreement
        })

consistency_data = []
for method in ['distilbert', 'roberta', 'bert']:
    polarity_col = f'polarity_{method}'
    if polarity_col in df.columns:
        aspect_std = df.groupby('aspect')[polarity_col].std().mean()
        print(f"\n{method.upper()}:")
        print(f"   Avg Std across aspects: {aspect_std:.3f}")
        consistency_data.append({
            'method': method,
            'avg_std': aspect_std
        })


ranking_data = []
for method in ['distilbert', 'roberta', 'bert']:
    conf = next((c for c in confidence_data if c['method'] == method), None)
    var = next((v for v in aspect_variation if v['method'] == method), None)
    agree = next((a for a in agreement_data if a['method'] == method), None)
    consist = next((c for c in consistency_data if c['method'] == method), None)

    ranking_data.append({
        'Model': method.upper(),
        'Confidence (↑)': conf['mean_confidence'] if conf else 0,
        'Aspect Variation (↑)': var['variation'] if var else 0,
        'Agreement (↑)': agree['avg_agreement'] if agree else 0,
        'Consistency (↓)': consist['avg_std'] if consist else 0
    })

ranking_df = pd.DataFrame(ranking_data)

for col in ['Confidence (↑)', 'Aspect Variation (↑)', 'Agreement (↑)']:
    ranking_df[f'{col}_norm'] = (ranking_df[col] - ranking_df[col].min()) / (ranking_df[col].max() - ranking_df[col].min())

ranking_df['Consistency (↓)_norm'] = 1 - ((ranking_df['Consistency (↓)'] - ranking_df['Consistency (↓)'].min()) /
                                          (ranking_df['Consistency (↓)'].max() - ranking_df['Consistency (↓)'].min()))

ranking_df['Composite Score'] = (ranking_df['Confidence (↑)_norm'] +
                                  ranking_df['Aspect Variation (↑)_norm'] +
                                  ranking_df['Agreement (↑)_norm'] +
                                  ranking_df['Consistency (↓)_norm']) / 4

ranking_df = ranking_df.sort_values('Composite Score', ascending=False)
print("\nRANKING SUMMARY:")
print(ranking_df[['Model', 'Confidence (↑)', 'Aspect Variation (↑)', 'Agreement (↑)', 'Consistency (↓)', 'Composite Score']].to_string(index=False))

print("\nGENERATING VISUALIZATIONS...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax1 = axes[0, 0]
for method in ['distilbert', 'roberta', 'bert']:
    polarity_col = f'polarity_{method}'
    if polarity_col in df.columns:
        ax1.hist(df[polarity_col].abs(), alpha=0.5, bins=50, label=method.upper())
ax1.set_title('Confidence Distribution by Model')
ax1.set_xlabel('Confidence Score')
ax1.set_ylabel('Frequency')
ax1.legend()

ax2 = axes[0, 1]
x = np.arange(len(aspect_variation))
variations = [v['variation'] for v in aspect_variation]
ax2.bar(x, variations, tick_label=[v['method'].upper() for v in aspect_variation])
ax2.set_title('Aspect Discrimination Power (Higher = Better)')
ax2.set_ylabel('Standard Deviation across Aspects')

ax3 = axes[1, 0]
agreements = [a['avg_agreement'] for a in agreement_data]
ax3.bar(x, agreements, tick_label=[a['method'].upper() for a in agreement_data])
ax3.set_title('Model Consensus (Higher = More Agreement)')
ax3.set_ylabel('Average Cohen\'s Kappa')
ax3.axhline(y=0.6, color='g', linestyle='--', label='Good agreement threshold')
ax3.legend()

ax4 = axes[1, 1]
composite_scores = ranking_df['Composite Score'].values
ax4.bar(range(len(composite_scores)), composite_scores, tick_label=ranking_df['Model'].values)
ax4.set_title('Composite Performance Score')
ax4.set_ylabel('Score (0-1)')
for i, v in enumerate(composite_scores):
    ax4.text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('data_driven_model_selection.png', dpi=150, bbox_inches='tight')
plt.show()


best_model = ranking_df.iloc[0]['Model']
best_score = ranking_df.iloc[0]['Composite Score']

print(f"\nBased on YOUR data, the best model is: {best_model}")
print(f"   Composite Score: {best_score:.3f}")

print("Data-driven selection complete!")

ranking_df.to_csv('model_ranking.csv', index=False)
print("\nSaved: model_ranking.csv")

