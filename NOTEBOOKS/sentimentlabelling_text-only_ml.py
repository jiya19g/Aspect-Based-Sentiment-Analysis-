# -*- coding: utf-8 -*-
"""SentimentLabelling-ML.ipynb

# **Loading Dataset**
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

file_path = '/content/drive/MyDrive/Colab Notebooks/aspect_sentiment_gpu_results.csv'
df = pd.read_csv(file_path)

"""# **Cleaning and Labelling data**"""

df = df[['feedback_text', 'aspect', 'sentiment_roberta']]
df = df.rename(columns={'sentiment_roberta': 'sentiment'})

label_map = {
    'Negative': 0,
    'Neutral': 1,
    'Positive': 2
}

df['sentiment'] = df['sentiment'].map(label_map)

df = df.dropna().reset_index(drop=True)

"""# **Class Distribution**"""

print(df['sentiment'].value_counts(normalize=True))

"""# **Creating Input**"""

df['model_input'] = df['feedback_text'] + " [ASPECT: " + df['aspect'] + "]"

"""# **Train - Test - Validate Split**"""

from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

"""# **TF-IDF**"""

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_df['model_input'])
X_val = vectorizer.transform(val_df['model_input'])
X_test = vectorizer.transform(test_df['model_input'])

y_train = train_df['sentiment']
y_val = val_df['sentiment']
y_test = test_df['sentiment']

"""# **Assigning Class Weights**"""

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)

class_weights = dict(zip(classes, weights))
print("Class Weights:", class_weights)

"""# **Eval metrics**"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def evaluate(model, X_test, y_test, model_name, y_proba=None):
    """
    Unified evaluation for any model.

    Usage:
        evaluate(model_lr, X_test, y_test, "Logistic Regression")
        evaluate(model_svm, X_test, y_test, "SVM", y_proba=model_svm.predict_proba(X_test))
    """

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')


    print(f"\n{'='*50}")
    print(f"{model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print(f"\nPer-Class Performance:")
    print(classification_report(y_test, y_pred,
                                target_names=['Negative', 'Neutral', 'Positive']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    return {'model': model_name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1}

all_results = []

"""# **LOGISTIC REGRESSION**"""

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(max_iter=1000, class_weight=class_weights)
model_lr.fit(X_train, y_train)

result = evaluate(model_lr, X_test, y_test, "Logistic Regression")
all_results.append(result)

"""# **NAIVE BAYES**"""

from sklearn.naive_bayes import MultinomialNB

model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

result = evaluate(model_nb, X_test, y_test, "Naive Bayes")
all_results.append(result)

"""# **Random Forest**"""

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight=class_weights,
    n_jobs=-1,
    random_state=42
)
model_rf.fit(X_train, y_train)

result = evaluate(model_rf, X_test, y_test, "Random Forest")
all_results.append(result)

"""# **Linear SVM**"""

from sklearn.svm import LinearSVC

model_svm = LinearSVC(class_weight=class_weights, random_state=42, max_iter=2000)
model_svm.fit(X_train, y_train)

result = evaluate(model_svm, X_test, y_test, "Linear SVM")
all_results.append(result)

"""# **XGB**"""

from xgboost import XGBClassifier

# Convert class_weights to sample weights for XGBoost
sample_weights = np.ones(len(y_train))
for class_label, weight in class_weights.items():
    sample_weights[y_train == class_label] = weight

model_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)
model_xgb.fit(X_train, y_train, sample_weight=sample_weights)

result = evaluate(model_xgb, X_test, y_test, "XGBoost")
all_results.append(result)

"""# **Ridge Classifier**"""

from sklearn.linear_model import RidgeClassifier

model_ridge = RidgeClassifier(random_state=42)
model_ridge.fit(X_train, y_train)

result = evaluate(model_ridge, X_test, y_test, "Ridge Classifier")
all_results.append(result)

"""# **DecisionTreeClassifier**"""

from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(
    max_depth=10,
    class_weight=class_weights,
    random_state=42
)
model_dt.fit(X_train, y_train)

result = evaluate(model_dt, X_test, y_test, "Decision Tree")
all_results.append(result)

"""# **KNeighborsClassifier**"""

from sklearn.neighbors import KNeighborsClassifier

# Note: KNN can be slow on sparse data, use smaller k
model_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
model_knn.fit(X_train, y_train)

result = evaluate(model_knn, X_test, y_test, "K-Nearest Neighbors")
all_results.append(result)

"""# **SGDClassifier**"""

from sklearn.linear_model import SGDClassifier

model_sgd = SGDClassifier(
    loss='hinge',
    class_weight=class_weights,
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
model_sgd.fit(X_train, y_train)

result = evaluate(model_sgd, X_test, y_test, "SGD Classifier")
all_results.append(result)

"""# **ExtraTreesClassifier**"""

from sklearn.ensemble import ExtraTreesClassifier

model_et = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight=class_weights,
    n_jobs=-1,
    random_state=42
)
model_et.fit(X_train, y_train)

result = evaluate(model_et, X_test, y_test, "Extra Trees")
all_results.append(result)

"""# **LGBMClassifier**"""

from lightgbm import LGBMClassifier

model_lgb = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
model_lgb.fit(X_train, y_train)

result = evaluate(model_lgb, X_test, y_test, "LightGBM")
all_results.append(result)

"""# **result**"""

print("\n" + "="*70)
print(" FINAL MODEL COMPARISON (All ML Models)")
print("="*70)

comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df.sort_values('f1', ascending=False).reset_index(drop=True)
comparison_df.index = comparison_df.index + 1

print(comparison_df.to_string())

print(f"\n Best ML Model: {comparison_df.iloc[0]['model']}")
print(f"   Accuracy: {comparison_df.iloc[0]['accuracy']:.4f}")
print(f"   F1-Score: {comparison_df.iloc[0]['f1']:.4f}")

# Save results
comparison_df.to_csv('ml_model_comparison.csv', index=False)
print("\n Saved: ml_model_comparison.csv")

from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []

models = [
    ("K-Nearest Neighbors", model_knn),
    ("Linear SVM", model_svm),
    ("Logistic Regression", model_lr),
    ("LightGBM", model_lgb),
    ("XGBoost", model_xgb),
    ("SGD Classifier", model_sgd),
    ("Ridge Classifier", model_ridge),
    ("Extra Trees", model_et),
    ("Random Forest", model_rf),
    ("Naive Bayes", model_nb),
    ("Decision Tree", model_dt)
]

print("="*60)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("="*60)

for name, model in models:
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)

    cv_results.append({
        'Model': name,
        'CV Mean F1': f"{cv_scores.mean():.4f}",
        'CV Std': f"{cv_scores.std():.4f}",
        'CV Range': f"{cv_scores.min():.4f} - {cv_scores.max():.4f}"
    })

    print(f"\n{name}:")
    print(f"  CV F1 Scores: {cv_scores}")
    print(f"  Mean F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

cv_df = pd.DataFrame(cv_results)
cv_df = cv_df.sort_values('CV Mean F1', ascending=False)

print(" CROSS-VALIDATION RANKING (by Mean F1)")
print(cv_df.to_string(index=False))

print("SINGLE SPLIT vs CROSS-VALIDATION COMPARISON")
comparison = []
for name, model in models:
    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average='macro')

    cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro').mean()

    diff = test_f1 - cv_f1
    reliable = "✅" if abs(diff) < 0.03 else "⚠️" if abs(diff) < 0.05 else "❌"

    comparison.append({
        'Model': name,
        'Test F1': f"{test_f1:.4f}",
        'CV F1': f"{cv_f1:.4f}",
        'Difference': f"{diff:+.4f}",
        'Reliable?': reliable
    })

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))

print("\n📌 Interpretation:")
print("  ✅ Difference < 0.03 → Model generalizes well")
print("  ⚠️ Difference 0.03-0.05 → Slight overfitting possible")
print("  ❌ Difference > 0.05 → Test score is optimistic, CV more reliable")

