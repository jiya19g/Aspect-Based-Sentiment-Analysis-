# 📊 Aspect-Based Sentiment Analysis of Student Feedback

An AI-powered educational analytics system that performs **Aspect-Based Sentiment Analysis (ABSA)** on student feedback using **Machine Learning, Deep Learning, NLP, and Streamlit**.
The project analyzes student reviews across multiple academic aspects such as teaching quality, assessments, labs, workload, mentoring, and learning outcomes to generate actionable institutional insights.

Built as a final year B.Tech major project at **SRM Institute of Science and Technology**. 

---

# 🚀 Dashboard Preview

## 🖥️ Main Dashboard

![Dashboard](<img width="894" height="480" alt="image" src="https://github.com/user-attachments/assets/f94b9109-12a6-46d8-85ca-442eecdd3f3e" />

*Fig 4.6 Dashboard*

---

## 🔍 Single Feedback Analysis

![Single Feedback Analysis](sandbox:/mnt/data/d8cf929e-938a-464c-940b-5070c6a59b85.png)

*Fig 4.7 Single Feedback Analysis*

---

## 📂 Bulk Feedback Analysis

![Bulk Feedback Analysis](sandbox:/mnt/data/6a9f6b2b-de50-43e5-866d-241fa62ef935.png)

*Fig 4.8 Bulk Feedback Analysis*

---

## 👨‍🏫 Faculty Wise Analysis

![Faculty Wise Analysis](sandbox:/mnt/data/173e33b8-f8fe-41ff-ae9b-211deb8986dd.png)

*Fig 4.9 Faculty Wise Analysis*

---

## 📈 Sentiment Distributions

![Sentiment Distributions](sandbox:/mnt/data/5b75694b-485b-492f-8fcf-5a82d1300432.png)

*Fig 4.10 Sentiment Distributions*

---

# ✨ Features

## 🔍 Aspect-Based Analysis

Detects sentiments for:

* Teaching Quality
* Teaching Methodology
* Assessment & Evaluation
* Mentoring & Support
* Practical & Lab
* Course Design & Workload
* Learning Outcomes

---

## 📈 Interactive Dashboard

Built using Streamlit with:

* Faculty-wise analysis
* Sentiment distributions
* Course analytics
* Feedback trends
* Interactive charts
* Bulk CSV upload
* Real-time feedback analysis

---

## 🤖 AI & NLP Pipeline

* Sentence-BERT based aspect labeling
* CNN + LSTM sentiment classifier
* Comparative ML benchmarking
* Semantic similarity analysis
* Text normalization & slang handling
* Multi-aspect feedback detection

---

## 📊 Visualization Support

Integrated with:

* Plotly
* Matplotlib
* Seaborn

For:

* Pie charts
* Heatmaps
* Distribution plots
* Sentiment graphs
* Faculty comparison charts

---

# 🧠 Tech Stack

## Frontend

* Streamlit

## Backend / NLP

* TensorFlow
* scikit-learn
* Sentence Transformers
* NLTK

## Data Processing

* Pandas
* NumPy

## Visualization

* Plotly
* Matplotlib
* Seaborn
* WordCloud

---

# 📂 Project Structure

```bash
├── app.py
├── dashboard/
├── models/
├── dataset/
├── notebooks/
├── utils/
├── requirements.txt
├── README.md
└── assets/
```

---

# 📦 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/aspect-based-student-feedback-analysis.git

cd aspect-based-student-feedback-analysis
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📋 Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
tensorflow>=2.13.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
plotly>=5.17.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=10.0.0
openpyxl>=3.1.0
wordCloud
nltk
torchvision
```

---

# 🧪 Models Evaluated

## Machine Learning Models

* Logistic Regression
* Linear SVM
* Random Forest
* XGBoost
* SGD Classifier
* KNN
* Naive Bayes
* Decision Tree
* AdaBoost
* Extra Trees
* Gradient Boosting

---

## Deep Learning Models

* CNN
* LSTM
* BiLSTM
* GRU
* CNN + LSTM ✅
* Attention-based architectures

---

# 📊 Dataset Information

The dataset was built using:

* Public educational review datasets
* Real student feedback
* Synthetic educational feedback generation
* Sentence-level aspect annotations

### Dataset Size

📌 `133,065` feedback records 

### Includes

* Feedback text
* Aspect labels
* Sentiment labels
* Course metadata
* Faculty metadata
* MCQ responses

---

# 🏗️ Methodology

## Pipeline

```text
Data Collection
      ↓
Data Cleaning
      ↓
Aspect Labeling using Sentence-BERT
      ↓
Sentiment Labeling
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Performance Benchmarking
      ↓
Interactive Dashboard Deployment
```

---

# 📈 Key Findings

✅ CNN + LSTM hybrid achieved highest performance
✅ Attention mechanisms underperformed
✅ MCQ feature fusion gave no major improvements
✅ Text-only representations were highly effective
✅ Sentence-BERT improved semantic aspect assignment



---

# 🖥️ Dashboard Modules

## 📌 Single Feedback Analysis

Analyze one student review in real-time.

## 📌 Bulk Feedback Analysis

Upload CSV files for large-scale sentiment analysis.

## 📌 Faculty Wise Analysis

Compare faculty performance using aspect sentiment scores.

## 📌 Sentiment Distribution

Visualize positive, negative, and neutral distributions.

## 📌 Analytics & Insights

Generate recommendations and institutional insights.

---

# 🎯 Sustainable Development Goals

This project aligns with:

* 🎓 **SDG 4 – Quality Education**
* ⚙️ **SDG 9 – Industry, Innovation and Infrastructure**



---

# 📚 Research Contributions

* Large-scale educational ABSA dataset
* Benchmark comparison of 17 models
* Educational-domain sentiment analysis framework
* Reproducible methodology for academic feedback analytics

---

# 👨‍💻 Authors

* Akshat Neolia
* Vikrant Kumar Singh
* Jiya Gayawer

Under the guidance of:

Dr. V. M. Gayathri
Associate Professor
Department of Networking and Communications
SRM Institute of Science and Technology 

---

# 📄 Publications

📌 *Aspect-Based Sentiment Analysis of Student Feedback: A Model Comparison Study* 

---

# 📜 License

This project is developed for academic and research purposes.

---

# ⭐ Future Enhancements

* Transformer fine-tuning
* Real-time institutional deployment
* Multilingual feedback analysis
* Voice-based feedback support
* Explainable AI integration
* LLM-powered educational insights

---

# 🙌 Acknowledgements

Special thanks to:

* SRM Institute of Science and Technology
* Department of Networking and Communications
* Faculty mentors and reviewers
* Open-source AI/NLP community

