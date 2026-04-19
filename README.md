# Aspect-Based Sentiment Analysis on Student Feedback

## Overview

This project focuses on analyzing student feedback using aspect-based sentiment analysis (ABSA) to extract meaningful and actionable insights. The system is designed to process unstructured textual feedback and align it with structured evaluation formats to support academic decision-making.

## Approach

The overall approach follows a pipeline-based design:

* Collection of publicly available student feedback data
* Construction of a semi-synthetic dataset by adding academic context (course, faculty, etc.)
* Generation of structured responses (MCQs) aligned with textual sentiment
* Development of an aspect-based analysis pipeline to extract key themes from feedback

## Methodology

* **Data Preparation**

  * Real feedback text was used as the base
  * Additional academic attributes were simulated to resemble institutional data
  * Structured MCQ responses were generated to mimic standard feedback forms

* **Aspect Extraction**

  * Multiple approaches were explored, including keyword-based methods, topic modeling (LDA), clustering, and transformer-based techniques
  * A hybrid strategy was adopted, where transformer-based extraction is refined using lightweight rule-based methods to improve coverage

* **Sentiment Analysis**

  * Sentiment is analyzed at the sentence level
  * Results are aggregated to derive aspect-level sentiment insights

* **Data Processing**

  * Multi-aspect handling is supported for each feedback
  * Non-informative or generic feedback is filtered or refined to improve analysis quality

## Dataset

* Semi-synthetic dataset combining:

  * Real-world feedback text
  * Simulated academic metadata
* Includes:

  * Feedback text
  * Course and faculty information
  * Extracted aspects
  * Sentiment labels and scores
  * Generated MCQ responses

## Use Case

The system is intended to assist academic departments in:

* Understanding student feedback at a granular (aspect) level
* Identifying key areas such as teaching quality, workload, and evaluation
* Supporting data-driven improvements in course delivery and structure

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Sentence Transformers
* NLTK
* Streamlit (for dashboard integration)

## Status

This is an ongoing project. The current implementation focuses on building and validating the analysis pipeline, with further improvements and deployment planned.
