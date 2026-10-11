# Sentiment Analysis for Customer Feedback: Commercial Airline Twitter Corpus

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

## Executive Overview
Automated sentiment analysis enables enterprise service operations to extract real-time intelligence from high-velocity consumer communication streams. In commercial aviation, passengers frequently turn to microblog platforms during irregular operations (IROPS)—such as maintenance delays, cancellations, and baggage handling disruptions.

This project delivers an end-to-end statistical Natural Language Processing (NLP) pipeline and machine learning benchmark evaluating customer feedback from the **Twitter Airline Sentiment Benchmark** ($N = 14,640$). The objective is to classify short unstructured reviews into **negative**, **neutral**, or **positive** sentiments to facilitate automated grievance routing and operational triage.

---

## Dataset Description & Summary Statistics
The dataset contains customer feedback directed at six major commercial US air carriers: American Airlines, Delta Air Lines, Southwest Airlines, United Airlines, US Airways, and Virgin America.

* **Total Observations ($N$):** 14,640 records across 15 attributes.
* **Target Variable:** `airline_sentiment` (3 classes).
* **Observed Class Distribution:**
  * **Negative:** 9,178 tweets (62.69%)
  * **Neutral:** 3,099 tweets (21.17%)
  * **Positive:** 2,363 tweets (16.14%)
* **Top Grievance Categories:**
  1. Customer Service Issue ($n = 2,910$)
  2. Late Flight ($n = 1,665$)
  3. Can't Tell / Unspecified ($n = 1,190$)
  4. Cancelled Flight ($n = 847$)
  5. Lost Luggage ($n = 724$)

---

## Methodology Pipeline

### 1. Text Preprocessing
* **Entity Stripping:** Twitter user mentions (`@[A-Za-z0-9_]+`) and URLs (`https?://\S+`) are excised using regular expressions to eliminate spurious brand bias.
* **Character Normalization:** Non-alphabetical symbols, emojis, and numeric digits are filtered out (`[^a-zA-Z\s]`).
* **Lowercasing:** Uniform character folding is applied across all tokens.
* **Stop-Word Pruning:** Syntactic particles with low entropy are filtered using the standard English stop-word lexicon.

### 2. Feature Extraction
* **Vector Space Model:** Term Frequency–Inverse Document Frequency (TF-IDF) vectorization.
* **N-Gram Range:** Unigrams and bigrams ($n \in \{1, 2\}$) to capture localized negation (e.g., *"not happy"*, *"never again"*).
* **Feature Dimension:** Vocabulary capped at the top 5,000 most informative features (`max_features=5000`) with sub-linear term frequency scaling and Euclidean $L_2$ document normalization.

### 3. Model Architecture & Training Strategy
* **Data Partitioning:** 80/20 Stratified Train-Test split ($N_{\text{train}} = 11,712$; $N_{\text{test}} = 2,928$) to preserve identical class balance across sets.
* **Classifiers Evaluated:**
  * **Multinomial Naive Bayes (MNB):** Laplace smoothing ($\alpha = 0.5$).
  * **Logistic Regression (LR):** $L_2$ regularization penalty ($C = 1.0$), L-BFGS solver, multi-class cross-entropy.
  * **Linear Support Vector Machine (Linear SVC):** Maximum-margin hyperplane classification with hinge loss ($C = 1.0$).

---

## Benchmark Experimental Results

The models were evaluated on the independent held-out test set ($n = 2,928$):

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro $F_1$-Score | Weighted $F_1$-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 74.62% | 0.7411 | 0.5833 | 0.6215 | 0.7140 |
| **Logistic Regression ($L_2$)** | **76.98%** | **0.7449** | 0.6418 | 0.6766 | 0.7525 |
| **Linear Support Vector Machine** | 76.67% | 0.7145 | **0.6735** | **0.6909** | **0.7612** |

### Per-Class $F_1$-Score Diagnostics
| Class | Support | Multinomial NB | Logistic Regression | Linear SVC |
| :--- | :---: | :---: | :---: | :---: |
| **Negative** | 1,835 | 0.84 | **0.86** | **0.86** |
| **Neutral** | 620 | 0.41 | 0.50 | **0.55** |
| **Positive** | 473 | 0.61 | **0.67** | **0.67** |

### Key Diagnostic Takeaway
* **Logistic Regression** achieved the highest overall classification accuracy (**76.98%**).
* **Linear SVC** achieved the most robust class-balanced discrimination, leading across Macro Recall (**0.6735**) and Macro $F_1$ (**0.6909**), driven by superior performance on the minority neutral class ($F_1 = 0.55$).

---

## Project Setup & Reproduction Guide

### Prerequisites
* Python 3.9 or higher
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/jaswinder-kaur/Case-Study-Part-1.git](https://github.com/jaswinder-kaur/Case-Study-Part-1.git)
cd Case-Study-Part-1
