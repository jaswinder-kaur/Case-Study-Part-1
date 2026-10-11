# Sentiment Analysis for Customer Feedback: Commercial Airline Twitter Corpus

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

## 1. Project Overview
Automated sentiment analysis enables enterprise service operations to extract real-time intelligence from high-velocity consumer communication streams. In commercial aviation, passengers frequently turn to microblog platforms during irregular operations (IROPS)—such as maintenance delays, cancellations, and baggage handling disruptions.

This project implements an end-to-end statistical Natural Language Processing (NLP) pipeline and machine learning benchmark evaluating customer feedback from the **Twitter Airline Sentiment Benchmark** ($N = 14,640$). The objective is to classify short unstructured reviews into **negative**, **neutral**, or **positive** sentiments to facilitate automated grievance routing and operational triage.

---

## 2. Dataset Description & Summary Statistics
The dataset contains customer feedback directed at six major commercial air carriers: American Airlines, Delta Air Lines, Southwest Airlines, United Airlines, US Airways, and Virgin America.

- **Total Observations ($N$):** 14,640 records across 15 attributes
- **Target Variable:** `airline_sentiment` (3 classes: `negative`, `neutral`, `positive`)
- **Observed Class Distribution:**
  - **Negative:** 9,178 tweets (62.69%)
  - **Neutral:** 3,099 tweets (21.17%)
  - **Positive:** 2,363 tweets (16.14%)
- **Distribution Across Air Carriers:**
  - **United:** 3,822 tweets (68.89% negative)
  - **US Airways:** 2,913 tweets (77.69% negative)
  - **American:** 2,759 tweets (71.04% negative)
  - **Southwest:** 2,420 tweets (49.01% negative)
  - **Delta:** 2,222 tweets (42.98% negative)
  - **Virgin America:** 504 tweets (35.91% negative)
- **Top Grievance Categories:**
  1. Customer Service Issue ($n = 2,910$)
  2. Late Flight ($n = 1,665$)
  3. Can't Tell / Unspecified ($n = 1,190$)
  4. Cancelled Flight ($n = 847$)
  5. Lost Luggage ($n = 724$)

---

## 3. Methodology & Technical Pipeline

The workflow proceeds in four distinct stages:

1. **Text Preprocessing:**
   - **Entity Stripping:** Mentions (`@[A-Za-z0-9_]+`) and URLs (`https?://\S+`) are removed using regular expressions to eliminate carrier-specific bias.
   - **Character Normalization:** Non-alphabetical symbols, emojis, and digits are filtered out (`[^a-zA-Z\s]`).
   - **Lowercasing:** Uniform character folding is applied across all tokens.
   - **Stop-Word Pruning:** Common syntactic words with low predictive entropy are filtered using the built-in English stop-word lexicon.
2. **Dataset Partitioning:**
   - Evaluated via an 80/20 Stratified Train-Test split ($N_{\text{train}} = 11,712$; $N_{\text{test}} = 2,928$, `random_state=42`) to maintain consistent sentiment class distribution across splits.
3. **Feature Extraction:**
   - **Model:** Term Frequency–Inverse Document Frequency (TF-IDF) vectorizer.
   - **N-Gram Range:** Unigrams and bigrams ($n \in \{1, 2\}$) to preserve localized contextual phrases (e.g., *"not happy"*, *"never again"*).
   - **Configuration:** Vocabulary capped at 5,000 terms (`max_features=5000`), sub-linear term frequency scaling enabled (`sublinear_tf=True`), and Euclidean $L_2$ document vector normalization applied.
4. **Supervised Classification:**
   - **Multinomial Naive Bayes (MNB):** Generative probabilistic model using Laplace smoothing ($\alpha = 0.5$).
   - **Logistic Regression (LR):** Convex discriminative classifier with $L_2$ regularization penalty ($C = 1.0$), L-BFGS solver, and multinomial cross-entropy loss.
   - **Linear Support Vector Machine (Linear SVC):** Maximum-margin hyperplanes using coordinate descent (`dual=False`, $C = 1.0$).

---

## 4. Benchmark Experimental Results

Evaluated on the independent 20% stratified test set ($n = 2,928$):

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

### Key Diagnostic Findings
- **Logistic Regression** achieved the highest overall accuracy (**76.98%**).
- **Linear SVC** produced the strongest class-balanced discrimination, leading in Macro Recall (**0.6735**) and Macro $F_1$ (**0.6909**), driven by superior performance on the minority neutral class ($F_1 = 0.55$).
- **Multinomial Naive Bayes** suffered heavily from class imbalance, misclassifying 408 neutral reviews as negative due to the strong prior probability of negative reviews.

---

## 5. Software Requirements & Dependencies

- **Python Version:** 3.9+ (compatible with 3.10, 3.11, 3.12)
- **Key Python Packages:**
  - `numpy>=1.24.0`
  - `pandas>=2.0.0`
  - `scikit-learn>=1.3.0`
  - `matplotlib>=3.7.0`
  - `seaborn>=0.12.0`

---

## 6. Installation & Execution Guide

### Step 1: Clone the Repository
```bash
git clone [https://github.com/jaswinder-kaur/Case-Study-Part-1.git](https://github.com/jaswinder-kaur/Case-Study-Part-1.git)
cd Case-Study-Part-1
```

### Step 2: Set Up a Virtual Environment
On Windows (PowerShell):
```PowerShell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv\Scripts\Activate.ps1
```
On Windows (Command Prompt):
```DOS
python -m venv .venv
.venv\Scripts\activate.bat
```
On macOS / Linux:
```Bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```Bash
pip install -r requirements.txt
```

### Step 4: Run the Pipeline
Ensure Tweets.csv is located in the root project folder, then run:

```Bash
python sentiment_analysis.py
```

### Step 5: Outputs Produced
Descriptive dataset statistics and per-carrier sentiment distribution printed to the console.

Multi-class classification reports (Precision, Recall, F1-Score) for MNB, Logistic Regression, and Linear SVC.

Final comparative benchmark table summarizing performance across all evaluated models.

High-resolution multi-class confusion matrix plot exported to the root directory as confusion_matrices.png.

---
## 7. Repository Structure
```Plaintext
Case-Study-Part-1/
├── .gitignore               # Excludes virtual environments, caches, and temporary files
├── LICENSE                  # MIT License
├── README.md                # Project documentation and execution guide
├── requirements.txt         # Pinned package dependencies
├── Tweets.csv               # Dataset file
├── sentiment_analysis.py    # Main processing and model evaluation pipeline
└── confusion_matrices.png   # Generated comparative confusion matrix plot
```

---
## 8. Author & Course Metadata
Author: Jaswinder Kaur

Program: Ph.D. in Artificial Intelligence

Course: Statistics for AI (PhDAI-730)

Institution: University of the Cumberlands

Dataset Reference: Crowdflower / Kaggle Twitter Airline Sentiment Dataset
