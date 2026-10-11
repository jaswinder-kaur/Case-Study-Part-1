"""
Sentiment Analysis for Customer Feedback
Course: PhDAI-730 Statistics for AI - Case Study Part 1
Dataset: Twitter Airline Sentiment Dataset (Tweets.csv)
Author: Jaswinder Kaur
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# 1. DATA LOADING AND EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------
def load_and_explore_data(filepath='Tweets.csv'):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape}")
    print("\nTarget Class Distribution:")
    print(df['airline_sentiment'].value_counts(normalize=True).round(4) * 100)
    
    print("\nFeedback by Carrier:")
    print(df.groupby(['airline', 'airline_sentiment']).size().unstack(fill_value=0))
    return df

# ---------------------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------------------
def clean_tweet_text(text):
    """
    Cleans raw tweet text by removing mentions, URLs, non-alphabetical characters,
    and converts to lowercase.
    """
    if not isinstance(text, str):
        return ""
    # Remove user handles (@name)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove web hyperlinks
    text = re.sub(r'https?://\S+', '', text)
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase and trim extra whitespace
    text = text.lower().strip()
    return text

def preprocess_dataframe(df):
    print("\n[INFO] Preprocessing text data...")
    df['clean_text'] = df['text'].apply(clean_tweet_text)
    # Filter empty cleaned text entries
    df_clean = df[df['clean_text'].str.strip() != ''].copy()
    print(f"Valid records remaining post-cleaning: {len(df_clean)}")
    return df_clean

# ---------------------------------------------------------
# 3. FEATURE EXTRACTION & DATA SPLITTING
# ---------------------------------------------------------
def prepare_features(df_clean):
    print("\n[INFO] Partitioning dataset (80/20 Stratified Split)...")
    X = df_clean['clean_text']
    y = df_clean['airline_sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("[INFO] Extracting TF-IDF Features (Unigrams + Bigrams, max_features=5000)...")
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )
    
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    return X_train_vec, X_test_vec, y_train, y_test, tfidf

# ---------------------------------------------------------
# 4. MODEL DEVELOPMENT AND EVALUATION
# ---------------------------------------------------------
def train_and_evaluate_models(X_train_vec, X_test_vec, y_train, y_test):
    labels = ['negative', 'neutral', 'positive']
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Linear Support Vector Machine": LinearSVC(C=1.0, dual=False, random_state=42)
    }
    
    results = []
    confusion_matrices = {}
    
    for name, model in models.items():
        print(f"\n==========================================")
        print(f" Training: {name}")
        print(f"==========================================")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        acc = accuracy_score(y_test, y_pred)
        rep = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        confusion_matrices[name] = cm
        
        print(classification_report(y_test, y_pred))
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Macro Precision': rep['macro avg']['precision'],
            'Macro Recall': rep['macro avg']['recall'],
            'Macro F1': rep['macro avg']['f1-score'],
            'Weighted F1': rep['weighted avg']['f1-score']
        })
        
    summary_df = pd.DataFrame(results)
    return summary_df, confusion_matrices

# ---------------------------------------------------------
# 5. VISUALIZATION MODULE
# ---------------------------------------------------------
def plot_confusion_matrices(confusion_matrices):
    labels = ['negative', 'neutral', 'positive']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"{name}\nConfusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=300)
    print("\n[INFO] Confusion matrices saved to 'confusion_matrices.png'")
    plt.show()

# ---------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ---------------------------------------------------------
if __name__ == '__main__':
    # Step 1: Load Data
    df = load_and_explore_data('Tweets.csv')
    
    # Step 2: Clean & Preprocess
    df_clean = preprocess_dataframe(df)
    
    # Step 3: Feature Extraction
    X_train_vec, X_test_vec, y_train, y_test, tfidf = prepare_features(df_clean)
    
    # Step 4: Model Training and Evaluation
    summary_df, cms = train_and_evaluate_models(X_train_vec, X_test_vec, y_train, y_test)
    
    print("\n" + "="*50)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("="*50)
    print(summary_df.to_string(index=False))
    
    # Step 5: Plot Results
    plot_confusion_matrices(cms)