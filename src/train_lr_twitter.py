import pandas as pd
import re
import emoji
import time
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache
from contractions import fix as fix_contractions
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer # Using SnowballStemmer as a small, free improvement
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score)
from scipy.sparse import hstack, csr_matrix
import numpy as np

# --- Download NLTK data (if not already downloaded) ---
try:
    stopwords.words('english')
except LookupError:
    import nltk
    print("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    print("NLTK data downloaded.")

print("Script started...")

# --- Step 1. Load Data ---
print("Step 1/7: Loading data...")
start_time = time.time()
DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv',
                 encoding='ISO-8859-1', names=DATASET_COLUMNS)

print(f"Running on FULL dataset ({len(df)} samples)")
print(f"Data ready in {time.time() - start_time:.2f} seconds.")

# --- Step 2. Preprocessing ---
print("Step 2/7: Preprocessing tweets...")
start_time = time.time()

df_clean = df[['sentiment', 'text']].copy()
df_clean['sentiment'] = df_clean['sentiment'].replace(4, 1)

stop_words = set(stopwords.words('english'))
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
# Using SnowballStemmer - it's an improved version of PorterStemmer
stemmer = SnowballStemmer("english")

@lru_cache(maxsize=100000)
def preprocess_tweet(text):
    text = str(text).lower()
    text = emoji.demojize(text)
    text = fix_contractions(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"(\w)\1{2,}", r"\1\1", text)
    tokens = tokenizer.tokenize(text)
    clean_tokens = [stemmer.stem(tok) for tok in tokens if tok not in stop_words]
    return " ".join(clean_tokens)

df_clean['clean'] = df_clean['text'].apply(preprocess_tweet)
print(f"Preprocessing done in {time.time() - start_time:.2f} seconds.")

# --- Step 3. Train/Validation/Test Split ---
print("Step 3/7: Splitting data into train, validation, and test sets...")
X = df_clean['clean']
y = df_clean['sentiment']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42)

print(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples.")

# --- Step 4. Feature Extraction (STRATEGY 2 APPLIED) ---
print("Step 4/7: Extracting features with increased feature set...")
start_time = time.time()

# Word-level TF-IDF with INCREASED features for higher accuracy
tfidf_word = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_word = tfidf_word.fit_transform(X_train)
X_val_word = tfidf_word.transform(X_val)
X_test_word = tfidf_word.transform(X_test)

# Character-level TF-IDF with INCREASED features for higher accuracy
tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=20000)
X_train_char = tfidf_char.fit_transform(X_train)
X_val_char = tfidf_char.transform(X_val)
X_test_char = tfidf_char.transform(X_test)

sia = SentimentIntensityAnalyzer()
train_vader = np.array([sia.polarity_scores(t)['compound'] for t in X_train]).reshape(-1, 1)
val_vader = np.array([sia.polarity_scores(t)['compound'] for t in X_val]).reshape(-1, 1)
test_vader = np.array([sia.polarity_scores(t)['compound'] for t in X_test]).reshape(-1, 1)

X_train_final = hstack([X_train_word, X_train_char, csr_matrix(train_vader)])
X_val_final = hstack([X_val_word, X_val_char, csr_matrix(val_vader)])
X_test_final = hstack([X_test_word, X_test_char, csr_matrix(test_vader)])

print(f"Features ready in {time.time() - start_time:.2f} seconds.")

# --- Step 5. Model Training ---
print("Step 5/7: Training Logistic Regression and measuring time...")
model = LogisticRegression(solver='saga', C=1.0, max_iter=1000, random_state=42, n_jobs=-1)

train_start_time = time.time()
model.fit(X_train_final, y_train)
train_end_time = time.time()
training_time_seconds = train_end_time - train_start_time
training_time_formatted = time.strftime("%Mm %Ss", time.gmtime(training_time_seconds))

print(f"Model trained in {training_time_formatted}.")

# --- Step 6. Calculate All Metrics ---
print("Step 6/7: Calculating all performance metrics...")
y_val_pred = model.predict(X_val_final)
validation_acc = accuracy_score(y_val, y_val_pred)

y_test_pred = model.predict(X_test_final)
y_test_prob = model.predict_proba(X_test_final)[:, 1]

test_acc = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='macro')
recall = recall_score(y_test, y_test_pred, average='macro')
f1 = f1_score(y_test, y_test_pred, average='macro')
roc_auc = roc_auc_score(y_test, y_test_prob)

# --- Step 7. Display Final Summary & Save Results ---
print("\n" + "="*40)
print("          MODEL PERFORMANCE SUMMARY")
print("="*40)
summary_data = {
    "Metric": ["Model", "Validation_acc", "Test_acc", "Precision", "Recall", "F1_score", "AUC", "Training_time"],
    "Value": [
        "Logistic Regression",
        f"{validation_acc*100:.2f}%",
        f"{test_acc*100:.2f}%",
        f"{precision*100:.2f}%",
        f"{recall*100:.2f}%",
        f"{f1*100:.2f}%",
        f"{roc_auc*100:.2f}%",
        training_time_formatted
    ]
}
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print("="*40 + "\n")

print("Saving detailed report and charts...")
os.makedirs("results_summary", exist_ok=True)
summary_df.to_csv("results_summary/performance_summary.csv", index=False)

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative","Positive"],
            yticklabels=["Negative","Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("results_summary/confusion_matrix.png", dpi=300)
plt.close()

# ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1],'--',color='gray', label='Random Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("results_summary/roc_curve.png", dpi=300)
plt.close()

print("Detailed results saved in /results_summary/ folder.")
print("Script finished successfully.")
