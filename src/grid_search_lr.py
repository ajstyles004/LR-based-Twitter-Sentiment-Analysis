
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from utils import clean_tweet

def load_sentiment140(csv_path: str, sample_size: int = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin-1", header=None, names=[
        "target", "ids", "date", "flag", "user", "text"
    ])
    df = df[df["target"].isin([0,4])].copy()
    df["sentiment"] = df["target"].map({0:0, 4:1})
    df = df[["text","sentiment"]].dropna()
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(sample_size, random_state=seed)
    df["text"] = df["text"].astype(str).apply(clean_tweet)
    df = df[df["text"].str.len()>0].reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser(description="Grid Search for Logistic Regression (Twitter Sentiment140)")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--sample-size", type=int, default=50000)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_sentiment140(args.data_path, args.sample_size, args.seed)

    # Pipeline manually: we grid over vectorizer + LR params
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=args.test_size, random_state=args.seed, stratify=df["sentiment"]
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    lr = LogisticRegression(max_iter=1000, solver="liblinear")

    param_grid = {
        "C": [0.1, 0.5, 1.0, 3.0, 10.0],
        "penalty": ["l1", "l2"]
    }

    grid = GridSearchCV(lr, param_grid, cv=3, scoring=make_scorer(f1_score), verbose=2, n_jobs=-1)
    grid.fit(X_train_tfidf, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best CV score (F1):", grid.best_score_)

    # Evaluate best on test
    best = grid.best_estimator_
    y_pred = best.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Test Accuracy:", acc)
    print("Test F1:", f1)

if __name__ == "__main__":
    main()
