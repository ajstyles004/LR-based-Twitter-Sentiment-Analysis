import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

from utils import clean_tweet


def load_sentiment140(csv_path: str, sample_size: int = None, seed: int = 42) -> pd.DataFrame:
    # Load Sentiment140 CSV (no header, latin-1). Columns: target, ids, date, flag, user, text
    df = pd.read_csv(csv_path, encoding='latin-1', header=None, names=[
        'target', 'ids', 'date', 'flag', 'user', 'text'
    ])
    # keep only pos/neg
    df = df[df['target'].isin([0, 4])].copy()
    df['sentiment'] = df['target'].map({0: 0, 4: 1})
    df = df[['text', 'sentiment']].dropna()
    # optional sampling
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(sample_size, random_state=seed)
    # clean
    df['text'] = df['text'].astype(str).apply(clean_tweet)
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    return df


def train_and_evaluate(
    texts: List[str], labels: np.ndarray, test_sizes: List[float],
    max_features: int, ngram_max: int, penalty: str, C: float, solver: str,
    out_dir: str, seed: int = 42
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max))
    X = vectorizer.fit_transform(texts)
    y = labels
    results = []
    # ROC curves figure
    plt.figure()
    for test_size in test_sizes:
        split_name = f"{int((1-test_size)*100)}-{int(test_size*100)}"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        lr = LogisticRegression(max_iter=1000, C=C, penalty=penalty, solver=solver,
                               n_jobs=None if solver!='liblinear' else None)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results.append({
            'Train/Test Split': f"{int((1-test_size)*100)}/{int(test_size*100)}",
            'Accuracy (%)': round(acc*100, 2),
            'Precision': round(prec, 3),
            'Recall': round(rec, 3),
            'Specificity': round(spec, 3),
            'F1 Score': round(f1, 3),
            'ROC-AUC': round(auc, 3)
        })
        # Confusion matrix figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest')
        ax.set_title(f'Confusion Matrix ({split_name} split)')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Pred 0','Pred 1'])
        ax.set_yticklabels(['Actual 0','Actual 1'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'confusion_matrix_{split_name}.png'))
        plt.close(fig)
        # ROC curve on aggregate figure
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{split_name} (AUC={auc:.2f})")
    # finalize ROC figure
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Logistic Regression (Twitter Sentiment140)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_curves.png'))
    plt.close()
    # Accuracy vs split
    import pandas as pd
    df_results = pd.DataFrame(results)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = list(range(len(df_results)))
    ax.plot(x, df_results['Accuracy (%)'], marker='o')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['Train/Test Split'])
    ax.set_xlabel('Train/Test Split')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Train/Test Split (LR)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'accuracy_vs_split.png'))
    plt.close(fig)
    df_results.to_csv(os.path.join(out_dir, 'logistic_regression_results.csv'), index=False)
    return df_results

def main():
    ap = argparse.ArgumentParser(description='Twitter Sentiment Analysis with Logistic Regression (TF-IDF)')
    ap.add_argument('--data-path', type=str, required=True,
                    help='Path to Sentiment140 CSV (training.1600000.processed.noemoticon.csv)')
    ap.add_argument('--sample-size', type=int, default=100000,
                    help='Optional sample size to speed up (set -1 for all rows)')
    ap.add_argument('--test-sizes', type=str, default='0.1,0.2,0.3',
                    help="Comma-separated list of test sizes (e.g., '0.1,0.2,0.3')")
    ap.add_argument('--max-features', type=int, default=8000,
                    help='TF-IDF max_features')
    ap.add_argument('--ngram-max', type=int, default=2,
                    help='TF-IDF ngram upper bound (1=unigram, 2=bigram)')
    ap.add_argument('--penalty', type=str, default='l2', choices=['l1','l2'],
                    help='Logistic regression regularization')
    ap.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength')
    ap.add_argument('--solver', type=str, default='liblinear', choices=['liblinear','saga'],
                    help='liblinear for small/medium, saga for large + l1/l2')
    ap.add_argument('--out-dir', type=str, default='results', help='Output directory for figures/CSV')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    args = ap.parse_args()
    test_sizes = [float(s) for s in args.test_sizes.split(',')]
    sample = None if args.sample_size == -1 else args.sample_size
    df = load_sentiment140(args.data_path, sample_size=sample, seed=args.seed)
    df_results = train_and_evaluate(
        texts=df['text'].tolist(), labels=df['sentiment'].values,
        test_sizes=test_sizes, max_features=args.max_features,
        ngram_max=args.ngram_max, penalty=args.penalty, C=args.C,
        solver=args.solver, out_dir=args.out_dir, seed=args.seed
    )
    print('\n=== Logistic Regression Results (Twitter Sentiment140) ===\n')
    print(df_results.to_string(index=False))

if __name__ == '__main__':
    main()
