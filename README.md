# 🐦 Twitter Sentiment Analysis using Logistic Regression

This project implements a **Twitter Sentiment Analysis pipeline** using **Logistic Regression** on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).  
It demonstrates preprocessing, TF-IDF feature extraction, and classification, and evaluates performance with multiple train/test splits.

---

## 📌 Features
- Dataset preprocessing (removes URLs, mentions, special characters)
- TF-IDF vectorization with uni-grams and bi-grams
- Logistic Regression classifier
- Evaluation metrics: Accuracy, Precision, Recall, Specificity, F1-score, ROC-AUC
- Confusion matrix & ROC curve plots
- Results export as CSV & Markdown (ready for research papers)
- **Optimized Training**: Faster runtime with dataset sampling and reduced grid search space

---

## 📂 Project Structure
```
twitter-lr-sentiment-project/
├── README.md                  <- Project documentation
├── requirements.txt            <- Dependencies
├── data/                       <- Place Sentiment140 dataset here
├── results/                    <- Outputs (metrics, plots, tables)
└── src/
    ├── utils.py                <- Tweet preprocessing functions
    ├── train_lr_twitter.py     <- Main training & evaluation script
    ├── grid_search_lr.py       <- Hyperparameter tuning (GridSearchCV)
    └── results_to_md.py        <- Convert CSV results to Markdown table
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/ajstyles004/twitter-lr-sentiment.git
cd twitter-lr-sentiment-project
```

### 2. Create virtual environment
```bash
python -m venv .venv
# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Activate (Linux/macOS)
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
- Get **Sentiment140** from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).  
- Place `training.1600000.processed.noemoticon.csv` inside the `data/` folder.

---

## 🚀 Usage

### Train Logistic Regression
```bash
python src/train_lr_twitter.py --data-path "data/training.1600000.processed.noemoticon.csv" --sample-size 50000 --test-sizes "0.1,0.2,0.3" --max-features 10000 --ngram-max 2 --penalty l2 --C 1.0 --solver liblinear --out-dir results
```

👉 Parameters:
- `--sample-size`: number of tweets to sample (`-1` = full dataset)
- `--test-sizes`: list of test splits (e.g. `"0.1,0.2,0.3"`)
- `--max-features`: max TF-IDF vocabulary size
- `--ngram-max`: use uni-grams (`1`), bi-grams (`2`), etc.
- `--penalty`: `l1` or `l2` regularization
- `--C`: regularization strength
- `--solver`: solver for Logistic Regression (`liblinear`, `saga`)

---

### Hyperparameter Search
```bash
python src/grid_search_lr.py --data-path "data/training.1600000.processed.noemoticon.csv" --sample-size 50000 --test-size 0.2
```

---

### Convert Results to Markdown
```bash
python src/results_to_md.py --csv results/logistic_regression_results.csv
```
Example output (ready to paste into papers):

| Train/Test Split | Accuracy (%) | Precision | Recall | Specificity | F1 Score | ROC-AUC |
|------------------|--------------|-----------|--------|-------------|----------|---------|
| 90/10            | 78.87        | 0.779     | 0.807  | 0.771       | 0.793    | 0.870   |
| 80/20            | 79.01        | 0.784     | 0.802  | 0.778       | 0.793    | 0.870   |
| 70/30            | 78.71        | 0.781     | 0.799  | 0.775       | 0.790    | 0.868   |

---

## 📊 Results
- **Accuracy**: ~79%
- **ROC-AUC**: ~0.87
- Balanced **precision & recall**
- Strong baseline for comparing with deep learning models (e.g., LSTM, CNN)

---

## 📝 Citation
If you use this project in research, cite the dataset:
> Go, Alec, Richa Bhayani, and Lei Huang. *Sentiment140 dataset with 1.6 million tweets*. 2009.

---

## 📌 Future Work
- Add deep learning baselines (LSTM, CNN)
- Handle emojis, sarcasm, and slang better
- Try larger TF-IDF vocab sizes and char-grams

---


