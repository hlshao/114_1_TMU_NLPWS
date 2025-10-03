# ROCLING-SIGAI 2025 Shared Task Dataset

## Overview
This repository provides data and baseline resources for the **ROCLING-SIGAI 2025 Shared Task on Dimensional Sentiment Analysis**.  
The task focuses on predicting **sentiment intensity in Valence–Arousal dimensions** for Chinese text at both the **word** and **phrase** levels.

The dataset is designed for research and educational purposes. In particular, it is used as **teaching material** in the course *Natural Language Processing and Large Language Models Workshop* at **Taipei Medical University (TMU)**. During classroom exercises, the training data remain the same, while the test set may be modified for evaluation purposes.

## Task Description
- **Input**: Chinese words or phrases  
- **Output**: Predicted **Valence** (positivity–negativity) and **Arousal** (calm–excited) scores, typically ranging from 1 to 9  
- **Evaluation**: Pearson correlation coefficient and mean squared error (MSE) between system predictions and gold-standard annotations

This task provides a benchmark to evaluate **machine learning** and **deep learning** approaches to **affective computing**, including regression, representation learning, and large language model fine-tuning.

## Dataset
The dataset consists of:  
- **Training set**: Annotated Chinese words/phrases with gold-standard valence and arousal scores  
- **Validation set**: Provided for model tuning  
- **Test set**: Gold-standard labels hidden; used for leaderboard evaluation  
- **(For TMU Workshop)**: A variant test set will be used in classroom exercises

### Data Format
Each entry includes:
- `id` (identifier)
- `text` (Chinese word or phrase)
- `valence` (float)
- `arousal` (float)

Example:
```

id,text,valence,arousal
001,快樂,7.8,6.3
002,憤怒,2.1,7.5

````

---

## Getting Started

### 1) Environment
We recommend Python 3.10+.

```bash
# Option A: conda
conda create -n rocling2025 python=3.10 -y
conda activate rocling2025
pip install -U pip

# Option B: venv
python -m venv .venv
source .venv/bin/activate   # (Windows) .venv\Scripts\activate
pip install -U pip
````

Install minimal dependencies:

```bash
pip install pandas scikit-learn scipy numpy
```

(Optional) for transformer baselines:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate peft datasets
```

### 2) Directory Layout (suggested)

```
.
├── data/
│   ├── Train_Data.csv
│   └── Test_Data.csv          # not release yet
├── src/
│   └── baseline.py
└── README.md
```

### 3) Quick Baseline (Ridge + TF-IDF)

`src/baseline.py` (copy/paste):

```python
import argparse, pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import random

def set_seed(seed=42):
    np.random.seed(seed); random.seed(seed)

def pearson(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return pearsonr(y_true, y_pred).statistic

def train_eval(train_path, dev_path=None, out_pred_path=None):
    set_seed(42)
    df_tr = pd.read_csv(train_path)
    text = df_tr["text"].astype(str).values
    y_v = df_tr["valence"].values
    y_a = df_tr["arousal"].values

    # Simple TF-IDF + Ridge per target dimension
    model_v = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000),
        Ridge(alpha=2.0, random_state=42)
    )
    model_a = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000),
        Ridge(alpha=2.0, random_state=42)
    )

    # 5-fold CV on train
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds_v = np.zeros(len(df_tr))
    preds_a = np.zeros(len(df_tr))
    for tr_idx, va_idx in kf.split(text):
        model_v.fit(text[tr_idx], y_v[tr_idx])
        model_a.fit(text[tr_idx], y_a[tr_idx])
        preds_v[va_idx] = model_v.predict(text[va_idx])
        preds_a[va_idx] = model_a.predict(text[va_idx])

    print("[Train-CV] Valence  Pearson:", round(pearson(y_v, preds_v), 4),
          " MSE:", round(mean_squared_error(y_v, preds_v), 4))
    print("[Train-CV] Arousal  Pearson:", round(pearson(y_a, preds_a), 4),
          " MSE:", round(mean_squared_error(y_a, preds_a), 4))

    # Fit on full train and evaluate on dev (if provided)
    model_v.fit(text, y_v)
    model_a.fit(text, y_a)

    if dev_path:
        df_dev = pd.read_csv(dev_path)
        x_dev = df_dev["text"].astype(str).values
        pv = model_v.predict(x_dev)
        pa = model_a.predict(x_dev)
        if "valence" in df_dev and "arousal" in df_dev:
            print("[Dev] Valence   Pearson:", round(pearson(df_dev["valence"].values, pv), 4),
                  " MSE:", round(mean_squared_error(df_dev["valence"].values, pv), 4))
            print("[Dev] Arousal   Pearson:", round(pearson(df_dev["arousal"].values, pa), 4),
                  " MSE:", round(mean_squared_error(df_dev["arousal"].values, pa), 4))
        if out_pred_path:
            out = df_dev[["id"]].copy()
            out["valence"] = pv
            out["arousal"] = pa
            out.to_csv(out_pred_path, index=False)
            print(f"[Saved] {out_pred_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="path to train.csv")
    ap.add_argument("--dev", required=False, help="path to dev.csv (labels optional)")
    ap.add_argument("--out", required=False, help="path to save predictions on dev/test")
    args = ap.parse_args()
    train_eval(args.train, args.dev, args.out)
```

Run:

```bash
python src/baseline.py --train data/train.csv --dev data/dev.csv --out preds_dev.csv
```

### 4) Submission / In-class Testing

* Expected prediction file format:

```
id,valence,arousal
001,7.61,6.12
002,2.05,7.42
...
```

* **TMU Workshop**: the instructor may replace `test.csv` with an **unseen variant**. Keep your inference script generic:

```bash
python src/baseline.py --train data/train.csv --dev data/test.csv --out preds_test.csv
```

(If `test.csv` has no labels, the script will simply generate predictions.)

---

## Stronger Baselines (Optional)

### A) Character-aware TF-IDF

Chinese benefit from character + bigram features. You can toggle:

* `ngram_range=(1,2)` for word-segmentation-based text
* Or tokenize by character before `TfidfVectorizer` via a custom preprocessor.

### B) Tree-based Models

Swap `Ridge` with `RandomForestRegressor` or `XGBRegressor` (for two heads: valence & arousal). Keep in mind overfitting on small data.

### C) Transformer Encoder Regression (Minimal)

* Use a Chinese pretrained model (e.g., `bert-base-chinese`) and regress two heads.
* Tips:

  * Freeze backbone + train small regression head for quick baseline.
  * Try PEFT/LoRA for resource-efficient fine-tuning.
  * Evaluate with Pearson + MSE; report both.

---

## Evaluation

### Metrics

We report **Pearson correlation** and **MSE** per dimension:

* Higher Pearson = better linear agreement with gold scores
* Lower MSE = better absolute error

You may also report **Spearman’s ρ** as a robustness check.

### Reproducibility

* Fix a seed (e.g., 42) for splits
* Log: model conf, vectorizer conf, random seed, timestamp, dataset hash

---

## License & Acknowledgment

* Please refer to the **official dataset website** for licensing, terms of use, and task rules: *ROCLING-SIGAI 2025 Shared Task* (see project homepage).
* This repository uses the dataset solely for **research** and **teaching**. For TMU workshop usage, the **test set may be replaced** for instructional evaluation.

---

## Citation

If you use this dataset or related baseline code, please cite the relevant works listed below.

## References

* Rafael A. Calvo, and Sunghwan Mac Kim. 2013. Emotions in text: dimensional and categorical models. *Computational Intelligence*, 29(3):527-543.
* Munmun De Choudhury, Scott Counts, and Michael Gamon. 2012. Not all moods are created equal! Exploring human emotional states in social media. In *Proc. of ICWSM-12*, pp. 66-73.
* Yu-Chih Deng, Cheng-Yu Tsai, Yih-Ru Wang, Sin-Horng Chen, and Lung-Hao Lee. 2022. Predicting Chinese Phrase-level Sentiment Intensity in Valence-Arousal Dimensions with Linguistic Dependency Features. *IEEE Access*, 10:126612-126620.
* Yu-Chih Deng, Yih-Ru Wang, Sin-Horng Chen, and Lung-Hao Lee. 2023. Towards Transformer Fusions for Chinese Sentiment Intensity Prediction in Valence-Arousal Dimensions. *IEEE Access*, 11:109974-109982.
* Steven Du and Xi Zhang. 2016. Aicyber’s system for IALP 2016 shared task: Character-enhanced word vectors and Boosted Neural Networks. In *Proc. of IALP-16*, pp. 161–163.
* Pranav Goel, Devang Kulshreshtha, Prayas Jain and Kaushal Kumar Shukla. 2017. Prayas at EmoInt 2017: An Ensemble of Deep Neural Architectures for Emotion Intensity Prediction in Tweets. In *Proc. of WASSA-17*, pp. 58–65.
* Sunghwan Mac Kim, Alessandro Valitutti, and Rafael A. Calvo. 2010. Evaluation of unsupervised emotion models to textual affect recognition. In *Proc. of NAACL HLT 2010 Workshop on Computational Approaches to Analysis and Generation of Emotion in Text*, pp. 62-70.
* Lung-Hao Lee, Jian-Hong Li, and Liang-Chih Yu. 2022. Chinese EmoBank: Building Valence-Arousal Resources for Dimensional Sentiment Analysis. *ACM TALLIP*, 21(4): Article 65.
* N. Malandrakis, A. Potamianos, E. Iosif, and S. Narayanan. 2013. Distributional semantic models for affective text analysis. *IEEE TASLP*, 21(11): 2379-2392.
* Myriam Munezero, Tuomo Kakkonen, and Calkin S. Montero. 2011. Towards automatic detection of antisocial behavior from texts. In *Proc. of SAAIP Workshop at IJCNLP-11*, pp. 20-27.
* Georgios Paltoglou, Mathias Theunis, Arvid Kappas, and Mike Thelwall. 2013. Predicting emotional responses to long informal text. *IEEE Trans. Affective Computing*, 4(1):106-115.
* Jie Ren and Jeffrey V. Nickerson. 2014. Online review systems: How emotional language drives sales. In *Proc. of AMCIS-14*.
* James A. Russell. 1980. A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6):1161.
* Wen-Li Wei, Chung-Hsien Wu, and Jen-Chun Lin. 2011. A regression approach to affective rating of Chinese words from ANEW. In *Proc. of ACII-11*, pp. 121-131.
* Liang-Chih Yu et al. 2016a. Building Chinese affective resources in valence-arousal dimensions. In *Proc. of NAACL/HLT-16*, pp. 540-545.
* Liang-Chih Yu et al. 2016b. Overview of the IALP 2016 shared task on dimensional sentiment analysis for Chinese words. In *Proc. of IALP-16*, pp. 156-160.
* Liang-Chih Yu, Jin Wang, and Kam-Fai Wong. 2017. IJCNLP-2017 Task 2: Dimensional sentiment analysis for Chinese phrases. In *Proc. of IJCNLP-17*, pp. 9-16.
* Liang-Chih Yu, Jin Wang, Bo Peng, and Chu-Ren Huang. 2021. ROCLING-2021 shared task: dimensional sentiment analysis for educational texts. In *Proc. of ROCLING-21*, pp. 385-388.
* Liang-Chih Yu et al. 2020. Pipelined neural networks for phrase-level sentiment intensity prediction. *IEEE TAC*, 11(3), 447-458.
* Jin Wang et al. 2016. Community-based weighted graph model for valence-arousal prediction of affective words. *IEEE/ACM TASLP*, 24(11):1957-1968.
* Jin Wang et al. 2020. Tree-structured regional CNN-LSTM model for dimensional sentiment analysis. *IEEE/ACM TASLP*, 28, 581–591.
* Chuhan Wu et al. 2017. THU NGN at IJCNLP-2017 Task 2: Dimensional sentiment analysis for Chinese phrases with deep LSTM. In *Proc. of IJCNLP-17*, pp. 42-52.
* Suyang Zhu, Shoushan Li and Guodong Zhou. 2019. Adversarial attention modeling for multi-dimensional emotion regression. In *Proc. of ACL-19*, pp. 471–480.


