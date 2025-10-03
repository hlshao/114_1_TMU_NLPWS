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

    model_v = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000),
        Ridge(alpha=2.0, random_state=42)
    )
    model_a = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000),
        Ridge(alpha=2.0, random_state=42)
    )

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
