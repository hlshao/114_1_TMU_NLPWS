"""
test 2025/10/18
TF-IDF + Ridge regression for Valence/Arousal prediction
-------------------------------------------------------
功能概述：
- 從訓練檔(train.csv)讀取 text、valence、arousal 欄位
- 以 TF-IDF (1–2-gram) 表徵文字，分別訓練兩個 Ridge 迴歸模型（valence 與 arousal）
- 以 5-fold K-Fold 做 out-of-fold 預測估計 CV 表現：Pearson 相關與 MSE
- 以所有訓練資料重新擬合模型
- 若提供 dev.csv：輸出開發集/測試集預測，並在有標籤時同樣回報 Pearson 與 MSE

使用方式：
python script.py --train path/to/train.csv [--dev path/to/dev.csv] [--out path/to/save.csv]

輸入需求：
- train.csv 需含欄位：["text", "valence", "arousal"]
- dev.csv 需含欄位：["id", "text"]；如同時含 ["valence","arousal"] 則會計算指標
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import random
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    設定隨機種子以提升重現性（reproducibility）。
    這裡控制 numpy 與 Python random；(本程式未使用 torch 等其他框架)
    """
    np.random.seed(seed)
    random.seed(seed)


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    安全計算 Pearson 相關係數：
    - 若 y_true 或 y_pred 無變異（unique 值 < 2），pearsonr 會報錯，因此先行返回 0.0。
    - 回傳 scipy.stats.pearsonr 的 statistic（皮爾森 r 值）。
    """
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return pearsonr(y_true, y_pred).statistic


def train_eval(
    train_path: str,
    dev_path: Optional[str] = None,
    out_pred_path: Optional[str] = None
) -> None:
    """
    主流程：
    1) 讀取訓練資料並抽取文字與兩個連續標籤（valence、arousal）
    2) 建立兩條等結構之 pipeline：TF-IDF -> Ridge（對 valence / arousal 各一）
    3) 以 KFold 進行 5 折 CV，產生 out-of-fold 預測，估算 CV 的 Pearson 與 MSE
    4) 使用全部訓練資料重訓模型，以便後續對 dev/test 做推論
    5) 若提供 dev.csv，則進行推論；若 dev 也含標籤，則同樣回報指標；若提供 --out，將輸出 id,valence,arousal
    """
    set_seed(42)

    # 讀取訓練資料；預期必含 text、valence、arousal 欄位
    df_tr = pd.read_csv(train_path)
    text = df_tr["text"].astype(str).values
    y_v = df_tr["valence"].values
    y_a = df_tr["arousal"].values

    # 為 valence 建立文字特徵 + Ridge 迴歸的 pipeline
    # - ngram_range=(1,2)：使用 uni-gram + bi-gram
    # - min_df=2：過濾極罕見詞
    # - max_features=200000：限制詞彙量上限，避免記憶體爆炸
    # 備註：Ridge 的 random_state 僅在 solver='sag' 或 'saga' 時有效；sklearn 預設 solver='auto' 通常不使用隨機性
    model_v = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000),
        Ridge(alpha=2.0, random_state=42)
    )

    # 為 arousal 建立相同配置的 pipeline（與 valence 分開訓練，避免共用殘差）
    model_a = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000),
        Ridge(alpha=2.0, random_state=42)
    )

    # 5 折 K-Fold（隨機洗牌 + 固定種子，以確保可重現）
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 用來收集 out-of-fold 預測（與原始 y 對齊，估計 CV 表現用）
    preds_v = np.zeros(len(df_tr))
    preds_a = np.zeros(len(df_tr))

    # 交叉驗證迴圈：每折皆以訓練子集擬合，再對該折的驗證索引做推論
    for tr_idx, va_idx in kf.split(text):
        # 擬合各自目標的模型
        model_v.fit(text[tr_idx], y_v[tr_idx])
        model_a.fit(text[tr_idx], y_a[tr_idx])

        # 產生對應折的驗證預測
        preds_v[va_idx] = model_v.predict(text[va_idx])
        preds_a[va_idx] = model_a.predict(text[va_idx])

    # 匯報 CV（out-of-fold）表現：Pearson 與均方誤差 MSE
    print(
        "[Train-CV] Valence  Pearson:",
        round(pearson(y_v, preds_v), 4),
        " MSE:",
        round(mean_squared_error(y_v, preds_v), 4)
    )
    print(
        "[Train-CV] Arousal  Pearson:",
        round(pearson(y_a, preds_a), 4),
        " MSE:",
        round(mean_squared_error(y_a, preds_a), 4)
    )

    # 以全訓練資料重訓模型（作最終推論用）
    model_v.fit(text, y_v)
    model_a.fit(text, y_a)

    # 若提供 dev/test 檔，則推論並（在有標籤時）計算指標
    if dev_path:
        df_dev = pd.read_csv(dev_path)
        x_dev = df_dev["text"].astype(str).values

        # 生成預測
        pv = model_v.predict(x_dev)
        pa = model_a.predict(x_dev)

        # 若 dev 同時含標籤，則報告性能表現
        # DataFrame 的成員測試 "col in df_dev" 會檢查欄名（等同於 "col in df_dev.columns"）
        if "valence" in df_dev and "arousal" in df_dev:
            print(
                "[Dev] Valence   Pearson:",
                round(pearson(df_dev["valence"].values, pv), 4),
                " MSE:",
                round(mean_squared_error(df_dev["valence"].values, pv), 4)
            )
            print(
                "[Dev] Arousal   Pearson:",
                round(pearson(df_dev["arousal"].values, pa), 4),
                " MSE:",
                round(mean_squared_error(df_dev["arousal"].values, pa), 4)
            )

        # 若指定輸出路徑，寫出提交檔（常用於 leaderboard/test）
        if out_pred_path:
            out = df_dev[["id"]].copy()  # 需要 dev.csv 內有 id 欄位
            out["valence"] = pv
            out["arousal"] = pa
            out.to_csv(out_pred_path, index=False)
            print(f"[Saved] {out_pred_path}")


if __name__ == "__main__":
    # 參數解析：--train 必填；--dev 與 --out 可選
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="path to train.csv")
    ap.add_argument("--dev", required=False, help="path to dev.csv (labels optional)")
    ap.add_argument("--out", required=False, help="path to save predictions on dev/test")
    args = ap.parse_args()

    # 執行主流程
    train_eval(args.train, args.dev, args.out)

