import argparse, pandas as pd, numpy as np 
# 引入 argparse（命令列參數解析）、pandas（資料處理）、numpy（數值計算）

from sklearn.model_selection import KFold
# 引入 K 折交叉驗證 (K-Fold Cross Validation)

from sklearn.feature_extraction.text import TfidfVectorizer
# 引入 TF-IDF 向量化工具，用來把文字轉成數值特徵

from sklearn.linear_model import Ridge
# 引入 Ridge regression（L2 正則化的線性迴歸）

from sklearn.pipeline import make_pipeline
# 引入管線工具 (pipeline)，方便把多個步驟串接起來

from sklearn.metrics import mean_squared_error
# 引入 MSE (均方誤差)，常用來評估迴歸模型效能

from scipy.stats import pearsonr
# 引入皮爾森相關係數 (Pearson correlation)，衡量兩組數值的相關性

import random
# 引入 Python 內建的隨機數工具


def set_seed(seed=42):
    # 設定隨機種子，確保實驗結果可重現
    np.random.seed(seed)
    random.seed(seed)


def pearson(y_true, y_pred):
    # 計算 y_true 與 y_pred 之間的皮爾森相關係數
    # 若兩組資料變異數過小（唯一值 < 2），無法計算，則回傳 0.0
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return pearsonr(y_true, y_pred).statistic
    # pearsonr 會回傳一個物件，包含相關係數與 p 值
    # 這裡取 .statistic，即相關係數本身
