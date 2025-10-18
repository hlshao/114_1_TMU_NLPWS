# 匯入必要套件
import argparse                 # 用於命令列參數解析
import pandas as pd             # 資料處理與分析
import numpy as np              # 數值運算
from sklearn.model_selection import KFold              # K 折交叉驗證工具
from sklearn.feature_extraction.text import TfidfVectorizer  # 將文字轉換為 TF-IDF 向量
from sklearn.linear_model import Ridge                 # 岭迴歸模型（線性迴歸的一種，具正則化）
from sklearn.pipeline import make_pipeline             # 用於建立機器學習流程（可串接前處理與模型）
from sklearn.metrics import mean_squared_error         # 均方誤差 (MSE) 評估指標
from scipy.stats import pearsonr                       # 計算皮爾森相關係數 (Pearson correlation)
import random                                           # 產生隨機數

# ------------------------------
# 固定隨機種子，使結果可重現
# ------------------------------
def set_seed(seed=42):
    np.random.seed(seed)   # 固定 numpy 的隨機種子
    random.seed(seed)      # 固定 Python random 模組的隨機種子

# ------------------------------
# 自訂皮爾森相關係數計算函式
# ------------------------------
def pearson(y_true, y_pred):
    # 若 y_true 或 y_pred 的值全相同（變異數為 0），皮爾森相關無法計算，直接回傳 0
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    # 否則使用 scipy.stats.pearsonr 計算皮爾森相關係數
    return pearsonr(y_true, y_pred).statistic
