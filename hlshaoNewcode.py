# 匯入所需的套件
import argparse, pandas as pd, numpy as np                  # argparse：命令列參數解析；pandas：資料處理；numpy：數值運算
from sklearn.model_selection import KFold                   # KFold：交叉驗證的資料分割方法
from sklearn.feature_extraction.text import TfidfVectorizer # TfidfVectorizer：將文字轉換為TF-IDF向量
from sklearn.linear_model import Ridge                      # Ridge：Ridge（L2）迴歸模型
from sklearn.pipeline import make_pipeline                  # make_pipeline：建立多步驟的處理流程（如向量化→模型訓練）
from sklearn.metrics import mean_squared_error              # mean_squared_error：計算均方誤差（MSE）
from scipy.stats import pearsonr                            # pearsonr：皮爾森相關係數
import random                                                # random：用於隨機種子設定

# 設定隨機種子以確保實驗結果可重現
def set_seed(seed=42):
    np.random.seed(seed)     # 設定 NumPy 隨機數種子
    random.seed(seed)        # 設定 Python random 模組的隨機數種子

# 自訂皮爾森相關係數函式，用於評估模型預測與真值的相關性
def pearson(y_true, y_pred):
    # 若真實值或預測值中的唯一值少於兩個（例如全為常數），相關係數無法計算，直接回傳 0
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    # 否則使用 scipy 的 pearsonr 計算相關係數，回傳 statistic 屬性（即 r 值）
    return pearsonr(y_true, y_pred).statistic
