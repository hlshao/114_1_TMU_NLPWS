# 匯入必要的套件
import argparse, pandas as pd, numpy as np                  # argparse 用於命令列參數，pandas 和 numpy 用於資料處理
from sklearn.model_selection import KFold                  # 用於建立 K 折交叉驗證
from sklearn.feature_extraction.text import TfidfVectorizer # 將文字轉換為 TF-IDF 特徵向量
from sklearn.linear_model import Ridge                     # Ridge Regression 模型
from sklearn.pipeline import make_pipeline                 # 建立前處理+模型流程
from sklearn.metrics import mean_squared_error             # 計算 MSE 評估模型
from scipy.stats import pearsonr                           # 計算皮爾森相關係數
import random                                              # 控制 Python 內建隨機性的種子

def set_seed(seed=42):
    """
    設定隨機種子以確保結果可重現
    - numpy 用於矩陣與數值運算
    - random 用於 Python 內建隨機相關功能
    """
    np.random.seed(seed)
    random.seed(seed)

def pearson(y_true, y_pred):
    """
    計算預測值與真實值之間的皮爾森相關係數
    - 若其中一組資料沒有變化（即唯一值少於 2），相關係數無意義，直接回傳 0
    - pearsonr() 回傳一個包含 (相關係數, p-value) 的物件，因此取 statistic
    """
    # 若真實值或預測值的唯一值太少，無法計算相關係數，避免錯誤
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    # 回傳 pearson correlation 的 statistic（即相關係數本身）
    return pearsonr(y_true, y_pred).statistic
