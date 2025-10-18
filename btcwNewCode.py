# 導入所需的套件
import argparse                     # 處理命令列參數
import pandas as pd                # 資料處理套件，常用於讀取與操作資料表
import numpy as np                 # 數值運算套件
from sklearn.model_selection import KFold               # 用於交叉驗證的K折切分方法
from sklearn.feature_extraction.text import TfidfVectorizer  # 將文字轉換成TF-IDF向量的工具
from sklearn.linear_model import Ridge                 # L2正則化的線性回歸模型（Ridge Regression）
from sklearn.pipeline import make_pipeline             # 用來串接多個前處理與模型步驟
from sklearn.metrics import mean_squared_error         # 評估預測結果的均方誤差（MSE）
from scipy.stats import pearsonr                       # 計算皮爾森相關係數
import random                                           # Python內建的隨機模組

# 設定隨機種子以確保實驗結果可重現
def set_seed(seed=42):
    np.random.seed(seed)       # 設定 NumPy 的隨機種子
    random.seed(seed)          # 設定 Python random 模組的隨機種子

# 自訂皮爾森相關係數的計算函式，避免無效輸入導致錯誤
def pearson(y_true, y_pred):
    # 若真實值或預測值中所有值都一樣（沒有變異），則皮爾森係數無法計算，回傳0.0
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    # 否則，計算並回傳皮爾森相關係數的統計量
    return pearsonr(y_true, y_pred).statistic
