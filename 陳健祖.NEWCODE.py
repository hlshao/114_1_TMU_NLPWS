# ===========================
# 匯入所需套件
# ===========================
import argparse               # 用於解析命令列參數（可讓此程式支援在終端執行時指定輸入輸出檔案）
import pandas as pd           # 資料處理套件，用來讀取 CSV 檔案並處理 DataFrame 結構
import numpy as np            # 科學運算套件，用於陣列運算與隨機種子設定
from sklearn.model_selection import KFold        # 用於建立 K 折交叉驗證 (K-Fold Cross Validation)
from sklearn.feature_extraction.text import TfidfVectorizer  # 將文字轉換為 TF-IDF 特徵
from sklearn.linear_model import Ridge           # 使用 Ridge Regression（L2 正則化的線性模型）
from sklearn.pipeline import make_pipeline       # 將 TF-IDF 與 Ridge 串成一個 pipeline（方便訓練與預測）
from sklearn.metrics import mean_squared_error   # 用於評估模型預測誤差 (MSE)
from scipy.stats import pearsonr                 # 用於計算皮爾森相關係數 (Pearson correlation)
import random                                   # Python 內建隨機模組（搭配 numpy 設定隨機種子）

# ===========================
# 設定隨機種子
# ===========================
def set_seed(seed=42):
    """
    為了實驗可重現性，固定所有隨機過程的種子。
    包含 numpy 與 Python 內建的 random。
    """
    np.random.seed(seed)
    random.seed(seed)

# ===========================
# 自訂皮爾森相關函數
# ===========================
def pearson(y_true, y_pred):
    """
    計算真實值 (y_true) 與預測值 (y_pred) 之間的皮爾森相關係數。
    若其中任一組資料變異數為 0（即無法計算相關），則回傳 0。
    """
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        # 若任一組全為同一數值，無法計算皮爾森，回傳 0.0
        return 0.0
    return pearsonr(y_true, y_pred).statistic  # 使用 scipy.stats.pearsonr 回傳 .statistic 即相關係數

# ===========================
# 訓練與評估主函數
# ===========================
def train_eval(train_path, dev_path=None, out_pred_path=None):
    """
    主函數：
    1. 讀取訓練資料 train.csv
    2. 將文字欄轉成 TF-IDF 特徵
    3. 以 Ridge Regression 訓練 valence / arousal 兩個子模型
    4. 進行 K 折交叉驗證 (若有提供 dev 集也可額外驗證)
    5. 可選擇將預測結果輸出到 CSV
    """
    # 固定隨機種子確保實驗可重現
    set_seed(42)
    
    # 讀取訓練資料
    df_tr = pd.read_csv(train_path)
    
    # 取出文字欄與目標變數
    text = df_tr["text"].astype(str).values     # 將文字資料轉為 numpy 陣列
    y_v = df_tr["valence"].values               # Valence（情緒正負向）
    y_a = df_tr["arousal"].values               # Arousal（情緒強度）
