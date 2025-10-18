# argparse: 用來解析命令列參數，方便程式接受外部輸入
import argparse

# pandas: 常用的資料操作工具，用於讀取、處理表格資料
import pandas as pd

# numpy: 數值運算工具，常用於陣列運算、隨機數生成
import numpy as np

# KFold: K 折交叉驗證，用於模型驗證，將資料分成 K 份
from sklearn.model_selection import KFold

# TfidfVectorizer: 將文字轉換成 TF-IDF 特徵向量
from sklearn.feature_extraction.text import TfidfVectorizer

# Ridge: Ridge 回歸（帶 L2 正則化的線性回歸），適合預測連續值
from sklearn.linear_model import Ridge

# make_pipeline: 建立機器學習流程管線，方便將多個步驟串接（如向量化 + 回歸）
from sklearn.pipeline import make_pipeline

# mean_squared_error: 均方誤差，用於迴歸模型的評估
from sklearn.metrics import mean_squared_error

# pearsonr: 計算皮爾森相關係數，用於評估兩組連續數據的線性相關性
from scipy.stats import pearsonr

# random: Python 內建的隨機數工具，可用於設定隨機種子或生成隨機數
import random
