# 設定隨機種子，確保結果可重現
def set_seed(seed=42):
    np.random.seed(seed)      # 設定 NumPy 的隨機種子
    random.seed(seed)         # 設定 Python 內建 random 模組的隨機種子

# 計算皮爾森相關係數（Pearson correlation coefficient）
def pearson(y_true, y_pred):
    # 若輸入值的變化太少（例如所有值都一樣），相關係數無法計算，則回傳 0
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    # 使用 scipy.stats.pearsonr 計算相關係數，並回傳結果中的 statistic 欄位
    return pearsonr(y_true, y_pred).statistic

# 訓練與評估模型的主函式
def train_eval(train_path, dev_path=None, out_pred_path=None):
    set_seed(42)  # 固定隨機種子，確保實驗結果一致

    # 讀取訓練資料集
    df_tr = pd.read_csv(train_path)

    # 取出文字欄位，確保轉為字串型態
    text = df_tr["text"].astype(str).values

    # 取出標籤欄位（情緒價值 valence 與喚醒度 arousal）
    y_v = df_tr["valence"].values
    y_a = df_tr["arousal"].values

    # 建立 valence 模型：
    # 先使用 TF-IDF 向量化文本，再以 Ridge Regression（嶺迴歸）訓練模型
    model_v = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000),  # 使用 1-2 元語法、最低詞頻 2、最多 20 萬特徵
        Ridge(alpha=2.0, random_state=42)  # Ridge 模型，正則化強度 alpha=2.0
    )

    # 建立 arousal 模型，結構與 valence 模型相同
    model_a = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000),
        Ridge(alpha=2.0, random_state=42)
    )
