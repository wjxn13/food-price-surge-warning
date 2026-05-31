"""
V4 增强数据预处理：涨幅>15% + 额外短期波动特征 (加速版)
新增特征: vol_3m, price_range_3m, ret_3m_accel
"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_DIR = Path("D:/.kaggle/食品价格/食品价格")
# 初始值会被自动更新，仅占位
CURRENT_DATE = pd.Timestamp('2026-05-31')
LOOKAHEAD = 3
SURGE_THRESHOLD = 0.15
WINDOW_SIZE = 12
N_JOBS = -1

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# -------------------- 1. 数据加载 --------------------
def load_and_clean_data():
    global CURRENT_DATE          # 声明为全局变量
    log("开始加载数据...")
    csv_files = sorted(DATA_DIR.glob("wfp_food_prices_global_*.csv"))
    df_list = []
    for f in csv_files:
        year = int(f.stem.split("_")[-1])
        tmp = pd.read_csv(f, parse_dates=['date'])
        tmp['year'] = year
        df_list.append(tmp)
    df = pd.concat(df_list, ignore_index=True)
    df = df[(df['priceflag'] == 'actual') & (df['pricetype'] == 'Retail')]
    df = df[df['date'] <= CURRENT_DATE]
    df = df.dropna(subset=['market_id', 'commodity_id', 'usdprice', 'price'])
    df['series_id'] = df['market_id'].astype(str) + '_' + df['commodity_id'].astype(str)
    log(f"数据加载完成，序列数: {df['series_id'].nunique()}")

    # 动态更新 CURRENT_DATE 为数据中实际的最大日期
    CURRENT_DATE = df['date'].max()
    log(f"当前数据截止日期已更新为: {CURRENT_DATE.date()}")

    return df

# -------------------- 2. 月度面板 --------------------
def build_monthly_panel(df):
    log("构建月度面板...")
    df['year_month'] = df['date'].dt.to_period('M')
    monthly = df.groupby(['series_id', 'year_month']).agg(
        usdprice=('usdprice', 'mean'),
        price=('price', 'mean'),
        date=('date', 'first')
    ).reset_index()
    monthly['date'] = monthly['year_month'].dt.to_timestamp()

    monthly = monthly.set_index('date').groupby('series_id').resample('MS').mean()
    monthly[['usdprice', 'price']] = monthly.groupby('series_id')[['usdprice', 'price']].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both', limit=3)
    )
    monthly[['usdprice', 'price']] = monthly.groupby('series_id')[['usdprice', 'price']].transform(
        lambda x: x.fillna(x.median())
    )
    monthly = monthly.reset_index()

    seq_lengths = monthly.groupby('series_id').size()
    valid = seq_lengths[seq_lengths >= WINDOW_SIZE + LOOKAHEAD].index
    monthly = monthly[monthly['series_id'].isin(valid)]

    static = df.groupby('series_id')[['countryiso3', 'admin1', 'market', 'category']].first().reset_index()
    monthly = monthly.merge(static, on='series_id', how='left')
    log(f"面板序列数: {monthly['series_id'].nunique()}")
    return monthly

# -------------------- 3. 新标签（涨幅>15%）--------------------
def create_labels_new(grp):
    grp = grp.sort_values('date')
    prices = grp['usdprice'].values
    n = len(prices)
    labels = np.zeros(n)
    for i in range(n - LOOKAHEAD):
        current = prices[i]
        future = prices[i + LOOKAHEAD]
        if current > 0 and (future - current) / current > SURGE_THRESHOLD:
            labels[i] = 1
    grp['label'] = labels
    return grp

def add_labels_parallel(monthly):
    log("并行构造标签（涨幅>15%）...")
    groups = [grp for _, grp in monthly.groupby('series_id')]
    results = Parallel(n_jobs=N_JOBS, verbose=10, backend='loky')(
        delayed(create_labels_new)(grp.copy()) for grp in groups
    )
    monthly = pd.concat(results, ignore_index=True)
    monthly = monthly.dropna(subset=['label'])
    monthly['label'] = monthly['label'].astype(int)
    log(f"标签完成，总样本: {len(monthly)}，正样本比例: {monthly['label'].mean():.4f}")
    return monthly

# -------------------- 4. 基础特征（并行）--------------------
def build_features_enhanced(grp):
    grp = grp.sort_values('date')
    prices = grp['usdprice'].values
    n = len(prices)
    feat_list = []
    for i in range(WINDOW_SIZE, n - LOOKAHEAD):
        past = prices[i - WINDOW_SIZE : i]
        ret_1 = (prices[i-1] - prices[i-2]) / prices[i-2] if i>=2 and prices[i-2]>0 else 0
        ret_3 = (prices[i-1] - prices[i-4]) / prices[i-4] if i>=4 and prices[i-4]>0 else 0
        ret_6 = (prices[i-1] - prices[i-7]) / prices[i-7] if i>=7 and prices[i-7]>0 else 0
        ret_12 = (prices[i-1] - prices[i-13]) / prices[i-13] if i>=13 and prices[i-13]>0 else 0
        vol_6 = np.std(past[-6:]) / (np.mean(past[-6:]) + 1e-6)
        vol_12 = np.std(past) / (np.mean(past) + 1e-6)
        slope = np.polyfit(np.arange(len(past)), past, 1)[0] if len(past) > 1 else 0
        month = pd.Timestamp(grp['date'].values[i]).month
        sin_m = np.sin(2 * np.pi * month / 12)
        cos_m = np.cos(2 * np.pi * month / 12)

        short_past = past[-3:] if len(past) >= 3 else past
        vol_3 = np.std(short_past) / (np.mean(short_past) + 1e-6) if len(short_past) > 1 else 0
        if len(short_past) > 1:
            range_3 = (np.max(short_past) - np.min(short_past)) / (np.mean(short_past) + 1e-6)
        else:
            range_3 = 0
        if i >= 4:
            prev_ret_3 = (prices[i-2] - prices[i-5]) / prices[i-5] if i>=5 and prices[i-5]>0 else 0
            ret_3m_accel = ret_3 - prev_ret_3
        else:
            ret_3m_accel = 0

        feat = [ret_1, ret_3, ret_6, ret_12, vol_6, vol_12, slope, sin_m, cos_m,
                vol_3, range_3, ret_3m_accel]
        feat_list.append({
            'date': grp['date'].values[i],
            'label': grp['label'].values[i],
            'series_id': grp['series_id'].values[i],
            'features': feat
        })
    return pd.DataFrame(feat_list)

def build_features_parallel(monthly):
    log("并行构建基础特征（含短期新增）...")
    groups = [grp for _, grp in monthly.groupby('series_id')]
    results = Parallel(n_jobs=N_JOBS, verbose=10, backend='loky')(
        delayed(build_features_enhanced)(grp.copy()) for grp in groups
    )
    feature_df = pd.concat(results, ignore_index=True)
    base_cols = ['ret_1m','ret_3m','ret_6m','ret_12m','vol_6m','vol_12m','trend','sin_month','cos_month',
                 'vol_3m', 'price_range_3m', 'ret_3m_accel']
    feature_df[base_cols] = pd.DataFrame(feature_df['features'].tolist(), index=feature_df.index)
    feature_df.drop(columns=['features'], inplace=True)
    log(f"基础特征样本: {len(feature_df)}")
    return feature_df, base_cols

# -------------------- 5. 全局增强特征（同 V3）--------------------
def add_global_features(feature_df, monthly):
    log("计算全局增强特征...")
    static = monthly[['series_id', 'countryiso3', 'category', 'market']].drop_duplicates()
    feature_df = feature_df.merge(static, on='series_id', how='left')
    feature_df['year_month'] = feature_df['date'].dt.to_period('M')

    cat_avg = feature_df.groupby(['countryiso3', 'category', 'year_month'])['ret_3m'].transform('mean')
    feature_df['rel_strength'] = feature_df['ret_3m'] - cat_avg

    monthly['local_ret_3'] = monthly.groupby('series_id')['price'].pct_change(3)
    monthly_key = monthly[['series_id', 'date', 'local_ret_3']].dropna()
    feature_df = feature_df.merge(monthly_key, on=['series_id', 'date'], how='left')
    feature_df['fx_pressure'] = feature_df['ret_3m'] - feature_df['local_ret_3']
    feature_df.drop(columns=['local_ret_3'], inplace=True)

    feature_df['up_flag'] = (feature_df['ret_3m'] > 0).astype(int)
    market_up = feature_df.groupby(['countryiso3', 'market', 'year_month'])['up_flag'].transform('mean')
    feature_df['market_stress'] = market_up
    feature_df.drop(columns=['up_flag'], inplace=True)

    monthly['price_12m_high'] = monthly.groupby('series_id')['usdprice'].transform(lambda x: x.rolling(12, min_periods=1).max())
    monthly_key2 = monthly[['series_id', 'date', 'usdprice', 'price_12m_high']].dropna()
    feature_df = feature_df.merge(monthly_key2, on=['series_id', 'date'], how='left')
    feature_df['dist_to_12m_high'] = (feature_df['price_12m_high'] - feature_df['usdprice']) / (feature_df['usdprice'] + 1e-6)
    feature_df.drop(columns=['usdprice', 'price_12m_high'], inplace=True)

    monthly['fx_rate'] = monthly['usdprice'] / (monthly['price'] + 1e-6)
    monthly['fx_vol'] = monthly.groupby('series_id')['fx_rate'].transform(lambda x: x.rolling(6, min_periods=2).std())
    monthly_key3 = monthly[['series_id', 'date', 'fx_vol']].dropna()
    feature_df = feature_df.merge(monthly_key3, on=['series_id', 'date'], how='left')
    feature_df['fx_volatility'] = feature_df['fx_vol'].fillna(0)
    feature_df.drop(columns=['fx_vol'], inplace=True)

    monthly['ma12'] = monthly.groupby('series_id')['usdprice'].transform(lambda x: x.rolling(12, min_periods=1).mean())
    monthly_key4 = monthly[['series_id', 'date', 'usdprice', 'ma12']].dropna()
    feature_df = feature_df.merge(monthly_key4, on=['series_id', 'date'], how='left')
    feature_df['price_vs_ma12'] = (feature_df['usdprice'] - feature_df['ma12']) / (feature_df['ma12'] + 1e-6)
    feature_df.drop(columns=['usdprice', 'ma12'], inplace=True)

    global_cat = feature_df.groupby(['category', 'year_month'])['ret_3m'].transform('mean')
    feature_df['global_cat_ret_3m'] = global_cat

    feature_df.drop(columns=['year_month', 'countryiso3', 'market', 'category'], inplace=True)
    log("全局特征添加完成")
    return feature_df

# -------------------- 6. 保存缓存 --------------------
def save_cache(feature_df, feat_cols):
    log("标准化并保存缓存...")
    train_mask = feature_df['date'] <= '2020-12-31'
    X = feature_df[feat_cols].values
    scaler = StandardScaler()
    scaler.fit(X[train_mask])
    feature_df.to_parquet('features_v4.parquet', index=False)
    joblib.dump(scaler, 'scaler_v4.pkl')
    log(f"缓存已保存：features_v4.parquet 和 scaler_v4.pkl，特征数: {len(feat_cols)}")

# ==================== 主流程 ====================
if __name__ == "__main__":
    log("🔥 V4 增强特征缓存生成启动 (阈值 15% + 短期特征) 🔥")
    t0 = time.time()
    df = load_and_clean_data()
    monthly = build_monthly_panel(df)
    monthly = add_labels_parallel(monthly)
    feature_df, base_cols = build_features_parallel(monthly)
    feature_df = add_global_features(feature_df, monthly)

    global_cols = ['rel_strength', 'fx_pressure', 'market_stress',
                   'dist_to_12m_high', 'fx_volatility', 'price_vs_ma12', 'global_cat_ret_3m']
    feat_cols = base_cols + global_cols
    feature_df = feature_df.dropna(subset=feat_cols)
    log(f"最终特征样本数: {len(feature_df)}，正样本比例: {feature_df['label'].mean():.4f}")
    save_cache(feature_df, feat_cols)
    log(f"✅ 全部完成，总耗时 {time.time()-t0:.1f} 秒")