"""
构建价格时间序列缓存（供 Streamlit 使用）
"""
import pandas as pd
from pathlib import Path
import time

DATA_DIR = Path("D:/.kaggle/食品价格/食品价格")
OUTPUT_PATH = "price_series.parquet"

def build_price_cache():
    print("开始提取价格序列...")
    csv_files = sorted(DATA_DIR.glob("wfp_food_prices_global_*.csv"))
    df_list = []
    for f in csv_files:
        tmp = pd.read_csv(f, usecols=['date', 'market_id', 'commodity_id', 'usdprice', 'priceflag', 'pricetype'])
        tmp['date'] = pd.to_datetime(tmp['date'])
        tmp = tmp[(tmp['priceflag'] == 'actual') & (tmp['pricetype'] == 'Retail')]
        tmp['series_id'] = tmp['market_id'].astype(str) + '_' + tmp['commodity_id'].astype(str)
        df_list.append(tmp[['series_id', 'date', 'usdprice']])
    df = pd.concat(df_list, ignore_index=True)
    # 按月聚合，取均值
    df['year_month'] = df['date'].dt.to_period('M')
    monthly = df.groupby(['series_id', 'year_month']).agg(
        usdprice=('usdprice', 'mean'),
        date=('date', 'first')
    ).reset_index()
    monthly['date'] = monthly['year_month'].dt.to_timestamp()
    monthly = monthly[['series_id', 'date', 'usdprice']]
    monthly.to_parquet(OUTPUT_PATH, index=False)
    print(f"价格缓存已保存至 {OUTPUT_PATH}，共 {len(monthly)} 条记录")

if __name__ == "__main__":
    t0 = time.time()
    build_price_cache()
    print(f"耗时: {time.time()-t0:.1f} 秒")