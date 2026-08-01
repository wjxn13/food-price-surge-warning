"""
轻量补列脚本：在不重跑 build_cache 的前提下，
按 series_id 从原始 WFP CSV 映射 countryiso3 / category，
整数编码为 country_code / cat_code，写回 features_v4.parquet。
避免 build_monthly_panel 的 groupby.resample 在 6.5 万序列上 OOM。
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

DATA_DIR = Path("D:/.kaggle/食品价格/食品价格")
SRC = "features_v4.parquet"

print("构建 series_id -> (countryiso3, category) 映射 ...")
series_map = {}
for f in sorted(DATA_DIR.glob("wfp_food_prices_global_*.csv")):
    tmp = pd.read_csv(
        f,
        usecols=["market_id", "commodity_id", "countryiso3", "category"],
    )
    tmp["series_id"] = tmp["market_id"].astype(str) + "_" + tmp["commodity_id"].astype(str)
    m = tmp.groupby("series_id")[["countryiso3", "category"]].first()
    for sid, row in m.iterrows():
        if sid not in series_map:
            series_map[sid] = (row["countryiso3"], row["category"])
print(f"  映射覆盖序列数: {len(series_map)}")

print("读取现有 features_v4.parquet ...")
df = pd.read_parquet(SRC)
print(f"  行数: {len(df)}  列: {list(df.columns)}")

df["countryiso3"] = df["series_id"].map(lambda s: series_map.get(s, (None, None))[0])
df["category"] = df["series_id"].map(lambda s: series_map.get(s, (None, None))[1])

country_map = {c: i for i, c in enumerate(sorted(df["countryiso3"].dropna().unique()))}
cat_map = {c: i for i, c in enumerate(sorted(df["category"].dropna().unique()))}
df["country_code"] = df["countryiso3"].map(country_map).fillna(-1).astype(int)
df["cat_code"] = df["category"].map(cat_map).fillna(-1).astype(int)

print(f"  国家数: {len(country_map)}  品类数: {len(cat_map)}")
print(f"  未映射行数: country={int((df['country_code']==-1).sum())} cat={int((df['cat_code']==-1).sum())}")

joblib.dump({"country": country_map, "cat": cat_map}, "cat_maps.pkl")
df.to_parquet(SRC, index=False)
print("已写回 features_v4.parquet 并保存 cat_maps.pkl")
print("最终列:", list(df.columns))
