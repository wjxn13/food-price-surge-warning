"""
食品价格飙升预警部署演示 (修正版 v2)
生成 HTML 报告，包含 Top 20 高风险预警和特征重要性
"""
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- 1. 加载模型和预处理 --------------------
with open('feat_cols.json', 'r') as f:
    feat_cols = json.load(f)

scaler = joblib.load('scaler_ensemble.pkl')
lgb_model = joblib.load('lgb_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

class MLPModel(nn.Module):
    def __init__(self, cont_dim, hidden_dims=[128,64,32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = cont_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(1)

mlp_model = MLPModel(cont_dim=len(feat_cols)).to(DEVICE)
mlp_model.load_state_dict(torch.load('mlp_model.pth', map_location=DEVICE))
mlp_model.eval()

meta_model = joblib.load('meta_model.pkl')

# -------------------- 2. 加载特征数据并筛选当前时间点 --------------------
feature_df = pd.read_parquet('features_v4.parquet')
target_date = pd.Timestamp('2026-05-31')

current_data = feature_df[feature_df['date'] <= target_date].sort_values('date').groupby('series_id').last().reset_index()
print(f"当前可预警的序列数: {len(current_data)}")

X_current = current_data[feat_cols].values
X_current_scaled = scaler.transform(X_current)

# -------------------- 3. 预测概率 --------------------
lgb_prob = lgb_model.predict_proba(X_current_scaled)[:, 1]
xgb_prob = xgb_model.predict_proba(X_current_scaled)[:, 1]
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_current_scaled).to(DEVICE)
    mlp_logits = mlp_model(X_tensor).cpu().numpy()
mlp_prob = 1 / (1 + np.exp(-mlp_logits))

stacked = np.column_stack([lgb_prob, xgb_prob, mlp_prob])
final_prob = meta_model.predict_proba(stacked)[:, 1]

current_data['warning_prob'] = final_prob

# -------------------- 4. 获取静态映射和当前价格（从原始CSV快速提取）--------------------
DATA_DIR = Path("D:/.kaggle/食品价格/食品价格")
csv_files = sorted(DATA_DIR.glob("wfp_food_prices_global_*.csv"))

all_static = []
price_data = []
for f in csv_files:
    # 只读取必要的列
    tmp = pd.read_csv(f, usecols=['countryiso3', 'category', 'commodity', 'market_id', 'commodity_id', 'usdprice', 'date'], parse_dates=['date'])
    tmp['series_id'] = tmp['market_id'].astype(str) + '_' + tmp['commodity_id'].astype(str)
    all_static.append(tmp[['series_id', 'countryiso3', 'category', 'commodity']].drop_duplicates())
    price_data.append(tmp[['series_id', 'date', 'usdprice']])

static_df = pd.concat(all_static).drop_duplicates(subset='series_id')
price_df = pd.concat(price_data)

# 获取每个 series_id 在 target_date 之前的最新价格
price_recent = price_df[price_df['date'] <= target_date].sort_values('date').groupby('series_id').last().reset_index()
price_recent = price_recent[['series_id', 'usdprice']]

print(f"静态映射条目: {len(static_df)}")

# 合并到 Top20 数据
top20 = current_data.nlargest(20, 'warning_prob')
top20 = top20.merge(static_df, on='series_id', how='left')
top20 = top20.merge(price_recent, on='series_id', how='left')

# 展示列
top20_display = top20[['series_id', 'countryiso3', 'category', 'commodity', 'usdprice', 'warning_prob']].copy()
top20_display['warning_prob'] = top20_display['warning_prob'].round(4)
top20_display = top20_display.rename(columns={
    'countryiso3': '国家', 'category': '类别', 'commodity': '商品',
    'usdprice': '当前 USD 价格', 'warning_prob': '飙升概率'
})
print("Top 20 高风险预警:")
print(top20_display.to_string(index=False))

# -------------------- 5. 特征重要性图 --------------------
importance_df = pd.DataFrame({
    'feature': feat_cols,
    'importance': lgb_model.booster_.feature_importance(importance_type='split')
}).sort_values('importance', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['feature'][::-1], importance_df['importance'][::-1], color='steelblue')
ax.set_xlabel('Split Importance')
ax.set_title('LightGBM Top 10 Feature Importance')
plt.tight_layout()
feat_imp_path = 'feature_importance.png'
plt.savefig(feat_imp_path)
plt.close()

# -------------------- 6. 生成 HTML 报告 --------------------
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>食品价格飙升预警报告 ({target_date.date()})</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .high {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🌍 全球食品价格异常飙升预警报告</h1>
    <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>Top 20 高风险预警 (未来3个月涨幅 >15% 概率)</h2>
    {top20_display.to_html(index=False, classes='warning-table')}
    <h2>特征重要性 (LightGBM)</h2>
    <img src="{feat_imp_path}" alt="Feature Importance" style="max-width:100%;">
    <p>模型: LightGBM + XGBoost + MLP 集成 (Stacking)</p>
</body>
</html>
"""

report_path = f'warning_report_{target_date.date()}.html'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n报告已生成: {report_path}")
print(f"特征重要性图: {feat_imp_path}")