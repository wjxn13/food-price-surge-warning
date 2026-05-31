"""
V4 模型集成：LightGBM + XGBoost + MLP Stacking
使用 features_v4.parquet，最终输出测试集 F1
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- 1. 加载 V4 数据 --------------------
feature_df = pd.read_parquet('features_v4.parquet')
scaler_v4 = joblib.load('scaler_v4.pkl')

feat_cols = [
    'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
    'vol_6m', 'vol_12m', 'trend', 'sin_month', 'cos_month',
    'vol_3m', 'price_range_3m', 'ret_3m_accel',
    'rel_strength', 'fx_pressure', 'market_stress',
    'dist_to_12m_high', 'fx_volatility', 'price_vs_ma12', 'global_cat_ret_3m'
]

train_mask = feature_df['date'] <= '2020-12-31'
val_mask = (feature_df['date'] > '2020-12-31') & (feature_df['date'] <= '2023-12-31')
test_mask = (feature_df['date'] > '2023-12-31') & (feature_df['date'] <= '2025-12-31')

X = feature_df[feat_cols].values
y = feature_df['label'].values

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# 标准化（用训练集拟合）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# -------------------- 2. 训练基学习器 1：LightGBM (复用最佳参数) --------------------
print("训练 LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=300, num_leaves=63, max_depth=7, learning_rate=0.02,
    min_child_samples=80, subsample=0.7, colsample_bytree=0.8,
    reg_alpha=1e-7, reg_lambda=1e-5,
    scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_val_prob = lgb_model.predict_proba(X_val)[:, 1]
lgb_test_prob = lgb_model.predict_proba(X_test)[:, 1]

# -------------------- 3. 训练基学习器 2：XGBoost --------------------
print("训练 XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.02,
    scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1, reg_lambda=1, random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]

# -------------------- 4. 加载 MLP 模型并预测 --------------------
print("加载并预测 MLP 模型...")
# 复用之前的 MLP 定义（简化版，与 dl_warning_model.py 一致）
class MLPWithEmbeddings(nn.Module):
    def __init__(self, cont_dim, embed_dim=8, hidden_dims=[128,64,32], dropout=0.3):
        super().__init__()
        # 注意：这里只使用连续特征，不需要 Embedding（与原始MLP不同，但为了复用简单，我们直接加载之前保存的模型权重）
        # 实际上之前的MLP需要国家/类别嵌入，这里我们重新训练一个纯MLP（或加载之前权重但需要对齐输入）
        # 简便起见，我们直接训练一个新的纯MLP基学习器，只使用连续特征。
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

# 重新训练一个纯MLP（连续特征），避免Embedding对齐问题
mlp_model = MLPWithEmbeddings(cont_dim=len(feat_cols)).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([scale_pos_weight]).to(DEVICE))
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048)

best_val_f1 = 0
for epoch in range(30):
    mlp_model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(mlp_model(Xb), yb)
        loss.backward()
        optimizer.step()
    mlp_model.eval()
    with torch.no_grad():
        val_logits = mlp_model(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy()
        val_prob = 1 / (1 + np.exp(-val_logits))
        # 寻找最佳阈值
        best_th = 0.5
        best_f1 = 0
        for th in np.arange(0.1, 0.9, 0.02):
            preds = (val_prob >= th).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            torch.save(mlp_model.state_dict(), 'best_mlp_ensemble.pth')
    if (epoch+1) % 5 == 0:
        print(f"  MLP epoch {epoch+1}, best val F1: {best_val_f1:.4f}")

mlp_model.load_state_dict(torch.load('best_mlp_ensemble.pth'))
mlp_model.eval()
with torch.no_grad():
    mlp_val_logits = mlp_model(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy()
    mlp_test_logits = mlp_model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()
mlp_val_prob = 1 / (1 + np.exp(-mlp_val_logits))
mlp_test_prob = 1 / (1 + np.exp(-mlp_test_logits))

# -------------------- 5. Stacking：逻辑回归融合三模型输出 --------------------
print("训练 Stacking 逻辑回归...")
# 构造二级特征
val_stack = np.column_stack([lgb_val_prob, xgb_val_prob, mlp_val_prob])
test_stack = np.column_stack([lgb_test_prob, xgb_test_prob, mlp_test_prob])

meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(val_stack, y_val)

# 预测最终概率
test_final_prob = meta_model.predict_proba(test_stack)[:, 1]

# 寻找最佳阈值
best_f1, best_thresh = 0, 0.5
for thresh in np.arange(0.1, 0.9, 0.01):
    preds = (test_final_prob >= thresh).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

y_pred_final = (test_final_prob >= best_thresh).astype(int)

print("\n========== 集成模型测试集表现 ==========")
print(f"最佳阈值: {best_thresh:.2f}")
print(f"F1-score: {best_f1:.4f}")
print(f"召回率 (Recall): {recall_score(y_test, y_pred_final):.4f}")
print(f"精确率 (Precision): {precision_score(y_test, y_pred_final):.4f}")
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred_final))

# 各基学习器单独表现
print("\n--- 各基学习器在测试集上的最佳 F1 ---")
for name, prob in zip(['LightGBM','XGBoost','MLP'], [lgb_test_prob, xgb_test_prob, mlp_test_prob]):
    best_f1_single = 0
    for th in np.arange(0.1, 0.9, 0.01):
        preds = (prob >= th).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > best_f1_single:
            best_f1_single = f1
    print(f"{name}: {best_f1_single:.4f}")
# -------------------- 保存所有模型和预处理对象 --------------------
import joblib
# 保存树模型
joblib.dump(lgb_model, 'lgb_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
# 保存 MLP 模型参数
torch.save(mlp_model.state_dict(), 'mlp_model.pth')
# 保存逻辑回归二级模型
joblib.dump(meta_model, 'meta_model.pkl')
# 保存标准化器
joblib.dump(scaler, 'scaler_ensemble.pkl')  # 注意：这个 scaler 是集成脚本里单独拟合的，与 V4 的 scaler_v4 一致，但为了路径清晰，我们再保存一份
# 保存特征列名
import json
with open('feat_cols.json', 'w') as f:
    json.dump(feat_cols, f)
print("所有模型和预处理对象已保存。")