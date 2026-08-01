"""
V5 模型集成（无泄漏 + MLP 类别嵌入）：
  - 基学习器：LightGBM + XGBoost + MLP(含 country/cat Embedding)
  - 关键修复①：用 TimeSeriesSplit 在 train 内生成「折外(OOF)概率」训练二级逻辑回归，
              val 仅用于选阈值，test 为唯一评估集，全程不参与任何训练/选参（消除 V4 的数据穿越）。
  - 关键修复②：MLPWithEmbeddings 真正实现 country_code / cat_code 的 Embedding 分支，
              与连续特征拼接，恢复 V4 中被注释掉的类别交叉能力。
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import json, time, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SPLITS = 4          # train 内时序折数（用于 OOF）
MLP_EPOCHS = 15
MLP_BATCH = 4096

# -------------------- 1. 加载数据 --------------------
feature_df = pd.read_parquet('features_v4.parquet')
scaler_v4 = joblib.load('scaler_v4.pkl')
cat_maps = joblib.load('cat_maps.pkl')
n_countries = len(cat_maps['country'])
n_cats = len(cat_maps['cat'])

with open('feat_cols.json') as f:
    cont_feats = json.load(f)

Xc = feature_df[cont_feats].values.astype(np.float32)
cc = feature_df['country_code'].values.astype(int)
ct = feature_df['cat_code'].values.astype(int)
y = feature_df['label'].values.astype(int)

train_mask = feature_df['date'] <= '2020-12-31'
val_mask = (feature_df['date'] > '2020-12-31') & (feature_df['date'] <= '2023-12-31')
test_mask = (feature_df['date'] > '2023-12-31') & (feature_df['date'] <= '2025-12-31')

# 标准化（用 train 拟合的 scaler_v4，只作用于连续特征）
Xc_tr = scaler_v4.transform(Xc[train_mask]);  Xc_val = scaler_v4.transform(Xc[val_mask]);  Xc_te = scaler_v4.transform(Xc[test_mask])
cc_tr, cc_val, cc_te = cc[train_mask], cc[val_mask], cc[test_mask]
ct_tr, ct_val, ct_te = ct[train_mask], ct[val_mask], ct[test_mask]
y_tr, y_val, y_te = y[train_mask], y[val_mask], y[test_mask]
SPW = (len(y_tr) - y_tr.sum()) / y_tr.sum()
print(f"train={len(y_tr)} val={len(y_val)} test={len(y_te)}  正样本比 train={y_tr.mean():.3f} test={y_te.mean():.3f}")
print(f"国家数={n_countries} 品类数={n_cats}  SPW={SPW:.2f}")

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# -------------------- 2. 树模型 OOF + 最终模型 --------------------
def oof_tree(make_fn):
    oof = np.zeros(len(y_tr))
    final = make_fn(); final.fit(Xc_tr, y_tr)
    val_p = final.predict_proba(Xc_val)[:, 1]
    te_p = final.predict_proba(Xc_te)[:, 1]
    for tr_i, va_i in tscv.split(Xc_tr):
        m = make_fn(); m.fit(Xc_tr[tr_i], y_tr[tr_i])
        oof[va_i] = m.predict_proba(Xc_tr[va_i])[:, 1]
    return oof, val_p, te_p

print("训练 LightGBM（OOF）...")
lgb_oof, lgb_val_p, lgb_te_p = oof_tree(lambda: lgb.LGBMClassifier(
    n_estimators=300, num_leaves=63, max_depth=7, learning_rate=0.02,
    min_child_samples=80, subsample=0.7, colsample_bytree=0.8,
    reg_alpha=1e-7, reg_lambda=1e-5, scale_pos_weight=SPW,
    random_state=42, n_jobs=-1, verbose=-1))

print("训练 XGBoost（OOF）...")
xgb_oof, xgb_val_p, xgb_te_p = oof_tree(lambda: xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.02, scale_pos_weight=SPW,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=1,
    random_state=42, n_jobs=-1, verbosity=0))

# -------------------- 3. MLP（真实类别嵌入）--------------------
class MLPWithEmbeddings(nn.Module):
    def __init__(self, cont_dim, n_countries, n_cats, embed_dim=8, hidden=(128,64,32), dropout=0.3):
        super().__init__()
        self.country_emb = nn.Embedding(max(n_countries, 1), embed_dim)
        self.cat_emb = nn.Embedding(max(n_cats, 1), embed_dim)
        prev = cont_dim + 2 * embed_dim
        layers = []
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, xc, cc, ct):
        cc = cc.clamp(min=0); ct = ct.clamp(min=0)
        e = torch.cat([self.country_emb(cc), self.cat_emb(ct)], dim=1)
        return self.net(torch.cat([xc, e], dim=1)).squeeze(1)

def _train_mlp(model, Xc_, cc_, ct_, y_, epochs=MLP_EPOCHS):
    model.to(DEVICE)
    ds = TensorDataset(torch.FloatTensor(Xc_), torch.LongTensor(cc_), torch.LongTensor(ct_), torch.FloatTensor(y_))
    dl = DataLoader(ds, batch_size=MLP_BATCH, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([SPW]).to(DEVICE))
    model.train()
    for _ in range(epochs):
        for xb, cb, tb, yb in dl:
            xb, cb, tb, yb = xb.to(DEVICE), cb.to(DEVICE), tb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb, cb, tb), yb); loss.backward(); opt.step()

def _mlp_proba(model, Xc_, cc_, ct_, batch=4096):
    model.eval()
    ds = TensorDataset(torch.FloatTensor(Xc_), torch.LongTensor(cc_), torch.LongTensor(ct_))
    dl = DataLoader(ds, batch_size=batch)
    out = []
    with torch.no_grad():
        for xb, cb, tb in dl:
            xb, cb, tb = xb.to(DEVICE), cb.to(DEVICE), tb.to(DEVICE)
            out.append(torch.sigmoid(model(xb, cb, tb)).cpu().numpy())
    return np.concatenate(out)

cont_dim = Xc_tr.shape[1]
def make_mlp(): return MLPWithEmbeddings(cont_dim, n_countries, n_cats)

print("训练 MLP（OOF + 最终模型，含类别嵌入）...")
mlp_oof = np.zeros(len(y_tr))
final_mlp = make_mlp(); _train_mlp(final_mlp, Xc_tr, cc_tr, ct_tr, y_tr)
mlp_val_p = _mlp_proba(final_mlp, Xc_val, cc_val, ct_val)
mlp_te_p = _mlp_proba(final_mlp, Xc_te, cc_te, ct_te)
for k, (tr_i, va_i) in enumerate(tscv.split(Xc_tr), 1):
    m = make_mlp(); _train_mlp(m, Xc_tr[tr_i], cc_tr[tr_i], ct_tr[tr_i], y_tr[tr_i])
    mlp_oof[va_i] = _mlp_proba(m, Xc_tr[va_i], cc_tr[va_i], ct_tr[va_i])
    print(f"  MLP fold {k}/{N_SPLITS} OOF done")

# -------------------- 4. Stacking 二级模型（仅用 train OOF，无泄漏）-------------------
print("训练 Stacking 逻辑回归（train OOF）...")
train_stack = np.column_stack([lgb_oof, xgb_oof, mlp_oof])
meta = LogisticRegression(max_iter=1000, class_weight='balanced')
meta.fit(train_stack, y_tr)

val_stack = np.column_stack([lgb_val_p, xgb_val_p, mlp_val_p])
te_stack = np.column_stack([lgb_te_p, xgb_te_p, mlp_te_p])
val_final_p = meta.predict_proba(val_stack)[:, 1]
te_final_p = meta.predict_proba(te_stack)[:, 1]

# -------------------- 5. 阈值只在 val 上选，test 仅报告 --------------------
best_f1, best_th = 0, 0.5
for th in np.arange(0.1, 0.9, 0.01):
    f1 = f1_score(y_val, (val_final_p >= th).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_th = f1, th
y_te_pred = (te_final_p >= best_th).astype(int)

print("\n========== V5 集成（无泄漏 + MLP嵌入）测试集表现 ==========")
print(f"选定阈值(基于val): {best_th:.2f}   val F1: {best_f1:.4f}")
print(f"TEST F1-score : {f1_score(y_te, y_te_pred, zero_division=0):.4f}")
print(f"TEST 召回率    : {recall_score(y_te, y_te_pred):.4f}")
print(f"TEST 精确率    : {precision_score(y_te, y_te_pred):.4f}")
print("TEST 混淆矩阵:")
print(confusion_matrix(y_te, y_te_pred))

print("\n--- 各基学习器在 TEST 上的 F1（阈值均用集成选定的 val 阈值 %.2f）---" % best_th)
for name, vp, tp in zip(['LightGBM', 'XGBoost', 'MLP'], [lgb_val_p, xgb_val_p, mlp_val_p], [lgb_te_p, xgb_te_p, mlp_te_p]):
    base_best = max(f1_score(y_val, (vp >= th).astype(int), zero_division=0) for th in np.arange(0.1, 0.9, 0.01))
    print(f"{name}: test F1 = {f1_score(y_te, (tp >= best_th).astype(int), zero_division=0):.4f}  (val最优阈值F1={base_best:.4f})")

# -------------------- 保存新模型（覆盖旧 leaky 版本）-------------------
# 重新训练最终树模型以保存（OOF 阶段的最终模型未单独保留引用）
lgb_final = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, max_depth=7, learning_rate=0.02,
    min_child_samples=80, subsample=0.7, colsample_bytree=0.8, reg_alpha=1e-7, reg_lambda=1e-5,
    scale_pos_weight=SPW, random_state=42, n_jobs=-1, verbose=-1)
lgb_final.fit(Xc_tr, y_tr)
xgb_final = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.02, scale_pos_weight=SPW,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=1, random_state=42, n_jobs=-1, verbosity=0)
xgb_final.fit(Xc_tr, y_tr)
joblib.dump(lgb_final, 'lgb_model.pkl')
joblib.dump(xgb_final, 'xgb_model.pkl')
torch.save(final_mlp.state_dict(), 'mlp_model.pth')
joblib.dump(meta, 'meta_model.pkl')
joblib.dump(scaler_v4, 'scaler_ensemble.pkl')
print("\n新模型已保存（lgb_model.pkl / xgb_model.pkl / mlp_model.pth / meta_model.pkl）")
