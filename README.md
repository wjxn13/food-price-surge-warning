# 全球食品价格异常飙升预警系统 (Global Food Price Surge Early Warning System)

基于 WFP 全球食品价格数据（2015-2026）的机器学习预警系统，提供交互式 Streamlit 仪表盘，支持自动数据更新与模型重训练。

## 功能

- **多模型集成预测**：LightGBM + XGBoost + MLP Stacking 预测未来3个月价格飙升（涨幅>15%）概率。
- **交互式仪表盘**：按国家/类别筛选高风险商品，查看历史价格走势与风险热力图。
- **全自动更新流水线**：每日自动检测 Kaggle 数据集更新，下载新数据并重训练模型。
- **可部署的静态报告**：生成 HTML 预警报告。

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/wjxn13/food-price-surge-warning.git
cd food-price-surge-warning
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 获取模型与特征数据
从 [Releases](https://github.com/wjxn13/food-price-surge-warning/releases) 下载最新 `model_and_data.zip`，解压到项目根目录，确保包含：
- `features_v4.parquet`
- `price_series.parquet`
- `lgb_model.pkl`、`xgb_model.pkl`、`mlp_model.pth`、`meta_model.pkl`
- `scaler_v4.pkl`、`scaler_ensemble.pkl`、`feat_cols.json`

### 4. 运行仪表盘
```bash
streamlit run app.py
```
浏览器访问 http://localhost:8501 即可交互分析。

## 自动更新（可选）

若要启用每日自动检测 Kaggle 数据集更新并重训练：

1. 获取 Kaggle API 密钥（`kaggle.json`），放入 `%userprofile%\.kaggle\`。
2. 设置 Windows 任务计划程序，每日执行：
   ```
   程序：python.exe
   参数："D:\你的路径\auto_update.py"
   起始于：D:\你的路径\
   ```

## 主要脚本

| 脚本 | 用途 |
|------|------|
| `build_cache_v4.py` | 从原始 CSV 生成 V4 特征缓存 |
| `build_price_cache.py` | 提取月度价格序列供仪表盘使用 |
| `ensemble_v4.py` | 训练 LightGBM + XGBoost + MLP 集成（**旧版，含数据穿越，保留作泄漏对照**） |
| `ensemble_v5.py` | 训练无泄漏 + MLP 类别嵌入的 V5 集成（**当前主模型**） |
| `add_cat_cols.py` | 为 `features_v4.parquet` 补回国家/品类编码列（绕过 OOM 重建） |
| `auto_update.py` | 自动检测数据更新并触发重训练 |
| `deploy_demo.py` | 生成静态 HTML 预警报告 |
| `app.py` | Streamlit 交互式仪表盘 |

## 数据来源

[Global Food Prices Database (WFP)](https://www.kaggle.com/datasets/abhishekgupta56447/global-food-prices-database-wfp)  
License: CC BY 3.0 IGO

## 模型性能（V5 集成，已修复数据穿越）

测试集为 2024–2025；验证集 2021–2023 **仅用于选阈值**；训练集 ≤2020。所有指标均为测试集（唯一评估集）结果。

| 协议 | TEST F1 | 召回率 | 精确率 | 说明 |
|------|---------|--------|--------|------|
| V4（旧，含数据穿越） | 0.3997 | 0.5057 | 0.3305 | 验证集同时用于 MLP 早停与二级模型训练，指标被轻微夸大 |
| **V5（新，无泄漏 + MLP 嵌入）** | **0.3962** | **0.4934** | **0.3309** | 阈值由 val 网格选定（0.61），测试集唯一评估 |

各基学习器在 test 上的 F1（阈值统一用集成选定的 val 阈值 0.61）：LightGBM **0.3977** / XGBoost **0.3978** / MLP（含类别嵌入）**0.3936**。

> 注：修复数据穿越后 F1 下降约 0.0035，证明旧版数字存在轻微高估——这是**诚实的真实泛化表现**，而非性能回退。本任务为强不平衡的时序预警（正样本稀少、价格波动具时序相关性），F1 受标注/样本结构约束。MLP 嵌入虽已恢复，但 15 轮训练下仍略逊于树模型，多样性增益有限，属后续可深挖方向（见优化路线）。

## 模型优化路线

### 已实现（V5）
1. **✅ 修复 Stacking 数据穿越**：V4 中验证集 `val_stack` 同时用于 MLP 早停与二级逻辑回归训练，造成信息泄漏、test 指标虚高。V5 改用 `TimeSeriesSplit` 在**训练集内**生成折外（OOF）概率训练二级模型，验证集仅用于选阈值，测试集为唯一评估集。修复后真实 TEST F1 由 0.3997（泄漏）降至 0.3962，属诚实泛化结果（详见「模型性能」对照表）。
2. **✅ 恢复 MLP 类别嵌入**：`ensemble_v4.py` 的 MLP 因 Embedding 对齐问题退化成纯连续特征网络，与树模型高度同质。`ensemble_v5.py` 的 `MLPWithEmbeddings` 真正实现 `country_code` / `cat_code` 的 `nn.Embedding` 分支并与连续特征拼接，恢复类别交叉能力。对应 `add_cat_cols.py` 为 `features_v4.parquet` 补回了国家/品类编码列（绕过 OOM 重建）。

### 待做 / 可提升方向
3. **MLP 嵌入增益挖掘**：当前 15 轮训练的 MLP 在 test 上（0.3936）仍略逊于树模型（0.3977 / 0.3978），多样性增益有限。可加训轮数、调大 embedding 维度、或改用更深的网络，拉开与树模型的差异。
4. **特征工程增量**（原路线 #3）：
   - 动量交叉（`ret_3m - ret_12m`）、波动率回归（近 3 月波动相对 12 月均值的倍数）；
   - 与 12 月滚动均线的偏离、`dist_to_12m_high` 已在用，可补 **回撤深度**；
   - 类别/国家层面的**滞后同比**（去年同期涨幅），捕捉年度周期。
5. **阈值策略**（原路线 #4）：改用在验证集上最大化 **PR 曲线 / Youden J** 的自动阈值，并对不同国家/类别做阈值校准（正样本比例差异大）。
6. **类别不平衡**（原路线 #5）：除现有 `scale_pos_weight` 外，可试 **focal loss（MLP）** 或 **EasyEnsemble / 过采样**，直接拉升少数类召回。

> 优化点与本项目数据/代码均已在本地 `D:\.kaggle\食品价格\` 验证可读，重训需本地 WFP CSV（见 `食品价格/` 子目录）与 `features_v4.parquet`。

## 文件结构

```
├── app.py                     # Streamlit 仪表盘
├── auto_update.py             # 自动更新脚本
├── build_cache_v4.py          # 特征工程脚本
├── ensemble_v4.py             # 集成训练脚本（旧版，含数据穿越，对照用）
├── ensemble_v5.py             # 集成训练脚本（当前主模型，无泄漏 + MLP 嵌入）
├── add_cat_cols.py            # 为 features_v4.parquet 补国家/品类编码列
├── deploy_demo.py             # 静态报告生成
├── build_price_cache.py       # 价格序列提取
├── feat_cols.json             # 特征列名
├── requirements.txt           # Python 依赖（新增，详见上文）
├── .gitignore                 # Git 忽略规则
├── features_v4.parquet        # 特征缓存（需从 Release 下载）
├── price_series.parquet       # 价格序列（需从 Release 下载）
├── lgb_model.pkl / xgb_model.pkl / mlp_model.pth / meta_model.pkl  # 模型文件（需从 Release 下载）
├── scaler_v4.pkl / scaler_ensemble.pkl  # 标准化器（需从 Release 下载）
└── 食品价格/                  # 原始 WFP CSV 子目录（本地数据，已被 .gitignore 忽略）
```

## 许可

本项目代码部分遵循 MIT License，数据部分遵循原始许可 (CC BY 3.0 IGO)。
