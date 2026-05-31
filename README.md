# 全球食品价格异常飙升预警系统 (Global Food Price Surge Early Warning System)

基于 WFP 全球食品价格数据（2015-2026）的机器学习预警系统，提供交互式 Streamlit 仪表盘，支持自动数据更新与模型重训练。

## 功能

- **多模型集成预测**：LightGBM + XGBoost + MLP Stacking 预测未来3个月价格飙升（涨幅>15%）概率。
- **交互式仪表盘**：按国家/类别筛选高风险商品，查看历史价格走势与风险热力图。
- **全自动更新流水线**：每日自动检测 Kaggle 数据集更新，下载新数据并重训练模型。
- **可部署的静态报告**：生成 HTML 预警报告。

## 快速开始

### 1. 克隆仓库
```bash   ”“bash
git clone https://github.com/wjxn13/food-price-surge-warning.git
cd food-price-surge-warning
```

### 2. 安装依赖
```bash   ”“bash
pip install -r requirements.txtPIP install -r requirements.txt
```

### 3. 获取模型与特征数据
从 [Releases](https://github.com/wjxn13/food-price-surge-warning/releases) 下载最新 `model_and_data.zip`，解压到项目根目录，确保包含：
- `features_v4.parquet`
- `price_series.parquet`
- `lgb_model.pkl`、`xgb_model.pkl`、`mlp_model.pth`、`meta_model.pkl`
- `scaler_v4.pkl`、`scaler_ensemble.pkl`、`feat_cols.json`

### 4. 运行仪表盘
```bash   ”“bash
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
| `ensemble_v4.py` | 训练 LightGBM + XGBoost + MLP 集成模型 |
| `auto_update.py` | 自动检测数据更新并触发重训练 |
| `deploy_demo.py` | 生成静态 HTML 预警报告 |
| `app.py` | Streamlit 交互式仪表盘 |

## 数据来源

[Global Food Prices Database (WFP)](https://www.kaggle.com/datasets/abhishekgupta56447/global-food-prices-database-wfp)  
License: CC BY 3.0 IGO

## 模型性能（V4 集成）

- F1-score: 0.40
- 召回率 (Recall): 48.9%
- 精确率 (Precision): 33.8%
- 数据截止: 动态更新

## 文件结构

```
├── app.py                     # Streamlit 仪表盘
├── auto_update.py             # 自动更新脚本
├── build_cache_v4.py          # 特征工程脚本
├── ensemble_v4.py             # 集成训练脚本
├── deploy_demo.py             # 静态报告生成
├── build_price_cache.py       # 价格序列提取
├── feat_cols.json             # 特征列名
├── requirements.txt           # Python 依赖
├── .gitignore                 # Git 忽略规则
├── features_v4.parquet        # 特征缓存（需从 Release 下载）
├── price_series.parquet       # 价格序列（需从 Release 下载）
├── lgb_model.pkl / xgb_model.pkl / mlp_model.pth / meta_model.pkl  # 模型文件（需从 Release 下载）
├── scaler_v4.pkl / scaler_ensemble.pkl  # 标准化器（需从 Release 下载）
└── 食品价格/                  # 原始 CSV（需自行下载或通过 auto_update.py 自动获取）
```

## 许可

本项目代码部分遵循 MIT License，数据部分遵循原始许可 (CC BY 3.0 IGO)。
