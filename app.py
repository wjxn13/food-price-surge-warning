"""
食品价格飙升预警仪表盘 (Streamlit) - 中英双语版（最终修复版）
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="食品价格飙升预警", layout="wide")
st.title("🌍 全球食品价格异常飙升预警系统")

COUNTRY_CN = {
    "AFG": "阿富汗", "ALB": "阿尔巴尼亚", "DZA": "阿尔及利亚", "AGO": "安哥拉", "ARG": "阿根廷",
    "ARM": "亚美尼亚", "AZE": "阿塞拜疆", "BGD": "孟加拉国", "BLR": "白俄罗斯", "BEN": "贝宁",
    "BTN": "不丹", "BOL": "玻利维亚", "BIH": "波黑", "BWA": "博茨瓦纳", "BRA": "巴西",
    "BFA": "布基纳法索", "BDI": "布隆迪", "CPV": "佛得角", "KHM": "柬埔寨", "CMR": "喀麦隆",
    "CAF": "中非", "TCD": "乍得", "CHL": "智利", "COL": "哥伦比亚", "COM": "科摩罗",
    "COG": "刚果（布）", "COD": "刚果（金）", "CRI": "哥斯达黎加", "CIV": "科特迪瓦", "CUB": "古巴",
    "DJI": "吉布提", "DOM": "多米尼加", "ECU": "厄瓜多尔", "EGY": "埃及", "SLV": "萨尔瓦多",
    "GNQ": "赤道几内亚", "ERI": "厄立特里亚", "EST": "爱沙尼亚", "SWZ": "斯威士兰", "ETH": "埃塞俄比亚",
    "FJI": "斐济", "GAB": "加蓬", "GMB": "冈比亚", "GEO": "格鲁吉亚", "GHA": "加纳",
    "GRC": "希腊", "GTM": "危地马拉", "GIN": "几内亚", "GNB": "几内亚比绍", "GUY": "圭亚那",
    "HTI": "海地", "HND": "洪都拉斯", "HUN": "匈牙利", "IND": "印度", "IDN": "印度尼西亚",
    "IRN": "伊朗", "IRQ": "伊拉克", "ISR": "以色列", "JAM": "牙买加", "JOR": "约旦",
    "KAZ": "哈萨克斯坦", "KEN": "肯尼亚", "KIR": "基里巴斯", "PRK": "朝鲜", "KOR": "韩国",
    "KWT": "科威特", "KGZ": "吉尔吉斯斯坦", "LAO": "老挝", "LVA": "拉脱维亚", "LBN": "黎巴嫩",
    "LSO": "莱索托", "LBR": "利比里亚", "LBY": "利比亚", "LTU": "立陶宛", "MDG": "马达加斯加",
    "MWI": "马拉维", "MYS": "马来西亚", "MDV": "马尔代夫", "MLI": "马里", "MRT": "毛里塔尼亚",
    "MEX": "墨西哥", "MDA": "摩尔多瓦", "MNG": "蒙古", "MNE": "黑山", "MAR": "摩洛哥",
    "MOZ": "莫桑比克", "MMR": "缅甸", "NAM": "纳米比亚", "NPL": "尼泊尔", "NIC": "尼加拉瓜",
    "NER": "尼日尔", "NGA": "尼日利亚", "MKD": "北马其顿", "PAK": "巴基斯坦", "PSE": "巴勒斯坦",
    "PAN": "巴拿马", "PNG": "巴布亚新几内亚", "PRY": "巴拉圭", "PER": "秘鲁", "PHL": "菲律宾",
    "POL": "波兰", "QAT": "卡塔尔", "ROU": "罗马尼亚", "RUS": "俄罗斯", "RWA": "卢旺达",
    "STP": "圣多美和普林西比", "SAU": "沙特阿拉伯", "SEN": "塞内加尔", "SRB": "塞尔维亚", "SLE": "塞拉利昂",
    "SVK": "斯洛伐克", "SVN": "斯洛文尼亚", "SLB": "所罗门群岛", "SOM": "索马里", "ZAF": "南非",
    "SSD": "南苏丹", "LKA": "斯里兰卡", "SDN": "苏丹", "SUR": "苏里南", "SYR": "叙利亚",
    "TJK": "塔吉克斯坦", "TZA": "坦桑尼亚", "THA": "泰国", "TLS": "东帝汶", "TGO": "多哥",
    "TON": "汤加", "TTO": "特立尼达和多巴哥", "TUN": "突尼斯", "TUR": "土耳其", "TKM": "土库曼斯坦",
    "UGA": "乌干达", "UKR": "乌克兰", "ARE": "阿联酋", "URY": "乌拉圭", "UZB": "乌兹别克斯坦",
    "VUT": "瓦努阿图", "VEN": "委内瑞拉", "VNM": "越南", "YEM": "也门", "ZMB": "赞比亚",
    "ZWE": "津巴布韦", "UNK": "未知"
}

CATEGORY_CN = {
    "cereals and tubers": "谷物与薯类",
    "milk and dairy": "乳制品",
    "meat, fish and eggs": "肉鱼蛋",
    "oil and fats": "油脂",
    "pulses and nuts": "豆类与坚果",
    "vegetables and fruits": "蔬菜水果",
    "miscellaneous food": "其他食品",
    "non-food": "非食品"
}

def format_country(code):
    if pd.isna(code):
        return "未知"
    return f"{COUNTRY_CN.get(code, code)} ({code})"

def format_category(cat):
    if pd.isna(cat):
        return "未知"
    return f"{CATEGORY_CN.get(cat, cat)}"

@st.cache_resource
def load_models():
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp_model = MLPModel(cont_dim=len(feat_cols)).to(device)
    mlp_model.load_state_dict(torch.load('mlp_model.pth', map_location=device))
    mlp_model.eval()

    meta_model = joblib.load('meta_model.pkl')
    return feat_cols, scaler, lgb_model, xgb_model, mlp_model, meta_model, device

feat_cols, scaler, lgb_model, xgb_model, mlp_model, meta_model, device = load_models()

@st.cache_data
def load_feature_data():
    feature_df = pd.read_parquet('features_v4.parquet')
    target_date = feature_df['date'].max()
    current = feature_df[feature_df['date'] <= target_date].sort_values('date').groupby('series_id').last().reset_index()
    return current, target_date

current_data, target_date = load_feature_data()

@st.cache_data
def load_static_map():
    DATA_DIR = Path("D:/.kaggle/食品价格/食品价格")
    csv_files = sorted(DATA_DIR.glob("wfp_food_prices_global_*.csv"))
    static_list = []
    for f in csv_files:
        tmp = pd.read_csv(f, usecols=['countryiso3', 'category', 'commodity', 'market_id', 'commodity_id'])
        tmp['series_id'] = tmp['market_id'].astype(str) + '_' + tmp['commodity_id'].astype(str)
        static_list.append(tmp[['series_id', 'countryiso3', 'category', 'commodity']].drop_duplicates())
    static_df = pd.concat(static_list).drop_duplicates(subset='series_id')
    return static_df

static_df = load_static_map()

current_with_static = current_data.merge(static_df, on='series_id', how='left')
current_with_static['country_display'] = current_with_static['countryiso3'].apply(format_country)
current_with_static['category_display'] = current_with_static['category'].apply(format_category)

def predict_warning_prob(data_df):
    X = data_df[feat_cols].values
    X_scaled = scaler.transform(X)
    lgb_prob = lgb_model.predict_proba(X_scaled)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_scaled)[:, 1]
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        mlp_logits = mlp_model(X_tensor).cpu().numpy()
    mlp_prob = 1 / (1 + np.exp(-mlp_logits))
    stacked = np.column_stack([lgb_prob, xgb_prob, mlp_prob])
    final_prob = meta_model.predict_proba(stacked)[:, 1]
    return final_prob

current_with_static['warning_prob'] = predict_warning_prob(current_with_static)

# 侧边栏
st.sidebar.header("筛选条件")
country_display_all = sorted(current_with_static['country_display'].unique())
selected_country_display = st.sidebar.selectbox("选择国家", ["全部"] + country_display_all)

category_display_all = sorted(current_with_static['category_display'].unique())
selected_category_display = st.sidebar.selectbox("选择类别", ["全部"] + category_display_all)

if selected_country_display != "全部":
    selected_country_code = selected_country_display.split("(")[-1].rstrip(")")
else:
    selected_country_code = None

if selected_category_display != "全部":
    reverse_cat = {v: k for k, v in CATEGORY_CN.items()}
    selected_category_code = reverse_cat.get(selected_category_display, selected_category_display)
else:
    selected_category_code = None

filtered = current_with_static.copy()
if selected_country_code:
    filtered = filtered[filtered['countryiso3'] == selected_country_code]
if selected_category_code:
    filtered = filtered[filtered['category'] == selected_category_code]

filtered_sorted = filtered.sort_values('warning_prob', ascending=False)

# 风险表格
st.subheader(f"高风险商品 Top 20 (当前筛选: {selected_country_display if selected_country_display!='全部' else '所有国家'}, {selected_category_display if selected_category_display!='全部' else '所有类别'})")

table_df = filtered_sorted.head(20)[['country_display', 'category_display', 'commodity', 'warning_prob']].copy()
table_df.columns = ['国家', '类别', '商品', '飙升概率']
table_df['飙升概率'] = table_df['飙升概率'].apply(lambda x: f"{x:.2%}")
st.dataframe(table_df, width='stretch')

# 风险热力图
heat_df = filtered_sorted.head(20).copy()
heat_df['label'] = heat_df['country_display'] + ' - ' + heat_df['commodity']

fig_bar = px.bar(
    heat_df.iloc[::-1],
    x='warning_prob',
    y='label',
    color='warning_prob',
    orientation='h',
    hover_data={'country_display': True, 'category_display': True, 'commodity': True},
    labels={'warning_prob': '飙升概率', 'label': '商品'},
    title="风险热力图 (Top 20)"
)
fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_bar, width='stretch')

# 历史价格走势
st.subheader("历史价格走势 & 预警概率")
if filtered_sorted.empty:
    st.warning("当前筛选条件下没有数据，请调整筛选条件")
else:
    series_choices = filtered_sorted[['series_id', 'country_display', 'commodity', 'warning_prob']].copy()
    # 关键修复：确保所有列为纯字符串，使用 .apply(str) 或 astype(str) 并填充 NaN
    series_choices = series_choices.fillna({'country_display': '未知', 'commodity': '未知商品', 'warning_prob': 0})
    series_choices['prob_str'] = series_choices['warning_prob'].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else "概率未知")
    series_choices['label'] = (
        series_choices['country_display'].astype(str)
        + ' - '
        + series_choices['commodity'].astype(str)
        + ' (概率: '
        + series_choices['prob_str']
        + ')'
    )
    labels = series_choices['label'].tolist()
    if not labels:
        st.warning("没有可供选择的商品序列")
    else:
        selected_label = st.selectbox("选择商品序列查看详情", labels)
        # 通过 label 查找 series_id
        match = series_choices[series_choices['label'] == selected_label]
        if not match.empty:
            selected_series = match['series_id'].values[0]
        else:
            selected_series = series_choices['series_id'].values[0]  # fallback

        @st.cache_data
        def load_price_history():
            return pd.read_parquet('price_series.parquet')

        price_df = load_price_history()
        series_price = price_df[price_df['series_id'] == selected_series].sort_values('date')
        if not series_price.empty:
            current_warning = filtered_sorted[filtered_sorted['series_id'] == selected_series]['warning_prob'].values[0]
        else:
            current_warning = 0.0

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=series_price['date'],
            y=series_price['usdprice'],
            mode='lines+markers',
            name='USD 价格'
        ))
        if not series_price.empty:
            fig_ts.add_annotation(
                x=series_price['date'].iloc[-1],
                y=series_price['usdprice'].iloc[-1],
                text=f"当前预警概率: {current_warning:.1%}",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=-30
            )
        fig_ts.update_layout(title=f"历史价格走势 - {selected_label}", xaxis_title="日期", yaxis_title="USD 价格")
        st.plotly_chart(fig_ts, width='stretch')

        with st.expander("查看该商品最新特征值"):
            series_feat = current_data[current_data['series_id'] == selected_series]
            if not series_feat.empty:
                st.dataframe(series_feat[feat_cols].T.rename(columns={series_feat.index[0]: '特征值'}), width='stretch')

st.markdown("---")
st.caption(f"数据截止: {target_date.date()} | 模型: LightGBM + XGBoost + MLP 集成")