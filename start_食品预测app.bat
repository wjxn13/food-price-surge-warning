@echo off
cd /d "D:\.kaggle\食品价格"

echo 正在检测数据更新...
python auto_update.py

echo 正在启动仪表盘...
streamlit run app.py
pause