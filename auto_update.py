"""
自动更新脚本：检测 Kaggle 更新 → 条件下载 → 重训练（增强日志与超时）
"""
import subprocess
from pathlib import Path
from datetime import datetime
import re

DATASET_PATH = "abhishekgupta56447/global-food-prices-database-wfp"
DATA_DIR = Path("D:/.kaggle/食品价格/食品价格")
VERSION_FILE = "last_updated.txt"
LOG_FILE = "update_log.txt"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_with_timeout(cmd, timeout=30):
    """带超时的 subprocess 调用"""
    log(f"执行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        log(f"命令成功，返回码 {result.returncode}")
        return result
    except subprocess.TimeoutExpired:
        log(f"❌ 命令超时（{timeout} 秒）")
        raise
    except subprocess.CalledProcessError as e:
        log(f"❌ 命令返回错误码 {e.returncode}: {e.stderr}")
        raise

def get_dataset_last_updated():
    """获取数据集最后更新时间（优先元数据，备用文件列表）"""
    log("正在获取数据集元数据...")
    try:
        result = run_with_timeout(["kaggle", "datasets", "metadata", DATASET_PATH], timeout=15)
        output = result.stdout
        match = re.search(r"Last Updated:\s+([\d\-]+\s+[\d:]+)", output)
        if match:
            return match.group(1).strip()
        else:
            log("⚠️ 元数据中未找到 Last Updated 字段，尝试文件列表")
    except Exception as e:
        log(f"⚠️ 获取元数据失败: {e}，尝试文件列表")

    # 备用方案：通过文件列表提取最新日期
    log("正在获取文件列表...")
    try:
        result = run_with_timeout(["kaggle", "datasets", "files", DATASET_PATH], timeout=20)
        output = result.stdout
        dates = re.findall(r"(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}", output)
        if dates:
            latest = max(dates)
            log(f"从文件列表提取的最新日期: {latest}")
            return latest
        else:
            log("❌ 文件列表中未找到日期")
            return None
    except Exception as e:
        log(f"❌ 文件列表获取失败: {e}")
        return None

def read_last_updated():
    path = Path(VERSION_FILE)
    if path.exists():
        return path.read_text().strip()
    return ""

def write_last_updated(updated_str):
    Path(VERSION_FILE).write_text(updated_str)

def download_dataset():
    log("开始下载最新数据集...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", DATASET_PATH, "-p", str(DATA_DIR), "--unzip", "--force"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        log("✅ 数据集下载并解压成功")
    else:
        log(f"❌ 下载失败: {result.stderr}")
        raise Exception("下载失败")

def run_training():
    log("开始构建特征缓存 V4...")
    subprocess.run(["python", "build_cache_v4.py"], check=True)
    log("特征构建完成，开始训练集成模型...")
    subprocess.run(["python", "ensemble_v4.py"], check=True)
    log("✅ 模型训练完成")

def main():
    log("========== 检查数据集更新 ==========")
    latest_update = get_dataset_last_updated()
    if not latest_update:
        log("无法获取最新更新时间，强制下载并重训练（兜底）")
    else:
        current_update = read_last_updated()
        log(f"当前记录: {current_update or '无'}, 最新更新: {latest_update}")
        if latest_update == current_update:
            log("数据集未更新，无需操作。")
            return
        log("检测到数据集更新！")

    try:
        download_dataset()
    except Exception as e:
        log(f"下载异常: {e}")
        return

    if latest_update:
        write_last_updated(latest_update)

    try:
        run_training()
        log("🎉 自动更新完成！")
    except Exception as e:
        log(f"训练过程出错: {e}")

if __name__ == "__main__":
    main()