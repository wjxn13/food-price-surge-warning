"""
自动更新脚本：通过 kaggle CLI 获取 Last Updated 时间 → 条件下载 → 重训练
纯文本解析，无额外依赖
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

def get_dataset_last_updated():
    """运行 kaggle datasets metadata 并提取 Last Updated 时间"""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "metadata", DATASET_PATH],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
        # 调试：可将输出记录到日志（可选）
        # log(f"元数据输出:\n{output}")

        # 匹配类似 "Last Updated: 2026-05-31 10:00:00" 的行
        match = re.search(r"Last Updated:\s+([\d\-]+\s+[\d:]+)", output)
        if match:
            return match.group(1).strip()
        else:
            log("⚠️ 在元数据中未找到 Last Updated 字段，尝试使用文件列表最新日期")
            # 备用方案：获取文件列表，提取最新日期
            files_result = subprocess.run(
                ["kaggle", "datasets", "files", DATASET_PATH],
                capture_output=True, text=True, check=True
            )
            # 简单寻找日期格式 YYYY-MM-DD 的最晚时间
            dates = re.findall(r"(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}", files_result.stdout)
            if dates:
                # 取最大日期作为版本（实际可能不够精确，但可用）
                latest = max(dates)
                log(f"从文件列表提取的最新日期: {latest}")
                return latest
            return None
    except subprocess.CalledProcessError as e:
        log(f"❌ 获取元数据失败: {e.stderr}")
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