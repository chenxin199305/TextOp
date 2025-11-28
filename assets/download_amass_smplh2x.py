import os
import requests
import zipfile  # 修复了原来使用 tarfile 的错误
import logging
from tqdm import tqdm

# --- 日志配置 ---
# 创建一个 Logger 实例
logger = logging.getLogger(__name__)
# 设定最低处理级别为 INFO
logger.setLevel(logging.INFO)

# 创建一个格式器 (Formatter)
formatter = logging.Formatter(
    '[%(asctime)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 1. 控制台处理器 (Console Handler)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# 2. 文件处理器 (File Handler)
fh = logging.FileHandler('amass_download.log', encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# --- 资源链接 (不变) ---
resource_links = [
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/ACCAD.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/BMLhandball.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/BMLmovi.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/BMLrub.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/CMU.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/DanceDB.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/DFaust.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/EKUT.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/EyesJapanDataset.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/GRAB.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/HDM05.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/HUMAN4D.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/HumanEva.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/KIT.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/MoSh.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/PosePrior.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/SFU.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/SOMA.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/SSM.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/TCDHands.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/TotalCapture.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/Transitions.zip",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H2X/WEIZMANN.zip",
]


# --- 辅助函数：带进度条的下载 ---
def download_file_with_progress(url, file_path):
    """Downloads a file from a URL to a path with a progress bar."""

    # 发送请求
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # 使用 logger.error 记录致命错误
        logger.error(f"下载 {os.path.basename(file_path)} 发生网络错误: {e}")
        return False

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    if total_size == 0:
        logger.warning(f"无法获取 {os.path.basename(file_path)} 的文件大小，将以流模式下载。")

    # 注意：tqdm 的 desc 参数在这里充当了进度条的描述信息
    progress_bar = tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=f"下载 {os.path.basename(file_path)}",
        miniters=1,
        ncols=80,
        # 文件句柄：确保日志消息不会覆盖进度条
        file=os.sys.stderr
    )

    try:
        with open(file_path, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))
    except Exception as e:
        progress_bar.close()
        logger.error(f"写入文件 {file_path} 时发生错误: {e}")
        # 清理可能已经部分写入的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        return False

    progress_bar.close()
    return True


# --- 主下载函数 ---
def download_amass(destination_folder):
    """下载、解压缩 AMASS 数据集，并保留原始压缩包。"""

    logger.info("-" * 50)
    logger.info(f"开始 AMASS 数据集下载。")
    logger.info(f"目标文件夹: {destination_folder}")
    os.makedirs(destination_folder, exist_ok=True)
    logger.info("-" * 50)

    total_files = len(resource_links)
    for i, url in enumerate(resource_links):
        filename = url.split("/")[-1]
        file_path = os.path.join(destination_folder, filename)

        # 使用 logger.info 记录当前处理的进度
        logger.info(f"\n▶️ 处理文件 {i + 1}/{total_files}: {filename}")

        # 1. 检查是否已存在且完整
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logger.info(f"文件 {filename} 已存在，跳过下载。")
        else:
            # 1. 下载文件
            if not download_file_with_progress(url, file_path):
                logger.error(f"跳过 {filename} 的解压步骤，因为下载失败。")
                continue  # 跳到下一个链接

        # 2. 解压缩文件
        logger.info(f"正在解压缩 {filename}...")
        try:
            if filename.endswith(".zip"):
                # 修复：使用 zipfile 模块
                with zipfile.ZipFile(file_path, 'r') as zf:
                    zf.extractall(path=destination_folder)
                logger.info(f"文件 {filename} 解压缩成功。")
            else:
                logger.warning(f"跳过解压缩: {filename} 文件类型未知或不支持。")

            # 3. 保留压缩包
            logger.info(f"保留原压缩包: {filename}。")

        # 捕获 zipfile 相关的错误
        except zipfile.BadZipFile:
            logger.error(f"解压缩 {filename} 失败: 文件不是一个有效的 ZIP 文件，可能下载不完整或已损坏。")
        except Exception as e:
            logger.error(f"解压缩 {filename} 发生未知错误: {e}")

    logger.info("\n" + "=" * 50)
    logger.info("🎉 所有 AMASS 数据集已处理完成（已下载并解压）。")
    logger.info(f"数据和压缩包都保存在: {destination_folder}")
    logger.info("请查看 amass_download.log 文件获取详细记录。")


if __name__ == "__main__":
    download_amass("assets/AMASS/SMPL-H2X")
