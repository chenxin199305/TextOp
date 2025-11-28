import os
import requests
import tarfile
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
# 检查是否已添加，避免重复
if not logger.handlers:
    logger.addHandler(ch)

# 2. 文件处理器 (File Handler)
fh = logging.FileHandler('amass_download_tar.log', encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# --- 资源链接 (不变) ---
resource_links = [
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/ACCAD.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/BMLhandball.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/BMLmovi.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/BMLrub.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/CMU.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/DanceDB.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/DFaust.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/EKUT.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/EyesJapanDataset.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/GRAB.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/HDM05.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/HUMAN4D.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/HumanEva.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/KIT.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/MoSh.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/PosePrior.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/SFU.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/SOMA.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/SSM.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/TCDHands.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/TotalCapture.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/Transitions.tar.bz2",
    "https://dataset-1302548221.cos.ap-shanghai.myqcloud.com/AMASS/SMPL-H/WEIZMANN.tar.bz2",
]


# --- 辅助函数：带进度条的下载 ---
def download_file_with_progress(url, file_path):
    """Downloads a file from a URL to a path with a progress bar."""

    # 发送请求
    try:
        # 增加 timeout 和 stream=True
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # 检查HTTP状态码 (4xx 或 5xx)
    except requests.exceptions.RequestException as e:
        # 使用 logger.error 替换 print
        logger.error(f"下载 {os.path.basename(file_path)} 发生网络错误: {e}")
        return False

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    if total_size == 0:
        logger.warning(f"无法获取 {os.path.basename(file_path)} 的文件大小，将以流模式下载。")

    # 初始化进度条，并设置 file=os.sys.stderr 确保进度条在控制台正确显示，不与日志混淆
    progress_bar = tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=f"下载 {os.path.basename(file_path)}",
        miniters=1,
        ncols=80,
        file=os.sys.stderr
    )

    # 写入文件内容并更新进度条
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
    logger.info(f"创建目标文件夹: {destination_folder}")
    os.makedirs(destination_folder, exist_ok=True)
    logger.info("-" * 50)

    total_files = len(resource_links)
    for i, url in enumerate(resource_links):
        filename = url.split("/")[-1]
        file_path = os.path.join(destination_folder, filename)

        logger.info(f"\n▶️ 处理文件 {i + 1}/{total_files}: {filename}")

        # 1. 检查是否已存在（可选的重复下载检查）
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
            if filename.endswith(".tar.bz2"):
                # 'r:bz2' 模式用于读取 bzip2 压缩的 tar 文件 (这是正确的模式)
                with tarfile.open(file_path, "r:bz2") as tar:
                    tar.extractall(path=destination_folder)
                logger.info(f"文件 {filename} 解压缩成功。")
            else:
                logger.warning(f"跳过解压缩: {filename} 文件类型未知。")

            # 3. 保留压缩包
            logger.info(f"保留原压缩包: {filename}。")

        except tarfile.TarError as e:
            logger.error(f"解压缩 {filename} 发生 Tar 文件错误: {e}")
        except OSError as e:
            logger.error(f"解压缩 {filename} 发生文件操作错误: {e}")
        except Exception as e:
            logger.error(f"解压缩 {filename} 发生未知错误: {e}")

    logger.info("\n" + "=" * 50)
    logger.info("🎉 所有 AMASS 数据集已处理完成（已下载并解压）。")
    logger.info(f"数据和压缩包都保存在: {destination_folder}")
    logger.info("请查看 amass_download_tar.log 文件获取详细记录。")


if __name__ == "__main__":
    download_amass("assets/AMASS/SMPL-H")
