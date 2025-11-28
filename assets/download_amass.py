import os
import requests
import tarfile
from tqdm import tqdm  # Install tqdm: pip install tqdm

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
        print(f"\n\t🚨 下载 {os.path.basename(file_path)} 发生错误: {e}")
        return False

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    # 初始化进度条
    progress_bar = tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=f"下载 {os.path.basename(file_path)}",
        miniters=1,
        ncols=80  # 进度条列宽
    )

    # 写入文件内容并更新进度条
    with open(file_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)

    progress_bar.close()
    return True


# --- 主下载函数 ---
def download_amass(destination_folder):
    """下载、解压缩 AMASS 数据集，并保留原始压缩包。"""

    print(f"创建目标文件夹: {destination_folder}")
    os.makedirs(destination_folder, exist_ok=True)

    print("-" * 50)

    for i, url in enumerate(resource_links):
        filename = url.split("/")[-1]
        file_path = os.path.join(destination_folder, filename)

        print(f"\n▶️ 开始处理文件 {i + 1}/{len(resource_links)}: **{filename}**")

        # 1. 下载文件
        if not download_file_with_progress(url, file_path):
            print(f"跳过 {filename} 的解压步骤，因为下载失败。")
            continue  # 跳到下一个链接

        # 2. 解压缩文件
        print(f"\n\t解压缩 {filename}...")
        try:
            if filename.endswith(".tar.bz2"):
                # 'r:bz2' 模式用于读取 bzip2 压缩的 tar 文件
                with tarfile.open(file_path, "r:bz2") as tar:
                    # 💡 注意：为了安全起见，通常会检查 tar 文件中的成员路径，但此处假设数据集是可信的
                    tar.extractall(path=destination_folder)
                print("\t✅ 解压缩成功。")
            else:
                print(f"\t⚠️ 跳过解压缩: {filename} 文件类型未知。")

            # 3. *** 移除清理压缩包的代码，以保留它 ***
            print(f"\t📦 保留原压缩包: {filename}。")

        except tarfile.TarError as e:
            print(f"\t🚨 解压缩 {filename} 发生错误: {e}")
        except OSError as e:
            print(f"\t🚨 文件操作错误: {e}")

    print("\n" + "=" * 50)
    print("🎉 所有 AMASS 数据集已处理完成（已下载并解压）。")
    print(f"数据和压缩包都保存在: {destination_folder}")


if __name__ == "__main__":
    download_amass("assets/AMASS/SMPL-H")
