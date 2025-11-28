import os
import requests
import tarfile
import shutil
# Install tqdm: pip install tqdm
from tqdm import tqdm

# --- Resource Links (Unchanged) ---
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


# --- Helper function for robust downloading with progress bar ---
def download_file_with_progress(url, file_path):
    """Downloads a file from a URL to a path with a progress bar."""

    # Send a request
    try:
        response = requests.get(url, stream=True, timeout=10)  # Added timeout for robustness
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"\n\t🚨 Error downloading {os.path.basename(file_path)}: {e}")
        return False

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    # Initialize progress bar
    progress_bar = tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=f"Downloading {os.path.basename(file_path)}",
        miniters=1,
        ncols=80  # Set column width for progress bar display
    )

    # Write content to file with progress bar updates
    with open(file_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)

    progress_bar.close()
    return True


# --- Main Download Function ---
def download_amass(destination_folder):
    """Downloads, extracts, and cleans up the AMASS datasets."""

    print(f"Creating destination folder: {destination_folder}")
    os.makedirs(destination_folder, exist_ok=True)

    print("-" * 50)

    for i, url in enumerate(resource_links):
        filename = url.split("/")[-1]
        file_path = os.path.join(destination_folder, filename)

        print(f"\n▶️ Starting file {i + 1}/{len(resource_links)}: **{filename}**")

        # 1. Download the file with progress
        if not download_file_with_progress(url, file_path):
            print(f"Skipping extraction and cleanup for {filename} due to download error.")
            continue  # Move to the next link

        # 2. Extract the file
        print(f"\n\tExtracting {filename}...")
        try:
            if filename.endswith(".tar.bz2"):
                # 'r:bz2' mode for reading a bzip2-compressed tar file
                with tarfile.open(file_path, "r:bz2") as tar:
                    tar.extractall(path=destination_folder)
                print("\t✅ Extraction successful.")
            else:
                print(f"\t⚠️ Skipping extraction: Unknown file type for {filename}.")

            # 3. Remove the compressed file after extraction
            os.remove(file_path)  # Use os.remove for a single file
            print(f"\t🗑️ Cleaned up {filename}.")

        except tarfile.TarError as e:
            print(f"\t🚨 Error extracting {filename}: {e}")
            # Do not remove the file if extraction failed, in case the user wants to inspect it.
        except OSError as e:
            print(f"\t🚨 Error during cleanup for {filename}: {e}")

    print("\n" + "=" * 50)
    print("🎉 All available AMASS datasets have been processed.")
    print(f"Check the contents in: {destination_folder}")


if __name__ == "__main__":
    download_amass("assets/AMASS/SMPL-H")
