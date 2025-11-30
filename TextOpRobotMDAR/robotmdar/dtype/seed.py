"""模块说明：
用于设置全局随机种子以提高实验可重复性。该模块将同步 Python、NumPy、PyTorch（CPU/GPU）
以及 Python 哈希随机化相关的种子，并可选地开启 PyTorch 的确定性模式以尽量保证可复现性。
"""

import random
import numpy as np
import torch
import os
from loguru import logger


def set(seed=0, torch_deterministic=False):
    """设置全局随机种子并可选开启 PyTorch 确定性模式。

    参数:
        seed (int): 随机种子值，默认 0。
        torch_deterministic (bool): 若为 True，则尽可能启用 PyTorch/CuDNN 的确定性选项，
            以提高可复现性（可能带来性能开销或部分操作无法使用）。

    返回:
        int: 返回传入的 seed（便于记录或链式使用）。
    """
    logger.info("Setting seed: {}".format(seed))

    # Python 内置 random 模块的种子（影响 random.random 等）
    random.seed(seed)
    # NumPy 随机数种子（影响 np.random）
    np.random.seed(seed)
    # PyTorch CPU 随机种子（影响 torch.rand 等）
    torch.manual_seed(seed)
    # 控制 Python 的哈希随机化（影响 dict/key 等的哈希种子）
    os.environ["PYTHONHASHSEED"] = str(seed)
    # PyTorch GPU 随机种子（单 GPU）
    torch.cuda.manual_seed(seed)
    # PyTorch GPU 随机种子（多 GPU）
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # 启用 cuBLAS 工作区配置以提高可复现性（NVIDIA 推荐的设置）
        # 参考: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # 关闭 benchmark 模式，避免因选择不同算法导致不可复现
        torch.backends.cudnn.benchmark = False
        # 尽量使用确定性算法（若后端支持）
        torch.backends.cudnn.deterministic = True
        # 在 PyTorch 级别强制使用确定性算法（某些不确定操作会报错）
        torch.use_deterministic_algorithms(True)
    else:
        # 默认性能模式：允许 cudnn 根据运行时选择最快算法
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed
