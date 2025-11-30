"""模块说明：
为项目定义常用类型别名（Dataset、VAE、Diffusion 等），便于配置/注入使用。
类型别名说明：把具体实现映射为更抽象的名称，便于在配置中替换实现
"""

import torch
from robotmdar.dataloader.data import SkeletonPrimitiveDataset
from robotmdar.diffusion.gaussian_diffusion import GaussianDiffusion
from robotmdar.diffusion.resample import UniformSampler
from robotmdar.model.mld_vae import AutoMldVae
from robotmdar.model.mld_denoiser import DenoiserTransformer
from robotmdar.train.manager import BaseManager

Dataset = SkeletonPrimitiveDataset
Diffusion = GaussianDiffusion
VAE = AutoMldVae
Denoiser = DenoiserTransformer
Manager = BaseManager
Optimizer = torch.optim.Optimizer
Distribution = torch.distributions.Distribution
SSampler = UniformSampler
