import pdb
from functools import reduce
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution

# from robotmdar.model.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from robotmdar.model.operator import PositionalEncoding
from robotmdar.model.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from robotmdar.model.operator.position_encoding import build_position_encoding
from robotmdar.model.modules import Patcher1D, UnPatcher1D

"""
vae

skip connection encoder 
skip connection decoder

mem for each decoder layer
"""


class AutoMldVae(nn.Module):
    """
    AutoMldVae: A Transformer-based Variational Autoencoder (VAE) for motion data.

    This class implements a VAE architecture using Transformer encoders and decoders.
    It supports skip connections, latent distribution modeling, and optional wavelet-based
    patching for input data. The architecture can be configured as either "all_encoder"
    or "encoder_decoder".

    Args:
        nfeats (int): Number of features per frame.
        latent_dim (list): Latent space dimensions as [number of latent tokens, token feature dimension].
        h_dim (int): Hidden dimension size for the Transformer.
        ff_size (int): Feed-forward network expansion size.
        num_layers (int): Number of Transformer layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        arch (str): Architecture type, either "all_encoder" or "encoder_decoder".
        normalize_before (bool): Whether to normalize before attention layers.
        activation (str): Activation function to use (e.g., "gelu").
        position_embedding (str): Type of positional embedding ("learned" or others).
        use_patcher (bool): Whether to use wavelet-based patching.
        patch_size (int): Size of patches for wavelet transformation.
        patch_method (str): Wavelet transformation method (e.g., "haar").

    Attributes:
        latent_size (int): Number of latent tokens.
        latent_dim (int): Latent token feature dimension.
        h_dim (int): Hidden dimension size for the Transformer.
        arch (str): Architecture type.
        use_patcher (bool): Whether wavelet patching is enabled.
        patch_size (int): Patch size for wavelet transformation.
        patch_method (str): Wavelet transformation method.
        encoder (nn.Module): Transformer encoder with skip connections.
        decoder (nn.Module): Transformer decoder (or encoder if "all_encoder").
        global_motion_token (nn.Parameter): Latent distribution modeling token.
        skel_embedding (nn.Linear): Linear layer for skeleton embedding.
        final_layer (nn.Linear): Final linear layer for output.

    Transformer-based Motion Latent Diffusion VAE 为动作预测、全身运动建模服务，所以结构上有很多定制化的地方。

    AutoMldVae 的目标其实不是纯 VAE，而是一个 适配动作预测 / diffusion denoiser 的 “latent motion encoder-decoder”
    它结合：

    ✔ Transformer 的序列编码优势
    ✔ VAE 的潜空间建模（distribution + reparameterization）
    ✔ Skip-connection transformer（便于跨层特征传递）
    ✔ Wavelet patching（减少序列长度，提取 multi-scale 运动特征）
    ✔ Latent memory token（专门用于表示动作的高层语义）

    最终构建出一个 动作序列 → 低维潜空间 → 动作序列 的结构。
    """

    def __init__(
            self,
            nfeats: int,  # 每帧姿态特征维度
            latent_dim: list = [1, 256],  # [latent token 数, token 特征维度]
            h_dim: int = 512,  # Transformer hidden dim
            ff_size: int = 1024,  # FFN expansion
            num_layers: int = 9,  # Transformer 层数
            num_heads: int = 4,
            dropout: float = 0.1,
            arch: str = "all_encoder",  # 或 "encoder_decoder"
            normalize_before: bool = False,
            activation: str = "gelu",
            position_embedding: str = "learned",
            use_patcher=False,  # 是否使用 wavelet patch 技术
            patch_size=1,
            patch_method="haar",
    ) -> None:

        super().__init__()

        # Initialize latent space dimensions
        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.h_dim = h_dim
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = False
        self.pe_type = "mld"

        # Wavelet patching configuration
        self.use_patcher = use_patcher
        self.patch_size = patch_size
        self.patch_method = patch_method

        """
        如果 use_patcher=True：
        - encode 前 → motion 切 patch
        - decode 后 → patch 逆变换回来
        """
        if self.use_patcher:
            # Initialize wavelet transformation modules
            self.wavelet_transform = Patcher1D(patch_size, patch_method)
            self.inverse_wavelet_transform = UnPatcher1D(patch_size, patch_method)

        # Positional encoding for encoder and decoder
        # 位置编码
        self.query_pos_encoder = build_position_encoding(self.h_dim,
                                                         position_embedding=position_embedding)
        self.query_pos_decoder = build_position_encoding(self.h_dim,
                                                         position_embedding=position_embedding)

        # Transformer encoder setup
        encoder_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.h_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer,
                                              num_layers,
                                              encoder_norm)
        self.encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)

        # Transformer decoder setup based on architecture type
        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer,
                                                  num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.h_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        self.decoder_latent_proj = nn.Linear(self.latent_dim, self.h_dim)  # # latent_dim -> h_dim

        # 用于建模 latent distribution
        # 2*latent_size 是为了 split mu 和 logvar
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.h_dim))

        # Skeleton embedding and final output layer
        self.skel_embedding = nn.Linear(input_feats, self.h_dim)
        self.final_layer = nn.Linear(self.h_dim, output_feats)

        # Latent distribution scaling parameters
        self.register_buffer('latent_mean', torch.tensor(0))
        self.register_buffer('latent_std', torch.tensor(1))

    def encode(
            self,
            future_motion,
            history_motion,
            scale_latent: bool = False,
    ) -> Tuple[Tensor, Distribution]:
        """
        Encode future motion into a latent distribution.

        Args:
            future_motion (Tensor): Future motion data of shape [bs, nfuture, nfeats].
            history_motion (Tensor): History motion data of shape [bs, nhistory, nfeats].
            scale_latent (bool): Whether to scale the latent vector.

        Returns:
            Tuple[Tensor, Distribution]: Latent vector and its distribution.
        """
        bs, nfuture, nfeats = future_motion.shape
        nhistory = history_motion.shape[1]

        # Concatenate history and future motion
        x = torch.cat((history_motion, future_motion), dim=1)  # [bs, H+F, nfeats]
        # Embed each human poses into latent vectors
        # breakpoint()

        if self.use_patcher:
            # Apply wavelet transformation if enabled
            x = self.wavelet_transform(x)

        # Embed skeleton features
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, h_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # Add positional encoding and pass through the encoder
        xseq = torch.cat((dist, x), 0)
        xseq = self.query_pos_encoder(xseq)

        dist = self.encoder(xseq)[:dist.shape[0]]  # [2*latent_size, bs, h_dim]
        dist = self.encoder_latent_proj(dist)  # [2*latent_size, bs, latent_dim]

        # Split latent distribution into mean and log variance
        mu = dist[0:self.latent_size, ...]
        logvar = dist[self.latent_size:, ...]
        logvar = torch.clamp(logvar, min=-10, max=10)  # avoid numerical issues, otherwise denoiser rollout can break
        # if torch.isnan(mu).any() or torch.isinf(mu).any() or torch.isnan(logvar).any() or torch.isinf(logvar).any():
        #     pdb.set_trace()

        # Resample latent vector
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()

        # only used during denoiser training
        if scale_latent:
            latent = latent / self.latent_std

        return latent, dist

    def decode(
            self,
            z: Tensor,
            history_motion,
            nfuture,
            scale_latent: bool = False,
    ):
        """
        Decode a latent vector into future motion.

        Args:
            z (Tensor): Latent vector of shape [latent_size, bs, latent_dim].
            history_motion (Tensor): History motion data of shape [bs, nhistory, nfeats].
            nfuture (int): Number of future frames to predict.
            scale_latent (bool): Whether to scale the latent vector.

        Returns:
            Tensor: Decoded future motion of shape [bs, nfuture, nfeats].
        """
        # nfuture = 8
        bs = history_motion.shape[0]

        device = next(self.parameters()).device

        if scale_latent:  # only used during denoiser training
            z = z * self.latent_std
        z = self.decoder_latent_proj(z).to(device)  # [latent_size, bs, latent_dim] => [latent_size, bs, h_dim]

        # Initialize query tokens and embed history motion
        queries = torch.zeros(nfuture, bs, self.h_dim, device=z.device).to(device)
        history_embedding = self.skel_embedding(history_motion).permute(1, 0, 2).to(device)  # [nhistory, bs, h_dim]

        # Pass through the decoder based on architecture type
        if self.arch == "all_encoder":
            xseq = torch.cat((z, history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq)[-nfuture:]

        elif self.arch == "encoder_decoder":
            xseq = torch.cat((history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(
                tgt=xseq,
                memory=z,
            )
            # print('output:', output.shape)
            output = output[-nfuture:]

        # Final output layer
        output = self.final_layer(output)  # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)

        if self.use_patcher:
            # Apply inverse wavelet transformation if enabled
            output = self.inverse_wavelet_transform(output)

        return feats

    def forward(self, z, history_motion):
        """
        Forward pass for the model.

        Args:
            z (Tensor): Latent vector.
            history_motion (Tensor): History motion data.

        Returns:
            Tensor: Decoded future motion.
        """
        return self.decode(z, history_motion)
