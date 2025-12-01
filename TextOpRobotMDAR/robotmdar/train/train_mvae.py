"""
模块说明：
训练 Multi-Variate Autoencoder (MVAE) 的主脚本。

功能概述：
- 基于传入的 Hydra 配置 (cfg) 初始化日志与随机种子。
- 构建训练/验证数据集、VAE 模型、优化器与训练管理器(MVAEManager)。
- 在主循环中交替执行训练步与评估步，通过 manager 管理学习率、日志与检查点。
"""

import torch
import toolz

from omegaconf import DictConfig
from hydra.utils import instantiate

from robotmdar.dtype import seed, logger
from robotmdar.dtype.abc import Dataset, VAE, Optimizer
from robotmdar.train.manager import MVAEManager


def evaluate_distribution_match(normalized_data):
    """评估归一化后数据的统计特性是否接近标准正态分布。

    参数:
        normalized_data (torch.Tensor): 形状为 [N, D] 的张量，表示已归一化后的样本特征。

    返回:
        dict: 包含以下键：
            - "mean_error" (float): 各维度均值与 0 的平均绝对偏差。
            - "std_error" (float): 各维度标准差与 1 的平均绝对偏差。
            - "is_well_normalized" (bool): 简单布尔判定，若 mean_error < 0.1 且 std_error < 0.2 则为 True。
    """
    # 计算关键统计量
    actual_mean = normalized_data.mean(dim=0)  # 各特征均值
    actual_std = normalized_data.std(dim=0)  # 各特征标准差

    # 理想值对比
    perfect_mean = torch.zeros_like(actual_mean)  # 期望均值=0
    perfect_std = torch.ones_like(actual_std)  # 期望标准差=1

    # 计算偏差
    mean_error = (actual_mean - perfect_mean).abs().mean()
    std_error = (actual_std - perfect_std).abs().mean()

    return {
        "mean_error": mean_error.item(),
        "std_error": std_error.item(),
        "is_well_normalized": mean_error < 0.1 and std_error < 0.2
    }


def main(cfg: DictConfig):
    """程序入口：根据 cfg 初始化并运行训练与评估循环。

    参数:
        cfg (DictConfig): Hydra 配置对象，常用字段包括：
            - seed: 随机种子
            - data.train / data.val: 数据集配置（用于 instantiate）
            - vae: VAE 模型配置（用于 instantiate）
            - train.manager: MVAEManager 配置（用于 instantiate）
            - train.opt: 优化器参数（传递给 torch.optim.Adam）
            - device: 运行设备（如 "cpu" 或 "cuda"）
            - data.batch_size / data.num_primitive / data.future_len / data.history_len: 数据相关超参

    行为:
        - 初始化日志与随机种子
        - 构造数据集、模型、优化器与 manager，并将模型/优化器交给 manager 管理
        - 在可迭代的 manager 上循环：执行训练步、必要时运行评估步，并通过 manager 记录/保存状态。
    """
    logger.set(cfg)
    seed.set(cfg.seed)

    train_data: Dataset = instantiate(cfg.data.train)
    val_data: Dataset = instantiate(cfg.data.val)

    vae: VAE = instantiate(cfg.vae)

    # 为 MVAE 创建 Adam 优化器
    optimizer: Optimizer = torch.optim.Adam(vae.parameters(), **cfg.train.opt)

    # 实例化训练管理器 MVAEManager（封装训练状态、日志、检查点、调度等）。
    manager: MVAEManager = instantiate(cfg.train.manager)

    # 把模型、优化器和训练数据交给 manager 管理
    manager.hold_model(vae, optimizer, train_data)

    train_dataiter = iter(train_data)
    val_dataiter = iter(val_data)

    num_primitive = cfg.data.num_primitive
    future_len = cfg.data.future_len
    history_len = cfg.data.history_len

    print(
        f"Training MVAE with {num_primitive} primitives, "
        f"each with future length {future_len} and history length {history_len}."
    )

    # all_normalized = []
    # for i in range(100):
    #     batch = next(train_dataiter)
    #     for pidx in range(num_primitive):
    #         all_normalized.append(batch[pidx][0])
    # normalized_data = torch.cat(all_normalized)
    # dist_result = evaluate_distribution_match(normalized_data)
    # print(dist_result)
    # breakpoint()

    while manager:

        # --------------------------------------------------

        # Training Loop
        vae.train()
        batch = next(train_dataiter)

        prev_motion = None

        for pidx in range(num_primitive):
            # manager.pre_step()：训练前的钩子，可能更新 step 计数、记录时间等。
            # motion, cond = batch[pidx]：从 batch 中取得第 pidx 个 primitive 的数据
            # （motion 是运动序列张量，cond 是条件，例如文本 embedding）。
            # 把数据移动到目标设备（GPU/CPU）cfg.device。
            manager.pre_step()
            motion, cond = batch[pidx]
            motion, cond = motion.to(cfg.device), cond.to(cfg.device)

            # 从 motion 中分割出 ground-truth 的未来序列 future_motion_gt（最后 future_len 帧）
            # 和 ground-truth 历史 gt_history（前 history_len 帧）。
            future_motion_gt = motion[:, -future_len:, :]
            gt_history = motion[:, :history_len, :]

            # 使用统一的history选择函数
            # 使用 manager.choose_history 得到用于模型输入的历史 motion。
            # 该函数可以基于 gt_history 和 prev_motion（上一次 roll-out）
            # 决定是否使用 ground-truth history、融合历史或使用生成的历史等（实现策略由 manager 定义）。
            history_motion = manager.choose_history(gt_history,
                                                    prev_motion,
                                                    history_len)

            # Encode using VAE
            latent, dist = vae.encode(future_motion=future_motion_gt,
                                      history_motion=history_motion)

            future_motion_pred = vae.decode(latent,
                                            history_motion,
                                            nfuture=future_len)  # [B, F, D]

            loss_dict, extras = manager.calc_loss(future_motion_gt,
                                                  future_motion_pred,
                                                  dist,
                                                  history_motion=history_motion)
            loss = loss_dict["total"]

            # optimizer.zero_grad()：清零梯度。
            # loss.backward()：反向传播计算梯度。
            optimizer.zero_grad()
            loss.backward()

            # 遍历 denoiser 的参数检查是否存在 NaN 或 Inf 的梯度（防止梯度爆炸或数值问题）。如果检测到则 has_nan_grad = True。
            has_nan_grad = False
            for param in vae.parameters():
                if param.grad is not None:
                    # 检查 NaN 和 Inf
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True

            # 若没有 NaN/Inf 梯度：
            # manager.grad_clip(denoiser)：调用 manager 对梯度进行剪裁（比如 clip_by_norm）。
            # optimizer.step()：更新参数。
            if not has_nan_grad:
                manager.grad_clip(vae)
                optimizer.step()

            prev_motion = future_motion_pred.detach()

            # manager.post_step(...)：
            # 训练步后钩子，传入是否为 eval、损失字典（把 tensor detach 并转 CPU）、额外信息 extras（同样处理）。
            # manager 可能在此记录到日志、写入 TensorBoard、保存检查点、更新学习率调度等。
            manager.post_step(
                is_eval=False,
                loss_dict=toolz.valmap(lambda x: x.detach().cpu(), loss_dict),
                extras=toolz.valmap(lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x, extras),
            )

        # --------------------------------------------------

        # Evaluation Loop
        vae.eval()
        while manager.should_eval():
            batch = next(val_dataiter)
            for pidx in range(num_primitive):
                manager.pre_step(is_eval=True)
                motion, cond = batch[pidx]
                motion, cond = motion.to(cfg.device), cond.to(cfg.device)

                future_motion_gt = motion[:, -future_len:, :]
                history_motion = motion[:, :history_len, :]

                # Encode using VAE
                latent, dist = vae.encode(future_motion=future_motion_gt,
                                          history_motion=history_motion)

                future_motion_pred = vae.decode(latent,
                                                history_motion,
                                                nfuture=future_len)

                loss_dict, extras = manager.calc_loss(future_motion_gt,
                                                      future_motion_pred,
                                                      dist,
                                                      history_motion=history_motion)

                # manager.post_step(...)：
                # 训练步后钩子，传入是否为 eval、损失字典（把 tensor detach 并转 CPU）、额外信息 extras（同样处理）。
                # manager 可能在此记录到日志、写入 TensorBoard、保存检查点、更新学习率调度等。
                manager.post_step(
                    is_eval=True,
                    loss_dict=toolz.valmap(lambda x: x.detach().cpu(), loss_dict),
                    extras=toolz.valmap(lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x, extras),
                )

        # --------------------------------------------------
