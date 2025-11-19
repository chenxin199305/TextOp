"""
Training script for the Diffusion-based Action Repetition (DAR) model.
This script sets up the training loop, including data loading, model instantiation,
loss calculation, and optimization.

Jason Q&A 2025-11-18:
Q: primitive 是什么概念，如何理解?
A: 在机器人运动生成领域，"primitive" 通常指的是基本的运动单元或动作片段。
    这些基本单元可以是简单的运动模式，如走路、跑步、跳跃等， 也可以是更复杂的动作组合。
    通过将复杂的运动分解为多个 primitive，模型可以更有效地学习和生成多样化的运动序列。
    例如，在训练一个机器人生成行走动作的模型时， 可以将行走动作分解为多个 primitive，
    每个 primitive 代表一个步态周期。
    这样，模型可以通过学习这些基本的步态单元， 然后将它们组合起来生成完整的行走动作。
    将长序列拆成多个短的片段 —— primitive。
    primitive = 一个独立的未来动作片段
Q: 为什么要 primitive？
A: 长动作序列不好训练：
    1）GPU memory 限制
    2）diffusion模型通常处理固定长度输入
    3）VAE/denoiser 推理一次处理整个序列太大
Q: primitive 可能是什么长度？
    history_len = 10 (历史动作 = 10帧)
    future_len = 20 (每段未来动作 = 20帧)
    num_primitive = 4
    那么总动作长度 = 10 + (4×20) = 90 帧

Q: B, H, T, D 分别代码什么含义
A: B, H, T, D 是典型的 动作序列 / 时序模型的张量维度命名
    | 符号    | 代表含义                             | 示例值               | 解释                 |
    | ----- | -------------------------------- | ----------------- | ------------------ |
    | **B** | Batch size                       | 16                | 一次训练喂多少个样本         |
    | **H** | History length                   | 10                | 历史帧数（已知动作序列长度）     |
    | **T** | Future length (primitive length) | 20                | 一个 primitive 的预测长度 |
    | **D** | Feature dimension                | 263 / 72 / 3N etc | 每一帧的人体姿态/动作特征维度    |

"""

import torch

from omegaconf import DictConfig
from hydra.utils import instantiate

from robotmdar.dtype import seed, logger
from robotmdar.dtype.abc import VAE, Dataset, Denoiser, Diffusion, Optimizer, SSampler
from robotmdar.train.manager import DARManager

USE_VAE = True


def main(cfg: DictConfig):
    seed.set(cfg.seed)
    logger.set(cfg)

    train_data: Dataset = instantiate(cfg.data.train)
    val_data: Dataset = instantiate(cfg.data.val)

    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)

    # 实例化一个调度采样器（负责在训练中采样时间步 t 与权重），schedule_sampler 通常包装了具体的 diffusion 对象。
    # diffusion 直接从 schedule_sampler 中取出，方便后面调用 diffusion 的采样/前向函数。
    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    # 为 denoiser 创建 AdamW 优化器，学习率由配置给出。
    optimizer: Optimizer = torch.optim.AdamW(denoiser.parameters(),
                                             lr=cfg.train.manager.learning_rate)

    # 实例化训练管理器 DARManager（封装训练状态、日志、检查点、学习率调度、评价控制等）。
    manager: DARManager = instantiate(cfg.train.manager)

    # 把模型、优化器和训练数据“交给” manager 管理
    # （hold_model 可能会做模型移动到设备、封装到 manager 内部的引用、创建检查点结构等）。
    manager.hold_model(vae, denoiser, optimizer, train_data)

    train_dataiter = iter(train_data)
    val_dataiter = iter(val_data)

    num_primitive: int = cfg.data.num_primitive
    future_len: int = cfg.data.future_len
    history_len: int = cfg.data.history_len

    print(
        f"Training DAR with {num_primitive} primitives, "
        f"each with future length {future_len} and history length {history_len}."
    )

    # Training loop following train_mvae.py approach
    # 以 manager 的真值（通常 while manager: 等价于 while not manager.should_stop()）
    # 控制训练循环，manager 负责知道何时终止训练（例如达到最大步数或收到早停信号）。
    while manager:

        # --------------------------------------------------

        # Training Loop
        denoiser.train()
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

            # Sample timesteps
            # 通过 schedule_sampler.sample 为每个样本采样扩散时间步 t（通常是整形向量）
            # 和对应的损失权重 weights（用于加权不同时间步的损失）。
            batch_size = motion.shape[0]
            t, weights = schedule_sampler.sample(batch_size, device=cfg.device)

            if USE_VAE:
                # Encode using VAE
                latent_gt, _ = vae.encode(future_motion=future_motion_gt,
                                          history_motion=history_motion)  # [T=1, B, D]   latent_gt: (1, 512, 128)
                x_start = latent_gt.permute(1, 0, 2)  # [B, T=1, D]

            else:
                latent_gt = None
                x_start = torch.cat((history_motion, future_motion_gt), dim=1)

            # Forward diffusion
            # 对 x_start 在时间步 t 上进行前向扩散（q_sample），
            # 给输入加噪（这里用随机 noise），以生成带噪的 x_t（去噪器的输入）。
            x_t = diffusion.q_sample(x_start=x_start,
                                     t=t,
                                     noise=torch.randn_like(x_start))

            # Denoise
            # 构建字典 y 作为条件（文本 embedding 与历史 motion），送入去噪网络。
            y = {
                'text_embedding': cond,  # cond is already the text_embedding tensor
                'history_motion_normalized': history_motion,
            }

            # denoiser(...) 返回对 x_start 的预测（x_start_pred），
            # 这里 timesteps 通过 diffusion._scale_timesteps(t)（将整形时间步映射到模型期望的浮点或缩放的表示）。
            x_start_pred = denoiser(x_t=x_t,
                                    timesteps=diffusion._scale_timesteps(t),
                                    y=y)  # [B, T=1, D]
            # breakpoint()
            if USE_VAE:
                latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]

                # Decode
                # 把 x_start_pred 转换维度得到 latent_pred，
                # 然后调用 vae.decode(latent_pred, history_motion, nfuture=...)，
                # 将潜在变量解码成未来 motion（可能是归一化的）。
                future_motion_pred = vae.decode(
                    latent_pred,
                    history_motion,
                    nfuture=future_len)  # [B, F, D], normalized
            else:
                latent_pred = None

                # 直接从 x_start_pred 中截取最后 future_len 步作为未来 motion 的预测。
                future_motion_pred = x_start_pred[:, -future_len:]

            # Calculate loss
            # 把 ground-truth 未来、预测未来、潜在 gt/pred、权重等传给 manager.calc_loss，
            # 由 manager 统一计算各种损失项并返回字典 loss_dict（例如重构损失、KL、对抗或 perceptual loss）。
            loss_dict, extras = manager.calc_loss(future_motion_gt,
                                                  future_motion_pred,
                                                  latent_gt,
                                                  None,
                                                  latent_pred,
                                                  weights,
                                                  history_motion=history_motion,  # dist=None for DAR
                                                  )
            loss = loss_dict['total']  # 取得总损失标量（用于反向传播）。

            # optimizer.zero_grad()：清零梯度。
            # loss.backward()：反向传播计算梯度。
            optimizer.zero_grad()
            loss.backward()

            # 遍历 denoiser 的参数检查是否存在 NaN 或 Inf 的梯度（防止梯度爆炸或数值问题）。如果检测到则 has_nan_grad = True。
            has_nan_grad = False
            for param in denoiser.parameters():
                if param.grad is not None:
                    # 检查 NaN 和 Inf
                    if torch.isnan(param.grad).any() \
                            or torch.isinf(param.grad).any():
                        has_nan_grad = True

            # 若没有 NaN/Inf 梯度：
            # manager.grad_clip(denoiser)：调用 manager 对梯度进行剪裁（比如 clip_by_norm）。
            # optimizer.step()：更新参数。
            if not has_nan_grad:
                manager.grad_clip(denoiser)
                optimizer.step()

            # 更新prev_motion，如果启用full sample则使用更高质量的采样
            if manager.should_use_full_sample():
                with torch.no_grad():
                    # 如果 manager 判断当前需要使用完整采样（should_use_full_sample()，
                    # 例如在某些步骤进行更慢但更准确的采样用于生成历史）：
                    # 使用完整的DDPM采样循环来生成更高质量的rollout history
                    sample_fn = diffusion.p_sample_loop
                    x_start_full = sample_fn(
                        denoiser,
                        x_start.shape,
                        clip_denoised=False,
                        model_kwargs={'y': y},  # Wrap y in the expected structure
                        skip_timesteps=0,
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                    )
                    # 确保x_start_full是tensor并转换维度
                    if isinstance(x_start_full, torch.Tensor):
                        latent_full = x_start_full.permute(1, 0,
                                                           2)  # [T=1, B, D]
                    else:
                        # 如果返回的是其他格式，直接使用原始预测
                        latent_full = latent_pred
                    future_motion_full = vae.decode(latent_full,
                                                    history_motion,
                                                    nfuture=future_len)
                    prev_motion = torch.cat(
                        [history_motion, future_motion_full], dim=1).detach()
            else:
                # 如果不使用 full-sample，直接用当前预测的 future_motion_pred 拼接历史作为 prev_motion（并 detach）。
                prev_motion = torch.cat([history_motion, future_motion_pred], dim=1).detach()

            # manager.post_step(...)：
            # 训练步后钩子，传入是否为 eval、损失字典（把 tensor detach 并转 CPU）、额外信息 extras（同样处理）。
            # manager 可能在此记录到日志、写入 TensorBoard、保存检查点、更新学习率调度等。
            manager.post_step(
                is_eval=False,
                loss_dict={k: v.detach().cpu() for k, v in loss_dict.items()},
                extras={k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in extras.items()},
            )

        # --------------------------------------------------

        # Evaluation Loop
        denoiser.eval()
        while manager.should_eval():
            batch = next(val_dataiter)

            # 验证每个 primitive 的处理（类似训练，但不反向传播）
            for pidx in range(num_primitive):

                # 对每个 primitive：
                # manager.pre_step(is_eval=True)：告知 manager 这是 evaluation 步（可能更新 eval-specific 状态）。
                # 拿到 motion/cond，移动设备，分割未来与历史。
                manager.pre_step(is_eval=True)
                motion, cond = batch[pidx]
                motion, cond = motion.to(cfg.device), cond.to(cfg.device)

                future_motion_gt = motion[:, -future_len:, :]
                history_motion = motion[:, :history_len, :]

                with torch.no_grad():
                    t, weights = schedule_sampler.sample(motion.shape[0],
                                                         device=cfg.device)

                    # 与训练相同的分支逻辑：如果使用 VAE，则编码并构造 x_start；否则拼接原始 motion。
                    if USE_VAE:
                        latent_gt, _ = vae.encode(
                            future_motion=future_motion_gt,
                            history_motion=history_motion)
                        # Forward diffusion
                        x_start = latent_gt.permute(1, 0, 2)  # [B, T=1, D]
                    else:
                        latent_gt = None
                        x_start = torch.cat((history_motion, future_motion_gt),
                                            dim=1)

                    # 对 x_start 做 q_sample 得到 x_t，构建条件 y，
                    # 并通过 denoiser 得到预测 x_start_pred（所有操作都在 no_grad 下）。
                    x_t = diffusion.q_sample(x_start=x_start,
                                             t=t,
                                             noise=torch.randn_like(x_start))

                    y = {
                        'text_embedding':
                            cond,  # cond is already the text_embedding tensor
                        'history_motion_normalized': history_motion,
                    }
                    x_start_pred = denoiser(
                        x_t=x_t, timesteps=diffusion._scale_timesteps(t), y=y)

                    if USE_VAE:
                        latent_pred = x_start_pred.permute(1, 0, 2)

                        future_motion_pred = vae.decode(latent_pred,
                                                        history_motion,
                                                        nfuture=future_len)

                    else:
                        latent_pred = None
                        future_motion_pred = x_start_pred[:, -future_len:]

                    loss_dict, extras = manager.calc_loss(
                        future_motion_gt,
                        future_motion_pred,
                        latent_gt,
                        None,
                        latent_pred,
                        weights,
                        history_motion=history_motion)

                manager.post_step(
                    is_eval=True,
                    loss_dict={
                        k: v.detach().cpu()
                        for k, v in loss_dict.items()
                    },
                    extras={
                        k:
                            v.detach().cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in extras.items()
                    })

        # --------------------------------------------------
