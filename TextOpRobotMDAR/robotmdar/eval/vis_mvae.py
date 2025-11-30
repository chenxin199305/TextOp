"""模块说明：
该脚本用于可视化 MVAE 的预测与真实运动（future vs gt）。
通过从验证数据集中取批次，使用 VAE 的 encode/decode 生成预测，
然后把预测/真实的 feature 重建为 qpos 并推入可视化缓冲队列供渲染循环使用。
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Literal, Callable, List, Tuple
from dataclasses import dataclass
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from robotmdar.dtype import seed, logger
from robotmdar.dtype.motion import motion_dict_to_qpos, QPos, get_zero_abs_pose, motion_dict_to_abs_pose, FeatureVersion, get_blended_feature, transform_feature_to_world, dict_concat, dict_to_tensor
from robotmdar.dtype.device import tree_to_numpy
from robotmdar.dtype.abc import Dataset, VAE, Optimizer, Distribution
from robotmdar.train.manager import MVAEManager
from robotmdar.dtype.vis_mjc import mjc_load_everything, VisState, get_keycb_fn, mjc_autoloop_mdar


def add_batch_fn(
        motion_buff,
        val_dataiter,
        vae,
        val_data,
        num_primitive,
        future_len,
        history_len,
        cfg
):
    """返回一个用于填充 motion_buff 的闭包函数 add_batch。

    参数:
        motion_buff (dict): 全局缓冲，包含 'pd' 和 'gt' 列表，用于存放预测和真实 qpos。
        val_dataiter (iterator): 验证集的迭代器，next(val_dataiter) 返回一个批次。
        vae (VAE): 已加载并设置为 eval 模式的 VAE 模型，提供 encode/decode 接口。
        val_data (Dataset): 验证集 Dataset 实例，提供 reconstruct_motion 等工具函数。
        num_primitive (int): 场景中 primitive 的数量（批次中独立 motion 条目数）。
        future_len (int): 未来帧长度（预测的帧数）。
        history_len (int): 历史帧长度（作为条件输入的帧数）。
        cfg (DictConfig): 配置对象，包含 device、batch_size 等字段。

    返回:
        function: 无参闭包 add_batch()，调用后会将一批 pd 和 gt qpos 数据追加到 motion_buff。
    """

    def add_batch():
        """从验证集取一批数据，使用 VAE 生成预测，并将预测/真实的 qpos 追加到 motion_buff。

        操作:
            1. 从 val_dataiter 获取一批 motion/cond，按 primitive 处理。
            2. 切分历史和未来帧，调用 vae.encode/vae.decode 得到预测未来。
            3. 使用 val_data.reconstruct_motion 将 feature 重建为字典形式。
            4. 将 feature dict 转换为 qpos（tree_to_numpy(motion_dict_to_qpos(...))）。
            5. 将本批次每个 primitive 的 pd 和 gt qpos 列表追加到 motion_buff['pd'] 与 motion_buff['gt']。

        注意:
            - 该函数直接修改外部的 motion_buff（通过闭包捕获）。
            - 不返回值，异常会向上抛出以便外部处理。
        """
        # 初始化用于重建绝对位姿（abs pose）的占位张量，形状为 (batch_size, ...)
        pd_abs_pose = get_zero_abs_pose((cfg.data.batch_size,), device=cfg.device)
        gt_abs_pose = get_zero_abs_pose((cfg.data.batch_size,), device=cfg.device)

        # 从验证数据迭代器获取一批数据
        batch = next(val_dataiter)

        # 准备当前批次的预测/真实缓冲（按 primitive 列表）
        pd_buff: List[Tuple[QPos, torch.Tensor]] = []
        gt_buff: List[Tuple[QPos, torch.Tensor]] = []

        for pidx in range(num_primitive):
            # 每个 primitive 对应一个 motion, cond
            # motion 形状通常为 [batch, seq_len, feature_dim]
            motion, cond = batch[pidx]
            motion, cond = motion.to(cfg.device), cond.to(cfg.device)

            # 将序列切分为历史和未来部分
            future_motion_gt = motion[:, -future_len:, :]  # 未来 ground-truth
            history_motion = motion[:, :history_len, :]  # 历史输入

            # 使用 VAE 编码未来（GT）与历史 -> 得到潜变量与分布
            latent, dist = vae.encode(future_motion=future_motion_gt,
                                      history_motion=history_motion)

            # 使用潜变量与历史解码出未来预测
            future_motion_pred = vae.decode(latent,
                                            history_motion,
                                            nfuture=future_len)

            # 将 history + future_pred 拼接并重建为 feature dict（用于转成 qpos）
            # 注意：reconstruct_motion 返回的是包含关节旋转、位移等的字典结构
            future_motion_pred_dict = val_data.reconstruct_motion(
                torch.cat([history_motion, future_motion_pred], dim=1),
                abs_pose=pd_abs_pose,
                ret_fk=False)
            future_motion_gt_dict = val_data.reconstruct_motion(
                torch.cat([history_motion, future_motion_gt], dim=1),
                abs_pose=gt_abs_pose,
                ret_fk=False)

            # 通过 motion_dict_to_abs_pose 更新绝对位姿（通常取倒数第二帧作为参考 idx=-2）
            pd_abs_pose = motion_dict_to_abs_pose(future_motion_pred_dict, idx=-2)
            gt_abs_pose = motion_dict_to_abs_pose(future_motion_gt_dict, idx=-2)

            # 将 feature dict 转换为 qpos（关节角度等），并转为 numpy 以便可视化渲染使用
            pd_buff.append(tree_to_numpy(motion_dict_to_qpos(future_motion_pred_dict)))  # type: ignore
            gt_buff.append(tree_to_numpy(motion_dict_to_qpos(future_motion_gt_dict)))  # type: ignore

        # 将本批次按 primitive 的预测/真实结果追加到全局 motion 缓冲
        motion_buff['pd'].append(pd_buff)
        motion_buff['gt'].append(gt_buff)

    return add_batch


def main(cfg: DictConfig):
    """程序入口，初始化环境并启动可视化渲染循环。

    参数:
        cfg (DictConfig): 配置对象，包含以下常用字段：
            - seed: 随机种子
            - data.batch_size: 验证集 batch size
            - data.val: 验证集配置（用于 instantiate）
            - vae: VAE 模型配置（用于 instantiate）
            - train.manager: 训练管理器配置（用于 instantiate）
            - device: 运行设备（如 'cpu' 或 'cuda'）
            - data.num_primitive / data.future_len / data.history_len: 序列维度信息

    行为:
        1. 设置日志与随机种子。
        2. 实例化验证集与 VAE，并通过 MVAEManager 挂载模型。
        3. 构建 motion_buff 和 VisState，创建 add_batch 闭包。
        4. 调用 mjc_autoloop_mdar 开始 mujoco 自动渲染循环，传入缓冲和回调。
    """

    logger.set(cfg)
    seed.set(cfg.seed)

    # 实例化验证集并得到其迭代器
    val_data: Dataset = instantiate(cfg.data.val)
    val_dataiter = iter(val_data)

    # 实例化 VAE 并设置为评估模式（不启用 dropout/batchnorm 更新）
    vae: VAE = instantiate(cfg.vae)
    vae.eval()

    # 通过训练管理器把模型、优化器等挂载起来（此处只 hold 模型，第二个参数为 None）
    manager: MVAEManager = instantiate(cfg.train.manager)
    manager.hold_model(vae, None, val_data)

    # 从配置中读取 primitive 数量与序列长度信息
    num_primitive = cfg.data.num_primitive
    future_len = cfg.data.future_len
    history_len = cfg.data.history_len

    # 初始化用于可视化渲染的缓冲区（保存预测/真实 qpos 列表）
    motion_buff = {
        'pd': [],
        'gt': [],
    }

    # VisState 存储可视化的运行时状态（相机、播放状态等）
    vs = VisState()

    # 生成用于不断向缓冲里追加批次的闭包函数
    add_batch = add_batch_fn(motion_buff,
                             val_dataiter,
                             vae,
                             val_data,
                             num_primitive,
                             future_len,
                             history_len,
                             cfg)

    # 获取键盘回调函数（用于交互控制）
    keycb_fn = get_keycb_fn(vs)

    fps = val_data.fps

    # 启动 mujoco 的自动循环渲染函数，传入所有必要的参数与回调
    mjc_autoloop_mdar(vs,
                      fps,
                      num_primitive,
                      future_len,
                      history_len,
                      motion_buff,
                      add_batch,
                      keycb_fn)
