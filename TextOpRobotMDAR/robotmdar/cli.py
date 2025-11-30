#!/usr/bin/env python3
"""
RobotMDAR CLI - 简单的入口脚本，使用 Hydra 管理配置

此模块为 RobotMDAR 项目的命令行入口。使用 Hydra 加载配置并根据
cfg.task 映射选择对应的任务模块并执行其 main 函数。

环境变量:
    HYDRA_FULL_ERROR: 设置为 "1" 以在出现错误时显示完整的回溯信息。
"""
import os

# 设置环境变量以便在 Hydra 抛出异常时显示完整的错误信息（便于调试）
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

import sys
from pathlib import Path
from importlib import import_module

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_path="config",
    config_name="base",
    version_base="1.1",
)
def main(cfg: DictConfig):
    """
    主入口函数（由 Hydra 装饰器包裹）。

    Args:
        cfg (DictConfig): 从配置文件和命令行参数合并得到的配置对象。

    行为:
        - 打印当前加载的配置（以 YAML 格式输出）
        - 根据 cfg.task 在本地映射表中查找对应模块路径
        - 动态导入模块并调用其 main(cfg)
        - 如果 cfg.task 未知，则抛出 ValueError
    """
    # 打印分隔线和当前配置，便于日志查看
    print("-=-" * 30)
    print(OmegaConf.to_yaml(cfg))
    print("=-=" * 30)

    # 任务名称到模块路径的映射表（以字符串表示相对于 robotmdar 包的模块）
    task_modules = {
        "train-dar": "train.train_dar",
        "train-mvae": "train.train_mvae",
        "vis-mvae": "eval.vis_mvae",
        "vis-dar": "eval.vis_dar",
        "loop-dar": "eval.loop_dar",
        "freq-dar": "eval.freq_dar",
        "export-dar": "export.export_dar_onnx",
        "noise-opt": "opt.noise_opt",
    }

    # 从配置中读取要执行的任务名称
    task = cfg.task
    task_path = task_modules.get(task)

    if task_path is None:
        # 未知任务：抛出有说明的错误，列出可用任务
        raise ValueError(
            f"Unknown task: {task}\n"
            f"Available tasks: {', '.join(task_modules.keys())}"
        )

    # 通过 import_module 动态导入对应的任务模块并调用其 main 函数
    module = import_module(f"robotmdar.{task_path}")
    module.main(cfg)


if __name__ == "__main__":
    # 直接运行脚本时调用 main（Hydra 会处理工作目录和配置）
    main()
