#!/usr/bin/env python3
"""
RobotMDAR CLI - Simple entry point using Hydra framework
"""

import os

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
    print("-=-" * 30)
    print(OmegaConf.to_yaml(cfg))
    print("=-=" * 30)

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

    task = cfg.task
    task_path = task_modules.get(task)

    if task_path is None:
        raise ValueError(
            f"Unknown task: {task}\n"
            f"Available tasks: {', '.join(task_modules.keys())}"
        )

    module = import_module(f"robotmdar.{task_path}")
    module.main(cfg)


if __name__ == "__main__":
    main()
