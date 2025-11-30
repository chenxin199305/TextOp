"""模块说明：
封装 loguru 与标准 logging 的桥接，提供统一的日志初始化函数。
"""

import logging
from loguru import logger
import sys
import os
from pathlib import Path

from omegaconf import OmegaConf


class HydraLoggerBridge(logging.Handler):

    def emit(self, record):
        """将标准 logging 记录转发到 loguru 并保留原始调用栈信息。

        参数:
            record (logging.LogRecord): 标准 logging 生成的日志记录对象。

        行为:
            - 将 logging 的 level 转换为 loguru 的 level；
            - 查找原始调用帧以保证日志显示正确的文件/行号；
            - 使用 logger.opt 设置 depth 与 exception 信息后调用 loguru.log 输出。
        """
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth,
                   exception=record.exc_info).log(level, record.getMessage())


def set(cfg):
    """初始化实验日志系统并保存配置文件。

    参数:
        cfg: 包含 experiment_dir, expname 等字段的配置对象（通常为 OmegaConf / DictConfig）。

    返回:
        loguru.logger: 已配置的 loguru logger 实例。

    说明:
        - 在 cfg.experiment_dir 下创建 run.log 并将控制台日志也输出到 stdout；
        - 将 Hydra 的配置保存为 cfg.yaml 以便复现。
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(cfg.experiment_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    hydra_log_path = os.path.join(cfg.experiment_dir, "run.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    logger.info(f"Logging initialized for experiment: {cfg.expname}")
    logger.info(f"Experiment directory: {cfg.experiment_dir}")

    hydra_cfg_path = os.path.join(cfg.experiment_dir, "cfg.yaml")
    OmegaConf.save(cfg, hydra_cfg_path, resolve=True)

    return logger
