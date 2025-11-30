"""模块说明：
工具函数：在 numpy 与 torch 张量之间递归转换（支持 list/tuple/dict 嵌套），
以及一个装饰器用于将接受/返回 torch.Tensor 的函数包装为 numpy 输入/输出 使用。
"""

import numpy as np
import torch
from typing import Union, Dict, Tuple, Any

to_torch = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
to_numpy = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def tree_to_torch(x: Union[np.ndarray, torch.Tensor, Dict[str, Any], Tuple, list]) -> Union[torch.Tensor, Dict[str, Any], Tuple, list]:
    """递归将 numpy ndarray 转为 torch.Tensor；保持 dict/list/tuple 结构不变。

    参数:
        x: 支持 np.ndarray / torch.Tensor / dict / list / tuple

    返回:
        对应的 torch.Tensor 或嵌套结构，其中所有 ndarray 都转为 torch.Tensor
    """
    if isinstance(x, (list, tuple)):
        return type(x)(tree_to_torch(i) for i in x)
    elif isinstance(x, dict):
        return {k: tree_to_torch(v) for k, v in x.items()}
    else:
        return to_torch(x)


def tree_to_numpy(x: Union[torch.Tensor, Dict[str, Any], Tuple, list]) -> Union[np.ndarray, Dict[str, Any], Tuple, list]:
    """递归将 torch.Tensor 转为 numpy ndarray；保持 dict/list/tuple 结构不变。

    参数:
        x: 支持 torch.Tensor / dict / list / tuple

    返回:
        对应的 numpy ndarray 或嵌套结构，其中所有 torch.Tensor 都转为 numpy
    """
    if isinstance(x, (list, tuple)):
        return type(x)(tree_to_numpy(i) for i in x)
    elif isinstance(x, dict):
        return {k: tree_to_numpy(v) for k, v in x.items()}
    else:
        return to_numpy(x)


def wrap_torch_to_numpy(func):
    """
    装饰器：把一个以 torch.Tensor 为输入输出的函数包装为接受 numpy 的函数并返回 numpy。
    使用场景：当外部代码基于 numpy 数据时，能透明调用内部的 torch 实现。
    """

    def wrapper(*args, **kwargs):
        # 将输入参数（含嵌套）转换为 torch.Tensor
        torch_args = tree_to_torch(args)
        torch_kwargs: Dict[str, Any] = tree_to_torch(kwargs)  # type: ignore
        result = func(*torch_args, **torch_kwargs)
        # 将结果（可能是 tensor 或嵌套结构）转换回 numpy
        return tree_to_numpy(result)

    return wrapper
