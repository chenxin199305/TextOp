"""模块说明：
用于基于 MuJoCo 的简单可视化工具，加载模型并提供一个自动播放循环用于展示
预测/真实（pd/gt）qpos 序列。包含按键交互（翻帧、切 primitive、切批次、切换 pd/gt、播放控制等）。
"""

import atexit
from dataclasses import dataclass
from typing import Callable, Literal, Optional
import os
import numpy as np
import mujoco
import mujoco.viewer
import time


def mjc_load_everything(
        dt: float,
        keycb_fn: Optional[Callable[[int], None]] = None,
        humanoid_xml:
        str = "./description/robots/g1/g1_23dof_lock_wrist.xml"):
    """加载 MuJoCo 模型并启动被动 viewer，返回用于渲染单帧的回调 show_fn 与 viewer 对象。

    参数:
        dt (float): 模拟/渲染时间步长（秒），通常取 1/fps。
        keycb_fn (Optional[Callable[[int], None]]): 键盘回调函数（接收 keycode）。
        humanoid_xml (str): MuJoCo XML 文件路径。

    返回:
        Tuple[Callable[[np.ndarray, np.ndarray], None], mujoco.viewer.Viewer]:
            - show_fn(qpos, contact): 将 qpos 填入 mj_data 并执行前向计算与渲染同步。
            - viewer: MuJoCo viewer 对象（用于外部循环判断 is_running 等）。
    """
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)  # type: ignore
    # 创建数据容器（状态、关节等）
    mj_data = mujoco.MjData(mj_model)  # type: ignore
    # 设置模型时间步长
    mj_model.opt.timestep = dt

    # 启动被动 viewer（不在主线程自动循环渲染），传入键盘回调
    viewer = mujoco.viewer.launch_passive(mj_model,
                                          mj_data,
                                          key_callback=keycb_fn)

    # 摄像机基础视角设置（可根据需要调整）
    viewer.cam.lookat[:] = np.array([0, 0, 0.7])
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30  # 负值表示从上往下看

    def show_fn(qpos: np.ndarray, contact: np.ndarray):
        # 将外部提供的 qpos 写入 mj_data（注意 qpos 的格式需匹配模型）
        mj_data.qpos[:] = qpos
        # 将 quaternion 从 xyzw 转为 MuJoCo 期望的 wxyz 排序
        mj_data.qpos[3:7] = (mj_data.qpos[3:7])[[3, 0, 1, 2]]  # xyzw -> wxyz

        # 运行正向动力学以更新 body frame（渲染前必须调用）
        mujoco.mj_forward(mj_model, mj_data)  # type: ignore
        # 与 viewer 同步并提交渲染
        viewer.sync()
        ...

    # 注册退出时关闭 viewer 的处理，确保资源释放
    atexit.register(viewer.close)
    return show_fn, viewer


@dataclass
class VisState:
    """可视化运行时状态（用于交互控制与播放定位）。
    字段:
        midx (int): 全局批次索引（用于区分已加载的批次）。
        pidx (int): 当前 primitive 在 batch 内的索引（类似关节组/片段索引）。
        fidx (int): 当前帧索引（帧序号，包含历史+未来）。
        mode (Literal['pd','gt']): 当前模式，'pd' 为预测，'gt' 为真实。
        _trig (bool): 触发 flag，若 False 则暂停在当前帧（等待用户交互）。
        _autoplay (bool): 是否自动播放帧（True 表示按帧递增）。
    """
    midx: int = 0
    pidx: int = 0
    fidx: int = 0
    mode: Literal['pd', 'gt'] = 'pd'
    _trig: bool = True
    _autoplay: bool = True


def get_keycb_fn(vs: VisState):
    """生成一个键盘回调函数，回调内部会修改传入的 VisState。

    参数:
        vs (VisState): 运行时状态对象，回调会直接修改其字段。

    返回:
        Callable[[int], None]: 接收键码的回调函数 keycb_fn。
    """

    def keycb_fn(keycode: int):
        """键盘回调（内部）：响应常用按键并更新 vs。

        支持键（按键描述基于 ASCII / GLFW keycode）：
            P: 切换 pd/gt
            N/M: 下/上 一个批次（midx +/-）
            J/K: 下/上 一个 primitive（pidx +/-）
            左/右 箭头: fidx -/+ （翻帧）
            R: 重置 fidx=0,pidx=0
            空格: 切换 autoplay
            Q 或 Esc: 退出程序
        """
        # 使用 chr 对可打印键进行判断（注意特殊按键会超过 127）
        if chr(keycode) == 'P':
            if vs.mode == 'pd':
                vs.mode = 'gt'
            else:
                vs.mode = 'pd'
            print(f"Mode: {vs.mode}")
        elif chr(keycode) == 'N':
            vs.midx += 1
            print(f"Next batch: {vs.midx}")
        elif chr(keycode) == 'M':
            vs.midx -= 1
            print(f"Previous batch: {vs.midx}")
        elif chr(keycode) == 'J':
            vs.pidx += 1
            print(f"Next primitive: {vs.pidx}")
        elif chr(keycode) == 'K':
            vs.pidx -= 1
            print(f"Previous primitive: {vs.pidx}")
        elif keycode == 263:  # Left arrow
            vs.fidx -= 1
            print(f"Previous frame: {vs.fidx}")
        elif keycode == 262:  # Right arrow
            vs.fidx += 1
            print(f"Next frame: {vs.fidx}")
        elif chr(keycode) == 'R':
            vs.fidx = 0
            vs.pidx = 0
            print(f"Reset to fidx=0, pidx=0")
        elif chr(keycode) == ' ':
            vs._autoplay = not vs._autoplay
            print(f"Autoplay: {vs._autoplay}")
        elif keycode == 256 or chr(keycode) == 'Q':
            # 直接强制退出进程（用于调试/快速关闭）
            print("Esc")
            os._exit(0)
        else:
            # 对未知按键做简单输出，便于调试
            print(
                f"Unknown key: {keycode} ({chr(keycode) if keycode < 128 else 'special'})"
            )

        # 每次按键后触发一次更新（以便外层循环处理）
        vs._trig = True

    return keycb_fn


def mjc_autoloop_mdar(
        vs: VisState,
        fps: float,
        num_primitive: int,
        future_len: int,
        history_len: int,
        motion_buff: dict,
        add_batch: Callable,
        keycb_fn: Callable,
):
    """主循环：在 viewer 上自动/交互地播放 motion_buff 中的 qpos 序列。

    参数:
        vs (VisState): 运行时可视化状态。
        fps (float): 帧率（用于 sleep 与 mj 模型时间步长）。
        num_primitive (int): 每个批次包含的 primitive 数量。
        future_len (int): 未来帧长度（用于界定 fidx 的上限）。
        history_len (int): 历史帧长度（用于界定 fidx 的下限）。
        motion_buff (dict): 缓冲，包含 'pd' 和 'gt' 两种模式，每个为已追加的批次列表。
                            每个批次项为 per-primitive 的 (qpos_array, contact_array) 列表。
        add_batch (Callable): 无参回调，必要时向 motion_buff 中追加新批次。
        keycb_fn (Callable): 键盘回调函数，传入给 mujoco viewer。
    """
    show_fn, viewer = mjc_load_everything(dt=1 / fps, keycb_fn=keycb_fn)

    while viewer.is_running():
        if vs._trig:
            # 计算当前 midx 映射到 motion_buff 的索引：按 num_primitive 分组
            batch_index = vs.midx // num_primitive
            batch_offset = vs.midx % num_primitive
            # 若请求的批次未加载，调用 add_batch 来补充
            if batch_index >= len(motion_buff[vs.mode]):
                add_batch()

            # 处理 primitive 索引越界（前进到下一个 batch 或回退到上一个 batch）
            if vs.pidx >= num_primitive:
                vs.fidx = 0
                vs.pidx = 0
                if not vs._autoplay:
                    vs.midx += 1
                continue
            elif vs.pidx < 0:
                vs.fidx = 0
                vs.pidx = num_primitive - 1
                if not vs._autoplay:
                    vs.midx -= 1
                continue

            # 处理帧索引越界（超过历史+未来或小于 0 时切换 primitive）
            if vs.fidx >= future_len + history_len:
                vs.fidx = history_len
                vs.pidx += 1
                continue
            elif vs.fidx < 0:
                vs.fidx = future_len - 1
                vs.pidx -= 1
                continue

            # 从缓冲中读取当前 primitive 的 (qpos, contact) 序列
            motion_prim = motion_buff[vs.mode][batch_index][vs.pidx]
            qpos, contact = motion_prim
            # 取出当前 batch_offset 中的帧（batch_offset 用于在同一批次内索引不同样本）
            qpos_curr = qpos[batch_offset][vs.fidx]
            contact_curr = contact[batch_offset][vs.fidx]
            # 渲染当前帧
            show_fn(qpos_curr, contact_curr)

            # 根据 autoplay 决定是否自动推进帧或等待下一次触发
            if not vs._autoplay:
                vs._trig = False
            else:
                vs.fidx += 1
            # 打印状态便于调试
            print(vs)
        # 控制循环频率，避免忙等
        time.sleep(1 / fps)
