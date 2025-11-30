import os
import argparse
import numpy as np
import joblib
import json
import yaml
from tqdm import tqdm
from pathlib import Path

# 脚本元信息：用于将 BABEL 注释与 AMASS 动作数据匹配并合并为训练/验证集合
DATASET_NAME = "Babel-teach X AMASS Robot"
BABEL_SPLIT = ['train', 'val']
FPS = 50  # 帧率，BABEL / AMASS 数据的采样帧率（帧/s）

# 终端颜色，便于调试输出识别
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
BLUE = "\033[34m"
BOLD = "\033[1m"
RESET = "\033[0m"


def process_babel_json(babel_json_path):
    """
    解析单个 BABEL JSON 文件（train.json / val.json）。
    主要工作：
    - 读取 JSON 中每个样本条目
    - 校验并规范化 feat_p 路径（.npz -> .pkl，转为相对路径）
    - 优先使用 frame_ann；若缺失则将 seq_ann 转为 frame_ann（将整段时间作为一个标签）
    - 将每条 label 归纳为 (start_t, end_t, proc_label, act_cat)
    返回：dict mapping feat_p -> { babel_sid, frame_ann(list), duration }
    """
    with open(babel_json_path, 'r') as f:
        babel_json = json.load(f)
    result = {}

    for k, v in babel_json.items():
        try:
            babel_id = int(k)
            duration = float(v.get("dur", -1))
            assert duration >= 0

            feat_p = v.get("feat_p", None)

            assert feat_p and feat_p.endswith(".npz"), f"Invalid feat_p for {babel_id}"
            feat_p = feat_p.replace(".npz", ".pkl")

            # 将 feat 路径规整为相对路径（相对于第一个路径组件）
            feat_p = Path(feat_p).relative_to(feat_p.split("/")[0])  # relative to the first component
            feat_p = str(feat_p.as_posix())  # 规范化路径分隔符为 '/'

            frame_ann_raw = v.get("frame_ann", None)
            has_frame_ann = frame_ann_raw is not None and isinstance(frame_ann_raw, dict)

            if not has_frame_ann:
                # 如果缺少逐帧标注，则使用 seq_ann（序列标注）并把整段时间当作一个标签
                seq_ann = v.get("seq_ann", None)
                assert seq_ann and isinstance(seq_ann, dict) and "labels" in seq_ann, f"Missing seq_ann for {babel_id}"
                seq_labels = seq_ann["labels"]
                for label in seq_labels:
                    label["start_t"] = 0
                    label["end_t"] = duration
                frame_ann_raw = {"labels": seq_labels}

            frame_ann_list = []
            for label in frame_ann_raw["labels"]:
                start_t = label.get("start_t")
                end_t = label.get("end_t")
                proc_label = label.get("proc_label")
                act_cat = label.get("act_cat")

                # 基本完整性校验：时间区间、处理标签与动作类别
                assert start_t is not None and end_t is not None, f"Missing time range, got {start_t=}, {end_t=}"
                assert proc_label, f"Missing proc_label for {babel_id}"
                assert act_cat and isinstance(act_cat, list), f"Missing act_cat for {babel_id}"

                frame_ann_list.append((start_t, end_t, proc_label, act_cat))

            result[feat_p] = {
                "babel_sid": babel_id,
                "frame_ann": frame_ann_list,
                "duration": duration
            }

        except AssertionError as e:
            # 出现断言错误时打印信息并跳过该条目（保留 breakpoint 方便调试）
            print(f"Skipping {k} due to error: {e}, where value is {v}")
            breakpoint()  # Debugging breakpoint
            continue

    return result


def load_babel(babel_dir):
    """
    加载 BABEL 数据集目录下的 train.json 和 val.json 文件并解析。
    返回一个 dict: {'train': {...}, 'val': {...}}，内部为 process_babel_json 的结果。
    """
    print(f"Loading BABEL dataset from: {babel_dir}")

    babel_data = {}

    for split in BABEL_SPLIT:
        json_path = os.path.join(babel_dir, f"{split}.json")
        babel_data[split] = process_babel_json(json_path)

    return babel_data


def load_amass(amass_dir):
    """
    遍历 AMASS 目录，加载所有 .pkl 文件（joblib 格式），并收集 motion 信息。
    规范化每个 motion 的 fps 并将相对路径截断为最后 4 个路径组件以便匹配 BABEL 的 feat_p。
    返回一个 dict: rel_path -> motion_dict
    """
    print(f"Loading AMASS motion data from: {amass_dir}")
    all_motion_files = list(Path(amass_dir).rglob("*.pkl"))
    print(f"Found {len(all_motion_files)} motion files")

    amass_data = {}

    for motion_file in tqdm(all_motion_files):
        try:
            data = joblib.load(motion_file)
            for k, motion in data.items():
                motion['fps'] = 50
                assert abs(motion['fps'] - FPS) < 1e-5, f"FPS mismatch for {motion_file}: {motion['fps']} != {FPS}"
                rel_path = motion_file.relative_to(amass_dir).as_posix()
                # 仅保留路径的最后 4 个组件以匹配 BABEL 中的 feat_p 约定
                rel_path = "/".join(rel_path.split("/")[-4:])
                amass_data[rel_path] = motion
        except Exception as e:
            # 若某个文件加载失败则打印错误并跳过，不中断整个流程
            print(f"Error loading {motion_file}: {e}")
            continue

    return amass_data


def merge_datasets(amass_data, babel_data_all, output_dir, custom_exclusions):
    """
    合并 BABEL 注释和 AMASS 动作数据：
    - 对每个 split（train/val）遍历 BABEL 条目，查找对应的 amass_data（基于 feat_p）
    - 过滤掉自定义排除项、缺失 motion、或过短的 motion
    - 为每条合并项计算 length（帧数）并保存 merged pkl 与统计信息（yaml）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 数据汇总和存储
    merged_data = {}
    for split in BABEL_SPLIT:  # "train" and "val" splits
        merged = []
        babel_data = babel_data_all[split]
        for feat_p, babel_entry in tqdm(babel_data.items(), desc=f"Merging {split}", disable=True):
            # feat_p 示例: "BMLrub/BioMotionLab_NTroje/rub055/0020_lifting_heavy2_poses.npz" 标识动作文件路径

            # breakpoint()
            motion = amass_data.get(feat_p, None)

            # print(f"split = {split}, feat_p = {feat_p}, motion = {motion}")

            # 自定义排除：如果路径包含这些字符串则跳过（例如过难或不可行的动作）
            if any(content in feat_p for content in custom_exclusions):
                print(f"{YELLOW}Skipping {feat_p} in {split} due to excluded content{RESET}")
                continue  # exclude hard and infeasible motions

            if motion is None:
                # 没有匹配到 AMASS 数据则跳过
                print(f"{RED}Skipping {feat_p} in {split} due to missing{RESET}")
                continue  # no matching AMASS motion

            # 过滤过短的 motion 数据（帧数阈值为 67）
            if motion['root_trans_offset'].shape[0] <= 67:
                print(f"{BLUE}Skipping {feat_p} in {split} due to short motion data{RESET}")
                continue  # no matching AMASS motion

            if 'motion_len' not in motion.keys():
                motion['motion_len'] = motion['root_trans_offset'].shape[0]

            merged.append({
                "feat_p": feat_p,
                "babel_sid": babel_entry["babel_sid"],
                "frame_ann": babel_entry["frame_ann"],
                "duration": babel_entry["duration"],
                "length": int(np.ceil(babel_entry["duration"] * FPS)),
                "motion": motion
            })

        merged_data[split] = merged
        output_file = os.path.join(output_dir, f"{split}.pkl")
        joblib.dump(merged, output_file)
        print(f"{GREEN}{BOLD}Saved {len(merged)} motions to {output_file}{RESET}")

    # 汇总统计信息并写入 statistics.yaml
    stats = {
        'dataset name': DATASET_NAME,
        'fps': FPS,
        'babel dir': BABEL_DIR,
        'amass robot dir': AMASS_ROBOT_DIR,
        'output dir': OUTPUT_DIR,
        'babel count': {k: len(v) for k, v in babel_data_all.items()},
        'amass count': len(amass_data),
        'merged count': {k: len(v) for k, v in merged_data.items() if k in BABEL_SPLIT},
        # 计算总时长（dof 长度之和 / fps）
        'total duration': sum(
            sum(len(entry['motion']['dof']) for entry in merged_data[split])
            for split in BABEL_SPLIT
        ) / FPS
    }
    stats_path = os.path.join(output_dir, "statistics.yaml")
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f)
    print(f"Statistics written to {stats_path}")


def main():
    # 核心执行流程：加载 babel、加载 amass、设置排除规则并进行合并
    babel_data = load_babel(BABEL_DIR)
    amass_data = load_amass(AMASS_ROBOT_DIR)
    output_dir = OUTPUT_DIR

    """
    Jason 2025-11-29:
    custom_exclusions 标识一些比较难的动作，然后进行过滤掉，避免在训练集中出现
    这些动作包括翻滚、爬行、上下楼梯等，主要是因为这些动作比较复杂，可能会影响模型的训练效果
    """
    custom_exclusions = ["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]

    merge_datasets(amass_data, babel_data, output_dir, custom_exclusions)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument('--amass_robot', type=str, required=True)
    args.add_argument('--babel', type=str, required=True)
    parsed_args = args.parse_args()

    AMASS_ROBOT_DIR = parsed_args.amass_robot
    BABEL_DIR = parsed_args.babel
    OUTPUT_DIR = "./dataset/BABEL-AMASS-ROBOT-23dof-50fps-TEACH"

    main()
