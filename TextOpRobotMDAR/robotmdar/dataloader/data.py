# python
"""
Final Clean High-Performance Data Loader for Robot Motion Primitive Dataset

This is a clean, well-structured implementation with:
1. Motion-first generation ensuring primitive continuity
2. Small, focused functions for better readability
3. 100% interface compatibility with original
"""
# 高性能数据加载器，用于机器人运动原语数据集（含中文注释）

from pathlib import Path
import numpy as np
import joblib
import yaml
from typing import Any, Tuple, Dict, List, Optional
import sys
import random
from omegaconf import DictConfig
from loguru import logger

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
from robotmdar.model.clip import load_and_freeze_clip, encode_text
from robotmdar.skeleton.robot import RobotSkeleton
from robotmdar.dtype.motion import MotionDict, motion_dict_to_feature, AbsolutePose, motion_feature_to_dict, MotionKeys, FeatureVersion
import json


class SkeletonPrimitiveDataset(data.IterableDataset):
    """
    Clean, high-performance SkeletonPrimitiveDataset with motion-first generation.
    Key features:
    - Motion-first generation ensuring primitive continuity
    - Small, focused functions for better readability
    - 100% interface compatibility with original
    """

    # 数据集类：按运动优先（motion-first）生成原语，保证原语间连续性

    # ===============================================================
    # Load & build dataset

    def __init__(
            self,
            robot_cfg: DictConfig,
            batch_size: int,
            nfeats: int,
            history_len: int,
            future_len: int,
            num_primitive: int,
            datadir: str,
            action_statistics_path: str,
            weighted_sample: bool = False,
            frame_weight: bool = False,
            use_weighted_meanstd: bool = False,
            split: str = 'train',
            device: str = 'cuda',
            **kwargs: Any
    ):
        super().__init__()

        # Store parameters
        self.batch_size = batch_size
        self.history_len = history_len
        self.future_len = future_len
        self.num_primitive = num_primitive

        self.nfeats = nfeats

        # segment_len: 每次采样的总帧数（历史 + 多个未来原语 * future_len + 1）
        self.segment_len = self.history_len + self.future_len * self.num_primitive + 1
        # context_len: 用于单个原语上下文的长度（历史 + 1 个 future）
        self.context_len = self.history_len + self.future_len

        self.weighted_sample = weighted_sample
        self.frame_weight = frame_weight
        self.action_statistics_path = action_statistics_path
        self.use_weighted_meanstd = use_weighted_meanstd

        self.datadir = Path(datadir)
        self.split = split
        self.device = "cpu"  # Keep embeddings on CPU initially  \# 初始将文本嵌入放在 CPU

        # Load and prepare data
        self._load_data()

        # Initialize skeleton and normalization
        self.skeleton = RobotSkeleton(device=self.device, cfg=robot_cfg)

        if self.weighted_sample and self.use_weighted_meanstd:
            self._load_weighted_meanstd()
        else:
            self._load_meanstd()

    def _load_data(self) -> None:
        """Load and prepare data efficiently"""
        logger.info(f" Loading {self.split} data...")
        self._load_statistics()

        # Load data files
        if self.split == 'none':
            return
        splits = ['train', 'val'] if self.split == 'all' else [self.split]
        all_data = []
        for split in splits:
            datapkl = self.datadir / f'{split}.pkl'
            assert datapkl.exists(), f"Data file {datapkl} does not exist"
            all_data.extend(joblib.load(datapkl))

        # Fix length labels and filter valid samples
        self.valid_indices = []
        for i, item in enumerate(all_data):
            # item['motion']['motion_len'] 存储原始序列帧长
            item['length'] = int(item['motion']['motion_len'])
            # 仅保留长度 >= segment_len 的样本
            if item['length'] >= self.segment_len:
                self.valid_indices.append(i)

        self.raw_data = all_data

        if self.weighted_sample:
            self._cal_sample_weight()

        logger.info(f" Found {len(self.valid_indices)} valid samples out of {len(self.raw_data)}")

        # Load text embeddings (缓存或计算)
        self._load_text_embeddings()

    def _cal_sample_weight(self):
        # 计算基于动作统计的序列与帧权重（用于加权采样）
        logger.info(f" ====================Use Weighted Sample====================")

        with open(self.action_statistics_path, 'r') as f:
            action_statistics = json.load(f)

        for data in self.raw_data:
            seq_weight = 0
            for seg in data['frame_ann']:
                seg_act_cat = seg[3]
                act_weights = 0
                for act_cat in seg_act_cat:
                    # 若该动作类别在统计中不存在则跳过
                    if act_cat not in action_statistics:
                        continue
                    else:
                        act_weights += action_statistics[act_cat]['weight']
                # 使用每个 segment 的持续时间乘以动作权重累加
                seq_weight += (seg[1] - seg[0]) * act_weights
            data['weight'] = seq_weight
            num_frames = data['length']

            frame_weights = []
            # 针对每个可选的起始帧计算帧级权重
            for frame_idx in range(0, num_frames - self.segment_len + 1):
                start_t = frame_idx / self.fps
                end_t = (frame_idx + self.segment_len - 1) / self.fps
                frame_weight = 0
                for seg in data['frame_ann']:
                    overlap_len = self._get_overlap([seg[0], seg[1]], [start_t, end_t])
                    if overlap_len > 0:
                        act_weights = 0
                        for act_cat in seg[3]:
                            if act_cat not in action_statistics:
                                continue
                            else:
                                act_weights += action_statistics[act_cat]['weight']
                        frame_weight += overlap_len * act_weights
                frame_weights.append(frame_weight)
            data['frame_weights'] = frame_weights

        babel_sum = sum([data['weight'] for data in self.raw_data])
        print('babel sum: ', babel_sum)
        samp_percent = 0.0
        print('samp percent: ', samp_percent)
        if babel_sum > 0:
            for data in self.raw_data:
                data['weight'] = data['weight'] / babel_sum * (1 - samp_percent)

        seq_weights = np.array([data['weight'] for data in self.raw_data])
        seq_weights = seq_weights / seq_weights.sum()
        self.seq_weights = seq_weights

        # self._statistic_sample_weight()
        # breakpoint()

    def _statistic_sample_weight(self):
        # 将样本权重写入文件用于分析（开发/调试辅助函数）
        import re
        act_weight = {}
        for i, data in enumerate(self.raw_data):
            act_weight[i] = data['weight']

        with open('data_sample_weight_statistics_norm.txt', 'w', encoding='utf-8') as f:
            for seg in act_weight.items():
                line = f"{seg[0]}\t{seg[1]}"
                f.write(line + '\n')

    def _load_statistics(self) -> None:
        """Load motion statistics"""
        statistics_yaml = self.datadir / 'statistics.yaml'
        with open(statistics_yaml, 'r') as f:
            self.statistics = yaml.safe_load(f)
        # fps 用于将帧索引转换为时间
        self.fps = self.statistics['fps']

    def _load_text_embeddings(self) -> None:
        """Load or compute text embeddings"""
        text_embedding_path = self.datadir / f'{self.split}_text_embed.pkl'
        if text_embedding_path.exists():
            logger.info(" Loading cached text embeddings...")
            # 从磁盘加载预计算的文本嵌入（CPU）
            self.text_embeddings_dict = torch.load(text_embedding_path, map_location="cpu")
        else:
            logger.info(" Computing text embeddings...")
            # 仅在必要时加载 CLIP 模型并计算所有唯一文本的嵌入
            clip_model = load_and_freeze_clip(
                clip_version='ViT-B/32', device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.text_embeddings_dict = self._compute_text_embeddings(self.raw_data, clip_model)
            torch.save(self.text_embeddings_dict, text_embedding_path)

    @staticmethod
    def _compute_text_embeddings(raw_data: List[Dict[str, Any]],
                                 clip_model: nn.Module,
                                 batch_size: int = 64) -> Dict[str, torch.Tensor]:
        """Compute text embeddings efficiently"""
        # 提取所有唯一文本标签
        all_texts = set()
        for item in raw_data:
            for ann in item['frame_ann']:
                all_texts.add(ann[2])

        uni_texts = list(all_texts)

        # Batch encode for efficiency
        embeddings_list = []
        for i in range(0, len(uni_texts), batch_size):
            batch_texts = uni_texts[i:i + batch_size]
            batch_embeddings = encode_text(clip_model, batch_texts)
            embeddings_list.append(batch_embeddings.detach().float())

        text_embeddings = torch.cat(embeddings_list, dim=0)

        # Create dictionary mapping text->embedding
        text_embeddings_dict = dict(zip(uni_texts, text_embeddings))
        # 空字符串对应全零向量（作为默认）
        text_embeddings_dict[''] = torch.zeros_like(text_embeddings[0])

        return text_embeddings_dict

    def _load_meanstd(self) -> None:
        """Load or compute mean/std for normalization"""
        meanstd_cache_path = self.datadir / 'meanstd.pkl'
        if meanstd_cache_path.exists():
            logger.info(f" Loading cached mean/std from {meanstd_cache_path}...")
            meanstd = torch.load(meanstd_cache_path, map_location="cpu")
        else:
            logger.info(f" Computing mean/std..")
            assert self.split == 'train', "Compute mean and std from 'train' set"

            # zjk: DART meanstd cal method
            meanstd = self._compute_meanstd()
            # meanstd = self._compute_meanstd_V2()

            torch.save(meanstd, meanstd_cache_path)
            logger.info(f" Saved mean/std to {meanstd_cache_path}")

        self.mean, self.std = meanstd

    def _load_weighted_meanstd(self) -> None:
        """Load or compute mean/std for weighted normalization"""
        meanstd_cache_path = self.datadir / 'weighted_meanstd.pkl'
        if meanstd_cache_path.exists():
            logger.info(f" Loading cached mean/std from {meanstd_cache_path}...")
            meanstd = torch.load(meanstd_cache_path, map_location="cpu")
        else:
            logger.info(f" Computing mean/std..")
            assert self.split == 'train', "Compute mean and std from 'train' set"

            meanstd = self._compute_meanstd()
            torch.save(meanstd, meanstd_cache_path)
            logger.info(f" Saved mean/std to {meanstd_cache_path}")

        self.mean, self.std = meanstd

    def _compute_meanstd(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std efficiently by sampling batches"""
        motion_sum = torch.zeros(self.nfeats)
        motion_square_sum = torch.zeros(self.nfeats)
        count = 0

        # Sample a subset for statistics, 控制采样量
        N = 10000 // self.batch_size + 1

        # fake a mean and std, so that we can call _generate_batch_optimized
        self.mean = torch.zeros(self.nfeats)
        self.std = torch.ones(self.nfeats)

        # Iterate sampled batches and累积求和、求平方和
        for i in tqdm(range(N)):
            batch_data = self._generate_batch_optimized(generator=torch.Generator().manual_seed(i))

            for primitive_idx in range(self.num_primitive):
                motion_features, _ = batch_data[primitive_idx]
                motion_sum += motion_features.sum(dim=(0, 1))
                motion_square_sum += motion_features.square().sum(dim=(0, 1))
                count += motion_features.shape[0] * motion_features.shape[1]

        mean = motion_sum / count
        std = (motion_square_sum / count - mean.square()).sqrt()
        return mean, std

    def _compute_meanstd_V2(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 备用的更精确但可能更慢的 mean/std 计算方法（将所有原语收集后一次性计算）
        all_mp_data = []
        for seq_data in self.raw_data:
            motion_data = seq_data['motion']
            num_frames = motion_data['root_trans_offset'].shape[0]
            primitive_data_list = []
            for start_frame in range(0, num_frames - self.context_len, self.future_len):
                end_frame = start_frame + self.context_len
                primitive_data_list.append(self._extract_single_primitive(seq_data, start_frame, end_frame)[0])

            primitive_dict = {}
            for key in MotionKeys:
                primitive_dict[key] = torch.cat([data[key] for data in primitive_data_list], dim=0)

            batch_start_idx = 0
            while batch_start_idx < len(primitive_dict['root_trans_offset']):
                batch_end_idx = min(batch_start_idx + self.batch_size, len(primitive_dict['root_trans_offset']))
                batch_primitive_dict = {key: primitive_dict[key][batch_start_idx:batch_end_idx] for key in MotionKeys}
                motion_tensor = motion_dict_to_feature(batch_primitive_dict)[0]
                all_mp_data.append(motion_tensor)
                batch_start_idx = batch_end_idx

        all_mp_data = torch.cat(all_mp_data, dim=0)
        tensor_mean = all_mp_data.mean(dim=[0, 1], keepdim=True)
        tensor_std = all_mp_data.std(dim=[0, 1], keepdim=True)
        return tensor_mean, tensor_std

    # ================================================================
    # Data reconstruction

    def normalize(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize features"""
        # 将 feat 标准化到 mean/std（支持不同设备）
        return (feat - self.mean.to(feat.device)) / self.std.to(feat.device)

    def denormalize(self, feat: torch.Tensor) -> torch.Tensor:
        """Denormalize features"""
        return feat * self.std.to(feat.device) + self.mean.to(feat.device)

    def reconstruct_motion(
            self,
            motion_feature: torch.Tensor,
            abs_pose: Optional[AbsolutePose] = None,
            need_denormalize: bool = True,
            ret_fk: bool = True,
            ret_fk_full: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct motion from features"""
        if need_denormalize:
            motion_feature = self.denormalize(motion_feature)

        # 根据当前导入的 motion_feature_to_dict 版本选择调用方式
        if motion_feature_to_dict.__name__ == 'motion_dict_to_feature_v4':
            motion_dict = motion_feature_to_dict(motion_feature, abs_pose, self.skeleton)
        else:
            motion_dict = motion_feature_to_dict(motion_feature, abs_pose)

        if ret_fk:
            # 使用 skeleton 进行正向运动学，返回关节位置/旋转等
            return self.skeleton.forward_kinematics(motion_dict, return_full=ret_fk_full)
        else:
            return motion_dict

    # ================================================================
    # Sampling from dataset

    def _get_overlap(self, seg1, seg2):
        # 计算两个时间区间的重叠长度（秒）
        overlap_len = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
        return overlap_len

    def have_overlap(self, seg1, seg2):
        # 判断两个区间是否有重叠
        if seg1[0] > seg2[1] or seg2[0] > seg1[1]:
            return False
        else:
            return True

    def _extract_single_primitive(self, sample: Dict[str, Any], prim_start: int,
                                  prim_end: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Extract a single primitive from motion data"""
        # 从 sample['motion'] 中提取所需的键并转换为 tensor
        motion_data = {}
        for k in MotionKeys:
            if k in sample['motion']:
                motion_data[k] = torch.tensor(sample['motion'][k][prim_start:prim_end], dtype=torch.float32)

        # Find text label for the future portion
        prim_labels = []

        # 计算未来区间的帧索引（history 在前，因此未来从 prim_start + history_len 开始）
        future_start = prim_start + self.history_len
        future_end = prim_end - 1

        for ann in sample['frame_ann']:
            # 如果注释与未来区间有重叠，则视为该原语的标签候选
            if self.have_overlap([ann[0] * self.fps, ann[1] * self.fps], [future_start, future_end]):
                prim_labels.append(ann[2])

        # 随机选择一个标签（若没有则为空字符串）
        text_label = random.choice(prim_labels) if prim_labels else ''
        text_embedding = self.text_embeddings_dict.get(text_label, torch.zeros(512))

        return motion_data, text_embedding

    def _generate_motion_primitives(self, sample: Dict[str, Any],
                                    seg_start: int) -> List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """Generate all primitives from a single motion segment with proper overlapping"""
        primitives = []

        for primitive_idx in range(self.num_primitive):
            # 为保证原语连续性：第 i 个原语的最后 history_len 帧 == 第 i+1 个原语的前 history_len 帧
            prim_start = seg_start + primitive_idx * self.future_len
            prim_end = prim_start + self.future_len + self.history_len + 1

            motion_data, text_embedding = self._extract_single_primitive(sample, prim_start, prim_end)
            primitives.append((motion_data, text_embedding))

        return primitives

    def _sample_motion_batch(self,
                             generator: Optional[torch.Generator
                             ] = None) -> List[List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]]:
        """Sample a batch of motions and generate all their primitives"""
        # 根据是否使用加权采样选择索引生成方式
        if not self.weighted_sample:
            rand_idx = torch.randint(0, len(self.valid_indices), (self.batch_size,), generator=generator)
        else:
            rand_idx = torch.from_numpy(
                np.random.choice(len(self.raw_data), size=self.batch_size, replace=True, p=self.seq_weights)
            )

        all_motion_primitives = []
        for batch_idx in range(self.batch_size):
            # Get sample by index
            sample_idx = self.valid_indices[rand_idx[batch_idx].item()]  # type:ignore
            sample = self.raw_data[sample_idx]

            # Sample segment start ONCE per motion using the generator for reproducibility
            max_start = sample['length'] - self.segment_len

            if self.weighted_sample and self.frame_weight:
                # 使用帧权重进行加权选择起始帧
                seg_start = random.choices(range(max_start + 1), weights=sample['frame_weights'], k=1)[0]
            else:
                # 使用 torch 的 generator 来保证多线程可复现
                seg_start = int(torch.randint(0, max_start, (1,), generator=generator).item())

            # Generate ALL primitives for this motion using the SAME seg_start
            motion_primitives = self._generate_motion_primitives(sample, seg_start)
            all_motion_primitives.append(motion_primitives)

        return all_motion_primitives

    def _organize_primitives_by_index(
            self, all_motion_primitives: List[List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Organize primitives by primitive index for batching"""
        batch_primitives = []

        for primitive_idx in range(self.num_primitive):
            # 收集当前原语索引下整个 batch 的 motion 与 text
            motion_batch = []
            text_batch = []

            for batch_idx in range(self.batch_size):
                motion_data, text_embedding = all_motion_primitives[batch_idx][primitive_idx]
                motion_batch.append(motion_data)
                text_batch.append(text_embedding)

            # Convert to tensors and motion features
            motion_features = self._convert_to_motion_features(motion_batch)
            text_features = torch.stack(text_batch)

            # 对 motion_features 标准化并与 text 一起作为单个原语的批次输出
            batch_primitives.append((self.normalize(motion_features), text_features))

        return batch_primitives

    def _convert_to_motion_features(self, motion_batch: List[MotionDict]) -> torch.Tensor:
        """Convert batch of motion data to motion features"""
        # 将 motion_batch 中相同 key 的张量按 batch 维度堆叠
        motion_tensors = {}
        for k in MotionKeys:
            motion_tensors[k] = torch.stack([m[k] for m in motion_batch])

        # motion_dict_to_feature 将 motion dict 转换为特征张量
        motion_features, _ = motion_dict_to_feature(motion_tensors, self.skeleton)

        return motion_features

    def _generate_batch_optimized(self,
                                  generator: Optional[torch.Generator
                                  ] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate a batch using motion-first approach"""
        # Step 1: Sample motions and generate all their primitives
        all_motion_primitives = self._sample_motion_batch(generator)

        # Step 2: Organize primitives by index for batching
        batch_primitives = self._organize_primitives_by_index(all_motion_primitives)

        return batch_primitives

    def __iter__(self):
        """Iterator that yields batches in the expected format"""
        worker_info = data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        generator = torch.Generator()
        # 使用 worker_id + 随机偏移设置种子确保每个 worker 的随机性和可复现性
        generator.manual_seed(worker_id + np.random.randint(0, 1000000))

        while True:
            yield self._generate_batch_optimized(generator=generator)

    def __len__(self) -> int:
        # 数据集长度定义为有效样本数
        return len(self.valid_indices)
