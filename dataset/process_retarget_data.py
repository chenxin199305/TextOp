import numpy as np
import sys
# sys.modules['numpy._core'] = np.core
from scipy import interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
import joblib
import argparse
from robotmdar.skeleton.robot import RobotSkeleton
from omegaconf import DictConfig, OmegaConf
import torch


def foot_detect(positions, thres=0.002):
    fid_r, fid_l = 12, 6
    velfactor, heightfactor = np.array([thres]), np.array([0.08])
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[1:, fid_l, 2]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(int) & (feet_l_h < heightfactor).astype(int)).astype(np.float32)
    feet_l = np.expand_dims(feet_l, axis=1)
    feet_l = np.concatenate([np.array([[1.]]), feet_l], axis=0)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[1:, fid_r, 2]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(int) & (feet_r_h < heightfactor).astype(int)).astype(np.float32)
    feet_r = np.expand_dims(feet_r, axis=1)
    feet_r = np.concatenate([np.array([[1.]]), feet_r], axis=0)
    return feet_l, feet_r


def cal_contact_mask(dof, root_trans, root_rot):
    motion_dict = {
        "dof": torch.from_numpy(dof).float(),
        "root_trans_offset": torch.from_numpy(root_trans).float(),
        "root_rot": torch.from_numpy(root_rot).float()
    }
    fk_return = SKELETON.forward_kinematics(motion_dict, return_full=False)
    feet_l, feet_r = foot_detect(fk_return['global_translation_extend'][0].numpy())
    contact_mask = np.concatenate([feet_l, feet_r], axis=-1)

    return contact_mask


def interpolate_motion_data_slerp(root_trans_offset, root_rot, dof, original_fps=30, target_fps=50):
    original_frames = root_trans_offset.shape[0]
    original_time = np.linspace(0, original_frames / original_fps, original_frames)

    target_frames = int(original_frames * target_fps / original_fps)
    target_time = np.linspace(0, original_frames / original_fps, target_frames)

    def interpolate_linear(data, target_time):
        result = np.zeros((len(target_time), data.shape[1]))
        for i in range(data.shape[1]):
            interpolator = interpolate.interp1d(original_time, data[:, i],
                                                kind='linear',
                                                fill_value="extrapolate")
            result[:, i] = interpolator(target_time)
        return result

    def interpolate_slerp(quaternions, target_time):
        rotations = R.from_quat(quaternions)
        slerp = Slerp(original_time, rotations)

        interpolated_rotations = slerp(target_time)

        return interpolated_rotations.as_quat()

    interpolated_trans = interpolate_linear(root_trans_offset, target_time)
    interpolated_rot = interpolate_slerp(root_rot, target_time)
    interpolated_dof = interpolate_linear(dof, target_time)

    return interpolated_trans, interpolated_rot, interpolated_dof


def process_pkl_file(input_path, output_path):
    try:
        data = joblib.load(input_path)

        keyname = input_path.split('/')[-1].replace('.pkl', '')
        root_trans_offset = data['root_pos']
        root_rot = data['root_rot']
        dof = np.concatenate((data['dof_pos'][:, :19], data['dof_pos'][:, 22:26]), axis=1)
        fps = 50

        interpolated_trans, interpolated_rot, interpolated_dof = interpolate_motion_data_slerp(root_trans_offset, root_rot, dof)

        contact_mask = cal_contact_mask(dof, root_trans_offset, root_rot)

        processed_data = {}

        data_dump = {
            "root_trans_offset": interpolated_trans,
            "dof": interpolated_dof,
            "root_rot": interpolated_rot,
            "contact_mask": contact_mask,
            "fps": fps
        }
        processed_data[keyname] = data_dump
        joblib.dump(processed_data, output_path)

        return True
    except Exception as e:
        print(f"\nProcess {input_path} error: {str(e)}")
        return False


def process_directory(source_dir, target_dir, file_extension='.pkl'):
    os.makedirs(target_dir, exist_ok=True)

    pkl_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(file_extension):
                full_path = os.path.join(root, file)
                pkl_files.append(full_path)

    total_files = len(pkl_files)
    print(f"Find {total_files} files!")

    for file_path in tqdm(pkl_files, desc="Processing"):
        relative_path = os.path.relpath(file_path, source_dir)
        output_path = os.path.join(target_dir, relative_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        process_pkl_file(file_path, output_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument('--input_dir', type=str, required=True, help='source retargeted data folder')
    args.add_argument('--output_dir', type=str, required=True, help='target processed data folder')
    args.add_argument('--robot_config', type=str, required=True, help='path to robot config')
    parsed_args = args.parse_args()

    robot_config = parsed_args.robot_config
    robot_cfg = OmegaConf.load(robot_config)

    SKELETON = RobotSkeleton(device="cpu", cfg=robot_cfg)

    process_directory(parsed_args.input_dir, parsed_args.output_dir)
