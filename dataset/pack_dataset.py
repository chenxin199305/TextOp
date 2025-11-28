import os
import argparse
import numpy as np
import joblib
import json
import yaml
from tqdm import tqdm
from pathlib import Path

DATASET_NAME = "Babel-teach X AMASS Robot"
BABEL_SPLIT = ['train', 'val']
FPS = 50


def process_babel_json(babel_json_path):
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

            feat_p = Path(feat_p).relative_to(feat_p.split("/")[0])  # relative to the first component
            feat_p = str(feat_p.as_posix())  # normalize path

            frame_ann_raw = v.get("frame_ann", None)
            has_frame_ann = frame_ann_raw is not None and isinstance(frame_ann_raw, dict)

            if not has_frame_ann:
                # Use seq_ann instead
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
            print(f"Skipping {k} due to error: {e}, where value is {v}")
            breakpoint()  # Debugging breakpoint
            continue

    return result


def load_babel(BABEL_DIR):
    print(f"Loading BABEL dataset from: {BABEL_DIR}")
    babel_all = {}
    for split in BABEL_SPLIT:
        json_path = os.path.join(BABEL_DIR, f"{split}.json")
        babel_all[split] = process_babel_json(json_path)
    # print(f"babel_all = {babel_all}")
    return babel_all


def load_amass(amass_dir):
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
                rel_path = "/".join(rel_path.split("/")[-4:])
                amass_data[rel_path] = motion
        except Exception as e:
            print(f"Error loading {motion_file}: {e}")
            continue

    return amass_data


def merge_datasets(amass_data, babel_data_all, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    merged_data = {}
    for split in BABEL_SPLIT:  # "train" and "val" splits
        merged = []
        babel_data = babel_data_all[split]
        for feat_p, babel_entry in tqdm(babel_data.items(), desc=f"Merging {split}", disable=True):

            # breakpoint()
            motion = amass_data.get(feat_p, None)

            print(f"split = {split}, feat_p = {feat_p}, motion = {motion}")

            if motion is None or motion['root_trans_offset'].shape[0] <= 67:
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
        print(f"Saved {len(merged)} motions to {output_file}")

    stats = {
        'dataset name': DATASET_NAME,
        'fps': FPS,
        'babel dir': BABEL_DIR,
        'amass robot dir': AMASS_ROBOT_DIR,
        'output dir': OUTPUT_DIR,
        'babel count': {k: len(v) for k, v in babel_data_all.items()},
        'amass count': len(amass_data),
        'merged count': {k: len(v) for k, v in merged_data.items() if k in BABEL_SPLIT},
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
    babel_data = load_babel(BABEL_DIR)
    amass_data = load_amass(AMASS_ROBOT_DIR)
    merge_datasets(amass_data, babel_data, OUTPUT_DIR)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument('--amass_robot', type=str, required=True)
    args.add_argument('--babel', type=str, required=True)
    parsed_args = args.parse_args()

    AMASS_ROBOT_DIR = parsed_args.amass_robot
    BABEL_DIR = parsed_args.babel
    OUTPUT_DIR = "./dataset/BABEL-AMASS-ROBOT-23dof-50fps-TEACH"

    main()
