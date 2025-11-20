import pickle
import sys
from pathlib import Path

import joblib
import torch


def load_pkl(pkl_path):
    pkl_path = str(pkl_path)  # Convert Path to string if needed
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e1:
        try:
            print(f"Pickle failed, trying joblib for {pkl_path}")
            return joblib.load(pkl_path)
        except Exception as e2:
            try:
                print(f"Joblib failed, trying torch.load for {pkl_path}")
                return torch.load(pkl_path, map_location="cpu")
            except Exception as e3:
                raise RuntimeError(
                    f"Failed to load {pkl_path} as pickle, joblib, or torch.\nErrors:\n- pickle: {e1}\n- joblib: {e2}\n- torch: {e3}"
                )


def merge_motion_files(pkl_paths):
    merged = {}
    for pkl_path in pkl_paths:
        print(f"Loading: {pkl_path}")
        data = load_pkl(pkl_path)

        if not isinstance(data, dict):
            raise ValueError(f"{pkl_path} does not contain a dict")

        for k in data:
            if k in merged:
                raise KeyError(
                    f"Duplicate key '{k}' found in file: {pkl_path}")
        merged.update(data)

    return merged


def main():
    if len(sys.argv) != 2:
        print("Usage: python motion_package.py <folder_with_pkl_files>")
        sys.exit(1)

    folder_path = Path(sys.argv[1])
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    pkl_paths = sorted(folder_path.glob("*.pkl"))
    if not pkl_paths:
        print(f"No .pkl files found in {folder_path}")
        sys.exit(1)

    print(f"Found {len(pkl_paths)} pkl files.")
    merged_data = merge_motion_files(pkl_paths)

    output_path = folder_path.parent / (folder_path.name +
                                        "_merged_motion.pkl")
    joblib.dump(merged_data, output_path)

    print(f"Merged motion saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
