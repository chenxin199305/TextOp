# Dataset

## Statistics

The curated dataset for Tracker contains:

- 9,780 total sequences
- 9,758 AMASS sequences
- 22 LAFAN1 sequences

The curated dataset for RobotMDAR contains:

- 8,285 total sequences
- 6,209 training sequences
- 2,076 validation sequences
- 105,395 seconds of total motion duration

## Dataset Construction

The following guidance shows how we obtain our dataset.

### 1. Download data

Download

- [AMASS](https://amass.is.tue.mpg.de/)
- [BABEL-TEACH](https://download.is.tue.mpg.de/download.php?domain=teach&resume=1&sfile=babel-data/babel-teach.zip)

data from their websites.

### 2. Retargeting

> Jason 2025-11-20:
> When try to use GMR to do retarget, should run under GMR conda env, because it use different python version compared to this project.

Use [GMR](https://github.com/YanjieZe/GMR) to retarget whole AMASS data to G1:

- Install `GMR` and additional `joblib` .
- Run `dataset/smplx_to_robot_dataset.py`, a slightly modified script.

```bash
# Generate retaregeted data
python dataset/smplx_to_robot_dataset.py 
--src_folder <path_to_dir_of_smplx_data> 
--tgt_folder <path_to_dir_to_save_robot_data> 
--robot unitree_g1
```

Post process retargeted data:

- This script interpolates the motion data from 30fps to 50fps,
- computes the feet contact mask, and saves the data in a [PBHC](https://github.com/TeleHuman/PBHC)-pkl format.

```bash
# Update retargeted data to 50Hz
python dataset/process_retarget_data.py 
--input_dir <Path To Retargeted Data Dir> 
--output_dir <Path To Output Dir> 
--robot_config "TextOpRobotMDAR/robotmdar/config/skeleton/g1.yaml"
```

#### 3. Prepare Data for Tracker

> We manually remove some unsuitable data in AMASS for RL training.

Pack the motion directory to a single file

```bash
cd TextOpTracker
python scripts/motion_package.py <folder_with_pkl_files>
```

Transform the data format to meet Tracker's requirement:

- You can also use `dataset/pkl_to_npz.py` to transform the files one by one.

```bash
# Activate the Tracker's environment
python scripts/pklpack_to_npz.py 
--input_file /path/to/aaa.pkl
--output_dir ./artifacts/unpacked_motions 
--input_fps 50 
--output_fps 50
```

We also select and add some high-quality motions from `LAFAN1`.
These parts of data can be transformed from `.csv` format to `.npz` format by:

```bash
python scripts/csv_to_npz.py 
--input_file LAFAN/dance1_subject2.csv 
--input_fps 30 
--frame_range 122 722
--output_file ./artifacts/dance1_subject2/motion.npz 
--output_fps 50
```

Organize the data files as following for Tracker loading:

```
TextOpTracker/artifacts/
├── Dataset/ 
│   ├── motion1
│   │   └── motion.npz
│   ├── motion1
│   │   └── motion.npz
│   └── ...
└── ...

```

#### 4. Prepare Data for RobotMDAR

Pack motion and text label dataset to meet RobotMDAR's requirement:

- It'll split the dataset into training and validation set according to the annotation from `BABEL`.
- The training set `train.pkl` comprises **6,209** sequences,
- the validation set `val.pkl` contains **2,076** sequences,
- and the entire dataset has a total duration of **105,395** seconds.

```bash
# Generate packaged data
python dataset/pack_dataset.py 
--amass_robot <Path To Retargeted Data Dir> 
--babel <Path to BABEL Dir>
```

Calculate data sampling weights for RobotMDAR training:

```bash
# Generate json data
python dataset/cal_action_statistics.py 
--data_folder <Path To Packaged Data Dir> 
--trg_filename <Path To Save Json File>
```
