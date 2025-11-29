#!/bin/bash

# get current shell script path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This script moves all files from a specified source folder to target folder.
DATASET_SOURCE_DIR="$SCRIPT_DIR/AMASS_retarget_50hz"
DATASET_TARGET_DIR="$SCRIPT_DIR/AMASS_retarget_50hz_rearranged"

# remove DATASET_TARGET_DIR at first
rm -rf "$DATASET_TARGET_DIR"

# copy DATASET_SOURCE_DIR to DATASET_TARGET_DIR
cp -r "$DATASET_SOURCE_DIR" "$DATASET_TARGET_DIR"

# source folder names and target parent folder mapping
# source_folder -> target_folder
# KIT/KIT -> KIT/
declare -A SOURCE_FOLDERS_TO_MOVE=(
    ["ACCAD"]="ACCAD/ACCAD"
    ["BMLmovi"]="BMLmovi/BMLmovi"
    ["BMLrub"]="BMLrub/BioMotionLab_NTroje"
    ["CMU"]="CMU/CMU"
    ["DanceDB"]="DanceDB/DanceDB"
    ["DFaust"]="DFaust/DFaust_67"
    ["EKUT"]="EKUT/EKUT"
    ["EyesJapanDataset"]="EyesJapanDataset/Eyes_Japan_Dataset"
    ["HDM05"]="HDM05/MPI_HDM05"
    ["HUMAN4D"]="HUMAN4D/HUMAN4D"
    ["HumanEva"]="HumanEva/HumanEva"
    ["KIT"]="KIT/KIT"
    ["MoSh"]="MoSh/MPI_mosh"
    ["PosePrior"]="PosePrior/MPI_Limits"
    ["SFU"]="SFU/SFU"
    ["SSM"]="SSM/SSM_synced"
    ["TCDHands"]="TCDHands/TCD_handMocap"
    ["TotalCapture"]="TotalCapture/TotalCapture"
    ["Transitions"]="Transitions/Transitions_mocap"
)
declare -A TARGET_FOLDERS_TO_MOVE=(
    ["ACCAD"]="ACCAD"
    ["BMLmovi"]="BMLmovi"
    ["BMLrub"]="BioMotionLab_NTroje"
    ["CMU"]="CMU"
    ["DanceDB"]="DanceDB"
    ["DFaust"]="DFaust_67"
    ["EKUT"]="EKUT"
    ["EyesJapanDataset"]="Eyes_Japan_Dataset"
    ["HDM05"]="MPI_HDM05"
    ["HUMAN4D"]="HUMAN4D"
    ["HumanEva"]="HumanEva"
    ["KIT"]="KIT"
    ["MoSh"]="MPI_mosh"
    ["PosePrior"]="MPI_Limits"
    ["SFU"]="SFU"
    ["SSM"]="SSM_synced"
    ["TCDHands"]="TCD_handMocap"
    ["TotalCapture"]="TotalCapture"
    ["Transitions"]="Transitions_mocap"
)

for FOLDER_NAME in "${!SOURCE_FOLDERS_TO_MOVE[@]}"; do
    SOURCE_FOLDER="${SOURCE_FOLDERS_TO_MOVE[$FOLDER_NAME]}"
    TARGET_FOLDER="${TARGET_FOLDERS_TO_MOVE[$FOLDER_NAME]}"

    echo "Processing folder: $FOLDER_NAME"
    echo "Source folder: $SOURCE_FOLDER"
    echo "Target folder: $TARGET_FOLDER"

    SOURCE_DIR="$DATASET_TARGET_DIR/$SOURCE_FOLDER"
    TARGET_DIR="$DATASET_TARGET_DIR/$TARGET_FOLDER"

    echo "Source directory: $SOURCE_DIR"
    echo "Target directory: $TARGET_DIR"

    # Check if source directory exists
    if [ -d "$SOURCE_DIR" ]; then
        echo "Moving files from $SOURCE_DIR to $TARGET_DIR"

        # Create target directory if it doesn't exist
        mkdir -p "$TARGET_DIR"

        # Move files from child folder to parent folder
        mv "$SOURCE_DIR"/* "$TARGET_DIR"/

        # Remove the now empty child folder
        rmdir "$SOURCE_DIR"
    else
        echo "Source directory $SOURCE_DIR does not exist. Skipping."
    fi
done