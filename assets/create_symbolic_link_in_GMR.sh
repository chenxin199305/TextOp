#!/bin/bash

# get current shell script path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# set amass dataset path
AMASS_DIR="$SCRIPT_DIR/AMASS/SMPL-H2X"

# set retarget folder path
RETARGET_DIR="$SCRIPT_DIR/AMASS_retarget"

# create symbolic link for AMASS if it doesn't exist
if [ ! -e "$SCRIPT_DIR/../GMR/assets/AMASS" ]; then
    echo "Creating symbolic link for AMASS in GMR"
    ln -s "$AMASS_DIR" "$SCRIPT_DIR/../GMR/assets/AMASS"
else
    echo "Symbolic link for AMASS already exists. Skipping."
fi

# create symbolic link for AMASS_retarget if it doesn't exist
if [ ! -e "$SCRIPT_DIR/../GMR/assets/AMASS_retarget" ]; then
    echo "Creating symbolic link for AMASS_retarget in GMR"
    ln -s "$RETARGET_DIR" "$SCRIPT_DIR/../GMR/assets/AMASS_retarget"
else
    echo "Symbolic link for AMASS_retarget already exists. Skipping."
fi

# set smplx model path
SMPLX_DIR="$SCRIPT_DIR/SMPL/smplx"

# create symbolic link for smpl if it doesn't exist
if [ ! -e "$SCRIPT_DIR/../GMR/assets/body_models/smplx" ]; then
    echo "Creating symbolic link for smplx in GMR"
    ln -s "$SMPLX_DIR" "$SCRIPT_DIR/../GMR/assets/body_models/smplx"
else
    echo "Symbolic link for smplx already exists. Skipping."
fi