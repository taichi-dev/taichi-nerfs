#!/bin/bash

set -euo pipefail

export ROOT_DIR="./360_v2/"
export DOWNSAMPLE=0.25 # to avoid OOM

python3 train.py \
    --root_dir $ROOT_DIR/garden --dataset_name colmap \
    --exp_name garden --downsample $DOWNSAMPLE \
    --scale 8.0 --batch_size 4096 
