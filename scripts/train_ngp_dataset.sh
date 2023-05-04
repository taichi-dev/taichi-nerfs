#!/bin/bash

set -euo pipefail

export ROOT_DIR="./ngp_data"
export DOWNSAMPLE=0.5 # to avoid OOM

python3 train.py \
    --root_dir $ROOT_DIR/ --dataset_name ngp \
    --exp_name custom_ngp --downsample $DOWNSAMPLE \
    --scale 8.0 --batch_size 8192
