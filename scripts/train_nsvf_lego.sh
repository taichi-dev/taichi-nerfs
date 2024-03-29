#!/bin/bash

set -euo pipefail

export DATA_DIR=./data/Synthetic_NeRF

python3 train.py \
    --root_dir $DATA_DIR/Lego \
    --exp_name Lego \
    --batch_size 8192 --lr 1e-2 \
    --gpu 0 \
    #--gui \
    #--half2_opt \
    #--val_only --ckpt_path ckpts/nsvf/Lego/epoch=19-v6.ckpt
