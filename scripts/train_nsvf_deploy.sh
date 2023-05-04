#!/bin/bash

set -euo pipefail

export DATA_DIR=./Synthetic_NeRF

python3 train.py \
    --root_dir $DATA_DIR/Lego \
    --exp_name Lego \
    --max_steps 20000 --batch_size 8192 --lr 1e-2 \
    --deployment --deployment_model_path=. --gui
