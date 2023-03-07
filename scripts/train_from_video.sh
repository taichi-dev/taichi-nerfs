#!/bin/bash

set -euo pipefail

VIDEO_FILE='IMG_3117.MOV'
SCALE=1 # choose from 1, 4, 8, 16, 64

pushd data
python3 colmap2nerf.py --video_in $VIDEO_FILE --video_fps 2 --run_colmap --aabb_scale $SCALE --images images
mv colmap_sparse sparse
popd

python3 train.py --root_dir data --dataset_name colmap --exp_name custom --downsample 0.25  --num_epochs 20 --scale $SCALE --gui
