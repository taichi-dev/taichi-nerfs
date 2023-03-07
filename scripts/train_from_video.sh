#!/bin/bash

VIDEO_FILE='' # Put your video in data/ folder and update filename here
SCALE=1 # choose from 1, 4, 8, 16, 64; 16 is recommended for a real screne
VIDEO_FPS=2 # 2 is suitable for a one minute video 

while getopts v:s:f: flag
do
    case "${flag}" in
        v) VIDEO_FILE=${OPTARG};;
        s) SCALE=${OPTARG};;
        f) VIDEO_FPS=${OPTARG};;
    esac
done

echo "video path $VIDEO_FILE"
echo "scale $SCALE"
echo "video fps $VIDEO_FPS"

pushd data
python3 colmap2nerf.py --video_in $VIDEO_FILE --video_fps $VIDEO_FPS --run_colmap --aabb_scale $SCALE --images images
mv colmap_sparse sparse
popd

python3 train.py --root_dir data --dataset_name colmap --exp_name custom --downsample 0.25  --num_epochs 20 --scale $SCALE --gui
