#  Put your video in data/ folder and update filename VIDEO_FILE
#  SCALE choose from 1, 4, 8, 16, 64; 16 is recommended for a real screne
#  VIDEO_FPS = 2 is suitable for a one minute video 
set VIDEO_FILE 'video.mp4' 
set SCALE 16 
set VIDEO_FPS 2 

echo "video path $VIDEO_FILE"
echo "scale $SCALE"
echo "video fps $VIDEO_FPS"

cd "data"

python colmap2nerf.py --video_in $VIDEO_FILE --video_fps $VIDEO_FPS --run_colmap --aabb_scale $SCALE --images images

Move-Item colmap_sparse sparse
cd ..


python train.py --root_dir data --dataset_name colmap --exp_name custom --downsample 0.25  --num_epochs 20 --scale $SCALE  --gui  