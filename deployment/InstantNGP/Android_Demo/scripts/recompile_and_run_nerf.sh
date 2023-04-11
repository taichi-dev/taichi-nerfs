# regenerate aot files
rm -rf instant_ngp_fp16/assets/compiled
rm -rf instant_ngp/assets/compiled

# recompile Taichi C-API & Nerf project
TAICHI_REPO_DIR=/home/taichigraphics/workspace/TaichiRepos/taichi_android ./scripts/build-taichi-android.sh
./scripts/build-android.sh

# Copy and test on Android
adb shell "mkdir -p /data/local/tmp/NERF"
adb shell "cd /data/local/tmp/NERF && rm -rf *"
adb push build-taichi-android-aarch64/install/c_api/lib/libtaichi_c_api.so instant_ngp instant_ngp_fp16 build-android-aarch64/nerf build-android-aarch64/nerf_fp16 /data/local/tmp/NERF
adb shell "cd /data/local/tmp/NERF && LD_LIBRARY_PATH=. ./nerf_fp16"
