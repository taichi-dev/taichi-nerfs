# recompile Taichi C-API & Nerf project
export TAICHI_REPO_DIR=/home/taichigraphics/workspace/TaichiRepos/taichi_android

./scripts/build-taichi-android.sh
./scripts/build-android.sh

# Copy and test on Android
adb shell "mkdir -p /data/local/tmp/NERF"
adb shell "cd /data/local/tmp/NERF && rm -rf *"
adb push build-taichi-android-aarch64/install/c_api/lib/libtaichi_c_api.so ../taichi_ngp/compiled build-android-aarch64/nerf /data/local/tmp/NERF
adb shell "cd /data/local/tmp/NERF && LD_LIBRARY_PATH=. ./nerf"
adb pull /data/local/tmp/NERF/out.png
