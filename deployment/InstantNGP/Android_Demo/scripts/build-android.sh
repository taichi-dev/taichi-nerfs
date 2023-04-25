#!/bin/bash
set -e

rm -rf build-android-aarch64
mkdir build-android-aarch64
pushd build-android-aarch64
TAICHI_C_API_INSTALL_DIR="${PWD}/../c_api" cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_PLATFORM=android-29 \
    -DANDROID_ABI="arm64-v8a" \
    -G "Ninja"
if [ $? -ne 0 ]; then
    echo "Configuration failed"
    exit -1
fi

cmake --build .
if [ $? -ne 0 ]; then
    echo "Build failed"
    exit -1
fi
popd
