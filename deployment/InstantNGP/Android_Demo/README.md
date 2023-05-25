# Taichi NGP Android Demo
Our Android Demo is a semi-finished product at this moment, due to lack of expertise in Android development. 

We support the entire infrastructure from setting up the `camera position` to finally generate the `inferenced image`. However the camera position is directly set in the code, while the inferenced image is written to the disk. To make it a real Demo, consider connecting the camera position with a `Touch Event`, and pass the inferenced image to `Android GUI`.

Please be aware that there are known issues with certain Android devices, which is primarily related to their Vulkan support.

There's also performance optimization opportunities wrt Android deployment. 

**We welcome all contributions to improve the quality of the Android Demo!**

## Prerequisites 
1. Device running Android with Vulkan support
2. Android Studio: https://developer.android.com/studio
3. Android Studio - adb: https://developer.android.com/tools/adb
4. Export ANDROID_NDK_ROOT and ANDROID_SDK_ROOT, for example:
5. Install Ninja: `sudo apt-get update && sudo apt-get install ninja-build`
```
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/25.1.8937393/
export ANDROID_SDK_ROOT=~/Android/Sdk/
```

## Build and Install the Demo
1. Change camera position if neccessary
open `main.cpp` and modify the following lines:
```
float angle_x, angle_y, angle_z = 0.0;
float radius = 2.5;
app.pose_rotate_scale(angle_x, angle_y, angle_z, radius);
```

2. Build and run the inference
```
sh scripts/compile_and_run_nerf.sh
```

3. Checkout the output image
  * You'll notice a `out.png` generated in the same folder, which contains the **inferenced image** with current **camera position**.


## Train and deploy your own NGP model
Follow instructions from: [How to train a deployable NGP model](../RetrainInstruction.md)

In the end, you will obtain a bunch of AOT files at `taichi_ngp/compiled`

4. Modify the width and height in C++ code if you've changed `--res_w` or `res_h`:
  * open `main.cpp` and modify the following lines:
```
int img_width  = CUSTOM_WIDTH;
int img_height = CUSTOM_HEIGHT;
app.initialize(img_width, img_height,
               aot_file_root,
               hash_embedding_path,
               sigma_weights_path,
               rgb_weights_path,
               density_bitfield_path,
               pose_path,
               directions_path);
```

5. Install the Demo following steps from **Build and Install the Demo** and have fun!
