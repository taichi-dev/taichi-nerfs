# Taichi NGP Android Demo
Our Android Demo is a semi-finished product at this moment, due to lack of expertise in Android development. 

We support the entire infrastructure from setting up the **camera position** to finally generate the **inferenced image**. However the camera position is directly set in the code, while the inferenced image is written to the disk. To make it a real Demo, consider connecting the camera position with a **Touch Event**, and pass the inferenced image to **Android GUI**.

Besides, there's also performance optimization opportunities wrt Android deployment. 

**We welcome all contributions to improve the quality of the Android Demo!**

## Prerequisites 
1. Device running Android with Vulkan support
2. Android Studio: https://developer.android.com/studio
3. Android Studio - abd: https://developer.android.com/tools/adb

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
You'll notice a `out.png` generated in the same folder, which contains the **inferenced image** with current **camera position**.

## Deploy with modified taichi_ngp.py
Sometimes you may want to modify the Taichi NGP model in `taichi_ngp.py` for fun. Once done editing `taichi_ngp.py`, you will need to regenerate the AOT files (`InstantNGP/taichi_ngp/compiled`). 

For example, let's say we want to adjust the camera resolution. 

1. Modify `taichi_ngp.py`:
For our example only, `taichi_ngp.py` already supports adjusting the camera resolution via `--res_w` `--res_h`

2. Regenerate AOT files:
`python3 InstantNGP/taichi_ngp.py --scene smh_lego --aot --res_w=100 --res_h=200`

3. **Modify the `width` and `height` in C++ code accordingly**
open `main.cpp` and modify the following lines:
```
int img_width  = 300;
int img_height = 600;
app.initialize(img_width, img_height,
               aot_file_root,
               hash_embedding_path,
               sigma_weights_path,
               rgb_weights_path,
               density_bitfield_path,
               pose_path,
               directions_path);
```

4. Install the Demo following steps from **Build and Install the Demo**

## Train and deploy your own NGP model
[TODO: train + deploy]
