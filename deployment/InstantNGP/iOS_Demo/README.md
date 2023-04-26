# Taichi NGP iOS Demo

## Prerequisites 
1. Machine running MacOS 
2. Device running iOS with Metal support (typically iPhones or iPads)
3. Xcode: https://developer.apple.com/xcode

4. Turn on **Developer Mode** for your device
[Image]


## Build and Install the Demo
1. Open the TaichiNGPDemo.xcodeproj in Xcode

2. Configure app signature
[Image]

3. Select the target device
[Image]

4. Click `Product --> Run`
[Image]

5. Trust your app on the device
[Image]

Now, re-run the code and enjoy Nerf on your phone!
[Demo]

## Deploy with modified taichi_ngp.py
Sometimes you may want to modify the Taichi NGP model in `taichi_ngp.py` for fun. Once done editing `taichi_ngp.py`, you will need to regenerate the AOT files (`InstantNGP/taichi_ngp/compiled`). 

For example, let's say we want to adjust the camera resolution. 

1. Modify `taichi_ngp.py`:
For our example only, `taichi_ngp.py` already supports adjusting the camera resolution via `--res_w` `--res_h`

2. Regenerate AOT files:
`python3 InstantNGP/taichi_ngp.py --scene smh_lego --aot --res_w=100 --res_h=200`

3. Modify the width and height in C++ code accordingly 

open ViewController.mm and modify the following lines:
```
int img_width = 100;
int img_height = 200;
app_f32.initialize(img_width, img_height,
                   std::string([aotFilePath UTF8String]),
                   std::string([hashEmbeddingFilePath UTF8String]),
                   std::string([sigmaWeightsFilePath UTF8String]),
                   std::string([rgbWeightsFilePath UTF8String]),
                   std::string([densityBitfieldFilePath UTF8String]),
                   std::string([poseFilePath UTF8String]),
                   std::string([directionsFilePath UTF8String]));
```

4. Install the Demo following steps from **Build and Install the Demo**

## Train and deploy your own NGP model
[TODO: train + deploy]
