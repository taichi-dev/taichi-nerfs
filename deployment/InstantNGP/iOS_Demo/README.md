# Taichi NGP iOS Demo

<p align="center">
<img src="../../../assets/NeRF_on_iPad.gif" width="200"> 
<img src="../../../assets/NeRF_iPhone14_Pro_Max.gif" width="200">
</p>

<p align="center">
<img src="../../../assets/Chair_iPad.gif" width="200"> 
<img src="../../../assets/Ficus_iPad.gif" width="200">
<img src="../../../assets/Lego_iPad.gif" width="200">
<img src="../../../assets/Mic_iPad.gif" width="200">
</p>

## Prerequisites 
1. Machine running MacOS 
2. Device running iOS with Metal support (typically iPhones or iPads)
3. Xcode: https://developer.apple.com/xcode

4. Turn on **Developer Mode** for your device
  * `Settings: Privacy & Security --> Developer Mode`

<img width="200" alt="image" src="https://user-images.githubusercontent.com/22334008/234472321-54f6d270-3b88-448a-8941-fdbb1ef834b1.png">


## Build and Install the Demo
1. Open the TaichiNGPDemo.xcodeproj in Xcode

2. Configure app signature
- Click on `TaichiNGPDemo` on the sidebar, then switch to the `Signing & Capabilities` tab
- Add your Apple account to `Team` and just randomly make up a Bundle identifier

<img width="500" alt="image" src="https://user-images.githubusercontent.com/22334008/234472482-49b786fe-f1d5-4936-854c-cf09dd47ca98.png">


3. Select the target device

<img width="500" alt="image" src="https://user-images.githubusercontent.com/22334008/234473478-122a151f-c841-4990-be6a-0799447aa69d.png">


4. Click `Product --> Run`

<img width="500" alt="image" src="https://user-images.githubusercontent.com/22334008/234473561-07a06376-91bc-4e57-8454-83c0683251b9.png">


5. Trust your app on the device
  * `Settings: General --> VPN & Device Management --> App Development --> Trust`

<img width="400" alt="image" src="https://user-images.githubusercontent.com/22334008/234473586-be8d8032-cf54-43ed-81a8-f636d7c2b37c.png">


Now, re-run the code and enjoy Nerf on your phone!

<img src="../../../assets/NeRF_iPhone14_Pro_Max.gif" width="200">

## Train and deploy your own NGP model

Typically, you need two different workstations to train and deploy an NGP model:
- A Windows/Linux machine with Nvidia GPUs to train the model
- A MacOS to compile and install the deployment model (copied from the Windows/Linux machine) to your iOS device


**Let's start with training the NGP model on the Windows/Linux machine with GPU cards**

Follow the instructions from: [How to train a deployable NGP model](../RetrainInstruction.md)

In the end, you will obtain a bunch of AOT files at `taichi_ngp/compiled`

**Now copy the generated AOT files under `taichi_ngp/compiled` to the same folder on your MacOS**

4. [MacOS] Modify the width and height in C++ code if you've changed `--res_w` or `res_h`:
  * open `ViewController.mm` and modify the following lines:
```
int img_width = CUSTOM_WIDTH;
int img_height = CUSTOM_HEIGHT;
app_f32.initialize(img_width, img_height,
                   std::string([aotFilePath UTF8String]),
                   std::string([hashEmbeddingFilePath UTF8String]),
                   std::string([sigmaWeightsFilePath UTF8String]),
                   std::string([rgbWeightsFilePath UTF8String]),
                   std::string([densityBitfieldFilePath UTF8String]),
                   std::string([poseFilePath UTF8String]),
                   std::string([directionsFilePath UTF8String]));
```


5. [MacOS] Install the Demo following steps from **Build and Install the Demo** and have fun!

![Ship](https://user-images.githubusercontent.com/22334008/236150277-ed64032c-c021-4e22-8914-02aa65b960f2.gif)
