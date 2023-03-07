# Taichi NeRFs
A PyTorch + Taichi implementation of [instant-ngp](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf) NeRF training pipeline. 

<p align="center">
<img src="assets/office.gif" width="200">
</p>

## Installation
1. Install PyTorch by `python -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116` (update the url with your installed CUDA Toolkit version number).
2. Install taichi nightly via `pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly`. 
3. Install requirements by `pip install -r requirements.txt`.
4. If you plan to train with your own video, please install `colmap` via `sudo apt install colmap` or follow instructions at https://colmap.github.io/install.html.

## Train with preprocessed datasets

### Synthetic NeRF

Download [Synthetic NeRF dataset](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NeRF.zip) and unzip it. Please keep the folder name unchanged.


We also provide a script to train the Lego scene from scratch, and display an interactive GUI at the end of the training.

```
./scripts/train_nsvf_lego.sh
```

Performance is measured on a Ubuntu 20.04 with an RTX3090 GPU. 

| Scene    | avg PSNR | Training Time(20 epochs)   | GPU     |
| :---:     | :---:    | :---: | :---:   |
| Lego | 35.0    | 208s  | RTX3090 |

To reach the best performance, here are the steps to follow:
1. Your work station is running on Linux and has RTX 3090 Graphics card
2. Follow the steps in [Installation Section](https://github.com/taichi-dev/taichi-nerfs#installation)
3. Uncomment `--half2_opt` to enable half2 optimization in the script, then `sh scripts/train_nsvf_lego.sh`. For now, half2 optimization is only supported on Linux with Graphics Card Architecture >Pascal.


### 360_v2 dataset
Download [360 v2 dataset](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) and unzip it. Please keep the folder name unchanged.

```
./scripts/train_360_v2_garden.sh
```
## Train with your own video

Place your video in `data` folder and update `VIDEO_FILE` in the script below accordingly. Running this script will preprocess your video and start training a NeRF out of it:

```
./scripts/train_from_video.sh
```

## [Preview] Mobile Deployment

Using [Taichi AOT](https://docs.taichi-lang.org/docs/tutorial), you can easily deploy a NeRF rendering pipeline on any mobile devices! 

<p align="center">
<img src="assets/NeRF_on_iPad.gif" width="200">
</p>

Stay tuned, more cool demos are on the way! For business inquiries, please reach out us at `contact@taichi.graphics`.
## Frequently asked questions (FAQ)

__Q:__ Is CUDA the only supported Taichi backend? How about vulkan backend? 

__A:__ For the most efficient interop with PyTorch CUDA backend, training is mostly tested with Taichi CUDA backend. However it's pretty straightforward to switch to Taichi vulkan backend if interop is removed, check out this awesome [taichi-ngp inference demo](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/rendering/taichi_ngp.py)!

__Q:__ I got OOM(Out of Memory) error on my GPU, what can I do?
__A:__ Reduce `batch_size` passed to `train.py`! By default it's `8192` which fits a RTX3090, you should reduce this accordingly. For instance, `batch_size=2048` is recommended on a RTX3060Ti. 

# Acknowledgement

The PyTorch interface of the training pipeline and colmap preprocessing are highly referred to:

*  [ngp_pl](https://github.com/kwea123/ngp_pl)
*  [instant-ngp CUDA implementation](https://github.com/NVlabs/instant-ngp/tree/master)
