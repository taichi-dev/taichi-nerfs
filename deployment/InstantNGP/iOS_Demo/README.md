# Regenerate AOT files
The Taichi-Python to be used for generating AOT files should match the version of the `libtaichi_c_api.a`. 
For inference with float32 precision:
```
python3 deployment/instant_ngp/assets/taichi_ngp.py --scene smh_lego --aot --res_w=800 --res_h=800
```

For inference with float16 precision:
```
python3 deployment/instant_ngp/assets/taichi_ngp.py --scene smh_lego --aot --fp16 --res_w=800 --res_h=800 --output deployment/instant_ngp_fp16/assets/compiled
```

# How to regenerate iOS AOT library
```
cd taichi-nerf/deployment
TAICHI_REPO_DIR=${TAICHI_REPO} sh scripts/build-taichi-ios.sh
cp build-taichi-ios-arm64/install/c_api/lib/libtaichi_c_api.a TaichiNerf_IOS/TaichiNerfTestbench/c_api/lib/
```

# How to check output image
We enabled File Sharing for this the iOS App, and you can simply view the output image using "Finder".
