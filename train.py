import glob
import os
import time
import tqdm
import random
import warnings

import torch
import imageio
import numpy as np
import taichi as ti
from einops import rearrange
import torch.nn.functional as F

from gui import NGPGUI
from opt import get_opts
from datasets import dataset_dict

from modules.networks import NGP
from modules.intersection import get_rays
from modules.distortion import distortion_loss
from modules.rendering import MAX_SAMPLES, render
from modules.utils import depth2img, save_deployment_model

from torchmetrics import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
)

warnings.filterwarnings("ignore")

def taichi_init(args):
    taichi_init_args = {"arch": ti.cuda,}
    if args.half_opt:
        taichi_init_args["half2_vectorization"] = True

    ti.init(**taichi_init_args)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    hparams = get_opts()
    taichi_init(hparams)

    val_dir = 'results/'

    # rendering configuration
    exp_step_factor = 1 / 256 if hparams.scale > 0.5 else 0.

    # occupancy grid update configuration
    warmup_steps = 256
    update_interval = 16

    # datasets
    dataset = dataset_dict[hparams.dataset_name]
    train_dataset = dataset(
        root_dir=hparams.root_dir,
        split=hparams.split,
        downsample=hparams.downsample,
    ).to(device)
    train_dataset.batch_size = hparams.batch_size
    train_dataset.ray_sampling_strategy = hparams.ray_sampling_strategy

    test_dataset = dataset(
        root_dir=hparams.root_dir,
        split='test',
        downsample=hparams.downsample,
    ).to(device)
    # TODO: add test set rendering code


    # metric
    val_psnr = PeakSignalNoiseRatio(
        data_range=1
    ).to(device)
    val_ssim = StructuralSimilarityIndexMeasure(
        data_range=1
    ).to(device)

    if hparams.deployment:
        model_config = {
            'scale': hparams.scale,
            'pos_encoder_type': 'hash',
            'levels': 4,
            'feature_per_level': 4,
            'base_res': 32,
            'max_res': 128,
            'log2_T': 21,
            'xyz_net_width': 16,
            'rgb_net_width': 16,
            'rgb_net_depth': 1,
        }
    else:
        model_config = {
            'scale': hparams.scale,
            'pos_encoder_type': hparams.encoder_type,
            'max_res': 1024 if hparams.scale == 0.5 else 4096,
            'half_opt': hparams.half_opt,
        }

    # model
    model = NGP(**model_config).to(device)

    # load checkpoint if ckpt path is provided
    if hparams.ckpt_path:
        state_dict = torch.load(hparams.ckpt_path)
        model.load_state_dict(state_dict)
        print("Load checkpoint from %s" % hparams.ckpt_path)

    model.mark_invisible_cells(
        train_dataset.K,
        train_dataset.poses, 
        train_dataset.img_wh,
    )

    # use large scaler, the default scaler is 2**16 
    # TODO: investigate why the gradient is small
    if hparams.half_opt:
        scaler = 2**16
    else:
        scaler = 2**19
    grad_scaler = torch.cuda.amp.GradScaler(scaler)
    # optimizer
    try:
        import apex
        optimizer = apex.optimizers.FusedAdam(
            model.parameters(), 
            lr=hparams.lr, 
            eps=1e-15,
        )
    except ImportError:
        print("Failed to import apex FusedAdam, use torch Adam instead.")
        optimizer = torch.optim.Adam(
            model.parameters(), 
            hparams.lr, 
            eps=1e-15,
        )

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        hparams.max_steps,
        hparams.lr/30
    )


    # training loop
    tic = time.time()
    for step in range(hparams.max_steps+1):
        model.train()

        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]

        direction = data['direction']
        pose = data['pose']

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if step % update_interval == 0:
                model.update_density_grid(
                    0.01 * MAX_SAMPLES / 3**0.5,
                    warmup=step < warmup_steps,
                )
            # get rays
            rays_o, rays_d = get_rays(direction, pose)
            # render image
            results = render(
                model, 
                rays_o, 
                rays_d,
                exp_step_factor=exp_step_factor,
            )

            loss = F.mse_loss(results['rgb'], data['rgb'])
            if hparams.distortion_loss_w > 0:
                loss += hparams.distortion_loss_w * distortion_loss(results).mean()

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()

        if step % 1000 == 0:
            elapsed_time = time.time() - tic
            with torch.no_grad():
                mse = F.mse_loss(results['rgb'], data['rgb'])
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
            print(
                f"elapsed_time={elapsed_time:.2f}s | "
                f"step={step} | psnr={psnr:.2f} | "
                f"loss={loss:.6f} | "
                # number of rays
                f"rays={len(data['rgb'])} | "
                # ray marching samples per ray (occupied space on the ray)
                f"rm_s={results['rm_samples'] / len(data['rgb']):.1f} | "
                # volume rendering samples per ray 
                # (stops marching when transmittance drops below 1e-4)
                f"vr_s={results['vr_samples'] / len(data['rgb']):.1f} | "
            )

    if hparams.deployment:
        save_deployment_model(
            model=model, 
            dataset=train_dataset, 
            save_dir=hparams.deployment_model_path,
        )

    # check if val_dir exists, otherwise create it
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    # save model
    torch.save(
        model.state_dict(),
        os.path.join(val_dir, 'model.pth'),
    )
    # test loop
    progress_bar = tqdm.tqdm(total=len(test_dataset), desc=f'evaluating: ')
    with torch.no_grad():
        model.eval()
        w, h = test_dataset.img_wh
        directions = test_dataset.directions
        test_psnrs = []
        test_ssims = []
        for test_step in range(len(test_dataset)):
            progress_bar.update()
            test_data = test_dataset[test_step]

            rgb_gt = test_data['rgb']
            poses = test_data['pose']

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # get rays
                rays_o, rays_d = get_rays(directions, poses)
                # render image
                results = render(
                    model, 
                    rays_o, 
                    rays_d,
                    test_time=True,
                    exp_step_factor=exp_step_factor,
                )
            # TODO: get rid of this
            rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
            rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
            # get psnr
            val_psnr(rgb_pred, rgb_gt)
            test_psnrs.append(val_psnr.compute())
            val_psnr.reset()
            # get ssim
            val_ssim(rgb_pred, rgb_gt)
            test_ssims.append(val_ssim.compute())
            val_ssim.reset()

            # save test image to disk
            if test_step == 0:
                test_idx = test_data['img_idxs']
                # TODO: get rid of this
                rgb_pred = rearrange(
                    results['rgb'].cpu().numpy(),
                    '(h w) c -> h w c',
                    h=h
                )
                rgb_pred = (rgb_pred * 255).astype(np.uint8)
                depth = depth2img(
                    rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
                imageio.imsave(
                    os.path.join(
                        val_dir, 
                        f'rgb_{test_idx:03d}.png'
                        ),
                    rgb_pred
                )
                imageio.imsave(
                    os.path.join(
                        val_dir, 
                        f'depth_{test_idx:03d}.png'
                    ),
                    depth
                )

        progress_bar.close()
        test_psnr_avg = sum(test_psnrs) / len(test_psnrs)
        test_ssim_avg = sum(test_ssims) / len(test_ssims)
        print(f"evaluation: psnr_avg={test_psnr_avg} | ssim_avg={test_ssim_avg}")


    if hparams.gui:
        ti.reset()
        hparams.ckpt_path = os.path.join(val_dir, 'model.pth')
        taichi_init(hparams)
        dataset = dataset_dict[hparams.dataset_name](
            root_dir=hparams.root_dir,
            downsample=hparams.downsample,
            read_meta=True,
        )
        NGPGUI(
            hparams, 
            model_config, 
            dataset.K, 
            dataset.img_wh, 
            dataset.poses
        ).render()

if __name__ == '__main__':
    main()
