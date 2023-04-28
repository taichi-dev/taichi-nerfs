import glob
import os
import time
import warnings

import imageio
import numpy as np
import taichi as ti
import torch
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from einops import rearrange
# models
from kornia.utils.grid import create_meshgrid3d
from modules.losses import NeRFLoss
from modules.networks import TaichiNGP
from modules.rendering import MAX_SAMPLES, render
from modules.utils import load_ckpt, depth2img, save_deployment_model
from opt import get_opts
from show_gui import NGPGUI
# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
# optimizer, losses
# from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
# data
from torch.utils.data import DataLoader
# metrics
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

warnings.filterwarnings("ignore")


class NeRFSystem(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)

        rgb_act = 'Sigmoid'
        self.model = TaichiNGP(
            self.hparams,
            scale=self.hparams.scale,
            rgb_act=rgb_act,
            deployment=self.hparams.deployment,
        )
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
                                   torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer(
            'grid_coords',
            create_meshgrid3d(G, G, G, False,
                              dtype=torch.int32).reshape(-1, 3))

        self.tic = 0.0
        self.test_id = 0
        self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/training'
        os.makedirs(self.val_dir, exist_ok=True)
        self.validation_step_outputs =[]
        
    def forward(self, batch, split):
        if split == 'train':
            poses = self.poses[batch['img_idxs'].type(torch.long)]
            directions = self.directions[batch['pix_idxs'].type(torch.long)]
        else:
            poses = batch['pose']
            directions = self.directions

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {
            'test_time': split != 'train',
            'random_bg': self.hparams.random_bg
        }
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1 / 256

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {
            'root_dir': self.hparams.root_dir,
            'downsample': self.hparams.downsample
        }
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.register_buffer('directions',
                             self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        load_ckpt(self.model, self.hparams.ckpt_path)

        self.test_dataset = dataset(split='test', **kwargs)
        if self.hparams.dataset_name == 'colmap':
            self.test_dataset_traj = dataset(split='test_traj', **kwargs)
            self.test_saving_training = 20000 // len(self.test_dataset_traj)
        else:
            self.test_saving_training = 20000 // len(self.test_dataset)

        self.test_saving_training = 5

    def configure_optimizers(self):
        # define additional parameters
        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']:
                net_params += [p]

        opts = []
        self.net_opt = torch.optim.Adam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        net_sch = CosineAnnealingLR(self.net_opt, self.hparams.num_epochs,
                                    self.hparams.lr / 30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses, self.train_dataset.img_wh)
        self.tic = time.time()

    def training_step(self, batch, batch_nb, *args):
        if self.global_step % self.update_interval == 0:
            self.model.update_density_grid(
                0.01 * MAX_SAMPLES / 3**0.5,
                warmup=self.global_step < self.warmup_steps,
                erode=self.hparams.dataset_name == 'colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        loss = sum(lo.mean() for lo in loss_d.values())

        if not self.hparams.perf:

            with torch.no_grad():
                self.train_psnr(results['rgb'], batch['rgb'])
            self.log('lr', self.net_opt.param_groups[0]['lr'])
            self.log('train/loss', loss)
            # ray marching samples per ray (occupied space on the ray)
            self.log('train/rm_s', results['rm_samples'] / len(batch['rgb']),
                     True)
            # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
            self.log('train/vr_s', results['vr_samples'] / len(batch['rgb']),
                     True)
            self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        self.elapsed_time = time.time() - self.tic
        print(f"total training time: {system.elapsed_time:.2f}")

        torch.cuda.empty_cache()
        if not hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/rendering'
            os.makedirs(self.val_dir, exist_ok=True)

            self.eval()
            batch_data_set = self.test_dataset_traj if self.hparams.dataset_name == 'colmap' else self.test_dataset
            for batch_val in batch_data_set:
                for k, v in batch_val.items():
                    if isinstance(v, torch.Tensor):
                        batch_val[k] = v.to(self.device)
                self.val_on_training(batch_val)

            imgs = sorted(glob.glob(os.path.join(system.val_dir, 'rgb_*.png')))
            imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                            [imageio.imread(img) for img in imgs],
                            fps=24,
                            macro_block_size=2)
            imgs = sorted(
                glob.glob(os.path.join(system.val_dir, 'depth_*.png')))
            imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                            [imageio.imread(img) for img in imgs],
                            fps=24,
                            macro_block_size=2)

            torch.cuda.empty_cache()
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/'
            os.makedirs(self.val_dir, exist_ok=True)

    def val_on_training(self, batch):
        if self.hparams.dataset_name == 'colmap':
            results = self(batch, split='test_traj')
        else:
            results = self(batch, split='test')
        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)

        idx = batch['img_idxs']
        rgb_pred = rearrange(results['rgb'].cpu().numpy(),
                             '(h w) c -> h w c',
                             h=h)
        rgb_pred = (rgb_pred * 255).astype(np.uint8)
        depth = depth2img(
            rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
        imageio.imsave(os.path.join(self.val_dir, f'rgb_{idx:03d}.png'),
                       rgb_pred)
        imageio.imsave(os.path.join(self.val_dir, f'depth_{idx:03d}.png'),
                       depth)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()

        if not self.hparams.no_save_test:  # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(),
                                 '(h w) c -> h w c',
                                 h=h)
            rgb_pred = (rgb_pred * 255).astype(np.uint8)
            depth = depth2img(
                rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'rgb_{idx:03d}.png'),
                           rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'depth_{idx:03d}.png'),
                           depth)

        self.validation_step_outputs.append(logs)
        return logs

    def on_validation_epoch_end(self):
        outputs=self.validation_step_outputs
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = psnrs.mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = ssims.mean()
        self.log('test/ssim', mean_ssim)


def taichi_init(args):
    taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
    if args.half2_opt:
        taichi_init_args["half2_vectorization"] = True

    ti.init(**taichi_init_args)


if __name__ == '__main__':
    hparams = get_opts()

    taichi_init(hparams)

    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams).to(torch.device('cuda'))

    ckpt_cb = ModelCheckpoint(
        dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
        filename='{epoch:d}',
        save_weights_only=True,
        every_n_epochs=hparams.num_epochs,
        save_on_train_epoch_end=True,
        save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        check_val_every_n_epoch=hparams.num_epochs,
        callbacks=callbacks,
        logger=None,
        enable_model_summary=False,
        accelerator='gpu',
        devices=1,
        # //strategy=None,
        num_sanity_val_steps=0,
        precision=16,
    )

    if hparams.val_only:
        trainer.validate(system, verbose=True)
    else:
        trainer.fit(system)

    if hparams.deployment:
        save_deployment_model(system, hparams.deployment_model_path)

    if not hparams.no_save_test:  # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, 'rgb_*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs],
                        fps=24,
                        macro_block_size=1)
        imgs = sorted(glob.glob(os.path.join(system.val_dir, 'depth_*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs],
                        fps=24,
                        macro_block_size=1)
    
    if hparams.gui:
        ti.reset()
        if not hparams.val_only:
            hparams.ckpt_path = ckpt_cb.best_model_path
        taichi_init(hparams)
        kwargs = {
            'root_dir': hparams.root_dir,
            'downsample': hparams.downsample,
            'read_meta': True
        }
        dataset = dataset_dict[hparams.dataset_name](**kwargs)

        NGPGUI(hparams, dataset.K, dataset.img_wh, dataset.poses).render()

