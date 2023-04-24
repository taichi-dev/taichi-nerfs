import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """

    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)
    
    def to(self, device):
        self.images = self.images.to(device)
        self.poses = self.poses.to(device)
        self.K = self.K.to(device)
        self.directions = self.directions.to(device)
        return self

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images':  # randomly select images
                # img_idxs = np.random.choice(len(self.poses), self.batch_size)
                img_idxs = torch.randint(
                    0,
                    len(self.poses),
                    size=(self.batch_size,),
                    device=self.images.device,
                )
            elif self.ray_sampling_strategy == 'same_image':  # randomly select ONE image
                # img_idxs = np.random.choice(len(self.poses), 1)[0]
                img_idxs = [idx]
            # randomly select pixels
            # pix_idxs = np.random.choice(self.img_wh[0] * self.img_wh[1],
            #                             self.batch_size)
            x = torch.randint(
                0, self.img_wh[0], size=(self.batch_size,), device=self.images.device
            )
            y = torch.randint(
                0, self.img_wh[1], size=(self.batch_size,), device=self.images.device
            )
            pix_idxs = y * self.img_wh[0] + x
            rays = self.rays[img_idxs, pix_idxs]
            sample = {
                'img_idxs': img_idxs,
                'pix_idxs': pix_idxs,
                'rgb': rays[:, :3]
            }
            # HDR-NeRF data
            if self.rays.shape[-1] == 4:
                sample['exposure'] = rays[:, 3:]
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
             # if ground truth available
            if len(self.rays) > 0: 
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                # HDR-NeRF data
                if rays.shape[1] == 4:  
                    # same exposure for all rays
                    sample['exposure'] = rays[0, 3]  

        return sample
