import json
import os

import numpy as np
import torch
from tqdm import tqdm

from .base import BaseDataset
from .color_utils import read_image
from .ray_utils import get_ray_directions

class NGPDataset(BaseDataset):

    def __init__(self, root_dir, split='train', downsample=1.0, read_meta=True):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if read_meta:
            self.read_meta(split)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, "transforms.json"),
                  'r') as f:
            meta = json.load(f)

        w = int(meta['w'] * self.downsample)
        h = int(meta['h'] * self.downsample)
        # fx = fy = 0.5 * 800 / np.tan(
        #     0.5 * meta['camera_angle_x']) * self.downsample
    
        # fx = 0.5 * w / np.tan(0.5 * meta['camera_angle_x']) * self.downsample
        # fy = 0.5 * h / np.tan(0.5 * meta['camera_angle_y']) * self.downsample
        fx = meta['fl_x'] * self.downsample
        fy = meta['fl_y'] * self.downsample

        K = np.float32([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        with open(os.path.join(self.root_dir, "transforms.json"),
                    'r') as f:
            frames = json.load(f)["frames"]

        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            img_path = os.path.join(
                self.root_dir,
                f"{frame['file_path']}"
            )

            if not os.path.exists(img_path):
                continue
            
            img = read_image(img_path, self.img_wh)
            self.rays += [img]

            c2w = np.array(frame['transform_matrix'])[:3, :4]
            c2w[:, 1:3] *= -1 
            self.poses += [c2w]


        if len(self.rays) > 0:
            self.rays = torch.FloatTensor(np.stack(self.rays))  # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses)  # (N_images, 3, 4)
