from typing import Callable, Optional

# from .custom_functions import TruncExp
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd

from .hash_encoder import HashEncoder
from .ray_march import RayMarcher
from .rendering import NEAR_DISTANCE
from .spherical_harmonics import DirEncoder
from .utils import morton3D, morton3D_invert, packbits
from .volume_train import VolumeRenderer as VolumeRendererTaichi


class TruncExp(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))


class TaichiNGP(nn.Module):

    def __init__(self, args, scale, rgb_act='Sigmoid'):
        super().__init__()
        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
        self.grid_size = 128
        self.register_buffer(
            'density_bitfield',
            torch.zeros(self.cascades * self.grid_size**3 // 8,
                        dtype=torch.uint8))

        # constants
        L = 16
        F = 2
        log2_T = 19
        N_min = 16
        b = np.exp(np.log(2048 * scale / N_min) / (L - 1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')
        self.b = b
        # self.b = self.hash_encoder.native_tcnn_module.hyperparams(
        # )['per_level_scale']

        self.ray_marching = RayMarcher(args.batch_size)

        self.hash_encoder = HashEncoder(self.b,
                                        args.batch_size,
                                        half2_opt=args.half2_opt)

        self.dir_encoder = DirEncoder(args.batch_size)

        self.render_func = VolumeRendererTaichi(args.batch_size)

        self.xyz_encoder = \
            MLP(
                input_dim=32,
                output_dim=16,
                net_depth=1,
                net_width=64,
                bias_enabled=False,
            )

        self.rgb_net = \
            MLP(
                input_dim=32,
                output_dim=3,
                net_depth=2,
                net_width=64,
                bias_enabled=False,
                output_activation=nn.Sigmoid()
            )

        if self.rgb_act == 'None':  # rgb_net output is log-radiance
            for i in range(3):  # independent tonemappers for r,g,b
                tonemapper_net = \
                    MLP(
                        input_dim=1,
                        output_dim=1,
                        net_depth=1,
                        net_width=64,
                        bias_enabled=False,
                        output_activation=nn.Sigmoid()
                    )
                setattr(self, f'tonemapper_net_{i}', tonemapper_net)

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        embedding = self.hash_encoder(x)
        h = self.xyz_encoder(embedding)
        sigmas = TruncExp.apply(h[:, 0])
        if return_feat:
            return sigmas, h
        return sigmas

    def log_radiance_to_rgb(self, log_radiances, **kwargs):
        """
        Convert log-radiance to rgb as the setting in HDR-NeRF.
        Called only when self.rgb_act == 'None' (with exposure)

        Inputs:
            log_radiances: (N, 3)

        Outputs:
            rgbs: (N, 3)
        """
        if 'exposure' in kwargs:
            log_exposure = torch.log(kwargs['exposure'])
        else:  # unit exposure by default
            log_exposure = 0

        out = []
        for i in range(3):
            inp = log_radiances[:, i:i + 1] + log_exposure
            out += [getattr(self, f'tonemapper_net_{i}')(inp)]
        rgbs = torch.cat(out, 1)
        return rgbs

    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, h = self.density(x, return_feat=True)
        d = d / torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d + 1) / 2)
        rgbs = self.rgb_net(torch.cat([d, h], 1))

        if self.rgb_act == 'None':  # rgbs is log-radiance
            if kwargs.get('output_radiance', False):  # output HDR map
                rgbs = TruncExp.apply(rgbs)
            else:  # convert to LDR using tonemapper networks
                rgbs = self.log_radiance_to_rgb(rgbs, **kwargs)

        return sigmas, rgbs

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.

        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold

        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3),
                                    dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(
                self.density_grid[c] > density_threshold)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(len(indices2), (M, ),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1,
                                  indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=32**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a')  # (N_cams, 3, 3)
        w2c_T = -w2c_R @ poses[:, :3, 3:]  # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i + chunk] / (self.grid_size - 1) * 2 - 1
                s = min(2**(c - 1), self.scale)
                half_grid_size = s / self.grid_size
                xyzs_w = (xyzs * (s - half_grid_size)).T  # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N_cams, 3, chunk)
                uvd = K @ xyzs_c  # (N_cams, 3, chunk)
                uv = uvd[:, :2] / uvd[:, 2:]  # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2] >=
                                  NEAR_DISTANCE) & in_image  # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2] <
                                   NEAR_DISTANCE) & in_image  # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count > 0) & (~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self,
                            density_threshold,
                            warmup=False,
                            decay=0.95,
                            erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup:  # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(
                self.grid_size**3 // 4, density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c - 1), self.scale)
            half_grid_size = s / self.grid_size
            xyzs_w = (coords /
                      (self.grid_size - 1) * 2 - 1) * (s - half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1 / self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid > 0].mean().item()

        packbits(
            self.density_grid.reshape(-1).contiguous(),
            min(mean_density, density_threshold), self.density_bitfield)


class MLP(nn.Module):
    '''
        A simple MLP with skip connections from:
        https://github.com/KAIR-BAIR/nerfacc/blob/master/examples/radiance_fields/mlp.py
    '''

    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled))
            if ((self.skip_layer is not None) and (i % self.skip_layer == 0)
                    and (i > 0)):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(in_features,
                                          self.output_dim,
                                          bias=bias_enabled)
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):

        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    # @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if ((self.skip_layer is not None) and (i % self.skip_layer == 0)
                    and (i > 0)):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x
