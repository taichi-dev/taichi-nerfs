from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange
from kornia.utils.grid import create_meshgrid3d
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd

from .rendering import NEAR_DISTANCE
from .spherical_harmonics import DirEncoder
from .triplane import TriPlaneEncoder
from .utils import morton3D, morton3D_invert, packbits
from .volume_train import VolumeRenderer
from .sh_utils import eval_sh


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


class NGP(nn.Module):

    def __init__(
            self,
            scale: float=0.5,
            # position encoder config
            pos_encoder_type: str='hash',
            levels: int=16, # number of levels in hash table
            feature_per_level: int=2, # number of features per level
            log2_T: int=19, # maximum number of entries per level 2^19
            base_res: int=16, # minimum resolution of  hash table
            max_res: int=2048, # maximum resolution of the hash table
            half_opt: bool=False, # whether to use half precision, available for hash
            # mlp config
            xyz_net_width: int=64,
            xyz_net_depth: int=1,
            xyz_net_out_dim: int=16,
            rgb_net_depth: int=2,
            rgb_net_width: int=64,
        ):
        super().__init__()

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
            torch.zeros(
                self.cascades * self.grid_size**3 // 8,
                dtype=torch.uint8
            )
        )

        self.register_buffer(
            'density_grid',
            torch.zeros(self.cascades, self.grid_size**3),
        )
        self.register_buffer(
            'grid_coords',
            create_meshgrid3d(
                self.grid_size,
                self.grid_size,
                self.grid_size,
                False,
                dtype=torch.int32
            ).reshape(-1, 3)
        )

        if pos_encoder_type == 'hash':
            if half_opt:
                from .hash_encoder_half import HashEncoder
            else:
                from .hash_encoder import HashEncoder

            self.pos_encoder = HashEncoder(
                max_params=2**log2_T,
                base_res=base_res,
                max_res=max_res,
                levels=levels,
                feature_per_level=feature_per_level,
            )
        elif pos_encoder_type == 'triplane':
            self.pos_encoder = TriPlaneEncoder(
                base_res=16,
                max_res=max_res,
                levels=8,
                feature_per_level=4,
            )
        else:
            raise NotImplementedError

        self.xyz_encoder = MLP(
            input_dim=self.pos_encoder.out_dim,
            output_dim=xyz_net_out_dim,
            net_depth=xyz_net_depth,
            net_width=xyz_net_width,
            bias_enabled=False,
        )

        self.dir_encoder = DirEncoder()

        rgb_input_dim = (
            self.dir_encoder.out_dim + \
            self.xyz_encoder.output_dim
        )
        self.rgb_net =  MLP(
            input_dim=rgb_input_dim,
            output_dim=3,
            net_depth=rgb_net_depth,
            net_width=rgb_net_width,
            bias_enabled=False,
            output_activation=nn.Sigmoid()
        )

        self.render_func = VolumeRenderer()

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature
        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        embedding = self.pos_encoder(x)
        h = self.xyz_encoder(embedding)
        sigmas = TruncExp.apply(h[:, 0])
        if return_feat:
            return sigmas, h
        return sigmas

    def forward(self, x, d):
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

class VoxelGrid(NGP):
    def __init__(
            self,
            scale: float=0.5,
            half_opt: bool=False, # whether to use half precision, available for hash
            # grid configs
            sh_degree: int=2,
            grid_size: int=256,
            grid_radius: float=0.0125,
            origin_sh: float=0.,
            origin_sigma: float=0.1,
        ):
        super().__init__()

        self.sh_degree = sh_degree
        self.grid_size = grid_size
        self.grid_radius = grid_radius
        self.scale = scale
        self.origin_sh = origin_sh
        self.origin_sigma = origin_sigma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sh_dim = (1 + self.sh_degree) ** 2
        self.cascades = max(1 + int(np.ceil(np.log2(2 * self.scale))), 1)
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)
        self.register_buffer(
            'density_bitfield',
            torch.zeros(
                self.cascades * self.grid_size**3 // 8,
                dtype=torch.uint8
            )
        )

        self.register_buffer(
            'density_grid',
            torch.zeros(self.cascades, self.grid_size**3),
        )
        self.register_buffer(
            'grid_coords',
            create_meshgrid3d(
                self.grid_size,
                self.grid_size,
                self.grid_size,
                False,
                dtype=torch.int32
            ).reshape(-1, 3)
        )

        # initialize the grids
        self.initialize_grid()

    def initialize_grid(self):
        """
        Initialize a voxel grid according to the configs

        Params:
            grid_normalized_coords: (sx * sy * sz, 3), normalized coordinates of the grids
            grid_fields: (sx, sy, sz, sh_dim + 1), data fields(sh and density) of the grids
        """
        if isinstance(self.grid_size, float) or isinstance(self.grid_size, int):
            grid_res = [self.grid_size] * 3
        else:
            grid_res = self.grid_size
        assert len(grid_res) == 3, "grid resolution must be 3 dimension"
        sx, sy, sz = grid_res[0], grid_res[1], grid_res[2]
        gx_idxs, gy_idxs, gz_idsx= torch.arange(sx, device=self.device), \
                                   torch.arange(sy, device=self.device), \
                                   torch.arange(sz, device=self.device)
        cx_idxs, cy_idxs, cz_idxs = torch.meshgrid(gx_idxs, gy_idxs, gz_idsx, indexing='ij')

        # self.grid_idxs = create_meshgrid3d(grid_res[0], grid_res[1], grid_res[2], False, dtype=torch.int32).reshape(-1, 3)

        # center grid
        cx_idxs, cy_idxs, cz_idxs = cx_idxs - np.ceil(sx / 2) + 1, \
                                    cy_idxs - np.ceil(sy / 2) + 1, \
                                    cz_idxs - np.ceil(sz / 2) + 1

        # edit grid spacing
        cx, cy, cz = cx_idxs * self.grid_radius, cy_idxs * self.grid_radius, cz_idxs * self.grid_radius

        grids = torch.stack([cx, cy, cz], dim=-1)
        self.grid_normalized_coords = grids.reshape(sx * sy * sz, 3)

        # initialize grid datas
        self.sh_fields = nn.Parameter(
            torch.ones(
                (grids.shape[0],  grids.shape[1], grids.shape[2], self.sh_dim * 3),
                dtype=torch.float32,
                device=self.device
            ) * self.origin_sh,
            requires_grad=True,
        )

        self.density_fields = nn.Parameter(
            torch.ones(
                (grids.shape[0],  grids.shape[1], grids.shape[2], 1),
                dtype=torch.float32,
                device=self.device
            ) * self.origin_sigma,
            requires_grad=True,
        )

        self.grid_fields = torch.cat((self.sh_fields, self.density_fields), dim=3)

    def out_of_grid(self, idx):
        """
        Checks if the given indices are out of bounds of the grid.

        Inputs:
            idx: (N, 3), the indices of the points to check
        Outputs:
            idx_valid_mask: (N, 1)

        """
        x_idx, y_idx, z_idx = idx.unbind(-1)

        # find which points are outside the grid
        sx, sy, sz, _ = self.grid_fields.shape
        x_idx_valid = (x_idx < sx) & (x_idx >= 0)
        y_idx_valid = (y_idx < sy) & (y_idx >= 0)
        z_idx_valid = (z_idx < sz) & (z_idx >= 0)
        idx_valid_mask = x_idx_valid & y_idx_valid & z_idx_valid

        return idx_valid_mask

    def fix_out_of_grid(self, idx):
        x_idx, y_idx, z_idx = idx.unbind(-1)

        # find which points are outside the grid
        sx, sy, sz, _ = self.grid_fields.shape
        x_idx %= sx
        y_idx %= sy
        z_idx %= sz

        return x_idx, y_idx, z_idx

    def normalize_samples(self, pts):
        return (pts- self.grid_normalized_coords.min(0)[0]) / self.grid_radius

    def trilinear_interpolation(self, bundles, weight_a, weight_b):
        c00 = bundles[0] * weight_a[:, 2:] + bundles[1] * weight_b[:, 2:]
        c01 = bundles[2] * weight_a[:, 2:] + bundles[3] * weight_b[:, 2:]
        c10 = bundles[4] * weight_a[:, 2:] + bundles[5] * weight_b[:, 2:]
        c11 = bundles[6] * weight_a[:, 2:] + bundles[7] * weight_b[:, 2:]
        c0 = c00 * weight_a[:, 1:2] + c01 * weight_b[:, 1:2]
        c1 = c10 * weight_a[:, 1:2] + c11 * weight_b[:, 1:2]
        results = c0 * weight_a[:, :1] + c1 * weight_b[:, :1]

        return results

    def query_grids(self, idx, use_trilinear=False):
        """
        Query the grid fields at the given indices.

        Input:
            idx: (N, 3)
            use_trilinear: bool, whether use trilinear interpolation

        Outputs:
            samples_results: (N, sh_dim + 1)
        """
        aligned_idx = torch.round(idx).to(torch.long)

        idx_mask = self.out_of_grid(aligned_idx)
        x_idx, y_idx, z_idx = self.fix_out_of_grid(aligned_idx)

        query_results = self.grid_fields[x_idx, y_idx, z_idx]
        query_results = query_results * idx_mask.unsqueeze(-1)  # zero the samples that are out of the grid

        if use_trilinear:
            weight_b = torch.abs(idx - aligned_idx)
            weight_a = 1.0 - weight_b
            query_sh, query_density = query_results[..., :-1], query_results[..., -1]
            samples_density = self.trilinear_interpolation(query_density, weight_a, weight_b)
            samples_sh = self.trilinear_interpolation(query_sh, weight_a, weight_b)
            samples_result = torch.cat((samples_sh, samples_density), dim=3)
            return samples_result

        return query_results


    def forward(self, pts, dirs):
        normalized_idx = self.normalize_samples(pts)
        samples_result = self.query_grids(normalized_idx)
        samples_sh, samples_density = samples_reuslt[..., :-1], samples_reuslt[..., -1]
        samples_rgb = torch.empty((pts.shape(0), pts.shape(1), 3), device=samples_sh.device)
        sh_dim = self.net.sh_dim
        for i in range(3):
            sh_coeffs = samples_sh[:, :, sh_dim*i:sh_dim*(i+1)]
            samples_rgb[:, :, i] = eval_sh(self.sh_degree, sh_coeffs, viewdirs)
        return samples_density, samples_rgb


MODEL_DICT = {
    'ngp': NGP,
    'svox': VoxelGrid,
}
