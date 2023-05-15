import torch
import taichi as ti
from taichi.math import uvec3, ivec3
from torch.cuda.amp import custom_bwd, custom_fwd

from .utils import (
    data_type, 
    torch_type, 
    scale_in_level_np
)

def build_triplane_encoder_kernel(
    log_scale,
    base_res: int=16,
    max_res: int=2048,
    levels: int=16,
    feat_dim: int=2,
    max_param: int=(4096**2)*3*2
):
    # Type
    uvec6 = ti.types.vector(n=6, dtype=ti.u32)
    tf_vec3 = ti.types.vector(n=3, dtype=data_type)
    tf_vec6 = ti.types.vector(n=6, dtype=data_type)
    offset = int(max_res**2)*feat_dim

    @ti.func
    def grid_scale(level, log_scale, base_res):
        exp_scale = ti.exp(level * log_scale)
        return base_res * exp_scale - 1.0
    
    @ti.func
    def grid_resolution(scale):
        return ti.uint32(ti.ceil(scale)) + 1

    @ti.kernel
    def triplane_encoder_kernel(
        xyzs: ti.types.ndarray(), 
        plane_table: ti.types.ndarray(),  
        output_embedding: ti.types.ndarray(),  
        B: ti.i32,
    ):
        ti.loop_config(block_dim=256)
        for i, sn in ti.ndrange(B, levels*feat_dim):
            j = sn // levels
            level = sn % levels
            xyz = tf_vec6([
                xyzs[i, 0], xyzs[i, 1],
                xyzs[i, 1], xyzs[i, 2],
                xyzs[i, 2], xyzs[i, 0],
            ])
            scale = grid_scale(level, log_scale, base_res)
            resolution = grid_resolution(scale)
            local_feat = tf_vec3(0)

            # plane query
            pos = xyz * (resolution - 1) + 0.5
            pos_grid = ti.cast(ti.floor(pos), ti.uint32)
            pos -= ti.cast(pos_grid, data_type)

            for idx in ti.static(range(4)):
                w = tf_vec3(1.)
                pos_grid_local = uvec6(0)

                for d in ti.static(range(2)):
                    if (idx & (1 << d)) == 0:
                        pos_grid_local[d::2] = pos_grid[d::2]
                        w *= 1 - pos[d::2]
                    else:
                        pos_grid_local[d::2] = pos_grid[d::2] + 1
                        w *= pos[d::2]

                # convert pos_grid_local to high res
                pos_grid_local_ori = ti.cast(
                    pos_grid_local / resolution * (max_res-1),
                    ti.u32
                )

                stride = ti.u32(1)
                index = uvec3(0)
                for i in ti.static(range(2)):
                    index += pos_grid_local_ori[i::2] * stride
                    stride *= ti.cast(max_res, ti.u32)

                for fd in ti.static(range(3)):
                    plane_base = offset * fd
                    index_base = index[fd] * feat_dim
                    final_index = ti.u32(plane_base + index_base + j)
                    # if final_index >= max_param:
                    #     print(f"shit happend: {plane_base}, {index_base}, {final_index}, pos_grid_local: {pos_grid_local}, pos_grid_local_ori: {pos_grid_local_ori}")
                    local_feat[fd] += (
                        w[fd] * plane_table[final_index]
                    )

            cumprod = 1.
            for fd in ti.static(range(3)):
                cumprod *= local_feat[fd]

            output_embedding[i, sn] = cumprod

    return triplane_encoder_kernel


class TriPlaneEncoder(torch.nn.Module):

    def __init__(
            self,
            base_res: int=16,
            max_res: int=2048,
            levels: int=16,
            feature_per_level: int=2,
        ):
        super(TriPlaneEncoder, self).__init__()

        self.base_res = base_res
        self.max_res = max_res
        self.levels = levels
        self.feature_per_level = feature_per_level
        self.out_dim = levels * feature_per_level

        self.log_b = scale_in_level_np(
            base_res=base_res,
            max_res=max_res,
            levels=levels,
        )

        self.total_param_size = (
            int(self.max_res**2) * 3 * self.feature_per_level
        )
        self.plane_embedding = torch.nn.Parameter(
            torch.zeros(
                self.total_param_size, 
                dtype=torch_type,
            ),
            requires_grad=True
        )
        torch.nn.init.uniform_(self.plane_embedding)

        print(
            f'TriPlane Encoder: '
            f'base_res={base_res} '
            f'max_res={max_res} '
            f'levels={levels} '
            f'feat_per_level={feature_per_level} '
            f'per_level_scale={self.log_b} '
            f'total_param_size={self.total_param_size} '
        )
                    

        self._encode_kernel = build_triplane_encoder_kernel(
            self.log_b,
            base_res=self.base_res,
            max_res=self.max_res,
            levels=self.levels,
            feat_dim=self.feature_per_level,
            max_param=self.total_param_size,
        )

        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_pos, params):

                output_embedding = torch.empty(
                    input_pos.shape[0], self.out_dim,
                    dtype=torch_type,
                    device=input_pos.device, 
                    requires_grad=True,
                )
                # print("output_embedding shape: ", output_embedding.shape)
                self._encode_kernel(
                    input_pos,
                    params,
                    output_embedding,
                    input_pos.shape[0],
                )
                ctx.save_for_backward(
                    input_pos, 
                    output_embedding, 
                    params
                )
                # print("output_embedding: ", output_embedding)

                return output_embedding

            @staticmethod
            def backward(ctx, doutput):

                input_pos, output_embedding, params = ctx.saved_tensors
                output_embedding.grad = doutput

                self._encode_kernel.grad(
                    input_pos,
                    params,
                    output_embedding,
                    input_pos.shape[0],
                )
                return None, params.grad

        self._module_function = _module_function.apply

    def forward(self, positions):
        return self._module_function(
            positions.contiguous(), 
            self.plane_embedding.contiguous()
        )
