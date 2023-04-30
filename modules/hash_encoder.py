import numpy as np
import taichi as ti
import torch
from taichi.math import uvec3
from torch.cuda.amp import custom_bwd, custom_fwd

from .utils import (data_type, torch_type)

half2 = ti.types.vector(n=2, dtype=ti.f16)

@ti.kernel
def random_initialize(data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = (ti.random() * 2.0 - 1.0) * 1e-4

@ti.func
def fast_hash(pos_grid_local):
    result = ti.uint32(0)
    # primes = uvec3(ti.uint32(1), ti.uint32(1958374283), ti.uint32(2654435761))
    primes = uvec3(ti.uint32(1), ti.uint32(2654435761), ti.uint32(805459861))
    for i in ti.static(range(3)):
        result ^= ti.uint32(pos_grid_local[i]) * primes[i]
    return result


@ti.func
def under_hash(pos_grid_local, resolution):
    result = ti.uint32(0)
    stride = ti.uint32(1)
    for i in ti.static(range(3)):
        result += ti.uint32(pos_grid_local[i] * stride)
        stride *= resolution
    return result


@ti.func
def grid_pos2hash_index(indicator, pos_grid_local, resolution, map_size):
    hash_result = ti.uint32(0)
    if indicator == 1:
        hash_result = under_hash(pos_grid_local, resolution)
    else:
        hash_result = fast_hash(pos_grid_local)

    return hash_result % map_size


@ti.kernel
def hash_encode_kernel(
        xyzs: ti.types.ndarray(), 
        table: ti.types.ndarray(),
        xyzs_embedding: ti.types.ndarray(), 
        hash_map_indicator: ti.types.ndarray(),
        hash_map_sizes_field: ti.types.ndarray(), 
        offsets: ti.types.ndarray(), 
        B: ti.i32,
        per_level_scale: ti.f32
    ):

    # get hash table embedding
    ti.loop_config(block_dim=32)
    for i, level in ti.ndrange(B, 16):
        xyz = ti.Vector([xyzs[i, 0], xyzs[i, 1], xyzs[i, 2]])

        scale = 16 * ti.exp(level * ti.log(per_level_scale)) - 1.0
        resolution = ti.cast(ti.ceil(scale), ti.uint32) + 1

        offset = offsets[level] * 2

        pos = xyz * scale + 0.5
        pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)
        pos -= pos_grid_uint

        indicator = hash_map_indicator[level]
        map_size = hash_map_sizes_field[level]

        local_feature_0 = 0.0
        local_feature_1 = 0.0

        for idx in ti.static(range(8)):
            w = 1.
            pos_grid_local = uvec3(0)

            for d in ti.static(range(3)):
                if (idx & (1 << d)) == 0:
                    pos_grid_local[d] = pos_grid_uint[d]
                    w *= 1 - pos[d]
                else:
                    pos_grid_local[d] = pos_grid_uint[d] + 1
                    w *= pos[d]

            index = grid_pos2hash_index(indicator, pos_grid_local, resolution,
                                        map_size)
            index_table = offset + index * 2
            index_table_int = ti.cast(index_table, ti.int32)
            local_feature_0 += w * table[index_table_int]
            local_feature_1 += w * table[index_table_int + 1]

        xyzs_embedding[i, level * 2] = local_feature_0
        xyzs_embedding[i, level * 2 + 1] = local_feature_1


@ti.kernel
def hash_encode_kernel_half2(
        xyzs: ti.template(), table: ti.template(),
        xyzs_embedding: ti.template(), hash_map_indicator: ti.template(),
        hash_map_sizes_field: ti.template(), offsets: ti.template(), B: ti.i32,
        per_level_scale: ti.f16):

    # get hash table embedding
    ti.loop_config(block_dim=32)
    for i, level in ti.ndrange(B, 16):
        xyz = ti.Vector([xyzs[i, 0], xyzs[i, 1], xyzs[i, 2]])

        scale = 16 * ti.exp(level * ti.log(per_level_scale)) - 1.0
        resolution = ti.cast(ti.ceil(scale), ti.uint32) + 1

        offset = offsets[level]

        pos = xyz * scale + 0.5
        pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)
        pos -= pos_grid_uint

        indicator = hash_map_indicator[level]
        map_size = hash_map_sizes_field[level]

        local_feature = half2(0.0)
        for idx in ti.static(range(8)):
            w = ti.f32(1.0)
            pos_grid_local = uvec3(0)

            for d in ti.static(range(3)):
                if (idx & (1 << d)) == 0:
                    pos_grid_local[d] = pos_grid_uint[d]
                    w *= 1 - pos[d]
                else:
                    pos_grid_local[d] = pos_grid_uint[d] + 1
                    w *= pos[d]

            index = grid_pos2hash_index(indicator, pos_grid_local, resolution,
                                        map_size)

            index_table = offset + index
            index_table_int = ti.cast(index_table, ti.int32)

            local_feature += w * table[index_table_int]
        xyzs_embedding[i, level] = local_feature


class HashEncoder(torch.nn.Module):

    def __init__(
        self,
        b=1.3195079565048218,
        max_params: float=2**19,
        hash_level: int=16,
        base_res: float=16
    ):
        super(HashEncoder, self).__init__()

        self.per_level_scale = b

        # per_level_scale = 1.3195079565048218
        print("per_level_scale: ", b)
        self.register_buffer(
            'offsets',
            torch.zeros(16, dtype=torch.int32),
            persistent=False
        )
        self.register_buffer(
            'hash_map_sizes',
            torch.zeros(16, dtype=torch.int32),
            persistent=False
        )
        self.register_buffer(
            'hash_map_indicator',
            torch.zeros(16, dtype=torch.int32),
            persistent=False
        )

        offset_ = 0
        for i in range(hash_level):
            resolution = int(
                np.ceil(base_res * np.exp(i * np.log(self.per_level_scale)) -
                        1.0)) + 1
            params_in_level = resolution**3
            params_in_level = int(resolution**
                                  3) if params_in_level % 8 == 0 else int(
                                      (params_in_level + 8 - 1) / 8) * 8
            params_in_level = min(max_params, params_in_level)
            self.offsets[i] = offset_
            self.hash_map_sizes[i] = params_in_level
            self.hash_map_indicator[
                i] = 1 if resolution**3 <= params_in_level else 0
            offset_ += params_in_level
        print("total_hash_size: ", offset_)

        self.total_hash_param = offset_ * 2
        print("total_hash_param: ", self.total_hash_param)

        self.hash_table = torch.nn.Parameter(torch.zeros(
            self.total_hash_param,
            dtype=torch_type),
            requires_grad=True
        )
        random_initialize(self.hash_table)

        self._hash_encode_kernel = hash_encode_kernel

        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_pos, params):

                output_embedding = torch.empty(
                    input_pos.shape[0], 32,
                    dtype=torch_type,
                    device=input_pos.device, 
                    requires_grad=True,
                )
                ctx.save_for_backward(
                    input_pos, 
                    output_embedding, 
                    params
                )

                self._hash_encode_kernel(
                    input_pos.contiguous(),
                    params.contiguous(),
                    output_embedding.contiguous(),
                    self.hash_map_indicator.contiguous(),
                    self.hash_map_sizes.contiguous(),
                    self.offsets.contiguous(),
                    input_pos.shape[0],
                    self.per_level_scale,
                )

                return output_embedding

            @staticmethod
            def backward(ctx, doutput):
                input_pos, output_embedding, params = ctx.saved_tensors
                output_embedding.grad = doutput

                self._hash_encode_kernel.grad(
                    input_pos.contiguous(),
                    params.contiguous(),
                    output_embedding.contiguous(),
                    self.hash_map_indicator.contiguous(),
                    self.hash_map_sizes.contiguous(),
                    self.offsets.contiguous(),
                    input_pos.shape[0],
                    self.per_level_scale,
                )
                return None, params.grad

        self._module_function = _module_function

    def forward(self, positions):
        return self._module_function.apply(positions, self.hash_table)
