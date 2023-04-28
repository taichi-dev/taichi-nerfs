import numpy as np
import taichi as ti
import torch
from taichi.math import uvec3
from torch.cuda.amp import custom_bwd, custom_fwd

from .utils import (
    data_type, ti2torch, ti2torch_grad, 
    torch2ti, torch2ti_grad, torch_type
)

half2 = ti.types.vector(n=2, dtype=ti.f16)


@ti.kernel
def random_initialize(data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = (ti.random() * 2.0 - 1.0) * 1e-4


@ti.kernel
def ti_copy(data1: ti.template(), data2: ti.template()):
    for I in ti.grouped(data1):
        data1[I] = data2[I]


@ti.kernel
def ti_copy_array(data1: ti.types.ndarray(), data2: ti.types.ndarray()):
    for I in ti.grouped(data1):
        data1[I] = data2[I]


@ti.kernel
def ti_copy_field_array(data1: ti.template(), data2: ti.types.ndarray()):
    for I in ti.grouped(data1):
        data1[I] = data2[I]


@ti.kernel
def hash_encode_kernel(
        xyzs: ti.template(), table: ti.template(),
        xyzs_embedding: ti.template(), offsets: ti.template(), B: ti.i32,
        per_level_scale: ti.f32):

    # get hash table embedding
    # ti.loop_config(block_dim=32)
    # TODO: use var to replace 4
    for i, level in ti.ndrange(B, 4):
        xyz = ti.Vector([xyzs[i, 0], xyzs[i, 1], xyzs[i, 2]])

        # TODO: use base_res var to replace 16
        scale = 32 * ti.exp(level * ti.log(per_level_scale)) - 1.0
        resolution = ti.cast(ti.ceil(scale), ti.uint32) + 1

        offset = offsets[level] * 4

        pos = xyz * scale + 0.5
        pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)
        pos -= pos_grid_uint

        local_feature_0 = 0.0
        local_feature_1 = 0.0
        local_feature_2 = 0.0
        local_feature_3 = 0.0

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

            index = 0
            stride = 1
            for c_ in ti.static(range(3)):
                index += pos_grid_local[c_] * stride
                stride *= resolution

            index_table = offset + index * 4
            index_table_int = ti.cast(index_table, ti.int32)
            local_feature_0 += w * table[index_table_int]
            local_feature_1 += w * table[index_table_int + 1]
            local_feature_2 += w * table[index_table_int + 2]
            local_feature_3 += w * table[index_table_int + 3]

        # TODO: use static for loop 
        xyzs_embedding[i, level * 4] = local_feature_0
        xyzs_embedding[i, level * 4 + 1] = local_feature_1
        xyzs_embedding[i, level * 4 + 2] = local_feature_2
        xyzs_embedding[i, level * 4 + 3] = local_feature_3




class HashEncoder(torch.nn.Module):

    def __init__(self,
                 b=1.3195079565048218,
                 max_params: float=2**19,
                 hash_level: int=16,
                 base_res: float=16,
                 batch_size=8192,
                 data_type=data_type,
                 feature_per_level=2):
        super(HashEncoder, self).__init__()

        self.per_level_scale = b

        print("per_level_scale: ", b)
        self.offsets = ti.field(ti.i32, shape=(hash_level, ))
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
            offset_ += params_in_level
        print("offset_: ", offset_)

        self.total_hash_size = offset_ * feature_per_level
        print("total_hash_size: ", self.total_hash_size)

        self.hash_table = torch.nn.Parameter(torch.zeros(self.total_hash_size,
                                                         dtype=torch_type),
                                             requires_grad=True)
        random_initialize(self.hash_table)

        output_dim = feature_per_level*hash_level

        self.parameter_fields = ti.field(
            data_type,
            shape=(self.total_hash_size, ),
            needs_grad=True
        )
        self.output_fields = ti.field(
            dtype=data_type,
            shape=(batch_size * 1024, output_dim),
            needs_grad=True
        )
        self.torch2ti = torch2ti
        self.ti2torch = ti2torch
        self.ti2torch_grad = ti2torch_grad
        self.torch2ti_grad = torch2ti_grad

        self._hash_encode_kernel = hash_encode_kernel

        self.input_fields = ti.field(
            dtype=data_type,
            shape=(batch_size * 1024, 3),
            needs_grad=True
        )

        self.register_buffer(
            'hash_grad',
            torch.zeros(self.total_hash_size, dtype=torch_type),
            persistent=False
        )
        self.register_buffer(
            'output_embedding',
            torch.zeros(batch_size * 1024, output_dim, dtype=torch_type),
            persistent=False
        )

        class _module_function(torch.autograd.Function):

            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, input_pos, params):
                output_embedding = self.output_embedding[:input_pos.
                                                         shape[0]].contiguous(
                                                         )
                torch2ti(self.input_fields, input_pos.contiguous())
                self.torch2ti(self.parameter_fields, params.contiguous())

                self._hash_encode_kernel(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.offsets,
                    input_pos.shape[0],
                    self.per_level_scale,
                )
                self.ti2torch(self.output_fields, output_embedding)

                return output_embedding

            @staticmethod
            @custom_bwd
            def backward(ctx, doutput):

                self.zero_grad()

                self.torch2ti_grad(self.output_fields, doutput.contiguous())
                self._hash_encode_kernel.grad(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.offsets,
                    doutput.shape[0],
                    self.per_level_scale,
                )
                self.ti2torch_grad(self.parameter_fields,
                                   self.hash_grad.contiguous())
                return None, self.hash_grad

        self._module_function = _module_function

    def zero_grad(self):
        self.parameter_fields.grad.fill(0.)

    def forward(self, positions):
        return self._module_function.apply(positions, self.hash_table)
