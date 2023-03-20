import numpy as np
import taichi as ti
import torch
from taichi.math import uvec3, ivec3
from torch.cuda.amp import custom_bwd, custom_fwd

from .utils import (data_type, ti2torch, ti2torch_grad, ti2torch_grad_vec,
                    ti2torch_vec, torch2ti, torch2ti_grad, torch2ti_grad_vec,
                    torch2ti_vec, torch_type)

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


tf_vec8 = ti.types.vector(n=8, dtype=data_type)
tf_vec4 = ti.types.vector(n=4, dtype=data_type)
tf_vec3 = ti.types.vector(n=3, dtype=data_type)
ivec2 = ti.types.vector(n=2, dtype=ti.i32)
feat_dim = 4

@ti.kernel
def fetch_kernel(
        xyzs: ti.template(), 
        plane_embedding: ti.template(),
        grid_embedding: ti.template(),
        xyzs_embedding: ti.template(), 
        B: ti.i32,
        grid_res: ti.i32,
        plane_res: ti.i32,):

    # get hash table embedding
    for i in ti.ndrange(B):
        xyz = ti.Vector([xyzs[i, 0], xyzs[i, 1], xyzs[i, 2]])

        grid_feat = tf_vec4(0.)
        xy_plane_feat = tf_vec4(0.)
        yz_plane_feat = tf_vec4(0.)
        zx_plane_feat = tf_vec4(0.)

        # plane query
        pos = xyz * (plane_res-1) + 0.5
        pos_grid_uint = ti.cast(ti.floor(pos), ti.int32)
        pos -= pos_grid_uint

        xy_pos = ti.Vector([pos[0], pos[1]])
        yz_pos = ti.Vector([pos[1], pos[2]])
        zx_pos = ti.Vector([pos[2], pos[0]])

        xy_pos_grid_uint = ti.Vector([pos_grid_uint[0], pos_grid_uint[1]])
        yz_pos_grid_uint = ti.Vector([pos_grid_uint[1], pos_grid_uint[2]])
        zx_pos_grid_uint = ti.Vector([pos_grid_uint[2], pos_grid_uint[0]])

        for idx in ti.static(range(4)):
            xy_w = 1.
            yz_w = 1.
            zx_w = 1.
            xy_pos_grid_local = ivec2(0)
            yz_pos_grid_local = ivec2(0)
            zx_pos_grid_local = ivec2(0)

            for d in ti.static(range(2)):
                if (idx & (1 << d)) == 0:
                    xy_pos_grid_local[d] = xy_pos_grid_uint[d]
                    yz_pos_grid_local[d] = yz_pos_grid_uint[d]
                    zx_pos_grid_local[d] = zx_pos_grid_uint[d]
                    xy_w *= 1 - xy_pos[d]
                    yz_w *= 1 - yz_pos[d]
                    zx_w *= 1 - zx_pos[d]
                else:
                    xy_pos_grid_local[d] = xy_pos_grid_uint[d] + 1
                    yz_pos_grid_local[d] = yz_pos_grid_uint[d] + 1
                    zx_pos_grid_local[d] = zx_pos_grid_uint[d] + 1
                    xy_w *= xy_pos[d]
                    yz_w *= yz_pos[d]
                    zx_w *= zx_pos[d]

            xy_index = 0
            yz_index = 0
            zx_index = 0
            stride = 1
            for i in ti.static(range(2)):
                xy_index += xy_pos_grid_local[i] * stride
                yz_index += yz_pos_grid_local[i] * stride
                zx_index += zx_pos_grid_local[i] * stride
                stride *= plane_res

            xy_index_table = xy_index * feat_dim
            yz_index_table = plane_res**2*feat_dim + yz_index * feat_dim
            zx_index_table = plane_res**2*feat_dim*2 + zx_index * feat_dim
            for j in ti.static(range(feat_dim)):
                xy_plane_feat[j] += xy_w * plane_embedding[xy_index_table + j]
                yz_plane_feat[j] += yz_w * plane_embedding[yz_index_table + j]
                zx_plane_feat[j] += zx_w * plane_embedding[zx_index_table + j]


        # grid query
        pos = xyz * (grid_res - 1) + 0.5
        pos_grid_uint = ti.cast(ti.floor(pos), ti.int32)
        pos -= pos_grid_uint

        for idx in ti.static(range(8)):
            w = 1.
            pos_grid_local = ivec3(0)

            for d in ti.static(range(3)):
                if (idx & (1 << d)) == 0:
                    pos_grid_local[d] = pos_grid_uint[d]
                    w *= 1 - pos[d]
                else:
                    pos_grid_local[d] = pos_grid_uint[d] + 1
                    w *= pos[d]

            index = 0
            stride = 1
            for i in ti.static(range(3)):
                index += pos_grid_local[i] * stride
                stride *= grid_res

            index_table = index * feat_dim
            for j in ti.static(range(feat_dim)):
                grid_feat[j] += w * grid_embedding[index_table + j]


        for j in ti.static(range(feat_dim)):
            xyzs_embedding[i, j] = xy_plane_feat[j]
            xyzs_embedding[i, j + feat_dim] = yz_plane_feat[j]
            xyzs_embedding[i, j + feat_dim*2] = zx_plane_feat[j]
            xyzs_embedding[i, j + feat_dim*3] = grid_feat[j]

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


class TriPlaneEncoder(torch.nn.Module):

    def __init__(self,\
                 batch_size=8192,
                 data_type=data_type,
                 half2_opt=False):
        super(TriPlaneEncoder, self).__init__()

        if batch_size < 2048:
            batch_size = 2048

        self.grid_res = 256
        self.grid_feat = feat_dim
        self.plane_res = 1024
        self.plane_feat = feat_dim

        self.plane_embedding = torch.nn.Parameter(
            torch.zeros((self.plane_res**2) * 3 * self.plane_feat, dtype=torch_type),
            requires_grad=True
        )
        self.grid_embedding = torch.nn.Parameter(
            torch.zeros(self.grid_res**3 * self.grid_feat, dtype=torch_type),
            requires_grad=True
        )
                        
        random_initialize(self.plane_embedding)
        random_initialize(self.grid_embedding)

        if half2_opt:
            assert self.total_hash_size % 2 == 0
            self.parameter_fields = half2.field(shape=(self.total_hash_size //
                                                       2, ),
                                                needs_grad=True)
            self.output_fields = half2.field(shape=(batch_size * 1024, 16),
                                             needs_grad=True)

            self.torch2ti = torch2ti_vec
            self.ti2torch = ti2torch_vec
            self.ti2torch_grad = ti2torch_grad_vec
            self.torch2ti_grad = torch2ti_grad_vec

            self._hash_encode_kernel = hash_encode_kernel_half2
        else:
            self.plane_embedding_fields = ti.field(
                data_type, 
                shape=(self.plane_res**2 * 3 * self.plane_feat, ), 
                needs_grad=True
            )
            self.grid_embedding_fields = ti.field(
                data_type,
                shape=(self.grid_res**3 * self.grid_feat, ),
                needs_grad=True
            )
            self.output_fields = ti.field(dtype=data_type,
                                          shape=(batch_size * 1024, 32),
                                          needs_grad=True)
            self.torch2ti = torch2ti
            self.ti2torch = ti2torch
            self.ti2torch_grad = ti2torch_grad
            self.torch2ti_grad = torch2ti_grad

            self._encode_kernel = fetch_kernel

        self.input_fields = ti.field(dtype=data_type,
                                     shape=(batch_size * 1024, 3),
                                     needs_grad=True)

        self.register_buffer(
            'plane_grad', 
            torch.zeros(
                self.plane_res**2 * 3 * self.plane_feat, 
                dtype=torch_type
            )
        )
        self.register_buffer(
            'grid_grad', 
            torch.zeros(
                self.grid_res**3 * self.grid_feat, 
                dtype=torch_type
            )
        )
        self.register_buffer(
            'output_embedding',
            torch.zeros(batch_size * 1024, 32, dtype=torch_type))

        class _module_function(torch.autograd.Function):

            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, input_pos, plane_params, grid_params):
                output_embedding = self.output_embedding[:input_pos.
                                                         shape[0]].contiguous(
                                                         )
                torch2ti(self.input_fields, input_pos.contiguous())
                self.torch2ti(self.plane_embedding_fields, plane_params.contiguous())
                self.torch2ti(self.grid_embedding_fields, grid_params.contiguous())

                self._encode_kernel(
                    self.input_fields,
                    self.plane_embedding_fields,
                    self.grid_embedding_fields,
                    self.output_fields,
                    input_pos.shape[0],
                    self.grid_res,
                    self.plane_res,
                )
                self.ti2torch(self.output_fields, output_embedding)

                return output_embedding

            @staticmethod
            @custom_bwd
            def backward(ctx, doutput):

                self.zero_grad()

                self.torch2ti_grad(self.output_fields, doutput.contiguous())
                self._encode_kernel.grad(
                    self.input_fields,
                    self.plane_embedding_fields,
                    self.grid_embedding_fields,
                    self.output_fields,
                    doutput.shape[0],
                    self.grid_res,
                    self.plane_res,
                )
                self.ti2torch_grad(self.plane_embedding_fields,
                                   self.plane_grad.contiguous())
                self.ti2torch_grad(self.grid_embedding_fields,
                                   self.grid_grad.contiguous())
                return None, self.plane_grad, self.grid_grad

        self._module_function = _module_function

    def zero_grad(self):
        self.plane_embedding_fields.grad.fill(0.)
        self.grid_embedding_fields.grad.fill(0.)

    def forward(self, positions):
        return self._module_function.apply(positions, self.plane_embedding, self.grid_embedding)
