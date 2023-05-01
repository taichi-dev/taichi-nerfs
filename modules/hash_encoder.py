import torch
import taichi as ti
from taichi.math import uvec3

from .utils import (
    data_type, 
    torch_type, 
    random_initialize, 
    align_to,
    res_in_level_np,
)

half2 = ti.types.vector(n=2, dtype=ti.f16)


def build_hash_encoder_kernel(
    base_res: float = 16,
    hash_level: int = 16,
    feature_per_level: int = 2,
    begin_fast_hash_level: int = 16,
):
    '''
    Build hash encoder kernel
    Construct taichi kernel with some fixed parameters
    '''

    # Type
    feat_vec = ti.types.vector(
        n=feature_per_level, 
        dtype=data_type,
    )

    # Functions
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
        if indicator:
            hash_result = under_hash(pos_grid_local, resolution)
        else:
            hash_result = fast_hash(pos_grid_local)

        return hash_result % map_size

    @ti.func
    def revert_scale(base_res, level, scale):
        log_scale = ti.log(scale)
        exp_scale = ti.exp(level * log_scale)
        return base_res * exp_scale - 1.0

    @ti.kernel
    def hash_encoder_kernel(
            xyzs: ti.types.ndarray(), 
            table: ti.types.ndarray(),
            xyzs_embedding: ti.types.ndarray(), 
            hash_map_sizes_field: ti.types.ndarray(), 
            offsets: ti.types.ndarray(), 
            B: ti.i32,
            per_level_scale: ti.f32
        ):
        # get hash table embedding
        ti.loop_config(block_dim=hash_level)
        for i, level in ti.ndrange(B, hash_level):
            xyz = ti.Vector([xyzs[i, 0], xyzs[i, 1], xyzs[i, 2]])

            scale = revert_scale(base_res, level, per_level_scale)
            resolution = ti.cast(ti.ceil(scale), ti.uint32) + 1

            offset = offsets[level] * feature_per_level

            pos = xyz * scale + 0.5
            pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)
            pos -= pos_grid_uint

            map_size = hash_map_sizes_field[level]

            local_features = feat_vec(0.)

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

                index = grid_pos2hash_index(
                            level >= begin_fast_hash_level, 
                            pos_grid_local, 
                            resolution,
                            map_size,
                        )
                index_table = offset + index * feature_per_level
                index_table_int = ti.cast(index_table, ti.int32)

                for l_f in ti.static(range(feature_per_level)):
                    local_features[l_f] += w * table[index_table_int+l_f]

            out_index_base = level * feature_per_level 
            for l_f in ti.static(range(feature_per_level)):
                xyzs_embedding[i, out_index_base + l_f] = local_features[l_f]

    return hash_encoder_kernel

class HashEncoder(torch.nn.Module):

    def __init__(
        self,
        b=1.3195079565048218,
        max_params: float=2**19,
        hash_level: int=16,
        base_res: float=16,
        feature_per_level: int=2,  
    ):
        super(HashEncoder, self).__init__()

        self.per_level_scale = b
        self.base_res = base_res
        self.hash_level = hash_level
        self.max_params = max_params
        self.feature_per_level = feature_per_level
        self.out_dim = feature_per_level * hash_level

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

        offset = 0
        begin_fast_hash_level = hash_level
        for i in range(hash_level):
            resolution = res_in_level_np(
                i, base_res, self.per_level_scale
            )
            full_size = resolution**3
            # Ensure that the parameter size is a multiple of 8.
            full_size_aligned = align_to(full_size, 8)

            # Restricted the parameter size using max_params.
            params_size_i = min(max_params, full_size_aligned)

            self.offsets[i] = offset
            self.hash_map_sizes[i] = params_size_i

            # Record the first level that begins to use fast_hash
            if full_size > params_size_i:
                if begin_fast_hash_level == hash_level:
                    begin_fast_hash_level = i
            
            offset += params_size_i

        self.begin_fast_hash_level = begin_fast_hash_level
        print("total_hash_size: ", offset)

        self.total_hash_param = offset * feature_per_level
        print("total_hash_param: ", self.total_hash_param)

        self.hash_table = torch.nn.Parameter(
            torch.zeros(
                self.total_hash_param,
                dtype=torch_type
            ),
            requires_grad=True
        )
        random_initialize(self.hash_table)

        self._hash_encoder_kernel = build_hash_encoder_kernel(
            base_res=self.base_res,
            hash_level=self.hash_level,
            feature_per_level=self.feature_per_level,
            begin_fast_hash_level=self.begin_fast_hash_level,
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
                ctx.save_for_backward(
                    input_pos, 
                    output_embedding, 
                    params
                )

                self._hash_encoder_kernel(
                    input_pos.contiguous(),
                    params.contiguous(),
                    output_embedding.contiguous(),
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

                self._hash_encoder_kernel.grad(
                    input_pos.contiguous(),
                    params.contiguous(),
                    output_embedding.contiguous(),
                    self.hash_map_sizes.contiguous(),
                    self.offsets.contiguous(),
                    input_pos.shape[0],
                    self.per_level_scale,
                )
                return None, params.grad

        self._module_function = _module_function

    def forward(self, positions):
        return self._module_function.apply(positions, self.hash_table)
