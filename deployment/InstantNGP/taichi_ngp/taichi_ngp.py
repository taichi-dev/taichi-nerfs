import argparse
import os
import platform
import shutil
import time
from typing import Tuple

import numpy as np
import taichi as ti
import wget
from matplotlib import pyplot as plt
from taichi.math import uvec3


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_w', type=int, default=800)
    parser.add_argument('--res_h', type=int, default=800)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--aot', action='store_true', default=False)
    return parser.parse_args()


args = parse_arguments()

ti.init(arch=ti.vulkan,
        enable_fallback=False,
        debug=False,
        kernel_profiler=False)

block_dim = 128

sigma_sm_preload = int((16 * 16 + 16 * 16) / block_dim)
rgb_sm_preload = int((16 * 32 + 16 * 16) / block_dim)
data_type = ti.f32
np_type = np.float32
tf_vec3 = ti.types.vector(3, dtype=data_type)
tf_vec8 = ti.types.vector(8, dtype=data_type)
tf_vec16 = ti.types.vector(16, dtype=data_type)
tf_vec32 = ti.types.vector(32, dtype=data_type)
tf_vec1 = ti.types.vector(1, dtype=data_type)
tf_vec2 = ti.types.vector(2, dtype=data_type)
tf_mat1x3 = ti.types.matrix(1, 3, dtype=data_type)
tf_index_temp = ti.types.vector(8, dtype=ti.i32)

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01

SQRT3 = 1.7320508075688772
SQRT3_MAX_SAMPLES = SQRT3 / 1024
SQRT3_2 = 1.7320508075688772 * 2

res_w = args.res_w
res_h = args.res_h
scale = 0.5
cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
grid_size = 128
base_res = 32
log2_T = 21
res = [res_w, res_h]
level = 4
exp_step_factor = 0
NGP_res = res
NGP_N_rays = res[0] * res[1]
NGP_grid_size = grid_size
NGP_exp_step_factor = exp_step_factor
NGP_scale = scale

# hash table variables
NGP_min_samples = 1 if exp_step_factor == 0 else 4
NGP_per_level_scales = 1.3195079565048218  # hard coded, otherwise it will be have lower percision
NGP_base_res = base_res
NGP_max_params = 2**log2_T
NGP_level = level
    
NGP_offsets = [0 for _ in range(16)]

#<----------------- hash table util code ----------------->
@ti.func
def calc_dt(t, exp_step_factor, grid_size, scale):
    return data_type(
        ti.math.clamp(t * exp_step_factor, SQRT3_MAX_SAMPLES,
                      SQRT3_2 * scale / grid_size))


@ti.func
def __expand_bits(v):
    v = (v * ti.uint32(0x00010001)) & ti.uint32(0xFF0000FF)
    v = (v * ti.uint32(0x00000101)) & ti.uint32(0x0F00F00F)
    v = (v * ti.uint32(0x00000011)) & ti.uint32(0xC30C30C3)
    v = (v * ti.uint32(0x00000005)) & ti.uint32(0x49249249)
    return v


@ti.func
def __morton3D(xyz):
    xyz = __expand_bits(xyz)
    return xyz[0] | (xyz[1] << 1) | (xyz[2] << 2)


@ti.func
def fast_hash(pos_grid_local):
    result = ti.uint32(0)
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


#<----------------- hash table util code ----------------->


@ti.func
def random_in_unit_disk():
    theta = 2.0 * np.pi * ti.random()
    return ti.Vector([ti.sin(theta), ti.cos(theta)])


@ti.func
def random_normal():
    x = ti.random() * 2. - 1.
    y = ti.random() * 2. - 1.
    return tf_vec2(x, y)


@ti.func
def dir_encode_func(dir_):
    input_val = tf_vec32(0.0)
    d = dir_ / dir_.norm()
    x = d[0]
    y = d[1]
    z = d[2]
    xy = x * y
    xz = x * z
    yz = y * z
    x2 = x * x
    y2 = y * y
    z2 = z * z

    temp = 0.28209479177387814
    input_val[0] = data_type(temp)
    input_val[1] = data_type(-0.48860251190291987 * y)
    input_val[2] = data_type(0.48860251190291987 * z)
    input_val[3] = data_type(-0.48860251190291987 * x)
    input_val[4] = data_type(1.0925484305920792 * xy)
    input_val[5] = data_type(-1.0925484305920792 * yz)
    input_val[6] = data_type(0.94617469575755997 * z2 - 0.31539156525251999)
    input_val[7] = data_type(-1.0925484305920792 * xz)
    input_val[8] = data_type(0.54627421529603959 * x2 -
                             0.54627421529603959 * y2)
    input_val[9] = data_type(0.59004358992664352 * y * (-3.0 * x2 + y2))
    input_val[10] = data_type(2.8906114426405538 * xy * z)
    input_val[11] = data_type(0.45704579946446572 * y * (1.0 - 5.0 * z2))
    input_val[12] = data_type(0.3731763325901154 * z * (5.0 * z2 - 3.0))
    input_val[13] = data_type(0.45704579946446572 * x * (1.0 - 5.0 * z2))
    input_val[14] = data_type(1.4453057213202769 * z * (x2 - y2))
    input_val[15] = data_type(0.59004358992664352 * x * (-x2 + 3.0 * y2))

    return input_val

@ti.kernel
def init_from_arr(data: ti.types.ndarray(), f: ti.template()):
    for I in ti.grouped(f):
        f[I] = data[I]


@ti.kernel
def rotate_scale(NGP_pose: ti.types.ndarray(dtype=ti.types.matrix(
    3, 4, dtype=data_type),
                                            ndim=0), angle_x: ti.f32,
                 angle_y: ti.f32, angle_z: ti.f32, radius: ti.f32):
    # first move camera to radius
    res = ti.math.eye(4)
    res[2, 3] -= radius

    # rotate
    rot = ti.math.eye(4)
    rot[:3, :3] = NGP_pose[None][:3, :3]

    rot = ti.math.rotation3d(angle_x, angle_y, angle_z) @ rot

    res = rot @ res
    # translate
    res[:3, 3] -= NGP_center

    NGP_pose[None] = res[:3, :4]

@ti.kernel
def reset(counter: ti.types.ndarray(dtype=ti.i32, ndim=1),
          NGP_alive_indices: ti.types.ndarray(ti.i32, ndim=1),
          NGP_opacity: ti.types.ndarray(dtype=data_type, ndim=1),
          NGP_rgb: ti.types.ndarray(dtype=tf_vec3, ndim=1)):
    for I in ti.grouped(NGP_opacity):
        NGP_opacity[I] = 0.0
    for I in ti.grouped(NGP_rgb):
        NGP_rgb[I] = tf_vec3(0.0)
    counter[0] = NGP_N_rays
    for i, j in ti.ndrange(NGP_N_rays, 2):
        NGP_alive_indices[i * 2 + j] = i


@ti.func
def _ray_aabb_intersec(ray_o, ray_d):
    inv_d = 1.0 / ray_d

    t_min = (NGP_center - NGP_half_size - ray_o) * inv_d
    t_max = (NGP_center + NGP_half_size - ray_o) * inv_d

    _t1 = ti.min(t_min, t_max)
    _t2 = ti.max(t_min, t_max)
    t1 = _t1.max()
    t2 = _t2.min()

    return tf_vec2(t1, t2)


@ti.kernel
def ray_intersect(
        counter: ti.types.ndarray(ti.i32, ndim=1),
        NGP_pose: ti.types.ndarray(dtype=ti.types.matrix(3, 4,
                                                         dtype=data_type),
                                   ndim=0),
        NGP_directions: ti.types.ndarray(dtype=ti.types.matrix(
            1, 3, dtype=data_type),
                                         ndim=1),
        NGP_hits_t: ti.types.ndarray(dtype=tf_vec2, ndim=1),
        NGP_rays_o: ti.types.ndarray(dtype=tf_vec3, ndim=1),
        NGP_rays_d: ti.types.ndarray(dtype=tf_vec3, ndim=1),
):
    for i in ti.ndrange(counter[0]):
        c2w = NGP_pose[None]
        mat_result = NGP_directions[i] @ c2w[:, :3].transpose()
        ray_d = tf_vec3(mat_result[0, 0], mat_result[0, 1], mat_result[0, 2])
        ray_o = c2w[:, 3]

        t1t2 = _ray_aabb_intersec(ray_o, ray_d)

        if t1t2[1] > 0.0:
            NGP_hits_t[i][0] = data_type(ti.max(t1t2[0], NEAR_DISTANCE))
            NGP_hits_t[i][1] = t1t2[1]

        NGP_rays_o[i] = ray_o
        NGP_rays_d[i] = ray_d


@ti.kernel
def raymarching_test_kernel(
        counter: ti.types.ndarray(ti.i32, ndim=1),
        NGP_density_bitfield: ti.types.ndarray(dtype=ti.u32, ndim=1),
        NGP_hits_t: ti.types.ndarray(dtype=tf_vec2, ndim=1),
        NGP_alive_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        NGP_rays_o: ti.types.ndarray(dtype=tf_vec3, ndim=1),
        NGP_rays_d: ti.types.ndarray(dtype=tf_vec3, ndim=1),
        NGP_current_index: ti.types.ndarray(dtype=ti.i32, ndim=0),
        NGP_xyzs: ti.types.ndarray(dtype=tf_vec3, ndim=1),
        NGP_dirs: ti.types.ndarray(dtype=tf_vec3, ndim=1),
        NGP_deltas: ti.types.ndarray(dtype=data_type, ndim=1),
        NGP_ts: ti.types.ndarray(dtype=data_type, ndim=1),
        NGP_run_model_ind: ti.types.ndarray(dtype=ti.i32, ndim=1),
        NGP_N_eff_samples: ti.types.ndarray(dtype=ti.i32,
                                            ndim=1), N_samples: int):
    #NGP_run_model_ind.fill(0)
    for n in ti.ndrange(counter[0]):
        c_index = NGP_current_index[None]
        r = NGP_alive_indices[n * 2 + c_index]
        grid_size3 = NGP_grid_size**3
        grid_size_inv = 1.0 / NGP_grid_size

        ray_o = NGP_rays_o[r]
        ray_d = NGP_rays_d[r]
        t1t2 = NGP_hits_t[r]

        d_inv = 1.0 / ray_d

        t = t1t2[0]
        t2 = t1t2[1]

        s = 0

        start_idx = n * N_samples

        while (0 <= t) & (t < t2) & (s < N_samples):
            # xyz = ray_o + t*ray_d
            xyz = ray_o + t * ray_d
            dt = calc_dt(t, NGP_exp_step_factor, NGP_grid_size, NGP_scale)
            # mip = ti.max(mip_from_pos(xyz, cascades),
            #             mip_from_dt(dt, grid_size, cascades))

            mip_bound = 0.5
            mip_bound_inv = 1 / mip_bound

            nxyz = ti.math.clamp(
                0.5 * (xyz * mip_bound_inv + 1) * NGP_grid_size, 0.0,
                NGP_grid_size - 1.0)
            # nxyz = ti.ceil(nxyz)

            idx = __morton3D(ti.cast(nxyz, ti.u32))
            # occ = density_grid_taichi[idx] > 5.912066756501768
            occ = ti.uint32(NGP_density_bitfield[ti.u32(idx // 32)]
                            & (ti.u32(1) << ti.u32(idx % 32)))

            if occ:
                sn = start_idx + s
                for p in ti.static(range(3)):
                    NGP_xyzs[sn][p] = xyz[p]
                    NGP_dirs[sn][p] = ray_d[p]
                NGP_run_model_ind[sn] = 1
                NGP_ts[sn] = t
                NGP_deltas[sn] = dt
                t += dt
                NGP_hits_t[r][0] = t
                s += 1

            else:
                txyz = (((nxyz + 0.5 + 0.5 * ti.math.sign(ray_d)) *
                         grid_size_inv * 2 - 1) * mip_bound - xyz) * d_inv

                t_target = t + ti.max(0, txyz.min())
                t += calc_dt(t, NGP_exp_step_factor, NGP_grid_size, NGP_scale)
                while t < t_target:
                    t += calc_dt(t, NGP_exp_step_factor, NGP_grid_size,
                                 NGP_scale)

        NGP_N_eff_samples[n] = s
        if s == 0:
            NGP_alive_indices[n * 2 + c_index] = -1


@ti.kernel
def rearange_index(
        NGP_model_launch: ti.types.ndarray(ti.i32, ndim=0),
        NGP_padd_block_network: ti.types.ndarray(ti.i32, ndim=0),
        NGP_temp_hit: ti.types.ndarray(ti.i32, ndim=1),
        NGP_run_model_ind: ti.types.ndarray(ti.i32, ndim=1), B: ti.i32):
    NGP_model_launch[None] = 0

    for i in ti.ndrange(B):
        if NGP_run_model_ind[i]:
            index = ti.atomic_add(NGP_model_launch[None], 1)
            NGP_temp_hit[index] = i

    NGP_model_launch[None] += 1
    NGP_padd_block_network[None] = (
        (NGP_model_launch[None] + block_dim - 1) // block_dim) * block_dim


@ti.kernel
def hash_encode(
        NGP_hash_embedding: ti.types.ndarray(dtype=data_type, ndim=1),
        NGP_model_launch: ti.types.ndarray(ti.i32, ndim=0),
        NGP_xyzs: ti.types.ndarray(dtype=tf_vec3, ndim=1),
        NGP_dirs: ti.types.ndarray(dtype=tf_vec3, ndim=1),
        NGP_deltas: ti.types.ndarray(dtype=data_type, ndim=1),
        NGP_xyzs_embedding: ti.types.ndarray(dtype=data_type, ndim=2),
        NGP_temp_hit: ti.types.ndarray(ti.i32, ndim=1),
):
    for sn in ti.ndrange(NGP_model_launch[None]):
        for level in ti.static(range(NGP_level)):
            xyz = NGP_xyzs[NGP_temp_hit[sn]] + 0.5
            offset = NGP_offsets[level] * 4

            init_val0 = tf_vec1(0.0)
            init_val1 = tf_vec1(1.0)
            local_feature_0 = init_val0[0]
            local_feature_1 = init_val0[0]
            local_feature_2 = init_val0[0]
            local_feature_3 = init_val0[0]

            scale = NGP_base_res * ti.exp(
                level * ti.log(NGP_per_level_scales)) - 1.0
            resolution = ti.cast(ti.ceil(scale), ti.uint32) + 1

            pos = xyz * scale + 0.5
            pos_grid_uint = ti.cast(ti.floor(pos), ti.uint32)
            pos -= pos_grid_uint

            for idx in ti.static(range(8)):
                w = init_val1[0]
                pos_grid_local = uvec3(0)

                for d in ti.static(range(3)):
                    if (idx & (1 << d)) == 0:
                        pos_grid_local[d] = pos_grid_uint[d]
                        w *= data_type(1 - pos[d])
                    else:
                        pos_grid_local[d] = pos_grid_uint[d] + 1
                        w *= data_type(pos[d])

                index = 0
                stride = 1
                for c_ in ti.static(range(3)):
                    index += pos_grid_local[c_] * stride
                    stride *= resolution

                local_feature_0 += data_type(
                    w * NGP_hash_embedding[offset + index * 4])
                local_feature_1 += data_type(
                    w * NGP_hash_embedding[offset + index * 4 + 1])
                local_feature_2 += data_type(
                    w * NGP_hash_embedding[offset + index * 4 + 2])
                local_feature_3 += data_type(
                    w * NGP_hash_embedding[offset + index * 4 + 3])

            NGP_xyzs_embedding[sn, level * 4] = local_feature_0
            NGP_xyzs_embedding[sn, level * 4 + 1] = local_feature_1
            NGP_xyzs_embedding[sn, level * 4 + 2] = local_feature_2
            NGP_xyzs_embedding[sn, level * 4 + 3] = local_feature_3


@ti.kernel
def sigma_rgb_layer(
        NGP_sigma_weights: ti.types.ndarray(dtype=data_type, ndim=1),
        NGP_rgb_weights: ti.types.ndarray(dtype=data_type, ndim=1),
        NGP_model_launch: ti.types.ndarray(dtype=ti.i32, ndim=0),
        NGP_padd_block_network: ti.types.ndarray(dtype=ti.i32, ndim=0),
        NGP_xyzs_embedding: ti.types.ndarray(dtype=data_type, ndim=2),
        NGP_dirs: ti.types.ndarray(dtype=tf_vec3, ndim=1),
        # NGP_final_embedding: ti.types.ndarray(dtype=data_type, ndim=2),
        NGP_out_1: ti.types.ndarray(dtype=data_type, ndim=1),
        NGP_out_3: ti.types.ndarray(data_type, ndim=2),
        NGP_temp_hit: ti.types.ndarray(ti.i32, ndim=1),
):
    ti.loop_config(block_dim=block_dim)  # DO NOT REMOVE
    for sn in ti.ndrange(NGP_padd_block_network[None]):
        ray_id = NGP_temp_hit[sn]
        tid = sn % block_dim
        did_launch_num = NGP_model_launch[None]
        init_val = tf_vec1(0.0)
        sigma_weight = ti.simt.block.SharedArray((16 * 16 + 16 * 16, ),
                                                 data_type)
        rgb_weight = ti.simt.block.SharedArray((16 * 32 + 16 * 16, ),
                                               data_type)

        for i in ti.static(range(sigma_sm_preload)):
            k = tid * sigma_sm_preload + i
            sigma_weight[k] = NGP_sigma_weights[k]
        for i in ti.static(range(rgb_sm_preload)):
            k = tid * rgb_sm_preload + i
            rgb_weight[k] = NGP_rgb_weights[k]
        ti.simt.block.sync()

        if sn < did_launch_num:

            s0 = init_val[0]
            s1 = init_val[0]
            s2 = init_val[0]

            dir_ = NGP_dirs[ray_id]
            rgb_input_val = dir_encode_func(dir_)
            sigma_output_val = tf_vec16(0.)

            for i in range(16):
                temp = init_val[0]
                for j in ti.static(range(16)):
                    temp += NGP_xyzs_embedding[sn,
                                               j] * sigma_weight[i * 16 + j]

                for j in ti.static(range(16)):
                    sigma_output_val[j] += data_type(ti.max(
                        0.0, temp)) * sigma_weight[16 * 16 + j * 16 + i]

            for i in ti.static(range(16)):
                rgb_input_val[16 + i] = sigma_output_val[i]

            for i in range(16):
                temp = init_val[0]
                for j in ti.static(range(32)):
                    temp += rgb_input_val[j] * rgb_weight[i * 32 + j]

                s0 += data_type((ti.max(0.0, temp))) * rgb_weight[16 * 32 + i]
                s1 += data_type(
                    (ti.max(0.0, temp))) * rgb_weight[16 * 32 + 16 + i]
                s2 += data_type(
                    (ti.max(0.0, temp))) * rgb_weight[16 * 32 + 32 + i]

            NGP_out_1[NGP_temp_hit[sn]] = data_type(ti.exp(
                sigma_output_val[0]))
            NGP_out_3[NGP_temp_hit[sn], 0] = data_type(1 / (1 + ti.exp(-s0)))
            NGP_out_3[NGP_temp_hit[sn], 1] = data_type(1 / (1 + ti.exp(-s1)))
            NGP_out_3[NGP_temp_hit[sn], 2] = data_type(1 / (1 + ti.exp(-s2)))


@ti.kernel
def composite_test(
        counter: ti.types.ndarray(dtype=ti.i32, ndim=1),
        NGP_alive_indices: ti.types.ndarray(dtype=ti.i32,
                                            ndim=1), NGP_rgb: ti.types.ndarray(
                                                dtype=tf_vec3, ndim=1),
        NGP_opacity: ti.types.ndarray(dtype=data_type, ndim=1),
        NGP_current_index: ti.types.ndarray(ti.i32, ndim=0),
        NGP_deltas: ti.types.ndarray(data_type,
                                     ndim=1), NGP_ts: ti.types.ndarray(
                                         data_type, ndim=1),
        NGP_out_3: ti.types.ndarray(data_type,
                                    ndim=2), NGP_out_1: ti.types.ndarray(
                                        dtype=data_type, ndim=1),
        NGP_N_eff_samples: ti.types.ndarray(dtype=ti.i32, ndim=1),
        max_samples: ti.i32, T_threshold: data_type):
    for n in ti.ndrange(counter[0]):
        N_samples = NGP_N_eff_samples[n]
        if N_samples != 0:
            c_index = NGP_current_index[None]
            r = NGP_alive_indices[n * 2 + c_index]

            T = data_type(1.0 - NGP_opacity[r])

            start_idx = n * max_samples

            rgb_temp = tf_vec3(0.0)
            depth_temp = tf_vec1(0.0)
            opacity_temp = tf_vec1(0.0)
            out_3_temp = tf_vec3(0.0)

            for s in range(N_samples):
                sn = start_idx + s
                a = data_type(1.0 - ti.exp(-NGP_out_1[sn] * NGP_deltas[sn]))
                w = a * T

                for i in ti.static(range(3)):
                    out_3_temp[i] = NGP_out_3[sn, i]

                rgb_temp += w * out_3_temp
                depth_temp[0] += w * NGP_ts[sn]
                opacity_temp[0] += w

                T *= data_type(1.0 - a)

                if T <= T_threshold:
                    NGP_alive_indices[n * 2 + c_index] = -1
                    break

            NGP_rgb[r] = NGP_rgb[r] + rgb_temp
            NGP_opacity[r] = NGP_opacity[r] + opacity_temp[0]


@ti.kernel
def init_current_index(NGP_current_index: ti.types.ndarray(ti.i32, ndim=0)):
    NGP_current_index[None] = 0


@ti.kernel
def re_order(counter: ti.types.ndarray(ti.i32, ndim=1),
             NGP_alive_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
             NGP_current_index: ti.types.ndarray(ti.i32, ndim=0), B: ti.i32):
    counter[0] = 0
    c_index = NGP_current_index[None]
    n_index = (c_index + 1) % 2
    NGP_current_index[None] = n_index

    for i in ti.ndrange(B):
        alive_temp = NGP_alive_indices[i * 2 + c_index]
        if alive_temp >= 0:
            index = ti.atomic_add(counter[0], 1)
            NGP_alive_indices[index * 2 + n_index] = alive_temp

@ti.kernel
def fill_ndarray(
        arr: ti.types.ndarray(dtype=ti.types.vector(2, dtype=data_type),
                              ndim=1), val: data_type):
    for I in ti.grouped(arr):
        arr[I] = val

def save_weights(aot_folder, np_arr, name):
    # Binary Header: int32(dtype) int32(num_elements)
    # Binary Contents: flat binary buffer

    # dtype: 0(float32), 1(float16), 2(int32), 3(int16), 4(uint32), 5(uint16)
    if np_arr.dtype == np.float32:
        dtype = 0
    elif np_arr.dtype == np.float16:
        dtype = 1
    elif np_arr.dtype == np.int32:
        dtype = 2
    elif np_arr.dtype == np.int16:
        dtype = 3
    elif np_arr.dtype == np.uint32:
        dtype = 4
    elif np_arr.dtype == np.uint16:
        dtype = 5
    else:
        print("Unrecognized dtype: ", np_arr.dtype)
        assert False
    num_elements = np_arr.size

    byte_arr = np_arr.flatten().tobytes()
    header = np.array([dtype, num_elements]).astype('int32').tobytes()

    filename = aot_folder + '/' + name + '.bin'
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, "wb+") as f:
        f.write(header)
        f.write(byte_arr)

def initialize():
    offset = 0
    for i in range(NGP_level):
        resolution = int(
            np.ceil(NGP_base_res * np.exp(i * np.log(NGP_per_level_scales)) -
                    1.0)) + 1
        print(f"level: {i}, res: {resolution}")
        params_in_level = resolution**3
        params_in_level = int(resolution**
                              3) if params_in_level % 8 == 0 else int(
                                  (params_in_level + 8 - 1) / 8) * 8
        params_in_level = min(NGP_max_params, params_in_level)
        NGP_offsets[i] = offset
        offset += params_in_level
    
def NGP_get_direction(res_w, res_h, camera_angle_x):
    w, h = int(res_w), int(res_h)
    fx = 0.5 * w / np.tan(0.5 * camera_angle_x)
    fy = 0.5 * h / np.tan(0.5 * camera_angle_x)
    cx, cy = 0.5 * w, 0.5 * h

    x, y = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5,
                       np.arange(h, dtype=np.float32) + 0.5,
                       indexing='xy')

    directions = np.stack([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], -1)

    return directions.reshape(-1, 3)

def load_deployment_model(model_path):
    if model_path is None:
        model_dir = os.path.dirname(__file__) + '/npy_models/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        npy_file = os.path.join(model_dir, 'smh_lego.npy')
        if not os.path.exists(npy_file):
            PRETRAINED_MODEL_URL = 'https://github.com/Linyou/taichi-ngp-renderer/releases/download/v0.1-models/{}.npy'
            url = PRETRAINED_MODEL_URL.format('smh_lego')
            wget.download(url, out=npy_file)
        model_path = npy_file
    
    print(f'Loading model from {model_path}')
    model = np.load(model_path, allow_pickle=True).item()
    
    global NGP_per_level_scales
    NGP_per_level_scales = model['model.per_level_scale']

    camera_angle_x = 0.5
    directions = NGP_get_direction(NGP_res[0], NGP_res[1],
                                   camera_angle_x)[:,
                                                   None, :].astype(np_type)
    model['model.directions'] = directions
    
    return model

def update_model_weights(model):
    NGP_hash_embedding.from_numpy(
        model['model.hash_encoder.params'].astype(np_type))
    NGP_sigma_weights.from_numpy(
        model['model.xyz_encoder.params'].astype(np_type))
    NGP_rgb_weights.from_numpy(model['model.rgb_net.params'].astype(np_type))
    NGP_density_bitfield.from_numpy(
        model['model.density_bitfield'].view("uint32"))

    pose = model['poses'][20].astype(np_type).reshape(3, 4)
    NGP_pose.from_numpy(pose)
    
    NGP_directions.from_numpy(model['model.directions'])
    

def render(max_samples,
           T_threshold,
           dist_to_focus=0.8,
           len_dis=0.0) -> Tuple[float, int, int]:
    samples = 0
    rotate_scale(NGP_pose, 0.5, 0.5, 0.0, 2.5)
    reset(NGP_counter, NGP_alive_indices, NGP_opacity, NGP_rgb)

    ray_intersect(NGP_counter, NGP_pose, NGP_directions, NGP_hits_t,
                  NGP_rays_o, NGP_rays_d)

    while samples < max_samples:
        N_alive = NGP_counter[0]
        if N_alive == 0:
            break

        # how many more samples the number of samples add for each ray
        N_samples = max(min(NGP_N_rays // N_alive, 64), NGP_min_samples)
        samples += N_samples
        launch_model_total = N_alive * N_samples

        raymarching_test_kernel(NGP_counter, NGP_density_bitfield, NGP_hits_t,
                                NGP_alive_indices, NGP_rays_o, NGP_rays_d,
                                NGP_current_index, NGP_xyzs, NGP_dirs,
                                NGP_deltas, NGP_ts, NGP_run_model_ind,
                                NGP_N_eff_samples, N_samples)
        rearange_index(NGP_model_launch, NGP_padd_block_network, NGP_temp_hit,
                       NGP_run_model_ind, launch_model_total)
        hash_encode(NGP_hash_embedding, NGP_model_launch, NGP_xyzs, NGP_dirs,
                    NGP_deltas, NGP_xyzs_embedding, NGP_temp_hit)
        sigma_rgb_layer(NGP_sigma_weights, NGP_rgb_weights, NGP_model_launch,
                        NGP_padd_block_network, NGP_xyzs_embedding, NGP_dirs,
                        NGP_out_1, NGP_out_3, NGP_temp_hit)

        composite_test(NGP_counter, NGP_alive_indices, NGP_rgb, NGP_opacity,
                       NGP_current_index, NGP_deltas, NGP_ts, NGP_out_3,
                       NGP_out_1, NGP_N_eff_samples, N_samples, T_threshold)
        re_order(NGP_counter, NGP_alive_indices, NGP_current_index, N_alive)

    return samples, N_alive, N_samples


def inference(n=1):
    samples, N_alive, N_samples = render(max_samples=100, T_threshold=1e-2)
    for _ in range(n):
        samples, N_alive, N_samples = render(max_samples=100, T_threshold=1e-2)
    ti.sync()
    print(f"samples: {samples}, N_alive: {N_alive}, N_samples: {N_samples}")
    rgb_np = NGP_rgb.to_numpy().reshape(NGP_res[1], NGP_res[0], 3)
    plt.imshow((rgb_np * 255).astype(np.uint8))
    plt.show()
    

def main(args):
    model = load_deployment_model(args.model_path)
    initialize()

    if args.aot:
        aot_folder = os.path.dirname(__file__) + '/compiled'
        shutil.rmtree(aot_folder, ignore_errors=True)
        os.makedirs(aot_folder)

        save_weights(aot_folder, 
                     model['model.hash_encoder.params'].astype(np_type),
                     'hash_embedding')
        save_weights(aot_folder,
                     model['model.xyz_encoder.params'].astype(np_type),
                     'sigma_weights')
        save_weights(aot_folder, 
                     model['model.rgb_net.params'].astype(np_type),
                     'rgb_weights')
        save_weights(aot_folder,
                     model['model.density_bitfield'].view("uint32"),
                     'density_bitfield')
        save_weights(aot_folder,
                     model['poses'][20].astype(np_type).reshape(3, 4), 'pose')
        save_weights(aot_folder,
                     model['model.directions'], 'directions')
    
        m = ti.aot.Module(
            caps=['spirv_has_int8', 'spirv_has_int16', 'spirv_has_float16'])
        m.add_kernel(reset)
        m.add_kernel(ray_intersect)
        m.add_kernel(raymarching_test_kernel)
        m.add_kernel(rearange_index)
        m.add_kernel(hash_encode)
        m.add_kernel(sigma_rgb_layer)
        m.add_kernel(composite_test)
        m.add_kernel(re_order)
        m.add_kernel(fill_ndarray)
        m.add_kernel(init_current_index)
        m.add_kernel(rotate_scale)

        m.save(aot_folder)
        print(f'Saved to {aot_folder}')
    else:
        update_model_weights(model)
        inference()

if __name__ == '__main__':
    # rays intersection parameters
    # t1, t2 need to be initialized to -1.0
    NGP_hits_t = ti.Vector.ndarray(n=2, dtype=data_type, shape=(NGP_N_rays))
    
    fill_ndarray(NGP_hits_t, -1.0)
    NGP_center = tf_vec3(0.0, 0.0, 0.0)
    NGP_xyz_min = -tf_vec3(scale, scale, scale)
    NGP_xyz_max = tf_vec3(scale, scale, scale)
    NGP_half_size = (NGP_xyz_max - NGP_xyz_min) / 2

    NGP_rays_o = ti.Vector.ndarray(n=3, dtype=data_type, shape=(NGP_N_rays))
    NGP_rays_d = ti.Vector.ndarray(n=3, dtype=data_type, shape=(NGP_N_rays))
    # use the pre-compute direction and scene pose
    NGP_directions = ti.Matrix.ndarray(n=1,
                                       m=3,
                                       dtype=data_type,
                                       shape=(NGP_N_rays, ))
    NGP_pose = ti.Matrix.ndarray(n=3, m=4, dtype=data_type, shape=())

    # density_bitfield is used for point sampling
    NGP_density_bitfield = ti.ndarray(ti.uint32,
                                      shape=(cascades * grid_size**3 // 32))

    # count the number of rays that still alive
    NGP_counter = ti.ndarray(ti.i32, shape=(1, ))
    NGP_counter[0] = NGP_N_rays
    # current alive buffer index
    NGP_current_index = ti.ndarray(ti.i32, shape=())
    # NGP_current_index[None] = 0
    init_current_index(NGP_current_index)

    # how many samples that need to run the model
    NGP_model_launch = ti.ndarray(ti.i32, shape=())

    # buffer for the alive rays
    NGP_alive_indices = ti.ndarray(ti.i32, shape=(2 * NGP_N_rays, ))

    # padd the thread to the factor of block size (thread per block)
    NGP_padd_block_network = ti.ndarray(ti.i32, shape=())

    # model parameters
    sigma_layer1_base = 16 * 16
    layer1_base = 32 * 16
    NGP_hash_embedding = ti.ndarray(dtype=data_type, shape=(11176096, ))
    NGP_sigma_weights = ti.ndarray(dtype=data_type,
                                   shape=(sigma_layer1_base + 16 * 16, ))
    NGP_rgb_weights = ti.ndarray(dtype=data_type,
                                 shape=(layer1_base + 16 * 16, ))

    # buffers that used for points sampling
    NGP_max_samples_per_rays = 1
    NGP_max_samples_shape = NGP_N_rays * NGP_max_samples_per_rays
    print(f"Max samples shape {NGP_max_samples_shape}")

    NGP_xyzs = ti.Vector.ndarray(3,
                                 dtype=data_type,
                                 shape=(NGP_max_samples_shape, ))
    NGP_dirs = ti.Vector.ndarray(3,
                                 dtype=data_type,
                                 shape=(NGP_max_samples_shape, ))
    NGP_deltas = ti.ndarray(data_type, shape=(NGP_max_samples_shape, ))
    NGP_ts = ti.ndarray(data_type, shape=(NGP_max_samples_shape, ))

    # buffers that store the info of sampled points
    NGP_run_model_ind = ti.ndarray(ti.int32, shape=(NGP_max_samples_shape, ))
    NGP_N_eff_samples = ti.ndarray(ti.int32, shape=(NGP_N_rays, ))

    # intermediate buffers for network
    NGP_xyzs_embedding = ti.ndarray(data_type,
                                    shape=(NGP_max_samples_shape, 32))
    NGP_final_embedding = ti.ndarray(data_type,
                                     shape=(NGP_max_samples_shape, 16))
    NGP_out_3 = ti.ndarray(data_type, shape=(NGP_max_samples_shape, 3))
    NGP_out_1 = ti.ndarray(data_type, shape=(NGP_max_samples_shape, ))
    NGP_temp_hit = ti.ndarray(ti.i32, shape=(NGP_max_samples_shape, ))

    # results buffers
    NGP_opacity = ti.ndarray(data_type, shape=(NGP_N_rays, ))
    NGP_rgb = ti.Vector.ndarray(3, dtype=data_type, shape=(NGP_N_rays, ))

    main(args)
