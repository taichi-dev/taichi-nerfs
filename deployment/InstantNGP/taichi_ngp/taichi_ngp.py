import argparse
import os
import shutil
from typing import Tuple

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from modules.intersection import ray_aabb_intersect

import numpy as np
import taichi as ti
from matplotlib import pyplot as plt
from kernels import args, np_type, data_type,\
                    rotate_scale, reset,\
                    ray_intersect, raymarching_test_kernel,\
                    rearange_index, hash_encode,\
                    sigma_rgb_layer, composite_test,\
                    re_order, fill_ndarray,\
                    init_current_index, rotate_scale,\
                    initialize, load_deployment_model,\
                    cascades, grid_size, scale, \
                    NGP_res, NGP_N_rays, NGP_min_samples

from new_kernels import get_rays

ti.init(arch=ti.vulkan,
        enable_fallback=False,
        debug=False,
        kernel_profiler=False)

#########################
# Compile for AOT files #
#########################
def save_aot_weights(aot_folder, np_arr, name):
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

def prepare_aot_files(model):
    aot_folder = os.path.dirname(__file__) + '/compiled'
    shutil.rmtree(aot_folder, ignore_errors=True)
    os.makedirs(aot_folder)

    save_aot_weights(aot_folder, 
                 model['model.hash_encoder.params'].astype(np_type),
                 'hash_embedding')
    save_aot_weights(aot_folder,
                 model['model.xyz_encoder.params'].astype(np_type),
                 'sigma_weights')
    save_aot_weights(aot_folder, 
                 model['model.rgb_net.params'].astype(np_type),
                 'rgb_weights')
    save_aot_weights(aot_folder,
                 model['model.density_bitfield'].view("uint32"),
                 'density_bitfield')
    save_aot_weights(aot_folder,
                 model['poses'][20].astype(np_type).reshape(3, 4), 'pose')
    save_aot_weights(aot_folder,
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


##########################################
# Inference on Host Machine (DEBUG ONLY) #
##########################################
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



def run_inference(max_samples,
                  T_threshold,
                  dist_to_focus=0.8,
                  len_dis=0.0) -> Tuple[float, int, int]:
    samples = 0
    #rotate_scale(NGP_pose, 0.5, 0.5, 0.0, 2.5)
    reset(NGP_counter, NGP_alive_indices, NGP_opacity, NGP_rgb)

    get_rays(NGP_pose, NGP_directions, NGP_rays_o, NGP_rays_d)
    ray_aabb_intersect(NGP_hits_t, NGP_rays_o, NGP_rays_d, scale)

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


def inference_local(n=1):
    
    for _ in range(n):
        samples, N_alive, N_samples = run_inference(max_samples=100, T_threshold=1e-2)
    
    ti.sync()
    
    # Show inferenced image
    rgb_np = NGP_rgb.to_numpy().reshape(NGP_res[1], NGP_res[0], 3)
    plt.imshow((rgb_np * 255).astype(np.uint8))
    plt.show()
    

if __name__ == '__main__':
    model = load_deployment_model(args.model_path)
    initialize()

    if args.aot:
        #####################################################
        # Prepare AOT files for inference on mobile devices #
        #####################################################
        prepare_aot_files(model)
    else:
        ##################################
        #     THIS IS FOR DEBUG ONLY     #
        # Run inference on local machine #
        ##################################
        # Others
        NGP_hits_t = ti.Vector.ndarray(n=2, dtype=data_type, shape=(NGP_N_rays))
        
        fill_ndarray(NGP_hits_t, -1.0)

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
        NGP_hash_embedding = ti.ndarray(dtype=data_type, shape=(17956864, ))
        NGP_sigma_weights = ti.ndarray(dtype=data_type,
                                       shape=(sigma_layer1_base + 16 * 16, ))
        NGP_rgb_weights = ti.ndarray(dtype=data_type,
                                     shape=(layer1_base + 16 * 16, ))

        # buffers that used for points sampling
        NGP_max_samples_per_rays = 1
        NGP_max_samples_shape = NGP_N_rays * NGP_max_samples_per_rays

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
        
        update_model_weights(model)
        inference_local()
