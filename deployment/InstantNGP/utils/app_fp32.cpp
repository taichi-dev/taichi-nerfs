//
//  app_fp32.m
//  TaichiNerfTestbench
//
//  Created by Zhanlue Yang on 2023/2/17.
//
#include "app_fp32.hpp"

#include <cmath>
#include <ctime>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>
#include <fstream>
#include "utils.hpp"

namespace {
void check_taichi_error(const std::string &msg) {
  ti::Error error = ti::get_last_error();
  if (error.error < TI_ERROR_SUCCESS) {
    std::string error_msg = msg + "\n" + error.message;
    throw std::runtime_error(error_msg);
  }
}
}  // namespace

template <typename T>
void copy_from_vector(ti::Runtime& runtime_, const std::vector<T> &from, ti::NdArray<T> &to) {
    auto tmp = runtime_.allocate_ndarray<T>(
        {static_cast<unsigned int>(from.size())}, {}, /*host_accessible=*/true);
    tmp.write(from);
    tmp.slice().copy_to(to.slice());
    tmp.destroy();
}

App_nerf_f32::App_nerf_f32(TiArch arch) {
    runtime_ = ti::Runtime(arch);
}

void App_nerf_f32::initialize(int img_width, int img_height,
                              const std::string& aot_file_path,
                              const std::string& hash_embedding_path,
                              const std::string& sigma_weights_path,
                              const std::string& rgb_weights_path,
                              const std::string& density_bitfield_path,
                              const std::string& pose_path,
                              const std::string& directions_path) {
    // ----------------- //
    // Initialize Params 
    // ----------------- //
    kWidth = img_width;
    kHeight = img_height;
    kNumRays = kWidth * kHeight;
    kNumMaxSamples = kNumRays * kMaxSamplePerRay;

    // ------------------------------ //
    // Load AOT Module and AOT Kernels 
    // ------------------------------ //
    module_ = runtime_.load_aot_module(aot_file_path);
    check_taichi_error("load_aot_module failed");

    k_fill_ndarray_ = module_.get_kernel("fill_ndarray");
    k_init_current_index_ = module_.get_kernel("init_current_index");
    k_rotate_scale_ = module_.get_kernel("rotate_scale");

    // ---------------------------------- //
    // Load Pre-trained Weights to Ndarray 
    // ---------------------------------- //
    hash_embedding_ = runtime_.allocate_ndarray<float>(
        {11176096}, {}, /*host_accessible=*/false);
    std::vector<float> hash_embedding = read_float32_array(hash_embedding_path);
    copy_from_vector<float>(runtime_, hash_embedding, hash_embedding_);

    sigma_weights_ = runtime_.allocate_ndarray<float>(
        { kSigmaLayerBase + 16 * 16}, {}, /*host_accessible=*/false);
    auto sigma_weights = read_float32_array(sigma_weights_path);
    copy_from_vector<float>(runtime_, sigma_weights, sigma_weights_);

    rgb_weights_ = runtime_.allocate_ndarray<float>({kLayer2Base}, {},
                                                    /*host_accessible=*/false);
    auto rgb_weights = read_float32_array(rgb_weights_path);
    copy_from_vector<float>(runtime_, rgb_weights, rgb_weights_);

    density_bitfield_ = runtime_.allocate_ndarray<unsigned int>(
        {kCascades * kGridSize * kGridSize * kGridSize / 32}, {},
        /*host_accessible=*/false);
    auto density_bitfield = read_uint32_array(density_bitfield_path);
    copy_from_vector<unsigned int>(runtime_, density_bitfield, density_bitfield_);

    pose_ =
        runtime_.allocate_ndarray<float>({}, {3, 4}, /*host_accessible=*/false);
    auto pose = read_float32_array(pose_path);
    copy_from_vector<float>(runtime_, pose, pose_);

    directions_ = runtime_.allocate_ndarray<float>({kNumRays}, {1, 3},
                                                   /*host_accessible=*/false);
    auto directions = read_float32_array(directions_path);
    copy_from_vector<float>(runtime_, directions, directions_);

    // -------------------------------- //
    // Allocate and Initialize Ndarrays
    // -------------------------------- //
    run_model_ind_ = runtime_.allocate_ndarray<int>({kNumMaxSamples}, {},
                                                    /*host_accessible=*/false);
    N_eff_samples_ = runtime_.allocate_ndarray<int>({kNumRays}, {},
                                                    /*host_accessible=*/false);

    counter_ =
        runtime_.allocate_ndarray<int>({1}, {}, /*host_accessible=*/false);
    counter_stage_ =
        runtime_.allocate_ndarray<int>({1}, {}, /*host_accessible=*/true);

    hits_t_ = runtime_.allocate_ndarray<float>({kNumRays}, {2},
                                               /*host_accessible=*/false);
    k_fill_ndarray_.push_arg(hits_t_);
    k_fill_ndarray_.push_arg(float(-1.0));
    k_fill_ndarray_.launch();

    alive_indices_ = runtime_.allocate_ndarray<int>({2 * kNumRays}, {},
                                                    /*host_accessible=*/false);

    current_index_ =
        runtime_.allocate_ndarray<int>({1}, {}, /*host_accessible=*/false);
    
    k_init_current_index_[0] = current_index_;
    k_init_current_index_.launch();
    
    model_launch_ =
        runtime_.allocate_ndarray<int>({1}, {}, /*host_accessible=*/false);
    pad_block_network_ =
        runtime_.allocate_ndarray<int>({1}, {}, /*host_accessible=*/false);

    opacity_ = runtime_.allocate_ndarray<float>({kNumRays}, {},
                                                /*host_accessible=*/false);
    rays_o_ = runtime_.allocate_ndarray<float>({kNumRays}, {3},
                                               /*host_accessible=*/false);
    rays_d_ = runtime_.allocate_ndarray<float>({kNumRays}, {3},
                                               /*host_accessible=*/false);
    rgb_ = runtime_.allocate_ndarray<float>({kNumRays}, {3},
                                            /*host_accessible=*/false);
    rgb_stage_ = runtime_.allocate_ndarray<float>({kNumRays}, {3},
                                                  /*host_accessible=*/true);

    xyzs_ = runtime_.allocate_ndarray<float>({kNumMaxSamples}, {3},
                                             /*host_accessible=*/false);
    dirs_ = runtime_.allocate_ndarray<float>({kNumMaxSamples}, {3},
                                             /*host_accessible=*/false);
    deltas_ = runtime_.allocate_ndarray<float>({kNumMaxSamples}, {},
                                               /*host_accessible=*/false);
    ts_ = runtime_.allocate_ndarray<float>({kNumMaxSamples}, {},
                                           /*host_accessible=*/false);

    xyzs_embedding_ = runtime_.allocate_ndarray<float>(
        {kNumMaxSamples, 32}, {}, /*host_accessible=*/false);
    final_embedding_ = runtime_.allocate_ndarray<float>(
        {kNumMaxSamples, 16}, {}, /*host_accessible=*/false);
    out_3_ = runtime_.allocate_ndarray<float>({kNumMaxSamples, 3}, {},
                                              /*host_accessible=*/false);
    out_1_ = runtime_.allocate_ndarray<float>({kNumMaxSamples}, {},
                                              /*host_accessible=*/false);
    temp_hit_ = runtime_.allocate_ndarray<int>({kNumMaxSamples}, {},
                                               /*host_accessible=*/false);

    runtime_.wait();
    check_taichi_error("Memory allocation failed");

    // -------------------- //
    // Prepare AOT Kernels
    // -------------------- //
    k_reset_ = module_.get_kernel("reset");
    k_reset_[0] = counter_;
    k_reset_[1] = alive_indices_;
    k_reset_[2] = opacity_;
    k_reset_[3] = rgb_;

    k_ray_intersect_ = module_.get_kernel("ray_intersect");
    k_ray_intersect_[0] = counter_;
    k_ray_intersect_[1] = pose_;
    k_ray_intersect_[2] = directions_;
    k_ray_intersect_[3] = hits_t_;
    k_ray_intersect_[4] = rays_o_;
    k_ray_intersect_[5] = rays_d_;

    k_raymarching_test_kernel_ = module_.get_kernel("raymarching_test_kernel");
    k_raymarching_test_kernel_[0] = counter_;
    k_raymarching_test_kernel_[1] = density_bitfield_;
    k_raymarching_test_kernel_[2] = hits_t_;
    k_raymarching_test_kernel_[3] = alive_indices_;
    k_raymarching_test_kernel_[4] = rays_o_;
    k_raymarching_test_kernel_[5] = rays_d_;
    k_raymarching_test_kernel_[6] = current_index_;
    k_raymarching_test_kernel_[7] = xyzs_;
    k_raymarching_test_kernel_[8] = dirs_;
    k_raymarching_test_kernel_[9] = deltas_;
    k_raymarching_test_kernel_[10] = ts_;
    k_raymarching_test_kernel_[11] = run_model_ind_;
    k_raymarching_test_kernel_[12] = N_eff_samples_;

    k_rearange_index_ = module_.get_kernel("rearange_index");
    k_rearange_index_[0] = model_launch_;
    k_rearange_index_[1] = pad_block_network_;
    k_rearange_index_[2] = temp_hit_;
    k_rearange_index_[3] = run_model_ind_;

    k_hash_encode_ = module_.get_kernel("hash_encode");
    k_hash_encode_[0] = hash_embedding_;
    k_hash_encode_[1] = model_launch_;
    k_hash_encode_[2] = xyzs_;
    k_hash_encode_[3] = dirs_;
    k_hash_encode_[4] = deltas_;
    k_hash_encode_[5] = xyzs_embedding_;
    k_hash_encode_[6] = temp_hit_;

    k_mlp_layer_ = module_.get_kernel("sigma_rgb_layer");
    k_mlp_layer_[0] = sigma_weights_;
    k_mlp_layer_[1] = rgb_weights_;
    k_mlp_layer_[2] = model_launch_;
    k_mlp_layer_[3] = pad_block_network_;
    k_mlp_layer_[4] = xyzs_embedding_;
    k_mlp_layer_[5] = dirs_;
    k_mlp_layer_[6] = out_1_;
    k_mlp_layer_[7] = out_3_;
    k_mlp_layer_[8] = temp_hit_;

    k_composite_test_ = module_.get_kernel("composite_test");
    k_composite_test_[0] = counter_;
    k_composite_test_[1] = alive_indices_;
    k_composite_test_[2] = rgb_;
    k_composite_test_[3] = opacity_;
    k_composite_test_[4] = current_index_;
    k_composite_test_[5] = deltas_;
    k_composite_test_[6] = ts_;
    k_composite_test_[7] = out_3_;
    k_composite_test_[8] = out_1_;
    k_composite_test_[9] = N_eff_samples_;
    k_composite_test_[11] = kThreshold;

    k_re_order_ = module_.get_kernel("re_order");
    k_re_order_[0] = counter_;
    k_re_order_[1] = alive_indices_;
    k_re_order_[2] = current_index_;

    check_taichi_error("get_kernel failed");
}

// This function rotates the camera
void App_nerf_f32::pose_rotate_scale(float angle_x, float angle_y, float angle_z, float radius) {
    k_rotate_scale_[0] = pose_;
    k_rotate_scale_[1] = angle_x;
    k_rotate_scale_[2] = angle_y;
    k_rotate_scale_[3] = angle_z;
    k_rotate_scale_[4] = radius;
    
    k_rotate_scale_.launch();
    runtime_.wait();
}

std::vector<float> App_nerf_f32::run() {

  for (int n_time = 0; n_time < kRepeat; n_time += 1) {
    k_reset_.launch();
    k_ray_intersect_.launch();

    samples = 0;
    while (samples < kMaxSamples) {
      counter_.slice().copy_to(counter_stage_.slice());
      runtime_.wait();

      counter_stage_.read(counter);
      int n_alive = counter[0];
      if (n_alive == 0) {
        break;
      }

      int N_samples = std::max(std::min(int(kNumRays / n_alive), 64), 1);
      samples += N_samples;
      int launch_model_total = n_alive * N_samples;

      k_raymarching_test_kernel_[13] = N_samples;
      k_raymarching_test_kernel_.launch();

      k_rearange_index_[4] = launch_model_total;
      k_rearange_index_.launch();

      k_hash_encode_.launch();

      k_mlp_layer_.launch();
      k_composite_test_[10] = N_samples;
      k_composite_test_.launch();

      k_re_order_[3] = n_alive;
      k_re_order_.launch();

      check_taichi_error("render a frame failed");
    }
    runtime_.wait();
  }

  rgb_.slice().copy_to(rgb_stage_.slice());
  runtime_.wait();
  std::vector<float> img(kWidth * kHeight * 3);
  rgb_stage_.read(img);

  return img;
}
