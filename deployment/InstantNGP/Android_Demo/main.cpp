#include <cmath>
#include <ctime>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>
#include "utils.hpp"
#include <vulkan/vulkan.h>
#include <taichi/cpp/taichi.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_writer.h"

namespace {
void check_taichi_error(const std::string &msg) {
  TiError error = ti_get_last_error(0, nullptr);
  if (error < TI_ERROR_SUCCESS) {
    throw std::runtime_error(msg);
  }
}
}  // namespace

struct App_nerf {
  ti::Runtime runtime_;
  ti::AotModule module_;
  ti::Kernel k_reset_;
  ti::Kernel k_ray_intersect_;
  ti::Kernel k_raymarching_test_kernel_;
  ti::Kernel k_rearange_index_;
  ti::Kernel k_hash_encode_;
  ti::Kernel k_mlp_layer_;
  ti::Kernel k_composite_test_;
  ti::Kernel k_re_order_;
  ti::Kernel k_fill_ndarray_;
  ti::Kernel k_init_current_index_;
  
  // ndarrays
  ti::NdArray<float> pose_;
  ti::NdArray<float> hash_embedding_;
  ti::NdArray<float> rgb_weights_;
  ti::NdArray<float> sigma_weights_;
  ti::NdArray<unsigned int> density_bitfield_;
  ti::NdArray<float> directions_;
  ti::NdArray<int> counter_;
  ti::NdArray<float> hits_t_;
  ti::NdArray<int> alive_indices_;
  ti::NdArray<float> opacity_;
  ti::NdArray<float> rays_o_;
  ti::NdArray<float> rays_d_;
  ti::NdArray<float> rgb_;
  ti::NdArray<int> current_index_;
  ti::NdArray<int> model_launch_;
  ti::NdArray<int> pad_block_network_;
  ti::NdArray<float> xyzs_;
  ti::NdArray<float> dirs_;
  ti::NdArray<float> deltas_;
  ti::NdArray<float> ts_;
  ti::NdArray<float> xyzs_embedding_;
  ti::NdArray<float> final_embedding_;
  ti::NdArray<float> out_3_;
  ti::NdArray<float> out_1_;
  ti::NdArray<int> temp_hit_;
  ti::NdArray<int> run_model_ind_;
  ti::NdArray<int> N_eff_samples_;

  // Persistent staging bufs
  ti::NdArray<int> counter_stage_;
  ti::NdArray<float> rgb_stage_;

  // constants
  unsigned int kSigmaLayerBase = 16 * 16;
  unsigned int kLayer1Base = 32 * 16;
  unsigned int kLayer2Base = kLayer1Base + 16 * 16;
  unsigned int kGridSize = 128;
  unsigned int kCascades = 1;
  unsigned int kWidth = 800;
  unsigned int kHeight = 800;
  unsigned int kNumRays = kWidth * kHeight;
  unsigned int kNumLevel = 16;
  unsigned int kMaxSamplePerRay = 1;
  unsigned int kNumMaxSamples = kNumRays * kMaxSamplePerRay;
  int kMaxSamples = 100;
  float kThreshold = 1e-2;
  int kRepeat = 100;

  // runtime variables
  int samples = 0;
  std::vector<int> counter{1};

  double elapsed_time;

  template <typename T>
  void copy_from_vector(const std::vector<T> &from, ti::NdArray<T> &to) {
    auto tmp = runtime_.allocate_ndarray<T>(
        {static_cast<unsigned int>(from.size())}, {}, /*host_accessible=*/true);
    tmp.write(from);
    tmp.slice().copy_to(to.slice());
    tmp.destroy();
  }

  App_nerf() {
    runtime_ = ti::Runtime(TI_ARCH_VULKAN);
    module_ = runtime_.load_aot_module("instant_ngp/assets/compiled");
    check_taichi_error("load_aot_module failed");

    k_fill_ndarray_ = module_.get_kernel("fill_ndarray");
    k_init_current_index_ = module_.get_kernel("init_current_index");

    hash_embedding_ = runtime_.allocate_ndarray<float>(
        {11176096}, {}, /*host_accessible=*/false);
    
    std::vector<float> hash_embedding = read_float32_array("instant_ngp/assets/compiled/hash_embedding.bin");
    copy_from_vector<float>(hash_embedding, hash_embedding_);

    sigma_weights_ = runtime_.allocate_ndarray<float>(
        { kSigmaLayerBase + 16 * 16}, {}, /*host_accessible=*/false);
    
    auto sigma_weights = read_float32_array("instant_ngp/assets/compiled/sigma_weights.bin");
    copy_from_vector<float>(sigma_weights, sigma_weights_);

    rgb_weights_ = runtime_.allocate_ndarray<float>({kLayer2Base}, {},
                                                    /*host_accessible=*/false);
    auto rgb_weights = read_float32_array("instant_ngp/assets/compiled/rgb_weights.bin");
    copy_from_vector<float>(rgb_weights, rgb_weights_);

    density_bitfield_ = runtime_.allocate_ndarray<unsigned int>(
        {kCascades * kGridSize * kGridSize * kGridSize / 32}, {},
        /*host_accessible=*/false);
    auto density_bitfield = read_uint32_array("instant_ngp/assets/compiled/density_bitfield.bin");
    copy_from_vector<unsigned int>(density_bitfield, density_bitfield_);

    pose_ =
        runtime_.allocate_ndarray<float>({}, {3, 4}, /*host_accessible=*/false);
    auto pose = read_float32_array("instant_ngp/assets/compiled/pose.bin");
    copy_from_vector<float>(pose, pose_);

    directions_ = runtime_.allocate_ndarray<float>({kNumRays}, {1, 3},
                                                   /*host_accessible=*/false);
    auto directions = read_float32_array("instant_ngp/assets/compiled/directions.bin");
    copy_from_vector<float>(directions, directions_);

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

    // Most kernel arguments don't change so let's initialize them here.
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
    std::cout << "Initialized!" << std::endl;
  }

  void run() {
    std::cout << "Running " << kRepeat << " frames" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

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
        std::cout << "samples: " << samples << " n_alive: " << n_alive
                  << " N_samples: " << N_samples << std::endl;
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
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    std::cout << "render a frame time: " << elapsed_time / kRepeat << " ms"
              << std::endl;

    rgb_.slice().copy_to(rgb_stage_.slice());
    runtime_.wait();
    std::vector<float> img(kWidth * kHeight * 3);
    rgb_stage_.read(img);
    unsigned char data[kWidth * kHeight * 3];
    for (int i = 0; i < kWidth * kHeight * 3; i++) {
      data[i] = uint8_t(img[i] * 255);
    }
    stbi_write_png("out.png", kWidth, kHeight, /*components=*/3, data,
                   kWidth * 3);
  }
};

int main(int argc, const char **argv) {
  App_nerf app;
  app.run();
  return 0;
}
