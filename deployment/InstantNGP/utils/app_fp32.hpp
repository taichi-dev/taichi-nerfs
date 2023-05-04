#pragma once

#include "taichi/cpp/taichi.hpp"

class App_nerf_f32 {
public:
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
  ti::Kernel k_rotate_scale_;
  
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
    
  unsigned int kWidth;
  unsigned int kHeight;
  unsigned int kNumRays;
  unsigned int kNumMaxSamples;
  unsigned int kNumLevel = 16;
  unsigned int kMaxSamplePerRay = 1;
  int kMaxSamples = 100;
  float kThreshold = 1e-2;
  int kRepeat = 1;

  // runtime variables
  int samples = 0;
  std::vector<int> counter{1};

  double elapsed_time;

  App_nerf_f32(TiArch arch);
    
  void initialize(int img_width, int img_height,
                  const std::string& aot_file_path,
                  const std::string& hash_embedding_path,
                  const std::string& sigma_weights_path,
                  const std::string& rgb_weights_path,
                  const std::string& density_bitfield_path,
                  const std::string& pose_path,
                  const std::string& directions_path);
    
  std::vector<float> run();
  void pose_rotate_scale(float angle_x, float angle_y, float angle_z, float radius);

};

