#include <iostream>
#include <vector>
#include <taichi/cpp/taichi.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_writer.h"
#include "app_fp32.hpp"
#include "utils.hpp"

int main(int argc, const char **argv) {
  std::string aot_file_root = "compiled/";
  std::string hash_embedding_path = aot_file_root + "hash_embedding.bin";
  std::string sigma_weights_path = aot_file_root + "sigma_weights.bin";
  std::string rgb_weights_path = aot_file_root + "rgb_weights.bin";
  std::string density_bitfield_path = aot_file_root + "density_bitfield.bin";
  std::string pose_path = aot_file_root + "pose.bin";
  std::string directions_path = aot_file_root + "directions.bin";
  
  // --------------- //
  // Initialization  //
  // --------------- //
  TiArch arch = TI_ARCH_VULKAN;
  App_nerf_f32 app = App_nerf_f32(arch);
  
  // Modify Width & Height to stay consistent with what used in the taichi code
  // In this demo, we used 300 x 600 since it's generated from:
  //      python3 taichi_ngp.py --scene smh_lego --aot --res_w=300 --res_h=600
  int img_width  = 300;
  int img_height = 600;
  app.initialize(img_width, img_height,
                 aot_file_root,
                 hash_embedding_path,
                 sigma_weights_path,
                 rgb_weights_path,
                 density_bitfield_path,
                 pose_path,
                 directions_path);
  std::cout << "Inference Initialized" << std::endl;
  
  // --------------- //
  // Inference Run   //
  // --------------- //
  float angle_x, angle_y, angle_z = 0.0;
  float radius = 2.5;
  // Camera rotation: connect "angles" & "radius" with Android Touch Event to make it a real Demo
  
  app.pose_rotate_scale(angle_x, angle_y, angle_z, radius);
  
  std::vector<float> img = app.run();
  std::cout << "Inference Finished" << std::endl;

  // ---------------------------------------- //
  // Postprocess and display inference output //
  // ---------------------------------------- //
  unsigned char data[img_width * img_height * 3];
  for (int i = 0; i < img_width * img_height * 3; i++) {
      data[i] = uint8_t(img[i] * 255);
  }
  
  // Write to disk: display this image to Android Screen to make it a real Demo
  stbi_write_png("out.png", img_width, img_height, /*components=*/3, data, img_width * 3);
  std::cout << "Inference End" << std::endl;

  return 0;
}
