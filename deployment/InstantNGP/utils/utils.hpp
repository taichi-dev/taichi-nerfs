#pragma once

#include <vector>
#include <string>

uint16_t to_float16(float in);

float to_float32(uint16_t in);

std::vector<float> read_float32_array(const std::string& filename);
std::vector<uint16_t> read_float16_array(const std::string& filename);
std::vector<int32_t> read_int32_array(const std::string& filename);
std::vector<int16_t> read_int16_array(const std::string& filename);
std::vector<uint32_t> read_uint32_array(const std::string& filename);
std::vector<uint16_t> read_uint16_array(const std::string& filename);
