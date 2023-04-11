//
//  utils.cpp
//  TaichiNerfTestbench
//
//  Created by Zhanlue Yang on 2023/2/22.
//

#include "utils.hpp"
#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>

// based on https://gist.github.com/martin-kallman/5049614
// float32
// Martin Kallman
//
// Fast half-precision to single-precision floating point conversion
//  - Supports signed zero and denormals-as-zero (DAZ)
//  - Does not support infinities or NaN
//  - Few, partially pipelinable, non-branching instructions,
//  - Core opreations ~6 clock cycles on modern x86-64
static void float32(float *__restrict out, const uint16_t in) {
    uint32_t t1;
    uint32_t t2;
    uint32_t t3;

    t1 = in & 0x7fffu;                       // Non-sign bits
    t2 = in & 0x8000u;                       // Sign bit
    t3 = in & 0x7c00u;                       // Exponent

    t1 <<= 13u;                              // Align mantissa on MSB
    t2 <<= 16u;                              // Shift sign bit into position

    t1 += 0x38000000;                       // Adjust bias

    t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero

    t1 |= t2;                               // Re-insert sign bit

    *((uint32_t *) out) = t1;
}

// float16
// Martin Kallman
//
// Fast single-precision to half-precision floating point conversion
//  - Supports signed zero, denormals-as-zero (DAZ), flush-to-zero (FTZ),
//    clamp-to-max
//  - Does not support infinities or NaN
//  - Few, partially pipelinable, non-branching instructions,
//  - Core opreations ~10 clock cycles on modern x86-64
static void float16(uint16_t *__restrict out, const float in) {
    uint32_t inu = *((uint32_t * ) & in);
    uint32_t t1;
    uint32_t t2;
    uint32_t t3;

    t1 = inu & 0x7fffffffu;                 // Non-sign bits
    t2 = inu & 0x80000000u;                 // Sign bit
    t3 = inu & 0x7f800000u;                 // Exponent

    t1 >>= 13u;                             // Align mantissa on MSB
    t2 >>= 16u;                             // Shift sign bit into position

    t1 -= 0x1c000;                         // Adjust bias

    t1 = (t3 < 0x38800000u) ? 0 : t1;       // Flush-to-zero
    t1 = (t3 > 0x8e000000u) ? 0x7bff : t1;  // Clamp-to-max
    t1 = (t3 == 0 ? 0 : t1);               // Denormals-as-zero

    t1 |= t2;                              // Re-insert sign bit

    *((uint16_t *) out) = t1;
}

uint16_t to_float16(float in) {
    uint16_t out;
    float16(&out, in);
    return out;
}

float to_float32(uint16_t in) {
    float out;
    float32(&out, in);
    return out;
}

enum class BufferDtype {
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    Int16 = 3,
    Uint32 = 4,
    Uint16 = 5
};

static std::pair<BufferDtype, std::vector<unsigned char>> read_binary_file_impl(const std::string& filename) {
    // Binary Header: int32(dtype) int32(num_elements)
    // Binary Contents: flat binary buffer

    std::ifstream input( filename, std::ios::binary );
    if(!input.is_open()) {
        assert(false && "Unable to read file");
    }
    
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
     
    /* -------------- */
    /* Parsing Header */
    /* -------------- */
    // int32(dtype): 0(float32), 1(float16)
    // int32(num_elements)
    unsigned char* dtype_ptr = buffer.data();
    int32_t* dtype_ptr_1 = reinterpret_cast<int32_t*>(dtype_ptr);
    int32_t dtype = *dtype_ptr_1;
    
    auto* numel_ptr = buffer.data() + sizeof(int32_t);
    int32_t num_elements = *reinterpret_cast<int32_t*>(numel_ptr);
    
    int vector_size = num_elements;
    BufferDtype buffer_dtype;
    switch(dtype) {
        case 0: {
            buffer_dtype = BufferDtype::Float32;
            vector_size *= sizeof(float);
            break;
        }
        case 1: {
            buffer_dtype = BufferDtype::Float16;
            vector_size *= sizeof(unsigned short);
            break;
        }
        case 2: {
            buffer_dtype = BufferDtype::Int32;
            vector_size *= sizeof(int32_t);
            break;
        }
        case 3: {
            buffer_dtype = BufferDtype::Int16;
            vector_size *= sizeof(int16_t);
            break;
        }
        case 4: {
            buffer_dtype = BufferDtype::Uint32;
            vector_size *= sizeof(uint32_t);
            break;
        }
        case 5: {
            buffer_dtype = BufferDtype::Uint16;
            vector_size *= sizeof(uint16_t);
            break;
        }
        default: {
            throw std::runtime_error("Invalid buffer dtype");
            break;
        }
    }

    
    // Validate buffer size
    int num_bytes_ref = sizeof(int32_t) /*dtype*/ + sizeof(int32_t) /*num_elements*/ + vector_size;
    assert(num_bytes_ref == buffer.size() && "Invalid Buffer size");

    /* ---------------- */
    /* Parsing Contents */
    /* ---------------- */
    auto* content_ptr = numel_ptr + sizeof(int32_t);
    std::vector<unsigned char> contents(vector_size);

    std::memcpy(contents.data(), content_ptr, vector_size);

    return {buffer_dtype, std::move(contents)};
}

template<typename T>
std::vector<T> read_binary_file(const std::string& filename, BufferDtype type) {
    auto res = read_binary_file_impl(filename);
    
    assert(type == res.first && "Dtype mismatch");

    int num_elements = res.second.size() / sizeof(T);
    std::vector<T> arr(num_elements);
    std::memcpy(arr.data(), res.second.data(), res.second.size());
    
    return arr;
}

std::vector<float> read_float32_array(const std::string& filename) {
    return read_binary_file<float>(filename, BufferDtype::Float32);
}

std::vector<uint16_t> read_float16_array(const std::string& filename) {
    return read_binary_file<uint16_t>(filename, BufferDtype::Float16);
}

std::vector<int32_t> read_int32_array(const std::string& filename) {
    return read_binary_file<int32_t>(filename, BufferDtype::Int32);
}

std::vector<int16_t> read_int16_array(const std::string& filename) {
    return read_binary_file<int16_t>(filename, BufferDtype::Int16);
}

std::vector<uint32_t> read_uint32_array(const std::string& filename) {
    return read_binary_file<uint32_t>(filename, BufferDtype::Uint32);
}

std::vector<uint16_t> read_uint16_array(const std::string& filename) {
    return read_binary_file<uint16_t>(filename, BufferDtype::Uint16);
}
