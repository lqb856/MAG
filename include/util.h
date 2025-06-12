#pragma once

#include <random>
#include <iostream>
#include <cstring>
#include <algorithm>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif
namespace MAG {

    static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }

    inline float* load_data(const char* filename, unsigned& num, unsigned& dim) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
          std::cerr << "Open file error" << std::endl;
          exit(-1);
        }
      
        in.read((char*)&num, 4);
        std::cout<< "Num = " << num << std::endl;
        
        in.read((char*)&dim, 4);
        std::cout << "Dim = " << dim << std::endl;
      
        float* data = new float[(size_t)num * (size_t)dim];
        in.read((char*)data, (size_t)num * (size_t)dim * sizeof(float));
        in.close();
      
        return data;
    }

    unsigned* load_true_nn(const char* filename, unsigned& num, unsigned& k) {
      std::ifstream in(filename, std::ios::binary);
      if (!in.is_open()) {
        throw std::runtime_error("Failed to open file");
      }
    
      unsigned num_u32 = 0, k_u32 = 0;
    
      // Read header
      in.read(reinterpret_cast<char *>(&num_u32), sizeof(uint32_t));
      in.read(reinterpret_cast<char *>(&k_u32), sizeof(uint32_t));
    
      num = num_u32;
      k = k_u32;
    
      std::cout << "Loading " << num << " x " << k << " ground truth" << std::endl;
    
      // Allocate and read data
      auto data = new unsigned[num * k];
      in.read(reinterpret_cast<char *>(data), sizeof(int32_t) * num * k);
      in.close();
      return data;
    }

    inline float* data_align(float* data_ori, unsigned point_num, unsigned& dim){
      #ifdef __GNUC__
      #ifdef __AVX__
        #define DATA_ALIGN_FACTOR 8
      #else
      #ifdef __SSE2__
        #define DATA_ALIGN_FACTOR 4
      #else
        #define DATA_ALIGN_FACTOR 1
      #endif
      #endif
      #endif

      //std::cout << "align with : "<<DATA_ALIGN_FACTOR << std::endl;
      float* data_new=0;
      unsigned new_dim = (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
      //std::cout << "align to new dim: "<<new_dim << std::endl;
      #ifdef __APPLE__
        data_new = new float[new_dim * point_num];
      #else
        data_new = (float*)memalign(DATA_ALIGN_FACTOR * 4, point_num * new_dim * sizeof(float));
      #endif

      for(unsigned i=0; i<point_num; i++){
        memcpy(data_new + i * new_dim, data_ori + i * dim, dim * sizeof(float));
        memset(data_new + i * new_dim + dim, 0, (new_dim - dim) * sizeof(float));
      }
      dim = new_dim;
      #ifdef __APPLE__
        delete[] data_ori;
      #else
        free(data_ori);
      #endif
      return data_new;
    }

}
