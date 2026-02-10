#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


inline void checkCudaDevice() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found.");
    }
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (deviceProp.major < 1) {
        throw std::runtime_error("CUDA device does not support required features.");
    }
    std::cout << "Device Name: " << deviceProp.name << std::endl;
}
