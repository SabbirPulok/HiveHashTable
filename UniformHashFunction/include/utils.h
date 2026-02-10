#ifndef UTILS_H
#define UTILS_H

#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>


//Init a vector with random values
void initVector(float* vec, int N);

bool compareVectors(const float* A, const float* B, const float* C, int N);

//check CUDA errors
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << " : line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


#endif


//declares utility function for memory management, error checking and initialization