#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include "cuda.h"
#include "cuda_runtime.h"

__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N);

void vectorAdd(const float* A, const float* B, float *C, int N);

#endif

//declare CUDA Kernel and host-side wrapper