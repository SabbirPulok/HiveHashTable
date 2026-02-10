#include "vector_add.h"
#include "utils.h"

__global__ void vectorAddKernel(const float *A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < N)
        C[idx] = A[idx] + B[idx];
}

void vectorAdd(const float* A, const float* B, float* C, int N)
{
    float *dA, *dB, *dC;

    cudaMalloc(&dA, N * sizeof(float));
    cudaMalloc(&dB, N * sizeof(float));
    cudaMalloc(&dC, N * sizeof(float));

    cudaMemcpy(dA, A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N*sizeof(float), cudaMemcpyHostToDevice);

    int threadPerBlock = 256;
    int blockPerGrid = (N + threadPerBlock -1) / threadPerBlock;
    vectorAddKernel<<<blockPerGrid, threadPerBlock>>> (dA, dB, dC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, N*sizeof(float), cudaMemcpyDeviceToHost);

    // for(int i=0; i<N; i++)
    // {
    //     std::cout<<"C["<<i<<"]: "<<C[i]<<std::endl;
    // }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}