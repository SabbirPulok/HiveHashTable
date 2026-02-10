#include <cuda_runtime.h>
#include <stdint.h>
#include "hash.hpp"
#include "cuda_helper.cuh"
#include "hash_table_struct.h"
#include "GPUTimer.h"
#include <vector>
#include <iostream>


//Coalesced memory access: each thread will read R consecutive elements
__global__ void coalesced_read(const uint64_t __restrict__* data, uint64_t __restrict__* out, size_t R, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t base = (long long) tid * R;

    if(base >= N) return;

    uint64_t acc = 0;

    #pragma unroll
    for(size_t i = 0; i < R; i++) {
        int idx = (base + i) % N;
        acc += data[idx];
    }

    out[tid] = acc;
}

__global__ void linear_probing_lookup_kernel(
    const HashEntry* __restrict__ table,
    const uint64_t* __restrict__ query_keys,
    uint64_t* __restrict__ results,
    size_t table_size,
    size_t num_queries,
    size_t max_probes
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= num_queries) return;

    uint64_t key = query_keys[tid];

    uint32_t hash = hash32(key, table_size);

    uint64_t result = 0; //0 means not found

    //Linear probing with bounded search
    for(uint32_t i = 0; i < max_probes; i++) {
        uint32_t idx = (hash + i) % table_size;
        HashEntry entry = table[idx];

        if(entry.key == key) {
            result = entry.value;
            break;
        } else if(entry.key == 0) {
            //Empty slot, key not found
            break;
        }
    }
}

void launch_linear_probing_lookup_kernel(
    std::vector<HashEntry> hash_table,
    std::vector<uint64_t>keys,
    std::vector<uint64_t>query_keys,
    size_t table_size,
    size_t num_keys,
    size_t num_queries,
    size_t threads_per_block,
    size_t num_blocks,
    size_t numIterations,
    double &elapsed_time
){
    std::cout<<"Launching Linear Probing Lookup Kernel..."<<std::endl;
    CoarseGraindGPUTimer gpuTimerCoarse;

    //Allocate device memory
    HashEntry* d_table;
    uint64_t* d_keys;
    uint64_t* d_query_keys;
    uint64_t* d_results;

    CUDA_CHECK(cudaMalloc(&d_table, table_size * sizeof(HashEntry)));
    CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_query_keys, num_queries * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_results, num_queries * sizeof(uint64_t)));

    //Copy data to device
    CUDA_CHECK(cudaMemcpy(d_table, hash_table.data(), table_size * sizeof(HashEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), num_keys * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query_keys, query_keys.data(), num_queries * sizeof(uint64_t), cudaMemcpyHostToDevice));

    //Launch kernel
    size_t max_probes = 16; //Set a limit on the number of probes to avoid long searches

    elapsed_time = 0.0;

    gpuTimerCoarse.start();
    for(int iter = 0; iter < numIterations; iter++) {
        linear_probing_lookup_kernel<<<num_blocks, threads_per_block>>>(
            d_table,
            d_query_keys,
            d_results,
            table_size,
            num_queries,
            max_probes
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    gpuTimerCoarse.stop();
    elapsed_time = (gpuTimerCoarse.getElapsedTime()) / numIterations; //Average time per iteration

    std::cout << "Linear Probing Lookup Kernel Time: " << elapsed_time << " ms" << std::endl;
    //Copy results back to host
    std::vector<uint64_t> results(num_queries);
    CUDA_CHECK(cudaMemcpy(results.data(), d_results, num_queries * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    //Free device memory
    CUDA_CHECK(cudaFree(d_table));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_query_keys));
    CUDA_CHECK(cudaFree(d_results));
}

__global__ void cuckoo_lookup_kernel(
    const HashEntry* __restrict__ table1,
    const HashEntry* __restrict__ table2,
    const uint64_t* __restrict__ query_keys,
    uint64_t* __restrict__ results,
    size_t table_size,
    size_t num_queries
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= num_queries) return;

    uint64_t key = query_keys[tid];

    uint32_t hash1 = hash32(key, table_size);
    uint32_t hash2 = hash32_alt(key, table_size);

    uint64_t result = 0; //0 means not found

    //Check first table
    HashEntry entry1 = table1[hash1];
    if(entry1.key == key) {
        result = entry1.value;
    } else {
        //Check second table
        HashEntry entry2 = table2[hash2];
        if(entry2.key == key) {
            result = entry2.value;
        }
    }

    results[tid] = result;
}



