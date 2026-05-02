#include "GPUTimer.h"
#include "warpspeed.cuh"
#include "hive_kernels.cuh"
#include "utils.h"
#include "cuda_helper.cuh"
#include <iostream>

using namespace warpspeed;

// Using standard 8 slots per bucket, 32-bit keys and values
using WS_Table = cuckoo_ht<
    uint32_t, 0, 0xFFFFFFFFu,
    uint32_t, 0, 0xFFFFFFFFu,
    TILE_SIZE, 8 
>;

__global__ void init_warpspeed_table(WS_Table::bucket_type* buckets, uint32_t* locks, size_t n_buckets) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_buckets) {
        buckets[tid].init();
        locks[tid] = 0;
    }
}

void run_warpspeed_mixed_workload(
    Operation* h_ops,
    size_t num_ops,
    size_t table_size, // this is total keys capacity
    size_t threads_per_block,
    size_t numIterations,
    double &elapsed_time,
    uint64_t* h_results,
    bool verify
) {
    std::cout << "Initializing Warpspeed Hash Table for Mixed Workload..." << std::endl;
    size_t n_buckets = (table_size + 7) / 8; // bucket size 8

    WS_Table::bucket_type* d_buckets;
    uint32_t* d_locks;
    
    CUDA_CHECK(cudaMalloc(&d_buckets, sizeof(WS_Table::bucket_type) * n_buckets));
    CUDA_CHECK(cudaMalloc(&d_locks, sizeof(uint32_t) * n_buckets));

    size_t num_blocks_init = (n_buckets + threads_per_block - 1) / threads_per_block;
    init_warpspeed_table<<<num_blocks_init, threads_per_block>>>(d_buckets, d_locks, n_buckets);
    CUDA_CHECK(cudaDeviceSynchronize());

    WS_Table table;
    table.primary_buckets = d_buckets;
    table.primary_locks = d_locks;
    table.n_buckets_primary = n_buckets;
    table.seed = 1337;

    WS_Table* d_table;
    CUDA_CHECK(cudaMalloc(&d_table, sizeof(WS_Table)));
    CUDA_CHECK(cudaMemcpy(d_table, &table, sizeof(WS_Table), cudaMemcpyHostToDevice));

    Operation* d_ops;
    CUDA_CHECK(cudaMalloc(&d_ops, sizeof(Operation) * num_ops));
    CUDA_CHECK(cudaMemcpy(d_ops, h_ops, sizeof(Operation) * num_ops, cudaMemcpyHostToDevice));

    uint64_t* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(uint64_t) * num_ops));
    CUDA_CHECK(cudaMemset(d_results, 0, sizeof(uint64_t) * num_ops));

    size_t num_blocks = (num_ops + threads_per_block - 1) / threads_per_block;

    // Warmup
    warpspeed_mixed_kernel<WS_Table><<<num_blocks, threads_per_block>>>(
        d_table, d_ops, num_ops, d_results
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Re-init for actual benchmarking
    init_warpspeed_table<<<num_blocks_init, threads_per_block>>>(d_buckets, d_locks, n_buckets);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float milliseconds = 0;

    for(size_t iter = 0; iter < numIterations; iter++) {
        // Clear table each iteration so it doesn't get 100% full
        init_warpspeed_table<<<num_blocks_init, threads_per_block, 0, stream>>>(d_buckets, d_locks, n_buckets);
        
        CoarseGrainedGPUTimer timer;
        timer.start(stream);
        warpspeed_mixed_kernel<WS_Table><<<num_blocks, threads_per_block, 0, stream>>>(
            d_table, d_ops, num_ops, d_results
        );
        timer.stop(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        milliseconds += timer.getElapsedTime();
    }

    elapsed_time = milliseconds / numIterations;
    std::cout << "Warpspeed Average Time per Iteration: " << elapsed_time << " ms\n";
    std::cout << "Warpspeed Throughput: " << static_cast<double>(num_ops)/(elapsed_time/1000.0)/1e6 << " Mops/sec\n";

    if (verify && h_results) {
        CUDA_CHECK(cudaMemcpy(h_results, d_results, sizeof(uint64_t) * num_ops, cudaMemcpyDeviceToHost));
        
        size_t unsuccessful_ops = 0;
        for (size_t i = 0; i < num_ops; ++i) {
            if (h_results[i] == 0) unsuccessful_ops++;
        }
        std::cout << "Warpspeed Unsuccessful ops: " << unsuccessful_ops << " out of " << num_ops 
                  << ", Success Rate: " << (1.0 - (unsuccessful_ops / static_cast<float>(num_ops))) * 100 << "%\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_buckets));
    CUDA_CHECK(cudaFree(d_locks));
    CUDA_CHECK(cudaFree(d_table));
    CUDA_CHECK(cudaFree(d_ops));
    CUDA_CHECK(cudaFree(d_results));
}
