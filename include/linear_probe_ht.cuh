#pragma once
#include <cuda_runtime.h>
#include "hash_table_struct.h"
#include <vector>
#include <cstdint>
#include <map>
#include "utils.h"

struct LinearProbeConfig{
    //User can launch multiple kernels with same params & calculate their elapsed time
    std::vector<KernelType>kernels; //kernels to launch

    //Have to store elapsed time for each kernel launched
    std::map<KernelType, double> elapsed_times;
    std::map<KernelType, double> throughput_mlops;

    double load_factor; //for insert
    size_t threads_per_block = 256;
    size_t blocks_per_grid = 1;

    size_t numIterations = 10; //for benchmarking

    //Build Inputs for insert, lookup and delete
    const std::vector<uint64_t>*insert_keys = nullptr; //for insert
    const std::vector<uint64_t>*query_keys = nullptr; //for lookup
    std::vector<uint64_t>* results = nullptr; //for lookup

    uint64_t* total_ops_results = nullptr; //for mix ops
    const std::vector<uint64_t>*delete_keys = nullptr; //for delete

    //Hash Table
    HashEntry* table = nullptr; //for insert and delete

    size_t table_size; //for insert and delete

    size_t num_queries; //for lookup
    size_t num_inserts; //for insert
    size_t num_deletes; //for delete
    size_t max_probes; //for lookup and insert
};

__global__ void lp_build_kernel(
    HashEntry* __restrict__ table,
    const uint64_t* __restrict__ keys,
    size_t table_size,
    size_t num_keys,
    size_t max_probes
);

__global__ void lp_lookup_kernel(
    const HashEntry* __restrict__ table,
    const uint64_t* __restrict__ query_keys,
    uint64_t* __restrict__ results,
    size_t table_size,
    size_t num_queries,
    size_t max_probes = 32
);

__global__ void lp_delete_kernel(
    HashEntry* __restrict__ table,
    const uint64_t* __restrict__ keys,
    size_t table_size,
    size_t num_deletes,
    size_t max_probes = 32
);

void lp_launch_kernel_with_no_mix_ops(
    LinearProbeConfig& config,
    size_t table_size, //always power of 2
    size_t threads_per_block,
    size_t num_blocks,
    size_t numIterations,
    bool verification_lp = false
);

__global__ void lp_mixed_ops(
    HashEntry* __restrict__ table,
    Operation* __restrict__ ops,
    size_t table_size,
    size_t num_ops,
    size_t max_probes,
    uint64_t* __restrict__ results
);

void lp_launch_kernel_with_mix_ops(
    LinearProbeConfig& config,
    std::vector<Operation> mix_ops,
    size_t threads_per_block,
    size_t num_blocks,
    size_t numIterations,
    bool verification_lp = false
);

