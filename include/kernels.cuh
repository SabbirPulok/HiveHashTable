#pragma once
#include <cuda_runtime.h>
#include "hash_table_struct.h"
#include <vector>
#include <cstdint>

//restrict = no pointer aliasing, no other memory locations will be used to access the same memory
__global__ void coalesced_read(const uint64_t __restrict__* data, uint64_t __restrict__* out, size_t R, size_t N);


__global__ void linear_probing_lookup_kernel(
    const HashEntry* __restrict__ table,
    const uint64_t* __restrict__ query_keys,
    uint64_t* __restrict__ results,
    size_t table_size,
    size_t num_queries,
    size_t max_probes = 32
);

__global__ void cuckoo_lookup_kernel(
    const HashEntry* __restrict__ table1,
    const HashEntry* __restrict__ table2,
    const uint64_t* __restrict__ query_keys,
    uint64_t* __restrict__ results,
    size_t table_size,
    size_t num_queries
);

void launch_linear_probing_lookup_kernel(
    const HashEntry* d_table,
    size_t table_size,
    const uint64_t* d_query_keys,
    uint64_t* d_results,
    size_t num_queries,
    int num_blocks,
    int threads_per_block,
    double &elapsed_time
);

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
);