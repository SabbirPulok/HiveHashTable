#pragma once
#ifdef __CUDACC__
#ifndef HIVE_STASH_TABLE_CUH
#define HIVE_STASH_TABLE_CUH

#include <cuda_runtime.h>
#include <cuda/atomic>
#include "cuda_helper.cuh"
#include "utils.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

using KeyType = key_type;
using ValueType = value_type;
// HiveOverflowStashBucket
template<typename KVType, size_t N = 8>
struct __align__(128) HiveOverflowStashBucket {

    static constexpr size_t SLOTS = N;

    alignas(sizeof(KVType)) KVType kv[SLOTS]; // EMPTY_KV=0 means empty
    int count;                                 // atomicAdd target
};


// compact_stash_kernel
// One thread per stash bucket. Scans kv[0..count-1], copies non-empty
// entries into a flat output array using an atomic counter for the base index.
template<typename KVType>
__global__ void compact_stash_kernel(
    HiveOverflowStashBucket<KVType>* stash_table,
    size_t num_buckets,
    uint64_t* out_flat_stash,
    uint32_t* out_count
)
{
    size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_buckets) return;

    auto count_ref = cuda::atomic_ref<int, cuda::thread_scope_device>(
        stash_table[b].count);
    int count = count_ref.load(cuda::memory_order_acquire);

    if (count <= 0) return;

    // Claim a contiguous region in the flat output array
    uint32_t base = atomicAdd(out_count, static_cast<uint32_t>(count));

    for (int s = 0; s < count; ++s)
    {
        auto kv_ref = cuda::atomic_ref<KVType, cuda::thread_scope_device>(
            stash_table[b].kv[s]);
        KVType kv = kv_ref.load(cuda::memory_order_acquire);

        if (kv != EMPTY_KV) 
        {
            out_flat_stash[base + s] = kv;
        }
    }
}


// reinsert_stash_into_next_batch
// Compacts all stash buckets into INSERT ops appended after d_ops.
// Returns the new total op count via num_ops_with_stash.
// Resets the entire stash array to zero after draining.
template<typename TableType, typename KVType>
void reinsert_stash_into_next_batch(
    TableType* __restrict__ table,
    HiveOverflowStashBucket<KVType>* __restrict__ stash_table,
    Operation* d_ops,
    size_t num_ops,
    size_t& num_ops_with_stash,
    cudaStream_t stream
)
{
    size_t num_buckets = 0;
    {
        decltype(table->num_buckets) h_num_buckets;
        CUDA_CHECK(cudaMemcpyAsync(
            &h_num_buckets,
            &table->num_buckets,
            sizeof(h_num_buckets),
            cudaMemcpyDeviceToHost,
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        num_buckets = static_cast<size_t>(h_num_buckets);
    }

    constexpr size_t STASH_SLOTS = HiveOverflowStashBucket<KVType>::SLOTS;
    size_t max_stash_entries = num_buckets * STASH_SLOTS;

    // Flat device buffer for compacted KV entries
    uint64_t* d_stash_flat = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_stash_flat,
        max_stash_entries * sizeof(uint64_t), stream));

    // Atomic counter: how many entries the compact kernel found
    uint32_t* d_stash_count = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_stash_count, sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_stash_count, 0, sizeof(uint32_t), stream));

    int compact_blocks = (num_buckets + 255) / 256; // FIX: was +256
    compact_stash_kernel<<<compact_blocks, 256, 0, stream>>>(
        stash_table,
        num_buckets,
        d_stash_flat,
        d_stash_count  
    );

    uint32_t stash_size = 0;
    CUDA_CHECK(cudaMemcpyAsync(&stash_size, d_stash_count,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (stash_size > 0)
    {
        std::cout << "Reinserting " << stash_size
                  << " stash entries into next batch" << std::endl;

        auto exec = thrust::cuda::par.on(stream);
        thrust::transform(
            exec,
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(stash_size),
            d_ops + num_ops,
            [d_stash_flat] __device__ (size_t i) {
                uint64_t kv = d_stash_flat[i];
                KeyType key = unpackKey(kv);                
                return Operation{OperationType::INSERT, static_cast<uint64_t>(key)};
            }
        );
        num_ops_with_stash = num_ops + stash_size;
    }
    else
    {
        num_ops_with_stash = num_ops;
    }

    // Single memset resets both count=0 and kv=EMPTY_KV=0
    CUDA_CHECK(cudaMemsetAsync(
        stash_table,
        0,
        num_buckets * sizeof(HiveOverflowStashBucket<KVType>),
        stream
    ));

    CUDA_CHECK(cudaFreeAsync(d_stash_flat, stream));
    CUDA_CHECK(cudaFreeAsync(d_stash_count, stream));
}

#endif // HIVE_STASH_TABLE_CUH
#endif
