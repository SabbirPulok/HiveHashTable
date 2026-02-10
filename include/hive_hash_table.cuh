//Plan to build a hash table which lcokss the bucket when a warp cooperatively inserts into it
//But when an entire warp hits the same bucket, it serializes hot buckets under common mixed load (multiple warps contending)
//Use per-bucket slot bitmap so most inserts don't need the lock at all
//We only grab it briefly during a real eviction
//Also have multiple d hash tables for load balancing

#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include "hash_table_struct.h"
#include "utils.h"
#include <cstdint>
#include "hive_hash_table_struct.cuh"
#include "HashPolicies.cuh"

#ifndef BREAKDOWN_INSERT
#define BREAKDOWN_INSERT 0
#endif

using KeyType = uint32_t;
using ValueType = uint32_t;
using KVType = uint64_t;


#ifdef __CUDACC__
//Free Slot for delete operation
__device__ __forceinline__ void freeSlot(uint32_t* pMask, int slot)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomic_mask(*(uint32_t*)pMask);
    //Sets the bit at 'slot' index to 1 (marking it as free)
    atomic_mask.fetch_or(1U << slot, cuda::memory_order_release);
}

//Bucket Locking
__device__ __forceinline__ void lockBucket(uint16_t* pLock)
{
    cuda::atomic_ref<uint16_t, cuda::thread_scope_device> atomic_lock(*pLock);
    //Spin until we successfully set the lock from 0 to 1
    while(atomic_lock.exchange(1, cuda::memory_order_acquire) != 0);
}

__device__ __forceinline__ void unlockBucket(uint16_t* pLock)
{
    cuda::atomic_ref<uint16_t, cuda::thread_scope_device> atomic_lock(*pLock);
    //Release the lock by setting it back to 0
    atomic_lock.store(0, cuda::memory_order_release);
}
#endif

//Mixed Operations (insert, delete, lookup) launch kernel
void hash_table_kernel_dispatch(
    Operation* h_ops,
    size_t num_ops,
    size_t table_size,
    size_t threads_per_block,
    size_t num_blocks,
    size_t numIterations,
    double &elapsed_time,
    uint64_t* h_results, //for lookup results
    bool verify,
    HashTableDataLayout layout,
    std::string hash_policy = "Default2Hash"
);


void hash_table_kernel_dispatch_YCSB(
    Operation* h_prefill_ops,
    Operation* h_workload_ops,
    size_t num_prefill_ops,
    size_t num_workload_ops,
    size_t table_size,
    size_t threads_per_block,
    size_t numIterations,
    double &elapsed_time,
    uint64_t* h_results, //for lookup results
    bool verify,
    HashTableDataLayout layout
);

void hive_launch_mix_ops_kernel(
    Operation* h_ops,
    size_t num_ops,
    size_t table_size,
    size_t threads_per_block,
    size_t num_blocks,
    size_t numIterations,
    double &elapsed_time,
    uint64_t* h_results, //for lookup results
    bool verify
);

void hiveStashTableCreate(
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device,
    size_t stash_capacity,
    bool stash_enabled
);

//Create Host and Device Side Hash Table from HiveHashTable
void hiveHashTableCreate(
    HiveHashTable& table_host,
    HiveHashTable* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device,
    size_t num_buckets,
    size_t max_num_buckets,
    size_t max_evictions,
    bool stash_enabled,
    size_t stash_capacity
);


void hiveHashTableCreate(
    HiveHashTableAoaS<KVType>& table_host,
    HiveHashTableAoaS<KVType>* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device,
    size_t num_buckets,
    size_t max_num_buckets,
    size_t max_evictions,
    bool stash_enabled,
    size_t stash_capacity
);

void hiveHashTableCreate(
    HiveHashTableAoaS_LeadMetaData<KVType>& table_host,
    HiveHashTableAoaS_LeadMetaData<KVType>* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device,
    size_t num_buckets,
    size_t max_num_buckets,
    size_t max_evictions,
    bool stash_enabled,
    size_t stash_capacity
);

void hiveHashTableDestroy(
    HiveHashTable& table_host, HiveHashTable* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host, HiveOverflowStash<KeyType, ValueType>* stash_device);

void hiveHashTableDestroy(
    HiveHashTableAoaS<KVType>& table_host,
    HiveHashTableAoaS<KVType>* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device
);

void hiveHashTableDestroy(
    HiveHashTableAoaS_LeadMetaData<KVType>& table_host,
    HiveHashTableAoaS_LeadMetaData<KVType>* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device
);
//Drain Stash from Device to Host
void hive_drain_stash(
    HiveOverflowStash<KeyType, ValueType>* stash_table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_table_host,
    uint32_t max_to_drain,
    cudaStream_t stream
);

#ifdef __CUDACC__
template<typename TableType, typename HashPolicy = Default2HashPolicy>
__global__ void hive_mixed_kernel(
    TableType* table,
    HiveOverflowStash<KeyType, ValueType>* stash,
    Operation* ops,
    size_t num_ops,
    uint64_t* results //output parameter
    #if BREAKDOWN_INSERT
    , InsertBreakdown* insert_breakdown
    #endif
);
#endif
