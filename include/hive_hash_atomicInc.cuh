#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include "hive_hash_table_struct.cuh"
#include "HashPolicies.cuh"
#include "hash_table_struct.h"
#include "hash.hpp"
#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template<typename TableType, typename Keytype>
__device__ __forceinline__ bool atomic_inc_value(
    TableType* __restrict__ table,
    uint32_t bucket_idx,
    Keytype key,
    uint32_t increment_value,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const size_t lane_id = tile.thread_rank();

    bool match_found = false;
    bool match_first = false;

    //vectorized load of KV pairs
    ulonglong2* bucket_vector = reinterpret_cast<ulonglong2*>(table->buckets[bucket_idx].kv);
    ulonglong2 two_kvs = load_cg_safe(&bucket_vector[lane_id]);

    match_first = (unpackKey(two_kvs.x) == key);
    match_found = match_first || (unpackKey(two_kvs.y) == key);

    unsigned mask = tile.ballot(match_found);

    if(!mask)
        return false;
    
    int winnner_lane = __ffs(mask) - 1; //Get the first matching lane
    bool success = false;

    if(lane_id == winnner_lane)
    {
        int slot_offset = match_first ? 0 : 1;
        auto* kv_ptr = table->loadKV(bucket_idx, lane_id * 2 + slot_offset);
        
        // Use native atomicAdd on the high 32 bits (the value part)
        // Since KV = (value << 32) | key, adding (inc << 32) increments value without touching key.
        atomicAdd((unsigned long long*)kv_ptr, (unsigned long long)increment_value << 32);
        success = true;
    }

    return tile.shfl(success, winnner_lane);
}

template<typename TableType, typename Keytype, typename Valuetype, typename HashPolicy>
__device__ __forceinline__ bool hive_atomic_inc_coop(
    const TableType* __restrict__  table,
    Keytype key,
    Valuetype increment_value,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const unsigned active = tile.ballot(true);
    const unsigned lane_id = tile.thread_rank();

    if(key == SENTINEL) return false;

    #pragma unroll
    for(size_t i = 0; i < HashPolicy::NumHashes; i++)
    {
        const uint32_t bucket = HashPolicy::get_bucket(i, key, table->num_buckets);
        if(atomic_inc_value<TableType, Keytype>(const_cast<TableType*>(table), bucket, key, increment_value, tile))
            return true;
    }

    return false;
}