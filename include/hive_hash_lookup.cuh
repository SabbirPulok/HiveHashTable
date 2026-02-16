#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdint>
#include "hive_hash_table_struct.cuh"
#include "hash_table_struct.h"
#include "HashPolicies.cuh"
#include "hash.hpp"
#include "utils.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;


template<typename TableType>
__device__ __forceinline__ bool scan_bucket_for_key(
    const TableType* __restrict__ table,
    uint32_t bucket_idx,
    KeyType key,
    ValueType* value_out,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const unsigned active = tile.ballot(true);
    const unsigned lane_id = tile.thread_rank();

    bool match = false;
    ValueType found_value = 0;
    
    //vectorized load using native ulonglong2 type
    ulonglong2* bucket_vec = reinterpret_cast<ulonglong2*>(table->buckets[bucket_idx].kv);
    
    //acquire load
    ulonglong2 two_kvs = load_two_kvs(bucket_vec, lane_id);

    // With kv packed as (value << 32) | key :
    
    bool match1 = (unpackKey(two_kvs.x) == key);
    bool match2 = (unpackKey(two_kvs.y) == key);

    match = match1 || match2;

    if (match) {
        found_value = match1 ? unpackValue(two_kvs.x) : unpackValue(two_kvs.y);
    }

    //return matching lane found value
    const unsigned match_mask = tile.ballot(match);

    if(match_mask == 0) //no match
        return false;
    
    int winner_lane = __ffs(match_mask) - 1; //-1 if no match

    *value_out = tile.shfl(found_value, winner_lane);
    
    return true;
}

template<typename TableType, typename KeyType, typename ValueType, typename HashPolicy>
__device__ __forceinline__  bool hive_lookup_one_coop(
    const TableType* __restrict__  table,
    const HiveOverflowStash<KeyType, ValueType>* __restrict__ stash,
    KeyType key,
    ValueType* value_out,
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
        if(scan_bucket_for_key(table, bucket, key, value_out, tile))

            return true;
    }

    return false;
}