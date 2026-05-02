#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdint>
#include "hive_hash_table_struct.cuh"
#include "hive_stash_table.cuh"
#include "hash_table_struct.h"
#include "HashPolicies.cuh"
#include "hash.hpp"
#include "utils.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;


template<typename TableType, bool BYPASS_L1 = true>
__device__ __forceinline__ bool scan_bucket_for_key(
    const TableType* __restrict__ table,
    uint32_t bucket_idx,
    key_type key,
    value_type* value_out,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const unsigned active = tile.ballot(true);
    const unsigned lane_id = tile.thread_rank();

    bool match = false;
    ValueType found_value = 0;

    //uint64_t kv = *(table->loadKV(bucket_idx, lane_id));
    
    //vectorized load using native ulonglong2 type
    ulonglong2* bucket_vec = reinterpret_cast<ulonglong2*>(table->buckets[bucket_idx].kv);
    //ulonglong2 two_kvs = load_cg_safe(&bucket_vec[lane_id]);
    
    //acquire load
    ulonglong2 two_kvs = load_two_kvs<BYPASS_L1>(bucket_vec, lane_id);

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

template<typename TableType, typename KeyType, typename ValueType, bool BYPASS_L1 = true>
__device__ __forceinline__ bool scan_stash_for_key(
    const TableType* __restrict__ table,
    const HiveOverflowStashBucket<kv_type>* __restrict__ stash,
    uint32_t bucket_idx,
    KeyType key,
    ValueType* value_out,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    // Metadata Gating: Fast local VRAM check. If bit 31 is 0, stash is empty.
    uint32_t primary_mask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
        *const_cast<uint32_t*>(table->getFreeMask(bucket_idx))).load(cuda::memory_order_relaxed);
    if ((primary_mask & 0x80000000u) == 0) return false;

    const int lane_id = tile.thread_rank();

    auto count_ref = cuda::atomic_ref<int, cuda::thread_scope_device>(
        const_cast<int&>(stash[bucket_idx].count)
    );

    int count = count_ref.load(cuda::memory_order_acquire);

    // Early-exit when no slots have been claimed — stash is empty.
    // Do NOT use count as the read upper bound: push_to_stash increments
    // count atomically BEFORE writing the KV, so a reader can see the new
    // count while the KV slot still holds EMPTY_KV.  Instead we scan all
    // SLOTS unconditionally — TILE_SIZE=4 lanes × 2 slots = 8 = SLOTS, so
    // the cost is exactly one 128-bit load per lane regardless.
    if(count <= 0) return false;

    const auto* bucket_vec = reinterpret_cast<const ulonglong2*>(stash[bucket_idx].kv);

    bool match = false;
    ValueType found_value = 0;

    // Always scan all SLOTS — every lane reads its fixed pair of slots.
    {
        ulonglong2 two_kvs = load_two_kvs<BYPASS_L1>(const_cast<ulonglong2*>(bucket_vec), lane_id);

        bool match1 = (unpackKey(two_kvs.x) == key);
        bool match2 = (unpackKey(two_kvs.y) == key);

        match = match1 || match2;

        if (match) {
            found_value = match1 ? unpackValue(two_kvs.x) : unpackValue(two_kvs.y);
        }
    }

    unsigned match_mask = tile.ballot(match);
    if(match_mask == 0) //no match
        return false;

    int winner_lane = __ffs(match_mask) - 1; //-1 if no match

    *value_out = tile.shfl(found_value, winner_lane);

    return true;
}


template<typename TableType, typename KeyType, typename ValueType, typename HashPolicy, bool BYPASS_L1 = false>
__device__ __forceinline__  bool hive_lookup_one_coop(
    const TableType* __restrict__  table,
    const HiveOverflowStashBucket<kv_type>* __restrict__ stash,
    kv_type key,
    value_type* __restrict__ value_out,
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
        if(scan_bucket_for_key<TableType, BYPASS_L1>(table, bucket, key, value_out, tile))

            return true;
    }

    // Check corresponding stash bucket (Canonical Anchoring)
    const uint32_t canonical = HashPolicy::get_bucket(0, key, table->num_buckets);
    if(scan_stash_for_key<TableType, KeyType, ValueType, BYPASS_L1>(table, stash, canonical, key, value_out, tile))
        return true;

    return false;
}