#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include "hive_hash_table_struct.cuh"
#include "HashPolicies.cuh"
#include "hash_table_struct.h"
#include "hive_stash_table.cuh"
#include "hash.hpp"
#include "utils.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template<typename TableType, typename KeyType, bool BYPASS_L1 = true>
__device__ __forceinline__ bool scan_bucket_and_delete(
    TableType* __restrict__ table,
    size_t bucket_idx,
    KeyType key,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const unsigned active = tile.ballot(true);
    const unsigned lane_id = tile.thread_rank();

    bool matchA = false;
    bool found  = false;

    ulonglong2* bucket_vec = reinterpret_cast<ulonglong2*>(table->buckets[bucket_idx].kv);
    ulonglong2 two_kvs = load_two_kvs<BYPASS_L1>(bucket_vec, lane_id);

    matchA = (unpackKey(two_kvs.x) == key);
    found  = matchA || (unpackKey(two_kvs.y) == key);

    unsigned found_mask = tile.ballot(found);

    if (found_mask == 0)
        return false;

    int winner_lane = __ffs(found_mask) - 1;

    bool deleted = false;

    if (lane_id == winner_lane)
    {
        int lane_offset = matchA ? 0 : 1;
        int slot_idx    = lane_id * 2 + lane_offset;

        uint64_t kv_to_delete = matchA ? two_kvs.x : two_kvs.y;

        auto* slot_ptr  = &table->buckets[bucket_idx].kv[slot_idx];
        auto atomic_kv  = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(*slot_ptr);

        uint64_t expected = kv_to_delete;

        while (true)
        {
            if (atomic_kv.compare_exchange_weak(
                    expected, EMPTY_KV,
                    cuda::memory_order_acq_rel,
                    cuda::memory_order_acquire))
            {
                auto atomic_mask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
                    *(table->getFreeMask(bucket_idx)));
                atomic_mask.fetch_or(1u << slot_idx, cuda::memory_order_release);
                deleted = true;
                break;
            }
            if (expected == EMPTY_KV || unpackKey(expected) != key)
            {
                // Another thread already deleted this key
                break;
            }
        }
    }

    return tile.shfl(deleted, winner_lane);
}

template<typename TableType, typename KeyType, bool BYPASS_L1 = true>
__device__ __forceinline__ bool scan_stash_for_delete(
    TableType* __restrict__ table,
    const HiveOverflowStashBucket<kv_type>* __restrict__ stash,
    uint32_t bucket_idx,
    KeyType key,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    // Metadata Gating: Fast local VRAM check. If bit 31 is 0, stash is empty.
    uint32_t primary_mask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
        *table->getFreeMask(bucket_idx)).load(cuda::memory_order_relaxed);
    if ((primary_mask & 0x80000000u) == 0) return false;

    const int lane_id = tile.thread_rank();

    auto count_ref = cuda::atomic_ref<int, cuda::thread_scope_device>(
        const_cast<int&>(stash[bucket_idx].count)
    );

    int count = count_ref.load(cuda::memory_order_acquire);

    if(count <= 0)
        return false;
    
    const auto* bucket_vec = reinterpret_cast<const ulonglong2*>(stash[bucket_idx].kv);

    bool match = false;

    ulonglong2 two_kvs = load_two_kvs<BYPASS_L1>(const_cast<ulonglong2*>(bucket_vec), lane_id);

    bool matchA = (unpackKey(two_kvs.x) == key);
    bool matchB = (unpackKey(two_kvs.y) == key);

    match = matchA || matchB;
   
    unsigned match_mask = tile.ballot(match);
    if(match_mask == 0)
        return false;
    
    int winner_lane = __ffs(match_mask) - 1;

    bool deleted = false;

    if(lane_id == winner_lane)
    {
        // Attempt to delete the matched entry by setting its key to EMPTY_KV
        int slot_offset = matchA ? 0 : 1;
        int slot_idx = lane_id * 2 + slot_offset;

        uint64_t kv_to_delete = matchA ? two_kvs.x : two_kvs.y;

        auto* slot_ptr = const_cast<uint64_t*>(&stash[bucket_idx].kv[slot_idx]);
        auto atomic_kv = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(*slot_ptr);

        uint64_t expected = kv_to_delete;

        while(true)
        {
            if (atomic_kv.compare_exchange_weak(
                    expected, EMPTY_KV,
                    cuda::memory_order_acq_rel,
                    cuda::memory_order_acquire))
            {
                // Successfully deleted the entry. 
                // DO NOT decrement count_ref; it acts as a monotonic allocation pointer.
                deleted = true;
                break;
            }
            if (expected == EMPTY_KV || unpackKey(expected) != key)
            {
                // Another thread already deleted this key or it was modified
                break;
            }
        }
    }

    return tile.shfl(deleted, winner_lane);
}
// hive_delete_one_coop — scan ALL candidate buckets, no early return.
//
// A key can reside in either of its NumHashes candidate buckets because
// cuckoo eviction may have displaced it from its canonical bucket (H0)
// to its alternate bucket (H1), or vice versa.  Stopping after the first
// successful delete would leave a stale copy in the other bucket if a
// cross-bucket duplication race previously occurred.
//
// Scanning all buckets ensures complete removal regardless of where the
// key currently lives, at the cost of one extra bucket scan per delete.
template<typename TableType, typename KeyType, typename HashPolicy, bool BYPASS_L1 = true>
__device__ __forceinline__ bool hive_delete_one_coop(
    TableType* __restrict__ table,
    const HiveOverflowStashBucket<kv_type>* __restrict__ stash,
    KeyType key,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    if (key == SENTINEL) return false;

    bool any_deleted = false;

    #pragma unroll
    for (int i = 0; i < HashPolicy::NumHashes; i++)
    {
        const size_t bucket = HashPolicy::get_bucket(i, key, table->num_buckets);
        if (scan_bucket_and_delete<TableType, KeyType, BYPASS_L1>(table, bucket, key, tile))
            any_deleted = true; 
    }

    // Check corresponding stash bucket (Canonical Anchoring)
    const uint32_t canonical = HashPolicy::get_bucket(0, key, table->num_buckets);
    if(scan_stash_for_delete<TableType, KeyType, BYPASS_L1>(table, stash, canonical, key, tile))
        any_deleted = true; 

    return any_deleted;
}