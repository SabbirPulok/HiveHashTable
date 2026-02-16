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

template<typename TableType, typename KeyType>
__device__ __forceinline__ bool scan_bucket_and_delete(
    TableType* __restrict__ table,
    size_t bucket_idx,
    KeyType key,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    // Implementation of the scan and delete operation
    // This function will scan the specified bucket for the key
    // and delete it if found, using the provided tag for
    // any necessary tagging operations.

    const unsigned active = tile.ballot(true);
    const unsigned lane_id = tile.thread_rank();

    bool matchA = false;
    bool found = false;

    // Use native unsigned long long2 for vectorized load
    ulonglong2* bucket_vec = reinterpret_cast<ulonglong2*>(table->buckets[bucket_idx].kv);
    ulonglong2 two_kvs = load_two_kvs(bucket_vec, lane_id);

    // Check both keys for a match
    matchA = (unpackKey(two_kvs.x) == key);
    found = matchA || (unpackKey(two_kvs.y) == key);

    //Collect results from all lanes
    unsigned found_mask = tile.ballot(found);

    if(found_mask == 0)
        return false; //Key not found in this bucket

    
    int winner_lane = __ffs(found_mask) - 1; //-1 if no match

    //return tile.shfl(found, winner);
    bool deleted = false;

    if(lane_id == winner_lane)
    {
        int lane_offset = matchA ? 0 : 1;
        int slot_idx = lane_id * 2 + lane_offset;

        uint64_t kv_to_delete = matchA ? two_kvs.x : two_kvs.y;


        auto* slot_ptr = &table->buckets[bucket_idx].kv[slot_idx];
        auto atomic_kv = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(*slot_ptr);
        
        uint64_t expected = kv_to_delete;
    
        while(true)
        {
            if(atomic_kv.compare_exchange_weak(
            expected,
            EMPTY_KV,
            cuda::memory_order_acq_rel,
            cuda::memory_order_acquire)) {                
                //Successfully deleted the key-value pair
                //Now mark the slot as free in the freeMask
                auto atomic_mask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
                    *(table->getFreeMask(bucket_idx)));
                atomic_mask.fetch_or(1u << slot_idx, cuda::memory_order_release);
                deleted = true;
                break;
            }
            if(expected == EMPTY_KV || unpackKey(expected) != key) {
                //Another thread has already deleted this key-value pair
                break;
            }
        }
    }

    return tile.shfl(deleted, winner_lane);
}

template<typename TableType, typename KeyType, typename HashPolicy>
__device__ __forceinline__ bool hive_delete_one_coop
(
    TableType* __restrict__ table,
    KeyType key,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    if(key == SENTINEL)
        return false; //Invalid key

    #pragma unroll
    for(int i = 0; i < HashPolicy::NumHashes; i++)
    {
        const size_t bucket = HashPolicy::get_bucket(i, key, table->num_buckets);
        if(scan_bucket_and_delete(table, bucket, key, tile))
            return true;
    }

    return false; //Key not found in either bucket
}
