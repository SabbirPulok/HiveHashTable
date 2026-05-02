#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include "hash_table_struct.h"
#include "hive_hash_table_struct.cuh"
#include "hash.hpp"
#include "utils.h"
#include "HashPolicies.cuh"
#include <cstdint>
#include <cstdio>
#include <array>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#ifndef BREAKDOWN_INSERT
#define BREAKDOWN_INSERT 0
#endif

template<typename TableType, typename KeyType, typename ValueType, typename HashPolicy>
__device__ void concurrent_kernel(
    TableType* __restrict__ table,
    HiveOverflowStash<KeyType, ValueType>* __restrict__ overflow_stash,
    OperationType operation,
    KeyType targetKey,
    ValueType targetValue,
    uint64_t& outResult,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const unsigned lane_id = tile.thread_rank();
    bool opResolved = false;

    ValueType foundValue = 0;

    #pragma unroll
    for (int probe = 0; probe < HashPolicy::NumHashes && !tile.any(opResolved); ++probe)
    {
        uint32_2 bucket_idx = HashPolicy::get_bucket(probe, targetKey, table->num_buckets);

        // ulonglong2* bucket_vec = reinterpret_cast<ulonglong2*>(table->buckets[bucket_idx].kv);

        // ulonglong2 two_kvs = load_two_kvs(bucket_vec, lane_id);

        uint64_t* slot_ptr = table->loadKV(bucket_idx, lane_id);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> atomic_slot(*slot_ptr);

        uint64_t loaded_kv = atomic_slot.load(cuda::memory_order_acquire);

        KeyType loaded_key = unpackKey(loaded_kv);

        bool isMatch = (loaded_key == targetKey);

        //Predictated lookup (flattened)
        bool shouldRead = (operation == OperationType::LOOKUP) && isMatch;
        uint32_t localValue = shouldRead * unpackValue(loaded_kv);

        if(operation == OperationType::LOOKUP)
        {
            //In-warp reduction to get the found value
            for(int offset = TILE_SIZE / 2; offset > 0; offset /= 2)
            {
                localValue |= tile.shfl_down(localValue, offset);
            }

            if(lane_id == 0)
            {
                outResult = localValue;
                opResolved = true;
            }
        }

        uint32_t localFreeMask = 0;
        if(lane_id == 0)
        {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomic_free_mask(*table->getFreeMask(bucket_idx, lane_id));
            localFreeMask = atomic_free_mask.load(cuda::memory_order_acquire);
        }
        
        //Predicated delete
        bool shouldDelete = (operation == OperationType::DELETE) && isMatch;
        if(shouldDelete)
        {
            //acquire and release
            uint64_t expected = loaded_kv;

            atomic_slot.compare_exchange_strong(expected, SENTINEL_KV, cuda::memory_order_acq_rel);
            atomic_free_mask.fetch_or(1 << lane_id, cuda::memory_order_release);
        }

        


        
        unsigned match_mask = tile.ballot(isMatch);


    }
}