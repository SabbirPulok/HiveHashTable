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

//load KV pair from table
__device__ __forceinline__ uint64_t loadKV(const HiveHashTable* __restrict__ table, const uint32_t bucket_idx, const uint32_t slot_idx)
{
    return table->buckets[bucket_idx].kv[slot_idx];
}

//Unlock Bucket
__device__ __forceinline__ void unlock_bucket(HiveHashTable* table, const uint32_t bucket_idx)
{
    auto atomic_lock = cuda::atomic_ref<uint16_t, cuda::thread_scope_device>(table->lock[bucket_idx]);
    atomic_lock.store(0, cuda::memory_order_release); //unlock
}

//Valid Slot Mask
template<typename TableType>
__device__ __forceinline__ uint32_t valid_slot_mask()
{
    return (TableType::SLOTS < 32) ? ((1u << TableType::SLOTS) - 1u) : 0xFFFFFFFFu;
}

//Generalized Alternate Bucket Selection
template<typename HashPolicy>
__device__ __forceinline__ uint32_t get_alternate_bucket(const uint32_t key, const uint32_t cur_bucket, size_t num_buckets)
{
    #pragma unroll
    for(int i = 0; i < HashPolicy::NumHashes; ++i)
    {
        if(HashPolicy::get_bucket(i, key, num_buckets) == cur_bucket)
        {
            return HashPolicy::get_bucket((i + 1) % HashPolicy::NumHashes, key, num_buckets);
        }
    }
    return HashPolicy::get_bucket(0, key, num_buckets); // Should not happen if cur_bucket is valid
}


//replace path
//Find all active lanes
//if use tags, all active lanes participate to match tags, unless match with key
//ballot to find which lanes matched
//elect lowest among matched lanes
//declare him winner and write to that slot
template<typename TableType>
__device__ bool try_replace(
    TableType* __restrict__ table,
    const uint32_t bucket_idx,
    const uint32_t key,
    const uint32_t value,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const size_t lane_id = tile.thread_rank();

    //Cached the KV to avoid double load
    uint64_t cached_kv = EMPTY_KV;
    bool match = false;
    bool match_a = false;

    // Use native uint4 for vectorized load
    //invalidate L1 cache
    uint4* bucket_vec = reinterpret_cast<uint4*>((table->buckets[bucket_idx].kv));
    uint4 two_kvs = bucket_vec[lane_id];

    // Check both keys for a match
    match_a = (two_kvs.x == key);
    match = match_a || (two_kvs.z == key);

    unsigned mask = tile.ballot(match);

    if (!mask) return false; // Early exit if no matches

    int winner = __ffs(mask) - 1; //first matching lane
    bool success = false;

    if (lane_id == winner)
    {
        int slot_offset = match_a ? 0 : 1;
        // Re-pack the KV pair for the atomic operation
        cached_kv = match_a ? packKV(two_kvs.x, two_kvs.y) : packKV(two_kvs.z, two_kvs.w);

        auto *slot = table->loadKV(bucket_idx, lane_id * 2 + slot_offset);

        //auto-deduce
        using T = typename std::remove_reference<decltype(*slot)>::type;
        auto atomic_slot = cuda::atomic_ref<T, cuda::thread_scope_device>(*slot);
        uint64_t desired = packKV(key, value);
        uint64_t expected = cached_kv;

        success = atomic_slot.compare_exchange_weak(
            expected,
            desired,
            cuda::memory_order_release,
            cuda::memory_order_relaxed);

    }
    //broadcast result to tile
    return tile.shfl(success, winner);
}

//Claim a free slot in the bucket based on freeMask
//Claim before write
//Immediately commit after claim the slot

#define FULL_MASK (HIVE_BUCKET_SLOTS < 32 ? ((1u << HIVE_BUCKET_SLOTS) - 1u) : 0xffffffffu)

template<typename TableType>
__device__ int claim_and_commit_slot(
    TableType* table,
    uint32_t bucket_idx,
    uint64_t new_kv,
    unsigned active_mask,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const size_t lane_id = tile.thread_rank();
    //Load mask once and broadcast to tile
    uint32_t free_mask = 0;
    if (lane_id == 0)
    {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomic_free_mask(*(table->getFreeMask(bucket_idx)));
        free_mask = atomic_free_mask.load(cuda::memory_order_acquire);
    }
    //free_mask = __shfl_sync(__activemask(), free_mask, 0);
    free_mask = tile.shfl(free_mask, 0);

    //Mask out unused slots
    //const uint32_t full_mask = (HIVE_BUCKET_SLOTS < 32) ? ((1u << HIVE_BUCKET_SLOTS) - 1u) : 0xffffffffu;
    free_mask &= valid_slot_mask<TableType>();
    if (!free_mask) return -1; //no free slot

    //Each thread i owns bits [2i, 2i+1]
    uint32_t two_bits_mask = 0x3u << (lane_id * 2); //0011 shifted to position

    //All threads in tile participate to find a free slot
    bool can_claim = (free_mask & two_bits_mask) != 0;
    unsigned claim_ballot = tile.ballot(can_claim);
    
    if (!claim_ballot) return -1; //no free slot

    
    //Winner election
    int winner_lane = __ffs(claim_ballot) - 1;
    int claimed = -1;

    if(lane_id == winner_lane)
    {
        uint32_t free_bits = free_mask & two_bits_mask;
        int target_slot_in_pair = __ffs(free_bits) - 1;
        int target_slot = lane_id * 2 + target_slot_in_pair;
        uint32_t slot_bit = 1u << target_slot;

        //Claim the slot atomically, atomic AND
        auto *mask = table->getFreeMask(bucket_idx);
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomic_free_mask(*mask);

        //Single atomic RMW operation instead of load + CAS
        uint32_t old_mask = atomic_free_mask.fetch_and(
            ~slot_bit, cuda::memory_order_release);

        //Check if we actually claimed the slot
        if (old_mask & slot_bit)
        {
            //Successfully claimed! Write KV immediately
            auto* slot = table->loadKV(bucket_idx, target_slot);
            using T = typename std::remove_reference<decltype(*slot)>::type;
            auto atomic_kv = cuda::atomic_ref<T, cuda::thread_scope_device>(*slot);
            atomic_kv.store(new_kv, cuda::memory_order_release);
            
            claimed = target_slot;
        }
        else
        {
            //Someone else got it, restore the bit
            //atomic_free_mask.fetch_or(slot_bit, cuda::memory_order_relaxed);
            claimed = -1;
        }
    }

    //broadcast result to tile
    return tile.shfl(claimed, winner_lane);
}

//bounded eviction path : slow path
//kicks (evicts) a victim from a full bucket, places the new kv in the vacated slot
//then re-inserts the evicted kv into its alternate bucket, repeat up to max_evictions times
template<typename TableType, typename HashPolicy>
__device__ bool cuckoo_evict_and_insert(
    cg::thread_block_tile<TILE_SIZE> tile,
    TableType* table,
    uint32_t bucket_idx,
    uint64_t kv
)
{
    const unsigned active = tile.ballot(true);
    const int lane_id = tile.thread_rank();
    const uint32_t VALID = valid_slot_mask<TableType>();

    uint64_t cur_kv = kv;
    uint32_t cur_bucket = bucket_idx;

    for(size_t kick = 0; kick < table->max_evictions; ++kick)
    {
        //If any thread freed a slot
        if(claim_and_commit_slot (table, cur_bucket, cur_kv, active, tile) != -1)
        {
            return true; //successfully inserted
        }

        int outcome = -1;
        uint64_t victim_kv = 0;

        //Lock the bucket for eviction
        //Picks a victim
        if(lane_id == 0)
        {
            table->lockBucket(cur_bucket);
            //lock_bucket(table, cur_bucket);

            //if anybody freed a slot while we were locking, just insert there
            auto atomic_free_mask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*table->getFreeMask(cur_bucket));
            uint32_t free_mask = atomic_free_mask.load(cuda::memory_order_relaxed) & VALID;
            if(free_mask != 0)
            {
                int free_slot = __ffs(free_mask) - 1;
                uint32_t free_bit = 1u << free_slot; //bit corresponding to the free slot

                //under the lock, update the slot without CAS
                uint32_t old_free_mask = atomic_free_mask.load(cuda::memory_order_relaxed);
                const uint32_t new_free_mask = old_free_mask & ~free_bit; //claim

                atomic_free_mask.store(new_free_mask, cuda::memory_order_release);

                //Publish KV
                auto* slot = table->loadKV(cur_bucket, free_slot);
                using T = typename std::remove_reference<decltype(*slot)>::type;
                auto atomic_kv = cuda::atomic_ref<T, cuda::thread_scope_device>(*slot);
                atomic_kv.store(cur_kv, cuda::memory_order_release);

                table->unlockBucket(cur_bucket);
                //unlock_bucket(table, cur_bucket);

                //done placed without eviction
                outcome = -2; //indicate no eviction
            }
            else
            {
                uint32_t occupied = ~atomic_free_mask.load(cuda::memory_order_relaxed) & VALID;
                const int victim_slot = __ffs(occupied) - 1; //pick first occupied slot as victim

                //Place my KV in the victim slot
                auto* slot = table->loadKV(cur_bucket, victim_slot);
                using T = typename std::remove_reference<decltype(*slot)>::type;
                auto atomic_kv = cuda::atomic_ref<T, cuda::thread_scope_device>(*slot);
                victim_kv = atomic_kv.load(cuda::memory_order_acquire);
                atomic_kv.store(kv, cuda::memory_order_release);

                table->unlockBucket(cur_bucket);
                //unlock_bucket(table, cur_bucket);
                outcome = victim_slot; //indicate eviction
            }
        }

        //broadcast outcome to tile
        outcome = tile.shfl(outcome, 0);
        victim_kv =  tile.shfl(victim_kv, 0);

        if(outcome == -2) return true; //placed without eviction
        //if(outcome == -1), failed to lock bucket, retry

        if(outcome >= 0)
        {
            //Successfully evicted a victim
            cur_kv = victim_kv;
            uint32_t cur_key = unpackKey(cur_kv);
            cur_bucket = get_alternate_bucket<HashPolicy>(cur_key, cur_bucket, table->num_buckets);
        }
    }

    return false;
}

//Push on stash (overflow buffer)
template<typename KeyType, typename ValueType>
__device__ __forceinline__ bool push_to_stash(
    HiveOverflowStash<KeyType, ValueType>* stash_table,
    uint64_t kv
)
{
    if(!stash_table->enabled || stash_table->capacity == 0) return false;

    uint32_t key = unpackKey(kv);
    uint32_t value = unpackValue(kv);

    //stash table head and tail atomics
    while(true)
    {
        uint32_t head = stash_table->head.load(cuda::memory_order_acquire);

        uint32_t tail = stash_table->tail.load(cuda::memory_order_acquire);

        if((tail - head) >= stash_table->capacity)
        {
            //stash full
            return false;
        }
        tail = stash_table->tail.fetch_add(1, cuda::memory_order_acq_rel); //reserve a slot
        //successfully reserved a slot at tail
        uint32_t index = tail % stash_table->capacity;
        stash_table->keys[index] = key;
        stash_table->values[index] = value;
        return true;
    }
}

__device__ __forceinline__ double min_start_time(
    double* start_time_array,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    double cur_time = start_time_array[tile.thread_rank()];

    for(int offset = tile.size()/2; offset > 0; offset /= 2)
    {
        // every lane compares its current candidate with the candidate of the lane offset away
        double other = tile.shfl_down(cur_time, offset);
        cur_time = min(cur_time, other);
    }
    double min_time = tile.shfl(cur_time, 0);

    return min_time;
}


//main insert function (tile-cooperative)
template<typename TableType, typename KeyType, typename ValueType, typename HashPolicy>
__device__ __forceinline__ bool hive_insert_one_coop(
    TableType* __restrict__ table,
    HiveOverflowStash<KeyType, ValueType>* __restrict__ stash_table,
    uint32_t key,
    uint32_t value,
    cg::thread_block_tile<TILE_SIZE> tile
    #if BREAKDOWN_INSERT
    , InsertBreakdown& breakdown_per_tile
    #endif
)
{
    const unsigned int active = tile.ballot(true);

    if(key == SENTINEL) return false; //invalid key

    const uint8_t tag = getTag(key);
    const uint64_t kv = packKV(key, value);
    
    int insertion_retries = 0;
    const int MAX_INSERT_RETRIES = 100;

    retry_insertion:

    if(insertion_retries++ > MAX_INSERT_RETRIES) return false;

    #if BREAKDOWN_INSERT
    double start_time[TILE_SIZE] = {0.0};
    double end_time = 0;
    start_time[tile.thread_rank()] = clock64();
    #endif

    //Stage A: replace in A or B if key already exists
    #pragma unroll
    for(int i = 0; i < HashPolicy::NumHashes; ++i)
    {
        uint32_t b_idx = HashPolicy::get_bucket(i, key, table->num_buckets);
        if(try_replace(table, b_idx, key, value, tile))
        {
            #if BREAKDOWN_INSERT
            end_time = clock64();
            breakdown_per_tile.stageA = end_time - min_start_time(start_time, tile);
            start_time[tile.thread_rank()] = clock64();
            #endif
            return true;
        }
    }

    //Stage B: free slot claim A then B
    #pragma unroll
    for(int i = 0; i < HashPolicy::NumHashes; ++i)
    {
        uint32_t b_idx = HashPolicy::get_bucket(i, key, table->num_buckets);
        int claimed_slot = claim_and_commit_slot(table, b_idx, kv, active, tile);
        if(claimed_slot != -1) 
        {
            // Double-check to ensure no duplicates were inserted concurrently
            bool duplicate_found = false;
            const int lane_id = tile.thread_rank();

            #pragma unroll
            for(int j = 0; j < HashPolicy::NumHashes; ++j)
            {
                uint32_t check_b_idx = HashPolicy::get_bucket(j, key, table->num_buckets);
                
                // Cooperative search in check_b_idx bypassing L1 cache
                uint4* bucket_vec = reinterpret_cast<uint4*>(table->buckets[check_b_idx].kv);
                uint4 two_kvs = load_cg_safe(&bucket_vec[lane_id]);

                bool m0 = (two_kvs.x == key);
                bool m1 = (two_kvs.z == key);
                
                // Filter out the slot we just claimed in the current bucket
                if(check_b_idx == b_idx)
                {
                    m0 = m0 && (claimed_slot != (lane_id * 2));
                    m1 = m1 && (claimed_slot != (lane_id * 2 + 1));
                }

                if(tile.ballot(m0 || m1))
                {
                    duplicate_found = true;
                }
            }

            if(duplicate_found)
            {
                // Rollback: clear the slot and release it
                int owner_lane = claimed_slot / 2;
                if(lane_id == owner_lane)
                {
                    auto* slot = table->loadKV(b_idx, claimed_slot);
                    using T = typename std::remove_reference<decltype(*slot)>::type;
                    cuda::atomic_ref<T, cuda::thread_scope_device> atomic_kv(*slot);
                    atomic_kv.store(EMPTY_KV, cuda::memory_order_release);
                }

                if(lane_id == 0)
                {
                    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomic_free_mask(*(table->getFreeMask(b_idx)));
                    atomic_free_mask.fetch_or(1u << claimed_slot, cuda::memory_order_release);
                }
                goto retry_insertion;
            }

            #if BREAKDOWN_INSERT
            end_time = clock64();
            breakdown_per_tile.stageB = end_time - min_start_time(start_time, tile);
            start_time[tile.thread_rank()] = clock64();
            #endif
            return true;
        }
    }

    //stage C: cuckoo eviction
    #pragma unroll
    for(int i = 0; i < HashPolicy::NumHashes; ++i)
    {
        uint32_t b_idx = HashPolicy::get_bucket(i, key, table->num_buckets);
        if(cuckoo_evict_and_insert<TableType, HashPolicy>(tile, table, b_idx, kv)) 
        {
            #if BREAKDOWN_INSERT
            end_time = clock64();
            breakdown_per_tile.stageC = end_time - min_start_time(start_time, tile);
            start_time[tile.thread_rank()] = clock64();
            #endif
            return true;
        }
    }

    //stage D: Push Stash
    push_to_stash(stash_table, kv);
    #if BREAKDOWN_INSERT
    end_time = clock64();
    breakdown_per_tile.stageD = end_time - min_start_time(start_time, tile);
    #endif

    return false;
}