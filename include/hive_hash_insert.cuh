#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include "hash_table_struct.h"
#include "hive_hash_table_struct.cuh"
#include "hive_stash_table.cuh"
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

//Lock Bucket
__device__ __forceinline__ void lock_bucket(HiveHashTable* table, const uint32_t bucket_idx)
{
    auto atomic_lock = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(table->lock[bucket_idx]);

    uint32_t expected = 0;

    while(!atomic_lock.compare_exchange_weak(
        expected,
        1,
        cuda::memory_order_acquire,
        cuda::memory_order_relaxed
    ))
    {
        expected = 0;
    }
}

//Unlock Bucket
__device__ __forceinline__ void unlock_bucket(HiveHashTable* table, const uint32_t bucket_idx)
{
    auto atomic_lock = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(table->lock[bucket_idx]);
    atomic_lock.store(0, cuda::memory_order_release);
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
    return HashPolicy::get_bucket(0, key, num_buckets);
}

// Stage A only: scan one bucket for an existing key and update if found.
// Does NOT insert — no CAS on empty slots, no freeMask update.
// Returns true if the key was found (and updated), false if absent.
//
// Used in Phase 1 of hive_insert_one_coop to scan ALL candidate buckets
// before any insert attempt. This catches:
//   (a) key already present in its primary bucket (normal update path)
//   (b) key previously cuckoo-evicted to its alternate bucket
// Both cases are resolved here without ever reaching Stage B, eliminating
// the re-insertion-after-eviction cross-bucket duplication risk.
template<typename TableType>
__device__ __forceinline__ bool scan_and_update(
    TableType* table,
    uint32_t bucket_idx,
    uint32_t key,
    uint32_t value,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const int lane_id = tile.thread_rank();
    const uint64_t kv_to_insert = packKV(key, value);
    ulonglong2* bucket_vec = reinterpret_cast<ulonglong2*>(table->buckets[bucket_idx].kv);

    ulonglong2 two_kvs = load_two_kvs(bucket_vec, lane_id);

    bool match_x = (unpackKey(two_kvs.x) == key);
    bool match_y = (unpackKey(two_kvs.y) == key);
    unsigned match_mask = tile.ballot(match_x || match_y);

    if (match_mask != 0)
    {
        int winner_lane = __ffs(match_mask) - 1;
        if (lane_id == winner_lane)
        {
            int slot_offset = match_x ? 0 : 1;
            uint32_t target_slot = (lane_id * 2) + slot_offset;
            auto* slot_ptr = &table->buckets[bucket_idx].kv[target_slot];
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                *reinterpret_cast<uint64_t*>(slot_ptr))
                .store(kv_to_insert, cuda::memory_order_release);
        }
        return true;
    }
    return false;
}

template<typename TableType>
__device__ __forceinline__ bool scan_and_update_insert_only(
    TableType* table,
    uint32_t bucket_idx,
    uint32_t key,
    uint32_t value,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const int lane_id = tile.thread_rank();
    const uint64_t kv_to_insert = packKV(key, value);
    ulonglong2* bucket_vec = reinterpret_cast<ulonglong2*>(table->buckets[bucket_idx].kv);

    ulonglong2 two_kvs = load_two_kvs(bucket_vec, lane_id);

    bool match_x = (unpackKey(two_kvs.x) == key);
    bool match_y = (unpackKey(two_kvs.y) == key);
    unsigned match_mask = tile.ballot(match_x || match_y);

    if (match_mask != 0)
    {
        int winner_lane = __ffs(match_mask) - 1;
        if (lane_id == winner_lane)
        {
            int slot_offset = match_x ? 0 : 1;
            uint32_t target_slot = (lane_id * 2) + slot_offset;
            auto* slot_ptr = &table->buckets[bucket_idx].kv[target_slot];
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                *reinterpret_cast<uint64_t*>(slot_ptr))
                .store(kv_to_insert, cuda::memory_order_release);
        }
        return true;
    }
    return false;
}

/**
 * scan_stash_and_update
 * 
 * Scans one stash bucket for an existing key and updates its value in-place.
 * Returns true if the key was found and updated, false if absent.
 * 
 * This is critical for preventing "Stash Shadowing" duplicates: if a key
 * was previously pushed to the stash, any subsequent update (upsert) for
 * that key must find it there rather than blindly inserting a second copy 
 * into a primary bucket that might have freed up.
 */
template<typename TableType>
__device__ __forceinline__ bool scan_stash_and_update(
    TableType* __restrict__ table,
    HiveOverflowStashBucket<kv_type>* __restrict__ stash_table,
    uint32_t bucket_idx,
    uint32_t key,
    uint32_t value,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    // Metadata Gating: Fast local VRAM check. If bit 31 is 0, stash is empty.
    uint32_t primary_mask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
        *table->getFreeMask(bucket_idx)).load(cuda::memory_order_relaxed);
    if ((primary_mask & 0x80000000u) == 0) return false;

    const int lane_id = tile.thread_rank();
    const uint64_t kv_to_insert = packKV(key, value);

    // check if this stash bucket has any entries
    auto count_ref = cuda::atomic_ref<int, cuda::thread_scope_device>(
        const_cast<int&>(stash_table[bucket_idx].count)
    );

    // acquire-load ensure we see the latest count
    int count = count_ref.load(cuda::memory_order_acquire);
    if (count <= 0) return false;

    // parallel vectorized load of the stash bucket
    ulonglong2* bucket_vec = reinterpret_cast<ulonglong2*>(stash_table[bucket_idx].kv);
    ulonglong2 two_kvs = load_two_kvs(bucket_vec, lane_id);

    bool match_x = (unpackKey(two_kvs.x) == key);
    bool match_y = (unpackKey(two_kvs.y) == key);
    unsigned match_mask = tile.ballot(match_x || match_y);

    if (match_mask != 0)
    {
        int winner_lane = __ffs(match_mask) - 1;
        if (lane_id == winner_lane)
        {
            int slot_offset = match_x ? 0 : 1;
            uint32_t target_slot = (lane_id * 2) + slot_offset;
            auto* slot_ptr = &stash_table[bucket_idx].kv[target_slot];
            
            // publish updated value with release semantics
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                *reinterpret_cast<uint64_t*>(slot_ptr))
                .store(kv_to_insert, cuda::memory_order_release);
        }
        return true;
    }
    return false;
}

// Stage B: find an empty slot in bucket_idx and CAS the KV pair into it.
//
// Includes a Stage A re-check at the top of every retry iteration.
// This is critical for the canonical insert design: if a concurrent tile
// committed the same key into the canonical bucket between the caller's
// Phase 1 scan and this Stage B call, the re-check detects it and converts
// the operation to an update instead of inserting a duplicate.
//
// Because the canonical bucket is always H0(key) — deterministic — any
// concurrent tile inserting the same key targets this same bucket, so
// the Stage A re-check here is guaranteed to find it.
//
// Returns true on success (inserted or converted to update).
// Returns false if the bucket is full (caller proceeds to cuckoo eviction).
template<typename TableType, typename KeyType, typename ValueType>
__device__ __forceinline__ bool insert_lockfree(
    TableType* table,
    uint32_t bucket_idx,
    KeyType key,
    ValueType value,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const int lane_id = tile.thread_rank();
    const uint64_t kv_to_insert = packKV(key, value);
    ulonglong2* bucket_vec = reinterpret_cast<ulonglong2*>(table->buckets[bucket_idx].kv);

    while(true)
    {
        // Fresh load on every iteration
        ulonglong2 two_kvs = load_two_kvs(bucket_vec, lane_id);

        // Stage A re-check: detects concurrent same-key commits since Phase 1
        bool match_x = (unpackKey(two_kvs.x) == key);
        bool match_y = (unpackKey(two_kvs.y) == key);
        unsigned match_mask = tile.ballot(match_x || match_y);
        if (match_mask != 0)
        {
            int winner_lane = __ffs(match_mask) - 1;
            if (lane_id == winner_lane)
            {
                int slot_offset = match_x ? 0 : 1;
                uint32_t target_slot = (lane_id * 2) + slot_offset;
                auto* slot_ptr = &table->buckets[bucket_idx].kv[target_slot];
                cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                    *reinterpret_cast<uint64_t*>(slot_ptr))
                    .store(kv_to_insert, cuda::memory_order_release);
            }
            return true;
        }

        // Stage B: claim an empty slot via CAS
        bool empty_x = (two_kvs.x == EMPTY_KV);
        bool empty_y = (two_kvs.y == EMPTY_KV);
        unsigned empty_mask = tile.ballot(empty_x || empty_y);
        if (empty_mask == 0)
            return false; // Bucket full — caller will invoke cuckoo eviction

        int winner_lane = __ffs(empty_mask) - 1;
        int action_state = 0; // 1=success, -1=retry

        if (lane_id == winner_lane)
        {
            int slot_offset = empty_x ? 0 : 1;
            int target_slot = (lane_id * 2) + slot_offset;

            auto* slot_ptr = &table->buckets[bucket_idx].kv[target_slot];
            auto atomic_kv = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                *reinterpret_cast<uint64_t*>(slot_ptr));

            uint64_t expected = EMPTY_KV;
            if (atomic_kv.compare_exchange_weak(
                    expected, kv_to_insert,
                    cuda::memory_order_acq_rel,
                    cuda::memory_order_acquire))
            {
                auto atomic_free_mask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
                    *table->getFreeMask(bucket_idx));
                atomic_free_mask.fetch_and(~(1u << target_slot), cuda::memory_order_release);
                action_state = 1;
            }
            else
            {
                action_state = -1; // Lost CAS race — reload and retry
            }
        }

        action_state = tile.shfl(action_state, winner_lane);
        if (action_state == 1) return true;
        // action_state == -1: loop back for fresh load
    }
}

// insert_freemask_first — Stage B for insert-only workloads.
//
// Instead of reading the full 64-byte bucket to find a free slot, this
// function reads only the 32-bit freeMask and claims a slot via a 32-bit CAS.
// The KV is then written with a plain release store (no second CAS needed —
// the freeMask CAS already uniquely owns the slot).
//
// Cost comparison (common case, no contention):
//   insert_only_lockfree:   4×LDG.128 (64B) + CAS-64 + atomicAnd-32
//   insert_freemask_first:  LDG-32 (4B)     + CAS-32 + ST-64
//
// The 32-bit CAS is cheaper than a 64-bit CAS on current NVIDIA architectures.
// On retry (lost race), only 4 bytes reload instead of 64 bytes.
//
// Lane 0 drives the CAS loop; all lanes participate in the while + shfl so
// the tile stays convergent and can collectively enter cuckoo if the bucket
// is full.
template<typename TableType, typename KeyType, typename ValueType>
__device__ __forceinline__ bool insert_freemask_first(
    TableType* table,
    uint32_t bucket_idx,
    KeyType key,
    ValueType value,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const int lane_id = tile.thread_rank();
    const uint64_t kv_to_insert = packKV(key, value);
    const uint32_t VALID = valid_slot_mask<TableType>();

    auto mask_ref = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
        *table->getFreeMask(bucket_idx));

    while (true)
    {
        // 0 = CAS lost (retry),  1 = inserted,  -1 = bucket full
        int action_state = 0;

        if (lane_id == 0)
        {
            uint32_t mask = mask_ref.load(cuda::memory_order_acquire) & VALID;

            if (mask == 0)
            {
                action_state = -1; // no free slots — caller enters cuckoo
            }
            else
            {
                int slot = __ffs(mask) - 1;
                uint32_t new_mask = mask & ~(1u << slot);

                if (mask_ref.compare_exchange_weak(
                        mask, new_mask,
                        cuda::memory_order_acq_rel,
                        cuda::memory_order_acquire))
                {
                    // Slot uniquely owned — plain release store (no CAS needed)
                    cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                        *table->loadKV(bucket_idx, slot))
                        .store(kv_to_insert, cuda::memory_order_release);
                    action_state = 1;
                }
                // compare_exchange_weak updates mask to current value on failure;
                // action_state stays 0 → all lanes loop back for a fresh load
            }
        }

        // All lanes converge here — shfl broadcasts lane 0's outcome
        action_state = tile.shfl(action_state, 0);
        if (action_state ==  1) return true;
        if (action_state == -1) return false;
        // action_state == 0: lost the CAS race — retry (only 4B reload)
    }
}

// insert_only_lockfree — original bucket-scan Stage B (kept for reference).
// Superseded by insert_freemask_first for the insert-only kernel.
template<typename TableType, typename KeyType, typename ValueType>
__device__ __forceinline__ bool insert_only_lockfree(
    TableType* table,
    uint32_t bucket_idx,
    KeyType key,
    ValueType value,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    const int lane_id = tile.thread_rank();
    const uint64_t kv_to_insert = packKV(key, value);
    ulonglong2* bucket_vec = reinterpret_cast<ulonglong2*>(table->buckets[bucket_idx].kv);

    while(true)
    {
        ulonglong2 two_kvs = load_two_kvs(bucket_vec, lane_id);

        bool empty_x = (two_kvs.x == EMPTY_KV);
        bool empty_y = (two_kvs.y == EMPTY_KV);

        unsigned empty_mask = tile.ballot(empty_x || empty_y);
        if (empty_mask == 0)
            return false; // Bucket full — caller invokes cuckoo eviction

        int winner_lane = __ffs(empty_mask) - 1;
        int action_state = 0;

        if (lane_id == winner_lane)
        {
            int slot_offset = empty_x ? 0 : 1;
            int target_slot = (lane_id * 2) + slot_offset;

            auto* slot_ptr = &table->buckets[bucket_idx].kv[target_slot];
            auto atomic_kv = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                *reinterpret_cast<uint64_t*>(slot_ptr));

            uint64_t expected = EMPTY_KV;
            if (atomic_kv.compare_exchange_weak(
                    expected, kv_to_insert,
                    cuda::memory_order_acq_rel,
                    cuda::memory_order_acquire))
            {
                auto atomic_free_mask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
                    *table->getFreeMask(bucket_idx));
                atomic_free_mask.fetch_and(~(1u << target_slot), cuda::memory_order_release);
                action_state = 1;
            }
            else
            {
                action_state = -1; // Lost CAS race — reload and retry
            }
        }

        // All lanes must participate in shfl — must be outside winner_lane guard
        action_state = tile.shfl(action_state, winner_lane);
        if (action_state == 1) return true;
        // action_state == -1: loop back for fresh load
    }
}

template<typename TableType>
__device__ __forceinline__ void lock_ordered(
    TableType* table,
    uint32_t bucket_a,  // Already held
    uint32_t bucket_b
)
{
    if(bucket_a == bucket_b) return;

    if(bucket_b < bucket_a) {
        table->unlockBucket(bucket_a);
        table->lockBucket(bucket_b);
        table->lockBucket(bucket_a);
    } else {
        table->lockBucket(bucket_b);
    }
}

template<typename TableType>
__device__ __forceinline__ void unlock_ordered(
    TableType* table,
    uint32_t bucket_a,
    uint32_t bucket_b
)
{
    if(bucket_a == bucket_b)
    {
        table->unlockBucket(bucket_a);
        return;
    }

    uint32_t low  = bucket_a < bucket_b ? bucket_a : bucket_b;
    uint32_t high = bucket_a < bucket_b ? bucket_b : bucket_a;

    table->unlockBucket(high);
    table->unlockBucket(low);
}

//Push on stash (overflow buffer)
template<typename TableType>
__device__ __forceinline__ bool push_to_stash(
    TableType* table,
    HiveOverflowStashBucket<kv_type>* stash_table,
    uint32_t bucket_idx,
    uint64_t kv
)
{
    auto count_ref = cuda::atomic_ref<int, cuda::thread_scope_device>(stash_table[bucket_idx].count);

    int slot = count_ref.fetch_add(1, cuda::memory_order_acq_rel);

    if(slot >= (int)HiveOverflowStashBucket<kv_type>::SLOTS)
    {
        count_ref.fetch_sub(1, cuda::memory_order_relaxed);
        return false;
    }

    auto kv_ref = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
        reinterpret_cast<uint64_t&>(stash_table[bucket_idx].kv[slot])
    );
    kv_ref.store(kv, cuda::memory_order_release);

    // Flag the primary bucket's metadata: Bit 31 indicates stash usage
    auto mask_ref = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*table->getFreeMask(bucket_idx));
    mask_ref.fetch_or(0x80000000u, cuda::memory_order_release);

    return true;
}


// Helper function: Double-Checked Locking for Duplicates
// Must be called WHILE holding the lock for bucket_idx.
template<typename TableType>
__device__ __forceinline__ bool check_duplicate_under_lock(
    TableType* table, 
    uint32_t bucket_idx, 
    uint64_t incoming_kv)
{
    const uint32_t incoming_key = unpackKey(incoming_kv);
    for (int i = 0; i < TableType::SLOTS; ++i) {
        uint64_t existing_kv = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
            *table->loadKV(bucket_idx, i)).load(cuda::memory_order_relaxed);
        
        if (existing_kv != EMPTY_KV && unpackKey(existing_kv) == incoming_key) {
            // It's already here! Just update the value (Upsert semantics)
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                *table->loadKV(bucket_idx, i)).store(incoming_kv, cuda::memory_order_release);
            return true;
        }
    }
    return false;
}

// cuckoo_evict_and_insert — multi-victim single-source eviction
//
// Design: each kick tries a DIFFERENT victim slot in the same source bucket.
// Every victim has a different alt_bucket, so across max_evictions kicks we
// probe up to SLOTS distinct destinations before giving up.
//
// Linearizability guarantee:
//   A victim is written to alt_bucket BEFORE its src slot is overwritten,
//   so it is always visible in at least one location.  No key-in-transit gap.
//   If all alt_buckets are full the INCOMING key (kv) is pushed to stash —
//   existing keys never move and are never temporarily invisible.
//
// No return inside if(lane_id==0): all lanes always reach tile.shfl together.
template<typename TableType, typename HashPolicy>
__device__ int cuckoo_evict_and_insert(
    cg::thread_block_tile<TILE_SIZE> tile,
    TableType* table,
    HiveOverflowStashBucket<kv_type>* stash_table,
    uint32_t bucket_idx,
    uint64_t kv
)
{
    const int lane_id = tile.thread_rank();
    const uint32_t VALID = valid_slot_mask<TableType>();
    int status = 0;

    if (lane_id == 0)
    {
        auto mask_ref = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
            *table->getFreeMask(bucket_idx));

        for (int kick = 0; kick < (int)table->max_evictions && status == 0; ++kick)
        {
            table->lockBucket(bucket_idx);

            // Double-Checked Locking: Did another tile insert our key while we were spinning?
            if (check_duplicate_under_lock(table, bucket_idx, kv))
            {
                table->unlockBucket(bucket_idx);
                status = 1;
                break;
            }

            uint32_t free_mask = mask_ref.load(cuda::memory_order_relaxed) & VALID;

            if (free_mask != 0)
            {
                // ── Fast path: space appeared (e.g. freed by concurrent delete) ──
                //
                // freeMask can be stale: lock-free delete sets freeMask AFTER
                // its CAS, but a concurrent insert_lockfree can race in between
                // and reoccupy the slot before delete updates freeMask.  We hold
                // the bucket lock here, so we re-read the actual KV to confirm
                // the slot is genuinely empty before writing.
                int slot = __ffs(free_mask) - 1;
                auto slot_ref = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                    *table->loadKV(bucket_idx, slot));
                uint64_t actual_kv = slot_ref.load(cuda::memory_order_acquire);
                if (actual_kv != EMPTY_KV)
                {
                    // freeMask was stale — slot got reoccupied. Repair the
                    // mask so future evictions don't keep re-entering this path.
                    mask_ref.fetch_and(~(1u << slot), cuda::memory_order_release);
                    table->unlockBucket(bucket_idx);
                    continue;
                }
                mask_ref.fetch_and(~(1u << slot), cuda::memory_order_release);
                slot_ref.store(kv, cuda::memory_order_release);
                table->unlockBucket(bucket_idx);
                status = 1;
                break;
            }

            // ── Eviction path: pick victim[kick % SLOTS] ──────────────────
            // Rotating the victim index across kicks spreads probes over
            // different alt_buckets — statistically one will have free space.
            int victim_slot = (int)(kick % TableType::SLOTS);

            auto src_ref = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                *table->loadKV(bucket_idx, victim_slot));
            uint64_t victim_kv = src_ref.load(cuda::memory_order_acquire);

            // Skip slots that were concurrently deleted (freeMask stale)
            if (victim_kv == EMPTY_KV)
            {
                table->unlockBucket(bucket_idx);
                continue;
            }

            uint32_t alt_bucket = get_alternate_bucket<HashPolicy>(
                unpackKey(victim_kv), bucket_idx, table->num_buckets);

            // Acquire both locks in address order — deadlock-free
            lock_ordered(table, bucket_idx, alt_bucket);

            // Re-read victim after both locks held: lock_ordered may have
            // released and re-acquired bucket_idx when alt < bucket_idx.
            victim_kv = src_ref.load(cuda::memory_order_acquire);

            if (victim_kv != EMPTY_KV && alt_bucket != bucket_idx)
            {
                auto alt_mask_ref = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
                    *table->getFreeMask(alt_bucket));
                uint32_t alt_free = alt_mask_ref.load(cuda::memory_order_relaxed) & VALID;

                if (alt_free != 0)
                {
                    int dst_slot = __ffs(alt_free) - 1;

                    // ① Claim dst in alt freeMask FIRST
                    alt_mask_ref.fetch_and(~(1u << dst_slot), cuda::memory_order_release);

                    // ② Write victim to dst — now visible in BOTH src and dst
                    cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                        *table->loadKV(alt_bucket, dst_slot))
                        .store(victim_kv, cuda::memory_order_release);

                    // ③ CAS src: victim_kv → kv
                    //   Detects if a concurrent lock-free delete cleared src
                    //   between ② and ③ (delete does not acquire bucket locks).
                    uint64_t expected = victim_kv;
                    if (src_ref.compare_exchange_strong(
                            expected, kv,
                            cuda::memory_order_acq_rel,
                            cuda::memory_order_acquire))
                    {
                        // Normal success: victim in alt, kv in src. ✓
                        status = 1;
                    }
                    else
                    {
                        // Concurrent delete freed src between ② and ③.
                        // Undo the dst write — we must not resurrect a deleted key.
                        // On the next iteration the fast-path will see the freed slot.
                        cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                            *table->loadKV(alt_bucket, dst_slot))
                            .store(EMPTY_KV, cuda::memory_order_release);
                        alt_mask_ref.fetch_or(1u << dst_slot, cuda::memory_order_release);

                        unlock_ordered(table, bucket_idx, alt_bucket);
                        kick--; // Retry this kick — we did not actually evict
                        continue;
                    }
                }
                // alt_free == 0: this victim's alt is full.
                // Unlock and let the loop try the next victim slot.
            }

            unlock_ordered(table, bucket_idx, alt_bucket);
        } // end for kick

        if (status == 0)
        {
            // All victim choices exhausted (every alt was full).
            // Stash the INCOMING key — existing keys never move, no gap.
            if(push_to_stash(table, stash_table, bucket_idx, kv))
                status = 2;
        }
    } // end if(lane_id == 0)

    // Single collective sync — lane 0 never returns before this point.
    return tile.shfl(status, 0);
}

// cuckoo_evict_and_insert_only — eviction path for insert-only workloads.
//
// Designed to work safely alongside insert_freemask_first, which claims slots
// via a lock-free 32-bit CAS on freeMask without holding the bucket lock.
// This creates two correctness requirements beyond the simpler prior version:
//
//   A. Fast path removed: insert_freemask_first fails only when freeMask==0.
//      In insert-only mode bits go 1→0 only (no deletes), so once
//      insert_freemask_first returns false the canonical bucket's freeMask
//      stays 0 — the fast path (checking if a slot freed up) is dead code.
//
//   B. EMPTY_KV victim guard: insert_freemask_first claims a freeMask bit
//      BEFORE writing the KV. In the narrow window between its CAS and its
//      store, a victim slot may read as EMPTY_KV even though its freeMask
//      bit is already 0 (occupied). We skip such slots and retry.
//
//   C. Alt-slot claiming via CAS (not fetch_and): insert_freemask_first on
//      alt_bucket can race with cuckoo's attempt to claim a dst_slot there.
//      Using fetch_and would not detect the race — both would "claim" the
//      same bit and cuckoo's subsequent KV store would overwrite
//      insert_freemask_first's write. Using CAS ensures exactly one winner
//      per slot; the loser retries with the next free bit.
//
//   D. Step ③ plain store (retained from before): both locks are held and
//      no delete can clear victim_kv, so the store always succeeds.
//
//   E. Conditional victim re-read (retained): only re-read victim_kv after
//      lock_ordered when alt_bucket < bucket_idx (the only case where
//      lock_ordered releases and re-acquires bucket_idx).
template<typename TableType, typename HashPolicy>
__device__ int cuckoo_evict_and_insert_only(
    cg::thread_block_tile<TILE_SIZE> tile,
    TableType* __restrict__ table,
    HiveOverflowStashBucket<kv_type>* __restrict__ stash_table,
    uint32_t bucket_idx,
    uint64_t kv
    #if BREAKDOWN_INSERT
    , InsertBreakdown& breakdown_per_tile
    #endif
)
{
    const uint32_t lane_id = tile.thread_rank();
    const uint32_t VALID = valid_slot_mask<TableType>();

    int status = 0;

    #if BREAKDOWN_INSERT
    double cuckoo_start = 0, cuckoo_end = 0;
    double stash_start = 0, stash_end = 0;
    #endif

    if(lane_id == 0)
    {
        #if BREAKDOWN_INSERT
        cuckoo_start = clock64();
        #endif

        for(int kick = 0; kick < (int)table->max_evictions && status == 0; ++kick)
        {
            table->lockBucket(bucket_idx);

            // (A) No fast path — canonical freeMask is 0 by the time we get here.

            // Eviction path: pick victim[kick % SLOTS]
            int victim_slot = (int)(kick % TableType::SLOTS);
            auto src_ref = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                *table->loadKV(bucket_idx, victim_slot));
            uint64_t victim_kv = src_ref.load(cuda::memory_order_acquire);

            // (B) Skip slots in the insert_freemask_first write window:
            //     freeMask bit cleared but KV not yet stored → still EMPTY_KV.
            if (victim_kv == EMPTY_KV)
            {
                table->unlockBucket(bucket_idx);
                continue; // try next victim slot
            }

            uint32_t alt_bucket = get_alternate_bucket<HashPolicy>(
                unpackKey(victim_kv), bucket_idx, table->num_buckets);

            // (E) Acquire both locks in address order — deadlock-free.
            lock_ordered(table, bucket_idx, alt_bucket);

            // Re-read victim only when lock_ordered may have released bucket_idx.
            if (alt_bucket < bucket_idx)
                victim_kv = src_ref.load(cuda::memory_order_acquire);

            if (alt_bucket != bucket_idx)
            {
                auto alt_mask_ref = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(
                    *table->getFreeMask(alt_bucket));

                // (C) Claim a free slot in alt_bucket via CAS on freeMask.
                //     insert_freemask_first on alt_bucket races here without
                //     holding the lock, so fetch_and is unsafe — CAS ensures
                //     exactly one owner per slot.
                uint32_t alt_free = alt_mask_ref.load(cuda::memory_order_acquire) & VALID;

                while (alt_free != 0)
                {
                    int dst_slot = __ffs(alt_free) - 1;
                    uint32_t new_alt = alt_free & ~(1u << dst_slot);

                    if (alt_mask_ref.compare_exchange_weak(
                            alt_free, new_alt,
                            cuda::memory_order_acq_rel,
                            cuda::memory_order_acquire))
                    {
                        // ① Claimed dst_slot uniquely in alt_bucket
                        // ② Write victim to dst — visible in both src and dst
                        cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(
                            *table->loadKV(alt_bucket, dst_slot))
                            .store(victim_kv, cuda::memory_order_release);

                        // ③ (D) Plain store — both locks held, no delete races
                        src_ref.store(kv, cuda::memory_order_release);
                        status = 1;
                        break;
                    }
                    // CAS failure: alt_free updated to current value — retry
                    alt_free &= VALID;
                }
                // alt_free == 0: all alt_bucket slots claimed — try next victim
            }

            unlock_ordered(table, bucket_idx, alt_bucket);
        }

        #if BREAKDOWN_INSERT
        cuckoo_end = clock64();
        #endif

        if(status == 0)
        {
            #if BREAKDOWN_INSERT
            stash_start = clock64();
            #endif

            if (push_to_stash(table, stash_table, bucket_idx, kv))
                status = 2;

            #if BREAKDOWN_INSERT
            stash_end = clock64();
            #endif
        }
    }

    #if BREAKDOWN_INSERT
    double cuckoo_time = tile.shfl(cuckoo_end - cuckoo_start, 0);
    double stash_time = tile.shfl(stash_end - stash_start, 0);
    breakdown_per_tile.stageC = cuckoo_time;
    breakdown_per_tile.stageD = stash_time;
    #endif

    return tile.shfl(status, 0);
}
__device__ __forceinline__ double min_start_time(
    double* start_time_array,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    double cur_time = start_time_array[tile.thread_rank()];

    for(int offset = tile.size()/2; offset > 0; offset /= 2)
    {
        double other = tile.shfl_down(cur_time, offset);
        cur_time = min(cur_time, other);
    }
    return tile.shfl(cur_time, 0);
}


// hive_insert_one_coop — canonical insert protocol
//
// Duplication is prevented through three structural guarantees:
//
//   1. Phase 1 (Stage A cross-bucket scan): scans ALL NumHashes candidate
//      buckets for the key before any insert attempt. This catches both
//      (a) a key already present in its primary bucket, and (b) a key
//      previously cuckoo-evicted to its alternate bucket. Either case is
//      resolved as an in-place update — Stage B is never reached.
//
//   2. Deterministic canonical bucket = H0(key): the insertion target for
//      Stage B depends only on the key, not on runtime load state. Any two
//      concurrent tiles inserting the same key ALWAYS target the same bucket,
//      so insert_lockfree's internal Stage A re-check is guaranteed to find
//      a concurrent commit in the same bucket and convert to update.
//
//   3. Cuckoo eviction targets canonical bucket only: victims move to their
//      own alternate bucket; the incoming key always lands at H0(key).
//      Phase 1 on the next insert of a displaced key finds it via cross-scan.
template<typename TableType, typename KeyType, typename ValueType, typename HashPolicy>
__device__ __forceinline__ bool hive_insert_one_coop(
    TableType* __restrict__ table,
    HiveOverflowStashBucket<kv_type>* __restrict__ stash_table,
    uint32_t key,
    uint32_t value,
    cg::thread_block_tile<TILE_SIZE> tile
)
{
    if (key == SENTINEL) return false;

    const uint64_t kv = packKV(key, value);

    // ── Phase 1: Stage A — scan ALL candidate buckets, no insert ───────────
    // Resolves: (a) normal key-present update, (b) post-eviction re-insert.
    #pragma unroll
    for (int i = 0; i < HashPolicy::NumHashes; ++i)
    {
        const uint32_t b = HashPolicy::get_bucket(i, key, table->num_buckets);

        if (scan_and_update<TableType>(table, b, key, value, tile))
        {
            return true;
        }

        // Prefetch the next candidate bucket while processing the current one
        if (i + 1 < HashPolicy::NumHashes && tile.thread_rank() == 0)
        {
            const uint32_t next_b = HashPolicy::get_bucket(i + 1, key, table->num_buckets);
            asm volatile("prefetch.global.L1 [%0];" ::
                "l"(reinterpret_cast<const void*>(&table->buckets[next_b])));
        }
    }

    // ── Phase 1.5: Stage A (Stash) — scan canonical stash for existing keys ──
    // Canonical Anchoring: keys are ONLY ever stashed in their canonical (H0) bucket.
    const uint32_t canonical = HashPolicy::get_bucket(0, key, table->num_buckets);
    if (scan_stash_and_update<TableType>(table, stash_table, canonical, key, value, tile))
    {
        return true;
    }

    // ── Canonical bucket: H0(key) — deterministic, runtime-state-free ─────
    // All concurrent inserts of the same key target this bucket, enabling
    // insert_lockfree's internal Stage A re-check to prevent duplicates.

    // Prefetch canonical bucket ahead of Stage B
    if (tile.thread_rank() == 0)
        asm volatile("prefetch.global.L1 [%0];" ::
            "l"(reinterpret_cast<const void*>(&table->buckets[canonical])));

    // ── Phase 2: Stage B — CAS into canonical bucket only ──────────────────
    // insert_lockfree re-checks Stage A on every CAS-retry iteration, so
    // a concurrent same-key commit landed between Phase 1 and here is caught
    // and the operation safely converts to an update.
    if (insert_lockfree<TableType, uint32_t, uint32_t>(table, canonical, key, value, tile))
    {
        return true;
    }

    // ── Cuckoo eviction from canonical bucket ─────────────────────────────
    if (cuckoo_evict_and_insert<TableType, HashPolicy>(tile, table, stash_table, canonical, kv))
    {
        return true;
    }

    // All paths exhausted — key could not be placed
    return false;
}


template<typename TableType, typename KeyType, typename ValueType, typename HashPolicy>
__device__ __forceinline__ bool hive_insert_one_only(
    TableType* __restrict__ table,
    HiveOverflowStashBucket<kv_type>* __restrict__ stash_table,
    uint32_t key,
    uint32_t value,
    cg::thread_block_tile<TILE_SIZE> tile
    #if BREAKDOWN_INSERT
    , InsertBreakdown& breakdown_per_tile
    #endif
)
{
    if(key == SENTINEL) return false;

    const uint64_t kv = packKV(key, value);

    #if BREAKDOWN_INSERT
    double start_time[TILE_SIZE] = {0.0};
    double end_time = 0;
    start_time[tile.thread_rank()] = clock64();
    #endif

    // Phase 1: scan ALL candidate buckets for an existing copy of this key.
    // Needed even in insert-only workloads: cuckoo eviction can displace a key
    // from its canonical bucket to its alternate bucket, so a later insert of
    // the same key must find it there and update in-place rather than
    // creating a duplicate.
    #pragma unroll
    for (int i = 0; i < HashPolicy::NumHashes; ++i)
    {
        const uint32_t b = HashPolicy::get_bucket(i, key, table->num_buckets);
        if (scan_and_update_insert_only<TableType>(table, b, key, value, tile))
        {
            #if BREAKDOWN_INSERT
            end_time = clock64();
            breakdown_per_tile.stageA = end_time - min_start_time(start_time, tile);
            #endif
            return true;
        }
    }

    // Canonical bucket = H0(key).
    // Prefetch the freeMask word for canonical: insert_freemask_first reads
    // only 4 bytes (freeMask), not the full 64-byte bucket. For SoA layout
    // the freeMask lives in a separate array — prefetching it while Phase 1
    // finishes hides the cache-miss latency of that first 4-byte load.
    const uint32_t canonical = HashPolicy::get_bucket(0, key, table->num_buckets);
    if (tile.thread_rank() == 0)
        asm volatile("prefetch.global.L1 [%0];" ::
            "l"(reinterpret_cast<const void*>(table->getFreeMask(canonical))));

    if (insert_freemask_first<TableType, uint32_t, uint32_t>(table, canonical, key, value, tile))
    {
        #if BREAKDOWN_INSERT
        end_time = clock64();
        breakdown_per_tile.stageB = end_time - min_start_time(start_time, tile);
        #endif
        return true;
    }

    int evict_status = cuckoo_evict_and_insert_only<TableType, HashPolicy>(
        tile, table, stash_table, canonical, kv
        #if BREAKDOWN_INSERT
        , breakdown_per_tile
        #endif
    );
    
    if(evict_status > 0)
    {
        return true;
    }

    return false;
}