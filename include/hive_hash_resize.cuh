#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdint>
#include "hive_hash_table_struct.cuh"
#include "hash.hpp"
#include "utils.h"
#include <algorithm> //std::min
#include <cooperative_groups.h>
#include <vector>
#include "cuda_helper.cuh"

#include "hive_hash_insert.cuh"
#include "hive_hash_delete.cuh"

namespace cg = cooperative_groups;

//H.index_mask -> current 2^m -1
//H.split_ptr -> how many buckets have already been split
//H.nBuckets -> how many buckets currently active
//H.max_buckets -> maximum pre-allocated buckets
//K -> how many new buckets you'd like to add on this call

inline void advance_round(HiveHashTable& H)
{
    H.index_mask = (H.index_mask << 1) | 1; //double the index mask
    H.split_ptr = 0; //reset the split pointer
}

template<typename HashPolicy>
__device__ bool hash_check(uint64_t kv, int bucket_index, uint32_t index_mask)
{
    uint32_t key = unpackKey(kv);

    for(int i = 0; i < HashPolicy::NumHashes; i++)
    {
        uint32_t hash = HashPolicy::get_bucket(i, key, index_mask);
        if (hash == bucket_index) return true;
    }
    return false;
}

template<typename HashPolicy>
__global__ void hive_hash_split_buckets(
    HiveHashTable* table,
    const uint32_t* __restrict__ split_buckets,
    size_t num_split_buckets,
    uint32_t base_index //the starting index of this round, i.e. 2^m
)
{
    auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    auto global_tile_id = blockIdx.x * tile.meta_group_size() + tile.meta_group_rank();
    auto lane_id = tile.thread_rank();

    if(global_tile_id >= num_split_buckets) return;

    uint32_t src_bucket_idx = split_buckets[global_tile_id];
    uint32_t dst_bucket_idx = base_index + src_bucket_idx;

    auto atomic_src_lock = cuda::atomic_ref<uint16_t, cuda::thread_scope_device>(table->lock[src_bucket_idx]);
    auto atomic_dst_lock = cuda::atomic_ref<uint16_t, cuda::thread_scope_device>(table->lock[dst_bucket_idx]);

    if(lane_id == 0)
    {
        //lock both src and dst buckets
        while(atomic_src_lock.exchange(1u, cuda::memory_order_acquire) != 0u) {};
        while(atomic_dst_lock.exchange(1u, cuda::memory_order_acquire) != 0u) {};
    }
    tile.sync();

    HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>* src_bucket = &table->buckets[src_bucket_idx];
    HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>* dst_bucket = &table->buckets[dst_bucket_idx];

    uint64_t kv = EMPTY_KV;
    bool occupied = false;
    if(lane_id < HIVE_BUCKET_SLOTS)
    {
        kv = src_bucket->kv[lane_id];
        occupied = (kv != EMPTY_KV);
    }
        
    //Decide mover under linear hashing next-level rule
    const uint32_t next_mask = (table->index_mask << 1) | 1u;
    bool should_move = false;

    if(occupied)
    {
        bool in_src_match = hash_check<HashPolicy>(kv, src_bucket_idx, table->index_mask);
        bool from_dst_match = hash_check<HashPolicy>(kv, dst_bucket_idx, next_mask);
        should_move = (in_src_match && from_dst_match);
    }

    //Build mask of movers and compute compact dst positions via tile prefix sums
    uint32_t mover_mask = tile.ballot(should_move);
    uint32_t my_rank = __popc(mover_mask & ((1u << lane_id) - 1)); //number of movers before me in the tile
    uint32_t num_movers = __popc(mover_mask);

    if(should_move && occupied)
    {
        dst_bucket->kv[my_rank] = kv; //move to dst bucket
        src_bucket->kv[lane_id] = EMPTY_KV; //remove from src bucket
    }

    tile.sync();

    //update the mask
    if(lane_id == 0)
    {
        auto atomic_src_freemask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(table->freeMask[src_bucket_idx]);
        auto atomic_dst_freemask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(table->freeMask[dst_bucket_idx]);

        uint32_t src_freeMask = atomic_src_freemask.load(cuda::memory_order_acquire);
        uint32_t dst_freeMask = atomic_dst_freemask.load(cuda::memory_order_acquire);

        //Mark moved slots in src as free
        src_freeMask |= mover_mask;

        //Mark moved slots in dst as occupied
        dst_freeMask = 0xFFFFFFFFu; //all free
        if(num_movers > 0)
        {
            uint32_t occ = (num_movers == HIVE_BUCKET_SLOTS) ? 0xFFFFFFFFu : ((1u << num_movers) - 1u);
            dst_freeMask &= ~occ; //mark occupied slots as 0
        }

        atomic_src_freemask.store(src_freeMask, cuda::memory_order_release);
        atomic_dst_freemask.store(dst_freeMask, cuda::memory_order_release);

        //Unlock both buckets
        atomic_src_lock.exchange(0u, cuda::memory_order_release);
        atomic_dst_lock.exchange(0u, cuda::memory_order_release);
    }
    tile.sync();

}

template <typename HashPolicy = Default2HashPolicy>
void inline hive_hash_grow( HiveHashTable* d_table, 
                     size_t add_buckets)
{
    HiveHashTable h_table;
    CoarseGraindGPUTimer timer;
    float t = 0.0f;

    size_t moved_keys = add_buckets * HIVE_BUCKET_SLOTS;

    std::cout << "Number of New Buckets Added: " << add_buckets << ", Number of KV Slots Included: " << moved_keys << std::endl;

    CUDA_CHECK(cudaMemcpy(&h_table, d_table, sizeof(HiveHashTable), cudaMemcpyDeviceToHost));

    while (add_buckets > 0 && (h_table.num_buckets + add_buckets) < h_table.max_num_buckets) { //no headroom
        uint32_t baseline_size = h_table.index_mask + 1; //end goal: have to split all of these buckets
        size_t remain = baseline_size - h_table.split_ptr; //how many buckets remain to be split in this round, split_ptr: already splitted of that baseline
        uint32_t batch = std::min({add_buckets, remain, h_table.max_num_buckets - h_table.num_buckets}); //how many buckets to split in this batch

        if(batch == 0)
        {
            advance_round(h_table);
            CUDA_CHECK(cudaMemcpy(d_table, &h_table, sizeof(HiveHashTable), cudaMemcpyHostToDevice));
            continue;
        }

        //Build the list of src buckets to split: split_ptr ... split_ptr + batch -1
        std::vector<uint32_t> h_src_buckets(batch);
        for(int i=0; i<batch; i++)
        {
            h_src_buckets[i] = h_table.split_ptr + i;

            uint32_t new_bucket = h_table.num_buckets + i;
            uint32_t fullMask = 0xFFFFFFFFu; //all slots free
            uint16_t unlock = 0; //unlocked

            CUDA_CHECK(cudaMemcpy(h_table.freeMask + new_bucket, &fullMask, sizeof(uint32_t), cudaMemcpyHostToDevice)); //init new buckets free mask
            CUDA_CHECK(cudaMemcpy(h_table.lock + new_bucket, &unlock, sizeof(uint16_t), cudaMemcpyHostToDevice)); //init new buckets lock
            CUDA_CHECK(cudaMemset(&h_table.buckets[new_bucket], 0, sizeof(HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>))); //init new buckets body
        }

        h_table.num_buckets += batch;

        CUDA_CHECK(cudaMemcpy(d_table, &h_table, sizeof(HiveHashTable), cudaMemcpyHostToDevice));

        //Upload src bucket list to device
        uint32_t* d_src_buckets;
        CUDA_CHECK(cudaMalloc(&d_src_buckets, sizeof(uint32_t)*batch));
        CUDA_CHECK(cudaMemcpy(d_src_buckets, h_src_buckets.data(), sizeof(uint32_t)*batch, cudaMemcpyHostToDevice));

        dim3 grid(batch), block(256);

        timer.start();
        hive_hash_split_buckets<HashPolicy><<<grid, block>>>(d_table, d_src_buckets, batch, baseline_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        t += timer.getElapsedTime();

        CUDA_CHECK(cudaFree(d_src_buckets));

        CUDA_CHECK(cudaMemcpy(&h_table, d_table, sizeof(HiveHashTable), cudaMemcpyDeviceToHost));
        h_table.split_ptr += batch;

        //Have we already split all baseline size buckets?
        if(h_table.split_ptr == baseline_size) //finished splitting this round
        {
            advance_round(h_table); //update index mask or set new goal and reset split to 0
        }

        CUDA_CHECK(cudaMemcpy(d_table, &h_table, sizeof(HiveHashTable), cudaMemcpyHostToDevice));
        add_buckets -= batch;

        if(h_table.num_buckets >= h_table.max_num_buckets)
            break;
    }
    double throughput_mlops = static_cast<double>(moved_keys - (add_buckets * HIVE_BUCKET_SLOTS)) / (t/1000.0f) / 1e6;
    std::cout << "Hive Hash Grow Time: " << t << " ms, Throughput: " << throughput_mlops << " Mops/s" << std::endl;
}

inline void regress_round(HiveHashTable& H)
{
    H.index_mask >>= 1; //halve the index mask
    H.split_ptr = H.index_mask + 1; //reset the split pointer to the end of previous round
}


// Kernel: do not use inline qualifiers for __global__ functions
__global__ void hive_hash_merge_buckets(
    HiveHashTable* table,
    HiveOverflowStash<KeyType, ValueType>* stash,
    const uint32_t* __restrict__ src_buckets,
    size_t num_src_buckets,
    uint32_t base_index, //the starting index of this round, i.e. 2^m
    bool* success_flag
)
{
    auto tile = cg::tiled_partition<HIVE_BUCKET_SLOTS>(cg::this_thread_block());
    auto global_tile_id = blockIdx.x * tile.meta_group_size() + tile.meta_group_rank();
    auto lane_id = tile.thread_rank();

    if(global_tile_id >= num_src_buckets) return;

    uint32_t dst_bucket_idx = src_buckets[global_tile_id];
    uint32_t src_bucket_idx = dst_bucket_idx + base_index;

    auto atomic_src_lock = cuda::atomic_ref<uint16_t, cuda::thread_scope_device>(table->lock[src_bucket_idx]);
    auto atomic_dst_lock = cuda::atomic_ref<uint16_t, cuda::thread_scope_device>(table->lock[dst_bucket_idx]);

    if(lane_id == 0)
    {
        //lock both src and dst buckets
        while(atomic_src_lock.exchange(1u, cuda::memory_order_acquire) != 0u) {};
        while(atomic_dst_lock.exchange(1u, cuda::memory_order_acquire) != 0u) {};
    }
    tile.sync();

    HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>* src_bucket = &table->buckets[src_bucket_idx];
    HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>* dst_bucket = &table->buckets[dst_bucket_idx];

    //Read Destination bucket free mask
    auto freeMask_dst = table->freeMask[dst_bucket_idx];

    //Read entry from source bucket
    uint64_t kv = (lane_id < HIVE_BUCKET_SLOTS) ? src_bucket->kv[lane_id] : EMPTY_KV;
    bool occupied_src = (kv != EMPTY_KV);

    uint32_t occupied_mask = tile.ballot(occupied_src);
    
    //Free slots in dst bucket
    uint32_t num_free_slots = __popc(freeMask_dst);

    //Compute prefix sum to get destination positions for each occupied entry
    uint32_t my_rank = __popc(occupied_mask & ((1u << lane_id) - 1)); //number of occupied slots before me in the tile

    //Track which free slots are used
    __shared__ uint32_t used_mask[8]; //256/32 = 8 warps per block
    if(lane_id == 0) used_mask[tile.meta_group_rank()] = 0;
    tile.sync();


    //Find Nth free slot in dst bucket for each moving entry
    if(occupied_src)
    {
        if(my_rank < num_free_slots)
        {
            //Find my_rank 
            uint32_t free_pos = 0;
            uint32_t count = 0;
            uint32_t temp_mask = freeMask_dst;

            while(temp_mask && count <= my_rank)
            {
                free_pos = __ffs(temp_mask) - 1; //-1
                if(count == my_rank) break; //found my position
                temp_mask &= ~(1u << free_pos); //clear this bit
                count++;
            }

            //Move entry from src to dst
            dst_bucket->kv[free_pos] = kv; //move to dst bucket
            src_bucket->kv[lane_id] = EMPTY_KV; //remove from src bucket

            atomicOr(&used_mask[tile.meta_group_rank()], (1u << free_pos));
        }
        else
        {
            //Overflow to stash
            bool pushed = push_to_stash(stash, kv);
            if(pushed)
            {
                src_bucket->kv[lane_id] = EMPTY_KV;
            }
            else
            {
                //Stash is full, fail the merge
                if(lane_id == 0) //Only one thread needs to report failure
                {
                   auto atomic_success = cuda::atomic_ref<bool, cuda::thread_scope_device>(*success_flag);
                   atomic_success.store(false, cuda::memory_order_release);
                }
            }
        }
    }

    //Update Free mask
    if(lane_id == 0)
    {
        auto atomic_src_freemask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(table->freeMask[src_bucket_idx]);
        auto atomic_dst_freemask = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(table->freeMask[dst_bucket_idx]);

        //Mark used slots in dst as occupied
        uint32_t new_dst_freeMask = freeMask_dst & ~used_mask[tile.meta_group_rank()];

        atomic_src_freemask.store(0xFFFFFFFFu, cuda::memory_order_release);
        atomic_dst_freemask.store(new_dst_freeMask, cuda::memory_order_release);

        //Unlock both buckets
        atomic_src_lock.exchange(0u, cuda::memory_order_release);
        atomic_dst_lock.exchange(0u, cuda::memory_order_release);
    }
    tile.sync();
    
}
void hive_hash_shrink(HiveHashTable* d_table, HiveOverflowStash<KeyType, ValueType>* d_stash, size_t remove_buckets)
{
    HiveHashTable h_table;
    CoarseGraindGPUTimer timer;
    float t = 0.0f;
    double removed_keys = remove_buckets * HIVE_BUCKET_SLOTS;

    std::cout << "Number of Buckets Merged: " << remove_buckets << ", Number of KV Slots Removed: " << removed_keys << std::endl;

    CUDA_CHECK(cudaMemcpy(&h_table, d_table, sizeof(HiveHashTable), cudaMemcpyDeviceToHost));

    bool *success_flag;
    CUDA_CHECK(cudaMalloc(&success_flag, sizeof(bool)));
    CUDA_CHECK(cudaMemset(success_flag, 1, sizeof(bool)));

    while (remove_buckets > 0 && h_table.num_buckets > 1) { //no headroom
        uint32_t base = h_table.index_mask + 1; //current round start index

        //std::cout << "Current round base: " << base << ", split_ptr: " << h_table.split_ptr << ", num_buckets: " << h_table.num_buckets << ", remove buckets: "<< remove_buckets << "\n";
        uint32_t batch = std::min({remove_buckets, (size_t)h_table.split_ptr, h_table.num_buckets - 1}); //how many buckets to merge in this batch

        //std::cout << "Shrinking " << batch << " buckets\n";

        if(batch == 0)
        {
            if(h_table.index_mask == 0)
                break; //cannot shrink anymore
            regress_round(h_table);
            CUDA_CHECK(cudaMemcpy(d_table, &h_table, sizeof(HiveHashTable), cudaMemcpyHostToDevice));
            continue;
        }

        //build src list for merging : split_ptr-1, split_ptr-2, ..., split_ptr-batch
        std::vector<uint32_t> h_dst_buckets(batch);
        for(int i=0; i<batch; i++)
        {
            h_dst_buckets[i] = h_table.split_ptr - 1 - i;
        }

        //Upload src bucket list to device
        uint32_t* d_dst_buckets;
        CUDA_CHECK(cudaMalloc(&d_dst_buckets, sizeof(uint32_t)*batch));
        CUDA_CHECK(cudaMemcpy(d_dst_buckets, h_dst_buckets.data(), sizeof(uint32_t)*batch, cudaMemcpyHostToDevice));

        bool h_success_flag = true;
        CUDA_CHECK(cudaMemcpy(success_flag, &h_success_flag, sizeof(bool), cudaMemcpyHostToDevice));

        dim3 grid(batch), block(256);
        timer.start();
        hive_hash_merge_buckets<<<grid, block>>>(d_table, d_stash, d_dst_buckets, batch, base, success_flag);
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.stop();
        t += timer.getElapsedTime();

        CUDA_CHECK(cudaMemcpy(&h_success_flag, success_flag, sizeof(bool), cudaMemcpyDeviceToHost));
        if(h_success_flag != true)
        {
            std::cerr << "Shrink failed due to insufficient space in destination buckets\n";
            CUDA_CHECK(cudaFree(d_dst_buckets));
            break;
        }
        CUDA_CHECK(cudaFree(d_dst_buckets));

        CUDA_CHECK(cudaMemcpy(&h_table, d_table, sizeof(HiveHashTable), cudaMemcpyDeviceToHost));
        h_table.split_ptr -= batch;
        h_table.num_buckets -= batch;
        remove_buckets -= batch;

        if(h_table.split_ptr == 0 && h_table.index_mask > 0) //finished merging this round
        {
            regress_round(h_table);
        }

        CUDA_CHECK(cudaMemcpy(d_table, &h_table, sizeof(HiveHashTable), cudaMemcpyHostToDevice));
    }
    double throughput_mlops = static_cast<double>(removed_keys - (remove_buckets * HIVE_BUCKET_SLOTS)) / (t/1000.0f) / 1e6;

    std::cout << "Hive Hash Shrink Time: " << t << " ms, throughput: " << throughput_mlops << " MLOPS\n";
    cudaFree(success_flag);
}
