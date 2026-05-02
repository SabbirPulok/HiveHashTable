#pragma once

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace warpspeed{

    #define N_CUCKOO_HASHES 3
    #define CUCKOO_MAX_PROBES 20
    #define MAX_CUCKOO_ATTEMPTS 500

template<typename KeyType, typename ValueType>
struct kv_pair{
    KeyType key;
    ValueType value;
};

__device__ __forceinline__ uint64_t hash(const void* key, int len , uint64_t seed)
{
   const uint64_t m = 0xc6a4a7935bd1e995;
    const int r = 47;
    uint64_t h = seed ^ (len * m);
    const uint64_t * data = (const uint64_t *)key;
    const uint64_t * end = data + (len/8);

    while(data != end) {
        uint64_t k = *data++;
        k *= m; 
        k ^= k >> r; 
        k *= m; 
        h ^= k;
        h *= m; 
    }

    const unsigned char * data2 = (const unsigned char*)data;
    switch(len & 7) {
        case 7: h ^= (uint64_t)data2[6] << 48;
        case 6: h ^= (uint64_t)data2[5] << 40;
        case 5: h ^= (uint64_t)data2[4] << 32;
        case 4: h ^= (uint64_t)data2[3] << 24;
        case 3: h ^= (uint64_t)data2[2] << 16;
        case 2: h ^= (uint64_t)data2[1] << 8;
        case 1: h ^= (uint64_t)data2[0];
                h *= m;
    };
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h; 
}

template <typename KeyType, KeyType DefaultKey, KeyType TombStoneKey, 
typename ValueType, ValueType DefaultValue, ValueType TombStoneValue, 
size_t partition_size, size_t bucket_size>
struct cuckoo_ht {
   using tile_type = cg::thread_block_tile<partition_size>;
   using pair = kv_pair<KeyType, ValueType>;

   struct bucket_type {
       pair slots[bucket_size];
       
       __device__ __forceinline__ void init(){
           for (size_t i = 0; i < bucket_size; ++i) {
               slots[i] = {DefaultKey, DefaultValue};
           }
       }
   };

   bucket_type* primary_buckets;
   uint32_t* primary_locks;
   uint64_t n_buckets_primary;
   uint64_t seed;

   __device__ __forceinline__ void stall_lock_one_thread(uint64_t bucket)
   {
        while(atomicCAS(&primary_locks[bucket], 0, 1) != 0);
   }

   __device__ __forceinline__ void unlock_one_thread(uint64_t bucket)
   {
       atomicExch(&primary_locks[bucket], 0);
   }

   __device__ __forceinline__ void stall_lock(tile_type tile, uint64_t bucket)
   {
       if(tile.thread_rank() == 0)
       {
           stall_lock_one_thread(bucket);
       }
       tile.sync();
   }

   __device__ __forceinline__ void unlock(tile_type tile, uint64_t bucket)
   {
       if(tile.thread_rank() == 0)
       {
           unlock_one_thread(bucket);
       }
       tile.sync();
   }

   __device__ __forceinline__ uint64_t get_bucket_index(const KeyType& key, size_t bucket_id)
   {
       bucket_id = bucket_id % N_CUCKOO_HASHES;
       return hash(&key, sizeof(KeyType), seed + bucket_id) % n_buckets_primary;
   }

   // Lookup
   __device__ __forceinline__ bool lookup(tile_type tile, KeyType key, ValueType& return_val)
   {
        for(size_t b = 0; b < N_CUCKOO_HASHES; b++)
        {
            uint64_t bucket_idx = get_bucket_index(key, b);
            bucket_type* b_ptr = &primary_buckets[bucket_idx];

            for(size_t i = tile.thread_rank(); i < bucket_size; i+= tile.size())
            {
                KeyType loaded_key = b_ptr->slots[i].key;

                bool match = (loaded_key == key);
                int found = __ffs(tile.ballot(match)) - 1;
                if(found != -1)
                {
                    // Non-atomic read, accepted in standard open-addressing without packing
                    return_val = b_ptr->slots[found].value;
                    return true;
                }
            }
        }
        return false;
   }

   __device__ __forceinline__ bool erase_key(tile_type tile, KeyType key)
   {
       for(size_t b = 0; b < N_CUCKOO_HASHES; b++)
       {
           uint64_t b_idx = get_bucket_index(key, b);
           stall_lock(tile, b_idx); 
           bucket_type* b_ptr = &primary_buckets[b_idx];

           for(size_t i = tile.thread_rank(); i < bucket_size; i+= tile.size())
           {
               KeyType loaded_key = b_ptr->slots[i].key;
               bool match = (loaded_key == key);

               uint32_t winner = __ffs(tile.ballot(match)) - 1;
               if(winner != -1)
               {
                   if (tile.thread_rank() == winner)
                   {
                       atomicExch((unsigned long long*)&b_ptr->slots[i].key, (unsigned long long)TombStoneKey);
                   }
                   unlock(tile, b_idx);
                   return true;
               }
           }
           unlock(tile, b_idx);
        }
        return false;
   }

   //Insert/ Upsert Replace
   __device__ __forceinline__ bool upsert_replace(tile_type tile, KeyType key, ValueType value)
   {
       for(size_t b = 0; b < N_CUCKOO_HASHES; b++)
       {
           uint64_t b_idx = get_bucket_index(key, b);
           stall_lock(tile, b_idx);
           bucket_type* b_ptr = &primary_buckets[b_idx];

           for(size_t i = tile.thread_rank(); i < bucket_size; i+= tile.size())
           {
               KeyType loaded_key = b_ptr->slots[i].key;
               bool is_empty_or_tomb = (loaded_key == DefaultKey) || (loaded_key == TombStoneKey);
               bool is_match = (loaded_key == key);
               bool valid_slot = is_empty_or_tomb || is_match;

               uint32_t ballot_res = tile.ballot(valid_slot);
               if(ballot_res)
               {
                   uint32_t winner = __ffs(ballot_res) - 1;
                   bool success = false;
                   if(tile.thread_rank() == winner)
                   {
                       unsigned long long old_k = atomicCAS((unsigned long long*)&b_ptr->slots[i].key, (unsigned long long)loaded_key, (unsigned long long)key);

                       if(old_k == (unsigned long long)loaded_key)
                       {
                           b_ptr->slots[i].value = value;
                           __threadfence();
                           success = true;
                       }
                   }

                   if(tile.ballot(success))
                   {
                       unlock(tile, b_idx);
                       return true;
                   }
               }
            }
            unlock(tile, b_idx);
        }
        
        KeyType current_key = key;
        ValueType current_value = value;
        int current_hash_id = 0;

        for(int attempt = 0; attempt < MAX_CUCKOO_ATTEMPTS; attempt++)
        {
            uint64_t evict_bucket_idx = get_bucket_index(current_key, current_hash_id);
            stall_lock(tile, evict_bucket_idx);

            bucket_type* b_ptr = &primary_buckets[evict_bucket_idx];
            KeyType evicted_k;
            ValueType evicted_v;

            if (tile.thread_rank() == 0) {
                evicted_k = b_ptr->slots[0].key;
                evicted_v = b_ptr->slots[0].value;
                
                b_ptr->slots[0].key = current_key;
                b_ptr->slots[0].value = current_value;
                __threadfence();
            }

            evicted_k = tile.shfl(evicted_k, 0);
            evicted_v = tile.shfl(evicted_v, 0);

            unlock(tile, evict_bucket_idx);

            if (evicted_k == DefaultKey || evicted_k == TombStoneKey) return true;

            current_key = evicted_k;
            current_value = evicted_v;
            current_hash_id = (current_hash_id + 1) % N_CUCKOO_HASHES;
        }

        return false;
   }

   template<typename TableType, typename K, typename V, int TSIZE>
    __device__ __forceinline__ bool warpspeed_insert_one_coop(
        TableType* table, 
        K key, 
        V value, 
        cg::thread_block_tile<TSIZE>& tile) 
    {
        return table->upsert_replace(tile, key, value);
    }

    template<typename TableType, typename K, typename V, int TSIZE>
    __device__ __forceinline__ bool warpspeed_lookup_one_coop(
        TableType* table, 
        K key, 
        V* value_out, 
        cg::thread_block_tile<TSIZE>& tile) 
    {
        V temp_val;
        bool found = table->lookup(tile, key, temp_val);
        if (found && value_out != nullptr && tile.thread_rank() == 0) {
            *value_out = temp_val;
        }
        return found;
    }

    template<typename TableType, typename K, int TSIZE>
    __device__ __forceinline__ bool warpspeed_delete_one_coop(
        TableType* table, 
        K key, 
        cg::thread_block_tile<TSIZE>& tile) 
    {
        return table->erase_key(tile, key);
    }

};
}
