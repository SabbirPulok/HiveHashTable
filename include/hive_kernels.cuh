#pragma once
#include "hive_hash_table.cuh"
#include "hive_hash_insert.cuh"
#include "hive_hash_lookup.cuh"
#include "hive_hash_atomicInc.cuh"
#include "hive_hash_delete.cuh"
#include "hive_hash_resize.cuh"
#include "HashPolicies.cuh"
#include "warpspeed.cuh"

template<typename TableType>
__global__ void warpspeed_mixed_kernel(
    TableType* table,
    Operation* ops,
    size_t num_ops,
    uint64_t* results //output parameter
)
{
    size_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

    bool active = global_thread_id < num_ops;
    
    if (tile.all(!active)) return;

    OperationType my_type = OperationType::NONE;
    uint32_t my_key = 0;
    
    if (active) {
        Operation op = ops[global_thread_id];
        my_type = op.type;
        my_key = static_cast<uint32_t>(op.key);
    }

    for(int i = 0; i < TILE_SIZE; ++i)
    {
        OperationType type = (OperationType)tile.shfl((int)my_type, i);
        uint32_t key = tile.shfl(my_key, i);
        uint32_t value = key + 1; 

        uint64_t result = 0;
        bool success = false;

        switch(type)
        {
            case OperationType::INSERT:
            {
                success = table->upsert_replace(tile, key, value);
                result = success ? 1 : 0;
                break;
            }
            case OperationType::LOOKUP:
            {
                uint32_t value_out = 0;
                success = table->lookup(tile, key, value_out);
                result = success ? 1 : 0;
                break;
            }
            case OperationType::DELETE:
            {
                success = table->erase_key(tile, key);
                result = success ? 1 : 0;
                break;
            }
            default:
                break;
        }

        if (tile.thread_rank() == i && active)
        {
             if (results != nullptr) results[global_thread_id] = result;
        }
    }
}

template<typename TableType, typename HashPolicy>
__global__ void hive_lookup_kernel(
    TableType* table,
    HiveOverflowStashBucket<kv_type>* stash_table,
    Operation* ops,
    size_t num_ops,
    uint64_t* results
)
{
    // Assign each thread a single operation
    // Threads in a tile cooperate to process each thread's operation one by one
    size_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    //Cooperative tile processing
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

    bool active = global_thread_id < num_ops;
    
    //Optimization: if all in tile are inactive, return
    if (tile.all(!active)) return;

    uint32_t my_key = 0; // Default dummy
    if (active) {
        my_key = static_cast<uint32_t>(ops[global_thread_id].key);
    }

    for(int i = 0; i < TILE_SIZE; ++i)
    {
        uint32_t key = tile.shfl(my_key, i);
        uint32_t value = 0;
        
        //All threads in tile cooperate for 'key'
        bool success = hive_lookup_one_coop<TableType, key_type, value_type, HashPolicy>(table, stash_table, key, &value, tile);
        
        if (tile.thread_rank() == i && active)
        {
             if (results != nullptr) results[global_thread_id] = success ? 1 : 0;
        }
    }

    //scan for key into stash
}

template<typename TableType, typename HashPolicy>
__global__ void hive_mixed_kernel(
    TableType* table,
    HiveOverflowStashBucket<kv_type>* stash_table,
    Operation* ops,
    size_t num_ops,
    uint64_t* results //output parameter
    #if BREAKDOWN_INSERT
    , InsertBreakdown* insert_breakdown
    #endif
)
{
    // Assign each thread a single operation
    // Threads in a tile cooperate to process each thread's operation one by one
    size_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    //Cooperative tile processing
    cg::thread_block_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

    bool active = global_thread_id < num_ops;
    
    //Optimization: if all in tile are inactive, return
    if (tile.all(!active)) return;

    OperationType my_type = OperationType::NONE;
    uint32_t my_key = 0;
    
    if (active) {
        Operation op = ops[global_thread_id];
        my_type = op.type;
        my_key = static_cast<uint32_t>(op.key);
    }

    for(int i = 0; i < TILE_SIZE; ++i)
    {
        OperationType type = (OperationType)tile.shfl((int)my_type, i);
        uint32_t key = tile.shfl(my_key, i);
        uint32_t value = key + 1; // Derived value as in original

        uint64_t result = 0;
        bool success = false;

        switch(type)
        {
            case OperationType::INSERT:
            {
                success = hive_insert_one_coop<TableType, key_type, value_type, HashPolicy>(table, stash_table, key, value, tile);
                result = success ? 1 : 0;
                break;
            }
            case OperationType::LOOKUP:
            {
                success = hive_lookup_one_coop<TableType, key_type, value_type, HashPolicy, true>(table, stash_table, key, &value, tile);
                result = success ? 1 : 0;
                break;
            }
            case OperationType::DELETE:
            {
                success = hive_delete_one_coop<TableType, key_type, HashPolicy>(table, stash_table, key, tile);
                result = success ? 1 : 0;
                break;
            }
            case OperationType::NONE:
            {
                break;
            }
        }

        if (tile.thread_rank() == i && active)
        {
             if (results != nullptr) results[global_thread_id] = result;
        }
    }
}

// hive_insert_kernel — bulk insert-only kernel
//
// Optimized for throughput when the workload consists entirely of inserts.
// Unlike the mixed kernel, this relies on a simplified lock-free protocol
// that assumes unique keys and ignores concurrent deletes.
//
// Work distribution:
//      Thread block tiles cooperatively process one key at a time.
//      Boundary threads (global_thread_id >= num_ops) get my_key=0=SENTINEL.
//      hive_insert_one_coop returns false immediately for SENTINEL keys.
//      Results are only written for active threads (active flag guard).
template<typename TableType, typename HashPolicy>
__global__ void hive_insert_kernel(
    TableType* table,
    HiveOverflowStashBucket<kv_type>* stash_table,
    Operation* ops,
    size_t num_ops,
    uint64_t* results
    #if BREAKDOWN_INSERT
    , InsertBreakdown* insert_breakdown
    #endif
)
{
    size_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    cg::thread_block_tile<TILE_SIZE> tile =
        cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

    bool active = global_thread_id < num_ops;

    // If every lane in this tile is out of range, exit early
    if (tile.all(!active)) return;

    // Each thread loads its own key. Inactive threads get SENTINEL (0)
    // so hive_insert_one_coop skips them without any branch divergence.
    uint32_t my_key = 0;
    if (active)
        my_key = static_cast<uint32_t>(ops[global_thread_id].key);

    // Process TILE_SIZE operations cooperatively: one per tile position.
    // Lane i broadcasts its key; all lanes call hive_insert_one_coop together.
    for (int i = 0; i < TILE_SIZE; ++i)
    {
        uint32_t key   = tile.shfl(my_key, i);
        uint32_t value = key + 1; // Derived value — matches mixed kernel convention

        bool success = false;

        #if BREAKDOWN_INSERT
        InsertBreakdown breakdown_per_tile;
        success = hive_insert_one_only<TableType, key_type, value_type, HashPolicy>(
            table, stash_table, key, value, tile, breakdown_per_tile
        );
        #else
        success = hive_insert_one_only<TableType, key_type, value_type, HashPolicy>(
            table, stash_table, key, value, tile
        );
        #endif

        // Only the owning lane writes its result; inactive lanes are silenced
        if (tile.thread_rank() == i && active)
        {
            if (results != nullptr)
                results[global_thread_id] = success ? 1 : 0;
            #if BREAKDOWN_INSERT
            if(insert_breakdown != nullptr)
            {
                insert_breakdown[global_thread_id] = breakdown_per_tile;
            }
            #endif
        }
    }
}

template<typename TableType>
__global__ void check_access_of_device_ht(
    TableType* table,
    HiveOverflowStashBucket<kv_type>* stash_table,
    Operation* ops,
    size_t num_ops,
    uint64_t* results //output parameter
)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= table->num_buckets * TableType::SLOTS) return;

    if(i >= num_ops) return;

    //Access each slot
    uint32_t bucket = i / TableType::SLOTS;
    uint32_t slot = i % TableType::SLOTS;

    //Read the KV
    uint64_t kv = table->buckets[bucket].kv[slot];
    if(kv != SENTINEL)
    {
        //Just a dummy operation
        kv ^= 0xDEADBEEF;
    }

    Operation op = ops[i];
    uint32_t key = static_cast<uint32_t>(op.key);
    uint32_t value = static_cast<uint32_t>(op.key + 1);

    //stash_table->keys[i%stash_table->capacity] = key;
    results[i]=0;
}

//Calculate Occupancy
template<typename TableType>
__global__ void count_occupancy_kernel(TableType* table, uint64_t* total_occupied)
{
    size_t bucket_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(bucket_idx < table->num_buckets)
    {
        uint32_t free_mask = *(table->getFreeMask(bucket_idx));

        uint32_t valid_mask = valid_slot_mask<TableType>();

        uint32_t occupied_mask = (~free_mask) & valid_mask;

        //Count number of occupied slots
        uint32_t occupied_count = __popc(occupied_mask);

        if(occupied_count > 0)
        {
            //Atomic add to total occupied
            atomicAdd((unsigned long long int*)total_occupied, (unsigned long long int)occupied_count);
        }
    }
}


//Only for AoaS buckets
template<typename BucketType>
__global__ void initAoaSBucket(
    BucketType* buckets,
    size_t num_buckets,
    uint32_t initial_mask
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < num_buckets)
    {
        buckets[tid].header.freeMask = initial_mask;
        buckets[tid].header.lock = 0;

        for(int i = 0; i < HIVE_BUCKET_SLOTS_AOAS; i++)
        {
            // Infer the KV element type from the bucket instance and initialize to zero
            buckets[tid].kv[i] = 0;
        }
    }
}
// Allow HiveHashTableAoaS_LeadMetaData to reuse the same AoAS creation logic.
// Forwarding overload calls the AoAS implementation (they must be layout-compatible).
