#include "hive_hash_table.cuh"
#include "hash.hpp"
#include "cuda_helper.cuh"
#include "GPUTimer.h"
#include<vector>
#include<algorithm>
#include<cuda_runtime.h>
#include<cuda/atomic>
#include<iostream>


#include "hive_hash_insert.cuh"
#include "hive_hash_lookup.cuh"
#include "hive_hash_atomicInc.cuh"
#include "hive_hash_delete.cuh"
#include "hive_hash_resize.cuh"
#include "HashPolicies.cuh"
#include <bit>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform.h>


template<typename TableType, typename HashPolicy>
__global__ void hive_lookup_kernel(
    TableType* table,
    HiveOverflowStash<KeyType, ValueType>* stash_table,
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
        bool success = hive_lookup_one_coop<TableType, KeyType, ValueType, HashPolicy>(table, stash_table, key, &value, tile);
        
        if (tile.thread_rank() == i && active)
        {
             if (results != nullptr) results[global_thread_id] = success ? 1 : 0;
        }
    }
}

template<typename TableType, typename HashPolicy>
__global__ void hive_mixed_kernel(
    TableType* table,
    HiveOverflowStash<KeyType, ValueType>* stash_table,
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
                #if BREAKDOWN_INSERT
                size_t key_owner_idx = global_thread_id - tile.thread_rank() + i;
                success = hive_insert_one_coop<TableType, KeyType, ValueType, HashPolicy>(table, stash_table, key, value, tile, insert_breakdown[key_owner_idx]);
                #else
                success = hive_insert_one_coop<TableType, KeyType, ValueType, HashPolicy>(table, stash_table, key, value, tile);
                #endif
                
                result = success ? 1 : 0;
                break;
            }
            case OperationType::LOOKUP:
            {
                success = hive_lookup_one_coop<TableType, KeyType, ValueType, HashPolicy>(table, stash_table, key, &value, tile);
                result = success ? 1 : 0;
                break;
            }
            case OperationType::ATOMIC_INC:
            {
                success = hive_atomic_inc_coop<TableType, KeyType, ValueType, HashPolicy>(table, key, 1u, tile);
                result = success ? 1 : 0;
                break;
            }
            case OperationType::DELETE:
            {
                success = hive_delete_one_coop<TableType, KeyType, HashPolicy>(table, key, tile);
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

template<typename TableType>
__global__ void check_access_of_device_ht(
    TableType* table,
    HiveOverflowStash<KeyType, ValueType>* stash_table,
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

    stash_table->keys[i%stash_table->capacity] = key;
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

template<typename TableType, typename HashPolicy = Default2HashPolicy>
void run_hash_table_benchmark(
    TableType* hive_table_device,
    HiveOverflowStash<KeyType, ValueType>* hive_stash_device,
    Operation* d_ops,
    size_t num_ops,
    size_t stash_capacity,
    uint64_t* h_results,
    size_t num_blocks,
    size_t threads_per_block,
    size_t numIterations,
    double &elapsed_time,
    bool verify,
    bool all_deletes,
    bool all_lookups,
    bool skip_clear
    #if BREAKDOWN_INSERT
    , InsertBreakdown* d_insert_breakdown = nullptr
    #endif
)
{
    //Device Results
    uint64_t* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(uint64_t) * (num_ops + stash_capacity)));
    CUDA_CHECK(cudaMemset(d_results, 0, sizeof(uint64_t) * (num_ops + stash_capacity)));
    
    size_t numWarmup = all_deletes ? 0 : 5;
    numIterations = all_deletes ? 1 : numIterations;
    
    std::vector<InsertBreakdown> h_insert_breakdown(num_ops);

    //Warmup
    for (size_t i = 0; i < numWarmup; i++) {
        if(all_lookups)
        {
            hive_lookup_kernel<TableType, HashPolicy><<<num_blocks, threads_per_block>>>(
                hive_table_device,
                hive_stash_device,
                d_ops,
                num_ops,
                d_results
            );
        }
        else
        {
            hive_mixed_kernel<TableType, HashPolicy><<<num_blocks, threads_per_block>>>(
                hive_table_device,
                hive_stash_device,
                d_ops,
                num_ops,
                d_results
                #if BREAKDOWN_INSERT
                , d_insert_breakdown
                #endif
            );
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        if(!skip_clear)
        {
            //Reset stash atoms
            CUDA_CHECK(cudaMemsetAsync(&(hive_stash_device->head), 0, sizeof(KeyType), 0));
            CUDA_CHECK(cudaMemsetAsync(&(hive_stash_device->tail), 0, sizeof(KeyType), 0));
        }
    }

    //After warmup, clear the hash table and stash to ensure a clean benchmark run.
    if(!all_lookups && !all_deletes && !skip_clear)
        hiveHashTableClear(hive_table_device, hive_stash_device);
    
    float total_ms = 0;
    auto num_ops_with_stash = num_ops;
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for(size_t iter = 0; iter < numIterations; iter++)
    {
        CoarseGraindGPUTimer timer;
        if(all_lookups)
        {
            timer.start(stream);
            hive_lookup_kernel<TableType, HashPolicy><<<num_blocks, threads_per_block, 0, stream>>>(
                hive_table_device,
                hive_stash_device,
                d_ops,
                num_ops_with_stash,
                d_results
            );
            timer.stop(stream);
        }
        else{
            timer.start(stream);
            hive_mixed_kernel<TableType, HashPolicy><<<num_blocks, threads_per_block, 0, stream>>>(
                hive_table_device,
                hive_stash_device,
                d_ops,
                num_ops_with_stash,
                d_results
                #if BREAKDOWN_INSERT
                , d_insert_breakdown
                #endif
            );
            timer.stop(stream);
        }
        
        total_ms += timer.getElapsedTime();
        //Check load factor
        uint64_t* d_total_occupied;
        CUDA_CHECK(cudaMalloc(&d_total_occupied, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_total_occupied, 0, sizeof(uint64_t)));

        TableType hash_table_host_mirror;
        CUDA_CHECK(cudaMemcpy(&hash_table_host_mirror, hive_table_device, sizeof(TableType), cudaMemcpyDeviceToHost));

        count_occupancy_kernel<<<(hash_table_host_mirror.num_buckets + 512) / 512, 512, 0, stream>>>(hive_table_device, d_total_occupied);
        uint64_t h_total_occupied = 0;
        CUDA_CHECK(cudaMemcpy(&h_total_occupied, d_total_occupied, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        
        double lf = static_cast<double>(h_total_occupied) / (static_cast<double>(hash_table_host_mirror.num_buckets) * static_cast<double>(TableType::SLOTS));

        #if DYNAMIC_RESIZE
        std::cout << "Dynamic Resizing Check: " << lf<< std::endl;
        if constexpr (std::is_same<TableType, HiveHashTable>::value)
        {
            if(lf > 0.9)
            {
                std::cout << "Load factor exceeded 0.9, consider growing the hash table 2x" << std::endl;
                hive_hash_grow<HashPolicy>(hive_table_device, hash_table_host_mirror.num_buckets);
            }
            if(lf < 0.25)
            {
                std::cout << "Load factor below 0.25, consider shrinking the hash table" << std::endl;
                uint64_t target_occupied_slots = (h_total_occupied * 4) / 3; // target 75% occupancy
                uint64_t target_buckets = (target_occupied_slots + HIVE_BUCKET_SLOTS - 1) / HIVE_BUCKET_SLOTS;

                std::cout << "Target Buckets after shrink: " << target_buckets <<", Num Buckets: " << hash_table_host_mirror.num_buckets << std::endl;
                if(target_buckets < hash_table_host_mirror.num_buckets)
                {
                    size_t shrink_amount = hash_table_host_mirror.num_buckets - target_buckets;

                    //Limit shrink to at most 50%
                    shrink_amount = std::min(shrink_amount, hash_table_host_mirror.num_buckets - 1);

                    std::cout << "Shrinking hash table from " << hash_table_host_mirror.num_buckets << " to " << target_buckets << " buckets" << std::endl;
                    
                    hive_hash_shrink(hive_table_device, hive_stash_device, shrink_amount);
                }

            }   
        }
        #endif

        //include stash on next batch
        using MaxCapacityType = typename HiveOverflowStash<KeyType, ValueType>::max_cap_type;

        if(!skip_clear)
        {
            auto exec = thrust::cuda::par.on(stream);

            HiveOverflowStash<KeyType, ValueType>stash_mirror;
            CUDA_CHECK(cudaMemcpyAsync(&stash_mirror, hive_stash_device, sizeof(HiveOverflowStash<KeyType, ValueType>), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            //Why on multiple iteration stash_size remains same?
            MaxCapacityType stash_size = stash_mirror.tail - stash_mirror.head;
            
            if(stash_size > 0)
            {
                std::cout << "Copy Stash Size: " << stash_size << " to next batch iteration" << std::endl;
                //Copy from stash to d_ops
                thrust::transform(
                    exec,
                    thrust::counting_iterator<size_t>(0),
                    thrust::counting_iterator<size_t>(stash_size),
                    d_ops + num_ops,
                    [hive_stash_device] __device__ (size_t idx)
                    {
                        return Operation{
                            OperationType::INSERT,
                            static_cast<uint64_t>(hive_stash_device->keys[(hive_stash_device->head + idx) % static_cast<MaxCapacityType>(hive_stash_device->capacity)])
                        };                
                    }
                );

                num_ops_with_stash = num_ops + stash_size;
            }        
            else
            {
                num_ops_with_stash = num_ops;
            }

            //Reset stash pointers
            cudaMemsetAsync(&(hive_stash_device->head), 0, sizeof(KeyType), stream);
            cudaMemsetAsync(&(hive_stash_device->tail), 0, sizeof(KeyType), stream);    
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    elapsed_time = total_ms / numIterations;

    std::cout << "Average Time over " << numIterations << " iterations: " << elapsed_time << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_results, d_results, sizeof(uint64_t) * num_ops, cudaMemcpyDeviceToHost));
    

    size_t unsuccessful_ops = 0;
    size_t successful_inserts = 0;
    size_t successful_lookups = 0;
    size_t successful_deletes = 0;
    size_t successful_atomic_incs = 0;

    if (verify)
    {
        Operation* h_ops = new Operation[num_ops];
        CUDA_CHECK(cudaMemcpy(h_ops, d_ops, sizeof(Operation) * num_ops, cudaMemcpyDeviceToHost));

        //Verify results
        unsuccessful_ops = std::count(h_results, h_results+num_ops, 0);
        successful_inserts = std::count_if(h_ops, h_ops + num_ops, [h_ops, h_results](const Operation& op) {
            size_t idx = &op - h_ops;
            return op.type == OperationType::INSERT && h_results[idx] != 0;
        });
        successful_lookups = std::count_if(h_ops, h_ops + num_ops, [h_ops, h_results](const Operation& op) {
            size_t idx = &op - h_ops;
            return op.type == OperationType::LOOKUP && h_results[idx] != 0;
        });
        successful_deletes = std::count_if(h_ops, h_ops + num_ops, [h_ops, h_results](const Operation& op) {
            size_t idx = &op - h_ops;
            return op.type == OperationType::DELETE && h_results[idx] != 0;
        });
        successful_atomic_incs = std::count_if(h_ops, h_ops + num_ops, [h_ops, h_results](const Operation& op) {
            size_t idx = &op - h_ops;
            return op.type == OperationType::ATOMIC_INC && h_results[idx] != 0;
        });

        std::cout << "Verification Successful" << std::endl;

        // std::cout << "Unsuccessful ops: " << unsuccessful_ops << " out of " << num_ops <<", Success Rate: " << (1.0 - (unsuccessful_ops / static_cast<float>(num_ops))) * 100 << "%" << std::endl;
        // std::cout << "Successful Inserts: " << successful_inserts << std::endl;
        // std::cout << "Successful Lookups: " << successful_lookups << std::endl;
        // std::cout << "Successful Deletes: " << successful_deletes << std::endl;
        // std::cout << "Successful Atomic Increments: " << successful_atomic_incs << std::endl;

    }

    #if BREAKDOWN_INSERT
    Operation* h_ops = new Operation[num_ops];
    CUDA_CHECK(cudaMemcpy(h_ops, d_ops, sizeof(Operation) * num_ops, cudaMemcpyDeviceToHost));

    //Copy back insert breakdown
    CUDA_CHECK(cudaMemcpy(h_insert_breakdown.data(), d_insert_breakdown, num_ops * sizeof(InsertBreakdown), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_insert_breakdown));
    
    //Aggregate breakdown
    double total_stageA = 0;
    double total_stageB = 0;
    double total_stageC = 0;
    double total_stageD = 0;
    for(size_t i=0; i<num_ops; i++)
    {
        if(h_ops[i].type == OperationType::INSERT)
        {
            total_stageA += h_insert_breakdown[i].stageA;
            total_stageB += h_insert_breakdown[i].stageB;
            total_stageC += h_insert_breakdown[i].stageC;
            total_stageD += h_insert_breakdown[i].stageD;
        }
    }

    size_t num_inserts = std::count_if(h_ops, h_ops + num_ops, [](const Operation& op) {
        return op.type == OperationType::INSERT;
    });

    int clock_rate_khz = 0;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
    
    double stageA_elapsed_time_per_insert_ms = (total_stageA / static_cast<double>(clock_rate_khz)) / 4.0;
    double stageB_elapsed_time_per_insert_ms = (total_stageB / static_cast<double>(clock_rate_khz)) / 4.0;
    double stageC_elapsed_time_per_insert_ms = (total_stageC / static_cast<double>(clock_rate_khz)) / 4.0;
    double stageD_elapsed_time_per_insert_ms = (total_stageD / static_cast<double>(clock_rate_khz)) / 4.0;

    double total_insert_time_per_insert_ms = stageA_elapsed_time_per_insert_ms + stageB_elapsed_time_per_insert_ms + stageC_elapsed_time_per_insert_ms + stageD_elapsed_time_per_insert_ms;

    if (total_insert_time_per_insert_ms > 0)
    {
        std::cout << "Insert Breakdown (in ms):" << std::endl;
        std::cout << "Stage A (Try Replace Path): " << stageA_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (stageA_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) << "%" << std::endl;
        std::cout << "Stage B (Claim And Commit Path): " << stageB_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (stageB_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) << "%" << std::endl;
        std::cout << "Stage C (Cuckoo Eviction Path): " << stageC_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (stageC_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) << "%" << std::endl;
        std::cout << "Stage D (Stash Path): " << stageD_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (stageD_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) << "%" << std::endl;
    }

    #endif

    CUDA_CHECK(cudaFree(d_results));
}

void hash_table_kernel_dispatch_YCSB(
    Operation* h_prefill_ops,
    Operation* h_workload_ops,
    size_t num_prefill_ops,
    size_t num_workload_ops,
    size_t table_size,
    size_t threads_per_block,
    size_t numIterations,
    double &elapsed_time,
    uint64_t* h_results, //for lookup results
    bool verify,
    HashTableDataLayout layout
)
{
    const size_t stash_capacity = max_bucket_and_stash_caps.second / (sizeof(KeyType) * 2); //key-value pairs

    HiveOverflowStash<KeyType, ValueType> hive_stash;
    HiveOverflowStash<KeyType, ValueType>* hive_stash_device;
    CUDA_CHECK(cudaMalloc(&hive_stash_device, sizeof(HiveOverflowStash<KeyType, ValueType>)));

    //Fill up device ops
    Operation* d_prefill_ops;
    CUDA_CHECK(cudaMalloc(&d_prefill_ops, sizeof(Operation) * num_prefill_ops));

    thrust::fill(
        thrust::device,
        thrust::device_pointer_cast(d_prefill_ops),
        thrust::device_pointer_cast(d_prefill_ops) + num_prefill_ops,
        Operation{OperationType::NONE, 0}
    );

    CUDA_CHECK(cudaMemcpy(d_prefill_ops, h_prefill_ops, sizeof(Operation) * num_prefill_ops, cudaMemcpyHostToDevice));

    Operation* d_workload_ops;
    CUDA_CHECK(cudaMalloc(&d_workload_ops, sizeof(Operation) * (num_workload_ops + stash_capacity)));

    thrust::fill(
        thrust::device,
        thrust::device_pointer_cast(d_workload_ops),
        thrust::device_pointer_cast(d_workload_ops) + (num_workload_ops + stash_capacity),
        Operation{OperationType::NONE, 0}
    );

    CUDA_CHECK(cudaMemcpy(d_workload_ops, h_workload_ops, sizeof(Operation) * num_workload_ops, cudaMemcpyHostToDevice));

    generateLookupTables();

    auto compute_num_blocks = [threads_per_block](size_t num_ops){
        size_t n_blocks = (num_ops + threads_per_block - 1) / threads_per_block;
        return n_blocks;
    };

    size_t num_blocks_prefill = compute_num_blocks(num_prefill_ops);
    size_t num_blocks = compute_num_blocks(num_workload_ops);

    std::cout<<"Num Blocks: "<<num_blocks<<", Threads per Block: "<<threads_per_block<<std::endl;

    if(layout == HashTableDataLayout::HYBRID_SOA_AOS)
    {
        HiveHashTable hive_table;
        HiveHashTable* hive_table_device;
        CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTable)));

        const auto slots = HIVE_BUCKET_SLOTS;
        const size_t num_buckets = table_size / slots;

        const size_t max_num_buckets = max_bucket_and_stash_caps.first / (
            sizeof(HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>) /* Size of HiveBucketBody */
            + sizeof(uint32_t) /* Size of freeMask */
            + sizeof(uint16_t) /* Size of lock */
        );

        std::cout << "Creating Hive Hash Table with " << num_buckets << " buckets and stash capacity " << stash_capacity << " buckets" << std::endl;
        hiveHashTableCreate(
            hive_table, 
            hive_table_device, 
            hive_stash, 
            hive_stash_device, 
            num_buckets, 
            max_num_buckets, 
            8 /*max_evictions*/, 
            (bool)STASH_ENABLED /*stash_enabled*/, 
            stash_capacity /*stash_capacity*/
        );

        //Launch prefill kernel
        std::cout << "Prefill the hash table with " << num_prefill_ops << " insert operations" << std::endl;
        hive_mixed_kernel<HiveHashTable, Default2HashPolicy><<<num_blocks_prefill, threads_per_block>>>(
            hive_table_device,
            hive_stash_device,
            d_prefill_ops,
            num_prefill_ops,
            nullptr
            #if BREAKDOWN_INSERT
            , nullptr
            #endif
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "Running workload with " << num_workload_ops << " operations" << std::endl;
        run_hash_table_benchmark(
            hive_table_device,
            hive_stash_device,
            d_workload_ops,
            num_workload_ops,
            stash_capacity,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            false,
            false,
            true
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device,
            hive_stash,
            hive_stash_device
        );
    }
    else if(layout == HashTableDataLayout::ARRAY_OF_ALIGNED_STRUCTS)
    {
        std::cout << "Hive Hash Table With Array of aligned structures Layout" << std::endl;

        // Handle ARRAY_OF_ALIGNED_STRUCTS layout
        HiveHashTableAoaS<KVType> hive_table;
        HiveHashTableAoaS<KVType>* hive_table_device;
        CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTableAoaS<KVType>)));
        const auto slots = HIVE_BUCKET_SLOTS;
        const size_t num_buckets = table_size / slots;

        const size_t max_num_buckets = max_bucket_and_stash_caps.first / (sizeof(HiveBucketAoaS<KVType>));

        std::cout << "Creating Hive Hash Table (AoaS) with " << num_buckets << " buckets and stash capacity " << stash_capacity << " buckets" << std::endl;
        hiveHashTableCreate(
            hive_table,
            hive_table_device,
            hive_stash,
            hive_stash_device,
            num_buckets,
            max_num_buckets,
            8 /*max_evictions*/,
            (bool)STASH_ENABLED /*stash_enabled*/,
            stash_capacity /*stash_capacity*/
        );

        //Launch prefill kernel
        hive_mixed_kernel<HiveHashTableAoaS<KVType>, Default2HashPolicy><<<num_blocks_prefill, threads_per_block>>>(
            hive_table_device,
            hive_stash_device,
            d_prefill_ops,
            num_prefill_ops,
            nullptr
            #if BREAKDOWN_INSERT
            , nullptr
            #endif
        );

        run_hash_table_benchmark(
            hive_table_device,
            hive_stash_device,
            d_workload_ops,
            num_workload_ops,
            stash_capacity,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            false,
            false,
            true
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device,
            hive_stash,
            hive_stash_device
        );
    }
    else if (layout == HashTableDataLayout::ARRAY_OF_ALIGNED_STRUCTS_LEAD_METADATA)
    {
        std::cout << "Hive Hash Table With Array of Aligned Structures Leading Metadata Layout" << std::endl;

        HiveHashTableAoaS_LeadMetaData<KVType> hive_table;
        HiveHashTableAoaS_LeadMetaData<KVType>* hive_table_device;
        CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTableAoaS_LeadMetaData<KVType>)));

        const auto slots = HIVE_BUCKET_SLOTS;
        const size_t num_buckets = table_size / slots;

        const size_t max_num_buckets = max_bucket_and_stash_caps.first / (sizeof(HiveBucketAoaS_LeadMetaData<KVType>));

        hiveHashTableCreate(
            hive_table,
            hive_table_device,
            hive_stash,
            hive_stash_device,
            num_buckets,
            max_num_buckets,
            32 /*max_evictions*/,
            (bool)STASH_ENABLED /*stash_enabled*/,
            stash_capacity /*stash_capacity*/
        );

        //Launch prefill kernel
        hive_mixed_kernel<HiveHashTableAoaS_LeadMetaData<KVType>, Default2HashPolicy><<<num_blocks_prefill, threads_per_block>>>(
            hive_table_device,
            hive_stash_device,
            d_prefill_ops,
            num_prefill_ops,
            nullptr
            #if BREAKDOWN_INSERT
            , nullptr
            #endif
        );

        run_hash_table_benchmark(
            hive_table_device,
            hive_stash_device,
            d_workload_ops,
            num_workload_ops,
            stash_capacity,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            false,
            false,
            true
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device,
            hive_stash,
            hive_stash_device
        );

    }

    CUDA_CHECK(cudaFree(d_prefill_ops));
    CUDA_CHECK(cudaFree(d_workload_ops));
}

template <typename HashPolicy>
void hash_table_kernel_dispatch_t(
    Operation* h_ops,
    size_t num_ops,
    size_t table_size,
    size_t threads_per_block,
    size_t num_blocks,
    size_t numIterations,
    double &elapsed_time,
    uint64_t* h_results, //for lookup results
    bool verify,
    HashTableDataLayout layout
)
{
    //stash table
    const size_t stash_capacity = max_bucket_and_stash_caps.second / (sizeof(uint32_t) * 2); //key-value pairs

    HiveOverflowStash<KeyType, ValueType> hive_stash;
    HiveOverflowStash<KeyType, ValueType>* hive_stash_device;
    CUDA_CHECK(cudaMalloc(&hive_stash_device, sizeof(HiveOverflowStash<KeyType, ValueType>)));

    //Fill up device ops
    Operation* d_ops;
    CUDA_CHECK(cudaMalloc(&d_ops, sizeof(Operation) * (num_ops + stash_capacity)));
    
    thrust::fill(
        thrust::device,
        thrust::device_pointer_cast(d_ops),
        thrust::device_pointer_cast(d_ops) + num_ops + stash_capacity,
        Operation{OperationType::NONE, 0}
    );

    CUDA_CHECK(cudaMemcpy(d_ops, h_ops, sizeof(Operation) * num_ops, cudaMemcpyHostToDevice));

    //Generate lookup table for lookup-based hash functions
    generateLookupTables();

    std::vector<InsertBreakdown> h_insert_breakdown(num_ops, InsertBreakdown());
    InsertBreakdown* d_insert_breakdown = nullptr;
    #if BREAKDOWN_INSERT
    CUDA_CHECK(cudaMalloc(&d_insert_breakdown, num_ops * sizeof(InsertBreakdown)));
    CUDA_CHECK(cudaMemset(d_insert_breakdown, 0, num_ops * sizeof(InsertBreakdown)));
    #endif

    bool all_lookup = std::count_if(h_ops, h_ops + num_ops, [](const Operation& op) {
        return op.type == OperationType::LOOKUP;
    }) == num_ops;

    bool all_delete = std::count_if(h_ops, h_ops + num_ops, [](const Operation& op) {
        return op.type == OperationType::DELETE;
    }) == num_ops;

    Operation* d_prefill_ops = nullptr;

    if(all_lookup || all_delete)
    {
        std::cout << "All ops are lookups or deletes, starts prefilling the table.\n";

        std::vector<Operation> prefill_ops(num_ops);
        for(size_t i = 0; i < num_ops; i++)
        {
            prefill_ops[i] = {OperationType::INSERT, h_ops[i].key};
        }
        CUDA_CHECK(cudaMalloc(&d_prefill_ops, num_ops * sizeof(Operation)));
        CUDA_CHECK(cudaMemcpy(d_prefill_ops, prefill_ops.data(), num_ops * sizeof(Operation), cudaMemcpyHostToDevice));

    }


    if(layout == HashTableDataLayout::HYBRID_SOA_AOS)
    {
        HiveHashTable hive_table;
        HiveHashTable* hive_table_device;
        CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTable)));

        const auto slots = HIVE_BUCKET_SLOTS;
        const size_t num_buckets = table_size / slots;

        const size_t max_num_buckets = max_bucket_and_stash_caps.first / (
            sizeof(HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>) /* Size of HiveBucketBody */
            + sizeof(uint32_t) /* Size of freeMask */
            + sizeof(uint16_t) /* Size of lock */
        );

        std::cout << "Creating Hive Hash Table with " << num_buckets << " buckets and stash capacity " << stash_capacity << " buckets" << std::endl;
        hiveHashTableCreate(
            hive_table, 
            hive_table_device, 
            hive_stash, 
            hive_stash_device, 
            num_buckets, 
            max_num_buckets, 
            8 /*max_evictions*/, 
            (bool)STASH_ENABLED /*stash_enabled*/, 
            stash_capacity /*stash_capacity*/
        );

        if(all_lookup || all_delete)
        {
            //Launch prefill kernel
            hive_mixed_kernel<HiveHashTable, HashPolicy><<<num_blocks, threads_per_block>>>(
                hive_table_device,
                hive_stash_device,
                d_prefill_ops,
                num_ops,
                nullptr
                #if BREAKDOWN_INSERT
                , nullptr
                #endif
            );
        }

        run_hash_table_benchmark<HiveHashTable, HashPolicy>(
            hive_table_device,
            hive_stash_device,
            d_ops,
            num_ops,
            stash_capacity,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            all_delete,
            all_lookup,
            false
            #if BREAKDOWN_INSERT
            , d_insert_breakdown
            #endif
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device,
            hive_stash,
            hive_stash_device
        );
    }
    else if(layout == HashTableDataLayout::ARRAY_OF_ALIGNED_STRUCTS)
    {
        HiveHashTableAoaS<KVType> hive_table;
        HiveHashTableAoaS<KVType>* hive_table_device;
        CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTableAoaS<KVType>)));

        const auto slots = HIVE_BUCKET_SLOTS;
        const size_t num_buckets = table_size / slots;

        // For AoAS layout the per-bucket size equals sizeof(HiveBucketAoaS<KVType>).
        const size_t max_num_buckets = max_bucket_and_stash_caps.first / sizeof(HiveBucketAoaS<KVType>);

        std::cout << "Creating Hive Hash Table with " << num_buckets << " buckets and stash capacity " << stash_capacity << " buckets" << std::endl;
        hiveHashTableCreate(
            hive_table, 
            hive_table_device, 
            hive_stash, 
            hive_stash_device, 
            num_buckets, 
            max_num_buckets, 
            8 /*max_evictions*/, 
            (bool)STASH_ENABLED /*stash_enabled*/, 
            stash_capacity /*stash_capacity*/
        );

        if(all_lookup || all_delete)
        {
            //Launch prefill kernel
            hive_mixed_kernel<HiveHashTableAoaS<KVType>, HashPolicy><<<num_blocks, threads_per_block>>>(
                hive_table_device,
                hive_stash_device,
                d_prefill_ops,
                num_ops,
                nullptr
                #if BREAKDOWN_INSERT
                , nullptr
                #endif
            );
        }

        run_hash_table_benchmark<HiveHashTableAoaS<KVType>, HashPolicy>(
            hive_table_device,
            hive_stash_device,
            d_ops,
            num_ops,
            stash_capacity,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            all_delete,
            all_lookup,
            false
            #if BREAKDOWN_INSERT
            , d_insert_breakdown
            #endif
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device,
            hive_stash,
            hive_stash_device
        );
    }
    else if(layout == HashTableDataLayout::ARRAY_OF_ALIGNED_STRUCTS_LEAD_METADATA)
    {
        std::cout << "Hash Table with Lead Metadata Layout" << std::endl;
        
        HiveHashTableAoaS_LeadMetaData<KVType> hive_table;
        HiveHashTableAoaS_LeadMetaData<KVType>* hive_table_device;
        CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTableAoaS_LeadMetaData<KVType>)));

        const auto slots = HIVE_BUCKET_SLOTS;
        const size_t num_buckets = table_size / slots;

        // For AoAS layout the per-bucket size equals sizeof(HiveBucketAoaS<KVType>).
        const size_t max_num_buckets = max_bucket_and_stash_caps.first / sizeof(HiveBucketAoaS_LeadMetaData<KVType>);

        std::cout << "Creating Hive Hash Table with " << num_buckets << " buckets and stash capacity " << stash_capacity << " buckets" << std::endl;
        hiveHashTableCreate(
            hive_table, 
            hive_table_device, 
            hive_stash, 
            hive_stash_device, 
            num_buckets, 
            max_num_buckets, 
            8 /*max_evictions*/, 
            (bool)STASH_ENABLED /*stash_enabled*/, 
            stash_capacity /*stash_capacity*/
        );

        if(all_lookup || all_delete )
        {
            //Launch prefill kernel
            hive_mixed_kernel<HiveHashTableAoaS_LeadMetaData<KVType>, HashPolicy><<<num_blocks, threads_per_block>>>(
                hive_table_device,
                hive_stash_device,
                d_prefill_ops,
                num_ops,
                nullptr
                #if BREAKDOWN_INSERT
                , nullptr
                #endif
            );
        }

        run_hash_table_benchmark<HiveHashTableAoaS_LeadMetaData<KVType>, HashPolicy>(
            hive_table_device,
            hive_stash_device,
            d_ops,
            num_ops,
            stash_capacity,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            all_delete,
            all_lookup,
            false
            #if BREAKDOWN_INSERT
            , d_insert_breakdown
            #endif
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device,
            hive_stash,
            hive_stash_device
        );
    }

    CUDA_CHECK(cudaFree(d_ops));
    if(d_prefill_ops) CUDA_CHECK(cudaFree(d_prefill_ops));
}

void hash_table_kernel_dispatch(
    Operation* h_ops,
    size_t num_ops,
    size_t table_size,
    size_t threads_per_block,
    size_t num_blocks,
    size_t numIterations,
    double &elapsed_time,
    uint64_t* h_results, //for lookup results
    bool verify,
    HashTableDataLayout layout,
    std::string hash_policy
)
{
    if(hash_policy == "Default2Hash")
    {
        hash_table_kernel_dispatch_t<Default2HashPolicy>(
            h_ops, num_ops, table_size, threads_per_block, num_blocks, numIterations, elapsed_time, h_results, verify, layout
        );
    }
    else if(hash_policy == "TripleHash")
    {
        hash_table_kernel_dispatch_t<TripleHashPolicy>(
            h_ops, num_ops, table_size, threads_per_block, num_blocks, numIterations, elapsed_time, h_results, verify, layout
        );
    }
    else if(hash_policy == "MurmurCityHash")
    {
        hash_table_kernel_dispatch_t<MurmurCityHashPolicy>(
            h_ops, num_ops, table_size, threads_per_block, num_blocks, numIterations, elapsed_time, h_results, verify, layout
        );
    }
    else if(hash_policy == "MurmurCityBitHash")
    {
        hash_table_kernel_dispatch_t<MurmurCityBitHashPolicy>(
            h_ops, num_ops, table_size, threads_per_block, num_blocks, numIterations, elapsed_time, h_results, verify, layout
        );
    }
    else if(hash_policy == "Lookup2Hash")
    {
        hash_table_kernel_dispatch_t<Lookup2HashPolicy>(
            h_ops, num_ops, table_size, threads_per_block, num_blocks, numIterations, elapsed_time, h_results, verify, layout
        );
    }
    else if(hash_policy == "Lookup3Hash")
    {
        hash_table_kernel_dispatch_t<Lookup3HashPolicy>(
            h_ops, num_ops, table_size, threads_per_block, num_blocks, numIterations, elapsed_time, h_results, verify, layout
        );
    }
    else
    {
        std::cerr << "Unknown Hash Policy: " << hash_policy << ". Using Default2HashPolicy." << std::endl;
        hash_table_kernel_dispatch_t<Default2HashPolicy>(
            h_ops, num_ops, table_size, threads_per_block, num_blocks, numIterations, elapsed_time, h_results, verify, layout
        );
    }
}

void hive_launch_mix_ops_kernel(
    Operation* h_ops,
    size_t num_ops,
    size_t table_size,
    size_t threads_per_block,
    size_t num_blocks,
    size_t numIterations,
    double &elapsed_time,
    uint64_t* h_results, //for lookup results
    bool verify
)
{
    const size_t num_buckets = table_size / HIVE_BUCKET_SLOTS;

    const size_t max_num_buckets = max_bucket_and_stash_caps.first / (sizeof(HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>) + sizeof(uint32_t) + sizeof(uint16_t));

    const size_t stash_capacity = max_bucket_and_stash_caps.second / (sizeof(uint32_t) * 2); //key-value pairs

    //Device Table and Stash
    HiveHashTable hive_table;
    HiveHashTable* hive_table_device;
    CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTable)));
    HiveOverflowStash<KeyType, ValueType> hive_stash;
    HiveOverflowStash<KeyType, ValueType>* hive_stash_device;
    CUDA_CHECK(cudaMalloc(&hive_stash_device, sizeof(HiveOverflowStash<KeyType, ValueType>)));


    std::cout << "Creating Hive Hash Table with " << num_buckets << " buckets and stash capacity " << stash_capacity << " buckets" << std::endl;
    hiveHashTableCreate(
        hive_table, 
        hive_table_device, 
        hive_stash, 
        hive_stash_device, 
        num_buckets, 
        max_num_buckets, 
        8 /*max_evictions*/, 
        (bool)STASH_ENABLED /*stash_enabled*/, 
        stash_capacity /*stash_capacity*/
    );

    //Fill up device Ops
    Operation* d_ops;
    CUDA_CHECK(cudaMalloc(&d_ops, sizeof(Operation) * (num_ops + stash_capacity)));

    thrust::fill(
        thrust::device, 
        thrust::device_pointer_cast(d_ops), 
        thrust::device_pointer_cast(d_ops) + num_ops + stash_capacity, 
        Operation{OperationType::NONE, 0}
    );
    CUDA_CHECK(cudaMemcpy(d_ops, h_ops, sizeof(Operation) * (num_ops), cudaMemcpyHostToDevice));

    std::cout << "Launching mixed operations kernel with " << num_blocks << " blocks of " << threads_per_block << " threads..." << std::endl;
    
    //Device Results
    uint64_t* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(uint64_t) * (num_ops + stash_capacity)));
    CUDA_CHECK(cudaMemset(d_results, 0, sizeof(uint64_t) * (num_ops + stash_capacity)));

    //Generate Lookup Tables
    generateLookupTables();

    #if BREAKDOWN_INSERT
     //Breakdown for insert stages
    std::vector<InsertBreakdown> h_insert_breakdown(num_ops, InsertBreakdown());
    InsertBreakdown* d_insert_breakdown;
    CUDA_CHECK(cudaMalloc(&d_insert_breakdown, num_ops * sizeof(InsertBreakdown)));
    CUDA_CHECK(cudaMemset(d_insert_breakdown, 0, num_ops * sizeof(InsertBreakdown)));
    #endif

    bool all_lookup = std::count_if(h_ops, h_ops + num_ops, [](const Operation& op) {
        return op.type == OperationType::LOOKUP;
    }) == num_ops;


    if(all_lookup)
    {
        std::cout << "100% lookup workload detected, pre-filling the table with lookup keys..." << std::endl;
        std::vector<Operation> insert_ops(num_ops);
        for(size_t i=0; i<num_ops; i++)
        {
            insert_ops[i] = {OperationType::INSERT, h_ops[i].key};
        }
        std::cout <<"Insert Ops[0] : " << h_ops[10].key << std::endl;
        Operation* d_insert_ops;
        CUDA_CHECK(cudaMalloc(&d_insert_ops, sizeof(Operation) * num_ops));
        CUDA_CHECK(cudaMemcpy(d_insert_ops, insert_ops.data(), sizeof(Operation) * num_ops, cudaMemcpyHostToDevice));
        hive_mixed_kernel<<<num_blocks, threads_per_block>>>(
            hive_table_device,
            hive_stash_device,
            d_insert_ops,
            num_ops,
            d_results
            #if BREAKDOWN_INSERT
            , nullptr
            #endif
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        HiveHashTable hive_table_mirror;
        CUDA_CHECK(cudaMemcpy(&hive_table_mirror, hive_table_device, sizeof(HiveHashTable), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hive_table_device, &hive_table_mirror, sizeof(HiveHashTable), cudaMemcpyHostToDevice));
    
    }
    
    //Warmup
    for(int i=0; i<5; i++)
    {
        hive_mixed_kernel<<<num_blocks, threads_per_block>>>(
            hive_table_device,
            hive_stash_device,
            d_ops,
            num_ops,
            d_results
            #if BREAKDOWN_INSERT
            , nullptr
            #endif
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        //reset stash, but hive_stash_device->head & hive_stash_device->tail are atomics;
        CUDA_CHECK(cudaMemsetAsync(&(hive_stash_device->head), 0, sizeof(KeyType), 0));
        CUDA_CHECK(cudaMemsetAsync(&(hive_stash_device->tail), 0, sizeof(KeyType), 0));

    }
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float milliseconds = 0;

    auto num_ops_with_stash = num_ops;

    for(size_t iter=0; iter<numIterations; iter++)
    {
        CoarseGraindGPUTimer timer;
        timer.start(stream);
        hive_mixed_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            hive_table_device,
            hive_stash_device,
            d_ops,
            num_ops_with_stash,
            d_results
            #if BREAKDOWN_INSERT
            , d_insert_breakdown
            #endif
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop(stream);
        milliseconds += timer.getElapsedTime();

        HiveHashTable hive_table_dev_mirror;
        CUDA_CHECK(cudaMemcpy(&hive_table_dev_mirror, hive_table_device, sizeof(HiveHashTable), cudaMemcpyDeviceToHost));

        std::vector<uint32_t>free_mask_host(hive_table_dev_mirror.num_buckets);
        //Copy back freeMask
        CUDA_CHECK(cudaMemcpy(free_mask_host.data(), hive_table_dev_mirror.freeMask, sizeof(uint32_t) * hive_table_dev_mirror.num_buckets, cudaMemcpyDeviceToHost));

        uint64_t occupied_slots = 0;
        for(size_t i=0; i<hive_table_dev_mirror.num_buckets; i++)
        {
            uint32_t free_mask = free_mask_host[i];
            occupied_slots += __builtin_popcount(~free_mask);
        }
        double load_factor = static_cast<double>(occupied_slots) / static_cast<double>(hive_table_dev_mirror.num_buckets * HIVE_BUCKET_SLOTS);
        std::cout << "Load Factor after operations: " << load_factor * 100 << "%" << std::endl;

        #if DYNAMIC_RESIZE
        std::cout << "Dynamic Resizing Check: " << std::endl;
        if(load_factor > 0.9)
        {
            std::cout << "Load factor exceeded 90%, growing table 2x, number of buckets added: " << hive_table_dev_mirror.num_buckets << std::endl;
            hive_hash_grow(hive_table_device, hive_table_dev_mirror.num_buckets); //double the size
        }

        if(load_factor < 0.25)
        {
            std::cout << "Load factor below 40%, shrinking table 2x..." << std::endl;
            //hive_hash_shrink(hive_table_device, hive_table_dev_mirror.num_buckets/4); //half the size

            uint64_t target_buckets = (occupied_slots + HIVE_BUCKET_SLOTS - 1) / HIVE_BUCKET_SLOTS; // Round up
            target_buckets = (target_buckets * 4 + 2) / 3; // Target 75% load factor after shrink
            
            if(target_buckets < hive_table_dev_mirror.num_buckets)
            {
                size_t shrink_amount = hive_table_dev_mirror.num_buckets - target_buckets;
                
                // Limit shrink to at most 50% per iteration for stability
                shrink_amount = std::min(shrink_amount, hive_table_dev_mirror.num_buckets / 2);
                
                // Ensure we don't shrink below minimum size
                shrink_amount = std::min(shrink_amount, hive_table_dev_mirror.num_buckets - 1);
                
                std::cout << "Shrinking by " << shrink_amount << " buckets (from " 
                        << hive_table_dev_mirror.num_buckets << " to " 
                        << (hive_table_dev_mirror.num_buckets - shrink_amount) << ")" << std::endl;
                
                hive_hash_shrink(hive_table_device, hive_stash_device, shrink_amount);
            }
        }
        #endif

        //include stash on next batch
        using MaxCapacityType = typename HiveOverflowStash<KeyType, ValueType>::max_cap_type;

        auto exec = thrust::cuda::par.on(stream);

        HiveOverflowStash<KeyType, ValueType>stash_mirror;
        CUDA_CHECK(cudaMemcpyAsync(&stash_mirror, hive_stash_device, sizeof(HiveOverflowStash<KeyType, ValueType>), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        MaxCapacityType stash_size = stash_mirror.tail - stash_mirror.head;
        
        if(stash_size > 0)
        {
            std::cout << "Copy Stash Size: " << stash_size << " to next batch iteration" << std::endl;
            //Copy from stash to d_ops
            thrust::transform(
                exec,
                thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator<size_t>(stash_size),
                d_ops + num_ops,
                [hive_stash_device] __device__ (size_t idx)
                {
                    return Operation{
                        OperationType::INSERT,
                        static_cast<uint64_t>(hive_stash_device->keys[(hive_stash_device->head + idx) % static_cast<MaxCapacityType>(hive_stash_device->capacity)])
                    };                
                }
            );

            num_ops_with_stash = num_ops + stash_size;
        }        
        else
        {
            num_ops_with_stash = num_ops;
        }

        //Reset stash pointers
        cudaMemsetAsync(&(hive_stash_device->head), 0, sizeof(KeyType), stream);
        cudaMemsetAsync(&(hive_stash_device->tail), 0, sizeof(KeyType), stream);    

        
        // exec = thrust::cuda::par.on(stream);
        // auto head_ptr = thrust::device_pointer_cast(&(hive_stash_device->head));
        // auto tail_ptr = thrust::device_pointer_cast(&(hive_stash_device->tail));
        // thrust::fill(exec, head_ptr, head_ptr + 1, static_cast<uint64_t>(0));
        // thrust::fill(exec, tail_ptr, tail_ptr + 1, static_cast<uint64_t>(0));
        
    }
    // CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK(cudaEventRecord(stop));
    // CUDA_CHECK(cudaEventSynchronize(stop));
    // float milliseconds = 0;
    // CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    elapsed_time = milliseconds / numIterations;
    std::cout << "Average Time per Iteration: " << elapsed_time << " ms\n";

    //Memcpy hash table back to host for verification
    // Do not overwrite hive_table's host pointers; use a temporary mirror instead
    
    // CUDA_CHECK(cudaEventDestroy(start));
    // CUDA_CHECK(cudaEventDestroy(stop));

    //Copy back results
    
    CUDA_CHECK(cudaMemcpy(h_results, d_results, sizeof(uint64_t) * num_ops, cudaMemcpyDeviceToHost));
    

    size_t unsuccessful_ops = 0;
    size_t successful_inserts = 0;
    size_t successful_lookups = 0;
    size_t successful_deletes = 0;

    if (verify)
    {

        //Verify results
        unsuccessful_ops = std::count(h_results, h_results+num_ops, 0);
        successful_inserts = std::count_if(h_ops, h_ops + num_ops, [h_ops, h_results](const Operation& op) {
            size_t idx = &op - h_ops;
            return op.type == OperationType::INSERT && h_results[idx] != 0;
        });
        successful_lookups = std::count_if(h_ops, h_ops + num_ops, [h_ops, h_results](const Operation& op) {
            size_t idx = &op - h_ops;
            return op.type == OperationType::LOOKUP && h_results[idx] != 0;
        });
        successful_deletes = std::count_if(h_ops, h_ops + num_ops, [h_ops, h_results](const Operation& op) {
            size_t idx = &op - h_ops;
            return op.type == OperationType::DELETE && h_results[idx] != 0;
        });

        std::cout << "Verification Successful" << std::endl;
    }

    #if BREAKDOWN_INSERT
    //Copy back insert breakdown
    CUDA_CHECK(cudaMemcpy(h_insert_breakdown.data(), d_insert_breakdown, num_ops * sizeof(InsertBreakdown), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_insert_breakdown));
    
    //Aggregate breakdown
    double total_stageA = 0;
    double total_stageB = 0;
    double total_stageC = 0;
    double total_stageD = 0;
    for(size_t i=0; i<num_ops; i++)
    {
        if(h_ops[i].type == OperationType::INSERT)
        {
            total_stageA += h_insert_breakdown[i].stageA;
            total_stageB += h_insert_breakdown[i].stageB;
            total_stageC += h_insert_breakdown[i].stageC;
            total_stageD += h_insert_breakdown[i].stageD;
        }
    }

    size_t num_inserts = std::count_if(h_ops, h_ops + num_ops, [](const Operation& op) {
        return op.type == OperationType::INSERT;
    });

    int clock_rate_khz = 0;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
    
    double stageA_elapsed_time_per_insert_ms = (total_stageA / static_cast<double>(clock_rate_khz)) / 4.0;
    double stageB_elapsed_time_per_insert_ms = (total_stageB / static_cast<double>(clock_rate_khz)) / 4.0;
    double stageC_elapsed_time_per_insert_ms = (total_stageC / static_cast<double>(clock_rate_khz)) / 4.0;
    double stageD_elapsed_time_per_insert_ms = (total_stageD / static_cast<double>(clock_rate_khz)) / 4.0;

    double total_insert_time_per_insert_ms = stageA_elapsed_time_per_insert_ms + stageB_elapsed_time_per_insert_ms + stageC_elapsed_time_per_insert_ms + stageD_elapsed_time_per_insert_ms;

    if (total_insert_time_per_insert_ms > 0)
    {
        std::cout << "Insert Breakdown (in ms):" << std::endl;
        std::cout << "Stage A (Try Replace Path): " << stageA_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (stageA_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) << "%" << std::endl;
        std::cout << "Stage B (Claim And Commit Path): " << stageB_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (stageB_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) << "%" << std::endl;
        std::cout << "Stage C (Cuckoo Eviction Path): " << stageC_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (stageC_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) << "%" << std::endl;
        std::cout << "Stage D (Stash Path): " << stageD_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (stageD_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) << "%" << std::endl;
    }

    #endif

    //Destroy tables
    hiveHashTableDestroy(hive_table, hive_table_device, hive_stash, hive_stash_device);
}


void hiveStashTableCreate(
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device,
    size_t stash_capacity,
    bool stash_enabled
){
    //Same Pattern for stashing
    uint32_t *d_stash_keys = nullptr;
    uint32_t *d_stash_values = nullptr;
    

    HiveOverflowStash<KeyType, ValueType> stash_dev_mirror;

    if(stash_enabled)
    {
        //Allocate device buffers for stash
        CUDA_CHECK(cudaMalloc(&d_stash_keys, sizeof(uint32_t) * stash_capacity));
        CUDA_CHECK(cudaMalloc(&d_stash_values, sizeof(uint32_t) * stash_capacity));
        
        //Manually assign members to avoid atomic assignment issues
        stash_dev_mirror.keys = d_stash_keys;
        stash_dev_mirror.values = d_stash_values;
        stash_dev_mirror.head.store(0, cuda::memory_order_relaxed);
        stash_dev_mirror.tail.store(0, cuda::memory_order_relaxed);
        stash_dev_mirror.capacity = static_cast<uint32_t>(stash_capacity);
        stash_dev_mirror.enabled = true;
        
        stash_host.keys = new uint32_t[stash_capacity];
        stash_host.values = new uint32_t[stash_capacity];
        stash_host.head.store(0, cuda::memory_order_relaxed);
        stash_host.tail.store(0, cuda::memory_order_relaxed);
        stash_host.capacity = static_cast<uint32_t>(stash_capacity);
        stash_host.enabled = true;
    }
    else
    {
        stash_dev_mirror.keys = nullptr;
        stash_dev_mirror.values = nullptr;
        stash_dev_mirror.head.store(0, cuda::memory_order_relaxed);
        stash_dev_mirror.tail.store(0, cuda::memory_order_relaxed);
        stash_dev_mirror.capacity = 0;
        stash_dev_mirror.enabled = false;

        stash_host.keys = nullptr;
        stash_host.values = nullptr;
        stash_host.head.store(0, cuda::memory_order_relaxed);
        stash_host.tail.store(0, cuda::memory_order_relaxed);
        stash_host.capacity = 0;
        stash_host.enabled = false;
    }

    //Copy the device struct to device memory
    CUDA_CHECK(cudaMemcpy(stash_device, &stash_dev_mirror, sizeof(HiveOverflowStash <KeyType, ValueType>), cudaMemcpyHostToDevice));
}

void hiveHashTableCreate(
    HiveHashTable& table_host,
    HiveHashTable* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device,
    size_t num_buckets,
    size_t max_num_buckets,
    size_t max_evictions,
    bool stash_enabled,
    size_t stash_capacity
)
{
    //Allocate device buffers (poiinters live on host variable)
    HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>* d_bodies = nullptr;
    uint32_t* d_freeMask = nullptr;
    uint16_t* d_lock = nullptr;
    // uint8_t* d_tags = nullptr;

    CUDA_CHECK(cudaMalloc(&d_bodies, sizeof(HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>) * max_num_buckets));
    CUDA_CHECK(cudaMalloc(&d_freeMask, sizeof(uint32_t) * max_num_buckets));
    CUDA_CHECK(cudaMalloc(&d_lock, sizeof(uint16_t) * max_num_buckets));

    num_buckets = std::min(num_buckets, max_num_buckets);

    CUDA_CHECK(cudaMemset(d_bodies, 0, sizeof(HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>) * num_buckets));
    CUDA_CHECK(cudaMemset(d_lock, 0, sizeof(uint16_t) * num_buckets));

    //init free bits into mask
    // std::vector<uint32_t> h_freeMask(num_buckets, (HIVE_BUCKET_SLOTS >= 32) ? 0xFFFFFFFFU : ((1ULL << HIVE_BUCKET_SLOTS) - 1));
    table_host.freeMask = new uint32_t[max_num_buckets];
    std::fill(table_host.freeMask, table_host.freeMask + num_buckets, (HIVE_BUCKET_SLOTS >= 32) ? 0xFFFFFFFFU : ((1ULL << HIVE_BUCKET_SLOTS) - 1));
    if (max_num_buckets > num_buckets) {
        std::fill(table_host.freeMask + num_buckets, table_host.freeMask + max_num_buckets, 0); // Zero out extra buckets
    }

    CUDA_CHECK(cudaMemcpy(d_freeMask, table_host.freeMask, sizeof(uint32_t) * num_buckets, cudaMemcpyHostToDevice));

    //Build host mirror of the device struct and then mirror it
    HiveHashTable table_dev_mirror = {
        d_bodies,
        d_freeMask,
        d_lock,
        num_buckets,
        max_num_buckets,
        max_evictions
        #if DYNAMIC_RESIZE
        , static_cast<uint32_t>((1U << (32 - __builtin_clz(num_buckets - 1))) - 1), //index_mask
        0 //split_ptr
        #endif
    };

    //Copy the device struct to device memory
    CUDA_CHECK(cudaMemcpy(table_device, &table_dev_mirror, sizeof(HiveHashTable), cudaMemcpyHostToDevice));

    //Initialize host side hash table mirror
    table_host.num_buckets = num_buckets;
    table_host.max_num_buckets = max_num_buckets;

    table_host.max_evictions = max_evictions;
 
    table_host.buckets = new HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>[max_num_buckets];
    // table_host.freeMask = h_freeMask.data();
    table_host.lock = new uint16_t[max_num_buckets]();
    table_host.index_mask = table_dev_mirror.index_mask;
    table_host.split_ptr = 0;

    hiveStashTableCreate(
        stash_host,
        stash_device,
        stash_capacity,
        stash_enabled
    );

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
void hiveHashTableCreate(
    HiveHashTableAoaS_LeadMetaData<KVType>& table_host,
    HiveHashTableAoaS_LeadMetaData<KVType>* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device,
    size_t num_buckets,
    size_t max_num_buckets,
    size_t max_evictions,
    bool stash_enabled,
    size_t stash_capacity
)
{
    HiveBucketAoaS_LeadMetaData<KVType>* d_buckets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buckets, sizeof(HiveBucketAoaS_LeadMetaData<KVType>) * max_num_buckets));

    size_t alloc_buckets = std::min(num_buckets, max_num_buckets);

    uint32_t initial_mask = (HIVE_BUCKET_SLOTS_AOAS >= 32) ? 0xFFFFFFFFU : ((1ULL << HIVE_BUCKET_SLOTS_AOAS) - 1);

    initAoaSBucket<<<(alloc_buckets + 255) / 256, 256>>>(d_buckets, alloc_buckets, initial_mask);
    CUDA_CHECK(cudaDeviceSynchronize());

    //Build Host Mirror
    HiveHashTableAoaS_LeadMetaData<KVType> table_dev_mirror = {
        d_buckets,
        alloc_buckets,
        max_num_buckets,
        max_evictions
        #if DYNAMIC_RESIZE
        , static_cast<uint32_t>((1U << (32 - __builtin_clz(alloc_buckets - 1))) - 1), //index_mask
        0 //split_ptr
        #endif
    };

    CUDA_CHECK(cudaMemcpy(table_device, &table_dev_mirror, sizeof(HiveHashTableAoaS_LeadMetaData<KVType>), cudaMemcpyHostToDevice));

    table_host.num_buckets = alloc_buckets;
    table_host.max_num_buckets = max_num_buckets;
    table_host.max_evictions = max_evictions;
    
    table_host.buckets = new HiveBucketAoaS_LeadMetaData<KVType>[max_num_buckets];
    table_host.index_mask = table_dev_mirror.index_mask;
    table_host.split_ptr = table_dev_mirror.split_ptr;

    hiveStashTableCreate(
        stash_host,
        stash_device,
        stash_capacity,
        stash_enabled
    );
   
}

// The actual implementation used for AoAS buckets (body continues after $SELECTION_PLACEHOLDER$).
void hiveHashTableCreate(
    HiveHashTableAoaS<KVType>& table_host,
    HiveHashTableAoaS<KVType>* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device,
    size_t num_buckets,
    size_t max_num_buckets,
    size_t max_evictions,
    bool stash_enabled,
    size_t stash_capacity
)
{
    HiveBucketAoaS<KVType>* d_buckets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buckets, sizeof(HiveBucketAoaS<KVType>) * max_num_buckets));
    
    size_t alloc_buckets = std::min(num_buckets, max_num_buckets);

    uint32_t initial_mask = (HIVE_BUCKET_SLOTS_AOAS >= 32) ? 0xFFFFFFFFU : ((1ULL << HIVE_BUCKET_SLOTS_AOAS) - 1);

    initAoaSBucket<<<(alloc_buckets + 255) / 256, 256>>>(d_buckets, alloc_buckets, initial_mask);
    CUDA_CHECK(cudaDeviceSynchronize());

    //Build Host Mirror
    HiveHashTableAoaS<KVType> table_dev_mirror = {
        d_buckets,
        alloc_buckets,
        max_num_buckets,
        max_evictions
        #if DYNAMIC_RESIZE
        , static_cast<uint32_t>((1U << (32 - __builtin_clz(alloc_buckets - 1))) - 1), //index_mask
        0 //split_ptr
        #endif
    };

    CUDA_CHECK(cudaMemcpy(table_device, &table_dev_mirror, sizeof(HiveHashTableAoaS<KVType>), cudaMemcpyHostToDevice));

    table_host.num_buckets = alloc_buckets;
    table_host.max_num_buckets = max_num_buckets;
    table_host.max_evictions = max_evictions;
    
    table_host.buckets = new HiveBucketAoaS<KVType>[max_num_buckets];
    table_host.index_mask = table_dev_mirror.index_mask;
    table_host.split_ptr = table_dev_mirror.split_ptr;

    hiveStashTableCreate(
        stash_host,
        stash_device,
        stash_capacity,
        stash_enabled
    );

}


void hiveHashTableDestroy(
    HiveHashTable& table_host, HiveHashTable* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host, HiveOverflowStash<KeyType, ValueType>* stash_device)
{
    std::cout << "Destroying Hive Hash Table..." << std::endl;
    //Pull back device data to host
    HiveHashTable table_dev_mirror{};
    CUDA_CHECK(cudaMemcpy(&table_dev_mirror, table_device, sizeof(HiveHashTable), cudaMemcpyDeviceToHost));

    //Free device inner buffers
    if(table_dev_mirror.buckets)
        CUDA_CHECK(cudaFree(table_dev_mirror.buckets));
    if(table_dev_mirror.freeMask)
        CUDA_CHECK(cudaFree(table_dev_mirror.freeMask));
    if(table_dev_mirror.lock)
        CUDA_CHECK(cudaFree(table_dev_mirror.lock));
    

    HiveOverflowStash<KeyType, ValueType> stash_dev_mirror{};
    CUDA_CHECK(cudaMemcpy(&stash_dev_mirror, stash_device, sizeof(HiveOverflowStash<KeyType, ValueType>), cudaMemcpyDeviceToHost));

    //Free device inner buffers
    if(stash_dev_mirror.keys)
        CUDA_CHECK(cudaFree(stash_dev_mirror.keys));
    if(stash_dev_mirror.values)
        CUDA_CHECK(cudaFree(stash_dev_mirror.values));

    //Free host inner buffers
    CUDA_CHECK(cudaFree(table_device));
    CUDA_CHECK(cudaFree(stash_device));


    //Free host mirror
    delete[] table_host.buckets;
    delete[] table_host.lock;
    delete[] table_host.freeMask;
    
    if(stash_host.enabled)
    {
        delete[] stash_host.keys;
        delete[] stash_host.values;
    }
}

void hiveHashTableClear(
    HiveHashTable* table_device,
    HiveOverflowStash<KeyType, ValueType>* stash_device
)
{
    HiveHashTable table_mirror;
    CUDA_CHECK(cudaMemcpy(&table_mirror, table_device, sizeof(HiveHashTable), cudaMemcpyDeviceToHost));

    // Reset buckets and locks to 0
    CUDA_CHECK(cudaMemset(table_mirror.buckets, 0, sizeof(HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>) * table_mirror.num_buckets));
    CUDA_CHECK(cudaMemset(table_mirror.lock, 0, sizeof(uint16_t) * table_mirror.num_buckets));

    // Reset freeMask to all 1s
    const uint32_t full_mask = (HIVE_BUCKET_SLOTS < 32) ? ((1u << HIVE_BUCKET_SLOTS) - 1u) : 0xFFFFFFFFu;
    std::vector<uint32_t> h_full_mask(table_mirror.num_buckets, full_mask);
    CUDA_CHECK(cudaMemcpy(table_mirror.freeMask, h_full_mask.data(), sizeof(uint32_t) * table_mirror.num_buckets, cudaMemcpyHostToDevice));

    // Reset stash pointers
    CUDA_CHECK(cudaMemset(stash_device, 0, sizeof(HiveOverflowStash<KeyType, ValueType>)));
}

void hiveHashTableClear(
    HiveHashTableAoaS<KVType>* table_device,
    HiveOverflowStash<KeyType, ValueType>* stash_device
)
{
    HiveHashTableAoaS<KVType> table_mirror;
    CUDA_CHECK(cudaMemcpy(&table_mirror, table_device, sizeof(HiveHashTableAoaS<KVType>), cudaMemcpyDeviceToHost));
    
    uint32_t initial_mask = (HIVE_BUCKET_SLOTS_AOAS >= 32) ? 0xFFFFFFFFU : ((1ULL << HIVE_BUCKET_SLOTS_AOAS) - 1);
    initAoaSBucket<<<(table_mirror.num_buckets + 255) / 256, 256>>>(table_mirror.buckets, table_mirror.num_buckets, initial_mask);
    
    // Reset stash pointers
    CUDA_CHECK(cudaMemset(stash_device, 0, sizeof(HiveOverflowStash<KeyType, ValueType>)));
}

void hiveHashTableClear(
    HiveHashTableAoaS_LeadMetaData<KVType>* table_device,
    HiveOverflowStash<KeyType, ValueType>* stash_device
)
{
    HiveHashTableAoaS_LeadMetaData<KVType> table_mirror;
    CUDA_CHECK(cudaMemcpy(&table_mirror, table_device, sizeof(HiveHashTableAoaS_LeadMetaData<KVType>), cudaMemcpyDeviceToHost));
    
    uint32_t initial_mask = (HIVE_BUCKET_SLOTS_AOAS >= 32) ? 0xFFFFFFFFU : ((1ULL << HIVE_BUCKET_SLOTS_AOAS) - 1);
    initAoaSBucket<<<(table_mirror.num_buckets + 255) / 256, 256>>>(table_mirror.buckets, table_mirror.num_buckets, initial_mask);
    
    // Reset stash pointers
    CUDA_CHECK(cudaMemset(stash_device, 0, sizeof(HiveOverflowStash<KeyType, ValueType>)));
}


void hiveHashTableDestroy(
    HiveHashTableAoaS<KVType>& table_host,
    HiveHashTableAoaS<KVType>* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device
)
{
    std::cout << "Destroying Hive Hash Table AoaS..." << std::endl;
    //Pull back device data to host
    HiveHashTableAoaS<KVType> table_dev_mirror{};
    CUDA_CHECK(cudaMemcpy(&table_dev_mirror, table_device, sizeof(HiveHashTableAoaS<KVType>), cudaMemcpyDeviceToHost));

    //Free device inner buffers
    if(table_dev_mirror.buckets)
        CUDA_CHECK(cudaFree(table_dev_mirror.buckets));

    // HiveOverflowStash<KeyType, ValueType> stash_dev_mirror{};
    // CUDA_CHECK(cudaMemcpy(&stash_dev_mirror, stash_device, sizeof(HiveOverflowStash<KeyType, ValueType>), cudaMemcpyDeviceToHost));

    // //Free device inner buffers
    // if(stash_dev_mirror.keys)
    //     CUDA_CHECK(cudaFree(stash_dev_mirror.keys));
    // if(stash_dev_mirror.values)
    //     CUDA_CHECK(cudaFree(stash_dev_mirror.values));

    //Free host inner buffers
    CUDA_CHECK(cudaFree(table_device));
//    CUDA_CHECK(cudaFree(stash_device));

    if(table_host.buckets)
    {
        delete[] table_host.buckets;
        table_host.buckets = nullptr;
    }

    // if(stash_host.enabled)
    // {
    //     delete[] stash_host.keys;
    //     delete[] stash_host.values;
    // }
}

void hiveHashTableDestroy(
    HiveHashTableAoaS_LeadMetaData<KVType>& table_host,
    HiveHashTableAoaS_LeadMetaData<KVType>* table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_host,
    HiveOverflowStash<KeyType, ValueType>* stash_device
)
{
    std::cout << "Destroying Hive Hash Table AoaS..." << std::endl;
    //Pull back device data to host
    HiveHashTableAoaS_LeadMetaData<KVType> table_dev_mirror{};
    CUDA_CHECK(cudaMemcpy(&table_dev_mirror, table_device, sizeof(HiveHashTableAoaS_LeadMetaData<KVType>), cudaMemcpyDeviceToHost));

    //Free device inner buffers
    if(table_dev_mirror.buckets)
        CUDA_CHECK(cudaFree(table_dev_mirror.buckets));

    //Free host inner buffers
    CUDA_CHECK(cudaFree(table_device));
//    CUDA_CHECK(cudaFree(stash_device));

    if(table_host.buckets)
    {
        delete[] table_host.buckets;
        table_host.buckets = nullptr;
    }

}

void hive_drain_stash(
    HiveOverflowStash<KeyType, ValueType>* stash_table_device,
    HiveOverflowStash<KeyType, ValueType>& stash_table_host,
    uint32_t max_to_drain,
    cudaStream_t stream
)
{
    std::cout << "Draining stash..." << std::endl;
    //Pull back device data to host
    HiveOverflowStash<KeyType, ValueType> stash_dev_mirror{};
    CUDA_CHECK(cudaMemcpyAsync(&stash_dev_mirror, stash_table_device, sizeof(HiveOverflowStash<KeyType, ValueType>), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));


    //Figure out how many items to drain
    uint32_t head = stash_dev_mirror.head;
    uint32_t tail = stash_dev_mirror.tail;
    uint32_t count = tail - head;
    if(count > max_to_drain)
        count = max_to_drain;

    if(count == 0)
    {
        std::cout << "Stash is empty, nothing to drain." << std::endl;
        return; //nothing to drain
    }

    //Copy the items from device to host
    uint32_t start_idx = head % stash_dev_mirror.capacity;
    if(start_idx + count <= stash_dev_mirror.capacity)
    {
        //Single contiguous copy
        CUDA_CHECK(cudaMemcpyAsync(
            stash_table_host.keys + (head % stash_table_host.capacity),
            stash_dev_mirror.keys + start_idx,
            sizeof(uint32_t) * count,
            cudaMemcpyDeviceToHost,
            stream
        ));
        CUDA_CHECK(cudaMemcpyAsync(
            stash_table_host.values + (head % stash_table_host.capacity),
            stash_dev_mirror.values + start_idx,
            sizeof(uint32_t) * count,
            cudaMemcpyDeviceToHost,
            stream
        ));
    }
    else
    {
        // Two-part copy due to wrap-around
        uint32_t first_part = stash_dev_mirror.capacity - start_idx;
        uint32_t second_part = count - first_part;

        // Copy first contiguous part
        CUDA_CHECK(cudaMemcpyAsync(
            stash_table_host.keys + (head % stash_table_host.capacity),
            stash_dev_mirror.keys + start_idx,
            sizeof(uint32_t) * first_part,
            cudaMemcpyDeviceToHost,
            stream
        ));
        CUDA_CHECK(cudaMemcpyAsync(
            stash_table_host.values + (head % stash_table_host.capacity),
            stash_dev_mirror.values + start_idx,
            sizeof(uint32_t) * first_part,
            cudaMemcpyDeviceToHost,
            stream
        ));

        // Copy wrapped-around part
        CUDA_CHECK(cudaMemcpyAsync(
            stash_table_host.keys + ((head + first_part) % stash_table_host.capacity),
            stash_dev_mirror.keys,
            sizeof(uint32_t) * second_part,
            cudaMemcpyDeviceToHost,
            stream
        ));
        CUDA_CHECK(cudaMemcpyAsync(
            stash_table_host.values + ((head + first_part) % stash_table_host.capacity),
            stash_dev_mirror.values,
            sizeof(uint32_t) * second_part,
            cudaMemcpyDeviceToHost,
            stream
        ));
    }   
    CUDA_CHECK(cudaStreamSynchronize(stream));

    //Consumer advance the head
    head += count;
    CUDA_CHECK(cudaMemcpyAsync(&stash_table_device->head, &head, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "Draining stash..." << std::endl;
}