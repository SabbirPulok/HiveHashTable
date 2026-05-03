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



#include "hive_kernels.cuh"
template<typename TableType, typename HashPolicy = Default2HashPolicy>
void run_hash_table_benchmark(
    TableType* hive_table_device,
    HiveOverflowStashBucket<kv_type>* hive_stash_table,
    Operation* d_ops,
    size_t num_ops,
    uint64_t* h_results,
    size_t num_blocks,
    size_t threads_per_block,
    size_t numIterations,
    double &elapsed_time,
    bool verify,
    bool all_deletes,
    bool all_lookups,
    bool all_inserts,
    bool skip_clear
    #if BREAKDOWN_INSERT
    , InsertBreakdown* d_insert_breakdown = nullptr
    #endif
)
{
    //Device Results
    uint64_t* d_results;
    //max stash elements
    size_t num_buckets = 0;
    CUDA_CHECK(cudaMemcpy(&num_buckets, &hive_table_device->num_buckets, sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t max_total_ops = num_ops + num_buckets * HiveOverflowStashBucket<kv_type>::SLOTS;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(uint64_t) * max_total_ops));
    CUDA_CHECK(cudaMemset(d_results, 0, sizeof(uint64_t) * max_total_ops));

    size_t numWarmup = all_deletes ? 0 : 5;
    numIterations = all_deletes ? 1 : numIterations;
    
    std::vector<InsertBreakdown> h_insert_breakdown(num_ops);

    //Warmup
    for (size_t i = 0; i < numWarmup; i++) {
        if(all_lookups)
        {
            hive_lookup_kernel<TableType, HashPolicy><<<num_blocks, threads_per_block>>>(
                hive_table_device,
                hive_stash_table,
                d_ops,
                num_ops,
                d_results
            );
        }
        else if (all_inserts)
        {
            hive_insert_kernel<TableType, HashPolicy><<<num_blocks, threads_per_block>>>(
                hive_table_device,
                hive_stash_table,
                d_ops,
                num_ops,
                d_results
                #if BREAKDOWN_INSERT
                , d_insert_breakdown
                #endif
            );
        }
        else
        {
            hive_mixed_kernel<TableType, HashPolicy><<<num_blocks, threads_per_block>>>(
                hive_table_device,
                hive_stash_table,
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
    }

    //After warmup, clear the hash table and stash to ensure a clean benchmark run.
    if(!all_lookups && !all_deletes && !skip_clear)
    {
        hiveHashTableClear(hive_table_device);
    }
    

    float total_ms = 0;
    auto num_ops_with_stash = num_ops;
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for(size_t iter = 0; iter < numIterations; iter++)
    {
        CoarseGrainedGPUTimer timer;
        if(all_lookups)
        {
            timer.start(stream);
            hive_lookup_kernel<TableType, HashPolicy><<<num_blocks, threads_per_block, 0, stream>>>(
                hive_table_device,
                hive_stash_table,
                d_ops,
                num_ops_with_stash,
                d_results
            );
            timer.stop(stream);
        }
        else if (all_inserts)
        {
            timer.start(stream);
            hive_insert_kernel<TableType, HashPolicy><<<num_blocks, threads_per_block, 0, stream>>>(
                hive_table_device,
                hive_stash_table,
                d_ops,
                num_ops_with_stash,
                d_results
                #if BREAKDOWN_INSERT
                , d_insert_breakdown
                #endif
            );
            timer.stop(stream);
        }
        else{
            timer.start(stream);
            hive_mixed_kernel<TableType, HashPolicy><<<num_blocks, threads_per_block, 0, stream>>>(
                hive_table_device,
                hive_stash_table,
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

        std::cout << "Iteration " << iter << std::endl;

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

                    hive_hash_shrink(hive_table_device, hive_stash_table, shrink_amount);
                }

            }   
        }
        #endif

        //include stash on next batch

        if(!skip_clear)
        {
            reinsert_stash_into_next_batch(
                hive_table_device,
                hive_stash_table,
                d_ops,
                num_ops,
                num_ops_with_stash,
                stream
            );
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

        std::cout << "Unsuccessful ops: " << unsuccessful_ops << " out of " << num_ops <<", Success Rate: " << (1.0 - (unsuccessful_ops / static_cast<float>(num_ops))) * 100 << "%" << std::endl;
        std::cout << "Successful Inserts: " << successful_inserts << std::endl;
        std::cout << "Successful Lookups: " << successful_lookups << std::endl;
        std::cout << "Successful Deletes: " << successful_deletes << std::endl;

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

    std::cout << "Insert Breakdown (in ms):" << std::endl;
    std::cout << "Stage A (Try Replace Path): " << stageA_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (total_insert_time_per_insert_ms > 0 ? (stageA_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) : 0) << "%" << std::endl;
    std::cout << "Stage B (Claim And Commit Path): " << stageB_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (total_insert_time_per_insert_ms > 0 ? (stageB_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) : 0) << "%" << std::endl;
    std::cout << "Stage C (Cuckoo Eviction Path): " << stageC_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (total_insert_time_per_insert_ms > 0 ? (stageC_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) : 0) << "%" << std::endl;
    std::cout << "Stage D (Stash Path): " << stageD_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (total_insert_time_per_insert_ms > 0 ? (stageD_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) : 0) << "%" << std::endl;

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
    const int max_stash_buckets = max_bucket_and_stash_caps.second / (sizeof(HiveOverflowStashBucket<kv_type>));

    // Overflow Table
    HiveOverflowStashBucket<kv_type> *hive_stash_table; //max capacity buckets

    CUDA_CHECK(cudaMallocManaged(&hive_stash_table, max_stash_buckets * sizeof(HiveOverflowStashBucket<kv_type>)));

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
    CUDA_CHECK(cudaMalloc(&d_workload_ops, sizeof(Operation) * (num_workload_ops + table_size / HIVE_BUCKET_SLOTS)));

    thrust::fill(
        thrust::device,
        thrust::device_pointer_cast(d_workload_ops),
        thrust::device_pointer_cast(d_workload_ops) + (num_workload_ops + table_size / HIVE_BUCKET_SLOTS),
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
            sizeof(HiveBucketBody<kv_type, HIVE_BUCKET_SLOTS>) /* Size of HiveBucketBody */
            + sizeof(uint32_t) /* Size of freeMask */
            + sizeof(uint16_t) /* Size of lock */
        );

        std::cout << "Creating Hive Hash Table with " << num_buckets << " buckets and stash capacity " << max_stash_buckets << " buckets" << std::endl;
        hiveHashTableCreate(
            hive_table,
            hive_table_device,
            num_buckets,
            max_num_buckets
        );

        //Launch prefill kernel
        std::cout << "Prefill the hash table with " << num_prefill_ops << " insert operations" << std::endl;
        hive_insert_kernel<HiveHashTable, Default2HashPolicy><<<num_blocks_prefill, threads_per_block>>>(
            hive_table_device,
            hive_stash_table,
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
            hive_stash_table,
            d_workload_ops,
            num_workload_ops,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            false,
            false,
            false,
            true
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device
        );
    }
    else if(layout == HashTableDataLayout::ARRAY_OF_ALIGNED_STRUCTS)
    {
        std::cout << "Hive Hash Table With Array of aligned structures Layout" << std::endl;

        // Handle ARRAY_OF_ALIGNED_STRUCTS layout
        HiveHashTableAoaS<kv_type> hive_table;
        HiveHashTableAoaS<kv_type>* hive_table_device;
        CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTableAoaS<kv_type>)));
        const auto slots = HIVE_BUCKET_SLOTS;
        const size_t num_buckets = table_size / slots;

        const size_t max_num_buckets = max_bucket_and_stash_caps.first / (sizeof(HiveBucketAoaS<kv_type>));

        std::cout << "Creating Hive Hash Table (AoaS) with " << num_buckets << " buckets and stash capacity " << max_stash_buckets << " buckets" << std::endl;
        hiveHashTableCreate(
            hive_table,
            hive_table_device,
            num_buckets,
            max_num_buckets
        );

        //Launch prefill kernel
        hive_insert_kernel<HiveHashTableAoaS<kv_type>, Default2HashPolicy><<<num_blocks_prefill, threads_per_block>>>(
            hive_table_device,
            hive_stash_table,
            d_prefill_ops,
            num_prefill_ops,
            nullptr
            #if BREAKDOWN_INSERT
            , nullptr
            #endif
        );

        run_hash_table_benchmark(
            hive_table_device,
            hive_stash_table,
            d_workload_ops,
            num_workload_ops,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            false,
            false,
            false,
            true
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device
        );
    }
    else if (layout == HashTableDataLayout::ARRAY_OF_ALIGNED_STRUCTS_LEAD_METADATA)
    {
        std::cout << "Hive Hash Table With Array of Aligned Structures Leading Metadata Layout" << std::endl;

        HiveHashTableAoaS_LeadMetaData<kv_type> hive_table;
        HiveHashTableAoaS_LeadMetaData<kv_type>* hive_table_device;
        CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTableAoaS_LeadMetaData<kv_type>)));

        const auto slots = HIVE_BUCKET_SLOTS;
        const size_t num_buckets = table_size / slots;

        const size_t max_num_buckets = max_bucket_and_stash_caps.first / (sizeof(HiveBucketAoaS_LeadMetaData<kv_type>));

        hiveHashTableCreate(
            hive_table,
            hive_table_device,
            num_buckets,
            max_num_buckets
        );

        //Launch prefill kernel
        hive_insert_kernel<HiveHashTableAoaS_LeadMetaData<kv_type>, Default2HashPolicy><<<num_blocks_prefill, threads_per_block>>>(
            hive_table_device,
            hive_stash_table,
            d_prefill_ops,
            num_prefill_ops,
            nullptr
            #if BREAKDOWN_INSERT
            , nullptr
            #endif
        );

        run_hash_table_benchmark(
            hive_table_device,
            hive_stash_table,
            d_workload_ops,
            num_workload_ops,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            false,
            false,
            false,
            true
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device
        );

    }

    CUDA_CHECK(cudaFree(hive_stash_table));
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
    const size_t max_stash_buckets = max_bucket_and_stash_caps.second / (sizeof(HiveOverflowStashBucket<kv_type>));

    HiveOverflowStashBucket<kv_type> *hive_stash_table; //max capacity buckets

    CUDA_CHECK(cudaMallocManaged(&hive_stash_table, sizeof(HiveOverflowStashBucket<kv_type>) * max_stash_buckets));

    //Fill up device ops
    Operation* d_ops;
    CUDA_CHECK(cudaMalloc(&d_ops, sizeof(Operation) * (num_ops + max_stash_buckets * HiveOverflowStashBucket<kv_type>::SLOTS)));

    thrust::fill(
        thrust::device,
        thrust::device_pointer_cast(d_ops),
        thrust::device_pointer_cast(d_ops) + num_ops + max_stash_buckets * HiveOverflowStashBucket<kv_type>::SLOTS,
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

    bool all_insert = std::count_if(h_ops, h_ops + num_ops, [](const Operation& op) {
        return op.type == OperationType::INSERT;
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
            sizeof(HiveBucketBody<kv_type, HIVE_BUCKET_SLOTS>) /* Size of HiveBucketBody */
            + sizeof(uint32_t) /* Size of freeMask */
            + sizeof(uint16_t) /* Size of lock */
        );

        std::cout << "Creating Hive Hash Table with " << num_buckets << " buckets and stash capacity " << max_num_buckets << " buckets" << std::endl;
        hiveHashTableCreate(
            hive_table, 
            hive_table_device,
            num_buckets, 
            max_num_buckets
        );

        if(all_lookup || all_delete)
        {
            //Launch prefill kernel
            hive_mixed_kernel<HiveHashTable, HashPolicy><<<num_blocks, threads_per_block>>>(
                hive_table_device,
                hive_stash_table,
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
            hive_stash_table,
            d_ops,
            num_ops,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            all_delete,
            all_lookup,
            all_insert,
            false
            #if BREAKDOWN_INSERT
            , d_insert_breakdown
            #endif
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device
        );
    }
    else if(layout == HashTableDataLayout::ARRAY_OF_ALIGNED_STRUCTS)
    {
        HiveHashTableAoaS<kv_type> hive_table;
        HiveHashTableAoaS<kv_type>* hive_table_device;
        CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTableAoaS<kv_type>)));

        const auto slots = HIVE_BUCKET_SLOTS;
        const size_t num_buckets = table_size / slots;

        // For AoAS layout the per-bucket size equals sizeof(HiveBucketAoaS<kv_type>).
        const size_t max_num_buckets = max_bucket_and_stash_caps.first / sizeof(HiveBucketAoaS<kv_type>);

        std::cout << "Creating Hive Hash Table with " << num_buckets << " buckets and stash capacity " << max_num_buckets << " buckets" << std::endl;
        hiveHashTableCreate(
            hive_table,
            hive_table_device,
            num_buckets,
            max_num_buckets
        );

        if(all_lookup || all_delete)
        {
            //Launch prefill kernel
            hive_mixed_kernel<HiveHashTableAoaS<kv_type>, HashPolicy><<<num_blocks, threads_per_block>>>(
                hive_table_device,
                hive_stash_table,
                d_prefill_ops,
                num_ops,
                nullptr
                #if BREAKDOWN_INSERT
                , nullptr
                #endif
            );
        }

        run_hash_table_benchmark<HiveHashTableAoaS<kv_type>, HashPolicy>(
            hive_table_device,
            hive_stash_table,
            d_ops,
            num_ops,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            all_delete,
            all_lookup,
            all_insert,
            false
            #if BREAKDOWN_INSERT
            , d_insert_breakdown
            #endif
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device
        );
    }
    else if(layout == HashTableDataLayout::ARRAY_OF_ALIGNED_STRUCTS_LEAD_METADATA)
    {
        std::cout << "Hash Table with Lead Metadata Layout" << std::endl;
        
        HiveHashTableAoaS_LeadMetaData<kv_type> hive_table;
        HiveHashTableAoaS_LeadMetaData<kv_type>* hive_table_device;
        CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTableAoaS_LeadMetaData<kv_type>)));

        const auto slots = HIVE_BUCKET_SLOTS;
        const size_t num_buckets = table_size / slots;

        // For AoAS layout the per-bucket size equals sizeof(HiveBucketAoaS<kv_type>).
        const size_t max_num_buckets = max_bucket_and_stash_caps.first / sizeof(HiveBucketAoaS_LeadMetaData<kv_type>);

        std::cout << "Creating Hive Hash Table with " << num_buckets << " buckets and stash capacity " << max_stash_buckets << " buckets" << std::endl;
        hiveHashTableCreate(
            hive_table, 
            hive_table_device,
            num_buckets, 
            max_num_buckets
        );

        if(all_lookup || all_delete )
        {
            //Launch prefill kernel
            hive_mixed_kernel<HiveHashTableAoaS_LeadMetaData<kv_type>, HashPolicy><<<num_blocks, threads_per_block>>>(
                hive_table_device,
                hive_stash_table,
                d_prefill_ops,
                num_ops,
                nullptr
                #if BREAKDOWN_INSERT
                , nullptr
                #endif
            );
        }

        run_hash_table_benchmark<HiveHashTableAoaS_LeadMetaData<kv_type>, HashPolicy>(
            hive_table_device,
            hive_stash_table,
            d_ops,
            num_ops,
            h_results,
            num_blocks,
            threads_per_block,
            numIterations,
            elapsed_time,
            verify,
            all_delete,
            all_lookup,
            all_insert,
            false
            #if BREAKDOWN_INSERT
            , d_insert_breakdown
            #endif
        );

        hiveHashTableDestroy(
            hive_table,
            hive_table_device
        );
    }

    CUDA_CHECK(cudaFree(hive_stash_table));
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

    const size_t max_stash_buckets = max_bucket_and_stash_caps.second / (sizeof(HiveOverflowStashBucket<kv_type>));

    const size_t max_stash_slots = max_stash_buckets * HiveOverflowStashBucket<kv_type>::SLOTS;

    //Device Table and Stash
    HiveHashTable hive_table;
    HiveHashTable* hive_table_device;
    CUDA_CHECK(cudaMalloc(&hive_table_device, sizeof(HiveHashTable)));
    
    HiveOverflowStashBucket<kv_type> *hive_stash_table = nullptr; //max capacity buckets
    CUDA_CHECK(cudaMallocManaged(&hive_stash_table, sizeof(HiveOverflowStashBucket<kv_type>) * max_stash_buckets));

    std::cout << "Creating Hive Hash Table with " << num_buckets << " buckets and stash capacity " << max_stash_buckets << " buckets" << std::endl;
    hiveHashTableCreate(
        hive_table, 
        hive_table_device,
        num_buckets, 
        max_num_buckets
    );

    //Fill up device Ops
    Operation* d_ops;
    CUDA_CHECK(cudaMalloc(&d_ops, sizeof(Operation) * (num_ops + max_stash_slots)));

    thrust::fill(
        thrust::device, 
        thrust::device_pointer_cast(d_ops), 
        thrust::device_pointer_cast(d_ops) + num_ops + max_stash_slots, 
        Operation{OperationType::NONE, 0}
    );
    CUDA_CHECK(cudaMemcpy(d_ops, h_ops, sizeof(Operation) * (num_ops), cudaMemcpyHostToDevice));

    std::cout << "Launching mixed operations kernel with " << num_blocks << " blocks of " << threads_per_block << " threads..." << std::endl;
    
    //Device Results
    uint64_t* d_results;
    size_t nb = 0;
    CUDA_CHECK(cudaMemcpy(&nb, &hive_table_device->num_buckets, sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t max_total_ops = num_ops + nb + max_stash_slots;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(uint64_t) * (num_ops + max_stash_slots)));
    CUDA_CHECK(cudaMemset(d_results, 0, sizeof(uint64_t) * (num_ops + max_stash_slots)));

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
            hive_stash_table,
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
            hive_stash_table,
            d_ops,
            num_ops,
            d_results
            #if BREAKDOWN_INSERT
            , nullptr
            #endif
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        //reset stash
        CUDA_CHECK(cudaMemsetAsync(hive_stash_table, 0, sizeof(HiveOverflowStashBucket<kv_type>) * max_stash_buckets));

    }
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float milliseconds = 0;

    auto num_ops_with_stash = num_ops;

    for(size_t iter=0; iter<numIterations; iter++)
    {
        CoarseGrainedGPUTimer timer;
        timer.start(stream);
        hive_mixed_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            hive_table_device,
            hive_stash_table,
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
                
                hive_hash_shrink(hive_table_device, hive_stash_table, shrink_amount);
            }
        }
        #endif

        //include stash on next batch
        reinsert_stash_into_next_batch(
            hive_table_device,
            hive_stash_table,
            d_ops,
            num_ops,
            num_ops_with_stash,
            stream
        );
        
    }
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

        std::cout << "Unsuccessful ops: " << unsuccessful_ops << " out of " << num_ops <<", Success Rate: " << (1.0 - (unsuccessful_ops / static_cast<float>(num_ops))) * 100 << "%" << std::endl;
        std::cout << "Successful Inserts: " << successful_inserts << std::endl;
        std::cout << "Successful Lookups: " << successful_lookups << std::endl;
        std::cout << "Successful Deletes: " << successful_deletes << std::endl;

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

    std::cout << "Insert Breakdown (in ms):" << std::endl;
    std::cout << "Stage A (Try Replace Path): " << stageA_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (total_insert_time_per_insert_ms > 0 ? (stageA_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) : 0) << "%" << std::endl;
    std::cout << "Stage B (Claim And Commit Path): " << stageB_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (total_insert_time_per_insert_ms > 0 ? (stageB_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) : 0) << "%" << std::endl;
    std::cout << "Stage C (Cuckoo Eviction Path): " << stageC_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (total_insert_time_per_insert_ms > 0 ? (stageC_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) : 0) << "%" << std::endl;
    std::cout << "Stage D (Stash Path): " << stageD_elapsed_time_per_insert_ms << " ms" << ", Percentage of total: " << (total_insert_time_per_insert_ms > 0 ? (stageD_elapsed_time_per_insert_ms / total_insert_time_per_insert_ms * 100) : 0) << "%" << std::endl;

    #endif

    //Destroy tables
    hiveHashTableDestroy(hive_table, hive_table_device);
    CUDA_CHECK(cudaFree(hive_stash_table));
    CUDA_CHECK(cudaFree(d_ops));
    CUDA_CHECK(cudaFree(d_results));
}

