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
void hiveHashTableCreate(
    HiveHashTable& table_host,
    HiveHashTable* table_device,
    size_t num_buckets,
    size_t max_num_buckets
)
{
    //Allocate device buffers (poiinters live on host variable)
    HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>* d_bodies = nullptr;
    uint32_t* d_freeMask = nullptr;
    uint32_t* d_lock = nullptr;
    // uint8_t* d_tags = nullptr;

    CUDA_CHECK(cudaMalloc(&d_bodies, sizeof(HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>) * max_num_buckets));
    CUDA_CHECK(cudaMalloc(&d_freeMask, sizeof(uint32_t) * max_num_buckets));
    CUDA_CHECK(cudaMalloc(&d_lock, sizeof(uint32_t) * max_num_buckets));

    num_buckets = std::min(num_buckets, max_num_buckets);

    CUDA_CHECK(cudaMemset(d_bodies, 0, sizeof(HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>) * num_buckets));
    CUDA_CHECK(cudaMemset(d_lock, 0, sizeof(uint32_t) * num_buckets));

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
        MAX_EVICT
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

    table_host.max_evictions = MAX_EVICT;

    table_host.buckets = new HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>[max_num_buckets];
    // table_host.freeMask = h_freeMask.data();
    table_host.lock = new uint32_t[max_num_buckets]();
    table_host.index_mask = table_dev_mirror.index_mask;
    table_host.split_ptr = 0;

}

void hiveHashTableCreate(
    HiveHashTableAoaS_LeadMetaData<kv_type>& table_host,
    HiveHashTableAoaS_LeadMetaData<kv_type>* table_device,
    size_t num_buckets,
    size_t max_num_buckets
)
{
    HiveBucketAoaS_LeadMetaData<kv_type>* d_buckets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buckets, sizeof(HiveBucketAoaS_LeadMetaData<kv_type>) * max_num_buckets));

    size_t alloc_buckets = std::min(num_buckets, max_num_buckets);

    uint32_t initial_mask = (HIVE_BUCKET_SLOTS_AOAS >= 32) ? 0xFFFFFFFFU : ((1ULL << HIVE_BUCKET_SLOTS_AOAS) - 1);

    initAoaSBucket<<<(alloc_buckets + 255) / 256, 256>>>(d_buckets, alloc_buckets, initial_mask);
    CUDA_CHECK(cudaDeviceSynchronize());

    //Build Host Mirror
    HiveHashTableAoaS_LeadMetaData<kv_type> table_dev_mirror = {
        d_buckets,
        alloc_buckets,
        max_num_buckets,
        MAX_EVICT
        #if DYNAMIC_RESIZE
        , static_cast<uint32_t>((1U << (32 - __builtin_clz(alloc_buckets - 1))) - 1), //index_mask
        0 //split_ptr
        #endif
    };

    CUDA_CHECK(cudaMemcpy(table_device, &table_dev_mirror, sizeof(HiveHashTableAoaS_LeadMetaData<kv_type>), cudaMemcpyHostToDevice));

    table_host.num_buckets = alloc_buckets;
    table_host.max_num_buckets = max_num_buckets;
    table_host.max_evictions = MAX_EVICT;

    table_host.buckets = new HiveBucketAoaS_LeadMetaData<kv_type>[max_num_buckets];
    table_host.index_mask = table_dev_mirror.index_mask;
    table_host.split_ptr = table_dev_mirror.split_ptr;
}

// The actual implementation used for AoAS buckets (body continues after $SELECTION_PLACEHOLDER$).
void hiveHashTableCreate(
    HiveHashTableAoaS<kv_type>& table_host,
    HiveHashTableAoaS<kv_type>* table_device,
    size_t num_buckets,
    size_t max_num_buckets
)
{
    HiveBucketAoaS<kv_type>* d_buckets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buckets, sizeof(HiveBucketAoaS<kv_type>) * max_num_buckets));
    
    size_t alloc_buckets = std::min(num_buckets, max_num_buckets);

    uint32_t initial_mask = (HIVE_BUCKET_SLOTS_AOAS >= 32) ? 0xFFFFFFFFU : ((1ULL << HIVE_BUCKET_SLOTS_AOAS) - 1);

    initAoaSBucket<<<(alloc_buckets + 255) / 256, 256>>>(d_buckets, alloc_buckets, initial_mask);
    CUDA_CHECK(cudaDeviceSynchronize());

    //Build Host Mirror
    HiveHashTableAoaS<kv_type> table_dev_mirror = {
        d_buckets,
        alloc_buckets,
        max_num_buckets,
        MAX_EVICT
        #if DYNAMIC_RESIZE
        , static_cast<uint32_t>((1U << (32 - __builtin_clz(alloc_buckets - 1))) - 1), //index_mask
        0 //split_ptr
        #endif
    };

    CUDA_CHECK(cudaMemcpy(table_device, &table_dev_mirror, sizeof(HiveHashTableAoaS<kv_type>), cudaMemcpyHostToDevice));

    table_host.num_buckets = alloc_buckets;
    table_host.max_num_buckets = max_num_buckets;
    table_host.max_evictions = MAX_EVICT;

    table_host.buckets = new HiveBucketAoaS<kv_type>[max_num_buckets];
    table_host.index_mask = table_dev_mirror.index_mask;
    table_host.split_ptr = table_dev_mirror.split_ptr;
}


void hiveHashTableDestroy(
    HiveHashTable& table_host, HiveHashTable* table_device)
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
    



    //Free host inner buffers
    CUDA_CHECK(cudaFree(table_device));


    //Free host mirror
    delete[] table_host.buckets;
    delete[] table_host.lock;
    delete[] table_host.freeMask;
}

void hiveHashTableClear(
    HiveHashTable* table_device
)
{
    HiveHashTable table_mirror;
    CUDA_CHECK(cudaMemcpy(&table_mirror, table_device, sizeof(HiveHashTable), cudaMemcpyDeviceToHost));

    // Reset buckets and locks to 0
    CUDA_CHECK(cudaMemset(table_mirror.buckets, 0, sizeof(HiveBucketBody<uint64_t, HIVE_BUCKET_SLOTS>) * table_mirror.num_buckets));
    CUDA_CHECK(cudaMemset(table_mirror.lock, 0, sizeof(uint32_t) * table_mirror.num_buckets));

    // Reset freeMask to all 1s
    const uint32_t full_mask = (HIVE_BUCKET_SLOTS < 32) ? ((1u << HIVE_BUCKET_SLOTS) - 1u) : 0xFFFFFFFFu;
    std::vector<uint32_t> h_full_mask(table_mirror.num_buckets, full_mask);
    CUDA_CHECK(cudaMemcpy(table_mirror.freeMask, h_full_mask.data(), sizeof(uint32_t) * table_mirror.num_buckets, cudaMemcpyHostToDevice));
}

void hiveHashTableClear(
    HiveHashTableAoaS<kv_type>* table_device
)
{
    HiveHashTableAoaS<kv_type> table_mirror;
    CUDA_CHECK(cudaMemcpy(&table_mirror, table_device, sizeof(HiveHashTableAoaS<kv_type>), cudaMemcpyDeviceToHost));
    
    uint32_t initial_mask = (HIVE_BUCKET_SLOTS_AOAS >= 32) ? 0xFFFFFFFFU : ((1ULL << HIVE_BUCKET_SLOTS_AOAS) - 1);
    initAoaSBucket<<<(table_mirror.num_buckets + 255) / 256, 256>>>(table_mirror.buckets, table_mirror.num_buckets, initial_mask);
}

void hiveHashTableClear(
    HiveHashTableAoaS_LeadMetaData<kv_type>* table_device    
)
{
    HiveHashTableAoaS_LeadMetaData<kv_type> table_mirror;
    CUDA_CHECK(cudaMemcpy(&table_mirror, table_device, sizeof(HiveHashTableAoaS_LeadMetaData<kv_type>), cudaMemcpyDeviceToHost));
    
    uint32_t initial_mask = (HIVE_BUCKET_SLOTS_AOAS >= 32) ? 0xFFFFFFFFU : ((1ULL << HIVE_BUCKET_SLOTS_AOAS) - 1);
    initAoaSBucket<<<(table_mirror.num_buckets + 255) / 256, 256>>>(table_mirror.buckets, table_mirror.num_buckets, initial_mask);
}


void hiveHashTableDestroy(
    HiveHashTableAoaS<kv_type>& table_host,
    HiveHashTableAoaS<kv_type>* table_device
)
{
    std::cout << "Destroying Hive Hash Table AoaS..." << std::endl;
    //Pull back device data to host
    HiveHashTableAoaS<kv_type> table_dev_mirror{};
    CUDA_CHECK(cudaMemcpy(&table_dev_mirror, table_device, sizeof(HiveHashTableAoaS<kv_type>), cudaMemcpyDeviceToHost));

    //Free device inner buffers
    if(table_dev_mirror.buckets)
        CUDA_CHECK(cudaFree(table_dev_mirror.buckets));

    // HiveOverflowStash<key_type, value_type> stash_dev_mirror{};
    // CUDA_CHECK(cudaMemcpy(&stash_dev_mirror, stash_device, sizeof(HiveOverflowStash<key_type, value_type>), cudaMemcpyDeviceToHost));

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
    HiveHashTableAoaS_LeadMetaData<kv_type>& table_host,
    HiveHashTableAoaS_LeadMetaData<kv_type>* table_device
)
{
    std::cout << "Destroying Hive Hash Table AoaS..." << std::endl;
    //Pull back device data to host
    HiveHashTableAoaS_LeadMetaData<kv_type> table_dev_mirror{};
    CUDA_CHECK(cudaMemcpy(&table_dev_mirror, table_device, sizeof(HiveHashTableAoaS_LeadMetaData<kv_type>), cudaMemcpyDeviceToHost));

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