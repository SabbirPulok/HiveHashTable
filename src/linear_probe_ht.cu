#include "linear_probe_ht.cuh"
#include "hash.hpp"
#include "cuda_helper.cuh"
#include "GPUTimer.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda/atomic>
#include <nvtx3/nvToolsExt.h>

__host__ __device__ __constant__ uint64_t EMPTY_KEY = 0ull;
__host__ __device__ __constant__ uint64_t TOMBSTONE = 0xFFFFFFFFFFFFFFFFull;
__host__ __device__ __constant__ uint64_t IN_PROGRESS = 0xFFFFFFFFFFFFFFFEull;

inline size_t probe_budget(double load_factor) {
    if(load_factor <= 0.5) return 64;
    else if(load_factor <= 0.7) return 64;
    else if(load_factor <= 0.8) return 64;
    else if(load_factor <= 0.9) return 64;
    else return 64;
}

inline bool verify_lp_found(
    const std::vector<HashEntry>table,
    size_t table_size,
    uint64_t key,
    size_t max_probes
)
{
    uint32_t hash = hash32(key, table_size);

    for(size_t j = 0; j < max_probes; j++)
    {
        uint32_t idx = (hash + j) & (table_size - 1);
        if(table[idx].key == key)
        {
            //std::cout << "Key found: " << key << " at index: " << idx << std::endl;
            return true;
        }
    }
    //std::cout << "Key not found: " << key << std::endl;
    return false;
}

__device__ bool lp_insert_slot(
    HashEntry* __restrict__ table,
    uint32_t idx,
    uint64_t key,
    uint64_t value
)
{
    //printf("Inserting key: %llu at index: %u\n", key, idx);

    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> atomic_key(table[idx].key);
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> atomic_value(table[idx].value);

    uint64_t observed = atomic_key.load(cuda::std::memory_order_acquire);

    if(observed == key)
    {
        //update exisiting value with release to order prior writes
        atomic_value.store(value, cuda::std::memory_order_release);

        return true;
    }

    uint64_t expected = EMPTY_KEY;

    if(!atomic_key.compare_exchange_strong(
        expected,
        IN_PROGRESS,
        cuda::std::memory_order_acq_rel,
        cuda::std::memory_order_relaxed
    ))
    {
        if (expected == TOMBSTONE){
            if(!atomic_key.compare_exchange_strong(
                expected,
                IN_PROGRESS,
                cuda::std::memory_order_acq_rel,
                cuda::std::memory_order_relaxed
            )){
                return false; //if failed to claim, keep probing
            }
        }
        else if(expected == key){
            atomic_value.store(value, cuda::std::memory_order_release);
            return true;
        }
        else{
            return false; //if failed to claim, keep probing
        }   
    }

    atomic_value.store(value, cuda::std::memory_order_relaxed);
    atomic_key.store(key, cuda::std::memory_order_release);
    return true;
}

__device__ bool lp_lookup_slot(
    HashEntry* __restrict__ table,
    uint32_t idx,
    uint64_t key,
    uint64_t* out
)
{
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> atomic_key(table[idx].key);

    uint64_t cur_key = atomic_key.load(cuda::std::memory_order_acquire);

    if(cur_key == key)
    {
        //Acquire the value to see the lates value
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> atomic_value(table[idx].value);

        *out = atomic_value.load(cuda::std::memory_order_acquire);
        return true;
    }
    
    if(cur_key == EMPTY_KEY)
    {
        *out = EMPTY_KEY; //not found
        return true;
    }

    //IN_PROGRESS or TOMBSTONE or some other key, keep probing
    return false;
}

__device__ bool lp_delete_slot(
    HashEntry* __restrict__ table,
    uint32_t idx,
    uint64_t key
)
{
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> atomic_key(table[idx].key);

    uint64_t cur_key = atomic_key.load(cuda::std::memory_order_acquire);

    if(cur_key == key)
    {
        //Mark as deleted with release order to order prior writes
        atomic_key.store(TOMBSTONE, cuda::std::memory_order_release);
        return true;
    }

    if(cur_key == EMPTY_KEY)
    {
        //miss not found in the table
        return true; //not found
    }

    //if IN_PROGRESS or TOMBSTONE or some other key, keep probing
    return false;
}

__global__ void lp_mixed_ops(
    HashEntry* __restrict__ table,
    Operation* __restrict__ ops,
    size_t table_size,
    size_t num_ops,
    size_t max_probes,
    uint64_t* __restrict__ results
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= num_ops) return;

    Operation op = ops[tid];    

    uint64_t key = op.key;

    uint32_t hash = hash32(key, table_size);

    results[tid] = EMPTY_KEY;;
    
    uint64_t value = key + 1;
    uint64_t out = EMPTY_KEY;

    //printf("Operation Type: %d Key: %llu Thread: %llu\n", static_cast<int>(op.type), key, tid);

    if(op.key == EMPTY_KEY || op.key == TOMBSTONE || op.key == IN_PROGRESS) {
        //Invalid key for any operation
        return;
    }

    if(op.type == OperationType::INSERT) {

        for(uint32_t i = 0; i < max_probes; ++i)
        {
            uint32_t idx = (hash + i) & (table_size -1);
            
            if(lp_insert_slot(table, idx, key, value)){
                results[tid] = value;
                break;
            }
        }
        return;
    }

    if (op.type == OperationType::LOOKUP) {
        for (size_t i = 0; i < max_probes; ++i) {

            uint32_t idx = (hash + i) & (table_size - 1);
            if (lp_lookup_slot(table, idx, key, &value))
            {
                results[tid] = value;
                break;
            }
        }
        return;
    }

    if(op.type == OperationType::DELETE) {
        for (size_t i = 0; i < max_probes; ++i)
        {
            uint32_t idx = (hash + i) & (table_size -1);

            if(lp_delete_slot(table, idx, key)){
                results[tid] = key + 1; //indicate deleted
                break;
            }
        }
        return; //not found within max_probes
    }
    return;
}

void lp_launch_kernel_with_mix_ops(
    LinearProbeConfig& config,
    std::vector<Operation> mix_ops,
    size_t threads_per_block,
    size_t num_blocks,
    size_t numIterations,
    bool verification_lp
)
{
    std::cout << "Launching kernels with mix ops" << std::endl;
    size_t max_probes = probe_budget(config.load_factor);
    std::cout << "Max probes set to: " << max_probes << std::endl;

    // device allocations
    HashEntry* d_table = nullptr;

    CUDA_CHECK(cudaMalloc(&d_table, sizeof(HashEntry) * config.table_size));
    CUDA_CHECK(cudaMemcpy(d_table, config.table, sizeof(HashEntry) * config.table_size, cudaMemcpyHostToDevice));

    //const auto& ops = *(config.mix_ops);
    uint64_t* d_results = nullptr;
    Operation* d_ops = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ops, sizeof(Operation) * mix_ops.size()));
    CUDA_CHECK(cudaMemcpy(d_ops, mix_ops.data(), sizeof(Operation) * mix_ops.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_results, sizeof(uint64_t) * mix_ops.size()));
    //CUDA_CHECK(cudaMemset(d_results, static_cast<uint64_t>(EMPTY_KEY), sizeof(uint64_t) * mix_ops.size()));


    //Warmup 5 runs
    for(size_t i = 0; i < 5; i++)
    {
        //nvtxRangePushA("LP_Mixed_Ops_Warmup");
        lp_mixed_ops<<<num_blocks, threads_per_block>>>(
            d_table,
            d_ops,
            config.table_size,
            mix_ops.size(),
            max_probes,
            d_results
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        //nvtxRangePop();
        //reset the table
        CUDA_CHECK(cudaMemcpy(d_table, config.table, sizeof(HashEntry) * config.table_size, cudaMemcpyHostToDevice));
    }

    CoarseGraindGPUTimer timer;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for(size_t iter = 0; iter < numIterations; iter++)
    {
        CUDA_CHECK(cudaMemsetAsync(d_results, EMPTY_KEY, sizeof(uint64_t) * mix_ops.size(), stream));

        nvtxRangePushA("LP_Mixed_Ops_Measured");
        timer.start(stream);
        lp_mixed_ops<<<num_blocks, threads_per_block>>>(
            d_table,
            d_ops,
            config.table_size,
            mix_ops.size(),
            max_probes,
            d_results
        );
        timer.stop(stream);
        nvtxRangePop();

        config.elapsed_times[KernelType::MIX_OPS] += timer.getElapsedTime();

        if(iter!= numIterations - 1)
        {
            //reset the table
            CUDA_CHECK(cudaMemcpyAsync(d_table, config.table, sizeof(HashEntry) * config.table_size, cudaMemcpyHostToDevice, stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    config.elapsed_times[KernelType::MIX_OPS] /= numIterations;
    std::cout <<"Elapsed time for MIX_OPS kernel: " << config.elapsed_times[KernelType::MIX_OPS] << " ms" << std::endl;

    //Dump the hash table back to host
    // CUDA_CHECK(cudaMemcpy(config.table, d_table, sizeof(HashEntry) * config.table_size, cudaMemcpyDeviceToHost));
    // std::cout<<"Copied table back to host"<<std::endl;
    // for(size_t i = 0; i < config.table_size; i++)
    // {
    //     HashEntry& entry = config.table[i];
    //     //if(entry.key != EMPTY_KEY && entry.key != TOMBSTONE)
    //     //{
    //         std::cout << "Table[" << i << "] = " << entry.key << " : " << entry.value << std::endl;
    //     //}
    // }

    CUDA_CHECK(cudaMemcpy(config.total_ops_results, d_results, sizeof(uint64_t) * mix_ops.size(), cudaMemcpyDeviceToHost));
    std::cout<<"Copied results back to host"<<std::endl;
    
    if(verification_lp)
    {
        size_t success_count_insert = 0;
        size_t success_count_lookup = 0;
        size_t success_count_delete = 0;

        size_t num_ops = mix_ops.size();
        size_t num_inserts = std::count_if(mix_ops.begin(), mix_ops.end(), [](const Operation& op){ return op.type == OperationType::INSERT; });
        size_t num_lookups = std::count_if(mix_ops.begin(), mix_ops.end(), [](const Operation& op){ return op.type == OperationType::LOOKUP; });
        size_t num_deletes = std::count_if(mix_ops.begin(), mix_ops.end(), [](const Operation& op){ return op.type == OperationType::DELETE; });

        for(size_t i = 0; i < mix_ops.size(); i++)
        {
            Operation op = mix_ops[i];

            //std::cout << "Op[" << i << "]: " << "Type: " << static_cast<int>(op.type) << " Result: " << config.total_ops_results[i] << " Key: " << op.key << std::endl;
            switch (op.type)
            {
            case OperationType::INSERT:
                success_count_insert += (config.total_ops_results[i] != EMPTY_KEY) ? 1 : 0;
                break;
            case OperationType::LOOKUP:
                success_count_lookup += (config.total_ops_results[i] != EMPTY_KEY) ? 1 : 0;
                break;
            case OperationType::DELETE:
                success_count_delete += (config.total_ops_results[i] != EMPTY_KEY) ? 1 : 0;
                break;
            default:
                break;
            }
        }
        std::cout << "Verification Results: " << std::endl;
        std::cout << "Successful Inserts: " << success_count_insert << " Insertions Success Rate: " << (success_count_insert / static_cast<double>(num_inserts) * 100.0) << " %" << std::endl;
        std::cout << "Successful Lookups: " << success_count_lookup << " Lookups Success Rate: " << (success_count_lookup / static_cast<double>(num_lookups) * 100.0) << " %" << std::endl;
        std::cout << "Successful Deletions: " << success_count_delete << " Deletions Success Rate: " << (success_count_delete / static_cast<double>(num_deletes) * 100.0) << " %" << std::endl;

        size_t total_success = success_count_insert + success_count_lookup + success_count_delete;

        config.throughput_mlops[KernelType::MIX_OPS] = static_cast<double>(total_success) / (config.elapsed_times[KernelType::MIX_OPS] / 1000.0) / 1e6; //MLOPS
        std::cout << "Overall Success Rate: " << (total_success / static_cast<double>(num_ops) * 100.0) << " %" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_ops));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_table));
    
    return;
}

//__launch_bounds__(256, 4) 
__global__ void lp_build_kernel(
    HashEntry* __restrict__ table,
    const uint64_t* __restrict__ keys,
    size_t table_size,
    size_t num_keys,
    size_t max_probes
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) return;

    uint64_t key = keys[tid];
    uint64_t value = keys[tid] + 1;

    if(key == EMPTY_KEY || key == TOMBSTONE) return;

    uint32_t hash = hash32(key, table_size);

    for(size_t i = 0; i < max_probes; i++) {
        uint32_t idx = (hash + i) & (table_size - 1);

        uint64_t cur_key = table[idx].key;

        if(cur_key == key) {
            table[idx].value = value;
            return;
        }

        if(cur_key == EMPTY_KEY || cur_key == TOMBSTONE) {

            uint64_t expected = cur_key;

            //Use nv_atomic builtin to perform atomic CAS on 64-bit value

            // bool success = __nv_atomic_compare_exchange_n(
            //     reinterpret_cast<unsigned long long*>(&table[idx].key),
            //     reinterpret_cast<unsigned long long*>(&expected), /*Expected*/
            //     reinterpret_cast<unsigned long long*>(&key),
            //     false, /*weak*/
            //     __ATOMIC_RELAXED, /*success order*/
            //     __ATOMIC_RELAXED, /*failure order*/
            //     __NV_THREAD_SCOPE_DEVICE /*scope*/
            // );

            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> atomic_key(table[idx].key);

            bool success = atomic_key.compare_exchange_weak(
                expected, //Expected
                key,
                cuda::std::memory_order_relaxed,
                cuda::std::memory_order_relaxed
            );
        

            //CCL atomic primitives
            

            // uint64_t old = atomicCAS(
            //     reinterpret_cast<unsigned long long*>(&table[idx].key),
            //     reinterpret_cast<unsigned long long&>(expected),
            //     reinterpret_cast<unsigned long long&>(key)
            // );
            // bool success = (old == expected);
            

            if(success){
                //printf("Success[%d]: %d\n",tid,success);
                table[idx].value = value;
                return;
            }
            //Continue probing if failed
            continue;
        }
    }
    return; //Failed to insert within max_probes
}

__global__ void lp_lookup_kernel(
    const HashEntry* __restrict__ table,
    const uint64_t* __restrict__ query_keys,
    uint64_t* __restrict__ results,
    size_t table_size,
    size_t num_queries,
    size_t max_probes
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;

    const uint64_t key = query_keys[tid];

    if(key == EMPTY_KEY || key == TOMBSTONE) {
        results[tid] = EMPTY_KEY;
        return;
    }

    uint32_t hash = hash32(key, table_size);

    for(size_t i = 0; i < max_probes; i++) {
        uint32_t idx = (hash + i) & (table_size - 1);

        uint64_t cur_key = table[idx].key;

        if(cur_key == key) {
            results[tid] = table[idx].value;
            return;
        }

        if(cur_key == EMPTY_KEY) {
            results[tid] = EMPTY_KEY; //not found
            return;
        }
        //Continue probing if cur_key is TOMBSTONE or some other key
    }
    results[tid] = EMPTY_KEY; //not found within max_probes
}


__global__ void lp_delete_kernel(
    HashEntry* __restrict__ table,
    const uint64_t* __restrict__ delete_keys,
    size_t table_size,
    size_t num_deletes,
    size_t max_probes
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_deletes) return;

    const uint64_t key = delete_keys[tid];

    if(key == EMPTY_KEY || key == TOMBSTONE) return;

    uint32_t hash = hash32(key, table_size);

    for(size_t i = 0; i < max_probes; i++) {
        uint32_t idx = (hash + i) & (table_size - 1);

        uint64_t cur_key = table[idx].key;

        if(cur_key == key) {
            //Mark as deleted using a tombstone
            table[idx].key = TOMBSTONE;
            return;
        }

        if(cur_key == EMPTY_KEY) {
            return; //not found
        }
        //Continue probing if cur_key is TOMBSTONE or some other key
    }
    return; //not found within max_probes
}

__device__ __forceinline__ void lp_delete_mix_ops(
    HashEntry* __restrict__ table,
    const uint64_t* __restrict__ keys,
    size_t table_size,
    size_t num_deletes,
    size_t max_probes
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_deletes) return;

    const uint64_t key = keys[tid];

    if(key == EMPTY_KEY || key == TOMBSTONE || key == IN_PROGRESS) return;

    uint32_t hash = hash32(key, table_size);

    for(size_t i = 0; i < max_probes; i++) {
        uint32_t idx = (hash + i) & (table_size - 1);

        //Acquire the current key, so we see the latest value
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> atomic_key(table[idx].key);

        uint64_t cur_key = atomic_key.load(cuda::std::memory_order_acquire);

        if(cur_key == key) {
            //Mark as deleted using a tombstone
            atomic_key.compare_exchange_strong(cur_key, TOMBSTONE, cuda::std::memory_order_release, cuda::std::memory_order_relaxed);
            return;
        }

        if(cur_key == EMPTY_KEY) {
            return; //not found
        }
        //Continue probing if cur_key is TOMBSTONE or some other key
    }
    return; //not found within max_probes
}


void lp_launch_kernel_with_no_mix_ops(
    LinearProbeConfig& config,
    size_t table_size, //always power of 2
    size_t threads_per_block,
    size_t num_blocks,
    size_t numIterations,
    bool verification_lp
)
{
    std::cout << "Launching kernels with no mix ops" << std::endl;
    size_t max_probes = probe_budget(config.load_factor);
    std::cout << "Max probes set to: " << max_probes << std::endl;

    // device allocations
    HashEntry* d_table = nullptr;

    CUDA_CHECK(cudaMalloc(&d_table, sizeof(HashEntry) * table_size));
    CUDA_CHECK(cudaMemcpy(d_table, config.table, sizeof(HashEntry) * table_size, cudaMemcpyHostToDevice));

    for(KernelType kernel : config.kernels)
    {
        if(kernel == KernelType::INSERT)
        {
            std::cout << "Launching INSERT kernel" << std::endl;
            const auto& K = *(config.insert_keys);

            uint64_t* d_keys = nullptr;
            
            CUDA_CHECK(cudaMalloc(&d_keys, sizeof(uint64_t) * config.num_inserts));
            CUDA_CHECK(cudaMemcpy(d_keys, K.data(), sizeof(uint64_t) * config.num_inserts, cudaMemcpyHostToDevice));

            std::cout << "Warmup 5 runs" << std::endl;
            //Warmup 5 runs
            for(size_t i = 0; i < 5; i++)
            {
                lp_build_kernel<<<num_blocks, threads_per_block>>>(
                    d_table,
                    d_keys,
                    table_size,
                    config.num_inserts,
                    max_probes
                );
                CUDA_CHECK(cudaDeviceSynchronize());

                //reset the table
                CUDA_CHECK(cudaMemcpy(d_table, config.table, sizeof(HashEntry) * table_size, cudaMemcpyHostToDevice));

            }

            std::cout << "Warmup complete, Real Runs Starting" << std::endl;
            
            // Launch insert kernel

            CoarseGraindGPUTimer timer;
            
            for(size_t iter = 0; iter < numIterations; iter++)
            {
                timer.start();
                lp_build_kernel<<<num_blocks, threads_per_block>>>(
                    d_table,
                    d_keys,
                    table_size,
                    config.num_inserts,
                    max_probes
                );
                CUDA_CHECK(cudaDeviceSynchronize());
                timer.stop();
                config.elapsed_times[kernel] += timer.getElapsedTime();

                if(iter!= numIterations - 1)
                {
                    //reset the table
                    CUDA_CHECK(cudaMemcpy(d_table, config.table, sizeof(HashEntry) * table_size, cudaMemcpyHostToDevice));
                }
            }

            config.elapsed_times[kernel] /= numIterations;

            std::cout <<"Elapsed time for INSERT kernel: " << config.elapsed_times[kernel] << " ms" << std::endl;

            //refill the host table
            CUDA_CHECK(cudaMemcpy(config.table, d_table, sizeof(HashEntry) * table_size, cudaMemcpyDeviceToHost));

            if(verification_lp)
            {
                size_t match_count = 0;
                // Verify the results
                for(size_t i = 0; i < config.num_inserts; i++)
                {
                    if(verify_lp_found(std::vector<HashEntry>(config.table, config.table + table_size), table_size, K[i], max_probes))
                    {
                        match_count++;
                    }
                }

                config.throughput_mlops[kernel] = (match_count / (config.elapsed_times[kernel] / 1000.0)) / 1e6; //in Mops/sec
                std::cout << "Number of non-empty entries after insert: " << match_count << ", Number of inserted entries: " << config.num_inserts << ", Success Rate: " <<  (match_count / static_cast<float>(config.num_inserts)) * 100 << std::endl;
            }

            CUDA_CHECK(cudaFree(d_keys));

        }
        else if(kernel == KernelType::LOOKUP)
        {
            const auto& Q = *(config.query_keys);

            uint64_t* d_query_keys = nullptr;
            uint64_t* d_results = nullptr;

            CUDA_CHECK(cudaMalloc(&d_query_keys, sizeof(uint64_t) * config.num_queries));
            CUDA_CHECK(cudaMemcpy(d_query_keys, Q.data(), sizeof(uint64_t) * config.num_queries, cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(&d_results, sizeof(uint64_t) * config.num_queries));

            //Warmup 5 runs
            for(size_t i = 0; i < 5; i++)
            {
                lp_lookup_kernel<<<num_blocks, threads_per_block>>>(
                    d_table,
                    d_query_keys,
                    d_results,
                    table_size,
                    config.num_queries,
                    max_probes
                );
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            CoarseGraindGPUTimer timer;
            timer.start();
            for(size_t iter = 0; iter < numIterations; iter++)
            {
                lp_lookup_kernel<<<num_blocks, threads_per_block>>>(
                    d_table,
                    d_query_keys,
                    d_results,
                    table_size,
                    config.num_queries,
                    max_probes
                );
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop();
            config.elapsed_times[kernel] = timer.getElapsedTime() / numIterations;

            CUDA_CHECK(cudaMemcpy(config.results->data(), d_results, sizeof(uint64_t) * config.num_queries, cudaMemcpyDeviceToHost));

            //Check how many queries were found
            if(verification_lp)
            {
                size_t found_count = 0;
                for(size_t i = 0; i < config.num_queries; i++)
                {
                    uint64_t result_key = config.results->at(i)-1;

                    if(result_key != EMPTY_KEY)
                    {
                        found_count++;
                    }
                }
                std::cout << "Number of queries found: " << found_count << ", Total queries: " << config.num_queries << ", Hit Rate: " << (found_count / static_cast<float>(config.num_queries)) * 100 << "%" << std::endl;
                config.throughput_mlops[kernel] = (found_count / (config.elapsed_times[kernel] / 1000.0)) / 1e6; //in Mops/sec
            }
            CUDA_CHECK(cudaFree(d_query_keys));
            CUDA_CHECK(cudaFree(d_results));
    
        }
        else if(kernel == KernelType::DELETE)
        {
            const auto& D = *(config.delete_keys);

            uint64_t* d_delete_keys = nullptr;

            CUDA_CHECK(cudaMalloc(&d_delete_keys, sizeof(uint64_t) * config.num_deletes));
            CUDA_CHECK(cudaMemcpy(d_delete_keys, D.data(), sizeof(uint64_t) * config.num_deletes, cudaMemcpyHostToDevice));

            //Warmup 5 runs
            for(size_t i = 0; i < 5; i++)
            {
                lp_delete_kernel<<<num_blocks, threads_per_block>>>(
                    d_table,
                    d_delete_keys,
                    table_size,
                    config.num_deletes,
                    max_probes
                );
                CUDA_CHECK(cudaDeviceSynchronize());
                //back to original table
                CUDA_CHECK(cudaMemcpy(d_table, config.table, sizeof(HashEntry) * table_size, cudaMemcpyHostToDevice));
            }
            

            CoarseGraindGPUTimer timer;
            
            for(size_t iter = 0; iter < numIterations; iter++)
            {
                timer.start();
                lp_delete_kernel<<<num_blocks, threads_per_block>>>(
                    d_table,
                    d_delete_keys,
                    table_size,
                    config.num_deletes,
                    max_probes
                );
                CUDA_CHECK(cudaDeviceSynchronize());
                timer.stop();
                config.elapsed_times[kernel] += timer.getElapsedTime();
            }
            
            config.elapsed_times[kernel] /= numIterations;

            //refill the host table
            CUDA_CHECK(cudaMemcpy(config.table, d_table, sizeof(HashEntry) * table_size, cudaMemcpyDeviceToHost));

            if(verification_lp)
            {
                size_t not_found_count = 0;
                // Verify the results
                for(size_t i = 0; i < config.num_deletes; i++)
                {
                    if(!verify_lp_found(std::vector<HashEntry>(config.table, config.table + table_size), table_size, D[i], max_probes))
                    {
                        not_found_count++;
                    }
                }
                config.throughput_mlops[kernel] = (not_found_count / (config.elapsed_times[kernel] / 1000.0)) / 1e6; //in Mops/sec
                std::cout << "Number of successfully deleted entries: " << not_found_count << ", Number of delete requests: " << config.num_deletes << ", Success Rate: " <<  (not_found_count / static_cast<float>(config.num_deletes)) * 100 << "%" << std::endl;
            }
            CUDA_CHECK(cudaFree(d_delete_keys));
        }
    }
    CUDA_CHECK(cudaFree(d_table));

}