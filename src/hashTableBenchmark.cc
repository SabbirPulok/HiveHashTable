#include "hashTableBenchmark.h"
#include "hash_table_struct.h"
#include "hash.hpp"
#include "GPUTimer.h"
#include "linear_probe_ht.cuh"
#include "utils.h"
#include "hive_hash_table.cuh"
#include "zipf_distribution.h"
#include "cuda_helper.cuh"
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <array>

#ifndef VERIFICATION_VERBOSE
#define VERIFICATION_VERBOSE true
#endif

void HashTableBenchmark::generateTestKeys(std::vector<uint64_t>& keys, size_t num_keys, const std::string& distribution)
{
    keys.resize(num_keys);

    if(distribution == "unique")
    {
        std::iota(keys.begin(), keys.end(), static_cast<uint64_t>(1)); //Unique keys from 1 to num_keys
        std::shuffle(keys.begin(), keys.end(), rng);
    }
    else if (distribution == "uniform") {
        std::uniform_int_distribution<uint64_t> dist(1, UINT64_MAX - 1); // Avoid EMPTY_KEY
        for (size_t i = 0; i < num_keys; ++i) {
            keys[i] = dist(rng) % num_keys + 1; // Ensure keys are in range [1, num_keys]
        }
    } 
    else if (distribution == "clustered")
    {
        //Create clustered distribution
        std::uniform_int_distribution<uint64_t> cluster_dist(1, 1000);
        std::uniform_int_distribution<uint64_t> offset_dist(0, 100);

        for(size_t i = 0; i < num_keys; i++) {
            uint64_t cluster_center = cluster_dist(rng);
            keys[i] = cluster_center * 10000 + offset_dist(rng);
        }
    }
    else if(distribution == "skewed")
    {
        // Simple skewed distribution implementation
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < num_keys; ++i) {
            double x = dist(rng);
            keys[i] = static_cast<uint64_t>(pow(x, 3) * (UINT64_MAX - 1)) + 1; // Skewed towards lower values
        }
    }
    else {
        throw std::invalid_argument("Unsupported distribution type");
    }
}


void HashTableBenchmark::generateTestKeys(std::vector<uint32_t>& keys, size_t num_keys, const std::string& distribution)
{
    keys.resize(num_keys);

    if(distribution == "unique")
    {
        std::iota(keys.begin(), keys.end(), static_cast<uint64_t>(1)); //Unique keys from 1 to num_keys
        std::shuffle(keys.begin(), keys.end(), rng);
    }
    else if (distribution == "uniform") {
        std::uniform_int_distribution<uint32_t> dist(1, UINT32_MAX - 1); // Avoid EMPTY_KEY
        for (size_t i = 0; i < num_keys; ++i) {
            keys[i] = dist(rng) % num_keys + 1; // Ensure keys are in range [1, num_keys]
        }
    } 
    else if (distribution == "clustered")
    {
        //Create clustered distribution
        std::uniform_int_distribution<uint32_t> cluster_dist(1, 1000);
        std::uniform_int_distribution<uint32_t> offset_dist(0, 100);

        for(size_t i = 0; i < num_keys; i++) {
            uint32_t cluster_center = cluster_dist(rng);
            keys[i] = cluster_center * 10000 + offset_dist(rng);
        }
    }
    else if(distribution == "skewed")
    {
        // Simple skewed distribution implementation
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < num_keys; ++i) {
            double x = dist(rng);
            keys[i] = static_cast<uint32_t>(pow(x, 3) * (UINT32_MAX - 1)) + 1; // Skewed towards lower values
        }
    }
    else {
        throw std::invalid_argument("Unsupported distribution type");
    }
}

//Init hash table
void HashTableBenchmark::initHashTable(std::vector<HashEntry>& table, size_t table_size)
{
    table.resize(table_size);
    std::fill(table.begin(), table.end(), HashEntry{EMPTY_KEY, 0});
}

//build hash table on host
void HashTableBenchmark::buildLinearHashTable(std::vector<HashEntry>& table, std::vector<uint64_t>& keys, double load_factor)
{
    if(load_factor <= 0.0 || load_factor > 1.0) {
        throw std::invalid_argument("Load factor must be in (0, 1]");
    }
    size_t table_size = static_cast<size_t>(std::ceil(keys.size() / load_factor));

    table.resize(table_size);

    std::fill(table.begin(), table.end(), HashEntry{EMPTY_KEY, 0});

    //fill the table
    for(size_t i = 0; i < keys.size(); i++) {
        uint64_t key = keys[i];
        uint64_t value = i + 1;
        uint32_t idx = hash32(key, table_size);

        // Linear probing for collision resolution
        while(true) {
            if(table[idx].key == EMPTY_KEY) {
                table[idx].key = key;
                table[idx].value = value;
                break;
            } else if(table[idx].key == key) {
                // Update existing key
                table[idx].value = value;
                break;
            }
            idx = (idx + 1) % table_size; // Wrap around
        }
    }
}

void HashTableBenchmark::generateAccessPatterns(std::vector<uint32_t>& pattern, size_t count, size_t data_size, size_t pattern_length)
{
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(0, data_size - 1);

    pattern.resize(count * pattern_length);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < pattern_length; ++j) {
            pattern[i * pattern_length + j] = dist(rng);
        }
    }
}

void HashTableBenchmark::runBenchmarkLinearProbingWithNoMixOps(
    const std::vector<size_t> table_sizes,
    const std::vector<double> load_factors,
    const std::vector<std::string> distributions,
    int num_iterations,
    std::vector<HashTableBenchmarkResult>& results
)
{ 
    // Implementation of the benchmark
    for(size_t table_size : table_sizes)
    {
        for(double lf : load_factors)
        {
            for(const std::string& dist : distributions)
            {
                std::cout<<"Table Size: "<<pretty_print_number(table_size)<<", Load Factor: "<<lf<<", Distribution: "<<dist<<std::endl;

                // Calculate number of inserts as next power of 2
                size_t num_inserts = static_cast<size_t>(table_size * lf);
                // if (num_inserts == 0) num_inserts = 1;
                // else num_inserts = 1ULL << (sizeof(size_t) * 8 - __builtin_clzll(num_inserts - 1));

                size_t num_queries = num_inserts;

                size_t num_deletes = num_inserts;

                //Generate test keys
                std::vector<uint64_t>keys;
                std::vector<uint64_t>query_keys;
                std::vector<uint64_t>results_query(num_queries, 0);
                std::vector<uint64_t>delete_keys;
                std::vector<HashEntry>hash_table(table_size);

                generateTestKeys(keys, num_inserts, dist);
                //generateTestKeys(query_keys, num_queries, dist);
                //generateTestKeys(delete_keys, num_deletes, dist);
                
                query_keys = keys; //lookup same keys as inserted
                delete_keys = keys; //delete same keys as inserted

                initHashTable(hash_table, table_size);

                std::cout<< "Num Inserts: " << pretty_print_number(num_inserts) << ", Num Queries: " << pretty_print_number(num_queries) << ", Num Deletes: " << pretty_print_number(num_deletes) << std::endl;

                std::vector<KernelType> kernels_array = {
                    KernelType::INSERT,
                    KernelType::LOOKUP,
                    KernelType::DELETE
                };

                //Launch Parameters
                int threads_per_block = 1024;
                int num_blocks = (num_inserts + threads_per_block -1)/threads_per_block;
                std::cout<<"Num Blocks: "<<num_blocks<<", Threads per Block: "<<threads_per_block<<std::endl;

                LinearProbeConfig lp_config;
                lp_config.load_factor = lf;
                lp_config.threads_per_block = threads_per_block;
                lp_config.blocks_per_grid = num_blocks;
                lp_config.numIterations = num_iterations;
                lp_config.insert_keys = &keys;
                lp_config.query_keys = &query_keys;
                lp_config.results = &results_query;
                lp_config.delete_keys = &delete_keys;
                lp_config.table = hash_table.data();
                lp_config.table_size = table_size;
                lp_config.num_inserts = num_inserts;
                lp_config.num_queries = num_queries;
                lp_config.num_deletes = num_deletes;
                lp_config.max_probes = 64;
                lp_config.kernels = kernels_array;

                //launch linear probing kernel with no mix ops
                {
                    lp_launch_kernel_with_no_mix_ops(
                        lp_config,
                        table_size,
                        threads_per_block,
                        num_blocks,
                        num_iterations,
                        VERIFICATION_VERBOSE //verification verbose, turn it true for small table sizes (< 1<<16)
                    );

                    HashTableBenchmarkResult result(
                    "Linear Probing (No Mix Ops)- " + dist + " - LF: " + std::to_string(lf) + " - Size: " + std::to_string(table_size),
                    {
                        {KernelType::INSERT, lp_config.elapsed_times[KernelType::INSERT]},
                        {KernelType::LOOKUP, lp_config.elapsed_times[KernelType::LOOKUP]},
                        {KernelType::DELETE, lp_config.elapsed_times[KernelType::DELETE]}
                    },
                    {
                        {KernelType::INSERT, bool(VERIFICATION_VERBOSE) ? lp_config.throughput_mlops[KernelType::INSERT] : static_cast<double>(num_inserts)/(lp_config.elapsed_times[KernelType::INSERT]/1000.0)/1e6},
                        {KernelType::LOOKUP, bool(VERIFICATION_VERBOSE) ? lp_config.throughput_mlops[KernelType::LOOKUP] : static_cast<double>(num_queries)/(lp_config.elapsed_times[KernelType::LOOKUP]/1000.0)/1e6},
                        {KernelType::DELETE, bool(VERIFICATION_VERBOSE) ? lp_config.throughput_mlops[KernelType::DELETE] : static_cast<double>(num_deletes)/(lp_config.elapsed_times[KernelType::DELETE]/1000.0)/1e6}
                    });
                    
                    results.push_back(result);
                    results.back().print();
                }
            }
        }
    }
}

void HashTableBenchmark::runBenchmarkLinearProbingWithMixOps(
    const std::vector<size_t> table_sizes,
    const std::vector<double> load_factors,
    const std::vector<std::string> distributions,
    const std::array<double, 3> distribution_ratio, //{insert_ratio, lookup_ratio, delete_ratio}
    int num_iterations,
    std::vector<HashTableBenchmarkResult>& results
)
{
    // Implementation of the benchmark
    for(size_t table_size : table_sizes)
    {
        for(double lf : load_factors)
        {
            for(const std::string& dist : distributions)
            {
                std::cout<<"Table Size: "<<pretty_print_number(table_size)<<", Load Factor: "<<lf<<", Distribution: "<<dist<<std::endl;

                // Calulcate number of inserts, lookup, deletes based on distribution ratio and load factor
                size_t num_inserts = static_cast<size_t>(table_size * lf);
                size_t total_ops = static_cast<size_t>(num_inserts / distribution_ratio[0]);
                size_t num_queries = static_cast<size_t>(total_ops * distribution_ratio[1]);
                size_t num_deletes = static_cast<size_t>(total_ops * distribution_ratio[2]);

                std::cout<< "Num Inserts: " << pretty_print_number(num_inserts) << ", Num Queries: " << pretty_print_number(num_queries) << ", Num Deletes: " << pretty_print_number(num_deletes) << std::endl;

                std::cout<<"Distribution Ratio (Insert:Lookup:Delete): "<<distribution_ratio[0]<<":"<<distribution_ratio[1]<<":"<<distribution_ratio[2]<<std::endl;

                std::cout<<"Total Ops: "<<pretty_print_number(num_inserts + num_queries + num_deletes)<<std::endl;

                std::vector<uint64_t>keys;
                std::vector<uint64_t>query_keys;
                std::vector<uint64_t>delete_keys;
                std::vector<uint64_t>totalops_results(total_ops, static_cast<uint64_t>(EMPTY_KEY));

                std::vector<Operation> mix_ops;
                mix_ops.reserve(num_inserts + num_queries + num_deletes);
                
                //Generate test keys
                generateTestKeys(keys, num_inserts, dist);

                //fill the mix_ops with insert first
                for(auto key : keys) {
                    mix_ops.push_back(Operation{OperationType::INSERT, key});
                }

                query_keys = keys; 
                delete_keys = keys;

                //chose lookup and delete keys from inserted keys
                query_keys.resize(num_queries);

                for(auto key : query_keys) {
                    mix_ops.push_back(Operation{OperationType::LOOKUP, key});
                }
                delete_keys.resize(num_deletes);

                for(auto key : delete_keys) {
                    mix_ops.push_back(Operation{OperationType::DELETE, key});
                }

                //shuffle the mix_ops
                std::shuffle(mix_ops.begin(), mix_ops.end(), rng);

                std::vector<HashEntry> hash_table(table_size);
                initHashTable(hash_table, table_size);

                //Launch Parameters
                int threads_per_block = 1024;
                int num_blocks = (mix_ops.size() + threads_per_block -1)/threads_per_block;

                std::cout<<"Num Blocks: "<<num_blocks<<", Threads per Block: "<<threads_per_block<<std::endl;

                LinearProbeConfig lp_config;

                lp_config.load_factor = lf;
                lp_config.threads_per_block = threads_per_block;
                lp_config.blocks_per_grid = num_blocks;
                lp_config.numIterations = num_iterations;
                lp_config.total_ops_results = totalops_results.data();
                lp_config.table = hash_table.data();
                lp_config.table_size = table_size;
                lp_config.max_probes = 64;
                lp_config.kernels = {KernelType::MIX_OPS};

                //Linear Probe with mix Ops
                {
                    lp_launch_kernel_with_mix_ops(
                        lp_config,
                        mix_ops,
                        threads_per_block,
                        num_blocks,
                        num_iterations,
                        (bool)VERIFICATION_VERBOSE //verification verbose, turn it true for small table sizes (< 1<<16)
                    );

                    HashTableBenchmarkResult result(
                    "Linear Probing (Mix Ops)- " + dist + " - LF: " + std::to_string(lf) + " - Size: " + std::to_string(table_size),
                    {
                        {KernelType::MIX_OPS, lp_config.elapsed_times[KernelType::MIX_OPS]}
                    },
                    {
                        {KernelType::MIX_OPS, (bool)VERIFICATION_VERBOSE ? lp_config.throughput_mlops[KernelType::MIX_OPS] : static_cast<double>(mix_ops.size())/(lp_config.elapsed_times[KernelType::MIX_OPS]/1000.0)/1e6} //MOPS
                    });
                    
                    results.push_back(result);
                    results.back().print();
                }
            }
        }
    }
}

void HashTableBenchmark::runBenchmarkHiveHashTableWithMixOps(
    HashTableDataLayout layout,
    const size_t table_size,
    const double load_factor,
    const std::string& distribution,
    const std::array<double, 3> distribution_ratio, //{insert_ratio, lookup_ratio, delete_ratio}
    const std::string& hash_policy,
    int num_iterations,
    int threads_per_block_,
    HashTableBenchmarkResult &result
)
{
    // Implementation of the benchmark for Hive Hash Table with mixed operations
    std::cout<<"Table Size: "<<pretty_print_number(table_size)<<", Load Factor: "<<load_factor<<std::endl;
    std::cout<<"Distribution: "<<distribution<<std::endl;
    std::cout<<"Hash Policy: "<<hash_policy<<std::endl;
    
    size_t total_ops = static_cast<size_t>(table_size * load_factor);
    size_t num_inserts = static_cast<size_t>(total_ops * distribution_ratio[0]);
    size_t num_queries = static_cast<size_t>(total_ops * distribution_ratio[1]);
    size_t num_deletes = static_cast<size_t>(total_ops * distribution_ratio[2]);

    std::cout<< "Num Inserts: " << pretty_print_number(num_inserts) << ", Num Queries: " << pretty_print_number(num_queries) << ", Num Deletes: " << pretty_print_number(num_deletes) << std::endl;
      
    std::cout<<"Distribution Ratio (Insert:Lookup:Delete): "<<distribution_ratio[0]<<":"<<distribution_ratio[1]<<":"<<distribution_ratio[2]<<std::endl;

    std::cout<<"Total Ops: "<<pretty_print_number(num_inserts + num_queries + num_deletes)<<std::endl;

    std::vector<uint64_t>keys;
    std::vector<uint64_t>query_keys;
    std::vector<uint64_t>delete_keys;
    std::vector<uint64_t>totalops_results(total_ops, 0);

    std::vector<Operation> mix_ops;
    mix_ops.reserve(num_inserts + num_queries + num_deletes);

    size_t num_keys = num_inserts > 0 ? num_inserts : (num_queries > 0 ? num_queries : (num_deletes > 0 ? num_deletes : 1));
    //Generate test keys
    generateTestKeys(keys, num_keys, distribution);

    //fill the mix_ops with insert first
    if(num_inserts > 0)
    {
        for(auto key : keys) {
            mix_ops.push_back(Operation{OperationType::INSERT, key});
        }
    }

    //chose lookup and delete keys from inserted keys
    query_keys = keys;
    delete_keys = keys;

    query_keys.resize(num_queries);
    if(num_queries > 0) {
        for(auto key : query_keys) {
            mix_ops.push_back(Operation{OperationType::LOOKUP, key});
        }
    }
    
    
    delete_keys.resize(num_deletes);
    if(num_deletes > 0) {
        for(auto key : delete_keys) {
            mix_ops.push_back(Operation{OperationType::DELETE, key});
        }
    }

    //shuffle the mix_ops
    std::shuffle(mix_ops.begin(), mix_ops.end(), rng);

    //Launch Parameters
    //Assign each thread a single operation
    int threads_per_block = threads_per_block_;
    int num_blocks = (mix_ops.size() + threads_per_block - 1) / threads_per_block;
    
    std::cout<<"Num Blocks: "<<num_blocks<<", Threads per Block: "<<threads_per_block<<std::endl;
    
    double elapsed_time = 0.0;

    hash_table_kernel_dispatch(
        mix_ops.data(),
        mix_ops.size(),
        table_size,
        threads_per_block,
        num_blocks,
        num_iterations,
        elapsed_time,
        totalops_results.data(),
        (bool)VERIFICATION_VERBOSE,
        layout,
        hash_policy
    );

    std::cout << "Throughput: " << static_cast<double>(mix_ops.size())/(elapsed_time/1000.0)/1e6 << std::endl;
    result = HashTableBenchmarkResult(
        "Hive Hash Table (Mix Ops)- " + distribution + " - LF: " + std::to_string(load_factor) + " - Size: " + std::to_string(table_size),
        {
            {KernelType::MIX_OPS, elapsed_time}
        },
        {
            {KernelType::MIX_OPS, static_cast<double>(mix_ops.size())/(elapsed_time/1000.0)/1e6} //MOPS
        }
    );       
}

//Run Hash Table All Lookups operation (varying positive ratio of keys (0-100%))
void HashTableBenchmark::runAllLookupsBenchmark(
    HashTableDataLayout layout,
    const size_t table_size,
    const double load_factor,
    const std::string& distribution,
    size_t num_iterations,
    size_t num_operations,
    int threads_per_block_,
    HashTableBenchmarkResult &result
)
{
    std::cout<<"Table Size: "<<pretty_print_number(table_size)<<", Load Factor: "<<load_factor<<std::endl;
    std::cout<<"Hash Policy: "<<distribution<<std::endl;

    size_t record_count = static_cast<size_t>(table_size * load_factor);

    //prefill with 1..N distinct keys
    std::vector<uint32_t> keys_universe;
    keys_universe.reserve(record_count);
    generateTestKeys(keys_universe, record_count, "unique");

    std::mt19937 rng(std::random_device{}());

    std::vector<Operation> prefill_ops;
    for(const auto& key : keys_universe) {
        prefill_ops.push_back(Operation{OperationType::INSERT, key});
    }
    std::shuffle(prefill_ops.begin(), prefill_ops.end(), rng);


    std::vector<Operation> lookup_ops_100pct;
    for(const auto& key : keys_universe) {
        lookup_ops_100pct.push_back(Operation{OperationType::LOOKUP, key});
    }
    std::shuffle(lookup_ops_100pct.begin(), lookup_ops_100pct.end(), rng);

    double elapsed_time_100pct = 0.0;
    std::vector<uint64_t> results_100pct(lookup_ops_100pct.size(), 0);
    hash_table_kernel_dispatch_YCSB(
        prefill_ops.data(),
        lookup_ops_100pct.data(),
        record_count,
        lookup_ops_100pct.size(),
        table_size,
        threads_per_block_,
        num_iterations,
        elapsed_time_100pct,
        results_100pct.data(),
        (bool)VERIFICATION_VERBOSE,
        layout
    );

    std::cout << "Throughput (100% exist): " << static_cast<double>(lookup_ops_100pct.size())/(elapsed_time_100pct/1000.0)/1e6 << std::endl;
    
    //75pct exist
    std::vector<Operation> lookup_ops_75pct;
    for(size_t i=0; i < keys_universe.size() * 0.75; i++) {
        lookup_ops_75pct.push_back(Operation{OperationType::LOOKUP, keys_universe[i]});
    }
    for(size_t i=0; i < keys_universe.size() * 0.25; i++) {
        lookup_ops_75pct.push_back(Operation{OperationType::LOOKUP, keys_universe[i] + keys_universe.size()});
    }
    std::shuffle(lookup_ops_75pct.begin(), lookup_ops_75pct.end(), rng);

    double elapsed_time_75pct = 0.0;
    std::vector<uint64_t> results_75pct(lookup_ops_75pct.size(), 0);
    hash_table_kernel_dispatch_YCSB(
        prefill_ops.data(),
        lookup_ops_75pct.data(),
        record_count,
        lookup_ops_75pct.size(),
        table_size,
        threads_per_block_,
        num_iterations,
        elapsed_time_75pct,
        results_75pct.data(),
        (bool)VERIFICATION_VERBOSE,
        layout
    );
    std::cout << "Throughput (75% exist): " << static_cast<double>(lookup_ops_75pct.size())/(elapsed_time_75pct/1000.0)/1e6 << std::endl;

    //50 pct exist
    std::vector<Operation> lookup_ops_50pct;
    for(size_t i=0; i < keys_universe.size() * 0.5; i++) {
        lookup_ops_50pct.push_back(Operation{OperationType::LOOKUP, keys_universe[i]});
    }
    for(size_t i=0; i < keys_universe.size() * 0.5; i++) {
        lookup_ops_50pct.push_back(Operation{OperationType::LOOKUP, keys_universe[i] + keys_universe.size()});
    }

    double elapsed_time_50pct = 0.0;
    std::vector<uint64_t> results_50pct(lookup_ops_50pct.size(), 0);
    hash_table_kernel_dispatch_YCSB(
        prefill_ops.data(),
        lookup_ops_50pct.data(),
        record_count,
        lookup_ops_50pct.size(),
        table_size,
        threads_per_block_,
        num_iterations,
        elapsed_time_50pct,
        results_50pct.data(),
        (bool)VERIFICATION_VERBOSE,
        layout
    );
    std::cout << "Throughput (50% exist): " << static_cast<double>(lookup_ops_50pct.size())/(elapsed_time_50pct/1000.0)/1e6 << std::endl;

    //25 pct exist
    std::vector<Operation> lookup_ops_25pct;
    for(size_t i=0; i < keys_universe.size() * 0.25; i++) {
        lookup_ops_25pct.push_back(Operation{OperationType::LOOKUP, keys_universe[i]});
    }
    for(size_t i=0; i < keys_universe.size() * 0.75; i++) {
        lookup_ops_25pct.push_back(Operation{OperationType::LOOKUP, keys_universe[i] + keys_universe.size()});
    }
    std::shuffle(lookup_ops_25pct.begin(), lookup_ops_25pct.end(), rng);

    double elapsed_time_25pct = 0.0;
    std::vector<uint64_t> results_25pct(lookup_ops_25pct.size(), 0);
    hash_table_kernel_dispatch_YCSB(
        prefill_ops.data(),
        lookup_ops_25pct.data(),
        record_count,
        lookup_ops_25pct.size(),
        table_size,
        threads_per_block_,
        num_iterations,
        elapsed_time_25pct,
        results_25pct.data(),
        (bool)VERIFICATION_VERBOSE,
        layout
    );
    std::cout << "Throughput (25% exist): " << static_cast<double>(lookup_ops_25pct.size())/(elapsed_time_25pct/1000.0)/1e6 << std::endl;

    //0 pct exist
    std::vector<Operation> lookup_ops_0pct;
    for(size_t i=0; i < keys_universe.size(); i++) {
        lookup_ops_0pct.push_back(Operation{OperationType::LOOKUP, keys_universe[i] + keys_universe.size()});
    }
    std::shuffle(lookup_ops_0pct.begin(), lookup_ops_0pct.end(), rng);

    double elapsed_time_0pct = 0.0;
    std::vector<uint64_t> results_0pct(lookup_ops_0pct.size(), 0);
    hash_table_kernel_dispatch_YCSB(
        prefill_ops.data(),
        lookup_ops_0pct.data(),
        record_count,
        lookup_ops_0pct.size(),
        table_size,
        threads_per_block_,
        num_iterations,
        elapsed_time_0pct,
        results_0pct.data(),
        (bool)VERIFICATION_VERBOSE,
        layout
    );
    std::cout << "Throughput (0% exist): " << static_cast<double>(lookup_ops_0pct.size())/(elapsed_time_0pct/1000.0)/1e6 << std::endl;
}

void HashTableBenchmark::runYCSBBenchmarkWithHiveHashTable(
    HashTableDataLayout layout,
    const size_t table_size,
    const double load_factor,
    const YCSBWorkLoadType workload_type,
    size_t num_operations,
    size_t num_iterations,
    int threads_per_block_,
    HashTableBenchmarkResult &result
)
{
    // Implementation of the benchmark for Hive Hash Table with YCSB workload
    std::cout<<"Table Size: "<<pretty_print_number(table_size)<<", Load Factor: "<<load_factor<<std::endl;
    std::cout<<"YCSB Workload Type: ";

    double read_ratio = 0.0;
    double update_ratio = 0.0;
    double rmw_ratio = 0.0;
    double latest_insert_ratio = 0.0;

    switch(workload_type)
    {
        case YCSBWorkLoadType::WORKLOAD_A:
            std::cout<<"WORKLOAD_A (50% read, 50% write)"<<std::endl;
            read_ratio = 0.5;
            update_ratio = 0.5;
            break;
        case YCSBWorkLoadType::WORKLOAD_B:
            std::cout<<"WORKLOAD_B (95% read, 5% write)"<<std::endl;
            read_ratio = 0.95;
            update_ratio = 0.05;
            break;
        case YCSBWorkLoadType::WORKLOAD_C:
            std::cout<<"WORKLOAD_C (100% read)"<<std::endl;
            read_ratio = 1.0;
            update_ratio = 0.0;
            break;
        case YCSBWorkLoadType::WORKLOAD_D:
            // 5% INSERT: Inserting new keys (increasing the load factor during the run). 95% LOOKUP: Querying these recently inserted keys.
            std::cout<<"WORKLOAD_D (95% read, 5% new inserts)"<<std::endl;
            read_ratio = 0.95;
            latest_insert_ratio = 0.05;
            rmw_ratio = 0.0;
            break;
        case YCSBWorkLoadType::WORKLOAD_F:
            std::cout<<"WORKLOAD_F (50% read, 50% read-modify-write)"<<std::endl;
            read_ratio = 0.5;
            rmw_ratio = 0.5;
            break;
        default:
            std::cerr<<"Unsupported YCSB Workload Type"<<std::endl;
            return;
    }

    size_t record_count = static_cast<size_t>(table_size * load_factor);
    
    std::vector<Operation> prefill_ops;
    prefill_ops.reserve(record_count);

    //prefill with 1..N distinct keys
    for(size_t i = 1; i <= record_count; i++) {
        prefill_ops.push_back(Operation{OperationType::INSERT, static_cast<uint64_t>(i)});
    }

    std::vector<Operation> ycsb_workload_ops;
    ycsb_workload_ops.reserve(num_operations);

    size_t num_reads = std::min(static_cast<size_t>(num_operations * read_ratio), record_count);
    size_t num_updates = std::min(static_cast<size_t>(num_operations * update_ratio), record_count);
    size_t num_rmws = std::min(static_cast<size_t>(num_operations * rmw_ratio), record_count);
    size_t num_latest_inserts = std::min(static_cast<size_t>(num_operations * latest_insert_ratio), record_count);
    std::cout<<"YCSB Workload Ops - Reads: "<<num_reads<<", Updates: "<<num_updates<<", RMWs: "<<num_rmws<<", Latest Inserts: "<<num_latest_inserts<<std::endl;

    ZipfAlias zipf_dist(record_count, 1.5 /*skewness*/);

    //Push n_reads
    
    //Latest inserts with new keys
    std::vector<uint64_t> latest_insert_keys;

    for(size_t i = 0; i < num_latest_inserts; i++){
        ycsb_workload_ops.push_back(Operation{OperationType::INSERT, record_count + i + 1});
        latest_insert_keys.push_back(record_count + i + 1);
    }
    
    for(size_t i = 0; i < num_reads; i++){
        if(workload_type == YCSBWorkLoadType::WORKLOAD_D){
            ycsb_workload_ops.push_back(Operation{OperationType::LOOKUP, latest_insert_keys[i % latest_insert_keys.size()]});
        }
        else{
            ycsb_workload_ops.push_back(Operation{OperationType::LOOKUP, zipf_dist.sample()});
        }
    }

    for(size_t i = 0; i < num_updates; i++){
        ycsb_workload_ops.push_back(Operation{OperationType::INSERT, zipf_dist.sample()}); //Insert can perform blind update
    }

    for(size_t i = 0; i < num_rmws; i++){
        ycsb_workload_ops.push_back(Operation{OperationType::ATOMIC_INC, zipf_dist.sample()});
    }


    //Shuffle it with mt twister rng
    std::mt19937 rng(std::random_device{}());
    std::shuffle(ycsb_workload_ops.begin(), ycsb_workload_ops.end(), rng);

    double elapsed_time = 0.0;
    std::vector<uint64_t> totalops_results(num_operations, 0);

    hash_table_kernel_dispatch_YCSB(
        prefill_ops.data(),
        ycsb_workload_ops.data(),
        record_count,
        num_operations,
        table_size,
        threads_per_block_,
        num_iterations,
        elapsed_time,
        totalops_results.data(),
        (bool)VERIFICATION_VERBOSE,
        layout

    );

    result = HashTableBenchmarkResult(
        "Hive Hash Table YCSB - " + std::to_string(num_operations) + " ops - LF: " + std::to_string(load_factor) + " - Size: " + std::to_string(table_size),
        {
            {KernelType::MIX_OPS, elapsed_time}
        },
        {
            {KernelType::MIX_OPS, static_cast<double>(num_operations)/(elapsed_time/1000.0)/1e6} //MOPS
        }
    );
    std::cout << "Throughput: " << static_cast<double>(num_operations)/(elapsed_time/1000.0)/1e6 << std::endl;

}
