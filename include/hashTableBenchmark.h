#pragma once
#include <random>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <map>

#include "utils.h"
#include "hash_table_struct.h"

static constexpr uint32_t EMPTY_KEY = 0ull; //sentinel

using KeyType = uint32_t;
using ValueType = uint32_t;

struct HashTableBenchmarkResult {
   std::string test_name;

   std::map<KernelType, double> latency_ms;
   std::map<KernelType, double> throughput_mlops;
   double memory_bandwidth_gbps;
   size_t memory_access;

    HashTableBenchmarkResult(std::string _name, std::map<KernelType, double> _latency_ms, std::map<KernelType, double> _throughput_mlops)
     : test_name(_name),
     latency_ms(std::move(_latency_ms)), 
      throughput_mlops(std::move(_throughput_mlops)), 
      memory_bandwidth_gbps(0.0), 
      memory_access(0) {
    }

   void inline print() const {
        std::cout << std::left << std::setw(25) << test_name << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::fixed << std::setprecision(3);

        //You know often times MIX_OPS is the only kernel being run, so we should check for every kernel existence
        if(latency_ms.find(KernelType::LOOKUP) != latency_ms.end()) {
            std::cout << "Avg Lookup Time (ms): " << latency_ms.at(KernelType::LOOKUP) << ", Throughput (MLOPS): " << throughput_mlops.at(KernelType::LOOKUP) << "\n";
        }
        if(latency_ms.find(KernelType::MIX_OPS) != latency_ms.end()) {
            std::cout << "Avg Mix Ops Time (ms): " << latency_ms.at(KernelType::MIX_OPS) << ", Throughput (MLOPS): " << throughput_mlops.at(KernelType::MIX_OPS) << "\n";
        }
        if(latency_ms.find(KernelType::INSERT) != latency_ms.end()) {
            std::cout << "Avg Insert Time (ms): " << latency_ms.at(KernelType::INSERT) << ", Throughput (MLOPS): " << throughput_mlops.at(KernelType::INSERT) << "\n";
        }
        if(latency_ms.find(KernelType::DELETE) != latency_ms.end()) {
            std::cout << "Avg Delete Time (ms): " << latency_ms.at(KernelType::DELETE) << ", Throughput (MLOPS): " << throughput_mlops.at(KernelType::DELETE) << "\n";
        }
        std::cout << "----------------------------------------" << std::endl;
   }
};

class HashTableBenchmark {
    private:
        std::mt19937_64 rng;
    public:
        HashTableBenchmark() : rng(std::random_device{}()) {};

        //Generate test keys for hash table based on distribution
        void generateTestKeys(std::vector<uint64_t>&keys, size_t num_keys, const std::string& distribution = "uniform");

        void generateTestKeys(std::vector<uint32_t>& keys, size_t num_keys, const std::string& distribution = "unique");
        //Init hash table
        void initHashTable(std::vector<HashEntry>& table, size_t table_size);
        
        //build hash table
        void buildLinearHashTable(std::vector<HashEntry>& table, std::vector<uint64_t>& keys, double load_factor);

        //generate random access patterns for memory benchmarking
        void generateAccessPatterns(std::vector<uint32_t>& pattern, size_t count, size_t data_size, size_t pattern_length);

        void runBenchmarkLinearProbingWithNoMixOps(
            const std::vector<size_t> table_sizes,
            const std::vector<double> load_factors,
            const std::vector<std::string> distributions,
            int num_iterations,
            std::vector<HashTableBenchmarkResult>& results
        );

        void runBenchmarkLinearProbingWithMixOps(
            const std::vector<size_t> table_sizes,
            const std::vector<double> load_factors,
            const std::vector<std::string> distributions,
            const std::array<double, 3> distribution_ratio, //{insert_ratio, lookup_ratio, delete_ratio}
            int num_iterations,
            std::vector<HashTableBenchmarkResult>& results
        );

        void runBenchmarkHiveHashTableWithMixOps(
            HashTableDataLayout layout,
            const size_t table_size,
            const double load_factor,
            const std::string& distribution,
            const std::array<double, 3> distribution_ratio, //{insert_ratio, lookup_ratio, delete_ratio}
            const std::string& hash_policy,
            int num_iterations,
            int threads_per_block_,
            HashTableBenchmarkResult& result
        );

        void runYCSBBenchmarkWithHiveHashTable(
            HashTableDataLayout layout,
            const size_t table_size,
            const double load_factor,
            const YCSBWorkLoadType workload_type,
            size_t num_operations,
            size_t num_iterations,
            int threads_per_block_,
            HashTableBenchmarkResult &result
        );

        void runAllLookupsBenchmark(
            HashTableDataLayout layout,
            const size_t table_size,
            const double load_factor,
            const std::string& distribution,
            size_t num_iterations,
            size_t num_operations,
            int threads_per_block_,
            HashTableBenchmarkResult &result
        );
};

// Warpspeed Benchmark Wrapper
void run_warpspeed_mixed_workload(
    Operation* h_ops,
    size_t num_ops,
    size_t table_size,
    size_t threads_per_block,
    size_t numIterations,
    double &elapsed_time,
    uint64_t* h_results,
    bool verify
);
