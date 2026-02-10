#include "utils.h"

#include <iostream>
#include <vector>
#include <map>
#include <array>
#include <string> // Required for std::stoi, std::stod
#include <iomanip> // Required for std::fixed and std::setprecision

#include "hashTableBenchmark.h"
#include "GPUTimer.h"
#include "hash_table_struct.h"
#include "kernels.cuh"
#include "cuda_helper.cuh"
#include "linear_probe_ht.cuh"
#include "utils.h"

// Default benchmark parameters
size_t g_table_size = 25; // Default to 2^25
double g_load_factor = 0.5;    // Default to 90%
int g_num_iterations = 10;    // Default to 10 iterations
int g_threads_per_block = 256; // Default to 256 threads per block
size_t n_operations_ycsb = 1<<23; // Default to 2^20 operations

//data layout type
//HybridSoA-AoS, AaoS, Aaos-LeadMetaData
std::string g_data_layout = "AaoS-LeadMetaData";   // Array of Aligned Structs (AaoS), Hybrid SOA-AOS (SoA)
std::string ycsb_workload_type = "C"; //Default YCSB Workload Type

void print_help() {
    std::cout << "Usage: ./bin/hive_hash_table_ycsb [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  --table_size <size>        Set the hash table size (default: 2^25)\n";
    std::cout << "  --load_factor <factor>     Set the load factor (e.g., 0.9 for 90%) (default: 0.9)\n";
    std::cout << "  --num_iterations <count>   Set the number of benchmark iterations (default: 10)\n";
    std::cout << "  --data_layout <type>       Set the data layout type (HybridSoA-AoS, AaoS, AaoS-LeadMetaData) (default: HybridSoA-AoS)\n";
    std::cout << "  --threads-per-block <count> Set the number of threads per block (default: 256)\n";
    std::cout << "  --ycsb_workload_type <type> Set the YCSB workload type (A, B, C, D) (default: A)\n";
    std::cout << "  --num_operations_ycsb <count> Set the number of operations for YCSB benchmark (default: 2^20)\n";
    std::cout << "  --help                     Display this help message\n";
}

int main(int argc, char* argv[])
{
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--table_size") {
            if (i + 1 < argc) {
                g_table_size = std::stoul(argv[++i]);
            }
        } else if (arg == "--load_factor") {
            if (i + 1 < argc) {
                g_load_factor = std::stod(argv[++i]);
            }
        } else if (arg == "--num_iterations") {
            if (i + 1 < argc) {
                g_num_iterations = std::stoi(argv[++i]);
            } 
        } else if (arg == "--threads-per-block") {
            if (i + 1 < argc) {
                g_threads_per_block = std::stoi(argv[++i]);
            }
        } else if (arg == "--data_layout") {
            if (i + 1 < argc) {
                g_data_layout = argv[++i];
            }
        } else if (arg == "--ycsb_workload_type") {
            if (i + 1 < argc) {
                ycsb_workload_type = argv[++i];
            }
        } else if (arg == "--num_operations_ycsb") {
            if (i + 1 < argc) {
                n_operations_ycsb = std::stoul(argv[++i]);
            }
        } else if (arg == "--help") {
            print_help();
            return 0;
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            print_help();
            return 1;
        }
    }

    checkCudaDevice();

    HashTableDataLayout layout = HashTableDataLayout::ARRAY_OF_ALIGNED_STRUCTS;

    if(strcmp(g_data_layout.c_str(), "HybridSoA-AoS") == 0)
    {
        std::cout<<"Using HybridSoA-AoS data layout for Hive Hash Table Benchmark"<<std::endl;
        layout = HashTableDataLayout::HYBRID_SOA_AOS;
    }
    else if(strcmp(g_data_layout.c_str(), "AaoS") == 0)
    {
        std::cout<<"Using AaoS data layout for Hive Hash Table Benchmark"<<std::endl;
        layout = HashTableDataLayout::ARRAY_OF_ALIGNED_STRUCTS;
    }
    else if(strcmp(g_data_layout.c_str(), "AaoS-LeadMetaData") == 0)
    {
        std::cout<<"Using AaoS-LeadMetaData data layout for Hive Hash Table Benchmark"<<std::endl;
        layout = HashTableDataLayout::ARRAY_OF_ALIGNED_STRUCTS_LEAD_METADATA;
    }
    else
    {
        std::cerr<<"Unknown data layout type. Supported types: HybridSoA-AoS, AaoS, AaoS-LeadMetaData"<<std::endl;
        return 1;
    }

    HashTableBenchmark benchmark;
    
    HashTableBenchmarkResult result("Hive Hash Table with Mix Ops", {}, {});

    YCSBWorkLoadType workload_type = YCSBWorkLoadType::WORKLOAD_A;

    if(strcmp(ycsb_workload_type.c_str(), "A") == 0)
        workload_type = YCSBWorkLoadType::WORKLOAD_A;
    else if(strcmp(ycsb_workload_type.c_str(), "B") == 0)
        workload_type = YCSBWorkLoadType::WORKLOAD_B;
    else if(strcmp(ycsb_workload_type.c_str(), "C") == 0)
        workload_type = YCSBWorkLoadType::WORKLOAD_C;
    else if(strcmp(ycsb_workload_type.c_str(), "D") == 0)
        workload_type = YCSBWorkLoadType::WORKLOAD_D;

    std::cout << "Starting YCSB Workload: " << ycsb_workload_type << std::endl;
    std::cout << "Config: Table Size=" << (1 << g_table_size) 
              << ", Load Factor=" << std::fixed << std::setprecision(2) << g_load_factor * 100 << "%"
              << ", Operations=" << n_operations_ycsb
              << ", Iterations=" << g_num_iterations << "\n";

    benchmark.runYCSBBenchmarkWithHiveHashTable(
        layout,
        static_cast<size_t>(1) << g_table_size,
        g_load_factor,
        workload_type,
        n_operations_ycsb,
        g_num_iterations,
        g_threads_per_block,
        result
    );
    return 0;
}
