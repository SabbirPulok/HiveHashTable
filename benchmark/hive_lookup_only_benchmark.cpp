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
#include "utils.h"

// Default benchmark parameters
size_t g_table_size = 24; 
double g_load_factor = 0.7;    
double g_insert_ratio = 0.0;   // Default to 50% insert
double g_lookup_ratio = 1.0;   // Default to 30% lookup
double g_delete_ratio = 0.0;   // Default to 20% delete
int g_num_iterations = 10;    // Default to 10 iterations
size_t g_num_operations = 1<<23; // Default to 2^23 operations
int g_threads_per_block = 256; // Default to 256 threads per block
std::string g_distribution = "uniform"; // Default distribution


//data layout type
//HybridSoA-AoS, AaoS, Aaos-LeadMetaData
std::string g_data_layout = "HybridSoA-AoS";   // Array of Aligned Structs (AaoS), Hybrid SOA-AOS (SoA)
std::string g_hash_policy = "Default2Hash"; // Default Hash Policy


enum class HashTableType{
    LINEAR_PROBING,
    CUCKOO_HASHING
};

void print_help() {
    std::cout << "Usage: ./bin/hive_hash_table_benchmark [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  --table_size <size>        Set the hash table size (default: 2^25)\n";
    std::cout << "  --load_factor <factor>     Set the load factor (e.g., 0.9 for 90%) (default: 0.9)\n";
    std::cout << "  --insert_ratio <ratio>     Set the insertion ratio (e.g., 1.0 for 100%) (default: 1.0)\n";
    std::cout << "  --lookup_ratio <ratio>     Set the lookup ratio (e.g., 0.5 for 50%) (default: 0.0)\n";
    std::cout << "  --delete_ratio <ratio>     Set the deletion ratio (e.g., 0.1 for 10%) (default: 0.0)\n";
    std::cout << "  --num_iterations <count>   Set the number of benchmark iterations (default: 10)\n";
    std::cout << "  --n_operations <count>     Set the number of lookup operations (default: 2^23)\n";
    std::cout << "  --data_layout <type>       Set the data layout type (HybridSoA-AoS, AaoS, AaoS-LeadMetaData) (default: HybridSoA-AoS)\n";
    std::cout << "  --hash_policy <type>       Set the hash policy (Default2Hash, TripleHash, MurmurCityHash, MurmurCityBitHash, Lookup2Hash, Lookup3Hash) (default: Default2Hash)\n";
    std::cout << "  --threads-per-block <count> Set the number of threads per block (default: 256)\n";
    std::cout << "  --distribution <type>      Set the key distribution (uniform, skewed, clustered, unique) (default: uniform)\n";
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
        } else if (arg == "--insert_ratio") {
            if (i + 1 < argc) {
                g_insert_ratio = std::stod(argv[++i]);
            }
        } else if (arg == "--lookup_ratio") {
            if (i + 1 < argc) {
                g_lookup_ratio = std::stod(argv[++i]);
            } 
        } else if (arg == "--delete_ratio") {
            if (i + 1 < argc) {
                g_delete_ratio = std::stod(argv[++i]);
            } 
        } else if (arg == "--num_iterations") {
            if (i + 1 < argc) {
                g_num_iterations = std::stoi(argv[++i]);
            }
        } else if (arg == "--n_operations") {
            if (i + 1 < argc) {
                g_num_operations = std::stoul(argv[++i]);
            }
        } else if (arg == "--distribution") {
            if (i + 1 < argc) {
                g_distribution = argv[++i];
            }
        } else if (arg == "--threads-per-block") {
            if (i + 1 < argc) {
                g_threads_per_block = std::stoi(argv[++i]);
            }
        } else if (arg == "--data_layout") {
            if (i + 1 < argc) {
                g_data_layout = argv[++i];
            }
        } else if (arg == "--hash_policy") {
            if (i + 1 < argc) {
                g_hash_policy = argv[++i];
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

    std::cout << "Starting Hash Table Benchmarks...\n";
    std::cout << "Config: Table Size=" << (1 << g_table_size) 
              << ", Load Factor=" << std::fixed << std::setprecision(2) << g_load_factor * 100 << "%"
              << ", Ratios(I:L:D)=" << g_insert_ratio * 100 << ":" << g_lookup_ratio * 100 << ":" << g_delete_ratio * 100 
              << "%, Iterations=" << g_num_iterations << "\n";

    std::cout << "Running Hive Hash Table with Mix Ops...\n";

    benchmark.runAllLookupsBenchmark(
        layout,
        static_cast<size_t>(1) << g_table_size,
        g_load_factor,
        g_distribution,
        g_num_iterations,
        g_num_operations,
        g_threads_per_block,
        result
    );

    return 0;
}
