#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <iostream>
#include <iomanip>
#include <chrono>

using KeyType = uint32_t;
using ValueType = uint32_t;

int main(int argc, char** argv){
    if(argc < 4) {
        std::cerr << "Usage: ./cuco_bench <table_size_power> <load_factor> <num_operations> [lookup_exist_ratio]" << std::endl;
        return 1;
    }

    int table_size_power = std::stoi(argv[1]);
    float load_factor = std::stof(argv[2]);
    size_t num_operations = std::stoull(argv[3]);
    float lookup_exist_ratio = 1.0f;
    if(argc >= 5) {
        lookup_exist_ratio = std::stof(argv[4]);
    }


    KeyType empty_key = 0xFFFFFFFF;
    ValueType empty_value = 0xFFFFFFFF;

    // The physical capacity is 2^power
    size_t capacity = 1ULL << table_size_power;
    
    // Warmup CUDA context
    cudaFree(0);

    // Initialize CUCO static map
    cuco::static_map<KeyType, ValueType> map(
        capacity, 
        cuco::empty_key<KeyType>{empty_key}, 
        cuco::empty_value<ValueType>{empty_value}
    );

    // Generate sequential unique keys [1, 2, ..., N]
    thrust::device_vector<KeyType> keys(num_operations);
    thrust::sequence(keys.begin(), keys.end(), 1); // Start at 1 to avoid empty key

    // Shuffle keys by sorting with random weights
    thrust::device_vector<uint32_t> weights(num_operations);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    thrust::transform(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(num_operations),
                      weights.begin(),
                      [seed] __device__ (size_t idx) {
                          thrust::default_random_engine rng(seed + idx);
                          thrust::uniform_int_distribution<uint32_t> dist;
                          return dist(rng);
                      });
    
    thrust::sort_by_key(weights.begin(), weights.end(), keys.begin());

    // Generate sequential values [1, 2, ..., N]
    thrust::device_vector<ValueType> values(num_operations);
    thrust::sequence(values.begin(), values.end(), 1);

    // Combine keys and values into cuco::pair
    auto pairs = cuda::make_transform_iterator(
        cuda::counting_iterator<std::size_t>{0},
        cuda::proclaim_return_type<cuco::pair<KeyType, ValueType>>(
            [k = thrust::raw_pointer_cast(keys.data()), v = thrust::raw_pointer_cast(values.data())] __device__(auto i) { 
                return cuco::make_pair(k[i], v[i]); 
            }
        )
    );
    
    // Timed bulk insertion
    cudaDeviceSynchronize(); 
    auto start = std::chrono::high_resolution_clock::now();
    
    map.insert(pairs, pairs + num_operations);
    
    cudaDeviceSynchronize(); 
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> insert_duration = end - start;
    std::cout << "CuCollections Insertion time: " << insert_duration.count() << " ms" << std::endl;

    double actual_load_factor = static_cast<double>(map.size()) / static_cast<double>(capacity);
    std::cout << "Load factor after insertion: " << actual_load_factor << std::endl;

    // Million Ops Per seconds
    double insert_mops = (static_cast<double>(num_operations) / 1e6) / (insert_duration.count() / 1000.0);
    std::cout << "CuCollections Insertion Throughput: " << insert_mops << " Mops" << std::endl;

    // Generate query keys based on exist ratio
    size_t num_queries = num_operations;
    size_t num_exist = static_cast<size_t>(num_queries * lookup_exist_ratio);
    thrust::device_vector<KeyType> query_keys(num_queries);
    
      // First, copy the existing keys up to num_exist
      if (num_exist > 0) {
          thrust::copy(keys.begin(), keys.begin() + num_exist, query_keys.begin());
      }
      
      // Then, generate non-existing keys for the remainder
      if (num_queries > num_exist) {
          // Generate non-existing keys starting above the maximum existing key
          KeyType max_key = num_operations + 1; // Assuming max key inserted is around num_operations
          thrust::sequence(query_keys.begin() + num_exist, query_keys.end(), max_key + 1);
      }
      
      // Shuffle the query keys so exists and non-exists are mixed
      thrust::transform(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(num_queries),
                        weights.begin(),
                        [seed] __device__ (size_t idx) {
                            thrust::default_random_engine rng(seed + idx + 100); // Different seed offset
                            thrust::uniform_int_distribution<uint32_t> dist;
                            return dist(rng);
                        });
      thrust::sort_by_key(weights.begin(), weights.end(), query_keys.begin());
    

    // Output array for finds. Cuco contains() outputs booleans.
    thrust::device_vector<bool> found_results(num_queries);
    
    // Timed bulk lookup
    cudaDeviceSynchronize(); 
    auto start_query = std::chrono::high_resolution_clock::now();
    
    map.contains(query_keys.begin(), query_keys.end(), found_results.begin());
    
    cudaDeviceSynchronize(); 
    auto end_query = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> lookup_duration = end_query - start_query;
    std::cout << "CuCollections Lookup time: " << lookup_duration.count() << " ms" << std::endl;

    size_t n_values_found = thrust::count(found_results.begin(), found_results.end(), true);

    auto lookup_mops = (static_cast<double>(num_queries) / 1e6) / (lookup_duration.count() / 1000.0);
    std::cout << "CuCollections Lookup Throughput: " << lookup_mops << " Mops" << std::endl;
    std::cout << "Successfully Found: " << n_values_found << " out of " << num_queries << std::endl;

    return 0;
}