#include <warpcore/single_value_hash_table.cuh>
#include "../include/benchmark_common.cuh"

template<class HashTable>
HOSTQUALIFIER INLINEQUALIFIER
void single_value_benchmark(
    const typename HashTable::key_type * keys_d,
    const uint64_t max_keys,
    std::vector<uint64_t> input_sizes,
    std::vector<float> load_factors,
    bool print_headers = true,
    uint8_t iters = 5,
    std::chrono::milliseconds thermal_backoff = std::chrono::milliseconds(100))
{
    using key_t = typename HashTable::key_type;
    using value_t = typename HashTable::value_type;

    value_t* values_d = nullptr;
    cudaMalloc(&values_d, sizeof(value_t)*max_keys); CUERR

    key_t* query_keys_d = nullptr;
    cudaMalloc(&query_keys_d, sizeof(key_t)*max_keys); CUERR

    const auto max_input_size =
        *std::max_element(input_sizes.begin(), input_sizes.end());
    const auto min_load_factor =
        *std::min_element(load_factors.begin(), load_factors.end());

    if(max_input_size > max_keys)
    {
        std::cerr << "Maximum input size exceeded." << std::endl;
        exit(1);
    }

    if(!sufficient_memory_oa<HashTable>(max_input_size / min_load_factor))
    {
        std::cerr << "Not enough GPU memory." << std::endl;
        exit(1);
    }

    std::string results_file = "results/single_value_hash_table.csv";
    fs::path results_path = fs::path(results_file).parent_path();
    if(!fs::exists(results_path))
    {
        fs::create_directories(results_path);
    }

    if(fs::exists(results_file))
    {
        std::ofstream ofs;
        ofs.open(results_file, std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }

    for(auto size : input_sizes)
    {
        for(auto load : load_factors)
        {
            const std::uint64_t capacity = size / load;

            HashTable hash_table(capacity);

            Output<key_t,value_t> output;
            output.sample_size = size;
            output.key_capacity = hash_table.capacity();

            output.insert_ms = benchmark_insert(
                hash_table, keys_d, values_d, size,
                iters, thermal_backoff);

            output.query_ms = benchmark_query(
                hash_table, keys_d, values_d, size,
                iters, thermal_backoff);
            
            //100% exist query
            output.query_exist_100_ms = benchmark_query(
                hash_table, keys_d, values_d, size,
                iters, thermal_backoff);
            
            auto gen_mixed_keys = [&](int percent) {
                helpers::lambda_kernel
                <<<SDIV(size, 1024), 1024>>>
                ([=] DEVICEQUALIFIER
                {
                    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
                    if(tid < size)
                    {
                        bool hit = false;
                        switch(percent) {
                            case 100: hit = true; break;
                            case 75: hit = (tid % 4) != 0; break;
                            case 50: hit = (tid % 2) == 0; break;
                            case 25: hit = (tid % 4) == 0; break;
                            case 0: hit = false; break;
                        }
                        
                        if(hit) {
                            query_keys_d[tid] = keys_d[tid];
                        } else {
                            query_keys_d[tid] = keys_d[size + tid];
                        }
                    }
                });
                cudaDeviceSynchronize(); CUERR
            };

            //75% exist query
            gen_mixed_keys(75);
            output.query_exist_75_ms = benchmark_query(
                hash_table, query_keys_d, values_d, size,
                iters, thermal_backoff);

            //50% exist query
            gen_mixed_keys(50);
            output.query_exist_50_ms = benchmark_query(
                hash_table, query_keys_d, values_d, size,
                iters, thermal_backoff);

            //25% exist query
            gen_mixed_keys(25);
            output.query_exist_25_ms = benchmark_query(
                hash_table, query_keys_d, values_d, size,
                iters, thermal_backoff);

            //0% exist query
            gen_mixed_keys(0);
            output.query_exist_0_ms = benchmark_query(
                hash_table, query_keys_d, values_d, size,
                iters, thermal_backoff);
            
            output.key_load_factor = hash_table.load_factor();
            output.density = output.key_load_factor;
            output.status = hash_table.pop_status();

            if(print_headers)
                output.print_with_headers();
            else
                output.print_without_headers();
            output.save_results_in_csv_with_headers(results_file);
        }
    }

    cudaFree(values_d); CUERR
    cudaFree(query_keys_d); CUERR
}

int main(int argc, char* argv[])
{
    using namespace warpcore;

    using key_t = std::uint32_t;
    using value_t = std::uint32_t;

    const uint64_t max_keys = 1UL << 28;

    std::vector<uint64_t> input_sizes = {1ULL << 22, 1ULL << 23, 1ULL << 24, 1ULL << 25, 1ULL << 26, 1ULL << 27};

    const bool print_headers = true;

    uint64_t dev_id = 0;
    if(argc > 2) dev_id = std::atoi(argv[2]);
    cudaSetDevice(dev_id); CUERR

    key_t * keys_d = nullptr;

    if(argc > 1)
        keys_d = load_keys<key_t>(argv[1], max_keys);
    else
        keys_d = generate_keys<key_t>(max_keys, 1);

    using hash_table_t = SingleValueHashTable<
        key_t,
        value_t,
        defaults::empty_key<key_t>(),
        defaults::tombstone_key<key_t>(),
        defaults::probing_scheme_t<key_t, 8>,
        storage::key_value::AoSStore<key_t, value_t>>;

    single_value_benchmark<hash_table_t>(
        keys_d, max_keys, input_sizes, {1.0}, print_headers);

    cudaFree(keys_d); CUERR
}
