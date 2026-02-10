#include <Common/helper_functions.h>
#include <Common/helper_cuda.h>
#include <stdint.h>
#include "../tools/gputimer.h"
#include "../data/data_layout.cuh"
#include "../core/static_cuckoo.cuh"
#include "../data/key_generator.h"

namespace ch = cuckoo_helpers;
using namespace std;

class StaticTest
{
public:
    using data_t = DataLayout<>::data_t;
    using key_t = DataLayout<>::key_t;
    using value_t = DataLayout<>::value_t;
    static key_t *read_data(char *file_name, int data_len)
    {
        FILE *fid;
        fid = fopen(file_name, "rb");
        key_t *pos = (key_t *)malloc(sizeof(key_t) * data_len);
        if (fid == NULL)
        {
            printf("file not found.\n");
            return pos;
        }
        fread(pos, sizeof(unsigned int), data_len, fid);
        fclose(fid);
        return pos;
    }

    static int check(value_t *check_pool_h, int32_t size)
    {
        uint32_t error_cnt = 0;
        for (int i = 0; i < size; i++)
        {
            if (check_pool_h[i] != i + 5)
            {
                ++error_cnt;
            }
        }
        if (error_cnt != 0)
        {
            printf("num error:%d \n", error_cnt);
        }
        else
        {
            printf("batch check ok\n");
        }
        return error_cnt;
    }

    static key_t *gen_unique_keys(size_t n,
                                  uint32_t seed = 42,
                                  bool shuffle = true)
    {
        auto *keys = new key_t[n];
        // Bijection mod 2^32 (odd multiplier)
        const uint32_t A = 2654435761u;
        const uint32_t B = 1013904223u ^ seed;

        for (uint32_t i = 0; i < n; i++)
        {
            keys[i] = static_cast<key_t>(i * A + B);
            if (keys[i] == 0)
                keys[i] = 1; // avoid sentinel
        }

        if (shuffle)
        {
            std::mt19937 rng(seed);
            std::shuffle(keys, keys + n, rng);
        }

        return keys;
    }
};
int main(int argc, char **argv)
{
    using test_t = StaticTest;

    if (argc < 5)
    {
        std::cout << "Usage: ./static_test [out.csv] [pStart] [pEnd] [load_factor]\n";
        std::cout << "Example: ./static_test results.csv 20 24 0.85\n";
        return -1;
    }

    const char *out_csv = argv[1];
    int pStart = std::atoi(argv[2]);
    int pEnd = std::atoi(argv[3]);
    double init_fill_factor = std::atof(argv[4]);

    // Open CSV (overwrite). If you prefer append, use ios::app.
    std::ofstream csv(out_csv);
    if (!csv)
    {
        std::cerr << "Failed to open output csv: " << out_csv << "\n";
        return -1;
    }
    csv << "num_keys,insert_mops,query_100_mops,query_75_mops,query_50_mops,query_25_mops,query_0_mops\n";

    for (int p = pStart; p <= pEnd; p++)
    {
        const size_t n = size_t(1) << p;

        // Generate workload
        test_t::key_t *keys_h = test_t::gen_unique_keys(n, /*seed=*/42u + p, /*shuffle=*/true);

        auto *values_h = new test_t::value_t[n];
        auto *check_h = new test_t::value_t[n];
        for (size_t i = 0; i < n; i++)
        {
            values_h[i] = static_cast<test_t::value_t>(i + 5);
            check_h[i] = 0;
        }

        // Build table sized for requested load factor
        StaticCuckoo<512, 512> static_cuckoo(static_cast<size_t>(double(n) / init_fill_factor));

        // --- Time insert
        double insert_time_second = static_cuckoo.hash_insert(keys_h, values_h, (int)n);
        std::cout << "Insert time (s): " << insert_time_second << "\n";

        double insert_throughput = (double)n / insert_time_second * 1e-6; // MOPS

        // exist ratio of query key = {1.0, 0.75, 0.5, 0.25, 0.0}
        // We must make sure each query array has exactly n keys. For the non-existent portion
        // we generate keys > n (simple increasing values starting at n+1).

        auto *query_keys = new test_t::key_t[n];

        auto run_and_measure = [&](const char *label, size_t exist_ratio) {
            size_t exist_count = static_cast<size_t>(n * exist_ratio);
            // fill first exist_count entries with existing keys, rest with non-existing (> n)
            for (size_t i = 0; i < exist_count; ++i) {
                query_keys[i] = keys_h[i];
            }
            for (size_t i = exist_count; i < n; ++i) {
                // generate simple non-existing key value > n
                query_keys[i] = static_cast<test_t::key_t>(n + (i - exist_count) + 1);
            }

            // reset check_h
            for (size_t i = 0; i < n; ++i) check_h[i] = 0;

            double query_time = static_cuckoo.hash_search(query_keys, check_h, (int)n);
            std::cout << label << " Query time (s): " << query_time << "\n";

            // Count successful lookups (non-zero values in check_h)
            size_t successful = 0;
            for (size_t i = 0; i < n; ++i) {
                if (check_h[i] != 0) ++successful;
            }
            size_t query_error_count = n - successful;
            double query_throughput = (double)(n - query_error_count) / query_time * 1e-6; // MOPS
            std::cout << label << " successful: " << successful << ", errors: " << query_error_count << "\n";
            return query_throughput;
        };

        // 100% exist
        double query_throughput_100, query_throughput_75, query_throughput_50, query_throughput_25, query_throughput_0 = 0.0;
        query_throughput_100 = run_and_measure("100% exist", 1.0f);
        query_throughput_75 = run_and_measure("75% exist", 0.75f);
        query_throughput_50 = run_and_measure("50% exist", 0.5f);
        query_throughput_25 = run_and_measure("25% exist", 0.25f);
        query_throughput_0 = run_and_measure("0% exist", 0.0f);

        // Write CSV row (we record the 100% exist throughput)
        csv << n << "," << insert_throughput << "," << query_throughput_100 << "," << query_throughput_75 << "," << query_throughput_50 << "," << query_throughput_25 << "," << query_throughput_0 << "\n";

        delete[] query_keys;
        csv.flush();

        std::cout << "[2^" << p << " = " << n << "] "
                  << "insert: " << insert_throughput << " MOPS, "
                  << "query: " << query_throughput_100 << " MOPS\n";

        delete[] keys_h;
        delete[] values_h;
        delete[] check_h;
    }

    return 0;
}
