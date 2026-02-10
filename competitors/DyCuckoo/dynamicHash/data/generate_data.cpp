#include <iostream>
#include <fstream>
#include <unordered_set>
#include <random>
#include <vector>
#include <algorithm>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./generate_dat [output_file.dat] [num_keys]" << std::endl;
        return 1;
    }

    const std::string filename = argv[1];
    const size_t num_keys = std::stoull(argv[2]);

    std::unordered_set<int> key_set;
    std::mt19937 rng(42);  // Seed for reproducibility
    std::uniform_int_distribution<int> dist(1, num_keys * 10);  // Spread out to ensure uniqueness

    // Generate unique keys
    while (key_set.size() < num_keys) {
        key_set.insert(dist(rng));
    }

    // Copy to vector and optionally shuffle
    std::vector<int> keys(key_set.begin(), key_set.end());
    std::shuffle(keys.begin(), keys.end(), rng);

    // Write to binary file
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 1;
    }

    out.write(reinterpret_cast<const char*>(keys.data()), sizeof(int) * keys.size());
    out.close();

    std::cout << "Wrote " << keys.size() << " keys to " << filename << std::endl;

    return 0;
}
