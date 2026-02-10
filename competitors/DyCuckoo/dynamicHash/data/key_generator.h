#pragma once
#include <cstdint>
#include <algorithm>
#include <random>

template <typename KeyT>
inline KeyT* gen_unique_keys_u32(size_t n,
                                 uint32_t seed = 42,
                                 bool shuffle = true)
{
    auto* keys = new KeyT[n];

    // Bijection mod 2^32 (odd multiplier)
    const uint32_t A = 2654435761u;
    const uint32_t B = 1013904223u ^ seed;

    for (uint32_t i = 0; i < n; i++) {
        keys[i] = static_cast<KeyT>(i * A + B);
        if (keys[i] == 0) keys[i] = 1; // avoid sentinel
    }

    if (shuffle) {
        std::mt19937 rng(seed);
        std::shuffle(keys, keys + n, rng);
    }

    return keys;
}
