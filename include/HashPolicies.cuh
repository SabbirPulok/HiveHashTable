#pragma once
#include "hash.hpp"
#include <cstdint>

// Hash Policy Interface
// Each policy must define:
// static constexpr int NumHashes;
// __device__ static inline uint32_t get_bucket(int idx, uint32_t key, uint32_t num_buckets);

struct Default2HashPolicy {
    static constexpr int NumHashes = 2;
    
    __device__ static inline uint32_t get_bucket(int idx, uint32_t key, uint32_t num_buckets) {
        if (idx == 0) return hash1(key, num_buckets);
        return hash2(key, num_buckets);
    }
};

struct TripleHashPolicy {
    static constexpr int NumHashes = 3;
    
    __device__ static inline uint32_t get_bucket(int idx, uint32_t key, uint32_t num_buckets) {
        if (idx == 0) return hash1(key, num_buckets);
        if (idx == 1) return hash2(key, num_buckets);
        return cityhash(key, num_buckets);
    }
};

struct MurmurCityHashPolicy {
    static constexpr int NumHashes = 2;
    
    __device__ static inline uint32_t get_bucket(int idx, uint32_t key, uint32_t num_buckets) {
        if (idx == 0) return murmurhash(key, num_buckets);
        return cityhash(key, num_buckets);
    }
};

struct MurmurCityBitHashPolicy {
    static constexpr int NumHashes = 3;
    
    __device__ static inline uint32_t get_bucket(int idx, uint32_t key, uint32_t num_buckets) {
        if (idx == 0) return murmurhash(key, num_buckets);
        if (idx == 1) return cityhash(key, num_buckets);
        return hash1(key, num_buckets);
    }
};

struct Lookup2HashPolicy {
    static constexpr int NumHashes = 2;

    __device__ static inline uint32_t get_bucket(int idx, uint32_t key, uint32_t num_buckets) {
        if (idx == 0) return crc64(key) % num_buckets;
        return crc32(key) % num_buckets;
    }
};

struct Lookup3HashPolicy {
    static constexpr int NumHashes = 3;

    __device__ static inline uint32_t get_bucket(int idx, uint32_t key, uint32_t num_buckets) {
        if (idx == 0) return crc64(key) % num_buckets;
        if (idx == 1) return crc32(key) % num_buckets;
        return hash1(key, num_buckets);
    }
};

