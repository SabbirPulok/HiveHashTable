#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdint>
#include <cstddef>

#include <cuda/std/cstdint>
#include <cuda_helper.cuh> 

#include "utils.h"

#ifdef DEBUG
#define DEBUG 0
#endif

#ifndef DYNAMIC_RESIZE
#define DYNAMIC_RESIZE 0 
#endif

#ifndef HIVE_BUCKET_SLOTS
#define HIVE_BUCKET_SLOTS 8 //16 * 8B = 128B per bucket payload (coalesced access)
#endif

#ifndef BREAKDOWN_INSERT
#define BREAKDOWN_INSERT 0
#endif

#define TILE_SIZE 4

#define STASH_ENABLED 1

using key_type = uint32_t;

using value_type = uint32_t;
using kv_type = uint64_t;


static const std::pair<size_t, size_t> max_bucket_and_stash_caps = []() {
    size_t free_byte;
    size_t total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));

    std::cout << "GPU Free Memory: " << free_byte / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "GPU Total Memory: " << total_byte / (1024.0 * 1024.0 * 1024.0) << " GB" <<  std::endl;

    // stash will be in pinned memory (host and gpu visible) Managed Memory
    return std::make_pair((free_byte * 0.30), (free_byte * 0.30)); //KV pairs 8 bytes
}();

struct __align__(16) ULong2{
    uint64_t x;
    uint64_t y;
};


template <typename KVType, size_t SLOT>
struct __align__(128) HiveBucketBody{
    alignas(sizeof(KVType)) KVType kv[SLOT]; //key-value pairs
};

//Device Table state (SoA metadata: tiny atomics)
struct HiveHashTable{
    static constexpr size_t SLOTS = HIVE_BUCKET_SLOTS;
    // Array of buckets, each 256 bytes wide, storing 32 packed key-value pairs
    // Each warp can load the entire 256 byte bucket in one coalesced memory transaction, matching the GPU cache line (two 128B L1 cacheline)
    HiveBucketBody<kv_type, SLOTS>* buckets; //[num_buckets]

    // One 32-bit word per bucket, each bit corresponding to a slot in the bucket
    // Instead of scanning the entire bucket to find a free slot, we can just look at this bitmask
    // Atomically claim a free slot with a single CAS
    uint32_t* freeMask; //1=free, 0=occupied [num_buckets]

    // A short-lived per-bucket lock
    // Sometimes both candidate buckets are full, but you might need to evict a victim slot (cuckoo step)
    // Only rare eviction slow-path touches it
    uint32_t* lock; //lock for the bucket

    //Number of buckets in the table
    size_t num_buckets;
    size_t max_num_buckets;

    //maximum number of cuckoo kicks allowed during insertion
    //prevents infinite loops and after max_evictions kicks, we give up and push it into the stash (overflow buffer)
    size_t max_evictions = 8;

    
    uint32_t index_mask; //how many low bits should look to decide the bucket number

    uint32_t split_ptr; //how many low index buckets have been split to next level

    

    #ifdef __CUDACC__
    __device__ __forceinline__ uint32_t* getFreeMask(size_t bucket_idx) const{
        return &freeMask[bucket_idx];
    }
    __device__ __forceinline__ uint32_t* getLock(size_t bucket_idx) const{
        return &lock[bucket_idx];
    }

    __device__ __forceinline__ void lockBucket(size_t bucket_idx) const{
        // Use an atomic_ref on the 32-bit lock word in device scope
        auto atomic_lock = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(lock[bucket_idx]);

        unsigned backoff = 1u;
        while(atomic_lock.exchange(1u, cuda::memory_order_acquire) != 0u) {
            // busy-wait with exponential backoff to reduce thundering herd
#if defined(__CUDA_ARCH__)
            for (unsigned i = 0; i < backoff; ++i) {
                __nanosleep(1);
            }
#else
            for (volatile unsigned i = 0; i < backoff; ++i) {}
#endif
            backoff = (backoff < 1024u) ? (backoff << 1) : 1024u;
        }
    }

    __device__ __forceinline__ void unlockBucket(size_t bucket_idx) const{
        auto atomic_lock = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(lock[bucket_idx]);
        atomic_lock.store(0u, cuda::memory_order_release);
    }

    __device__ __forceinline__ uint64_t* loadKV(size_t bucket_idx, size_t slot_idx) const {
        return &buckets[bucket_idx].kv[slot_idx];
    }
    #endif
};

static auto constexpr HIVE_BUCKET_SLOTS_AOAS = 8;

template<typename KVType, size_t SLOT=HIVE_BUCKET_SLOTS_AOAS> //Usually 64 bits (8 bytes)
struct __align__(128) HiveBucketAoaS{
    //256 bytes total - 8 bytes metadata = 248 bytes for key-value pairs
    KVType kv[SLOT];

    struct Header{
        uint32_t freeMask; // 4 bytes
        uint32_t lock;      // 2 bytes
    } header;

};

template<typename KVType>
struct HiveHashTableAoaS{

    static_assert(sizeof(KVType) == 8 || sizeof(KVType) == 16, "HiveHashTableAoaS requires 64-bit or 128-bit key-value pairs");
    //static_assert(sizeof(HiveBucketAoaS<KVType>) == 256, "HiveBucketAoaS must be 256 bytes in size");

    static constexpr size_t SLOTS = HIVE_BUCKET_SLOTS_AOAS;

    HiveBucketAoaS<KVType, SLOTS>* buckets; // Array of bucket structures
    size_t num_buckets;       // Number of buckets in the table
    size_t max_num_buckets;   // Maximum number of buckets
    size_t max_evictions = 8;     // Maximum number of evictions allowed

    uint32_t index_mask;           // How many low bits should look to decide the bucket number
    uint32_t split_ptr;            // How many low index buckets have been split to next level

    //Helper abstractions for memory access
#ifdef __CUDACC__
    __device__ __forceinline__ uint32_t* getFreeMask(size_t bucket_idx) const{
        return &buckets[bucket_idx].header.freeMask;
    }

    __device__ __forceinline__ void lockBucket(size_t bucket_idx) const{
        // Use an atomic_ref on the 16-bit lock word in device scope
        auto atomic_lock = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(buckets[bucket_idx].header.lock);

        unsigned backoff = 1u;

        while(atomic_lock.exchange(1u, cuda::memory_order_acquire) != 0u) {
            // busy-wait
#if defined(__CUDA_ARCH__)
            // On device, use a short warp-level pause
            for (unsigned i = 0; i < backoff; ++i) {
                __nanosleep(1);
            }
#else
            // Host fallback (shouldn't be compiled for device code path)
            for (volatile unsigned i = 0; i < backoff; ++i) {}
#endif
            backoff = (backoff < 1024u) ? (backoff << 1) : 1024u;
        }
    }

    __device__ __forceinline__ uint32_t* getLock(size_t bucket_idx) const{
        return &buckets[bucket_idx].header.lock;
    }

    __device__ __forceinline__ void unlockBucket(size_t bucket_idx) const{
        auto atomic_lock = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(buckets[bucket_idx].header.lock);
        atomic_lock.store(0u, cuda::memory_order_release);
    }

    __device__ __forceinline__ KVType* loadKV(size_t bucket_idx, size_t slot_idx) const {
        return &buckets[bucket_idx].kv[slot_idx];
    }
#endif
};


template<typename KVType, size_t SLOT=HIVE_BUCKET_SLOTS_AOAS> //Usually 64 bits (8 bytes)
struct __align__(128) HiveBucketAoaS_LeadMetaData{
    //256 bytes total - 8 bytes metadata = 248 bytes for key-value pairs
    KVType kv[SLOT];

    struct Header{
        uint32_t freeMask; // 4 bytes
        uint32_t lock;      // 2 bytes
    } header;

};

template<typename KVType>
struct HiveHashTableAoaS_LeadMetaData{

    static_assert(sizeof(KVType) == 8 || sizeof(KVType) == 16, "HiveHashTableAoaS requires 64-bit or 128-bit key-value pairs");
    //static_assert(sizeof(HiveBucketAoaS<KVType>) == 256, "HiveBucketAoaS must be 256 bytes in size");

    static constexpr size_t SLOTS = HIVE_BUCKET_SLOTS_AOAS;

    HiveBucketAoaS_LeadMetaData<KVType, SLOTS>* buckets; // Array of bucket structures
    size_t num_buckets;       // Number of buckets in the table
    size_t max_num_buckets;   // Maximum number of buckets
    size_t max_evictions = 8;     // Maximum number of evictions allowed

    uint32_t index_mask;           // How many low bits should look to decide the bucket number
    uint32_t split_ptr;            // How many low index buckets have been split to next level

    //Helper abstractions for memory access
#ifdef __CUDACC__
    __device__ __forceinline__ uint32_t* getFreeMask(size_t bucket_idx) const{
        return &buckets[bucket_idx].header.freeMask;
    }

    __device__ __forceinline__ uint32_t* getLock(size_t bucket_idx) const{
        return &buckets[bucket_idx].header.lock;
    }

    __device__ __forceinline__ void lockBucket(size_t bucket_idx) const{
        // Use an atomic_ref on the 16-bit lock word in device scope
        auto atomic_lock = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(buckets[bucket_idx].header.lock);

        unsigned backoff = 1u;

        while(atomic_lock.exchange(1u, cuda::memory_order_acquire) != 0u) {
            // busy-wait
#if defined(__CUDA_ARCH__)
            // On device, use a short warp-level pause
            for (unsigned i = 0; i < backoff; ++i) {
                __nanosleep(1);
            }
#else
            // Host fallback (shouldn't be compiled for device code path)
            for (volatile unsigned i = 0; i < backoff; ++i) {}
#endif
            backoff = (backoff < 1024u) ? (backoff << 1) : 1024u;
        }
    }

    __device__ __forceinline__ void unlockBucket(size_t bucket_idx) const{
        auto atomic_lock = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(buckets[bucket_idx].header.lock);
        atomic_lock.store(0u, cuda::memory_order_release);
    }

    __device__ __forceinline__ KVType* loadKV(size_t bucket_idx, size_t slot_idx) const {
        return &buckets[bucket_idx].kv[slot_idx];
    }
#endif
};


struct InsertBreakdown {
    double stageA;
    double stageB;
    double stageC;
    double stageD;

    __host__ __device__ InsertBreakdown() : stageA(0), stageB(0), stageC(0), stageD(0) {}

    __host__ __device__ InsertBreakdown(const InsertBreakdown& other)
        : stageA(other.stageA), stageB(other.stageB), stageC(other.stageC), stageD(other.stageD) {}

    __host__ __device__ InsertBreakdown& operator=(const InsertBreakdown& other) {
        if (this != &other) {
            stageA = other.stageA;
            stageB = other.stageB;
            stageC = other.stageC;
            stageD = other.stageD;
        }
        return *this;
    }
};


#ifdef __CUDACC__
__device__ __forceinline__ uint64_t packKV(uint32_t key, uint32_t value)
{
    return (static_cast<uint64_t>(value) << 32) | key;
}

__device__  __forceinline__ uint32_t unpackKey(uint64_t kv)
{
    return static_cast<uint32_t>(kv);
}

__device__  __forceinline__ uint32_t unpackValue(uint64_t kv)
{
    return static_cast<uint32_t>(kv >> 32);
}

// Without reading the whole 32 bit key at lookup,
// read tag
// tag turns 32 bit key into 7 bit mini-hash (Not useful now)
__device__  __forceinline__ uint8_t getTag(uint32_t key)
{
    uint32_t x = key;
    x ^= x >> 16; //Mix high and low bits
    x *= 0x7feb352d; //Magic odd multiplier
    return uint8_t(x & 0x7f); //Take the low 7 bits
}
#endif