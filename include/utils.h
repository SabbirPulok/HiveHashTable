#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstdint>

enum class OperationType{
    INSERT,
    LOOKUP,
    DELETE,
    ATOMIC_INC,
    NONE
};

struct Operation{
    OperationType type;
    uint64_t key;
};

enum class HashTableDataLayout {
    HYBRID_SOA_AOS,
    ARRAY_OF_ALIGNED_STRUCTS,
    ARRAY_OF_ALIGNED_STRUCTS_LEAD_METADATA,
    WARPSPEED
};

enum class YCSBWorkLoadType{
    WORKLOAD_A, // 50% read, 50% write
    WORKLOAD_B, // 95% read, 5% write
    WORKLOAD_C, // 100% read
    WORKLOAD_D, // 95% read, 5% new inserts
    WORKLOAD_F, // 50% read, 50% read-modify-write
    NONE
};

#ifdef __CUDACC__
__device__ __constant__ uint32_t SENTINEL = 0ull;
__device__ __constant__ uint64_t EMPTY_KV = 0ull;
__device__ __constant__ uint64_t LOCKED_KV = 0xFFFFFFFFFFFFFFFFull;
#else
static const uint32_t SENTINEL = 0;
static const uint64_t EMPTY_KV = 0;
static const uint64_t LOCKED_KV = 0xFFFFFFFFFFFFFFFFull;
#endif

std::string inline pretty_print_number(size_t n) noexcept
{
    std::ostringstream out;
    if(n >= (1ULL << 30))
    {
        out << std::fixed << std::setprecision(2) << static_cast<double>(n) / (1ULL << 30);
        return out.str() + " billions";
    }
    else if(n >= (1ULL << 20))
    {
        out << std::fixed << std::setprecision(2) << static_cast<double>(n) / (1ULL << 20);
        return out.str() + " millions";
    }
    else if(n >= (1ULL << 10))
    {
        out << std::fixed << std::setprecision(2) << static_cast<double>(n) / (1ULL << 10);
        return out.str() + " thousands";
    }
    
    return std::to_string(n);
}

#ifdef __CUDACC__
// Helper to bypass L1 cache (Load Cache Global)
// Ensures we read data from L2 (where atomic releases from other SMs are visible)
// and not stale data from a local L1 cache line.
template<typename T>
__device__ __forceinline__ T load_cg_safe(const T* ptr)
{
    //__ldcg is supported for basic types and vector types like uint4 on modern archs
    return __ldcg(ptr);
}

// If no cache modifier is specified, default is .ca (Cache All). 
// On GPUs L1 cache is not hardware-coherent across SMs.
// Use BYPASS_L1 = true for concurrent mixed workloads to ensure visibility.
// Use BYPASS_L1 = false for lookup-only or static workloads to maximize L1 hit rate.
// "ld.global.acquire.gpu.v2.u64 {%0, %1}, [%2];\n"
template<bool BYPASS_L1 = true>
__device__ __forceinline__ ulonglong2 load_two_kvs(ulonglong2* bucket, unsigned lane)
{
    ulonglong2 v;

    if (BYPASS_L1) {
        asm volatile (
            "ld.global.acquire.gpu.v2.u64 {%0, %1}, [%2];\n"
            : "=l"(v.x), "=l"(v.y)
            : "l"(&bucket[lane])
            : "memory"
        );
    } else {
        asm volatile (
            "ld.global.ca.v2.u64 {%0, %1}, [%2];\n"
            : "=l"(v.x), "=l"(v.y)
            : "l"(&bucket[lane])
            : "memory"
        );
    }
    return v;
}

// Cache Global - load.global.cg = reads from L2 bypassing L1 cache but does not invalidates cached L1 lines
// Volatile - load.global.cv =  invalidates L2 entry and strictly inhibits compiler optimizations for all access
// ld.global.acquire.gpu.v2.u64 = Prevents L2 stale reads and Orders value loads (ensure cummulative visibility))

#endif

void vectorAdd(const float* A, const float* B, float* C, int N);