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
    ARRAY_OF_ALIGNED_STRUCTS_LEAD_METADATA
};

enum class YCSBWorkLoadType{
    WORKLOAD_A, // 50% read, 50% write
    WORKLOAD_B, // 95% read, 5% write
    WORKLOAD_C, // 100% read
    WORKLOAD_D, // 95% read, 5% new inserts
    WORKLOAD_F, // 50% read, 50% read-modify-write
    NONE
};

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
#endif

void vectorAdd(const float* A, const float* B, float* C, int N);

