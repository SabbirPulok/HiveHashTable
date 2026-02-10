#pragma once
#include <cstdint>
#include <cuda/std/atomic>

// Hash table entries
struct HashEntry {
    uint64_t key;
    uint64_t value;
};

enum class KernelType{
    LOOKUP,
    INSERT,
    DELETE,
    MIX_OPS
};
