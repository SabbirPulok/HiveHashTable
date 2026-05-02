#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

__host__ __device__ inline uint32_t hash32(uint32_t key, uint32_t table_size)
{
    key ^= key >> 16;
    key *= 0xff51afd7ed558ccd;
    key ^= key >> 16;
    key *= 0xc4ceb9fe1a85ec53;
    key ^= key >> 16;

    return static_cast<uint32_t>(key % table_size);
}

__host__ __device__ inline uint32_t hash32_alt(uint32_t key, uint32_t table_size)
{
    key = key ^ 0xA24BAED4963EE407;
    key ^= key >> 16;
    key *= 0xff51afd7ed558ccd;
    key ^= key >> 16;
    key *= 0xc4ceb9fe1a85ec53;
    key ^= key >> 16;

    return static_cast<uint32_t>(key % table_size);
}

__device__ __forceinline__ uint32_t hash1(uint32_t key, uint32_t table_size) {
    key = ~key + (key << 15);
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;
    key = key ^ (key >> 16);
    return (key) % table_size;
}

__host__ __device__ inline uint32_t hash1(uint32_t key) {
    key = ~key + (key << 15);
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;
    key = key ^ (key >> 16);
    return key;
}
          
__device__ __forceinline__ uint32_t hash2(uint32_t key, uint32_t table_size) {
    uint32_t temp;
    key = (key + 0x7ed55d16) + (key << 12);
    // key = (key ^ 0xc761c23c) ^ (key >> 19);
    asm("shr.u32 %0, %1, 19;" : "=r"(temp) : "r"(key));
    asm("lop3.b32 %0, %0, %1, 0xc761c23c, 0x96;" : "+r"(key) : "r"(temp));
    key = (key + 0x165667b1) + (key << 5);
    key = (key + 0xd3a2646c) ^ (key << 9);
    key = (key + 0xfd7046c5) + (key << 3);
    // key = (key ^ 0xb55a4f09) ^ (key >> 16);
    asm("shr.u32 %0, %1, 16;" : "=r"(temp) : "r"(key));
    asm("lop3.b32 %0, %0, %1, 0xb55a4f09, 0x96;" : "+r"(key) : "r"(temp));
    return key % table_size;
}

__host__ __device__ inline uint32_t hash2(uint32_t key) {
    uint32_t temp;
    key = (key + 0x7ed55d16) + (key << 12);
    //key = (key ^ 0xc761c23c) ^ (key >> 19);
    asm("shr.u32 %0, %1, 19;" : "=r"(temp) : "r"(key));
    asm("lop3.b32 %0, %0, %1, 0xc761c23c, 0x96;" : "+r"(key) : "r"(temp));

    key = (key + 0x165667b1) + (key << 5);
    key = (key + 0xd3a2646c) ^ (key << 9);
    key = (key + 0xfd7046c5) + (key << 3);
    //key = (key ^ 0xb55a4f09) ^ (key >> 16);
    asm("shr.u32 %0, %1, 16;" : "=r"(temp) : "r"(key));
    asm("lop3.b32 %0, %0, %1, 0xb55a4f09, 0x96;" : "+r"(key) : "r"(temp));
    return key;
}

__host__ __device__ inline uint32_t cityhash(uint32_t key, uint32_t table_size) {
  const uint64_t k0 = 0x9ae16a3b;
  const uint64_t k1 = 0xc2b2ae35;
  const uint64_t k2 = 0x1b873593;
  const uint64_t k3 = 0x85ebca6b;

  uint32_t hash = 0;

  uint32_t k = key;
  k ^= k0;
  k *= k1;
  k ^= k2;
  k *= k3;

  hash ^= k;
  hash *= k1;
  hash ^= k2;
  hash *= k3;

  hash ^= 4; // Length is 4 bytes

  hash ^= (hash >> 16);
  hash *= 0x85ebca6b;
  hash ^= (hash >> 13);
  hash *= 0xc2b2ae35;
  hash ^= (hash >> 16);

  return static_cast<uint32_t>(hash % table_size);
}

__host__ __device__ inline uint32_t murmurhash(uint32_t key, uint32_t table_size) {
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;
  const uint32_t r1 = 15;
  const uint32_t r2 = 13;
  const uint32_t m = 5;
  const uint32_t n = 0xe6546b64;

  uint32_t hash = 0;

  uint32_t k = key;
  k *= c1;
  k = (k << r1) | (k >> (32 - r1));
  k *= c2;

  hash ^= k;
  hash = (hash << r2) | (hash >> (32 - r2));
  hash = hash * m + n;

  hash ^= 4; // Length is 4 bytes

  hash ^= (hash >> 16);
  hash *= 0x85ebca6b;
  hash ^= (hash >> 13);
  hash *= 0xc2b2ae35;
  hash ^= (hash >> 16);

  return static_cast<uint32_t>(hash % table_size);
}


__device__ __constant__ uint64_t d_CRC64_Table[256];
__device__ __constant__ uint32_t d_CRC32_Table[256];

inline std::vector<uint64_t> generate_CRC64_Table()
{
  uint64_t CRC64_Polynomial { 0x42F0'E1EB'A9EA'3693ULL };
  std::vector<uint64_t> table(256);
  for (uint64_t i = 0; i < 256; i++) {
    uint64_t crc = i;
    for (uint64_t j = 0; j < 8; j++) {
      if (crc & 1ULL) {
        crc = (crc >> 1) ^ CRC64_Polynomial;
      } else {
        crc >>= 1;
      }
    }
    table[i] = crc;
  }
  return table;
}

inline std::vector<uint32_t> generate_CRC32_Table()
{
  uint32_t CRC32_Polynomial { 0x04C1'1D87u };
  std::vector<uint32_t> table(256);
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t crc = i;
    for (uint32_t j = 0; j < 8; j++) {
      if (crc & 1ULL) {
        crc = (crc >> 1) ^ CRC32_Polynomial;
      } else {
        crc >>= 1;
      }
    }
    table[i] = crc;
  }
  return table;
}

__host__ __device__ __forceinline__ uint32_t crc64(uint32_t key)
{
  uint64_t crc = 0xFFFFFFFFFFFFFFFF;
    uint8_t *byte {reinterpret_cast<uint8_t*>(&key)};

  for(int i=0; i<4; ++i)
  {
      uint8_t index = (crc ^ *byte++) & 0xFF;
      crc = (crc >> 8) ^ d_CRC64_Table[index];
  }

  return (uint32_t)(crc ^ 0xFFFF'FFFF'FFFF'FFFF);
}

__host__ __device__ __forceinline__ uint32_t crc32(uint32_t key)
{
    uint32_t crc = 0xFFFF'FFFFu;
    uint8_t *byte {reinterpret_cast<uint8_t*>(&key)};

    for(int i=0; i<4; ++i)
    {
        uint8_t index = (crc ^ *byte++) & 0xFF;
        crc = (crc >> 8) ^ d_CRC32_Table[index];
    }

    return crc ^ 0xFFFF'FFFFu;
}

inline void generateLookupTables()
{
    std::vector<uint64_t> h_CRC64_Table = generate_CRC64_Table();
    cudaMemcpyToSymbol(d_CRC64_Table, h_CRC64_Table.data(), 256 * sizeof(uint64_t));

    std::vector<uint32_t> h_CRC32_Table = generate_CRC32_Table();
    cudaMemcpyToSymbol(d_CRC32_Table, h_CRC32_Table.data(), 256 * sizeof(uint32_t));
}