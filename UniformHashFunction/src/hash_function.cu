#include "hash_function.h"
#include <random>
using std::random_device;
using std::mt19937;
using std::uniform_int_distribution;

#include<algorithm>
using std::fill;


uint64_t CRC64_Polynomial { 0x42F0'E1EB'A9EA'3693ULL };
uint32_t CRC32_Polynomial { 0x04C1'1D87u };

void generate_CRC64_table(vector<uint64_t>& table)
{
    table.resize(256);
    for (uint64_t i = 0; i < 256; ++i) {
    uint64_t crc = i;
    for (int j = 0; j < 8; ++j) {
        if (crc & 1ULL) {
            crc = (crc >> 1) ^ CRC64_Polynomial;
        } else {
            crc >>= 1;
        }
    }
    table[i] = crc;
    }    
    return;
}

void generate_CRC32_table(vector<uint32_t>& table)
{
  table.resize(256);
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t crc = i;
    for (int j = 0; j < 8; ++j) {
      if (crc & 1ULL) {
          crc = (crc >> 1) ^ CRC32_Polynomial;
      } else {
          crc >>= 1;
      }
    }
    table[i] = crc;
  }
  return;
}

void generate_SpinWheel_table(vector<uint16_t>&table, int seed)
{
    struct KV{
        int value;
        uint16_t key;
    };
    
    vector<KV>keyvalues;
    keyvalues.resize(65536);

    mt19937 gen(seed);

    table.resize(65536);
    for(uint32_t i=0; i<65536; ++i)
    {
        keyvalues[i] = {(int)gen(), (uint16_t)i};
    }
    
    std::sort(keyvalues.begin(), keyvalues.end(), [](struct KV &left, struct KV &right){
        return left.value<right.value;
    });

    for(uint32_t i=0; i<65536; ++i)
    {
        table[i] = keyvalues[i].key;
    }
    
}

void generate_SpinWheel_table_8b(vector<uint8_t>&table, int seed)
{
    struct KV{
        int value;
        uint8_t key;
    };
    
    vector<KV>keyvalues;
    keyvalues.resize(256);

    mt19937 gen(seed);

    table.resize(256);
    for(uint32_t i=0; i<256; ++i)
    {
        keyvalues[i] = {(int)gen(), (uint8_t)i};
    }
    
    std::sort(keyvalues.begin(), keyvalues.end(), [](struct KV &left, struct KV &right){
        return left.value<right.value;
    });

    for(uint32_t i=0; i<256; ++i)
    {
        table[i] = keyvalues[i].key;
    }
    
}


__device__ uint32_t crc64(uint64_t *CRC64_Table, uint32_t key)
{
    uint64_t crc = 0xFFFFFFFFFFFFFFFF;
    uint8_t *byte {reinterpret_cast<uint8_t*>(&key)};

    for(int i=0; i<4; ++i)
    {
        uint8_t index = (crc ^ *byte++) & 0xFF;
        crc = (crc >> 8) ^ CRC64_Table[index];
    }

    return (uint32_t)(crc ^ 0xFFFF'FFFF'FFFF'FFFF);
}

__host__ __device__ uint32_t crc32(uint32_t* CRC32_Table, uint32_t key)
{
    uint32_t crc = 0xFFFF'FFFFu;
    uint8_t *byte {reinterpret_cast<uint8_t*>(&key)};

    for(int i=0; i<4; ++i)
    {
        uint8_t index = (crc ^ *byte++) & 0xFF;
        crc = (crc >> 8) ^ CRC32_Table[index];
    }

    return crc ^ 0xFFFF'FFFFu;
}

__host__ __device__ uint32_t spinwheel(uint16_t *SpinWheelTable, uint32_t key)
{
    union U_KV{
        uint32_t key32;
        uint16_t key16[2];
    };

    U_KV ukv;

    ukv.key32 = key;

    ukv.key16[0] = SpinWheelTable[ukv.key16[0]];
    ukv.key16[1] = SpinWheelTable[static_cast<uint16_t>(ukv.key16[0]+ukv.key16[1])];

    return ukv.key32;
}

__host__ __device__ uint32_t spinwheel8b(uint8_t *SpinWheelTable, uint32_t key)
{
    union U_KV{
        uint32_t key32;
        uint8_t key8[4];
    };

    U_KV ukv;

    ukv.key32 = key;

    uint8_t Spinner = ukv.key8[0] = SpinWheelTable[ukv.key8[0]];
    Spinner+=ukv.key8[1] = SpinWheelTable[static_cast<uint8_t>(Spinner+ukv.key8[1])];
    Spinner+=ukv.key8[2] = SpinWheelTable[static_cast<uint8_t>(Spinner+ukv.key8[2])];
    ukv.key8[3] = SpinWheelTable[static_cast<uint8_t>(Spinner+ukv.key8[3])];

    return ukv.key32;
}

__host__ __device__ uint32_t hash1(uint32_t key) {
    key = ~key + (key << 15);
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;
    key = key ^ (key >> 16);
    return (key) % PRIME_N;
}

__host__ __device__ uint32_t hash2(uint32_t key) {
    key = (key + 0x7ed55d16) + (key << 12);
    key = (key ^ 0xc761c23c) ^ (key >> 19);
    key = (key + 0x165667b1) + (key << 5);
    key = (key + 0xd3a2646c) ^ (key << 9);
    key = (key + 0xfd7046c5) + (key << 3);
    key = (key ^ 0xb55a4f09) ^ (key >> 16);
    return key % PRIME_N;
}

__host__ __device__ uint32_t cityhash(uint32_t key) {
  const uint32_t k0 = 0x9ae16a3b;
  const uint32_t k1 = 0xc2b2ae35;
  const uint32_t k2 = 0x1b873593;
  const uint32_t k3 = 0x85ebca6b;

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

  return hash;
}

__host__ __device__ uint32_t murmurhash(uint32_t key) {
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

  return hash;
}

__host__ __device__ uint32_t identityhash(uint32_t key)
{
    return key;
}