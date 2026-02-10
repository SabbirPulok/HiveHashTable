#ifndef HASH_FUNCTION_H
#define HASH_FUNCTION_H

#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>
#include<vector>
using std::vector;

#define PRIME_N 294'967'291u


void generate_CRC64_table(vector<uint64_t>&table);
void generate_CRC32_table(vector<uint32_t>&table);
void generate_SpinWheel_table(vector<uint16_t>&table, int seed = PRIME_N);
void generate_SpinWheel_table_8b(vector<uint8_t>&table, int seed = PRIME_N);


__host__ __device__ uint32_t identityhash(uint32_t key);

__host__ __device__ uint32_t crc64(uint64_t* CRC64_Table, uint32_t key);

__host__ __device__ uint32_t crc32(uint32_t* CRC32_Table, uint32_t key);

__host__ __device__ uint32_t spinwheel(uint16_t* SpinWheelTable, uint32_t key);

__host__ __device__ uint32_t spinwheel8b(uint8_t* SpinWheelTable, uint32_t key);

__host__ __device__ uint32_t hash1(uint32_t key);

__host__ __device__ uint32_t hash2(uint32_t key);

__host__ __device__ uint32_t cityhash(uint32_t key);

__host__ __device__ uint32_t murmurhash(uint32_t key);



#endif
