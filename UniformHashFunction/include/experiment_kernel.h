#ifndef EXPERIMENT_KERNEL_H
#define EXPERIMENT_KERNEL_H


#include <cstdint>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"

#include <vector>
using std::vector;

#include <random>
#include <limits>
#include <iomanip>
#include <cmath>

#include "tStats.h"
#include "tTabularResults.h"
#include "Utility.h"
#include "utils.h"

#include "hash_function.h"

using std::log;
using std::exp;

enum function_list:size_t{
    ecrc64,
    ecrc32, 
    ehash1, 
    ehash2, 
    emurmurhash3, 
    ecityhash32, 
    espinwheel_hash, 
    espinwheel_hash8b, 
    eidentityhash, 
    nFunctions};

extern std::string decodeFunc[];

enum key_sequeunce_list{erandom, esequential, nKeySequence};
extern std::string decodeKeySequence[];

struct AllStats
{
    tStats<double> elpTimeStats[nFunctions];
    tStats<double> emptyBucketStats[nFunctions];
    tStats<double> bucketCountStatsMin[nFunctions];
    tStats<double> bucketCountStatsMax[nFunctions];
    tStats<double> bucketCollisionsStats[nFunctions];

};

extern __device__ uint32_t (*noLookUpHash[]) (uint32_t);
extern __device__ uint32_t (*hash) (uint32_t);

extern __constant__ uint8_t spinwheel_table_const[256*256];

void LaunchExperimentWithRetries(size_t nbuckets, size_t nkeys, int nTry);

void LaunchExperiment(uint32_t *keys, vector<uint64_t>h_buckets, uint64_t* d_buckets, AllStats *stats, uint64_t *CRC64_Table, uint32_t *CRC32_Table, uint16_t *SpinWheelTable, uint8_t *SpinWheelTable8b, 
size_t nbuckets, size_t nkeys, double &avgDeviationFromMean, vector<vector<double>>&func_observed_k);

__global__ void ExperimentKernel();

__global__ void identityHashKernel(uint64_t* buckets, uint32_t* keys, size_t nbuckets, size_t nkeys);

__global__ void crc64Kernel(uint64_t* buckets, uint32_t* keys, size_t nbuckets, size_t nkeys, uint64_t* CRC64_Table);

__global__ void crc32Kernel(uint64_t* buckets, uint32_t* keys, size_t nbuckets, size_t nkeys, uint32_t* CRC32_Table);

__global__ void spinWheelKernel(uint64_t* buckets, uint32_t* keys, size_t nbuckets, size_t nkeys, uint16_t* SpinWheel_Table);

__global__ void spinWheelKernel8b(uint64_t* buckets, uint32_t* keys, size_t nbuckets, size_t nkeys, uint8_t* SpinWheel_Table8b);

__global__ void hash1Kernel(uint64_t* buckets, uint32_t* keys, size_t nbuckets, size_t nkeys);

__global__ void hash2Kernel(uint64_t* buckets, uint32_t* keys, size_t nbuckets, size_t nkeys);

__global__ void cityHashKernel(uint64_t* buckets, uint32_t* keys, size_t nbuckets, size_t nkeys);

__global__ void murmurHashKernel(uint64_t* buckets, uint32_t* keys, size_t nbuckets, size_t nkeys);

//__global__ void nonLookupHashKernel(uint64_t* buckets, uint32_t* keys, size_t nbuckets, size_t nkeys, uint32_t (*hash) (uint32_t));


#endif