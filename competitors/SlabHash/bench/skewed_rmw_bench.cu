#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#define private public
#include "gpu_hash_table.cuh"
#undef private
#include "../../../include/zipf_distribution.h"
template <typename KeyT, typename ValueT>
__global__ void rmw_batched_operations(
    uint32_t* d_operations,
    uint32_t num_operations,
    unsigned long long* d_insert_time,
    unsigned long long* d_search_time,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_operations)
    return;

  // initializing the memory allocator on each warp:
  AllocatorContextT local_allocator_ctx(slab_hash.getAllocatorContext());
  local_allocator_ctx.initAllocator(tid, laneId);

  uint32_t myOperation = 0;
  uint32_t myKey = 0;
  uint32_t myValue = 0;
  uint32_t myBucket = 0;

  if (tid < num_operations) {
    myOperation = d_operations[tid];
    myKey = myOperation & 0x3FFFFFFF;
    myBucket = slab_hash.computeBucket(myKey);
    myOperation = myOperation >> 30;
    myValue = myKey;
  }

  bool to_insert = (myOperation == 1) ? true : false;
  bool to_search = (myOperation == 3) ? true : false;

  long long start_insert = clock64();
  slab_hash.insertPair(to_insert, laneId, myKey, myValue, myBucket, local_allocator_ctx);
  long long end_insert = clock64();

  slab_hash.searchKey(to_search, laneId, myKey, myValue, myBucket);
  long long end_search = clock64();

  unsigned long long insert_diff = end_insert - start_insert;
  unsigned long long search_diff = end_search - end_insert;

  unsigned long long warp_insert = 0;
  unsigned long long warp_search = 0;
  for (int i = 0; i < 32; i++) {
     warp_insert += __shfl_sync(0xFFFFFFFF, insert_diff, i);
     warp_search += __shfl_sync(0xFFFFFFFF, search_diff, i);
  }

  if (laneId == 0) {
      atomicAdd(d_insert_time, warp_insert / 32);
      atomicAdd(d_search_time, warp_search / 32);
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cout << "Usage: ./skewed_rmw_bench <table_size_power> <alpha> <num_operations> <update_ratio>\n";
    return 1;
  }

  int table_size_power = std::stoi(argv[1]);
  float alpha = std::stof(argv[2]);
  uint32_t num_operations = std::stoull(argv[3]);
  float update_ratio = std::stof(argv[4]); // 0.0 to 1.0

  uint32_t num_keys = 1 << table_size_power;
  uint32_t num_elements_per_unit = 15;
  float expected_chain = 0.8f;
  uint32_t expected_elements_per_bucket = expected_chain * num_elements_per_unit;
  uint32_t num_buckets = (num_keys + expected_elements_per_bucket - 1) / expected_elements_per_bucket;

  int device_idx = 0;
  cudaSetDevice(device_idx);

  // Initialize GPU hash table
  gpu_hash_table<uint32_t, uint32_t, SlabHashTypeT::ConcurrentMap> hash_table(
      num_keys, num_buckets, device_idx, time(nullptr));

  // Prefill the table
  std::vector<uint32_t> prefill_ops(num_keys);
  for (uint32_t i = 0; i < num_keys; i++) {
    prefill_ops[i] = (1u << 30) | (i + 1); // Op 1 = Insert
  }

  uint32_t* d_prefill_results;
  CHECK_ERROR(cudaMalloc(&d_prefill_results, sizeof(uint32_t) * num_keys));
  
  auto gpu_context = hash_table.slab_hash_->gpu_context_;
  
  hash_table.batched_operations(prefill_ops.data(), d_prefill_results, num_keys, 0);

  // Generate Zipf queries
  std::vector<uint32_t> zipf_ops(num_operations);
  ZipfAlias zipf(num_keys, alpha, 12345);
  
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for (uint32_t i = 0; i < num_operations; i++) {
    uint32_t key = zipf.sample(); // 1-based index
    uint32_t op = (dist(rng) < update_ratio) ? 1u : 3u; // 1=insert (update), 3=search
    zipf_ops[i] = (op << 30) | (key & 0x3FFFFFFF);
  }

  uint32_t* d_zipf_ops;
  CHECK_ERROR(cudaMalloc(&d_zipf_ops, sizeof(uint32_t) * num_operations));
  CHECK_ERROR(cudaMemcpy(d_zipf_ops, zipf_ops.data(), sizeof(uint32_t) * num_operations, cudaMemcpyHostToDevice));

  unsigned long long *d_insert_time, *d_search_time;
  CHECK_ERROR(cudaMalloc(&d_insert_time, sizeof(unsigned long long)));
  CHECK_ERROR(cudaMalloc(&d_search_time, sizeof(unsigned long long)));
  CHECK_ERROR(cudaMemset(d_insert_time, 0, sizeof(unsigned long long)));
  CHECK_ERROR(cudaMemset(d_search_time, 0, sizeof(unsigned long long)));

  uint32_t num_blocks_zipf = (num_operations + 127) / 128;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  rmw_batched_operations<<<num_blocks_zipf, 128>>>(
      d_zipf_ops, num_operations, d_insert_time, d_search_time, gpu_context);
  cudaEventRecord(stop);
  
  CHECK_ERROR(cudaEventSynchronize(stop));

  float total_ms = 0;
  cudaEventElapsedTime(&total_ms, start, stop);

  unsigned long long h_insert_time, h_search_time;
  CHECK_ERROR(cudaMemcpy(&h_insert_time, d_insert_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CHECK_ERROR(cudaMemcpy(&h_search_time, d_search_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

  double total_cycles = static_cast<double>(h_insert_time + h_search_time);
  double insert_fraction = total_cycles > 0 ? (static_cast<double>(h_insert_time) / total_cycles) : 0.0;
  double lookup_fraction = total_cycles > 0 ? (static_cast<double>(h_search_time) / total_cycles) : 0.0;

  double insert_latency_ms = total_ms * insert_fraction;
  double lookup_latency_ms = total_ms * lookup_fraction;

  std::cout << "UPDATE_RATIO: " << update_ratio 
            << " TOTAL_MS: " << total_ms 
            << " INSERT_MS: " << insert_latency_ms 
            << " LOOKUP_MS: " << lookup_latency_ms << "\n";

  cudaFree(d_prefill_results);
  cudaFree(d_zipf_ops);
  cudaFree(d_insert_time);
  cudaFree(d_search_time);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
