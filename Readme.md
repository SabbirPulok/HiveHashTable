# Hive Hash Table: A Dynamically Resizable Concurrent Hash Table for GPUs

This repository contains the implementation and benchmarking suite for **Hive Hash Table**, a high-performance, concurrent, and dynamically resizable hash table designed for modern GPU architectures.


## Prerequisites

*   **OS**: Linux
*   **GPU**: NVIDIA GPU (Compute Capability 6.0+) or AMD GPU
*   **Compiler**: `nvcc` (for NVIDIA) or `hipcc` (for AMD)
*   **Language**: C++17
*   **Python**: Python 3.x (for running experiments)
    *   Required Python packages: `matplotlib`, `numpy`, `pandas` (if applicable)

## Build Instructions

To build the project, use the provided `Makefile`.

### Standard Benchmark
To build the standard benchmark executable:

```sh
make benchmark
```
This creates `bin/hive_hash_table_benchmark`.

### YCSB Benchmark
To build the YCSB benchmark executable:

```sh
make ycsb
```
This creates `bin/hive_hash_table_ycsb`.

### Real Workload Benchmark
To build the real workload benchmark executable:

```sh
make real_workload
```
This creates `bin/hive_hash_table_real_workload`.

### Clean Build
To remove all build artifacts:

```sh
make clean
```

### Build Flags
You can customize the build with the following flags:
*   `DYNAMIC_RESIZE=1`: Enable dynamic resizing support.
*   `BREAKDOWN_INSERT=1`: Enable detailed insertion time breakdown.

Example:
```sh
make benchmark DYNAMIC_RESIZE=1
```

## Running Experiments

We provide a set of Python scripts in the `scripts/` directory to orchestrate experiments, run benchmarks, and plot results. These scripts automatically handle rebuilding the project with the necessary flags.

**Note:** Results (CSV files and PNG plots) will be saved in the `results/` directory.

### 1. Mixed Workload (Concurrent Phase)
Evaluates performance under mixed workloads (inserts, lookups, deletes) across different table sizes.

```sh
python3 scripts/concurrent_phase_workload_experiment.py
```
*   **Output**: `results/mixed_workload_<timestamp>.csv`, `results/mixed_workload_<timestamp>.png`

### 2. YCSB Workload
Evaluates performance using YCSB-style workloads (A, B, C, D, F).

```sh
python3 scripts/ycsb_workload_experiment.py
```
*   **Output**: `results/ycsb_workload_experiment_<timestamp>.csv`, `results/ycsb_workload_experiment.png`

### 3. Dynamic Resizing
Measures the throughput and latency of the hash table while dynamically resizing (growing and shrinking).

```sh
python3 scripts/dynamic_resizing_experiment.py
```
*   **Output**: `results/dynamic_resizing_experiment_<timestamp>.csv`, `results/hive_hash_dynamic_resizing_time.png`, etc.

### 4. Insertion Breakdown
Analyzes the time spent in different stages of the insertion process.

```sh
python3 scripts/insertion_breakdown_experiment.py
```
*   **Output**: `results/insertion_breakdown_experiment_<timestamp>.csv`, `results/insertion_stage_breakdown.png`

## Configuration

You can customize experiment parameters (such as `table_size`, `load_factor`, `insert_ratio`, etc.) by directly modifying the variables at the top of the respective Python scripts in the `scripts/` directory.