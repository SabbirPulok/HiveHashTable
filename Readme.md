# Artifact Evaluation Guide
This repository contains the implementation and benchmarking suite for **Hive Hash Table**, a high-performance, concurrent, and dynamically resizable hash table designed for modern GPU architectures (H100 NVL, RTX 5090, RTX 4090, RTX 3080, RTX 2080 Ti).

---

## 1. Environment Setup

### Step 1: Install System Dependencies
Update the package manager and install the required build tools:
```bash
apt-get update && apt-get install -y build-essential git python3 python3-pip wget
pip3 install "cmake>=4.2" --upgrade
python3 -m pip install matplotlib numpy pandas
```

### Step 2: Install Python Libraries
Install the Python packages needed for your benchmark and plotting scripts:
```bash
pip3 install matplotlib numpy pandas
```

### Step 3: Initialize Git Submodules
Ensure you are inside the project directory (e.g., `/workspace/hivehashtable_gpu`) and pull down the linked submodules (competitors like cuCollections):
```bash
git submodule update --init --recursive
```

```
mkdir -p result
```
---

## 2. Compiling the Project

To compile the codebase for the absolute latest architectures (e.g., NVIDIA H100 NVL), we compile using a forward-compatible architecture flag. The NVIDIA driver will then automatically JIT (Just-In-Time) compile the code at runtime.

Run the following command to build Hive Hash Table and its dependencies:

```bash
make clean && CUDA_PATH=/usr/local/cuda SM=90 make all
```
*(This will generate `bin/hive_hash_table_benchmark`, `bin/hive_hash_table_ycsb`, and `bin/hive_hash_table_lookup_only_workload`)*.

---

## 3. Running Competitor Benchmarks (One-Time Setup)

When deploying to a new machine (like an H100 NVL instance), you need to generate the baseline CSV data for the competitor hash tables **before** running the main Python plotting scripts. 

Since building and running the competitors across multiple capacities and load factors takes time, do this exactly once using the following one-liners. *(Note: We use compute_capability='90' for the H100 NVL)*:

**1. Generate WarpCore Data**
```bash
# This sweeps load factors [0.65, 0.80, 0.9, 0.95, 1.0] internally
python3 -c "from scripts.benchmark_utils import run_warpcore_bench; run_warpcore_bench('./competitors/warpcore')"
```

**2. Generate BGHT / IHT Data**
```bash
# Explicitly passes the target load factors to the benchmark.sh script
python3 -c "import os, sys; sys.path.append(os.getcwd()+'/scripts'); from benchmark_utils import run_bght_iht_bench; run_bght_iht_bench('H100-NVL','./competitors/BGHT_IHT', compute_capability='90', load_factors=[0.65, 0.80, 0.9, 0.95, 1.0])"
```

**3. Generate DyCuckoo Data**
```bash
# Runs the 5 load factor points for sizes 2^22 to 2^27
python3 -c "import os, sys; sys.path.append(os.getcwd()+'/scripts'); from benchmark_utils import run_dycuckoo_bsp;run_dycuckoo_bsp('./competitors/DyCuckoo/dynamicHash/', 22, 27, [0.65, 0.80, 0.9, 0.95, 1.0], compute_capability='90')"
```

**4. Generate SlabHash Data**
```bash
python3 -c "from scripts.benchmark_utils import run_slabhash_bsp_bench; run_slabhash_bsp_bench('./competitors/SlabHash')"

python3 -c "from scripts.benchmark_utils import run_slabhash_concurrent_bench; run_slabhash_concurrent_bench('./competitors/SlabHash')"
```

Once these commands finish, the respective `build/` directories for each competitor will contain the CSVs needed by the plotting scripts.

---

## 4. Reproducing the Experiments

You can now run the automated Python scripts to reproduce each experiment from the paper. They will read the cached baseline data generated in Step 3, run Hive Hash Table dynamically, and save the charts (`.png`) and raw data (`.csv`) directly into the `results/` folder.

### Experiment 1: Bulk Synchronous Parallel (BSP) Workloads
Sweeps table sizes to evaluate Bulk Insertion (100% Inserts) and Bulk Query (100% Lookups).

```bash
python3 scripts/BSP_workload_experiment.py
```
*Expected Output: `results/bulk_insertion_*.png`, `results/bulk_lookup_*.png`*

### Experiment 2: Load Factor Sweep & Lookup Exist Ratios
Runs two benchmarks: First, it sweeps insertion throughput across varying load factors (0.65 to 1.0). Second, it tests query throughput scalability when the percentage of successful vs. unsuccessful lookups varies from 0% to 100%.

```bash
python3 scripts/sweep_load_factor_and_exist_ratio_experiment.py
```
*Expected Output: `results/insert_only_load_factor_*.png`, `results/lookup_exist_*.png`*

### Experiment 3: Concurrent Mixed Phase Workloads
Evaluates the tables under a fully concurrent workload of mixed operations (50% Inserts, 40% Lookups, 10% Deletes) sweeping across table sizes.

```bash
python3 scripts/concurrent_phase_workload_experiment.py
```
*Expected Output: `results/mixed_workload_50I_40L_10D.png`*

### Experiment 4: YCSB Database Workloads
Evaluates the hash table under highly imbalanced Zipfian distributions, matching real-world database request patterns (YCSB Workloads A, B, C, D).

```bash
python3 scripts/ycsb_workload_experiment.py
```
*Expected Output: `results/ycsb_workload_*.png`*

### Experiment 5: Hash Policy / Probe Experiment
Evaluates the performance impact of using different internal Hash Policies (e.g., Default2Hash vs MurmurCityHash).

```bash
python3 scripts/probing_experiment.py
```
*Expected Output: `results/hash_function_comparison_*.png`*

### Experiment 6: Insertion Time Breakdown (Requires Recompilation)
To see a detailed latency breakdown of the 4-stage insertion protocol (Try Replace -> Claim & Commit -> Cuckoo Eviction -> Overflow Stash).

```bash
make clean && CUDA_PATH=/usr/local/cuda SM=90 make BREAKDOWN_INSERT=1 all
python3 scripts/insertion_breakdown_experiment.py
```
*Expected Output: `results/insertion_stage_breakdown.png`*

---

## 5. Viewing the Results

All of the generated plots and CSV files will be instantly available inside the `results/` folder. 