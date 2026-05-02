# Hash Function Uniformity Study (GPU)

This project evaluates the uniformity of various hash functions on the GPU by measuring the Collision Speedup Ratio (CSR). It compares standard functions (MurmurHash, CityHash, CRC) against custom Jenkins-style bitwise mixing functions (`BitHash1`, `BitHash2`).

## Build Instructions

To build the C++ benchmark executable:

```bash
mkdir -p build
cd build
# Adjust CMAKE_CUDA_ARCHITECTURES based on your GPU (e.g., 89 for RTX 4090, 120 for RTX 5090)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
make
```

## Running Experiments

The experiments are configured to sweep through various key counts ($n$) while maintaining a fixed number of buckets ($m = 512^2 = 262,144$).

To run the full suite of experiments and generate the raw data (CSV files):

```bash
./hash_functions_gpu
```

The results will be saved as timestamped CSV files in the `build/` directory (e.g., `Hash_Function_Study_Chapter_GPU_random262144b*16777216k.csv`).

## Plotting Results

To recreate **Figure 4** from the Hive Hash paper (Collision Speedup Ratio grouped bar chart):

```bash
# Ensure you are in the Hash_Function_GPU directory
python3 plot_csr.py
```

This script parses the CSV files in `build/`, extracts the CSR values for each hash function, and generates `HashFunctionCSR_recreated.png`.

### CSR Formula
The Collision Speedup Ratio (CSR) is calculated as:
$$CSR = \frac{\mathbb{E}[Y]}{Y_{observed}}$$
Where $\mathbb{E}[Y]$ is the expected number of collisions under uniform hashing, and $Y_{observed}$ is the average number of collisions measured during the experiment.
