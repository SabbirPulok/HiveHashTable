#!/bin/bash

# Use SM environment variable if provided, else default to 89
cuda_arch="${SM:-89}"
build_dir="build"
targets=("all")

# Use CUDA_PATH environment variable if provided, else default to /usr/local/cuda
cuda_bin="${CUDA_PATH:-/usr/local/cuda}/bin/nvcc"

echo "Rebuilding BGHT_IHT with SM=${cuda_arch} and NVCC=${cuda_bin}"

cmake -B $build_dir -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch} -DCMAKE_CUDA_COMPILER=${cuda_bin}
cmake --build $build_dir --target "${targets[@]}" --parallel $(nproc)
