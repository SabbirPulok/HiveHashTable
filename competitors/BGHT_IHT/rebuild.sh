#!/bin/bash

cuda_arch="89" #specify your GPU SM/gencode
build_dir="build"
targets=("all")

# Explicitly set the CUDA compiler to version 13.0
cmake -B $build_dir -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch} -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
cmake --build $build_dir --target "${targets[@]}" --parallel $(nproc)
