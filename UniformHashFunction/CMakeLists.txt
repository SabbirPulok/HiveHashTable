cmake_minimum_required(VERSION 3.2)  # Use a more recent version for better CUDA support

# Project name and supported languages (C++ and CUDA)
project(HashFunctionsGPU LANGUAGES CXX CUDA)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Change this if using a different GPU architecture

# Ensure CUDA is enabled and detected properly
enable_language(CUDA)

# Set CUDA Home Path (Explicitly Specify CUDA)
if(NOT DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME "/usr/local/cuda")  # Change if installed elsewhere
else()
    set(CUDA_HOME $ENV{CUDA_HOME})
endif()

message(STATUS "Using CUDA_HOME: ${CUDA_HOME}")

# Include CUDA and project headers
include_directories(${CUDA_HOME}/include include)

# Link CUDA libraries
link_directories(${CUDA_HOME}/lib64)

# Enable C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable for the main application
add_executable(hash_functions_gpu src/main.cpp src/vector_add.cu src/utils.cpp src/hash_function.cu src/experiment_kernel.cu)

# Set CUDA compilation flags (optional)
set_target_properties(hash_functions_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA runtime
target_link_libraries(hash_functions_gpu ${CUDA_HOME}/lib64/libcudart.so)
