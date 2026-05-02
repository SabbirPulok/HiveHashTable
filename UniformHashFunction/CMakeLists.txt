cmake_minimum_required(VERSION 3.10)

# Project name and supported languages (C++ and CUDA)
project(HashFunctionsGPU LANGUAGES CXX CUDA)

# Set CUDA architecture (adjust based on your GPU)
# e.g., 89 for RTX 4090, 120 for RTX 5090
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# Enable C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include CUDA and project headers
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} include)

# Add executable for the main application
add_executable(hash_functions_gpu 
    src/main.cpp 
    src/vector_add.cu 
    src/utils.cpp 
    src/hash_function.cu 
    src/experiment_kernel.cu
)

# Set CUDA compilation flags
set_target_properties(hash_functions_gpu PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA libraries
target_link_libraries(hash_functions_gpu PRIVATE cudart)
