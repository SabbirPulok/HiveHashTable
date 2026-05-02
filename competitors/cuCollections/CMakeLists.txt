cmake_minimum_required(VERSION 3.23)
project(cucoBench LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Add cuCollections via CMake so it fetches CCCL automatically
add_subdirectory(external/cuCollections build_cuco EXCLUDE_FROM_ALL)

add_executable(cucoBench cucoBench.cu)
target_link_libraries(cucoBench PRIVATE cuco::cuco)
target_compile_features(cucoBench PRIVATE cuda_std_17)
target_compile_options(cucoBench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)