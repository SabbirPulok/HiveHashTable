add_library(random_gen batched_data_gen.cpp)
cuda_add_executable(benchmark main_benchmarks.cu
			OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})

target_link_libraries(benchmark random_gen)
cuda_add_executable(skewed_rmw_bench skewed_rmw_bench.cu OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})
target_link_libraries(skewed_rmw_bench random_gen)
