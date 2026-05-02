BUILD ?= release

DYNAMIC_RESIZE ?= 0

BREAKDOWN_INSERT ?= 0

ifeq ($(BUILD), release)
	HOST_DBG_FLAGS = -O3
	DEV_DBG_FLAGS = -lineinfo -O3 
else
	HOST_DBG_FLAGS = -g -O0
	DEV_DBG_FLAGS = -G -Xptxas="-v"
endif

NVTX_LIB := -lnvToolsExt

BINARY := bin

CODEDIRS = . src
CUDA_PATH ?= /usr/local/cuda
INCDIRS = . ./include $(CUDA_PATH)/include $(CUDA_PATH)/include/cccl

#Xptxas="-v" provides register usage information -Xptxas="-v" 
#O0 and -G (device side)are for debugging
# -G debug mode on GPU
#Compiler Selection based upon GPU Platform
GPU_PLATFORM ?= NVIDIA
ifeq ($(GPU_PLATFORM), AMD)
	GPUCC = hipcc
	GPUFLAGS = --amdgpu-target=gfx129
else ifeq ($(GPU_PLATFORM), NVIDIA)
	GPUCC = $(CUDA_PATH)/bin/nvcc
ifndef SM
	ARCH=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ".")
	SM=$(if $(ARCH),$(ARCH),89)
endif
	GPUFLAGS = -std=c++17 --compiler-options "" -arch=sm_$(SM) -gencode arch=compute_$(SM),code=sm_$(SM)
# 	using lambdas in device code
	GPUFLAGS += --extended-lambda 
else
	$(error "Unsupported GPU_PLATFORM: $(GPU_PLATFORM). Please set GPU_PLATFORM to either 'AMD' or 'NVIDIA'.")
endif

#Common Settings
CXX=g++
DEPFLAGS := -MP -MD


ifeq ($(DYNAMIC_RESIZE), 1)
	USER_FLAGS += -DDYNAMIC_RESIZE
endif

ifeq ($(BREAKDOWN_INSERT), 1)
	USER_FLAGS += -DBREAKDOWN_INSERT
endif

DEPFLAGS += $(USER_FLAGS)


INCLUDES=$(foreach dir,$(INCDIRS),-I$(dir))
CXXFLAGS=-std=c++17 -Wall $(HOST_DBG_FLAGS) $(INCLUDES) $(DEPFLAGS)
GPUCCFLAGS=$(GPUFLAGS) $(INCLUDES) $(DEPFLAGS) $(DEV_DBG_FLAGS)

#Source file discovery
ALL_SRCS=$(foreach dir,$(CODEDIRS), \
	$(wildcard $(dir)/*.cpp)	\
	$(wildcard $(dir)/*.cc)	\
	$(wildcard $(dir)/*.cu)	\
	$(wildcard $(dir)/*.hip.cpp))

# Filter out any benchmark files if they were accidentally picked up (though . is in CODEDIRS, benchmark/ is not)
SRCS := $(filter-out ./benchmark/*.cpp, $(ALL_SRCS))

#Object file generation
OBJECTS := $(SRCS:.cpp=.o)
OBJECTS := $(OBJECTS:.cc=.o)
OBJECTS := $(OBJECTS:.cu=.o)
OBJECTS := $(OBJECTS:.hip.cpp=.o)

#Depedency files
DEPS=$(OBJECTS:.o=.d)

#Output binaries
ifeq ($(BREAKDOWN_INSERT), 1)
	OUT_BENCH=$(BINARY)/hive_hash_table_benchmark_breakdown_insert
else
	OUT_BENCH=$(BINARY)/hive_hash_table_benchmark
endif

OUT_YCSB=$(BINARY)/hive_hash_table_ycsb
OUT_LOOKUP_ONLY_WORKLOAD=$(BINARY)/hive_hash_table_lookup_only_workload

.PHONY: all clean run diff profile benchmark ycsb

all: $(OUT_BENCH) $(OUT_YCSB) $(OUT_LOOKUP_ONLY_WORKLOAD)

benchmark: $(OUT_BENCH)
ycsb: $(OUT_YCSB)
lookup_only_workload: $(OUT_LOOKUP_ONLY_WORKLOAD)

cuda:
	$(MAKE) GPU_PLATFORM=cuda all

hip:
	$(MAKE) GPU_PLATFORM=hip all

$(BINARY):
	mkdir -p $(BINARY)

# Main Benchmark Target
$(OUT_BENCH): $(OBJECTS) benchmark/hive_table_benchmark.o | $(BINARY)
	$(GPUCC) $(GPUFLAGS) -o $@ $(OBJECTS) benchmark/hive_table_benchmark.o -lcudart

# YCSB Target
$(OUT_YCSB): $(OBJECTS) benchmark/ycsb_benchmark.o | $(BINARY)
	$(GPUCC) $(GPUFLAGS) -o $@ $(OBJECTS) benchmark/ycsb_benchmark.o -lcudart

$(OUT_LOOKUP_ONLY_WORKLOAD): $(OBJECTS) benchmark/hive_lookup_only_benchmark.o | $(BINARY)
	$(GPUCC) $(GPUFLAGS) -o $@ $(OBJECTS) benchmark/hive_lookup_only_benchmark.o -lcudart

%.o:%.cu
	$(GPUCC) $(GPUCCFLAGS) -c $< -o $@

%.o:%.cpp
	$(GPUCC) $(GPUCCFLAGS) -c $< -o $@

%.o:%.cc
	$(GPUCC) $(GPUCCFLAGS) -c $< -o $@

%.o:%.hip.cpp
	$(GPUCC) $(GPUCCFLAGS) -c $< -o $@

#Include dependency files
-include $(DEPS)

clean:
	rm -f $(OBJECTS) benchmark/hive_table_benchmark.o benchmark/ycsb_benchmark.o benchmark/hive_lookup_only_benchmark.o $(DEPS) $(OUT_BENCH) $(OUT_YCSB)
	rm -rf $(BINARY)

run: $(OUT_BENCH)
	./$(OUT_BENCH)

# Profile with ncu
profile: $(OUT_BENCH)
	ncu --kernel-name-base function -k hive_mixed_kernel --set full ./$(OUT_BENCH)

diff:
	@git status
	@git diff --color

help:
	@echo "Makefile Usage:"
	@echo "  make benchmark  : Build only the standard benchmark executable"
	@echo "  make ycsb       : Build only the YCSB executable"
	@echo "  make lookup_only_workload: Build only the lookup only workload executable"
	@echo "  make clean      : Clean up build artifacts"
	@echo "  make run        : Run the standard benchmark"
