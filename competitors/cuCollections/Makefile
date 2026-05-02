NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 --expt-extended-lambda
INCLUDES = -I external/cuCollections/include

all: build/cucoBench

build:
	mkdir -p build

build/cucoBench: cucoBench.cu | build
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -rf build
