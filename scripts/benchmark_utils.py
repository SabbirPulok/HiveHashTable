import subprocess  # For running shell commands
import os  # For file path manipulations
import csv
import datetime
import shutil

def run_benchmark(params, executable_path):
    
    # # Ensure clean build to prevent mixing flags
    subprocess.run(['make', 'clean'], capture_output=True, text=True, check=True)

    command = ['make', 'benchmark']
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    # """Run the benchmark with the given parameters and return the output."""
    command  = [executable_path]
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    
    #Join the command list into a string for printing
    print("Running command:", " ".join(command))

    try:
        #Capture the stdout and stderr
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running benchmark:"+ e.stderr + " \nstdout:"+ e.stdout)
        return None
    
def run_ycsb_workloads(params, executable_path):
    
    # # Ensure clean build to prevent mixing flags
    subprocess.run(['make', 'clean'], capture_output=True, text=True, check=True)

    command = ['make', 'ycsb']
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    # """Run the benchmark with the given parameters and return the output."""
    command  = [executable_path]
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    
    #Join the command list into a string for printing
    print("Running command:", " ".join(command))

    try:
        #Capture the stdout and stderr
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running benchmark:"+ e.stderr + " \nstdout:"+ e.stdout)
        return None
    
def run_dynamic_resize(params, executable_path):
    
    # # Ensure clean build to prevent mixing flags
    subprocess.run(['make', 'clean'], capture_output=True, text=True, check=True)

    command = ['make', 'benchmark', 'DYNAMIC_RESIZE=1']
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    # """Run the benchmark with the given parameters and return the output."""
    command  = [executable_path]
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    
    #Join the command list into a string for printing
    print("Running command:", " ".join(command))

    try:
        #Capture the stdout and stderr
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running benchmark:"+ e.stderr + " \nstdout:"+ e.stdout)
        return None


def run_benchmark_insert_breakdown(params, executable_path):
    
    # Ensure clean build to force recompilation with the new flag
    subprocess.run(['make', 'clean'], capture_output=True, text=True, check=True)

    command = ['make' , 'BREAKDOWN_INSERT=1']
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    # """Run the benchmark with the given parameters and return the output."""
    command  = [executable_path]
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    
    #Join the command list into a string for printing
    print("Running command:", " ".join(command))

    try:
        #Capture the stdout and stderr
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running benchmark:"+ e.stderr + " \nstdout:"+ e.stdout)
        return None


def run_lookup_kernel(params, executable_path):
    command = ['make', 'lookup_only_workload']
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    # """Run the benchmark with the given parameters and return the output."""
    command = [executable_path]
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])

    # Join the command list into a string for printing
    print("Running command:", " ".join(command))

    try:
        # Capture the stdout and stderr
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running benchmark:" + e.stderr + " \nstdout:" + e.stdout)
        return None

# bash rebuild.sh
# mkdir build && cd build
# cmake ..
# make
# # Running the benchmarks
# source ../scripts/benchmark.sh
def run_bght_iht_bench (device_name, bght_iht_root):
    bght_iht_root = os.path.abspath(bght_iht_root)     
    build_dir = os.path.join(bght_iht_root, "build")
    
    old_cwd = os.getcwd()

    try:
        # intentionally remove build dir if exists to ensure clean build
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)

        os.chdir(bght_iht_root)
        try:
            subprocess.run(["bash", "rebuild.sh"], check=True)
        except subprocess.CalledProcessError as e:
            print("Error running rebuild script:"+ e.stderr + " \nstdout:"+ e.stdout)
            return None
            
        os.makedirs(build_dir, exist_ok=True)
        
        os.chdir(build_dir)
        # cmake command
        print("Rebuild project for BP2HT/IHT:")
        cmake_command = ["cmake", ".."]
        res = subprocess.run(cmake_command, capture_output=True, text=True)

        # make command
        print("Making project for BP2HT/IHT:")
        make_command = ["make"]
        res = subprocess.run(make_command, capture_output=True, text=True)

        bench_cmd = "source ../scripts/benchmark.sh"
        # Have to run the benchmark script in a shell to source it
        print("Running BP2HT/IHT benchmark command:", bench_cmd)
        try:
            res = subprocess.run(
                ["bash", "-lc", bench_cmd],
                check = True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running BGHT/IHT benchmark command: {e.cmd}, error: "+  e.stderr + " \nstdout:" + e.stdout)
            return

        result_dir = os.path.join(bght_iht_root, "build/results", device_name)
        return result_dir
    finally:
        os.chdir(old_cwd)

# Run warpcore
# rm -rf build && mkdir build && cd build

# cmake .. \
#   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
#   -DCMAKE_CUDA_ARCHITECTURES=89 \
#   -DCMAKE_CUDA_STANDARD=17 -DCMAKE_CUDA_STANDARD_REQUIRED=ON \
#   -DCMAKE_CXX_STANDARD=17  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
#   -DWARPCORE_BUILD_TESTS=ON -DWARPCORE_BUILD_BENCHMARKS=ON -DWARPCORE_BUILD_EXAMPLES=ON

# cmake --build . -j
# ./benchmarks/single_value_benchmark

def run_warpcore_bench(warpcore_root):
    warpcore_root = os.path.abspath(warpcore_root)     
    build_dir = os.path.join(warpcore_root, "build")

    # intentionally remove build dir if exists to ensure clean build
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        
    os.makedirs(build_dir, exist_ok=True)

    old_cwd = os.getcwd()
    
    try:
        os.chdir(build_dir)
        # cmake command
        # find current device compute capability
        compute_capability = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        ).stdout.strip().splitlines()[0]
        compute_capability = compute_capability.replace('.', '')
        # print(f"Detected GPU compute capability: {compute_capability}")
        cmake_command = [
            "cmake", "..",
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc",
            "-DCMAKE_CUDA_ARCHITECTURES="+compute_capability,
            "-DCMAKE_CUDA_STANDARD=17", "-DCMAKE_CUDA_STANDARD_REQUIRED=ON",
            "-DCMAKE_CXX_STANDARD=17",  "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
            "-DWARPCORE_BUILD_TESTS=ON", "-DWARPCORE_BUILD_BENCHMARKS=ON", "-DWARPCORE_BUILD_EXAMPLES=ON"
        ]
        print("Rebuild project for warpcore:", " ".join(cmake_command))
        res = subprocess.run(cmake_command, check=True, capture_output=True, text=True)

        # make command
        make_command = ["cmake", "--build", ".", "-j"]
        res = subprocess.run(make_command, check=True, capture_output=True, text=True)
        bench_cmd = "./benchmarks/single_value_benchmark"
        # Have to run the benchmark binary
        print("Running Warpcore benchmark command:", bench_cmd)
        res = subprocess.run(
            [bench_cmd],
            check = True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Warpcore benchmark command: {e.cmd}, error: "+  e.stderr + " \nstdout:" + e.stdout)
        return
    finally:
        os.chdir(old_cwd)

    result_dir = os.path.join(warpcore_root, "build/results/single_value_hash_table.csv")
    print("Warpcore benchmark results at:", result_dir)
    return result_dir
# mkdir build && cd build
# cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
# make
# python3 ../bench/bencher.py -m 2 -d 0
# result save on slabhash_root/build/bench_result/table_size_experiment.csv
def run_slabhash_bsp_bench(slabhash_root):
    slabhash_root =  os.path.abspath(slabhash_root)     
    build_dir = os.path.join(slabhash_root, "build")

    # intentionally remove build dir if exists to ensure clean build
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        
    os.makedirs(build_dir, exist_ok=True)

    old_cwd = os.getcwd()
    
    try:
        os.chdir(build_dir)
        # cmake command
        print("Rebuild project for SlabHash BSP:")
        cmake_command = ["cmake", "..", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]
        res = subprocess.run(cmake_command, capture_output=True, text=True)

        # make command
        print("Making project for SlabHash BSP:")
        make_command = ["make"]
        res = subprocess.run(make_command, capture_output=True, text=True)

        bench_cmd = "python3 ../bench/bencher.py -m 2 -d 0"
        # Have to run the benchmark script in a shell to source it
        print("Running SlabHash BSP benchmark command:", bench_cmd)
        res = subprocess.run(
            ["bash", "-c", bench_cmd],
            check = True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running SlabHash BSP benchmark command: {e.cmd}, error: "+  e.stderr + " \nstdout:" + e.stdout)
        return
    finally:
        os.chdir(old_cwd)

    result_dir = os.path.join(slabhash_root, "build/bench_result/table_size_experiment.csv")
    return result_dir

# presumably bsp bench already build the project, so no need to build again
# just run the concurrent bench
# python3 ../bench/bencher.py -m 3 -d 0
# result save on slabhash_root/build/bench_result/concurrent_experiment.csv
def run_slabhash_concurrent_bench(slabhash_root):
    slabhash_root =  os.path.abspath(slabhash_root)     
    build_dir = os.path.join(slabhash_root, "build")

    old_cwd = os.getcwd()
    
    try:
        os.chdir(build_dir)

        bench_cmd = "python3 ../bench/bencher.py -m 3 -d 0"
        # Have to run the benchmark script in a shell to source it
        print("Running SlabHash Concurrent benchmark command:", bench_cmd)
        res = subprocess.run(
            ["bash", "-c", bench_cmd],
            check = True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running SlabHash Concurrent benchmark command: {e.cmd}, error: "+  e.stderr + " \nstdout:" + e.stdout)
        return
    finally:
        os.chdir(old_cwd)

    result_dir = os.path.join(slabhash_root, "build/bench_result/concurrent_experiment.csv")
    return result_dir


def run_slabhash_all_lookups_bench(slabhash_root):
    slabhash_root =  os.path.abspath(slabhash_root)     
    build_dir = os.path.join(slabhash_root, "build")

    old_cwd = os.getcwd()
    
    try:
        os.chdir(build_dir)

        bench_cmd = "python3 ../bench/bencher.py -m 6 -d 0"
        # Have to run the benchmark script in a shell to source it
        print("Running SlabHash All Lookups benchmark command:", bench_cmd)
        res = subprocess.run(
            ["bash", "-c", bench_cmd],
            check = True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running SlabHash All Lookups benchmark command: {e.cmd}, error: "+  e.stderr + " \nstdout:" + e.stdout)
        return
    finally:
        os.chdir(old_cwd)

    result_dir = os.path.join(slabhash_root, "build/bench_result/query_experiment_varied_exist_ratio.csv")
    return result_dir

def run_slabhash_dynamic_resizing_bench(slabhash_root):
    slabhash_root =  os.path.abspath(slabhash_root)     
    build_dir = os.path.join(slabhash_root, "build")

    # intentionally remove build dir if exists to ensure clean build
    # if os.path.exists(build_dir):
    #     shutil.rmtree(build_dir)
        
    os.makedirs(build_dir, exist_ok=True)

    old_cwd = os.getcwd()
    
    try:
        os.chdir(build_dir)
        # cmake command
        print("Rebuild project for SlabHash Resizing Benchmark:")
        cmake_command = ["cmake", "..", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]
        res = subprocess.run(cmake_command, capture_output=True, text=True)

        # make command
        print("Making project for SlabHash Resizing Benchmark:")
        make_command = ["make"]
        res = subprocess.run(make_command, capture_output=True, text=True)

        table_expansion_cmd = "python3 ../bench/bencher.py -m 4 -d 0"
        # Have to run the benchmark script in a shell to source it
        print("Running SlabHash Resizing (Expansion) Benchmark command:", table_expansion_cmd)
        res = subprocess.run(
            ["bash", "-c", table_expansion_cmd],
            check = True, capture_output=True, text=True
        )

        table_contraction_cmd = "python3 ../bench/bencher.py -m 5 -d 0"
        # Have to run the benchmark script in a shell to source it
        print("Running SlabHash Resizing (Contraction) Benchmark command:", table_contraction_cmd)
        res = subprocess.run(
            ["bash", "-c", table_contraction_cmd],
            check = True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running SlabHash Resizing Benchmark command: {e.cmd}, error: "+  e.stderr + " \nstdout:" + e.stdout)
        return
    finally:
        os.chdir(old_cwd)

    expansion_result = os.path.join(slabhash_root, "build/bench_result/rehash_experiment.csv")
    contraction_result = os.path.join(slabhash_root, "build/bench_result/merge_experiment.csv")
    return expansion_result, contraction_result

# mkdir build && cd build
# cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
# make
# ./static_test static_result.csv 20 24 0.9
# result stored at dycuckoo_root/build/static_result.csv
def run_dycuckoo_bsp(dycuckoo_root, output_filename, min_power, max_power, load_factor):
    dycuckoo_root =  os.path.abspath(dycuckoo_root)     
    build_dir = os.path.join(dycuckoo_root, "build")

    # intentionally remove build dir if exists to ensure clean build
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        
    os.makedirs(build_dir, exist_ok=True)

    old_cwd = os.getcwd()
    
    try:
        os.chdir(build_dir)
        # cmake command
        print("Rebuild project for DyCuckoo BSP:")
        cmake_command = ["cmake", "..", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]
        res = subprocess.run(cmake_command, capture_output=True, text=True)

        # make command
        print("Making project for DyCuckoo BSP:")
        make_command = ["make"]
        res = subprocess.run(make_command, capture_output=True, text=True)

        bench_cmd = f"./static_test {output_filename} {min_power} {max_power} {load_factor}"
        # Have to run the benchmark binary
        print("Running DyCuckoo BSP benchmark command:", bench_cmd)
        res = subprocess.run(
            bench_cmd.split(),
            check = True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running DyCuckoo BSP benchmark command: {e.cmd}, error: "+  e.stderr + " \nstdout:" + e.stdout)
        return
    finally:
        os.chdir(old_cwd)

    result_dir = os.path.join(dycuckoo_root, "build", output_filename)
    return result_dir

def run_dycuckoo_dynamic_resize(dycuckoo_root, min_power, max_power, min_bound, max_bound, init_load_factor):
    dycuckoo_root =  os.path.abspath(dycuckoo_root)     
    build_dir = os.path.join(dycuckoo_root, "build")

    # intentionally remove build dir if exists to ensure clean build
    # if os.path.exists(build_dir):
    #     shutil.rmtree(build_dir)
        
    os.makedirs(build_dir, exist_ok=True)

    old_cwd = os.getcwd()
    
    try:
        os.chdir(build_dir)
        # cmake command
        print("Rebuild project for DyCuckoo Dynamic Resize:")
        cmake_command = ["cmake", "..", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]
        res = subprocess.run(cmake_command, capture_output=True, text=True)

        # make command
        print("Making project for DyCuckoo Dynamic Resize:")
        make_command = ["make"]
        res = subprocess.run(make_command, capture_output=True, text=True)
        deletion_ratio = 5
        bench_cmd = f"./dynamic_test {max_power} {deletion_ratio} {min_power} {min_bound} {max_bound} {init_load_factor}"
        # Have to run the benchmark binary
        print("Running DyCuckoo Dynamic Resize benchmark command:", bench_cmd)
        res = subprocess.run(
            bench_cmd.split(),
            check = True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running DyCuckoo Dynamic Resize benchmark command: {e.cmd}, error: "+  e.stderr + " \nstdout:" + e.stdout)
        return
    finally:
        os.chdir(old_cwd)

    result_dir = os.path.join(dycuckoo_root, "build", "dynamic_resize.csv")
    return result_dir

def run_ncu_profile(params, executable_path, kernel_name):
    params["num_iterations"] = 1

    command = [
        "ncu",
        "--kernel-name-base", "function",
        "-k", kernel_name,
        "--set", "full",
        executable_path
    ]

    for key, value in params.items():
        command.extend([f"--{key}", str(value)])

    print("Running command:", " ".join(command))

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running ncu profile:" + e.stderr + " \nstdout:" + e.stdout)
        return None

def write_results_to_csv(results, file_dir, filename, fieldnames):
    """Write the benchmark results to a CSV file."""
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    csv_file_path = os.path.join(file_dir, filename)
    csv_file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='w', newline='') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not csv_file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow(result)
        
    print(f"Results written to {csv_file_path}")

def read_results_from_col(file_dir, col_name, cast=float):
    out = []
    with open(file_dir, "r", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # strip whitespace from field names
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        for row in reader:
            val = row.get(col_name, "")
            if val == "" or val is None:
                out.append(None)
            else:
                out.append(cast(val))
    return out

def read_results_from_multiple_cols(file_dir, col_names, cast=float):
    out = {col: [] for col in col_names}
    with open(file_dir, "r", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # strip whitespace from field names
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        for row in reader:
            for col in col_names:
                val = row.get(col, "")
                if val == "" or val is None:
                    out[col].append(None)
                else:
                    out[col].append(cast(val))
    return out

def read_results_from_csv(file_dir, filename):
    """Read benchmark results from a CSV file."""
    csv_file_path = os.path.join(file_dir, filename)
    results = []

    if os.path.isfile(csv_file_path):
        with open(csv_file_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append(row)

    return results