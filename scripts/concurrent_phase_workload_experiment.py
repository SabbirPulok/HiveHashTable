import subprocess  # For running shell commands
import os  # For file path manipulations
import csv
import datetime
import re # For regex operations
import matplotlib.pyplot as plt  # For plotting graphs
from benchmark_utils import run_benchmark as run
from benchmark_utils import run_lookup_kernel as run_lookup
from benchmark_utils import run_bght_iht_bench as run_BCHT_IHT
from benchmark_utils import run_warpcore_bench as run_WarpCore
from benchmark_utils import run_slabhash_bsp_bench as run_SlabHash_BSP
from benchmark_utils import run_slabhash_concurrent_bench as run_SlabHash_conc
from benchmark_utils import run_slabhash_all_lookups_bench as run_SlabHash_lookups
from benchmark_utils import run_dycuckoo_bsp

from benchmark_utils import write_results_to_csv as w_csv
from benchmark_utils import read_results_from_col as read_col
from benchmark_utils import read_results_from_multiple_cols as read_multi_col

#Define the path of benchmark executable
BENCHMARK_EXECUTABLE = "./bin/hive_hash_table_benchmark"
LOOKUP_BENCHMARK = "./bin/hive_hash_table_lookup_only_workload"

 #Results directory
RESULTS_DIR = "./results"

#Compile and run competitors
BCHT_DIR = "./competitors/BGHT_IHT/"
WARPCORE_DIR = "./competitors/warpcore/"
SLABHASH_BSP_DIR = "./competitors/SlabHash/"
DYCUCKOO_BSP_DIR = "./competitors/DyCuckoo/dynamicHash/"

table_sizes = [20, 21, 22, 23, 24, 25, 26] #2^n
load_factor = 0.9
insert_ratio = 0.5
query_ratio = 0.4
delete_ratio = 0.1
num_iterations = 1


def parse_output(output):

    metrics = {}
    if output:
        # Device Name: NVIDIA GeForce RTX 2080 Ti
        device_match = re.search(r"Device Name: (.+)", output)
        if device_match:
            metrics["device_name"] = device_match.group(1).strip()

        #Config: Table Size=33554432, Load Factor=90.00%
        config_match = re.search(r"Config: Table Size=(\d+), Load Factor=([\d.]+)%", output, flags=re.DOTALL)
                                
        if config_match:
            metrics["table_size"] = int(config_match.group(1)) # Table Size in millions
            metrics["load_factor"] = float(config_match.group(2))
        
        # Num Inserts
        num_inserts_match = re.search(r"Num Inserts: ([\d.]+) (billions|millions|thousands)", output, flags=re.DOTALL)
        if num_inserts_match:
            num_inserts_val = float(num_inserts_match.group(1))
            num_inserts_unit = num_inserts_match.group(2)
            metrics["num_inserts"] = num_inserts_val * 1000 if num_inserts_unit == "billions" else num_inserts_val / 1000 if num_inserts_unit == "thousands" else num_inserts_val

        # Num Queries
        num_queries_match = re.search(r"Num Queries: ([\d.]+)( (billions|millions|thousands))?", output, flags=re.DOTALL)
        if num_queries_match:
            num_queries_val = float(num_queries_match.group(1))
            num_queries_unit = num_queries_match.group(3) # group(3) captures 'billions', 'millions' or 'thousands' if present
            metrics["num_queries"] = num_queries_val * 1000 if num_queries_unit == "billions" else num_queries_val if num_queries_unit == "millions" else num_queries_val / 1000 if num_queries_unit == "thousands" else num_queries_val

        # Num Deletes
        num_deletes_match = re.search(r"Num Deletes: ([\d.]+)( (billions|millions|thousands))?", output, flags=re.DOTALL)
        if num_deletes_match:
            num_deletes_val = float(num_deletes_match.group(1))
            num_deletes_unit = num_deletes_match.group(3) # group(3) captures 'billions', 'millions' or 'thousands' if present
            metrics["num_deletes"] = num_deletes_val * 1000 if num_deletes_unit == "billions" else num_deletes_val if num_deletes_unit == "millions" else num_deletes_val / 1000 if num_deletes_unit == "thousands" else num_deletes_val
        total_ops_match = re.search(r"Total Ops: ([\d.]+) (billions|millions|thousands)", output, flags=re.DOTALL)
        if total_ops_match:
            total_ops_val = float(total_ops_match.group(1))
            total_ops_unit = total_ops_match.group(2)
            metrics["total_ops"] = total_ops_val * 1000 if total_ops_unit == "billions" else total_ops_val if total_ops_unit == "millions" else total_ops_val / 1000 if total_ops_unit == "thousands" else total_ops_val
        
        # Num Blocks: 943719, Threads per Block: 1024
        blocks_threads_match = re.search(r"Num Blocks: (\d+), Threads per Block: (\d+)", output, flags=re.DOTALL)
        if blocks_threads_match:
            metrics["num_blocks"] = int(blocks_threads_match.group(1))
            metrics["threads_per_block"] = int(blocks_threads_match.group(2))
        
        #Average Time over 1 iterations: 31.50 ms
        time_match = re.search(r"Average Time over (\d+) iterations: ([\d.]+) ms", output, flags=re.DOTALL)
        if time_match:
            metrics["avg_time_ms"] = float(time_match.group(2))

        # Unsuccessful ops: 0 out of 30198988, Success Rate: 100.00%
        success_rate_match = re.search(r"Success Rate: ([\d.]+)%", output, flags=re.DOTALL)
        if success_rate_match:
            metrics["success_rate"] = float(success_rate_match.group(1))
        
        throughput_match = re.search(r"Throughput: ([\d.]+)", output, flags=re.DOTALL)
        if throughput_match:
            metrics["throughput_mops"] = float(throughput_match.group(1))
        print("Throughput: ", metrics.get("throughput_mops", "N/A"))

        # Query Throughput (q% exist)
        lookup_100_exist_throughput_match = re.search(r"Throughput \(100% exist\): ([\d.]+)", output, flags=re.DOTALL)
        if lookup_100_exist_throughput_match:
            metrics["lookup_100_exist_throughput"] = float(lookup_100_exist_throughput_match.group(1))

        lookup_75_exist_throughput_match = re.search(r"Throughput \(75% exist\): ([\d.]+)", output, flags=re.DOTALL)
        if lookup_75_exist_throughput_match:
            metrics["lookup_75_exist_throughput"] = float(lookup_75_exist_throughput_match.group(1))

        lookup_50_exist_throughput_match = re.search(r"Throughput \(50% exist\): ([\d.]+)", output, flags=re.DOTALL)
        if lookup_50_exist_throughput_match:
            metrics["lookup_50_exist_throughput"] = float(lookup_50_exist_throughput_match.group(1))

        lookup_25_exist_throughput_match = re.search(r"Throughput \(25% exist\): ([\d.]+)", output, flags=re.DOTALL)
        if lookup_25_exist_throughput_match:
            metrics["lookup_25_exist_throughput"] = float(lookup_25_exist_throughput_match.group(1))

        lookup_0_exist_throughput_match = re.search(r"Throughput \(0% exist\): ([\d.]+)", output, flags=re.DOTALL)
        if lookup_0_exist_throughput_match:
            metrics["lookup_0_exist_throughput"] = float(lookup_0_exist_throughput_match.group(1))

    return metrics

def plot_results(x_vals, series, fig_name, fig_title, x_label, y_label, two_power=True, log_scale=True):

    plt.figure(figsize=(10,6))

    for s in series:
        plt.plot(
            x_vals,
            s["y"],
            label=s.get("label", "series"),
            color=s.get("color", None),
            linestyle=s.get("ls", '-'),
            marker=s.get("marker", 'o'),
            linewidth=3,
            markersize=6
        )
    if log_scale:
        plt.xscale("log", base=2)
    if two_power:
        plt.xticks(x_vals, [f"2^{k}" for k in x_vals])
    else:
        plt.xticks(x_vals, [str(k) for k in x_vals])
    
    plt.xlabel(x_label, fontsize=16, fontweight='bold')
    plt.ylabel(y_label, fontsize=16, fontweight='bold')
    plt.title(fig_title, fontsize=18, fontweight='bold')
    plt.legend(fontsize=14, loc='best', frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + '/' + fig_name, dpi=300, bbox_inches='tight')
    plt.close()
    
def main():
    if not os.path.exists(BENCHMARK_EXECUTABLE):
        print(f"Benchmark executable not found at {BENCHMARK_EXECUTABLE}. Please build the project first.")
        return

    data_layouts = ["HybridSoA-AoS", "AaoS-LeadMetaData"]

    common_params = {
        "load_factor": load_factor, #fixed load factor of 90%
        "distribution": "uniform",
        "num_iterations": num_iterations,
    }

    #Define all possible metric keys that parse_output can generate
    all_possible_metric_keys = [
        "device_name", "table_size", "data_layout","load_factor", "num_inserts", "num_queries", "num_deletes",
        "total_ops", "num_blocks", "threads_per_block", "avg_time_ms",
        "success_rate", "throughput_mops",
        "lookup_100_exist_throughput", "lookup_75_exist_throughput", "lookup_50_exist_throughput",
        "lookup_25_exist_throughput", "lookup_0_exist_throughput"
    ]

    results = []
    warpcore_root = os.path.abspath(WARPCORE_DIR)
    slabhash_root = os.path.abspath(SLABHASH_BSP_DIR)
    dycuckoo_root = os.path.abspath(DYCUCKOO_BSP_DIR)

    #Experiment 3: Mixed Workload (50% Inserts, 40% Lookups, 10% Deletes) Sweeping Table Sizes
    results = []
    print("Starting Experiment 3: Mixed Workload (50% Inserts, 40% Lookups, 10% Deletes) Sweeping Table Sizes")
    print("Running Hive Hash Table Benchmark...")
    common_params["num_iterations"] = 1
    
    for data_layout in data_layouts:
        common_params["data_layout"] = data_layout
        for table_size in table_sizes:
            params = {
                **common_params,
                "table_size": table_size,
                "insert_ratio": 0.5,
                "lookup_ratio": 0.4,
                "delete_ratio": 0.1,
            }
            output= run(params, BENCHMARK_EXECUTABLE)
            metrics = parse_output(output)
            results.append(metrics)
    
    hive_hybrid_concurrent_throughputs = [r.get("throughput_mops", 0) for r in results[0:len(table_sizes)]]
    hive_aoas_concurrent_throughputs = [r.get("throughput_mops", 0) for r in results[len(table_sizes):2*len(table_sizes)]]
    # Running Benchmark for Slabhash Concurrent Workload
    print("Running SlabHash BSP Concurrent Workload Benchmark...")
    slabhash_concurrent_result_dir = run_SlabHash_conc(slabhash_root)
    # slabhash_concurrent_result_dir = os.path.join(slabhash_root, "build/bench_result/concurrent_experiment.csv")
    slabhash_concurrent_throughputs = read_col(slabhash_concurrent_result_dir, "concurrent_rate_mps")[:len(table_sizes)]
    
    series = [
        { "label" : "Hive Hash Table (HybridSoA-AoS)", "y": hive_hybrid_concurrent_throughputs, "color": "red", "marker": 'o', "ls": '-'},
        { "label" : "Hive Hash Table (AoAS)", "y": hive_aoas_concurrent_throughputs, "color": "orange", "marker": 'o', "ls": '--'},
        { "label" : "SlabHash", "y": slabhash_concurrent_throughputs, "color": "brown", "marker": 'x', "ls": '-.'},
    ]
    #Write results to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mixed_workload_{timestamp}.csv"
    w_csv(results, RESULTS_DIR, filename, all_possible_metric_keys)
    
    plot_results(table_sizes, series, filename.replace(".csv", ".png"),
                 "", "Number of Operations", "Throughput (MOPS)")

  
if __name__ == "__main__":
    main()


    



