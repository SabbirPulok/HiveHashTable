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

number_of_ops = [22, 23, 24, 25, 26, 27] #2^n
load_factor = 0.9
insert_ratio = 0.5
query_ratio = 0.4
delete_ratio = 0.1
num_iterations = 10


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
    table_sizes = number_of_ops
    warpcore_root = os.path.abspath(WARPCORE_DIR)
    slabhash_root = os.path.abspath(SLABHASH_BSP_DIR)
    dycuckoo_root = os.path.abspath(DYCUCKOO_BSP_DIR)


    # Experiment 1: Bulk Insertion (100% Inserts) Sweeping Table Sizes
    print("Starting Experiment 1: Bulk Insertion (100% Inserts) Sweeping Table Sizes")

    print("Running WarpCore Benchmark...")
    # warpcore_result_dir = run_WarpCore(warpcore_root)
    warpcore_result_dir = os.path.join(warpcore_root, "build/results/single_value_hash_table.csv")
    print("Warpcore results at:", warpcore_result_dir)

    print("Running SlabHash BSP Benchmark...")
    
    # slabhash_result_dir = run_SlabHash_BSP(slabhash_root)
    slabhash_result_dir = os.path.join(slabhash_root, "build/bench_result/table_size_experiment.csv")

    print("Running Dycuckoo BSP Benchmark...")
    dycuckoo_result_dir = run_dycuckoo_bsp(dycuckoo_root, "static_result.csv", 22, 27, 0.9)
    # dycuckoo_result_dir = os.path.join(dycuckoo_root, "build/static_result.csv")

    for data_layout in data_layouts:
        common_params["data_layout"] = data_layout
        for table_size in table_sizes:
            params = {
                **common_params,
                "table_size": table_size,
                "insert_ratio": 1.0,
                "lookup_ratio": 0.0,
                "delete_ratio": 0.0,
            }
            output= run(params, BENCHMARK_EXECUTABLE)
            metrics = parse_output(output)
            metrics["data_layout"] = data_layout # Ensure layout is in metrics
            results.append(metrics)
    
    # run BCHT and IHT
    # device_name = results[0].get("device_name", "unknown_device").replace(" ", "-")
    device_name = "NVIDIA GeForce RTX 4090".replace(" ", "-")
    bcht_root = os.path.abspath(BCHT_DIR)

    # p2bht_iht_result_dir = run_BCHT_IHT(device_name, bcht_root)
    p2bht_iht_result_dir = os.path.join(bcht_root, "build/results", device_name)

    p2bht_rates_fixed_lf_dir = os.path.join(p2bht_iht_result_dir, "rates_fixed_lf/p2bht_rates_lfeq90.csv")
    iht_rates_fixed_lf_dir = os.path.join(p2bht_iht_result_dir, "rates_fixed_lf/iht_rates_lfeq90.csv")

    #hive hash table throughput
    hive_hybrid_bulk_insert_throughputs = [r.get("throughput_mops", 0) for r in results if r.get("data_layout") == "HybridSoA-AoS" and r.get("num_inserts", 0) > 0]
    hive_aoas_bulk_insert_throughputs = [r.get("throughput_mops", 0) for r in results if r.get("data_layout") == "AaoS-LeadMetaData" and r.get("num_inserts", 0) > 0]

    #p2bht throughput
    p2bht_bulk_insert_throughputs = read_col(p2bht_rates_fixed_lf_dir, "insert_32")[:len(table_sizes)]

    #iht throughput
    iht_bulk_insert_throughputs = read_col(iht_rates_fixed_lf_dir, "insert_32_20")[:len(table_sizes)]
    
    #warpcore throughput
    warpcore_bulk_insert_throughputs = read_col(warpcore_result_dir, "insert_mops")[:len(table_sizes)]

    #Slabhash bsp throughput
    slabhash_bulk_insert_throughputs = read_col(slabhash_result_dir, "build_rate_mps")[:len(table_sizes)]
    
    # DyCuckoo throughput
    dycuckoo_bulk_insert_throughputs = read_col(dycuckoo_result_dir, "insert_mops")[:len(table_sizes)]

    series = [
        { "label" : "Hive Hash Table (HybridSoA-AoS)", "y": hive_hybrid_bulk_insert_throughputs, "color": "red", "marker": 'o', "ls": '-'},
        { "label" : "Hive Hash Table (AoAS)", "y": hive_aoas_bulk_insert_throughputs, "color": "orange", "marker": 'o', "ls": '--'},
        { "label" : "BP2HT", "y": p2bht_bulk_insert_throughputs, "color": "blue", "marker": 's', "ls": '-.'},
        { "label" : "IHT", "y": iht_bulk_insert_throughputs, "color": "green", "marker": '^', "ls": ':'},
        { "label" : "WarpCore", "y": warpcore_bulk_insert_throughputs, "color": "purple", "marker": 'd', "ls": 'solid'},
        { "label" : "SlabHash", "y": slabhash_bulk_insert_throughputs, "color": "brown", "marker": 'x', "ls": 'dashed'},
        { "label" : "DyCuckoo", "y": dycuckoo_bulk_insert_throughputs, "color": "cyan", "marker": 'v', "ls": 'dotted'},
    ]
    #Write results to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bulk_insertion_{timestamp}.csv"
    w_csv(results, RESULTS_DIR, filename, all_possible_metric_keys)
    plot_results(table_sizes, series, filename.replace(".csv", ".png"),
                 "", "Number of KV Pairs Inserted", "Bulk Insertion Throughput (MOPS)")
    
    # Experiment 2: Bulk Lookup (100% Lookups) Sweeping Table Sizes
    results = []
    for data_layout in data_layouts:
        common_params["data_layout"] = data_layout
        for table_size in table_sizes:
            print(f"Running lookup experiment for table size: {table_size}")
            params = {
                **common_params,
                "table_size": table_size,
                "insert_ratio": 0.0,
                "lookup_ratio": 1.0,
                "delete_ratio": 0.0,
            }
            output= run(params, BENCHMARK_EXECUTABLE)
            metrics = parse_output(output)
            results.append(metrics)
    
    hive_hybrid_bulk_lookup_throughputs = [r.get("throughput_mops", 0) for r in results[0:len(table_sizes)]]
    hive_aoas_bulk_lookup_throughputs = [r.get("throughput_mops", 0) for r in results[len(table_sizes):2*len(table_sizes)]]
    #bp2ht throughput
    bp2ht_bulk_lookup_throughputs = read_col(p2bht_rates_fixed_lf_dir, "find_32_100")[:len(table_sizes)]
    #iht throughput
    iht_bulk_lookup_throughputs = read_col(iht_rates_fixed_lf_dir, "find_32_20_100")[:len(table_sizes)]
    #warpcore throughput
    warpcore_bulk_lookup_throughputs = read_col(warpcore_result_dir, "query_mops")[:len(table_sizes)]
    #slabhash bsp throughput
    slabhash_bulk_lookup_throughputs = read_col(slabhash_result_dir, "search_rate_mps")[:len(table_sizes)]
    # DyCuckoo throughput
    dycuckoo_bulk_lookup_throughputs = read_col(dycuckoo_result_dir, "query_100_mops")[:len(table_sizes)]

    series = [
        { "label" : "Hive Hash Table (HybridSoA-AoS)", "y": hive_hybrid_bulk_lookup_throughputs, "color": "red", "marker": 'o', "ls": '-'},
        { "label" : "Hive Hash Table (AoAS)", "y": hive_aoas_bulk_lookup_throughputs, "color": "orange", "marker": 'o', "ls": '--'},
        { "label" : "BP2HT", "y": bp2ht_bulk_lookup_throughputs, "color": "blue", "marker": 's', "ls": '-.'},
        { "label" : "IHT", "y": iht_bulk_lookup_throughputs, "color": "green", "marker": '^', "ls": ':'},
        { "label" : "WarpCore", "y": warpcore_bulk_lookup_throughputs, "color": "purple", "marker": 'd', "ls": 'solid'},
        { "label" : "SlabHash", "y": slabhash_bulk_lookup_throughputs, "color": "brown", "marker": 'x', "ls": 'dashed'},
        { "label" : "DyCuckoo", "y": dycuckoo_bulk_lookup_throughputs, "color": "cyan", "marker": 'v', "ls": 'dotted'},
    ]
    #Write results to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bulk_lookup_{timestamp}.csv"
    w_csv(results, RESULTS_DIR, filename, all_possible_metric_keys)
    
    plot_results(table_sizes, series, filename.replace(".csv", ".png"),
                "" ,"Number of Query Keys", "Bulk Query Throughput (MOPS)")
    
    #Experiment 3: Mixed Workload (50% Inserts, 40% Lookups, 10% Deletes) Sweeping Table Sizes
    results = []
    print("Starting Experiment 3: Mixed Workload (50% Inserts, 40% Lookups, 10% Deletes) Sweeping Table Sizes")
    print("Running Hive Hash Table Benchmark...")
    common_params["num_iterations"] = 1
    table_sizes = [20, 21, 22, 23, 24, 25, 26]
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
    # slabhash_concurrent_result_dir = run_SlabHash_conc(slabhash_root)
    slabhash_concurrent_result_dir = os.path.join(slabhash_root, "build/bench_result/concurrent_experiment.csv")
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

    #Experiment 4: Query Throughput With Exist Ratio
    table_size = 24
    for data_layout in data_layouts:
        common_params["data_layout"] = data_layout
        params = {
            "table_size": table_size,
            "insert_ratio": 0.0,
            "lookup_ratio": 1.0,
            "delete_ratio": 0.0,
            "n_operations":1<<24 #nLookup operations
        }
        output= run_lookup(params, LOOKUP_BENCHMARK)
        metrics = parse_output(output)
        metrics["data_layout"] = data_layout
        results.append(metrics)

    hybrid_results = [r for r in results if r.get("data_layout") == "HybridSoA-AoS" and "lookup_100_exist_throughput" in r]
    aoas_results = [r for r in results if r.get("data_layout") == "AaoS-LeadMetaData" and "lookup_100_exist_throughput" in r]

    exist_keys = [
        "lookup_0_exist_throughput",
        "lookup_25_exist_throughput",
        "lookup_50_exist_throughput",
        "lookup_75_exist_throughput",
        "lookup_100_exist_throughput",
    ]

    hive_hybrid_all_lookup_exist_throughput = [hybrid_results[0].get(k, 0) for k in exist_keys] if hybrid_results else [0]*5
    hive_aoas_all_lookup_exist_throughput = [aoas_results[0].get(k, 0) for k in exist_keys] if aoas_results else [0]*5

    #bp2ht throughput
    bp2ht_cols = ["num_keys", "find_32_100", "find_32_75", "find_32_50", "find_32_25", "find_32_0"]
    bp2ht_raw = read_multi_col(p2bht_rates_fixed_lf_dir, bp2ht_cols)
    try:
        # csv reads as float often, handle potential type mismatch
        # The benchmark_utils casting defaults to float
        target_key = float(1 << table_size)
        idx = bp2ht_raw["num_keys"].index(target_key)
        bp2ht_lookup_exist_throughputs = [bp2ht_raw["find_32_0"][idx], bp2ht_raw["find_32_25"][idx], bp2ht_raw["find_32_50"][idx], bp2ht_raw["find_32_75"][idx], bp2ht_raw["find_32_100"][idx]]
    except (ValueError, KeyError, IndexError):
        bp2ht_lookup_exist_throughputs = [0]*5

    #iht throughput
    iht_cols = ["num_keys", "find_32_20_100", "find_32_20_75", "find_32_20_50", "find_32_20_25", "find_32_20_0"]
    iht_raw = read_multi_col(iht_rates_fixed_lf_dir, iht_cols)
    try:
        target_key = float(1 << table_size)
        idx = iht_raw["num_keys"].index(target_key)
        iht_lookup_exist_throughputs = [iht_raw["find_32_20_0"][idx], iht_raw["find_32_20_25"][idx], iht_raw["find_32_20_50"][idx], iht_raw["find_32_20_75"][idx], iht_raw["find_32_20_100"][idx]]
    except (ValueError, KeyError, IndexError):
        iht_lookup_exist_throughputs = [0]*5

    warpcore_cols = ["sample_size", "query_100_mops", "query_75_mops", "query_50_mops", "query_25_mops", "query_0_mops"]
    warpcore_raw = read_multi_col(warpcore_result_dir, warpcore_cols)
    try:
        target_key = float(1 << table_size)
        idx = warpcore_raw["sample_size"].index(target_key)
        warpcore_lookup_exist_throughputs = [warpcore_raw[k][idx] for k in ["query_0_mops", "query_25_mops", "query_50_mops", "query_75_mops", "query_100_mops"]]
    except (ValueError, KeyError, IndexError):
        warpcore_lookup_exist_throughputs = [0]*5

    # DyCuckoo throughput
    dycuckoo_cols = ["num_keys", "query_100_mops", "query_75_mops", "query_50_mops", "query_25_mops", "query_0_mops"]
    dycuckoo_raw = read_multi_col(dycuckoo_result_dir, dycuckoo_cols)
    try:
        target_key = float(1 << table_size)
        idx = dycuckoo_raw["num_keys"].index(target_key)
        dycuckoo_lookup_exist_throughputs = [dycuckoo_raw[k][idx] for k in ["query_0_mops", "query_25_mops", "query_50_mops", "query_75_mops", "query_100_mops"]]
    except (ValueError, KeyError, IndexError):
        dycuckoo_lookup_exist_throughputs = [0]*5

    print("Running SlabHash All Lookup Experiemnt")
    # slabhash_lookup_exist_result_dir = run_SlabHash_lookups(slabhash_root)
    slabhash_lookup_exist_result_dir = os.path.join(slabhash_root, "build/bench_result/query_experiment_varied_exist_ratio.csv")
    slabhash_lookup_exist_throughputs = [0]*5
    if os.path.exists(slabhash_lookup_exist_result_dir):
        # Assuming slabhash csv might just have one column or specific structure, 
        # but previously it was read_col query_rate_mops.
        # If it's a single run for this table size, read_col should work if it returns list.
        # We need 5 values.
        sh_vals = read_col(slabhash_lookup_exist_result_dir, "query_rate_mps")
        if sh_vals and len(sh_vals) >= 5:
            slabhash_lookup_exist_throughputs = sh_vals[:5][::-1]

    series = [
        { "label" : "Hive Hash Table (HybridSoA-AoS)", "y": hive_hybrid_all_lookup_exist_throughput, "color": "red", "marker": 'o', "ls": '-'},
        { "label" : "Hive Hash Table (AoAS)", "y": hive_aoas_all_lookup_exist_throughput, "color": "orange", "marker": 'o', "ls": '--'},
        { "label" : "BP2HT", "y": bp2ht_lookup_exist_throughputs, "color": "blue", "marker": 's', "ls": '-.'},
        { "label" : "IHT", "y": iht_lookup_exist_throughputs, "color": "green", "marker": '^', "ls": ':'},
        { "label" : "WarpCore", "y": warpcore_lookup_exist_throughputs, "color": "purple", "marker": 'd', "ls": 'solid'},
        { "label" : "SlabHash", "y": slabhash_lookup_exist_throughputs, "color": "brown", "marker": 'x', "ls": 'dashed'},
        { "label" : "DyCuckoo", "y": dycuckoo_lookup_exist_throughputs, "color": "cyan", "marker": 'v', "ls": 'dotted'},
    ]
    #Write results to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lookup_exist_{timestamp}.csv"
    w_csv(results, RESULTS_DIR, filename, all_possible_metric_keys)

    plot_results([0.0, 0.25, 0.5, 0.75, 1.0], series, filename.replace(".csv", ".png"),
                "" ,"Positive Queries (%)", "Throughput (MOPS)", two_power=False, log_scale=False)

    # #Experiment 4: Mixed Workload with variable thread_blocks
    # results = []
    # thread_blocks = [64, 128, 256, 512, 1024]
    # for tb in thread_blocks:
    #     params = {
    #         **common_params,
    #         "table_size": table_sizes[-1],
    #         "insert_ratio": insert_ratio,
    #         "lookup_ratio": query_ratio,
    #         "delete_ratio": delete_ratio,
    #         "threads-per-block": tb
    #     }
    #     output= run(params, BENCHMARK_EXECUTABLE)
    #     metrics = parse_output(output)
    #     results.append(metrics)
    # #Write results to CSV
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"mixed_workload_{timestamp}.csv"
    # w_csv(results, RESULTS_DIR, filename, all_possible_metric_keys)
    # plot_results(thread_blocks, [ [r.get("throughput_mops", 0) for r in results] ], filename.replace(".csv", ".png"),
    #             f"Imbalanced Workload Throughput (Table Size = 2^{table_sizes[-1]})","Threads per Block", "Throughput (MOPS)", False)

if __name__ == "__main__":
    main()


    



