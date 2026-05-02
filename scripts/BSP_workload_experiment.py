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
from benchmark_utils import run_cucollections_single_bench as run_cucollections_bench

from benchmark_utils import write_results_to_csv as w_csv
from benchmark_utils import read_results_from_col as read_col
from benchmark_utils import read_results_from_multiple_cols as read_multi_col

# Define the path of benchmark executable
BENCHMARK_EXECUTABLE = "./bin/hive_hash_table_benchmark"
LOOKUP_BENCHMARK = "./bin/hive_hash_table_real_workload"

# Results directory
RESULTS_DIR = "./results"

# Compile and run competitors
BCHT_DIR = "./competitors/BGHT_IHT/"
WARPCORE_DIR = "./competitors/warpcore/"
SLABHASH_BSP_DIR = "./competitors/SlabHash/"
DYCUCKOO_BSP_DIR = "./competitors/DyCuckoo/dynamicHash/"
CUCOLLECTIONS_DIR = "./competitors/cuCollections/"

number_of_ops = [22, 23, 24, 25, 26, 27] #2^n
load_factor = 0.9
insert_ratio = 0.5
query_ratio = 0.4
delete_ratio = 0.1
num_iterations = 10


def parse_output(output):

    metrics = {}
    if output:
        # 
        device_match = re.search(r"Device Name: (.+)", output)
        if device_match:
            metrics["device_name"] = device_match.group(1).strip()

        # Config: Table Size=33554432, Load Factor=90.00%
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
        
        # Average Time over 1 iterations: 31.50 ms
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
        if len(s["y"]) != len(x_vals):
            # Pad with zeroes if needed
            s["y"] = s["y"] + [0] * (len(x_vals) - len(s["y"]))
            if len(s["y"]) > len(x_vals):
                s["y"] = s["y"][:len(x_vals)]
            
        plt.plot(
            x_vals,
            s["y"],
            label=s.get("label", "series"),
            color=s.get("color", None),
            linestyle=s.get("ls", '-'),
            marker=s.get("marker", 'o'),
            linewidth=5,
            markersize=10
        )
    if log_scale:
        plt.xscale("log", base=2)
    if two_power:
        plt.xticks(x_vals, [f"2^{k}" for k in x_vals], fontsize=20)
    else:
        plt.xticks(x_vals, [str(k) for k in x_vals], fontsize=20)
    
    plt.yticks(fontsize=20)
    plt.xlabel(x_label, fontsize=20, fontweight='bold')
    plt.ylabel(y_label, fontsize=20, fontweight='bold')
    plt.title(fig_title, fontsize=20, fontweight='bold')
    # plt.legend(fontsize=14, loc='best', frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + '/' + fig_name, dpi=300, bbox_inches='tight')
    plt.close()
    
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run BSP Workload Experiment")
    parser.add_argument("--sm", type=str, default=os.environ.get("SM", None),
                        help="Compute capability (e.g., 89, 120). Defaults to the SM environment variable.")
    args = parser.parse_args()
    
    # Pass the sm argument to the benchmark runner functions. We need to update benchmark_utils.py to accept this.
    compute_capability = args.sm

    if not os.path.exists(BENCHMARK_EXECUTABLE):
        print(f"Benchmark executable not found at {BENCHMARK_EXECUTABLE}. Please build the project first.")
        return

    data_layouts = ["HybridSoA-AoS"]

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
    cucollections_root = os.path.abspath(CUCOLLECTIONS_DIR)


    # Experiment 1: Bulk Insertion (100% Inserts) Sweeping Table Sizes
    print("Starting Experiment 1: Bulk Insertion (100% Inserts) Sweeping Table Sizes")

    print("Running WarpCore Benchmark...")
    # warpcore_result_dir = run_WarpCore(warpcore_root, compute_capability=compute_capability)
    warpcore_result_dir = os.path.join(warpcore_root, "build/results/single_value_hash_table.csv")
    print("Warpcore results at:", warpcore_result_dir)

    # print("Running SlabHash BSP Benchmark...")
    
    # slabhash_result_dir = run_SlabHash_BSP(slabhash_root)
    slabhash_result_dir = os.path.join(slabhash_root, "build/bench_result/table_size_experiment.csv")

    print("Running Dycuckoo BSP Benchmark...")
    # dycuckoo_result_dir = run_dycuckoo_bsp(dycuckoo_root, "static_result.csv", 22, 27, 0.9)
    dycuckoo_result_dir = os.path.join(dycuckoo_root, "build/static_result_0.9.csv")

    
    print("Running CuCollections Benchmark...")
    cuco_res_dir = os.path.join(cucollections_root, "build/results")
    cuco_csv_path = os.path.join(cuco_res_dir, "cuco_benchmark.csv")

    # If CSV already exists, read the columns; otherwise run benchmarks and write CSV.
    if not os.path.isfile(cuco_csv_path):
        cucollections_insert_throughputs = []
        cucollections_lookup_throughputs = []
        for table_size in table_sizes:
            print(f"Running cuCollections for table size: {table_size}")
            cuco_out = run_cucollections_bench(cucollections_root, table_size, load_factor, compute_capability=compute_capability)
            if cuco_out:
                ins = re.search(r"CuCollections Insertion Throughput:\s*([\d.]+)\s*Mops", cuco_out, flags=re.I)
                lkp = re.search(r"CuCollections Lookup Throughput:\s*([\d.]+)\s*Mops", cuco_out, flags=re.I)
                cucollections_insert_throughputs.append(float(ins.group(1)) if ins else 0.0)
                cucollections_lookup_throughputs.append(float(lkp.group(1)) if lkp else 0.0)
            else:
                cucollections_insert_throughputs.append(0.0)
                cucollections_lookup_throughputs.append(0.0)

        os.makedirs(cuco_res_dir, exist_ok=True)
        with open(cuco_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["table_size", "insert_mops", "lookup_mops"])
            for ts, ins, lkp in zip(table_sizes, cucollections_insert_throughputs, cucollections_lookup_throughputs):
                writer.writerow([ts, ins, lkp])
    else:
        cucollections_insert_throughputs = read_col(cuco_csv_path, "insert_mops")[:len(table_sizes)]
        cucollections_lookup_throughputs = read_col(cuco_csv_path, "lookup_mops")[:len(table_sizes)]

    print(f"cuCollections results saved to: {cuco_csv_path}")

    # Hive Hash Table
    hive_insert_csv_path = os.path.join(RESULTS_DIR, "bulk_insertion.csv")
    if not os.path.isfile(hive_insert_csv_path):
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
        os.makedirs(RESULTS_DIR, exist_ok=True)
        w_csv(results, RESULTS_DIR, "bulk_insertion.csv", all_possible_metric_keys)
    else:
        print(f"Reading Hive Hash Table insertion results from {hive_insert_csv_path}")
        with open(hive_insert_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "throughput_mops" in row and row["throughput_mops"]:
                    row["throughput_mops"] = float(row["throughput_mops"])
                if "num_inserts" in row and row["num_inserts"]:
                    row["num_inserts"] = float(row["num_inserts"])
                results.append(row)
    
    # run BCHT and IHT
    device_name = results[0].get("device_name", "unknown_device").replace(" ", "-")
    # device_name = "NVIDIA H100 NVL".replace(" ", "-")
    bcht_root = os.path.abspath(BCHT_DIR)

    # p2bht_iht_result_dir = run_BCHT_IHT(device_name, bcht_root, compute_capability=compute_capability)
    p2bht_iht_result_dir = os.path.join(bcht_root, "build/results", device_name)

    p2bht_rates_fixed_lf_dir = os.path.join(p2bht_iht_result_dir, "rates_fixed_lf/p2bht_rates_lfeq90.csv")
    iht_rates_fixed_lf_dir = os.path.join(p2bht_iht_result_dir, "rates_fixed_lf/iht_rates_lfeq90.csv")
    print("P2BHT results at:", p2bht_rates_fixed_lf_dir)
    print("IHT results at:", iht_rates_fixed_lf_dir)


    # Hive hash table throughput
    hive_hybrid_bulk_insert_throughputs = [r.get("throughput_mops", 0) for r in results if r.get("data_layout") == "HybridSoA-AoS" and r.get("num_inserts", 0) > 0]
    hive_aoas_bulk_insert_throughputs = [r.get("throughput_mops", 0) for r in results if r.get("data_layout") == "AaoS-LeadMetaData" and r.get("num_inserts", 0) > 0]

    # Slabhash bsp throughput (SlabHash doesn't have load factor collision)
    slabhash_bulk_insert_throughputs = read_col(slabhash_result_dir, "build_rate_mps")[:len(table_sizes)]

    # P2BHT / IHT throughput (filtered by 0.9 load factor)
    p2bht_bulk_insert_throughputs = []
    try:
        p2bht_raw = read_multi_col(p2bht_rates_fixed_lf_dir, ["num_keys", "load_factor", "insert_32"])
        for ts in table_sizes:
            target_keys = float(1 << ts)
            matches = [i for i, (k, l) in enumerate(zip(p2bht_raw["num_keys"], p2bht_raw["load_factor"])) if k == target_keys and abs(l - 0.9) < 0.01]
            p2bht_bulk_insert_throughputs.append(p2bht_raw["insert_32"][matches[-1]] if matches else 0)
    except:
        p2bht_bulk_insert_throughputs = [0]*len(table_sizes)

    iht_bulk_insert_throughputs = []
    try:
        iht_raw = read_multi_col(iht_rates_fixed_lf_dir, ["num_keys", "load_factor", "insert_32_20"])
        for ts in table_sizes:
            target_keys = float(1 << ts)
            matches = [i for i, (k, l) in enumerate(zip(iht_raw["num_keys"], iht_raw["load_factor"])) if k == target_keys and abs(l - 0.9) < 0.01]
            iht_bulk_insert_throughputs.append(iht_raw["insert_32_20"][matches[-1]] if matches else 0)
    except:
        iht_bulk_insert_throughputs = [0]*len(table_sizes)

    # Warpcore throughput (we want the 0.9 load factor, which is the 4th item if sweeping 5 LFs, or map by exact size)
    warpcore_bulk_insert_throughputs = []
    try:
        warpcore_raw = read_multi_col(warpcore_result_dir, ["sample_size", "insert_mops"])
        for ts in table_sizes:
            target_keys = float(1 << ts)
            indices = [i for i, x in enumerate(warpcore_raw["sample_size"]) if x == target_keys]
            if len(indices) >= 4:
                warpcore_bulk_insert_throughputs.append(warpcore_raw["insert_mops"][indices[3]]) # 0.9 is 4th
            else:
                warpcore_bulk_insert_throughputs.append(warpcore_raw["insert_mops"][indices[0]] if indices else 0)
    except:
        warpcore_bulk_insert_throughputs = [0]*len(table_sizes)

    # DyCuckoo throughput (read exactly from the 0.9 CSV)
    dycuckoo_bulk_insert_throughputs = []
    try:
        dy_raw = read_multi_col(dycuckoo_result_dir, ["num_keys", "insert_mops"])
        for ts in table_sizes:
            target_keys = float(1 << ts)
            try:
                idx = dy_raw["num_keys"].index(target_keys)
                dycuckoo_bulk_insert_throughputs.append(dy_raw["insert_mops"][idx])
            except ValueError:
                dycuckoo_bulk_insert_throughputs.append(0)
    except:
        dycuckoo_bulk_insert_throughputs = [0]*len(table_sizes)

    series = [
        { "label" : "Hive Hash Table", "y": hive_hybrid_bulk_insert_throughputs, "color": "red", "marker": 'o', "ls": '-'},
        { "label" : "BP2HT", "y": p2bht_bulk_insert_throughputs, "color": "blue", "marker": 's', "ls": '-.'},
        { "label" : "IHT", "y": iht_bulk_insert_throughputs, "color": "green", "marker": '^', "ls": ':'},
        { "label" : "WarpCore", "y": warpcore_bulk_insert_throughputs, "color": "purple", "marker": 'd', "ls": 'solid'},
        { "label" : "SlabHash", "y": slabhash_bulk_insert_throughputs, "color": "brown", "marker": 'x', "ls": 'dashed'},
        { "label" : "DyCuckoo", "y": dycuckoo_bulk_insert_throughputs, "color": "cyan", "marker": 'v', "ls": 'dotted'},
        { "label" : "cuCollections", "y": cucollections_insert_throughputs, "color": "magenta", "marker": "*", "ls": "-"},
    ]
    plot_results(table_sizes, series, "bulk_insertion.png",
                 "", "Inserted KV Pairs", "Insertion Throughput (M-KV/s)")
    
    # Experiment 2: Bulk Lookup (100% Lookups) Sweeping Table Sizes
    results = []
    hive_lookup_csv_path = os.path.join(RESULTS_DIR, "bulk_lookup.csv")
    if not os.path.isfile(hive_lookup_csv_path):
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
        os.makedirs(RESULTS_DIR, exist_ok=True)
        w_csv(results, RESULTS_DIR, "bulk_lookup.csv", all_possible_metric_keys)
    else:
        print(f"Reading Hive Hash Table lookup results from {hive_lookup_csv_path}")
        with open(hive_lookup_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "throughput_mops" in row and row["throughput_mops"]:
                    row["throughput_mops"] = float(row["throughput_mops"])
                results.append(row)
    
    hive_hybrid_bulk_lookup_throughputs = [r.get("throughput_mops", 0) for r in results[0:len(table_sizes)]]
    hive_aoas_bulk_lookup_throughputs = [r.get("throughput_mops", 0) for r in results[len(table_sizes):2*len(table_sizes)]]
    #slabhash bsp throughput
    slabhash_bulk_lookup_throughputs = read_col(slabhash_result_dir, "search_rate_mps")[:len(table_sizes)]

    #bp2ht throughput
    bp2ht_bulk_lookup_throughputs = []
    try:
        p2bht_raw = read_multi_col(p2bht_rates_fixed_lf_dir, ["num_keys", "load_factor", "find_32_100"])
        for ts in table_sizes:
            target_keys = float(1 << ts)
            matches = [i for i, (k, l) in enumerate(zip(p2bht_raw["num_keys"], p2bht_raw["load_factor"])) if k == target_keys and abs(l - 0.9) < 0.01]
            bp2ht_bulk_lookup_throughputs.append(p2bht_raw["find_32_100"][matches[-1]] if matches else 0)
    except:
        bp2ht_bulk_lookup_throughputs = [0]*len(table_sizes)

    #iht throughput
    iht_bulk_lookup_throughputs = []
    try:
        iht_raw = read_multi_col(iht_rates_fixed_lf_dir, ["num_keys", "load_factor", "find_32_20_100"])
        for ts in table_sizes:
            target_keys = float(1 << ts)
            matches = [i for i, (k, l) in enumerate(zip(iht_raw["num_keys"], iht_raw["load_factor"])) if k == target_keys and abs(l - 0.9) < 0.01]
            iht_bulk_lookup_throughputs.append(iht_raw["find_32_20_100"][matches[-1]] if matches else 0)
    except:
        iht_bulk_lookup_throughputs = [0]*len(table_sizes)

    #warpcore throughput
    warpcore_bulk_lookup_throughputs = []
    try:
        warpcore_raw = read_multi_col(warpcore_result_dir, ["sample_size", "query_mops"])
        for ts in table_sizes:
            target_keys = float(1 << ts)
            indices = [i for i, x in enumerate(warpcore_raw["sample_size"]) if x == target_keys]
            if len(indices) >= 3:
                warpcore_bulk_lookup_throughputs.append(warpcore_raw["query_mops"][indices[2]]) # 0.9 is 3rd
            else:
                warpcore_bulk_lookup_throughputs.append(warpcore_raw["query_mops"][indices[0]] if indices else 0)
    except:
        warpcore_bulk_lookup_throughputs = [0]*len(table_sizes)

    # DyCuckoo throughput
    dycuckoo_bulk_lookup_throughputs = []
    try:
        dy_raw = read_multi_col(dycuckoo_result_dir, ["num_keys", "query_100_mops"])
        for ts in table_sizes:
            target_keys = float(1 << ts)
            try:
                idx = dy_raw["num_keys"].index(target_keys)
                dycuckoo_bulk_lookup_throughputs.append(dy_raw["query_100_mops"][idx])
            except ValueError:
                dycuckoo_bulk_lookup_throughputs.append(0)
    except:
        dycuckoo_bulk_lookup_throughputs = [0]*len(table_sizes)

    series = [
        { "label" : "Hive Hash Table", "y": hive_hybrid_bulk_lookup_throughputs, "color": "red", "marker": 'o', "ls": '-'},
        { "label" : "BP2HT", "y": bp2ht_bulk_lookup_throughputs, "color": "blue", "marker": 's', "ls": '-.'},
        { "label" : "IHT", "y": iht_bulk_lookup_throughputs, "color": "green", "marker": '^', "ls": ':'},
        { "label" : "WarpCore", "y": warpcore_bulk_lookup_throughputs, "color": "purple", "marker": 'd', "ls": 'solid'},
        { "label" : "SlabHash", "y": slabhash_bulk_lookup_throughputs, "color": "brown", "marker": 'x', "ls": 'dashed'},
        { "label" : "DyCuckoo", "y": dycuckoo_bulk_lookup_throughputs, "color": "cyan", "marker": 'v', "ls": 'dotted'},
        { "label" : "cuCollections", "y": cucollections_lookup_throughputs, "color": "magenta", "marker": "*", "ls": "-"},
    ]
    plot_results(table_sizes, series, "bulk_lookup.png",
                "" ,"Number of Query Keys", "Query Throughput (M-KV/s)")
    



if __name__ == "__main__":
    main()
