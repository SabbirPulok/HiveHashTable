import subprocess
import os
import csv
import datetime
import re
import matplotlib.pyplot as plt
from benchmark_utils import run_lookup_kernel as run_lookup
from benchmark_utils import run_benchmark as run
from benchmark_utils import run_slabhash_all_lookups_bench as run_SlabHash_lookups
from benchmark_utils import write_results_to_csv as w_csv
from benchmark_utils import read_results_from_col as read_col
from benchmark_utils import read_results_from_multiple_cols as read_multi_col
from benchmark_utils import run_warpcore_bench
from benchmark_utils import run_cucollections_insert_bench  
from benchmark_utils import run_cucollections_lookup_bench

# Define the path of benchmark executable
LOOKUP_BENCHMARK = "./bin/hive_hash_table_lookup_only_workload"
BENCHMARK_EXECUTABLE = "./bin/hive_hash_table_benchmark"

# Results directory
RESULTS_DIR = "./results"

# Compile and run competitors
BCHT_DIR = "./competitors/BGHT_IHT/"
WARPCORE_DIR = "./competitors/warpcore/"
SLABHASH_BSP_DIR = "./competitors/SlabHash/"
DYCUCKOO_BSP_DIR = "./competitors/DyCuckoo/dynamicHash/"
CUCOLLECTIONS_DIR = "./competitors/cuCollections/"

def parse_output(output):
    metrics = {}
    if output:
        device_match = re.search(r"Device Name: (.+)", output)
        if device_match:
            metrics["device_name"] = device_match.group(1).strip()

        config_match = re.search(r"Config: Table Size=(\d+), Load Factor=([\d.]+)%", output, flags=re.DOTALL)
        if config_match:
            metrics["table_size"] = int(config_match.group(1))
            metrics["load_factor"] = float(config_match.group(2))
        
        num_inserts_match = re.search(r"Num Inserts: ([\d.]+) (billions|millions|thousands)", output, flags=re.DOTALL)
        if num_inserts_match:
            num_inserts_val = float(num_inserts_match.group(1))
            num_inserts_unit = num_inserts_match.group(2)
            metrics["num_inserts"] = num_inserts_val * 1000 if num_inserts_unit == "billions" else num_inserts_val / 1000 if num_inserts_unit == "thousands" else num_inserts_val

        num_queries_match = re.search(r"Num Queries: ([\d.]+)( (billions|millions|thousands))?", output, flags=re.DOTALL)
        if num_queries_match:
            num_queries_val = float(num_queries_match.group(1))
            num_queries_unit = num_queries_match.group(3)
            metrics["num_queries"] = num_queries_val * 1000 if num_queries_unit == "billions" else num_queries_val if num_queries_unit == "millions" else num_queries_val / 1000 if num_queries_unit == "thousands" else num_queries_val

        num_deletes_match = re.search(r"Num Deletes: ([\d.]+)( (billions|millions|thousands))?", output, flags=re.DOTALL)
        if num_deletes_match:
            num_deletes_val = float(num_deletes_match.group(1))
            num_deletes_unit = num_deletes_match.group(3)
            metrics["num_deletes"] = num_deletes_val * 1000 if num_deletes_unit == "billions" else num_deletes_val if num_deletes_unit == "millions" else num_deletes_val / 1000 if num_deletes_unit == "thousands" else num_deletes_val
        
        total_ops_match = re.search(r"Total Ops: ([\d.]+) (billions|millions|thousands)", output, flags=re.DOTALL)
        if total_ops_match:
            total_ops_val = float(total_ops_match.group(1))
            total_ops_unit = total_ops_match.group(2)
            metrics["total_ops"] = total_ops_val * 1000 if total_ops_unit == "billions" else total_ops_val if total_ops_unit == "millions" else total_ops_val / 1000 if total_ops_unit == "thousands" else total_ops_val
        
        blocks_threads_match = re.search(r"Num Blocks: (\d+), Threads per Block: (\d+)", output, flags=re.DOTALL)
        if blocks_threads_match:
            metrics["num_blocks"] = int(blocks_threads_match.group(1))
            metrics["threads_per_block"] = int(blocks_threads_match.group(2))
        
        time_match = re.search(r"Average Time over (\d+) iterations: ([\d.]+) ms", output, flags=re.DOTALL)
        if time_match:
            metrics["avg_time_ms"] = float(time_match.group(2))

        success_rate_match = re.search(r"Success Rate: ([\d.]+)%", output, flags=re.DOTALL)
        if success_rate_match:
            metrics["success_rate"] = float(success_rate_match.group(1))
        
        throughput_match = re.search(r"Throughput: ([\d.]+)", output, flags=re.DOTALL)
        if throughput_match:
            metrics["throughput_mops"] = float(throughput_match.group(1))
        print("Throughput: ", metrics.get("throughput_mops", "N/A"))

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
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    for s in series:
        if len(s["y"]) != len(x_vals):
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

def main():
    if not os.path.exists(LOOKUP_BENCHMARK):
        print(f"Benchmark executable not found at {LOOKUP_BENCHMARK}. Please build the project first.")
        return

    data_layouts = ["HybridSoA-AoS"]

    common_params = {
        "load_factor": 0.9,
        "distribution": "uniform",
        "num_iterations": 10,
    }

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
    bcht_root = os.path.abspath(BCHT_DIR)
    
    warpcore_result_dir = os.path.join(warpcore_root, "build/results/single_value_hash_table.csv")
    dycuckoo_result_dir = os.path.join(dycuckoo_root, "build/static_result_0.9.csv")
    device_name = "NVIDIA H100 NVL".replace(" ", "-")
    p2bht_iht_result_dir = os.path.join(bcht_root, "build/results", device_name)
    p2bht_rates_fixed_lf_dir = os.path.join(p2bht_iht_result_dir, "rates_fixed_lf/p2bht_rates_lfeq90.csv")
    iht_rates_fixed_lf_dir = os.path.join(p2bht_iht_result_dir, "rates_fixed_lf/iht_rates_lfeq90.csv")

    print("Starting Experiment 3: Insert Throughput vs Load Factor")
    insert_table_size = 24
    load_factors = [0.65, 0.80, 0.9, 0.95, 1.0]

    print("Running cuCollections benchmark...")
    cucollections_root = os.path.abspath(CUCOLLECTIONS_DIR)
    cuco_insert_csv = os.path.join(cucollections_root, "build", "results", "cuco_insert_lf.csv")

    if not os.path.isfile(cuco_insert_csv):
        print(f"Results not found at {cuco_insert_csv}. Running cuCollections Insert Benchmark...")
        # Note: Ensure run_cucollections_insert_bench is imported from benchmark_utils
        run_cucollections_insert_bench(cucollections_root, insert_table_size, load_factors)
    else:
        print(f"Found existing data at {cuco_insert_csv}. Skipping cuCollections insert execution...")

    cuco_insert_throughputs = [0] * len(load_factors)
    if os.path.exists(cuco_insert_csv):
        cuco_insert_throughputs = read_col(cuco_insert_csv, "insert_mops")[:len(load_factors)]        


    warpcore_insert_throughputs = [0]*5
    try:
        warpcore_cols = ["sample_size", "insert_mops"]
        warpcore_raw = read_multi_col(warpcore_result_dir, warpcore_cols)
        target_sample_size = float(1 << insert_table_size)
        
        # Find all indices where the sample size matches our target table size
        indices = [i for i, x in enumerate(warpcore_raw["sample_size"]) if x == target_sample_size]
        
        if len(indices) == 5:
            warpcore_insert_throughputs = [warpcore_raw["insert_mops"][i] for i in indices]
        else:
            print(f"Warning: Expected 5 WarpCore results for size {insert_table_size}, found {len(indices)}")
    except Exception as e:
        print(f"Warning: Failed to extract WarpCore insert throughputs: {e}")

    insert_results = []
    hive_insert_lf_csv = os.path.join(RESULTS_DIR, "insert_only_load_factor.csv")
    if not os.path.isfile(hive_insert_lf_csv):
        for data_layout in data_layouts:
            for lf in load_factors:
                params = {
                    **common_params,
                    "data_layout": data_layout,
                    "table_size": insert_table_size,
                    "load_factor": lf,
                    "insert_ratio": 1.0,
                    "lookup_ratio": 0.0,
                    "delete_ratio": 0.0,
                }
                output = run(params, BENCHMARK_EXECUTABLE)
                metrics = parse_output(output)
                metrics["data_layout"] = data_layout
                metrics["load_factor"] = lf
                insert_results.append(metrics)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        w_csv(insert_results, RESULTS_DIR, "insert_only_load_factor.csv", all_possible_metric_keys)
    else:
        print(f"Reading Hive Hash Table insert load factor results from {hive_insert_lf_csv}")
        with open(hive_insert_lf_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "throughput_mops" in row and row["throughput_mops"]:
                    row["throughput_mops"] = float(row["throughput_mops"])
                if "load_factor" in row and row["load_factor"]:
                    row["load_factor"] = float(row["load_factor"])
                insert_results.append(row)
            
    hybrid_insert_throughputs = [r.get("throughput_mops", 0) for r in insert_results if r.get("data_layout") == "HybridSoA-AoS"]

    # Extract P2BHT insert throughputs
    p2bht_insert_throughputs = [0]*5
    try:
        p2bht_cols = ["num_keys", "load_factor", "insert_32"]
        target_key = float(1 << insert_table_size)
        
        for i, lf in enumerate(load_factors):
            # Convert 0.65 to "65", 1.0 to "100", 0.9 to "90"
            lf_str = str(int(lf * 100))
            p2bht_file = os.path.join(p2bht_iht_result_dir, f"rates_fixed_lf/p2bht_rates_lfeq{lf_str}.csv")
            if os.path.exists(p2bht_file):
                p2bht_raw = read_multi_col(p2bht_file, p2bht_cols)
                matches = [idx for idx, (k, l) in enumerate(zip(p2bht_raw["num_keys"], p2bht_raw["load_factor"])) 
                           if k == target_key and abs(l - lf) < 0.01]
                if matches:
                    p2bht_insert_throughputs[i] = p2bht_raw["insert_32"][matches[-1]]
    except Exception as e:
        print(f"Warning: Failed to extract P2BHT insert throughputs: {e}")

    # Extract IHT insert throughputs
    iht_insert_throughputs = [0]*5
    try:
        iht_cols = ["num_keys", "load_factor", "insert_32_20"]
        target_key = float(1 << insert_table_size)
        
        for i, lf in enumerate(load_factors):
            lf_str = str(int(lf * 100))
            iht_file = os.path.join(p2bht_iht_result_dir, f"rates_fixed_lf/iht_rates_lfeq{lf_str}.csv")
            if os.path.exists(iht_file):
                iht_raw = read_multi_col(iht_file, iht_cols)
                matches = [idx for idx, (k, l) in enumerate(zip(iht_raw["num_keys"], iht_raw["load_factor"])) 
                           if k == target_key and abs(l - lf) < 0.01]
                if matches:
                    iht_insert_throughputs[i] = iht_raw["insert_32_20"][matches[-1]]
    except Exception as e:
        print(f"Warning: Failed to extract IHT insert throughputs: {e}")

    # Extract DyCuckoo insert throughputs
    dycuckoo_insert_throughputs = [0]*5
    try:
        for i, lf in enumerate(load_factors):
            dy_file = os.path.join(dycuckoo_root, "build", f"static_result_{lf}.csv")
            if os.path.exists(dy_file):
                dy_cols = ["num_keys", "insert_mops"]
                dy_raw = read_multi_col(dy_file, dy_cols)
                target_key = float(1 << insert_table_size)
                try:
                    idx = dy_raw["num_keys"].index(target_key)
                    dycuckoo_insert_throughputs[i] = dy_raw["insert_mops"][idx]
                except ValueError:
                    pass
    except Exception as e:
        print(f"Warning: Failed to extract DyCuckoo insert throughputs: {e}")

    insert_series = [
        { "label" : "Hive Hash Table", "y": hybrid_insert_throughputs, "color": "red", "marker": 'o', "ls": '-'},
        { "label" : "BP2HT", "y": p2bht_insert_throughputs, "color": "blue", "marker": 's', "ls": '-.'},
        { "label" : "IHT", "y": iht_insert_throughputs, "color": "green", "marker": '^', "ls": ':'},
        { "label" : "WarpCore", "y": warpcore_insert_throughputs, "color": "purple", "marker": 'd', "ls": 'solid'},
        { "label" : "DyCuckoo", "y": dycuckoo_insert_throughputs, "color": "cyan", "marker": 'v', "ls": 'dotted'},
        { "label" : "cuCollections", "y": cuco_insert_throughputs, "color": "magenta", "marker": 'p', "ls": '--'},
    ]

    plot_results(load_factors, insert_series, "insert_only_load_factor.png",
                "" ,"Load Factor", "Insert Throughput (M-KV/s)", two_power=False, log_scale=False)

    print("Starting Experiment 4: Query Throughput With Exist Ratio")
    table_size = 24
    exist_keys = [
        "lookup_0_exist_throughput",
        "lookup_25_exist_throughput",
        "lookup_50_exist_throughput",
        "lookup_75_exist_throughput",
        "lookup_100_exist_throughput",
    ]
    hive_lookup_exist_csv = os.path.join(RESULTS_DIR, "lookup_exist.csv")
    if not os.path.isfile(hive_lookup_exist_csv):
        for data_layout in data_layouts:
            common_params["data_layout"] = data_layout
            params = {
                **common_params,
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
        os.makedirs(RESULTS_DIR, exist_ok=True)
        w_csv(results, RESULTS_DIR, "lookup_exist.csv", all_possible_metric_keys)
    else:
        print(f"Reading Hive Hash Table lookup exist ratio results from {hive_lookup_exist_csv}")
        with open(hive_lookup_exist_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in exist_keys:
                    if k in row and row[k]:
                        row[k] = float(row[k])
                results.append(row)

    hybrid_results = [r for r in results if r.get("data_layout") == "HybridSoA-AoS" and "lookup_100_exist_throughput" in r]
    aoas_results = [r for r in results if r.get("data_layout") == "AaoS-LeadMetaData" and "lookup_100_exist_throughput" in r]

    hive_hybrid_all_lookup_exist_throughput = [hybrid_results[0].get(k, 0) for k in exist_keys] if hybrid_results else [0]*5
    hive_aoas_all_lookup_exist_throughput = [aoas_results[0].get(k, 0) for k in exist_keys] if aoas_results else [0]*5

    #bp2ht throughput
    bp2ht_cols = ["num_keys", "find_32_100", "find_32_75", "find_32_50", "find_32_25", "find_32_0"]
    bp2ht_raw = read_multi_col(p2bht_rates_fixed_lf_dir, bp2ht_cols)
    try:
        target_key = float(1 << table_size)
        idx = bp2ht_raw["num_keys"].index(target_key)
        bp2ht_lookup_exist_throughputs = [bp2ht_raw["find_32_0"][idx], bp2ht_raw["find_32_25"][idx], bp2ht_raw["find_32_50"][idx], bp2ht_raw["find_32_75"][idx], bp2ht_raw["find_32_100"][idx]]
    except (ValueError, KeyError, IndexError, TypeError):
        bp2ht_lookup_exist_throughputs = [0]*5

    #iht throughput
    iht_cols = ["num_keys", "find_32_20_100", "find_32_20_75", "find_32_20_50", "find_32_20_25", "find_32_20_0"]
    iht_raw = read_multi_col(iht_rates_fixed_lf_dir, iht_cols)
    try:
        target_key = float(1 << table_size)
        idx = iht_raw["num_keys"].index(target_key)
        iht_lookup_exist_throughputs = [iht_raw["find_32_20_0"][idx], iht_raw["find_32_20_25"][idx], iht_raw["find_32_20_50"][idx], iht_raw["find_32_20_75"][idx], iht_raw["find_32_20_100"][idx]]
    except (ValueError, KeyError, IndexError, TypeError):
        iht_lookup_exist_throughputs = [0]*5

    warpcore_cols = ["sample_size", "query_100_mops", "query_75_mops", "query_50_mops", "query_25_mops", "query_0_mops"]
    warpcore_raw = read_multi_col(warpcore_result_dir, warpcore_cols)
    try:
        target_key = float(1 << table_size)
        idx = warpcore_raw["sample_size"].index(target_key)
        warpcore_lookup_exist_throughputs = [warpcore_raw[k][idx] for k in ["query_0_mops", "query_25_mops", "query_50_mops", "query_75_mops", "query_100_mops"]]
    except (ValueError, KeyError, IndexError, TypeError):
        warpcore_lookup_exist_throughputs = [0]*5

    # DyCuckoo throughput
    dycuckoo_cols = ["num_keys", "query_100_mops", "query_75_mops", "query_50_mops", "query_25_mops", "query_0_mops"]
    dycuckoo_raw = read_multi_col(dycuckoo_result_dir, dycuckoo_cols)
    try:
        target_key = float(1 << table_size)
        idx = dycuckoo_raw["num_keys"].index(target_key)
        dycuckoo_lookup_exist_throughputs = [dycuckoo_raw[k][idx] for k in ["query_0_mops", "query_25_mops", "query_50_mops", "query_75_mops", "query_100_mops"]]
    except (ValueError, KeyError, IndexError, TypeError):
        dycuckoo_lookup_exist_throughputs = [0]*5

    # Cuco Collections throughput
    cuco_lookup_csv = os.path.join(cucollections_root, "build", "results", "cuco_lookup_ratios.csv")

    if not os.path.isfile(cuco_lookup_csv):
        print(f"Results not found at {cuco_lookup_csv}. Running cuCollections Lookup Benchmark...")
        # Note: Ensure run_cucollections_lookup_bench is imported from benchmark_utils
        run_cucollections_lookup_bench(cucollections_root, table_size, common_params["load_factor"])
    else:
        print(f"Found existing data at {cuco_lookup_csv}. Skipping cuCollections lookup execution...")

    cuco_lookup_exist_throughputs = [0] * 5
    if os.path.exists(cuco_lookup_csv):
        # Read the 'lookup_mops' column matching the 5 exist keys (0, 25, 50, 75, 100)
        cuco_lookup_exist_throughputs = read_col(cuco_lookup_csv, "lookup_mops")[:5]

    print("Running SlabHash All Lookup Experiemnt")
    slabhash_lookup_exist_result_dir = os.path.join(slabhash_root, "build/bench_result/query_experiment_varied_exist_ratio.csv")
    slabhash_lookup_exist_throughputs = [0]*5
    if os.path.exists(slabhash_lookup_exist_result_dir):
        sh_vals = read_col(slabhash_lookup_exist_result_dir, "query_rate_mps")
        if sh_vals and len(sh_vals) >= 5:
            slabhash_lookup_exist_throughputs = sh_vals[:5][::-1]

    series = [
        { "label" : "Hive Hash Table", "y": hive_hybrid_all_lookup_exist_throughput, "color": "red", "marker": 'o', "ls": '-'},
        { "label" : "BP2HT", "y": bp2ht_lookup_exist_throughputs, "color": "blue", "marker": 's', "ls": '-.'},
        { "label" : "IHT", "y": iht_lookup_exist_throughputs, "color": "green", "marker": '^', "ls": ':'},
        { "label" : "WarpCore", "y": warpcore_lookup_exist_throughputs, "color": "purple", "marker": 'd', "ls": 'solid'},
        { "label" : "DyCuckoo", "y": dycuckoo_lookup_exist_throughputs, "color": "cyan", "marker": 'v', "ls": 'dotted'},
        { "label" : "cuCollections", "y": cuco_lookup_exist_throughputs, "color": "magenta", "marker": 'p', "ls": '--'},
    ]
    
    plot_results([0.0, 0.25, 0.5, 0.75, 1.0], series, "lookup_exist.png",
                "" ,"Positive Queries (%)", "Query Throughput (M-KV/s)", two_power=False, log_scale=False)

if __name__ == "__main__":
    main()
