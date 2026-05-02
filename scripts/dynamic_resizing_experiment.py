import argparse
import datetime
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from benchmark_utils import run_dynamic_resize as run
from benchmark_utils import run_dycuckoo_dynamic_resize as run_dycuckoo
from benchmark_utils import run_slabhash_dynamic_resizing_bench as run_slabhash
from benchmark_utils import write_results_to_csv as w_csv
from benchmark_utils import read_results_from_col as read_col
import subprocess

BENCHMARK_EXECUTABLE = "./bin/hive_hash_table_benchmark"
RESULTS_DIR = "./results"
DYCUCKOO_DIR = "./competitors/DyCuckoo/dynamicHash/"
SLABHASH_DIR = "./competitors/SlabHash/"

# Configuration for the experiments
NUMBER_OF_OPS_POWERS = [21, 22, 23, 24]
NUMBER_OF_ITERATIONS = 10
GROW_LOAD_FACTOR = 1.5
SHRINK_LOAD_FACTOR = 0.15

METRIC_KEYS = [
    "num_ops_power",
    "device_name",
    # "load_factor",
    "buckets_added",
    "kv_slots_added",
    "hive_hash_grow_time",
    "hive_hash_grow_throughput",
    "buckets_merged",
    "kv_slots_removed",
    "hive_hash_shrink_time",
    "hive_hash_shrink_throughput"
]

def parse_output(output: str) -> dict:
    """Parses the output of the benchmark executable to extract metrics."""
    metrics = {}
    if not output: return metrics

    device_name = re.search(r"Device Name:\s+(.+)", output)
    if device_name:
        metrics["device_name"] = device_name.group(1).strip()

    buckets_and_slots_added = re.search(r"Number of New Buckets Added:\s+(\d+), Number of KV Slots Included:\s+(\d+)", output)
    if buckets_and_slots_added:
        metrics["buckets_added"] = int(buckets_and_slots_added.group(1))
        metrics["kv_slots_added"] = int(buckets_and_slots_added.group(2))

    hive_hash_grow_time = re.search(r"Hive Hash Grow Time:\s+([\d\.]+)\s+ms, Throughput:\s+([\d\.]+)\s+Mops/s", output)
    if hive_hash_grow_time:
        metrics["hive_hash_grow_time"] = float(hive_hash_grow_time.group(1))
        metrics["hive_hash_grow_throughput"] = float(hive_hash_grow_time.group(2))

    buckets_merged = re.search(r"Number of Buckets Merged:\s+(\d+), Number of KV Slots Removed:\s+(\d+)", output)
    if buckets_merged:
        metrics["buckets_merged"] = int(buckets_merged.group(1))
        metrics["kv_slots_removed"] = int(buckets_merged.group(2))

    hive_hash_shrink_time = re.search(r"Hive Hash Shrink Time:\s+([\d\.]+)\s+ms, throughput:\s+([\d\.]+)\s+MLOPS", output)
    if hive_hash_shrink_time:
        metrics["hive_hash_shrink_time"] = float(hive_hash_shrink_time.group(1))
        metrics["hive_hash_shrink_throughput"] = float(hive_hash_shrink_time.group(2))

    return metrics

def plot_grouped_bar_chart(x_labels, y_values, bar_names, fig_name, fig_title, x_label, y_label, colors, hatches):
    """Generates and saves a grouped bar chart."""
    n_groups = len(x_labels)
    n_bars = len(bar_names)
    x_idx = np.arange(n_groups)
    bar_width = 0.8 / n_bars

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for i in range(n_bars):
        offsets = x_idx + (i - (n_bars - 1) / 2.0) * bar_width
        bars = ax.bar(offsets, y_values[i], bar_width, label=bar_names[i], color=colors[i], hatch=hatches[i % len(hatches)], edgecolor='black')
        ax.bar_label(bars, padding=3, fontsize=14, fontweight='bold', fmt='%.2f')

    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_label, fontsize=18, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=18, fontweight='bold')
    ax.set_title(fig_title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=18, loc='best', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fig_name), dpi=300, bbox_inches='tight')
    plt.close()

def run_experiment(compute_capability=None):
    """Runs the dynamic resizing benchmark experiments and plots the results."""
    if not os.path.exists(BENCHMARK_EXECUTABLE):
        print(f"Benchmark executable not found at path: {BENCHMARK_EXECUTABLE}")
        return
        
    os.makedirs(RESULTS_DIR, exist_ok=True)

    common_params = {
        "distribution": "uniform",
        "data_layout": "HybridSoA-AoS",
        "num_iterations": NUMBER_OF_ITERATIONS,
    }

    # run dycuckoo dynamic resizing benchmark
    print('Running DyCuckoo Benchmark...')
    dycuckoo_root = os.path.abspath(DYCUCKOO_DIR)
    dycuckoo_result_dir = run_dycuckoo(dycuckoo_root, 16, 24, 0.25, 0.9, 0.5, compute_capability=compute_capability)
    
    dycuckoo_grow_throughputs = read_col(dycuckoo_result_dir, "grow_throughput")[:len(NUMBER_OF_OPS_POWERS)]
    dycuckoo_shrink_throughputs = read_col(dycuckoo_result_dir, "shrink_throughput")[:len(NUMBER_OF_OPS_POWERS)]
    
    # Fill with 0s if benchmark failed
    if not dycuckoo_grow_throughputs: dycuckoo_grow_throughputs = [0] * len(NUMBER_OF_OPS_POWERS)
    if not dycuckoo_shrink_throughputs: dycuckoo_shrink_throughputs = [0] * len(NUMBER_OF_OPS_POWERS)

    print('Running SlabHash Benchmark...')
    slabhash_root = os.path.abspath(SLABHASH_DIR)
    
    slabhash_results = run_slabhash(slabhash_root)
    if slabhash_results:
        slabhash_expansion_result, slabhash_contraction_result = slabhash_results
        slabhash_grow_throughputs = read_col(slabhash_expansion_result, "rehash_throughput_Mops")
        slabhash_shrink_throughputs = read_col(slabhash_contraction_result, "merge_throughput_Mops")
    else:
        slabhash_grow_throughputs = [0] * len(NUMBER_OF_OPS_POWERS)
        slabhash_shrink_throughputs = [0] * len(NUMBER_OF_OPS_POWERS)

    if not slabhash_grow_throughputs: slabhash_grow_throughputs = [0] * len(NUMBER_OF_OPS_POWERS)
    if not slabhash_shrink_throughputs: slabhash_shrink_throughputs = [0] * len(NUMBER_OF_OPS_POWERS)

    all_results = []

    for ops_power in NUMBER_OF_OPS_POWERS:
        combined_metrics = {"num_ops_power": ops_power}

        # --- Grow Experiment ---
        print(f"Running Dynamic Resizing (Expansion) for num_ops=2^{ops_power}...")
        grow_params = {
            **common_params,
            "table_size": ops_power,
            "load_factor": GROW_LOAD_FACTOR,
            "insert_ratio": 1.0,
            "lookup_ratio": 0.0,
            "delete_ratio": 0.0,
        }
        try:
            grow_output = run(grow_params, executable_path=BENCHMARK_EXECUTABLE)
            grow_parsed_output = parse_output(grow_output)
            combined_metrics.update(grow_parsed_output)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error running grow benchmark for 2^{ops_power}: {e}")
        
        # --- Shrink Experiment ---
        print(f"Running Dynamic Resizing (Contraction) for num_ops=2^{ops_power}...")
        shrink_params = {
            **common_params,
            "table_size": ops_power,
            "load_factor": SHRINK_LOAD_FACTOR,
            "insert_ratio": 0.1,
            "lookup_ratio": 0.0,
            "delete_ratio": 0.1,
        }
        try:
            shrink_output = run(shrink_params, executable_path=BENCHMARK_EXECUTABLE)
            shrink_parsed_output = parse_output(shrink_output)
            combined_metrics["buckets_merged"] = shrink_parsed_output.get("buckets_merged", 0)
            combined_metrics["kv_slots_removed"] = shrink_parsed_output.get("kv_slots_removed", 0)
            combined_metrics["hive_hash_shrink_time"] = shrink_parsed_output.get("hive_hash_shrink_time", 0)
            combined_metrics["hive_hash_shrink_throughput"] = shrink_parsed_output.get("hive_hash_shrink_throughput", 0)
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error running shrink benchmark for 2^{ops_power}: {e}")
            
        all_results.append(combined_metrics)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dynamic_resizing_experiment_{timestamp}.csv"
    w_csv(all_results, RESULTS_DIR, filename, METRIC_KEYS)

    # --- Plotting ---
    x_labels = [f"{int(((float(r.get('buckets_added', 0) + r.get('buckets_merged', 0))) / (2 * 10**3)))}K" for r in all_results]

    # Plot for Time
    grow_times = [r.get("hive_hash_grow_time", 0) for r in all_results]
    shrink_times = [r.get("hive_hash_shrink_time", 0) for r in all_results]
    colors_time = ['orange', 'green']
    hatches_time = ['x', '.']
    plot_grouped_bar_chart(
        x_labels=x_labels,
        y_values=[grow_times, shrink_times],
        bar_names=['Grow Time (ms)', 'Shrink Time (ms)'] ,
        fig_name="hive_hash_dynamic_resizing_time.png",
        colors=colors_time,
        hatches=hatches_time,
        fig_title="",
        x_label="Number of Buckets Added/Removed (Thousands)",
        y_label="Time (ms)"
    )

    # Plot for Throughput
    grow_throughputs = [r.get("hive_hash_grow_throughput", 0) for r in all_results]
    shrink_throughputs = [r.get("hive_hash_shrink_throughput", 0) for r in all_results]
    bar_names_=['Hive Hash Table', 'SlabHash', 'DyCuckoo']
    colors_tp = plt.get_cmap('Set2')(np.linspace(0, 1, len(bar_names_)))
    hatches_tp = ['/', '\\', 'x', '.', '*']
    
    plot_grouped_bar_chart(
        x_labels=x_labels,
        y_values=[grow_throughputs, slabhash_grow_throughputs, dycuckoo_grow_throughputs],
        bar_names=bar_names_,
        fig_name="dynamic_resizing_grow_throughput.png",
        fig_title="",
        x_label="Number of Buckets Added (Thousands)",
        y_label="Throughput (Mops/s)",
        colors=colors_tp,
        hatches=hatches_tp
    )

    plot_grouped_bar_chart(
        x_labels=x_labels,
        y_values=[shrink_throughputs, slabhash_shrink_throughputs, dycuckoo_shrink_throughputs],
        bar_names=bar_names_,
        fig_name="dynamic_resizing_shrink_throughput.png",
        fig_title="",
        x_label="Number of Buckets Merged (Thousands)",
        y_label="Throughput (Mops/s)",
        colors=colors_tp,
        hatches=hatches_tp
    )
    
    print(f"Experiments complete. Results saved to {RESULTS_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Run Dynamic Resizing Experiment")
    parser.add_argument("--sm", type=str, default=os.environ.get("SM", None),
                        help="Compute capability (e.g., 89, 120). Defaults to the SM environment variable.")
    args = parser.parse_args()
    
    run_experiment(compute_capability=args.sm)

if __name__ == "__main__":
    main()
