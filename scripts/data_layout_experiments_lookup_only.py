from math import log2
import subprocess
import os
import csv
import datetime
import re
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from benchmark_utils import run_benchmark as run
from benchmark_utils import run_ncu_profile as run_ncu
from benchmark_utils import write_results_to_csv as w_csv
from benchmark_utils import read_results_from_csv as r_csv
from concurrent_phase_workload_experiment import parse_output


#define the path of benchmark executable
BENCHMARK_EXECUTABLE = "./bin/hive_hash_table_benchmark"

#results directory
RESULTS_DIR = "./results"

number_of_ops = [20, 21, 22, 23, 24, 25, 26] #2^n
number_of_iterations = 100
load_factors = [0.9]
insert_ratio = 0.0
query_ratio = 1.0
delete_ratio = 0.0


#Measure peformance for diferent data layouts
data_layouts = ["HybridSoA-AoS", "AaoS", "AaoS-LeadMetaData"]

base_all_metric_keys = [
    "device_name", "table_size", "load_factor", "data_layout", "num_inserts", "num_queries", "num_deletes",
    "total_ops", "num_blocks", "threads_per_block", "avg_time_ms",
    "success_rate", "throughput_mops"
]

def ncu_parse_output(output):
    metrics = {}    
    #L1/TEX Cache Throughput           %         14.53
    l1_cache_throughput = re.search(r"L1/TEX Cache Throughput\s+%\s+([\d.]+)", output)
    if l1_cache_throughput:
        metrics["l1_cache_throughput"] = float(l1_cache_throughput.group(1))

    # L2 Cache Throughput               %         20.39
    l2_cache_throughput = re.search(r"L2 Cache Throughput\s+%\s+([\d.]+)", output)
    if l2_cache_throughput:
        metrics["l2_cache_throughput"] = float(l2_cache_throughput.group(1))

    # Memory Throughput                 %         55.08
    memory_throughput = re.search(r"Memory Throughput\s+%\s+([\d.]+)", output)

    if memory_throughput:
        metrics["memory_throughput"] = float(memory_throughput.group(1))

    # L1/TEX Hit Rate                                  %        32.77
    l1_cache_hit_rate = re.search(r"L1/TEX Hit Rate\s+%\s+([\d.]+)", output)
    if l1_cache_hit_rate:
        metrics["l1_cache_hit_rate"] = float(l1_cache_hit_rate.group(1))

    # L2 Cache Hit Rate                                  %        25.12
    l2_cache_hit_rate = re.search(r"L2 Hit Rate\s+%\s+([\d.]+)", output)
    if l2_cache_hit_rate:
        metrics["l2_cache_hit_rate"] = float(l2_cache_hit_rate.group(1))

    # warp stall cycle reduction =  To reduce the number of cycles waiting on L1TEX data accesses verify the        
    #   memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by        
    #   increasing data locality (coalescing), or by changing the cache configuration.
    
    # On average, each warp of this workload spends 15.9 cycles being stalled waiting for a scoreboard dependency on a L1TEX 
    warp_stall_cycles = re.search(r"each warp of this workload spends\s+([\d.]+)", output)
    if warp_stall_cycles:
        metrics["warp_stall_cycles"] = float(warp_stall_cycles.group(1))
    
    # Eligible warps are the subset of active warps that are ready to issue their next instruction. 
    # Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. 

    # Eligible Warps Per Scheduler        warp         0.41
    eligible_warps_per_scheduler = re.search(r"Eligible Warps Per Scheduler\s+warp\s+([\d.]+)", output)
    if eligible_warps_per_scheduler:
        metrics["eligible_warps_per_scheduler"] = float(eligible_warps_per_scheduler.group(1))

    return metrics

# x = indices of bar
#y[] = values of different bars [eg. y[0] = value for bar 0, y[1] = value for bar 1, ...]

colors = ['red', 'blue','green', 'orange', 'purple']
markers = ['o', 's', 'D', '^', 'v']
linestyles = ['-', '--', '-.', ':', '-..']
hatches = ['/', '\\', 'x', '.', '*']

def plot_multi_bar_chart(x, y, bar_names, fig_name, fig_title, x_label, y_label):
    n_groups = len(x)
    n_bars = len(bar_names)
    x_idx = np.arange(n_groups)

    width = 0.8 / n_bars  # Adjust width based on number of bars

    plt.figure(figsize=(10, 6))

    for i in range(n_bars):
        offsets = x_idx + (i - (n_bars - 1) / 2.0) * width
        bar = plt.bar(offsets, y[i], width, label=bar_names[i], color=colors[i % len(colors)], hatch=hatches[i % len(hatches)])
        plt.bar_label(bar, padding=3, fontsize=10, label_type='edge')

    # plt.yscale('log', base=2)
    
    plt.xticks(x_idx, x) # Set the labels to be the actual x values
    plt.xlabel(x_label, fontsize=16, fontweight='bold')
    plt.ylabel(y_label, fontsize=16, fontweight='bold')
    plt.title(fig_title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, loc='best', frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + "/" + fig_name, dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_chart(data, factors, fig_name, fig_title, rmax=100.0):
    num_vars = len(factors)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start to the end.
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    # Create a polar subplot
    ax = plt.subplot(111, polar=True)

    ax.set_ylim(0, rmax)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(factors, fontsize=12)
    ax.set_yticks(np.arange(0, int(rmax), 10))
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # data is a dict: { 'LayoutName': [val1, val2, ...] }
    for i, (label, values) in enumerate(data.items()):
        y_list = values.copy()
        y_list += y_list[:1] #close the loop

        bar = ax.plot(angles, y_list, linewidth=2, label=label, color=colors[i % len(colors)])
        ax.fill(angles, y_list, color=colors[i % len(colors)], alpha=0.25)
        # Value annotation
        for angle, val in zip(angles[:-1], values):
            ax.text(angle, val + rmax * 0.03, f"{val:.1f}", color='black', fontsize=8, ha='center', va='center')

    plt.title(fig_title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', bbox_to_anchor=(1.1, 1.1), fontsize=12, frameon=False)
    plt.tight_layout() # no overlapping
    plt.savefig(RESULTS_DIR + "/" + fig_name, dpi=300, bbox_inches='tight')
    plt.close()

def extract_xy_single_map(filter_col, filter_val, data, x_col, y_col):
    data_filtered = [ row for row in data if row[filter_col] == filter_val] 

    x = sorted({ float(row[x_col]) for row in data_filtered})

    #build a map per layout: total_ops -> layout
    layout_maps = {}
    for layout in data_layouts:
        m = { float(row[x_col]) : float(row.get(y_col, 0.0))for row in data_filtered if row.get('data_layout').strip() == layout.strip() }
        layout_maps[layout] = m
    
    # build y
    y = []
    for layout in data_layouts:
        y_series = [layout_maps[layout].get(xv, 0) for xv in x]
        y.append(y_series)
    return x, y

def filter_data(filter_col, filter_val, data, x_col, map_col):
    # filter_val might need type adjustment or comparison adjustment
    # csv stores as string.
    data_filtered = [ row for row in data if float(row[filter_col]) == float(filter_val)] 

    result = {}
    for row in data_filtered:
        key = row[x_col].strip() # Keep as string for layout name
        result[key] = [float(row[m]) for m in map_col]

    return result


def main ():
    if not os.path.exists(BENCHMARK_EXECUTABLE):
        print("Benchmark executable not found at path:", BENCHMARK_EXECUTABLE)
        return
    
    common_params = {
        "distribution": "uniform"
    }

    additional_metrics = [
        "warp_stall_cycles",
        "eligible_warps_per_scheduler",
        "memory_throughput",
        "l1_cache_hit_rate",
        "l2_cache_hit_rate",
        "l1_cache_throughput",
        "l2_cache_throughput",
    ]

    all_possible_metric_keys = list(base_all_metric_keys) + [m for m in additional_metrics if m not in base_all_metric_keys]

    results_concurrent_mix = []
    results_inserts_only = []
    results_lookups_only = []

    # Performance Throughput
    for load_factor in load_factors:
        for ops in number_of_ops:
            for layout in data_layouts:
                print(f"Running experiments for data layout: {layout}")
                
                #all lookups
                params = {
                    **common_params,
                    "num_iterations": number_of_iterations,
                    "table_size": ops,
                    "load_factor": load_factor,
                    "insert_ratio": 0.0,
                    "lookup_ratio": 1.0,
                    "delete_ratio": 0.0,
                    "data_layout": layout,
                }
                metrics = {}
                output = run(params, BENCHMARK_EXECUTABLE)
                metrics = parse_output(output)
                metrics["data_layout"] = layout.strip()
                ncu_metrics = ncu_parse_output(run_ncu(params, BENCHMARK_EXECUTABLE, "hive_mixed_kernel"))
                for k, v in ncu_metrics.items():
                    metrics[k] = v
                results_lookups_only.append(metrics)

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_lf = float(load_factors[-1]) * 100
    memory_behavior_cols = ['memory_throughput', 'l1_cache_hit_rate', 'l2_cache_hit_rate', 'l1_cache_throughput', 'l2_cache_throughput']
    target_table_size = 1 << number_of_ops[-1]
    target_table_size_in_millions = (float)(target_table_size / (1024 * 1024))

    # Lookups Only
    print("Writing Lookups Only Results and Generating Plots...")
    filename = f"data_layout_experiment_lookups_only_{timestamp}.csv"
    w_csv(results_lookups_only, RESULTS_DIR, filename, all_possible_metric_keys)

    x, y = extract_xy_single_map("load_factor", target_lf, results_lookups_only, 'total_ops', 'throughput_mops')
    fig_name = f"data_layout_experiment_lookups_only_throughput_{timestamp}.png"
    fig_title = f"Data layout Experiment - Lookups Only Throughput (lf={target_lf})"
    x_label = "Number of Operations (millions)"
    y_label = "Throughput (Mops/s)"

    plot_multi_bar_chart(x, y, data_layouts, fig_name, fig_title, x_label, y_label)

    fig_name = f"data_layout_experiment_lookups_only_memory_behavior_{timestamp}.png"
    fig_title = f"Data layout Experiment - Lookups Only Memory Behavior (number of queries = {target_table_size_in_millions} millions)"
    memory_behavior_data_lookups = filter_data("table_size", target_table_size, results_lookups_only, 'data_layout', memory_behavior_cols)
    plot_radar_chart(memory_behavior_data_lookups, memory_behavior_cols, fig_name, fig_title, rmax=100.0)

    
if __name__ == "__main__":
    main()