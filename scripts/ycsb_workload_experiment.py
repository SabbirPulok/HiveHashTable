import datetime
import re
import os
import matplotlib.pyplot as plt
import numpy as np

from benchmark_utils import run_ycsb_workloads as run
from benchmark_utils import write_results_to_csv as w_csv

NUMBER_OF_ITERATIONS = 10
WORKLOAD_TYPES = ['A', 'B', 'C', 'D']

#Define the path of benchmark executable
BENCHMARK_EXECUTABLE = "./bin/hive_hash_table_ycsb"

 #Results directory
RESULTS_DIR = "./results"

N_OPS = 1 << 23

METRIC_KEYS = [
    "device_name",
    "data_layout",
    "workload_type",
    "throughput"
]

def parse_output(output):
    metrics = {}
    if output:
        # Device Name: NVIDIA GeForce RTX 2080 Ti
        device_match = re.search(r"Device Name: (.+)", output)
        if device_match:
            metrics["device_name"] = device_match.group(1).strip()
        
        throughput_match = re.search(r"Throughput: ([\d.]+)", output, flags=re.DOTALL)
        if throughput_match:
            metrics["throughput"] = float(throughput_match.group(1))
        print("Throughput: ", metrics.get("throughput", "N/A"))
    return metrics

def plot_ycsb_stacked_bars(workloads, throughputs, fig_name):
    from matplotlib.patches import Patch
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Composition mapping: (lookup_pct, update_pct, insert_pct)
    comp = {
        'A': (0.50, 0.50, 0.00),
        'B': (0.95, 0.05, 0.00),
        'C': (1.00, 0.00, 0.00),
        'D': (0.95, 0.00, 0.05)
    }

    colors = {'Lookup': '#1f77b4', 'Update': '#d62728', 'Insert': '#8c564b'} # Blue, Red, Brown

    x = np.arange(len(workloads))
    bar_width = 0.5

    for i, wl in enumerate(workloads):
        wl_letter = wl.replace("Workload ", "").strip()
        total_thrp = throughputs[i]
        l_pct, u_pct, i_pct = comp.get(wl_letter, (0,0,0))
        
        bottom = 0
        if l_pct > 0:
            ax.bar(x[i], total_thrp * l_pct, bar_width, bottom=bottom, color=colors['Lookup'], edgecolor='black')
            bottom += total_thrp * l_pct
        if u_pct > 0:
            ax.bar(x[i], total_thrp * u_pct, bar_width, bottom=bottom, color=colors['Update'], edgecolor='black')
            bottom += total_thrp * u_pct
        if i_pct > 0:
            ax.bar(x[i], total_thrp * i_pct, bar_width, bottom=bottom, color=colors['Insert'], edgecolor='black')
            bottom += total_thrp * i_pct
        
        # Label total on top
        ax.text(x[i], total_thrp + max(throughputs)*0.02, f"{total_thrp:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontweight='bold', fontsize=20)
    ax.set_ylabel("Throughput (M-KV/s)", fontsize=20, fontweight='bold')
    
    # Custom legend at top of figure
    legend_elements = [
        Patch(facecolor=colors['Update'], edgecolor='black', label='Update'),
        Patch(facecolor=colors['Insert'], edgecolor='black', label='Insert'),
        Patch(facecolor=colors['Lookup'], edgecolor='black', label='Lookup')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=20, frameon=False)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Remove x-axis grid to look cleaner with vertical bars
    ax.grid(axis='x', visible=False)
    ax.set_ylim(0, max(throughputs) * 1.25)
    
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fig_name), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(BENCHMARK_EXECUTABLE):
        print(f"Benchmark executable not found at {BENCHMARK_EXECUTABLE}. Please build the project first.")
        return

    data_layouts = ["HybridSoA-AoS"]

    common_params = {
        "num_iterations": NUMBER_OF_ITERATIONS,
        "num_operations_ycsb": N_OPS,
        "data_layout": data_layouts[0]
    }

    results = []
    csv_filename = "ycsb_workload_experiment.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)

    if not os.path.isfile(csv_path):
        print("Benchmarking with YCSB Workload...")
        for workload in WORKLOAD_TYPES:
            print(f"Running experiments for Workload Type: {workload}")
            params = {
                **common_params,
                "ycsb_workload_type": workload
            }
            for layout in data_layouts:
                params["data_layout"] = layout
                output = run(params, BENCHMARK_EXECUTABLE)
                metrics = parse_output(output)
                metrics["workload_type"] = workload
                metrics["data_layout"] = layout
                results.append(metrics)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        w_csv(results, RESULTS_DIR, csv_filename, METRIC_KEYS)
        print(f"Experiments complete. Results saved to {csv_path}")
    else:
        print(f"Reading existing YCSB results from {csv_path}")
        import csv as csv_lib
        with open(csv_path, mode='r') as f:
            reader = csv_lib.DictReader(f)
            for row in reader:
                if "throughput" in row:
                    row["throughput"] = float(row["throughput"])
                results.append(row)


    # Plotting the results                  
    x_labels = [f"Workload {label}" for label in WORKLOAD_TYPES]
    layout1_throughputs = [next((res["throughput"] for res in results if res["workload_type"] == wl and res["data_layout"] == data_layouts[0]), 0) for wl in WORKLOAD_TYPES]

    plot_ycsb_stacked_bars(
        x_labels,
        layout1_throughputs,
        "ycsb_workload_experiment.png"
    )

if __name__ == "__main__":
    main()
