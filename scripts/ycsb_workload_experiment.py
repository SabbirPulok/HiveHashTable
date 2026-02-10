import datetime
import re
import os
import matplotlib.pyplot as plt
import numpy as np

from benchmark_utils import run_ycsb_workloads as run
from benchmark_utils import write_results_to_csv as w_csv

NUMBER_OF_ITERATIONS = 10
WORKLOAD_TYPES = ['A', 'B', 'C', 'D', 'F']

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

def plot_grouped_bar_chart(x_labels, y_values, bar_names, fig_name, fig_title, x_label, y_label):
    """Generates and saves a grouped bar chart."""
    n_groups = len(x_labels)
    n_bars = len(bar_names)
    y_idx = np.arange(n_groups)
    bar_width = 0.8 / n_bars

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_bars))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    hatches = ['x', '*', '/', '.', '\\']

    for i in range(n_bars):
        offsets = y_idx + (i - (n_bars - 1) / 2.0) * bar_width
        bars = ax.barh(offsets, y_values[i], bar_width, label=bar_names[i], color=colors[i], hatch=hatches[i % len(hatches)], edgecolor='black')
        # Make numeric bar labels bold for better visibility
        ax.bar_label(bars, padding=3, fontsize=15, fmt='%.2f', fontweight='bold')

    ax.set_yticks(y_idx)
    # Make bar (y-tick) names bold
    ax.set_yticklabels(x_labels, fontweight='bold', fontsize=12)
    ax.set_xlabel(y_label, fontsize=18, fontweight='bold')
    ax.set_ylabel(x_label, fontsize=18, fontweight='bold')
    ax.set_title(fig_title, fontsize=18, fontweight='bold')
    ax.legend(fontsize=18, loc='best', frameon=True)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fig_name), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Benchmark executable not found at {BENCHMARK_EXECUTABLE}. Please build the project first.")
        return

    data_layouts = ["HybridSoA-AoS", "AaoS-LeadMetaData"]

    common_params = {
        "num_iterations": NUMBER_OF_ITERATIONS,
        "num_operations_ycsb": N_OPS,
        "data_layout": data_layouts[0]
    }

    results = []

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

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ycsb_workload_experiment_{timestamp}.csv"
    w_csv(results, RESULTS_DIR, filename, METRIC_KEYS)
    print(f"Experiments complete. Results saved to {os.path.join(RESULTS_DIR, filename)}")


    # Plotting the results
    x_labels = [f"Workload {label}" for label in WORKLOAD_TYPES]
    layout1_throughputs = [next((res["throughput"] for res in results if res["workload_type"] == wl and res["data_layout"] == data_layouts[0]), 0) for wl in WORKLOAD_TYPES]
    layout2_throughputs = [next((res["throughput"] for res in results if res["workload_type"] == wl and res["data_layout"] == data_layouts[1]), 0) for wl in WORKLOAD_TYPES]

    # fig_title = f"YCSB Workload Experiment Throughput (nOps={1 << N_OPS})",
    plot_grouped_bar_chart(
        x_labels,
        [layout1_throughputs, layout2_throughputs],
        ["Hive Hash Table (HybridSoA-AoS)", "Hive Hash Table (AoAS)"],
        "ycsb_workload_experiment.png",
        '',
        "Workload Type",
        "Throughput (MOPS)"
    )

if __name__ == "__main__":
    main()
