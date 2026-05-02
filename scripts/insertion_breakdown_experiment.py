import datetime
import os
import re
import matplotlib.pyplot as plt
import numpy as np

from benchmark_utils import run_benchmark_insert_breakdown as run
from benchmark_utils import write_results_to_csv as w_csv

BENCHMARK_EXECUTABLE = "./bin/hive_hash_table_benchmark_breakdown_insert"
RESULTS_DIR = "./results"
NUMBER_OF_ITERATIONS = 1

load_factors = [0.50, 0.75, 0.90, 0.95, 1.00]
stage_labels = ['Step A', 'Step B', 'Step C', 'Step D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red
hatches = ['////', '\\\\', '....', 'xx']

# Insert Breakdown (in ms):
# Stage A (Try Replace Path): 47028.40 ms, Percentage of total: 24.88%
# Stage B (Claim And Commit Path): 69247.05 ms, Percentage of total: 36.63%
# Stage C (Cuckoo Eviction Path): 68638.00 ms, Percentage of total: 36.31%
# Stage D (Stash Path): 4121.63 ms, Percentage of total: 2.18%

def parse_output(output):
    metrics = {}
    if output:
        # Extract breakdown times
        breakdown_match = re.search(r'Insert Breakdown \(in ms\):\s+Stage A \(Try Replace Path\): ([\d.]+) ms, Percentage of total: ([\d.]+)%\s+Stage B \(Claim And Commit Path\): ([\d.]+) ms, Percentage of total: ([\d.]+)%\s+Stage C \(Cuckoo Eviction Path\): ([\d.]+) ms, Percentage of total: ([\d.]+)%\s+Stage D \(Stash Path\): ([\d.]+) ms, Percentage of total: ([\d.]+)%', output, flags=re.DOTALL)
        if breakdown_match:
            metrics["stage_a_time"] = float(breakdown_match.group(1))
            metrics["stage_a_percent"] = float(breakdown_match.group(2))
            metrics["stage_b_time"] = float(breakdown_match.group(3))
            metrics["stage_b_percent"] = float(breakdown_match.group(4))
            metrics["stage_c_time"] = float(breakdown_match.group(5))
            metrics["stage_c_percent"] = float(breakdown_match.group(6))
            metrics["stage_d_time"] = float(breakdown_match.group(7))
            metrics["stage_d_percent"] = float(breakdown_match.group(8))
    return metrics

def plot_stack_bars(load_factors, percent_data):
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(load_factors))
    bar_width = 0.6
    bottom = np.zeros(len(load_factors))

    # Stacked bars
    for i in range(len(stage_labels)):
        ax.bar(
            x, percent_data[:, i], bottom=bottom, width=bar_width,
            label=stage_labels[i], color=colors[i],
            edgecolor='black', hatch=hatches[i]
        )
        bottom += percent_data[:, i]

    # Add text labels for Stage D specifically since it's small
    for i in range(len(load_factors)):
        val = percent_data[i, 3]
        if val > 0:
            ax.text(x[i], 101, f'{val:.2f}%', ha='center', va='bottom', 
                    fontsize=10, color=colors[3], fontweight='bold')

    # Axis labels & ticks
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lf:.2f}' for lf in load_factors], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlabel('Load Factor', fontsize=16, fontweight='bold')
    ax.set_ylabel('Steps Contribution in Total Elapsed Time (%)', fontsize=16, fontweight='bold')

    # Grid and title
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.7)
    ax.set_ylim(0, 115) # Increased to give space for labels
    
    # Inset for Stage D Detail
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="30%", height="25%", loc='center right', borderpad=3)
    
    # Add a white background to the inset for better readability over the main bars
    axins.patch.set_facecolor('white')
    axins.patch.set_alpha(0.9)
    
    axins.bar(x, percent_data[:, 3], color=colors[3], hatch=hatches[3], edgecolor='black')
    axins.set_xticks(x)
    axins.set_xticklabels([f'{lf:.2f}' for lf in load_factors], fontsize=14, color='black')
    axins.tick_params(axis='y', labelsize=14, labelcolor='black')
    axins.set_title('Stage D (Stash) Detail (%)', fontsize=14, fontweight='bold', color='black')
    axins.grid(axis='y', linestyle='--', alpha=0.5)

    ax.legend(
        fontsize=10, ncol=4, frameon=False,
        loc='upper center', bbox_to_anchor=(0.5, 0.99)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(os.path.join(RESULTS_DIR, 'insertion_stage_breakdown.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    if not os.path.exists(BENCHMARK_EXECUTABLE):
        print(f"Benchmark executable not found at {BENCHMARK_EXECUTABLE}. Please build the project first.")
        return
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    data_layouts = ["HybridSoA-AoS"]

    metrics = {
        "load_factor",
        "stage_a_time",
        "stage_a_percent",
        "stage_b_time",
        "stage_b_percent",
        "stage_c_time",
        "stage_c_percent",
        "stage_d_time",
        "stage_d_percent"
    }
    common_params = {
        "data_layout": data_layouts[0],
        "insert_ratio": 1.0,
        "lookup_ratio": 0.0,
        "delete_ratio": 0.0,
        "table_size": 23,
    }

    print("Starting Insertion Breakdown Experiment...")
    all_results = []

    for load_factor in load_factors:
        params = {
            **common_params,
            "load_factor": load_factor  + 0.6,
            "distribution": "uniform",
        }
        output = run(params, executable_path=BENCHMARK_EXECUTABLE)
        parsed_metrics = parse_output(output)
        all_results.append(parsed_metrics)
    
    #save results to csv
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"insertion_breakdown_experiment_{timestamp}.csv"
    w_csv(all_results, RESULTS_DIR, filename, metrics)
    print(f"Experiment complete. Results saved to {os.path.join(RESULTS_DIR, filename)}")
    plot_stack_bars(
        load_factors,
        np.array([
            [res["stage_a_percent"], res["stage_b_percent"], res["stage_c_percent"], res["stage_d_percent"]]
            for res in all_results
        ])
    )


if __name__ == "__main__":
    main()
















