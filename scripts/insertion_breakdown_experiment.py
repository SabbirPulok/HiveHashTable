import datetime
import os
import re
import matplotlib.pyplot as plt
import numpy as np

from benchmark_utils import run_benchmark_insert_breakdown as run
from benchmark_utils import write_results_to_csv as w_csv

BENCHMARK_EXECUTABLE = "./bin/hive_hash_table_benchmark_breakdown_insert"
RESULTS_DIR = "./results"
NUMBER_OF_ITERATIONS = 10

load_factors = [0.40, 0.55, 0.70, 0.85, 0.93]
stage_labels = ['Step A', 'Step B', 'Step C', 'Step D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red
hatches = ['////', '\\\\', '....', 'xx']
obtained_lf = []
# Insert Breakdown (in ms):
# Stage A (Try Replace Path): 47028.40 ms, Percentage of total: 24.88%
# Stage B (Claim And Commit Path): 69247.05 ms, Percentage of total: 36.63%
# Stage C (Cuckoo Eviction Path): 68638.00 ms, Percentage of total: 36.31%
# Stage D (Stash Path): 4121.63 ms, Percentage of total: 2.18%

def parse_output(output):
    metrics = {}
    if output:
        # Iteration 9 Load Factor: 98.10%
        last_itr = NUMBER_OF_ITERATIONS - 1
        load_factor_match = re.search(r'Iteration ' + str(last_itr) + r' Load Factor:\s+([\d\.]+)%', output)
        if load_factor_match:
            metrics["load_factor"] = float(load_factor_match.group(1)) / 100.0
            obtained_lf.append(metrics["load_factor"])

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
    fig, ax = plt.subplots(figsize=(9, 6))
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

    # Axis labels & ticks
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lf:.2f}' for lf in load_factors], fontsize=12)
    ax.set_xlabel('Load Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Steps Contribution in Total Elapsed Time (%)', fontsize=14, fontweight='bold')

    # Grid and title
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.7)
    ax.set_ylim(0, 110)
    #plt.title('Insertion Stage Breakdown vs. Load Factor', fontsize=15, fontweight='bold')


    ax.legend(
        fontsize=10, ncol=4, frameon=False,
        loc='upper center', bbox_to_anchor=(0.5, 0.99)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(os.path.join(RESULTS_DIR, 'insertion_stage_breakdown.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Benchmark executable not found at {BENCHMARK_EXECUTABLE}. Please build the project first.")
        return
    
    data_layouts = ["HybridSoA-AoS", "AaoS-LeadMetaData"]

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
        "table_size": 24,
    }

    print("Starting Insertion Breakdown Experiment...")
    all_results = []

    for load_factor in load_factors:
        params = {
            **common_params,
            "load_factor": load_factor,
            "distribution": "unique",
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
        obtained_lf,
        np.array([
            [res["stage_a_percent"], res["stage_b_percent"], res["stage_c_percent"], res["stage_d_percent"]]
            for res in all_results
        ])
    )


if __name__ == "__main__":
    main()
















