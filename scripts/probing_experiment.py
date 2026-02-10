import datetime
import os
import re
import matplotlib.pyplot as plt
import numpy as np

from benchmark_utils import run_benchmark as run
from benchmark_utils import write_results_to_csv as w_csv


# =========================
# Experiment configuration
# =========================
table_sizes = [18, 19, 20, 21, 22]
hash_policies = [
    'Default2Hash',
    'TripleHash',
    'MurmurCityHash',
    'MurmurCityBitHash',
    'Lookup2Hash',
    'Lookup3Hash'
]

load_factor = 0.9
insert_ratio = 1.0
lookup_ratio = 0.0
delete_ratio = 0.0
num_iterations = 10

BENCHMARK_EXECUTABLE = "./bin/hive_hash_table_benchmark"
RESULTS_DIR = "./results"


labels = [
    'BitHash1 & BitHash2',
    'BitHash1 & BitHash2 & City Hash',
    'City Hash & Murmur Hash',
    'City Hash & Murmur Hash & BitHash1',
    'CRC32 & CRC64',
    'CRC32 & CRC64 & BitHash1'
]

colors = [
    '#1c3f95',  # denim
    '#1e90ff',  # dodger blue
    '#d62728',  # dark red
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#98df8a'   # light green
]

hatches = [
    '////',
    '\\\\',
    'xxxx',
    '++++',
    '....',
    '----'
]


def parse_output(output: str) -> dict:
    """
    Extract throughput from benchmark output.
    Expected line: "Throughput: <number>" somewhere in output.
    """
    metrics = {}
    throughput_match = re.search(r'Throughput:\s*([\d.]+)', output)
    metrics['throughput'] = float(throughput_match.group(1)) if throughput_match else 0.0
    return metrics


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_grouped_bars(keys, data, out_png_path: str):
    """
    keys: list[str] (x-axis tick labels)
    data: np.ndarray shape (len(keys), num_policies)
    """
    x = np.arange(len(keys))
    bar_width = 0.13

    plt.figure(figsize=(10, 6))

    for i in range(data.shape[1]):
        plt.bar(
            x + i * bar_width,
            data[:, i],
            width=bar_width,
            label=labels[i] if i < len(labels) else hash_policies[i],
            color=colors[i] if i < len(colors) else None,
            hatch=hatches[i] if i < len(hatches) else None,
            edgecolor='black'
        )

    # Center ticks under the group of bars
    center_offset = bar_width * (data.shape[1] - 1) / 2.0
    plt.xticks(x + center_offset, keys, fontsize=12)

    plt.xlabel('Number of Keys', fontsize=14, fontweight='bold')
    plt.ylabel('Throughput (Mops/sec)', fontsize=14, fontweight='bold')

    plt.legend(fontsize=10, loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300, bbox_inches='tight')

    plt.close()


def main():
    if not os.path.exists(BENCHMARK_EXECUTABLE):
        print(f"Benchmark executable not found at {BENCHMARK_EXECUTABLE}. Please build the project first.")
        return

    ensure_dir(RESULTS_DIR)

    common_params = {
        'load_factor': load_factor,
        'insert_ratio': insert_ratio,
        'lookup_ratio': lookup_ratio,
        'delete_ratio': delete_ratio,
        'num_iterations': num_iterations
    }

    results_rows = []

    data = np.zeros((len(table_sizes), len(hash_policies)), dtype=np.float64)

    for ti, table_size in enumerate(table_sizes):
        row = {'table_size': table_size}

        for pi, hash_policy in enumerate(hash_policies):
            params = {
                **common_params,
                'table_size': table_size,
                'hash_policy': hash_policy
            }

            output = run(params, BENCHMARK_EXECUTABLE)
            metrics = parse_output(output)
            thr = metrics['throughput']

            col_name = f"{hash_policy}_throughput"
            row[col_name] = thr
            data[ti, pi] = thr

            print(f"[table_size=2^{table_size}] {hash_policy}: {thr:.2f} Mops/sec")

        results_rows.append(row)

    all_metrics = ['table_size'] + [f"{hp}_throughput" for hp in hash_policies]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"hash_policy_sweep_{timestamp}.csv"
    w_csv(results_rows, RESULTS_DIR, csv_name, all_metrics)
    print(f"\nSaved CSV: {os.path.join(RESULTS_DIR, csv_name)}")

    keys = [f"2^{ts}" for ts in table_sizes]

    png_name = f"hash_function_comparison_{timestamp}.png"
    png_path = os.path.join(RESULTS_DIR, png_name)

    plot_grouped_bars(keys, data, png_path)
    print(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
