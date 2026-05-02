import subprocess
import os
import matplotlib.pyplot as plt
import csv

# Define benchmark executable
BENCH_EXE = "./competitors/SlabHash/build/bin/skewed_rmw_bench"

def run_experiment(table_size_power, alpha, num_ops, update_ratio):
    cmd = [BENCH_EXE, str(table_size_power), str(alpha), str(num_ops), str(update_ratio)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Parse output: UPDATE_RATIO: 0.5 TOTAL_MS: 7237.64 INSERT_MS: 7237.45 LOOKUP_MS: 0.1911
    insert_ms = 0.0
    lookup_ms = 0.0
    for line in result.stdout.strip().split('\n'):
        if "UPDATE_RATIO:" in line:
            parts = line.split()
            insert_ms = float(parts[5])
            lookup_ms = float(parts[7])
    return insert_ms, lookup_ms

def main():
    if not os.path.exists(BENCH_EXE):
        print(f"Executable {BENCH_EXE} not found. Please build it first.")
        return
        
    table_size_power = 22
    alpha = 1.5
    num_ops = 10_000_000
    update_ratios = [0.0, 0.1, 0.5, 1.0] # 0%, 10%, 50%, 100%
    
    labels = ["0%", "10%", "50%", "100%"]
    insert_times = []
    lookup_times = []
    
    # Run experiments
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "skewed_rmw_contention.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["update_ratio", "insert_ms", "lookup_ms"])
        for r in update_ratios:
            ins, lkp = run_experiment(table_size_power, alpha, num_ops, r)
            insert_times.append(ins)
            lookup_times.append(lkp)
            writer.writerow([r, ins, lkp])
    
    print(f"Results written to {csv_path}")

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bar_width = 0.5
    indices = range(len(update_ratios))
    
    # Stacked bar chart
    p1 = ax.bar(indices, insert_times, bar_width, label='Insert Latency', color='blue')
    p2 = ax.bar(indices, lookup_times, bar_width, bottom=insert_times, label='Lookup Latency', color='red')
    
    ax.set_ylabel('Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Percentage of RMW Operations', fontsize=14, fontweight='bold')
    ax.set_title(f'SlabHash Skewed RMW Contention Cliff\n(Zipf Alpha={alpha}, Ops={num_ops/1e6}M)', fontsize=16, fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = "results/skewed_rmw_contention_cliff.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
