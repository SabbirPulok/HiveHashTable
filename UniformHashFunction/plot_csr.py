import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def extract_csr_data(build_dir, target_buckets):
    csv_files = glob.glob(os.path.join(build_dir, "Hash_Function_Study_Chapter_GPU_random*b*k.csv"))
    
    hash_funcs = ['crc64', 'crc32', 'cityhash32', 'murmurhash3', 'hash1', 'hash2']
    
    data = {hf: {} for hf in hash_funcs}
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df_filtered = df[df['nBuckets'] == target_buckets]
            
            if df_filtered.empty:
                continue
                
            n_keys = int(df_filtered['nKeys'].iloc[0])
            
            for hf in hash_funcs:
                row = df_filtered[df_filtered['Hash Function'] == hf]
                if not row.empty:
                    expected_collisions = row['Expected # of collisions'].iloc[0]
                    observed_collisions = row['Bucket Collision Stats_Avg'].iloc[0]
                    
                    if observed_collisions > 0:
                        csr = expected_collisions / observed_collisions
                    else:
                        csr = 1.0
                        
                    data[hf][n_keys] = csr
        except Exception as e:
            print(f"Error parsing {f}: {e}")
                
    return data

def plot_csr(data):
    # Match the specific keys requested
    target_keys = [4096, 8192, 128*128, 256*256, 512*512]
    
    # Filter data to only include target_keys that actually have data
    available_keys = set()
    for hf_data in data.values():
        available_keys.update(hf_data.keys())
        
    x_keys = sorted([k for k in target_keys if k in available_keys])
    
    if not x_keys:
        print("No data found for the specified target keys.")
        return

    # Labels for X-axis matching the original graph's format
    x_labels = []
    for k in x_keys:
        if k == 128*128: x_labels.append("128*128")
        elif k == 256*256: x_labels.append("256*256")
        elif k == 512*512: x_labels.append("512*512")
        elif k == 1024*1024: x_labels.append("1024*1024")
        elif k == 2048*2048: x_labels.append("2048*2048")
        else: x_labels.append(str(k))

    hash_funcs = ['crc64', 'crc32', 'cityhash32', 'murmurhash3', 'hash1', 'hash2']
    labels = ['CRC 64', 'CRC 32', 'City Hash', 'Murmur Hash', 'BitHash1', 'BitHash2']
    
    # Colors visually matched to the original grouped bar chart
    colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853', '#FF6D01', '#46BDC6']

    plt.figure(figsize=(12, 6))
    
    bar_width = 0.12
    x = np.arange(len(x_keys))
    
    # Plot grouped bars
    for i, hf in enumerate(hash_funcs):
        y_values = [data[hf].get(k, 0) for k in x_keys]
        plt.bar(x + i * bar_width, y_values, bar_width, label=labels[i], color=colors[i])

    # Styling to match the original
    plt.xticks(x + bar_width * 2.5, x_labels, fontsize=20)
    plt.yticks([0.00, 0.50, 1.00], ['0.00', '0.50', '1.00'], fontsize=20)

    plt.xlabel('Number of Keys', fontsize=20, fontweight='bold', labelpad=15)
    plt.ylabel('Collision SpeedUp Ratio', fontsize=20, fontweight='bold', labelpad=15)
    
    # Legend at the top, horizontal
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=6, frameon=True, 
               handlelength=1, handleheight=1, fontsize=15)
               
    # Horizontal grid lines (Red like the original)
    plt.grid(axis='y', color='red', linestyle='-', linewidth=0.5, alpha=0.7)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('HashFunctionCSR_recreated.png', dpi=300, bbox_inches='tight')
    print("Plot saved to HashFunctionCSR_recreated.png")

if __name__ == "__main__":
    build_dir = "build/"
    target_buckets = 512*512
    data = extract_csr_data(build_dir, target_buckets)
    plot_csr(data)
