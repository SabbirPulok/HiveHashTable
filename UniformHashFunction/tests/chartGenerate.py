import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# List of CSV files (modify the pattern if needed)
csv_files = glob.glob("../build/Hash_Function_Study_Chapter_GPU_random1024b*")  # Matches all relevant files

# Ensure we have files to process
if not csv_files:
    raise FileNotFoundError("No matching CSV files found.")

# Create output folder if it doesn't exist
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

# Define the common filename prefix
output_filename = os.path.join(output_folder, "Hash_Function_Study_Chapter_GPU_random1024b.png")

# Process each file
for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    # Ensure required columns exist
    required_columns = {"nKeys", "Expected # of Empty Buckets", "Empty Bucket Stats_Avg", "Hash Function"}
    if not required_columns.issubset(df.columns):
        print(f"Skipping {csv_file}: Missing required columns.")
        continue

    # Sort data by nKeys for correct plotting
    df = df.sort_values(by="nKeys")

    # Extract unique hash functions
    hash_functions = df["Hash Function"].unique()

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot expected empty buckets (single line)
    plt.plot(df["nKeys"], df["Expected # of Empty Buckets"], label="Expected # of Empty Buckets", linestyle="dashed", color="black")

    # Plot Empty Bucket Stats_Avg for each hash function
    for func in hash_functions:
        subset = df[df["Hash Function"] == func]
        plt.plot(subset["nKeys"], subset["Empty Bucket Stats_Avg"], marker='o', label=func)

# Labels and title
plt.xlabel("nKeys")
plt.ylabel("Number of Empty Buckets")
plt.title(f"Empty Bucket Statistics 1024 buckets")
plt.legend()
plt.grid(True)


# Save the plot
plt.savefig(output_filename, dpi=300)
plt.close()

print(f"Plot saved as {output_filename}")
