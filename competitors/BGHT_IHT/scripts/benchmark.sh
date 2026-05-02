#!/bin/bash

# pick a suitable device id
device=0

min_keys=$((2**22))
max_keys=$((2**27))
step=$((2))

# We will let benchmark_utils.py pass the load factors as arguments.
# If no arguments are passed, default to 0.9.
if [ $# -eq 0 ]; then
    load_factors=(0.9)
else
    load_factors=("$@")
fi

for lf in "${load_factors[@]}"; do
    echo "Running BGHT/IHT for load factor $lf"
    ./bin/rates_per_technique_fixed_lf --validate=false --min-keys=$min_keys --max-keys=$max_keys --step=$step --device=$device --load-factor2=$lf
done
