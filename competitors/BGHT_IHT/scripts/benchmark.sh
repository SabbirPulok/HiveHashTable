#!/bin/bash

# pick a suitable device id
device=0

# probing counts per scheme
# ./bin/probes_per_technique --validate=false --num-keys=10000000 --device=$device
# rates for fixed number of keys
# ./bin/rates_per_technique --validate=false --num-keys=10000000 --device=$device
# rates for fixed load factor
min_keys=$((2**22))
max_keys=$((2**27))
step=$((2))
./bin/rates_per_technique_fixed_lf --validate=false --min-keys=$min_keys --max-keys=$max_keys --step=$step --device=$device