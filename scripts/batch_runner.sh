#!/bin/bash
target=$1
echo "version,seed,training_steps,reward_mean,reward_std"
for seed in {0..99}
do
time python scripts/run_rust_$target.py $seed
time python scripts/run_python_$target.py $seed
done