#!/bin/bash

echo "version,seed,training_steps,reward_mean,reward_std"
for seed in {0..99}
do
time python scripts/run_rust.py $seed
time python scripts/run_python.py $seed
done