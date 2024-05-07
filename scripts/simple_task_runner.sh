#!/bin/bash
target=$1
seed=$2
# time taskset --cpu-list 0 python scripts/run_rust_$target.py $seed false 1
time taskset --cpu-list 0 python scripts/run_python_$target.py $seed false 1
# time taskset --cpu-list 0 python scripts/run_python_jax_$target.py $seed false 1