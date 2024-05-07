#!/bin/bash
target=$1
seed=$2
time python scripts/run_rust_$target.py $seed False 1
time python scripts/run_python_$target.py $seed False 1
