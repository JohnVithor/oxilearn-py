#!/bin/bash
target=$1
time python scripts/run_rust_$target.py
time python scripts/run_python_$target.py