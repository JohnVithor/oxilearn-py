#!/bin/bash

echo rust
time taskset --cpu-list 0 python scripts/run_rust.py

echo python
time taskset --cpu-list 0 python scripts/run_python.py