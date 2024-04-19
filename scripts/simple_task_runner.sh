#!/bin/bash
time taskset --cpu-list 0 python scripts/run_rust.py
time taskset --cpu-list 0 python scripts/run_python.py