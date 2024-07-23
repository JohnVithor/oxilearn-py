#!/bin/bash
target=$1
seed=$2
/usr/bin/time -f "%e,%S,%U,%P,%M,%Z,%F,%R,%c,%w,%I,%O,%C" python scripts/run_rust_$target.py $seed False 1
/usr/bin/time -f "%e,%S,%U,%P,%M,%Z,%F,%R,%c,%w,%I,%O,%C" python scripts/run_python_$target.py $seed False 1
