#!/bin/bash
target=$1
seed=$2
verbose=$3
echo "version,seed,training_steps,reward_mean,reward_std"
>&2 echo elapsed_real_time_seconds,kernel_cpu_seconds,user_cpu_seconds,cpu_percentage,max_resident_set_size_kb,system_page_size_bytes,major_page_faults,minor_page_faults,involuntary_context_switches,voluntary_context_switches,filesystem_input_reads,filesystem_output_writes,command_name_args
/usr/bin/time -f "%e,%S,%U,%P,%M,%Z,%F,%R,%c,%w,%I,%O,%C" taskset --cpu-list 1 python scripts/run_rust_$target.py $seed false $verbose
/usr/bin/time -f "%e,%S,%U,%P,%M,%Z,%F,%R,%c,%w,%I,%O,%C" taskset --cpu-list 1 python scripts/run_python_$target.py $seed false $verbose
# /usr/bin/time -f "%e,%S,%U,%P,%M,%Z,%F,%R,%c,%w,%I,%O,%C" taskset --cpu-list 1 python scripts/run_jax_$target.py $seed false $verbose