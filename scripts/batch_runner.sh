#!/bin/bash
target=$1
echo "version,seed,training_steps,reward_mean,reward_std"
>&2 echo elapsed_real_time_seconds,kernel_cpu_seconds,user_cpu_seconds,cpu_percentage,max_resident_set_size_kb,avg_resident_set_size_kb,avg_total_memory_use_kb,avg_unshared_data_area_kb,avg_unshared_stack_size_kb,avg_shared_text_space_kb,system_page_size_bytes,page_faults,major_page_faults,minor_page_faults,out_swaps,involuntary_context_switches,voluntary_context_switches,filesystem_input_reads,filesystem_output_writes,received_signals,command_name_args
for seed in {0..99}
do
/usr/bin/time -f "%e,%S,%U,%P,%M,%t,%K,%D,$p,%X,%Z,%F,%R,%W,%c,%w,%I,%O,%k,%C" python scripts/run_rust_$target.py $seed
/usr/bin/time -f "%e,%S,%U,%P,%M,%t,%K,%D,$p,%X,%Z,%F,%R,%W,%c,%w,%I,%O,%k,%C" python scripts/run_python_$target.py $seed
done
