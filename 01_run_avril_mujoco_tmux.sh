#!/bin/bash

# Session name for tmux
SESSION="AVRIL_MUJOCO"

# Start a new tmux session detached
tmux new-session -d -s $SESSION

# Initial core number
core=2

log_dir="/home/ondrejb/results/irl-torch/birl/logs"
mkdir -p "$log_dir"  # Ensure the log directory exists

# List of demos_id values
demos_ids=("hopper_medium" "hopper_medium_expert" "hopper_mixed" "walker2d_medium_expert" "walker2d_medium" "walker_mixed")

# Loop over each demos_id and create a new tmux window with the corresponding command
for demos_id in "${demos_ids[@]}"; do
    # Current date and time for log file naming
    current_time=$(date +"%Y-%m-%d_%H-%M-%S")

    # Log file path
    log_file="$log_dir/${demos_id}_$current_time_log.txt"

    # Command to run, with logging and a read to keep the window open
    command="CUDA_VISIBLE_DEVICES='' taskset -c $core python3 experiments/birl/08b0_avril_mujoco.py --demos_id $demos_id >$log_file 2>&1; echo 'Run completed. Press enter to close.'; read"

    # Create a new tmux window with this command
    tmux new-window -t $SESSION -n "$demos_id" "$command"

    # Increment the core number for the next command
    ((core++))
done

# Attach to the tmux session
tmux attach -t $SESSION
