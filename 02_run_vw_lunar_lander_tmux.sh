#!/bin/bash

# Session name for tmux
SESSION="VW_LUNAR_PAPER"

# Start a new tmux session detached
tmux new-session -d -s $SESSION

# Initial core number
core=30

log_dir="/home/ondrejb/results/irl-torch/birl/logs"
mkdir -p "$log_dir"  # Ensure the log directory exists

splits=(0 1 2 3 4)
reps=(0 1)
num_trajs=(1 3 7)


# Loop over each demos_id and create a new tmux window with the corresponding command
for split in "${splits[@]}"; do
    for rep in "${reps[@]}"; do
      for traj_num in "${num_trajs[@]}"; do
        # Current date and time for log file naming
        current_time=$(date +"%Y-%m-%d_%H-%M-%S")

        # Log file path
        log_file="$log_dir/vw_lunar.hyper.split$split.c$core.$current_time.log.txt"

        # Command to run, with logging and a read to keep the window open
        command="CUDA_VISIBLE_DEVICES='' taskset -c $core python3 experiments/birl/06b_vw_lunar_lander.py --num_trajs $traj_num --split $split --core_id $core; echo 'Run completed. Press enter to close.'; read"
#        command="CUDA_VISIBLE_DEVICES='' taskset -c $core python3 experiments/birl/06b_vw_lunar_lander.py --split $split --core_id $core; echo 'Run completed. Press enter to close.'; read"

       # Run ID for the window name
        run_id="traj$traj_num.split$split.rep$rep"
        # Create a new tmux window with this command
        tmux new-window -t $SESSION -n "$run_id" "$command"

        # Increment the core number for the next command
        ((core++))
      done
    done
done

# Attach to the tmux session
tmux attach -t $SESSION
