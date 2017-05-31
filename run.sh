#!/bin/bash
tmux kill-session -t examplesess
tmux new-session -s examplesess -n ps-0 -d bash
tmux new-window -t examplesess -n ps-1 -d bash
tmux new-window -t examplesess -n worker-0 -d bash
tmux new-window -t examplesess -n worker-1 -d bash
sleep 1
tmux send-keys -t examplesess:ps-0 'CUDA_VISIBLE_DEVICES= python example.py --job_name ps --task_index 0' Enter
tmux send-keys -t examplesess:ps-1 'CUDA_VISIBLE_DEVICES= python example.py --job_name ps --task_index 1' Enter
tmux send-keys -t examplesess:worker-0 'CUDA_VISIBLE_DEVICES=0 python example.py --job_name worker --task_index 0' Enter
tmux send-keys -t examplesess:worker-1 'CUDA_VISIBLE_DEVICES=0 python example.py --job_name worker --task_index 1' Enter
sleep 1
tmux a
