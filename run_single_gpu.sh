#!/bin/bash
name=sess
num_workers=2
num_gpus=1
GPU_ID=(0)

tmux kill-session -t $name

tmux new-session -s $name -n ps -d bash
for (( i=0; i<$num_workers; i++ ))
do
	tmux new-window -t $name -n worker$i -d bash
done

sleep 1

tmux send-keys -t $name:ps "CUDA_VISIBLE_DEVICES= python main.py --num_workers $num_workers --num_gpus $num_gpus --job_name ps --task_index=0" Enter
for (( i=0; i<$num_workers; i++ ))
do
	ID=$((i % $num_gpus))
	tmux send-keys -t $name:worker$i "CUDA_VISIBLE_DEVICES=${GPU_ID[$ID]} python main.py --num_workers $num_workers --num_gpus $num_gpus --job_name worker --task_index $i" Enter
done

sleep 1

tmux a
