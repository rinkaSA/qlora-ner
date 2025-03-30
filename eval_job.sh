#!/bin/bash
#SBATCH --job-name=qlora_evaluation
#SBATCH --nodes=1
#SBATCH --output=logs/llm_evaluation_%j.out
#SBATCH --error=logs/llm_evaluation_%j.err
#SBATCH --partition=alpha            
#SBATCH --gres=gpu:1          
#SBATCH --time=03:00:00            
#SBATCH --mem=32G                 
#SBATCH --cpus-per-task=4     

module load CUDA/11.7.0

source /data/horse/ws/irve354e-uniNer_test/qlora-ner/qlora-ner/.env

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,temperature.gpu --format=csv -l 5 > output/gpu_usage_eval.log &
gpu_log_pid=$!

python src/eval_llm_batches.py

kill $gpu_log_pid
