#!/bin/bash
#SBATCH --job-name=qlora_training
#SBATCH --nodes=1
#SBATCH --output=logs/qlora_training_enetitytypesprompt%j.out
#SBATCH --error=logs/qlora_training_entitytypesprompt%j.err
#SBATCH --partition=alpha            
#SBATCH --gres=gpu:1          
#SBATCH --time=05:00:00            
#SBATCH --mem=32G                 
#SBATCH --cpus-per-task=4     

#module load CUDA/11.7.1


#source /software/rome/r24.04/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
#conda activate universal-ner

#source /data/horse/ws/irve354e-uniNer_test/qlora-ner/qlora-ner/.env

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,temperature.gpu --format=csv -l 5 > output/gpu_usage_10-newprompt.log &
gpu_log_pid=$!

singularity run --nv -B /data/horse/ws/irve354e-uniNer_test/qlora-ner/qlora-ner:/workspace lora-train-eval.sif /opt/venv/bin/python src/quantize_train.py
# for verbose container logs
#singularity run --debug --nv -B /data/horse/ws/irve354e-uniNer_test/qlora-ner/qlora-ner:/workspace lora-train-eval.sif /opt/venv/bin/python src/quantize_train.py

#python src/quantize_train.py
kill $gpu_log_pid

