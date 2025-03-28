#!/bin/bash
#SBATCH --job-name=qlora_training
#SBATCH --output=logs/qlora_training_%j.out
#SBATCH --error=logs/qlora_training_%j.err
#SBATCH --partition=alpha            
#SBATCH --gres=gpu:1          
#SBATCH --time=05:00:00            
#SBATCH --mem=32G                 
#SBATCH --cpus-per-task=4     

module load CUDA/11.7.0

source /software/rome/r24.04/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate universal-ner

export CUDA_VISIBLE_DEVICES=0

python quantize_train.py
