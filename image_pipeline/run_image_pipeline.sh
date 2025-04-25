#!/bin/bash
#SBATCH --job-name=image_pipeline
#SBATCH --output=image_pipeline_output.log
#SBATCH --error=image_pipeline_error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# === Load modules
module load python/3.8 cuda/11.3

# === Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml_env

# === Navigate to your image pipeline directory
cd /home/bna36/misinfo_detection/image_pipeline

# === Run the image pipeline
python main.py
